
use bytemuck;
use wgpu::util::DeviceExt;
use crate::elements::mesh;
use crate::render::gpu_structs::GPUCubeMapFaceHeader;
use crate::types::{GPUElements, GPU_NUM_MESH_BUFFERS};
use super::gpu_structs::{
    GPUCamera, 
    GPUCubeMapData, 
    GPUFreeTriangle, 
    GPUMeshData, 
    GPUMeshTriangle, 
    GPUMeshChunkHeader,
    GPUMeshHeader, 
    GPUPrimitiveHeader, 
    GPURenderInfo, 
    GPUSphere
};
use crate::elements::mesh::create_mesh_triangles_from_meshes;
use super::{RenderInfo, RenderTarget};
use pollster;
use futures_channel;


//
// ComputePipeline
//
struct ComputePipeline {
    canv_width: u32,
    canv_height: u32,
    pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,
    // Buffers can't go out of scope before the command buffer finishes executing
    // Or else bind groups could have dangling references
    _camera_buffer: wgpu::Buffer,
    _render_info_buffer: wgpu::Buffer,
    render_target_buffer: wgpu::Buffer,
    _mesh_chunk_hedaer_buffer: wgpu::Buffer,
    _mesh_header_buffer: wgpu::Buffer,
    _primitive_header_buffer: wgpu::Buffer,
    _mesh_data_buffers: Vec<wgpu::Buffer>,
    _mesh_triangle_buffer: wgpu::Buffer,
    _sphere_buffer: wgpu::Buffer,
    _cube_map_headers_buffer: wgpu::Buffer,
    _cube_map_data_buffer: wgpu::Buffer,
    _free_triangle_buffer: wgpu::Buffer,
}

impl ComputePipeline {
    fn create_camera_buffer(device: &wgpu::Device, camera: &GPUCamera) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::bytes_of(camera),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }
    
    fn create_render_info_buffer(device: &wgpu::Device, render_info: &GPURenderInfo) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RenderInfo Buffer"),
            contents: bytemuck::bytes_of(render_info),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn create_render_target_buffer(device: &wgpu::Device, render_target: &RenderTarget) -> wgpu::Buffer {
        let buffer_data = render_target.buff_mux.lock();
        let buffer_data_f32 = buffer_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(); 
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RenderTarget Buffer"),
            contents: bytemuck::cast_slice(&*buffer_data_f32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_iter_count_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Interation count buffer for RNG"),
            contents: bytemuck::bytes_of(&(0 as u64)),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn create_mesh_buffers(device: &wgpu::Device, meshes: &Vec<mesh::Mesh>) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, Vec<wgpu::Buffer>) {
        let mut all_mesh_headers: Vec<GPUMeshHeader> = vec![];
        let mut mesh_data_chunks: Vec<Vec<f32>> = vec![vec![]; GPU_NUM_MESH_BUFFERS];
        let mut all_primitive_headers: Vec<GPUPrimitiveHeader> = vec![];
        let mut curr_chunk = 0;
        let mut curr_chunk_size = 0;
        let chunk_limit =  1024 * 1024 * 1024;
        let mut chunk_headers: Vec<GPUMeshChunkHeader> = vec![GPUMeshChunkHeader::get_empty(); 4];
        let mut curr_chunk_meshes = 0;
        for mesh in meshes {
            let gpu_mesh = GPUMeshData::from_mesh(mesh);
            let (mut mesh_header, primitive_headers, mesh_data) = gpu_mesh.get_raw_buffers();
            let raw_data_size = mesh_data.len() * std::mem::size_of::<f32>();
            assert!(raw_data_size <= chunk_limit, "Mesh data too large for buffer");
            assert!(mesh_data.len() as u32 == mesh_header.length, "Mesh header size mismatch");
            if curr_chunk_size + raw_data_size > chunk_limit {
                chunk_headers[curr_chunk].num_meshes = curr_chunk_meshes;
                curr_chunk += 1;
                curr_chunk_size = 0;
                curr_chunk_meshes = 0;
                assert!(curr_chunk < GPU_NUM_MESH_BUFFERS, "Mesh data too large to fit in {} buffers", GPU_NUM_MESH_BUFFERS);
            }
            mesh_header.chunk_id = curr_chunk as u32;
            mesh_header.data_offset = mesh_data_chunks[curr_chunk].len() as u32;
            mesh_header.primitive_header_offset = all_primitive_headers.len() as u32;
            all_mesh_headers.push(mesh_header);
            mesh_data_chunks[curr_chunk].extend(mesh_data);
            all_primitive_headers.extend(primitive_headers);
            curr_chunk_size += raw_data_size;
            curr_chunk_meshes += 1;
        }
        chunk_headers[curr_chunk].num_meshes = curr_chunk_meshes;
        // Check if these buffers are empty
        // If they are, create empty buffers as placeholders since wgpu doesn't accept empty buffers
        assert!(all_mesh_headers.is_empty() == all_primitive_headers.is_empty(), "Mesh and primitive headers must be empty together");
        if all_mesh_headers.is_empty() {
            all_mesh_headers.push(GPUMeshHeader::get_empty());
        }
        if all_primitive_headers.is_empty() {
            all_primitive_headers.push(GPUPrimitiveHeader::get_empty());
        }
        let mesh_chunk_header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Chunk Header Buffer"),
            contents: bytemuck::cast_slice(&chunk_headers),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mesh_header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Header Buffer"),
            contents: bytemuck::cast_slice(&all_mesh_headers),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let primitive_header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Primitive Header Buffer"),
            contents: bytemuck::cast_slice(&all_primitive_headers),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut data_buffers: Vec<wgpu::Buffer> = vec![];
        mesh_data_chunks.iter_mut().enumerate().for_each(|(i, chunk)| {
            if chunk.is_empty() {
                chunk.push(0.0);
            }
            let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("Mesh Buffer {}", i).as_str()),
                contents: bytemuck::cast_slice(&chunk),
                usage: wgpu::BufferUsages::STORAGE,
            });
            data_buffers.push(data_buffer);
        });
        return (mesh_chunk_header_buffer, mesh_header_buffer, primitive_header_buffer, data_buffers);
    }

    fn create_renderables_buffer(device: &wgpu::Device, elements: &GPUElements) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let (spheres, cube_maps, free_triangles, meshes) = elements;
        let mut sphere_data: Vec<GPUSphere> = vec![];
        let mut cube_map_headers: Vec<GPUCubeMapFaceHeader> = vec![];
        let mut cube_map_data: Vec<f32> = vec![];
        // contains the offsets into the cube maps buffer
        // each offset points to the header of a cube map
        let mut free_triangle_data: Vec<GPUFreeTriangle> = vec![];
        let mut mesh_triangle_data: Vec<GPUMeshTriangle> = vec![];
        for sphere in spheres {
            let gpu_sphere = GPUSphere::from_sphere(sphere);
            sphere_data.push(gpu_sphere);
        }
        assert!(cube_maps.len() <= 1, "Expected maximum 1 cube map");
        for cube_map in cube_maps {
            let gpu_cube_map = GPUCubeMapData::from_cube_map(cube_map);
            let (headers, data) = gpu_cube_map.get_raw_buffers();
            cube_map_headers.extend(headers);
            cube_map_data.extend(data);
        }
        for free_triangle in free_triangles {
            let gpu_free_triangle = GPUFreeTriangle::from_free_triangle(free_triangle);
            free_triangle_data.push(gpu_free_triangle);
        }
        let mesh_triangles = create_mesh_triangles_from_meshes(meshes);
        for mesh_triangle in mesh_triangles {
            let gpu_mesh_triangle = GPUMeshTriangle::from_mesh_triangle(&mesh_triangle);
            mesh_triangle_data.push(gpu_mesh_triangle);
        }
        // WGSL does not support empty buffers
        // create a dummy value with is_valid = 0 if buffer is empty
        if sphere_data.is_empty() {
            sphere_data.push(GPUSphere::get_empty());
        }
        if free_triangle_data.is_empty() {
            free_triangle_data.push(GPUFreeTriangle::get_empty());
        }
        if cube_map_headers.is_empty() {
            cube_map_headers.push(GPUCubeMapFaceHeader::get_empty());
            cube_map_data.push(0.0);
        }
        if mesh_triangle_data.is_empty() {
            mesh_triangle_data.push(GPUMeshTriangle::get_empty());
        }

        let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Buffer"),
            contents: bytemuck::cast_slice(&sphere_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cube_map_headers_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CubeMap Header Buffer"),
            contents: bytemuck::cast_slice(&cube_map_headers),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cube_map_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CubeMap Buffer"),
            contents: bytemuck::cast_slice(&cube_map_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let free_triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FreeTriangle Buffer"),
            contents: bytemuck::cast_slice(&free_triangle_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mesh_triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MeshTriangle Buffer"),
            contents: bytemuck::cast_slice(&mesh_triangle_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        return (sphere_buffer, cube_map_headers_buffer, cube_map_data_buffer, free_triangle_buffer, mesh_triangle_buffer);
    }

    pub fn new(
        device: &wgpu::Device,
        camera: &GPUCamera,
        render_info: &GPURenderInfo,
        render_target: &RenderTarget,
        renderables: &GPUElements 
    ) -> Self {
        assert!(render_info.width % 8 == 0 , "Canv width must be a multiple of 8");
        assert!(render_info.height % 8 == 0 , "Canv height must be a multiple of 8");
        // Create buffers
        let camera_buffer = ComputePipeline::create_camera_buffer(device, camera);

        let render_info_buffer = ComputePipeline::create_render_info_buffer(device, render_info);

        let render_target_buffer = ComputePipeline::create_render_target_buffer(device, render_target);

        let iter_counter_buffer = ComputePipeline::create_iter_count_buffer(device);

        let (mesh_chunk_header_buffer, mesh_header_buffer, primitive_header_buffer, mesh_data_buffers) = 
            ComputePipeline::create_mesh_buffers(device, &renderables.3);
        assert!(mesh_data_buffers.len() == GPU_NUM_MESH_BUFFERS, "Currently supports only {} mesh buffers", GPU_NUM_MESH_BUFFERS);

        let (sphere_buffer, cube_map_headers_buffer, cube_map_data_buffer, free_triangle_buffer, mesh_triangle_buffer) = 
            ComputePipeline::create_renderables_buffer(device, renderables);

        // Create bind group layout
        // For uniform buffers camera and render info
        let first_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout 1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // For render_target_buffer which gets written to
        let second_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout 2"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,  // Note: This is now binding 0 in the new group
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // for mesh chunk header buffer, mesh header buffer, primitive buffer, mesh data buffers, mesh triangle buffer
        let mut third_bind_group_layouts: Vec<wgpu::BindGroupLayoutEntry> = vec![];

        for i in 0..(1 + 1 + 1 + GPU_NUM_MESH_BUFFERS + 1) {
            third_bind_group_layouts.push(
                wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
        }

        let third_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout 2"),
            entries: &third_bind_group_layouts,
        });

        // For free_triangle, sphere, distant cube map
        let forth_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout 3"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, 
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let first_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group 0"),
            layout: &first_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: render_info_buffer.as_entire_binding(),
                },
            ],
        });

        let second_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group 1"),
            layout: &second_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: render_target_buffer.as_entire_binding(),
                },
            ],
        });
        
        let mut third_bind_group_entries: Vec<wgpu::BindGroupEntry> = vec![];

        third_bind_group_entries.push(
            wgpu::BindGroupEntry {
            binding: third_bind_group_entries.len() as u32,
            resource: mesh_chunk_header_buffer.as_entire_binding(),
        });

        third_bind_group_entries.push(
            wgpu::BindGroupEntry {
            binding: third_bind_group_entries.len() as u32,
            resource: mesh_header_buffer.as_entire_binding(),
        });
        
        third_bind_group_entries.push(
            wgpu::BindGroupEntry {
            binding: third_bind_group_entries.len() as u32,
            resource: primitive_header_buffer.as_entire_binding(),
        });

        for i in 0..GPU_NUM_MESH_BUFFERS {
            third_bind_group_entries.push(
                wgpu::BindGroupEntry {
                binding: third_bind_group_entries.len() as u32,
                resource: mesh_data_buffers[i].as_entire_binding(),
            });
        }

        third_bind_group_entries.push(
            wgpu::BindGroupEntry {
            binding: third_bind_group_entries.len() as u32,
            resource: mesh_triangle_buffer.as_entire_binding(),
        });

        let third_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group 2"),
            layout: &third_bind_group_layout,
            entries: &third_bind_group_entries,
        });

        let forth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group 3"),
            layout: &forth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sphere_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cube_map_headers_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cube_map_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: free_triangle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: iter_counter_buffer.as_entire_binding(),
                },
            ],
        });


        let bind_groups = vec![first_bind_group, second_bind_group, third_bind_group, forth_bind_group];
        
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ray Trace Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("trace.wgsl"))),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ray Trace Compute Pipeline Layout"),
            bind_group_layouts: &[&first_bind_group_layout, &second_bind_group_layout, &third_bind_group_layout, &forth_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ray Trace Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        Self {
            canv_width: render_info.width,
            canv_height: render_info.height,
            pipeline,
            bind_groups,
            _camera_buffer: camera_buffer,
            _render_info_buffer: render_info_buffer,
            render_target_buffer,
            _mesh_chunk_hedaer_buffer: mesh_chunk_header_buffer,
            _mesh_header_buffer: mesh_header_buffer,
            _primitive_header_buffer: primitive_header_buffer,
            _mesh_data_buffers: mesh_data_buffers,
            _mesh_triangle_buffer: mesh_triangle_buffer,
            _sphere_buffer: sphere_buffer,
            _cube_map_headers_buffer: cube_map_headers_buffer,
            _cube_map_data_buffer: cube_map_data_buffer,
            _free_triangle_buffer: free_triangle_buffer,
        }
    }

    pub fn get_render_target_buffer(&self) -> &wgpu::Buffer {
        &self.render_target_buffer
    }
}

//
// GPU State
//
pub struct GPUState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    command_buffer: Option<wgpu::CommandBuffer>,
    compute_pipeline: Option<ComputePipeline>,
}

impl AsRef<GPUState> for GPUState {
    fn as_ref(&self) -> &GPUState {
        self
    }
}

impl GPUState {
    
    
    pub fn new() -> GPUState {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false
            },
        )).unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: 1024 * 1024 * 1024,
                max_buffer_size: 1024 * 1024 * 1024,
                ..Default::default()
            },
            label: Some("Ray Tracing Device"),
            memory_hints: Default::default()
        }, 
        None)).unwrap();

        Self { device, queue, command_buffer: None, compute_pipeline: None }
    }
    
    pub fn create_compute_pipeline(
        &mut self, 
        camera: &GPUCamera, 
        render_info: &GPURenderInfo, 
        render_target: &RenderTarget, 
        elements: &GPUElements 
    ) {
        assert!(self.compute_pipeline.is_none(), "An active compute pipeline already exists");
        let compute_pipeline = ComputePipeline::new(&self.device, camera, render_info, render_target, elements);
        self.compute_pipeline = Some(compute_pipeline);
    }

    pub fn dispatch_compute_pipeline(&mut self) {
        assert!(self.command_buffer.is_none(), "An active command buffer already exists");
        assert!(self.compute_pipeline.is_some(), "No compute pipeline to dispatch");
        let compute_pipeline = self.compute_pipeline.as_ref().unwrap();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Ray Trace Compute Pipeline Command Encoder"),
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            timestamp_writes: None,
            label: Some("Compute Pass"),
        });
        compute_pass.set_pipeline(&compute_pipeline.pipeline);
        for (i, bind_group) in compute_pipeline.bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(i as u32, bind_group, &[]);
        }
        let work_groups_x = (compute_pipeline.canv_width / 8) as u32;
        let work_groups_y = (compute_pipeline.canv_height / 8) as u32;
        compute_pass.dispatch_workgroups(work_groups_x, work_groups_y, 1);
        drop(compute_pass);
        self.command_buffer = Some(encoder.finish());
    }

    pub fn submit_compute_pipeline(&mut self) {
        assert!(self.command_buffer.is_some(), "No command buffer to submit");
        self.queue.submit(self.command_buffer.take());
    }

    pub fn block_and_get_single_result(&self) -> Vec<f32> {
        assert!(self.compute_pipeline.is_some(), "No compute pipeline to get result from");
        let compute_pipeline = self.compute_pipeline.as_ref().unwrap();
        let render_target_buffer = compute_pipeline.get_render_target_buffer();
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Pipeline Staging Buffer"),
            size: render_target_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Staging Buffer Command Encoder"),
        });

        encoder.copy_buffer_to_buffer(render_target_buffer, 0, &staging_buffer, 0, render_target_buffer.size());

        self.queue.submit(std::iter::once(encoder.finish()));

        // Need to slice the staging buffer to map and poll on it
        // Mapping here means making GPU buffer accessible to CPU i.e. map GPU memory to CPU
        let staging_buffer_slice = staging_buffer.slice(..);

        // Block until the callback is called when the staging_buffer_slice
        pollster::block_on(async {
            let (sender, receiver) = futures_channel::oneshot::channel();
            // Map the staging buffer slice, execute the callback when the data is mapped
            staging_buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            // Wait for all operations to finish on device
            self.device.poll(wgpu::Maintain::Wait);
            receiver.await.unwrap().unwrap();
        });
        
        // Get a CPU view of GPU memory
        let data = staging_buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        // Unmap the staging data, don't need view of GPU memory anymore
        staging_buffer.unmap();
        // self.compute_pipeline= Some(*compute_pipeline);
        result
    }

    // pub fn update_seed_time(& self) {
        
    //     let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
    //         label: Some("Update time seed in render_info buffer"),
    //     });

    //     let render_info = RenderInfo();
    //     let render_info_buffer = ComputePipeline::create_render_info_buffer(self.device, render_info);

    //     encoder.copy_buffer_to_buffer(render_target_buffer, 0, &staging_buffer, 0, render_target_buffer.size());

    //     self.compute_pipeline.as_ref().unwrap().render_info_buffer;
    // }

}

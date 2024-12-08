use bytemuck;
use wgpu::util::DeviceExt;
use super::gpu_structs::{GPUCamera, GPURenderInfo};
use super::RenderTarget;
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
}

impl ComputePipeline {
    fn create_camera_buffer(device: &wgpu::Device, camera: &GPUCamera) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::bytes_of(camera),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
    
    fn create_render_info_buffer(device: &wgpu::Device, render_info: &GPURenderInfo) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RenderInfo Buffer"),
            contents: bytemuck::bytes_of(render_info),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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

    pub fn new(
        device: &wgpu::Device,
        camera: &GPUCamera,
        render_info: &GPURenderInfo,
        render_target: &RenderTarget,
    ) -> Self {
        assert!(render_info.width % 8 == 0 , "Canv width must be a multiple of 8");
        assert!(render_info.height % 8 == 0 , "Canv height must be a multiple of 8");
        // Create buffers
        let camera_buffer = ComputePipeline::create_camera_buffer(device, camera);

        let render_info_buffer = ComputePipeline::create_render_info_buffer(device, render_info);

        let render_target_buffer = ComputePipeline::create_render_target_buffer(device, render_target);

        // Create bind group layout
        let first_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout 0"),
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

        let second_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout 1"),
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
        
        // Create bind group
        let mut bind_groups = Vec::new();
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
        
        bind_groups.push(first_bind_group);
        bind_groups.push(second_bind_group);
        
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ray Trace Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("trace.wgsl"))),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ray Trace Compute Pipeline Layout"),
            bind_group_layouts: &[&first_bind_group_layout, &second_bind_group_layout],
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
            required_limits: wgpu::Limits::default(),
            label: Some("Ray Tracing Device"),
            memory_hints: Default::default()
        }, 
        None)).unwrap();

        Self { device, queue, command_buffer: None, compute_pipeline: None }
    }
    
    pub fn create_compute_pipeline(&mut self, camera: &GPUCamera, render_info: &GPURenderInfo, render_target: &RenderTarget) {
        assert!(self.compute_pipeline.is_none(), "An active compute pipeline already exists");
        let compute_pipeline = ComputePipeline::new(&self.device, camera, render_info, render_target);
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

    pub fn block_and_get_single_result(&mut self) -> Vec<f32> {
        assert!(self.compute_pipeline.is_some(), "No compute pipeline to get result from");
        let compute_pipeline = self.compute_pipeline.take().unwrap();
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
        result
    }

}



use nalgebra::Vector3;
use serde::Deserialize;
use bytemuck;
use crate::accel::{PlaneBounds, Aabb};
use crate::elements::distant_cube_map::DistantCubeMap;
use crate::elements::sphere::{Sphere, Coloring};
use crate::material::{DivertRayMethod, UniformDiffuseSpec};
use crate::elements::triangle::FreeTriangle;
use crate::elements::mesh::{Mesh, MeshTriangle};
use crate::ray::Hitable;
use crate::scene::Cam;
use super::RenderInfo;
use static_assertions;

// Ensure that the size of the type is divisible by 16 for gpu alignment
// Assertion is evaluated at compile time
macro_rules! assert_gpu_aligned {
    ($t:ty) => {
        static_assertions::const_assert_eq!(std::mem::size_of::<$t>() % 16, 0);
    };
}

//
// Buffers
//
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)] 
pub struct GPUCamera {
    pub direction: [f32; 4],    // d: direction vector (w=0 for direction)
    pub origin: [f32; 4],       // o: camera position (w=1 for position)
    pub up: [f32; 4],          // up vector (w=0 for direction)
    pub screen_dims: [f32; 2],  // screen_width, screen_height
    pub lens_radius: f32,       // lens_r (0 if None)
    pub padding: f32,          // for alignment
}

impl GPUCamera {
    pub fn from_cam(camera: &Cam) -> Self {
        let lens_radius = match camera.lens_r {
            Some(a) => a,
            None => 0.0,
        };

        Self {
            direction: [camera.d.x, camera.d.y, camera.d.z, 0.0],
            origin: [camera.o.x, camera.o.y, camera.o.z, 1.0],
            up: [camera.up.x, camera.up.y, camera.up.z, 0.0],
            screen_dims: [camera.screen_width, camera.screen_height],
            lens_radius,
            padding: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPURenderInfo {
    pub width: u32, // canv_width
    pub height: u32, // canv_height
    pub samps_per_pix: u32, // samps_per_pix
    pub assured_depth: u32, // russ_roull_info.assured_depth
    pub max_threshold: f32, // russ_roull_info.max_thres
    pub kd_tree_depth: u32, // kd_tree_depth
    pub debug_single_ray: u32, // rad_info.debug_single_ray
    pub dir_light_samp: u32, // rad_info.dir_light_samp
    // pub seed_time: u32,      // To use as a seed for the random number generator
    // padding: [u32; 3],
}

impl GPURenderInfo {
    pub fn from_render_info(render_info: &RenderInfo) -> Self {
        Self {
            width: render_info.width as u32,
            height: render_info.height as u32,
            samps_per_pix: render_info.gpu_render_batch.unwrap() as u32,//render_info.samps_per_pix as u32,
            assured_depth: render_info.rad_info.russ_roull_info.assured_depth as u32,
            max_threshold: render_info.rad_info.russ_roull_info.max_thres,
            kd_tree_depth: render_info.kd_tree_depth as u32,
            debug_single_ray: render_info.rad_info.debug_single_ray as u32,
            dir_light_samp: render_info.rad_info.dir_light_samp as u32,
            // seed_time: (std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_secs()) as u32,
            // padding: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUUniformDiffuseSpec {
    pub emissive: [f32; 3],
    pub has_emissive: u32,
    pub divert_ray_type: u32,
    pub diffp: f32,      // For DiffSpec
    pub n_out: f32,      // For Dielectric
    pub n_in: f32,       // For Dielectric
}

impl GPUUniformDiffuseSpec {
    pub fn from_material(material: &UniformDiffuseSpec) -> Self {
        let (diffp, n_out, n_in) = match material.divert_ray {
            DivertRayMethod::DiffSpec { diffp } => (diffp, 0.0, 0.0),
            DivertRayMethod::Dielectric { n_out, n_in } => (0.0, n_out, n_in),
            _ => (0.0, 0.0, 0.0),
        };

        Self {
            emissive: material.emissive.map(|v| [v.x, v.y, v.z]).unwrap_or([0.0; 3]),
            has_emissive: material.emissive.is_some() as u32,
            divert_ray_type: match material.divert_ray {
                DivertRayMethod::Spec => 0,
                DivertRayMethod::Diff => 1,
                DivertRayMethod::DiffSpec { .. } => 2,
                DivertRayMethod::Dielectric { .. } => 3,
            },
            diffp,
            n_out,
            n_in,
        }
    }
}

//
// GPU Representation of triangle::FreeTriangle
//
#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUFreeTriangle {
    pub vert1: [f32; 4],
    pub vert2: [f32; 4],
    pub vert3: [f32; 4],
    pub norm: [f32; 4],
    pub rgb: [f32; 4],
    pub _padding: [f32; 3],
    pub is_valid: u32,
    pub material: GPUUniformDiffuseSpec,
}

impl GPUFreeTriangle {
    pub fn get_empty() -> Self {
        Self {
            vert1: [0.0; 4],
            vert2: [0.0; 4],
            vert3: [0.0; 4],
            norm: [0.0; 4],
            rgb: [0.0; 4],
            _padding: [0.0; 3],
            is_valid: 0,
            material: GPUUniformDiffuseSpec {
                emissive: [0.0; 3],
                has_emissive: 0,
                divert_ray_type: 0,
                diffp: 0.0,
                n_out: 0.0,
                n_in: 0.0,
            },
        }
    }
    pub fn from_free_triangle(triangle: &FreeTriangle) -> Self {
        let gpu_triangle = Self {
            vert1: [triangle.verts[0].x, triangle.verts[0].y, triangle.verts[0].z, 1.0],
            vert2: [triangle.verts[1].x, triangle.verts[1].y, triangle.verts[1].z, 1.0],
            vert3: [triangle.verts[2].x, triangle.verts[2].y, triangle.verts[2].z, 1.0],
            norm: [triangle.norm.0.x, triangle.norm.0.y, triangle.norm.0.z, 0.0],
            rgb: [triangle.rgb.x, triangle.rgb.y, triangle.rgb.z, 0.0],
            _padding: [0.0; 3],
            is_valid: 1,
            material: GPUUniformDiffuseSpec::from_material(&triangle.diverts_ray),
        };
        return gpu_triangle;
    }
}

//
// GPU Representation of elements::Sphere
//
#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUSphere {
    pub center: [f32; 4],
    pub coloring: [f32; 4],
    pub radius: f32,
    pub is_valid: u32,
    pub _padding: [f32; 2],
    pub material: GPUUniformDiffuseSpec,
}

impl GPUSphere {
    pub fn get_empty() -> Self {
        Self {
            center: [0.0; 4],
            coloring: [0.0; 4],
            radius: 0.0,
            is_valid: 0,
            _padding: [0.0; 2],
            material: GPUUniformDiffuseSpec {
                emissive: [0.0; 3],
                has_emissive: 0,
                divert_ray_type: 0,
                diffp: 0.0,
                n_out: 0.0,
                n_in: 0.0,
            },
        }
    }
    pub fn from_sphere(sphere: &Sphere) -> Self {
        Self {
            center: [sphere.c.x, sphere.c.y, sphere.c.z, 1.0],
            coloring: match &sphere.coloring {
                Coloring::Solid(c) => [c.x, c.y, c.z, 0.0],
            },
            radius: sphere.r,
            is_valid: 1,
            _padding: [0.0; 2],
            material: GPUUniformDiffuseSpec::from_material(&sphere.mat),
        }
    }
}

//
// GPU Representation of DistantCubeMap
//[Face 1 Header (GPUCubeMapFaceHeader)][Face 1 Pixels...][Face 2 Header][Face 2 Pixels...] etc...
#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUCubeMapFaceHeader {
    pub width: u32,
    pub height: u32,
    pub uv_scale_x: f32,
    pub uv_scale_y: f32,
}

impl GPUCubeMapFaceHeader {
    pub fn get_empty() -> Self {
        Self {
            width: 0,
            height: 0,
            uv_scale_x: 0.0,
            uv_scale_y: 0.0,
        }
    }
}

pub struct GPUCubeMapData {
    pub headers: [GPUCubeMapFaceHeader; 6],
    pub data: [Vec<f32>; 6],
}

impl GPUCubeMapData {
    pub fn from_cube_map(cube_map: &DistantCubeMap) -> GPUCubeMapData {
        let faces = [
            &cube_map.neg_z,
            &cube_map.pos_z,
            &cube_map.neg_x,
            &cube_map.pos_x,
            &cube_map.neg_y,
            &cube_map.pos_y,
        ];
        let headers: [GPUCubeMapFaceHeader; 6] = std::array::from_fn(|i| {
            GPUCubeMapFaceHeader {
                width: faces[i].0.get_width(),
                height: faces[i].0.get_height(),
                uv_scale_x: faces[i].1,
                uv_scale_y: faces[i].2,
            }
        });
        let data: [Vec<f32>; 6] = std::array::from_fn(|i| {
            let face = &faces[i].0;
            return face.as_raw();
        });
        GPUCubeMapData {
            headers,
            data,
        }
    }
    pub fn get_raw_buffers(&self) -> (Vec<GPUCubeMapFaceHeader>, Vec<f32>) {
        let mut header_buffer: Vec<GPUCubeMapFaceHeader> = Vec::new();
        let mut data_buffer: Vec<f32> = Vec::new();
        for i in 0..6 {
            let header = self.headers[i];
            let face = &self.data[i];
            assert!(header.width * header.height * 3 == face.len() as u32);
            header_buffer.push(header);
            data_buffer.extend(face);
        }
        return (header_buffer, data_buffer);
    }
}


#[repr(C)]
#[derive(Copy, Clone, Default, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUPrimitiveHeader {
    pub length: u32,
    // offset to the start of the primitive data in the mesh data sequence.
    // to access position, do chunk[mesh_header.data_offset + prim_header.mesh_data_offset + prim_header.position_offset]
    // to access normal, do chunk[mesh_header.data_offset + prim_header.mesh_data_offset + prim_header.normal_offset]
    // etc.
    // where mesh_header is the header of the mesh containing this primitive
    pub mesh_data_offset: u32,
    pub position_offset: u32,
    pub position_count: u32,

    pub normal_offset: u32,
    pub normal_count: u32,

    pub triangle_offset: u32,
    pub triangle_count: u32,

    pub rgb_info_factor_offset: u32,
    pub rgb_info_coords_offset: u32,
    pub rgb_info_coords_count: u32,

    pub norm_info_scale_offset: u32,
    pub norm_info_coords_offset: u32,
    pub has_norm_info: u32,
    pub norm_info_coords_count: u32,

    pub metal_rough_metal_offset: u32,
    pub metal_rough_rough_offset: u32,
    pub metal_rough_coords_offset: u32,
    pub metal_rough_coords_count: u32,

    pub texture_data_offset: u32,
    pub texture_data_width: u32,
    pub texture_data_height: u32,

    pub normal_map_data_offset: u32,
    pub normal_map_data_width: u32,
    pub normal_map_data_height: u32,

    pub metal_rough_map_data_offset: u32,
    pub metal_rough_map_data_width: u32,
    pub metal_rough_map_data_height: u32,

}

impl GPUPrimitiveHeader {
    pub fn get_empty() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn from_primitive(mesh: &Mesh, i: usize, mesh_data_offset: u32, my_length: u32) -> Self {
        let position_offset = 0;
        let normal_offset = position_offset + (mesh.poses[i].len() * 3) as u32;
        let triangle_offset = normal_offset + (mesh.norms[i].len() * 3) as u32;
        let rgb_info_factor_offset = triangle_offset + (mesh.indices[i].len() * 3) as u32;
        let rgb_info_coords_offset = rgb_info_factor_offset + mesh.rgb_info[i].factor.len() as u32;
        let norm_info_scale_offset = rgb_info_coords_offset + mesh.rgb_info[i].coords.as_ref().map_or(0, |v| v.len() as u32 * 2);
        let norm_info_coords_offset = norm_info_scale_offset + mesh.norm_info[i].as_ref().map_or(0, |_| 1);
        let metal_rough_metal_offset = norm_info_coords_offset + mesh.norm_info[i].as_ref().map_or(0, |v| v.coords.len() as u32 * 2);
        let metal_rough_rough_offset = metal_rough_metal_offset + 1;
        let metal_rough_coords_offset = metal_rough_rough_offset + 1;
        let texture_data_offset = metal_rough_coords_offset + mesh.metal_rough[i].coords.as_ref().map_or(0, |v| v.len() as u32 * 2);
        let normal_map_data_offset = texture_data_offset + mesh.textures[i].as_ref().map_or(0, |img| img.get_width() * img.get_height() * 3) as u32;
        let metal_rough_map_data_offset = normal_map_data_offset + mesh.normal_maps[i].as_ref().map_or(0, |img| img.get_width() * img.get_height() * 3) as u32;

        let prim_header = GPUPrimitiveHeader {
            length: my_length,
            mesh_data_offset,

            position_offset,
            position_count: mesh.poses[i].len() as u32,

            normal_offset,
            normal_count: mesh.norms[i].len() as u32,

            triangle_offset,
            triangle_count: mesh.indices[i].len() as u32,

            rgb_info_factor_offset,
            rgb_info_coords_offset,
            rgb_info_coords_count: mesh.rgb_info[i].coords.as_ref().map_or(0, |v| v.len() as u32),

            norm_info_scale_offset,
            norm_info_coords_offset,
            has_norm_info: mesh.norm_info[i].is_some() as u32,
            norm_info_coords_count: mesh.norm_info[i].as_ref().map_or(0,|v| v.coords.len() as u32),

            metal_rough_metal_offset,
            metal_rough_rough_offset,
            metal_rough_coords_offset,
            metal_rough_coords_count: mesh.metal_rough[i].coords.as_ref().map_or(0, |v| v.len() as u32),

            texture_data_offset,
            texture_data_width: mesh.textures[i].as_ref().map_or(0, |img| img.get_width() as u32),
            texture_data_height: mesh.textures[i].as_ref().map_or(0, |img| img.get_height() as u32),

            normal_map_data_offset,
            normal_map_data_width: mesh.normal_maps[i].as_ref().map_or(0, |img| img.get_width() as u32),
            normal_map_data_height: mesh.normal_maps[i].as_ref().map_or(0, |img| img.get_height() as u32),

            metal_rough_map_data_offset,
            metal_rough_map_data_width: mesh.metal_rough_maps[i].as_ref().map_or(0, |img| img.get_width() as u32),
            metal_rough_map_data_height: mesh.metal_rough_maps[i].as_ref().map_or(0, |img| img.get_height() as u32),
        };

        return prim_header;
    }
}

pub struct GPUPrimitiveData {
    pub positions: Vec<[f32; 3]>,
    pub norms: Vec<[f32; 3]>,
    pub triangles: Vec<[f32; 3]>,

    pub rgb_info_factor: [f32; 3],
    pub rgb_info_coords: Option<Vec<[f32; 2]>>,

    pub norm_info_scale: Option<f32>,
    pub norm_info_coords: Option<Vec<[f32; 2]>>,

    pub metal_rough_metal: f32,
    pub metal_rough_rough: f32,
    pub metal_rough_coords: Option<Vec<[f32; 2]>>,

    pub texture_data: Option<Vec<f32>>,

    pub normal_map_data: Option<Vec<f32>>,

    pub metal_rough_map_data: Option<Vec<f32>>,
}

impl GPUPrimitiveData {
    pub fn get_raw_buffer(&self) -> Vec<f32> {
        let mut buffer: Vec<f32> = Vec::new();
        buffer.extend_from_slice(bytemuck::cast_slice(&self.positions));
        buffer.extend_from_slice(bytemuck::cast_slice(&self.norms));
        buffer.extend_from_slice(bytemuck::cast_slice(&self.triangles));
        buffer.extend_from_slice(bytemuck::cast_slice(&self.rgb_info_factor));
        self.rgb_info_coords.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        self.norm_info_scale.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(&[*v])));
        self.norm_info_coords.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        buffer.extend_from_slice(bytemuck::cast_slice(&[self.metal_rough_metal]));
        buffer.extend_from_slice(bytemuck::cast_slice(&[self.metal_rough_rough]));
        self.metal_rough_coords.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        self.texture_data.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        self.normal_map_data.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        self.metal_rough_map_data.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        return buffer;
    }
}


#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUMeshChunkHeader {
    pub num_meshes: u32,
    pub _padding: [u32; 3],
}

impl GPUMeshChunkHeader {
    pub fn get_empty() -> Self {
        Self {
            num_meshes: 0,
            _padding: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUMeshHeader {
    pub length: u32,
    pub num_primitives: u32,
    pub chunk_id: u32,
    // offset to the start of the mesh data in the chunk buffer
    pub data_offset: u32,
    pub _padding: [u32; 3],
    pub primitive_header_offset: u32,
    pub trans_mat: [f32; 16],
}

impl GPUMeshHeader {
    pub fn get_empty() -> Self {
        Self {
            length: 0,
            num_primitives: 0,
            chunk_id: 0,
            data_offset: 0,
            primitive_header_offset: 0,
            _padding: [0; 3],
            trans_mat: [0.0; 16],
        }
    }
}


pub struct GPUMeshData {
    pub mesh_header: GPUMeshHeader,
    pub primitive_headers: Vec<GPUPrimitiveHeader>,
    pub primitive_data: Vec<Vec<f32>>,
}

impl GPUMeshData {
    pub fn from_mesh(mesh: &Mesh) -> GPUMeshData {
        mesh.check_num_primitives();
        let num_primitives = mesh.poses.len();
        let mut primitive_headers: Vec<GPUPrimitiveHeader>  = Vec::new();
        let mut primitive_data: Vec<Vec<f32>> = Vec::new();
        let mut total_length: u32 = 0;
        for i in 0..num_primitives {
            let primitive_data_f32 = GPUPrimitiveData {
                positions: mesh.poses[i].iter().map(|v| [v.x, v.y, v.z]).collect(),
                norms: mesh.norms[i].iter().map(|v| [v.x, v.y, v.z]).collect(),
                triangles: mesh.indices[i].iter().map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]).collect(),
                rgb_info_factor: mesh.rgb_info[i].factor.as_slice().try_into().unwrap(),
                rgb_info_coords: mesh.rgb_info[i].coords.as_ref().map(|v| v.iter().map(|v| [v.x, v.y]).collect()),
                norm_info_scale: mesh.norm_info[i].as_ref().map(|v| v.scale),
                norm_info_coords: mesh.norm_info[i].as_ref().map(|v| v.coords.iter().map(|v| [v.x, v.y]).collect()),
                metal_rough_metal: mesh.metal_rough[i].metal,
                metal_rough_rough: mesh.metal_rough[i].rough,
                metal_rough_coords: mesh.metal_rough[i].coords.as_ref().map(|v| v.iter().map(|v| [v.x, v.y]).collect()),
                texture_data: mesh.textures[i].as_ref().map(|v| v.as_raw()),
                normal_map_data: mesh.normal_maps[i].as_ref().map(|v| v.as_raw()),
                metal_rough_map_data: mesh.metal_rough_maps[i].as_ref().map(|v| v.as_raw()),
            }.get_raw_buffer();

            let prim_header = GPUPrimitiveHeader::from_primitive(mesh, i, total_length, primitive_data_f32.len() as u32);
            total_length += primitive_data_f32.len() as u32;
            primitive_data.push(primitive_data_f32);
            primitive_headers.push(prim_header);
        }
        let mesh_header = GPUMeshHeader {
            length: total_length,
            num_primitives: num_primitives as u32,
            // To be set later when generating the buffers
            chunk_id: 0,
            data_offset: 0,
            primitive_header_offset: 0,
            _padding: [0; 3],
            // column major, which is what wgpu expects
            trans_mat: mesh.trans_mat.as_slice().try_into().unwrap(),
        };

        GPUMeshData {
            mesh_header,
            primitive_headers,
            primitive_data,
        }
    }

    pub fn get_raw_buffers(&self) -> (GPUMeshHeader, Vec<GPUPrimitiveHeader>, Vec<f32>) {
        let mut primitive_headers_buffer: Vec<GPUPrimitiveHeader> = Vec::new();
        let mut mesh_data_buffer = Vec::new();
        for i in 0..self.primitive_data.len() {
            assert!(self.primitive_headers[i].length == self.primitive_data[i].len() as u32);
            primitive_headers_buffer.push(self.primitive_headers[i]);
            mesh_data_buffer.extend(&self.primitive_data[i]);
        }
        return (self.mesh_header, primitive_headers_buffer, mesh_data_buffer);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUMeshTriangle {
    pub mesh_index: u32,
    pub prim_index: u32,
    pub inner_index: u32,
    pub is_valid: u32,
    pub normal_transform_c1: [f32; 3],
    pub _padding: f32,
    pub normal_transform_c2: [f32; 3],
    pub _padding2: f32,
    pub normal_transform_c3: [f32; 3],
    pub _padding3: f32,
}

impl GPUMeshTriangle {
    pub fn get_empty() -> Self {
        Self {
            ..Default::default()
        }
    }
    pub fn from_mesh_triangle(mesh_triangle: &MeshTriangle) -> Self {
        assert!(mesh_triangle.norm.mesh_index == mesh_triangle.verts.mesh_index);
        assert!(mesh_triangle.rgb.mesh_index == mesh_triangle.verts.mesh_index);
        assert!(mesh_triangle.diverts_ray.mesh_index == mesh_triangle.verts.mesh_index);
        assert!(mesh_triangle.norm.index == mesh_triangle.verts.index);
        assert!(mesh_triangle.rgb.index == mesh_triangle.verts.index);
        assert!(mesh_triangle.diverts_ray.index == mesh_triangle.verts.index);

        let mesh_index = mesh_triangle.verts.mesh_index;
        let (prim_index, inner_index) = mesh_triangle.verts.index;

        Self {
            mesh_index,
            prim_index: prim_index as u32,
            inner_index: inner_index as u32,
            is_valid: 1,
            normal_transform_c1: mesh_triangle.norm.normal_transform.column(0).as_slice().try_into().unwrap(),
            _padding: 0.0,
            normal_transform_c2: mesh_triangle.norm.normal_transform.column(1).as_slice().try_into().unwrap(),
            _padding2: 0.0,
            normal_transform_c3: mesh_triangle.norm.normal_transform.column(2).as_slice().try_into().unwrap(),
            _padding3: 0.0,
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUIter {
    pub _padding: [u32; 3],
    pub ation: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUAabb {
    // aabb bounding box of whatever object. Only used for GPUKdTree for now.
    pub bounds: [[f32; 4]; 3],
    // pub padding: [f32; 2],
}

impl GPUAabb {  
    pub fn new (aabb: Aabb) -> Self {
        Self {
            bounds: [[aabb.bounds[0].low, aabb.bounds[0].high, 0.0, 0.0],
                     [aabb.bounds[1].low, aabb.bounds[1].high, 0.0, 0.0],
                     [aabb.bounds[2].low, aabb.bounds[2].high, 0.0, 0.0]]
        }
    }
    // We want the kd tree for the meshes only for the GPU mode. KD tree doesn't help much for the other primitives as we can't add as many
    // So, the KD tree is its own primitive in a way like the Spheres and FreeTriangles
    pub fn build_gpu_kd_tree(mesh_triangles: &Vec<MeshTriangle>, max_depth: usize) -> (GPUAabb, Vec<GPUTreeNode>, Vec<u32>){
        let mut nodes = Vec::<GPUTreeNode>::new();
        let mut leaf_node_meshes = Vec::<u32>::with_capacity(mesh_triangles.len()); // size of leaf_node_meshes is the same as the number of meshes
        let aabbs: Vec<(usize, Aabb)> = mesh_triangles.iter().enumerate().filter_map(|(i, tri)| tri.give_aabb().map(|aabb| (i, aabb))).collect();

        // Get the entire structure's kd tree
        // We can consider separating the kd tree for each mesh member for better granularity.
        let kd_tree_aabb = GPUAabb::give_aabb(&aabbs);
        
        // GPUTreeNode::new_branch_node();

        // Finally, build the tree nodes
        Self::build_gpu_tree_nodes(&aabbs, 0, max_depth, &mut nodes, &mut leaf_node_meshes, 0, false, kd_tree_aabb);

        // Return the aabb of the entire kd-tree and its nodes
        (kd_tree_aabb, nodes, leaf_node_meshes)

    }

    pub fn give_aabb(aabbs: &Vec<(usize, Aabb)>) -> GPUAabb {
        let aabb = {
            let min_axes: Vec<f32> = (0..3).map(
                |a| (&aabbs).into_iter().map(|(_, aabb)| aabb.bounds[a].low)
                    .reduce(|pl, l| pl.min(l))
                    .unwrap()
                )
                .collect();
            let max_axes: Vec<f32> = (0..3).map(
                |a| (&aabbs).into_iter().map(|(_, aabb)| aabb.bounds[a].high)
                    .reduce(|ph, h| ph.max(h))
                    .unwrap()
                )
                .collect();
            Aabb {
                bounds: [
                    PlaneBounds {low: min_axes[0], high: max_axes[0]},
                    PlaneBounds {low: min_axes[1], high: max_axes[1]},
                    PlaneBounds {low: min_axes[2], high: max_axes[2]},
                ]
            }
        };

        GPUAabb::new(aabb)
    }

    // Call this function recursively, but construct an easily accessible vector with it suitable for the GPU
    pub fn build_gpu_tree_nodes(index_and_aabbs: &Vec<(usize, Aabb)>, cur_depth: usize, max_depth: usize, nodes: &mut Vec<GPUTreeNode>, leaf_node_meshes: &mut Vec<u32>, parent_node_idx: usize, high_node: bool, curr_aabb: GPUAabb){
        // Iterate between each axis for the 
        let axis = cur_depth % 3;

        // Create a leaf node here
        // TODO: If performance impact of the push is high, initialize nodes with maximum theoretical size before running so it won't take as long. 
        if cur_depth >= max_depth || index_and_aabbs.len() <= 1 {
            nodes.push(GPUTreeNode::new_leaf_node(axis as u32, index_and_aabbs, leaf_node_meshes, curr_aabb));
            let cur_idx = nodes.len() - 1;
            // add child index for the parent of this leaf node
            if high_node {
                nodes.get_mut(parent_node_idx).unwrap().high = cur_idx as u32;
            } else {
                nodes.get_mut(parent_node_idx).unwrap().low = cur_idx as u32;
            }
        } else {
            let aabbs: Vec<Aabb> = index_and_aabbs.iter().map(|(_,aabb)| *aabb).collect();
            let split = (&aabbs).into_iter().map(|aabb| aabb.centroid()).sum::<Vector3<f32>>() / (aabbs.len() as f32);
            
            // Add new branch node
            nodes.push(GPUTreeNode::new_branch_node(axis as u32,  split[axis], curr_aabb));
        
            let cur_idx = nodes.len() - 1;
            
            // Update the parent's node pointer index with the child node based on whether it's low or high 
            // For head node, it'll get updated twice. Once when it updates its own low node, then for the action low node
            if high_node {
                nodes.get_mut(parent_node_idx).unwrap().high = cur_idx as u32;
            } else {
                nodes.get_mut(parent_node_idx).unwrap().low = cur_idx as u32;
            }

            let (low, high): (Vec<(usize, Aabb)>, Vec<(usize, Aabb)>) = {
                let mut low: Vec<(usize, Aabb)> = vec![];
                let mut high: Vec<(usize, Aabb)> = vec![];
        
                index_and_aabbs.iter().for_each(|(i, aabb)| {
                    // this can handle case of element in both nodes
                    if aabb.bounds[axis].high >= split[axis] {
                        high.push((*i, *aabb));
                    }
                    if aabb.bounds[axis].low <= split[axis] {
                        low.push((*i, *aabb));
                    }
                });
                (low, high)
            };

            // Low recursive call
            let mut low_aabb = curr_aabb;
            low_aabb.bounds[axis][1] = split[axis];
            GPUAabb::build_gpu_tree_nodes(&low, cur_depth + 1, max_depth, nodes, leaf_node_meshes, cur_idx, false, low_aabb);

            // High recursive call
            let mut high_aabb = curr_aabb;
            high_aabb.bounds[axis][0] = split[axis];
            GPUAabb::build_gpu_tree_nodes(&high, cur_depth + 1, max_depth, nodes, leaf_node_meshes, cur_idx, true, high_aabb);
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUTreeNode {
    pub _padding: f32,
    pub axis: u32, // The axis of this Node
    pub split: f32, // The split coordinate of this tree node. Used for finding whether to go in low or high tree node
    pub low: u32,  // Array index of lower Node. Acts like pointer
    pub high: u32, // Array index of higher Node. Acts like pointer
    pub is_leaf: u32, // Boolean value for whether this TreeNode is a leaf. If leaf, get intersections of mesh indices 
    pub leaf_mesh_index: u32, // Offset of triangle mesh indices array
    pub leaf_mesh_size: u32, // Number of triangle mesh indices within the leaf node
    pub aabb: GPUAabb, // In GPU mode, we require the AABB of each tree node for the sequential KD tree traversal
}

impl GPUTreeNode {
    pub fn new_leaf_node(axis: u32, aabbs: &Vec<(usize, Aabb)>, leaf_meshes: &mut Vec<u32>, aabb: GPUAabb) -> Self {
        let leaf_mesh_index = leaf_meshes.len() as u32; // Index where the leaf node's meshes start
        let leaf_mesh_size = aabbs.len() as u32; // How many meshes need to be added in the leaf node

        for (idx, _) in aabbs {
            leaf_meshes.push(*idx as u32);
        }
        
        Self {
            _padding: 0.0,
            axis,
            split: 0.0,
            low: 0,
            high: 0,
            is_leaf: 1,      // Relevant for leaf nodes only
            leaf_mesh_index, // Relevant for leaf nodes only
            leaf_mesh_size,  // Relevant for leaf nodes only
            aabb,
        }
    }

    pub fn new_branch_node(axis: u32, split: f32, aabb: GPUAabb) -> Self {
        
        Self {
            _padding: 0.0,     
            axis,
            split,
            low: 0,
            high: 0,
            is_leaf: 0,         //Irrelevant for branch nodes
            leaf_mesh_index: 0, //Irrelevant for branch nodes
            leaf_mesh_size: 0,  //Irrelevant for branch nodes
            aabb,
        }   
    }
}

assert_gpu_aligned!(GPUTreeNode);
assert_gpu_aligned!(GPUAabb);
assert_gpu_aligned!(GPUCamera);
assert_gpu_aligned!(GPURenderInfo);
assert_gpu_aligned!(GPUUniformDiffuseSpec);
assert_gpu_aligned!(GPUFreeTriangle);
assert_gpu_aligned!(GPUSphere);
assert_gpu_aligned!(GPUCubeMapFaceHeader);
assert_gpu_aligned!(GPUMeshChunkHeader);
assert_gpu_aligned!(GPUMeshHeader);
assert_gpu_aligned!(GPUPrimitiveHeader);
assert_gpu_aligned!(GPUMeshTriangle);
assert_gpu_aligned!(GPUIter);

use serde::Deserialize;
use bytemuck;
use crate::elements::distant_cube_map::DistantCubeMap;
use crate::elements::sphere::{Sphere, Coloring};
use crate::material::{DivertRayMethod, UniformDiffuseSpec};
use crate::elements::triangle::FreeTriangle;
use crate::elements::mesh::{Mesh, MeshTriangle, VertexFromMesh, NormFromMesh, RgbFromMesh, DivertsRayFromMesh};
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
}

impl GPURenderInfo {
    pub fn from_render_info(render_info: &RenderInfo) -> Self {
        Self {
            width: render_info.width as u32,
            height: render_info.height as u32,
            samps_per_pix: render_info.samps_per_pix as u32,
            assured_depth: render_info.rad_info.russ_roull_info.assured_depth as u32,
            max_threshold: render_info.rad_info.russ_roull_info.max_thres,
            kd_tree_depth: render_info.kd_tree_depth as u32,
            debug_single_ray: render_info.rad_info.debug_single_ray as u32,
            dir_light_samp: render_info.rad_info.dir_light_samp as u32,
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
    pub is_valid: u32,
    _padding: [f32; 3],
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
            is_valid: 0,
            _padding: [0.0; 3],
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
        Self {
            vert1: [triangle.verts[0].x, triangle.verts[0].y, triangle.verts[0].z, 1.0],
            vert2: [triangle.verts[1].x, triangle.verts[1].y, triangle.verts[1].z, 1.0],
            vert3: [triangle.verts[2].x, triangle.verts[2].y, triangle.verts[2].z, 1.0],
            norm: [triangle.norm.0.x, triangle.norm.0.y, triangle.norm.0.z, 0.0],
            rgb: [triangle.rgb.x, triangle.rgb.y, triangle.rgb.z, 0.0],
            is_valid: 1,
            _padding: [0.0; 3],
            material: GPUUniformDiffuseSpec::from_material(&triangle.diverts_ray),
        }
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
    uv_scale_x: f32,
    uv_scale_y: f32,
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


//
// GPU Representation of mesh::Mesh
// Gonna have a header/data_chunk per primitive
//
#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPURgbF32ImageHeader {
    pub width: u32,
    pub height: u32,
    pub _padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUPrimitiveHeader {
    pub length: u32,
    pub position_count: u32,
    pub normal_count: u32,
    pub triangle_count: u32,
    pub rgb_info_coords_count: u32,
    pub has_norm_info: u32,
    pub norm_info_coords_count: u32,
    pub tangents_count: u32,
    pub has_texture: u32,
    pub has_normal_map: u32,
    pub has_metal_rough_map: u32,
    pub _padding: u32,
}

impl GPUPrimitiveHeader {
    pub fn get_empty() -> Self {
        Self {
            length: 0,
            position_count: 0,
            normal_count: 0,
            triangle_count: 0,
            rgb_info_coords_count: 0,
            has_norm_info: 0,
            norm_info_coords_count: 0,
            tangents_count: 0,
            has_texture: 0,
            has_normal_map: 0,
            has_metal_rough_map: 0,
            _padding: 0,
        }
    }
}

pub struct GPUPrimitiveData {
    pub positions: Vec<[f32; 3]>,
    pub norms: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,

    pub rgb_info_factor: [f32; 3],
    pub rgb_info_coords: Option<Vec<[f32; 2]>>,

    pub norm_info_scale: Option<f32>,
    pub norm_info_coords: Option<Vec<[f32; 2]>>,

    pub tangents: Option<Vec<[f32; 3]>>,

    pub metal_rough_metal: f32,
    pub metal_rough_rough: f32,
    pub metal_rough_coords: Option<Vec<[f32; 2]>>,

    pub texture_header: Option<GPURgbF32ImageHeader>,
    pub texture_data: Option<Vec<f32>>,

    pub normal_map_header: Option<GPURgbF32ImageHeader>,
    pub normal_map_data: Option<Vec<f32>>,

    pub metal_rough_map_header: Option<GPURgbF32ImageHeader>,
    pub metal_rough_map_data: Option<Vec<f32>>,
}

impl GPUPrimitiveData {
    pub fn get_buffer_size(&self) -> usize {
        let mut buffer_size = 0;
        buffer_size += self.positions.len() * std::mem::size_of::<[f32; 3]>();
        buffer_size += self.norms.len() * std::mem::size_of::<[f32; 3]>();
        buffer_size += self.triangles.len() * std::mem::size_of::<[u32; 3]>();
        buffer_size += std::mem::size_of::<[f32; 3]>();  // rgb_info_factor
        buffer_size += self.rgb_info_coords.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<[f32; 2]>());
        buffer_size += std::mem::size_of::<f32>();  // norm_info_scale if Some
        buffer_size += self.norm_info_coords.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<[f32; 2]>());
        buffer_size += self.tangents.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<[f32; 3]>());
        buffer_size += std::mem::size_of::<f32>();  // metal_rough_metal
        buffer_size += std::mem::size_of::<f32>();  // metal_rough_rough
        buffer_size += self.metal_rough_coords.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<[f32; 2]>());
        buffer_size += self.texture_header.as_ref().map_or(0, |_| std::mem::size_of::<GPURgbF32ImageHeader>());
        buffer_size += self.texture_data.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<f32>());
        buffer_size += self.normal_map_header.as_ref().map_or(0, |_| std::mem::size_of::<GPURgbF32ImageHeader>());
        buffer_size += self.normal_map_data.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<f32>());
        buffer_size += self.metal_rough_map_header.as_ref().map_or(0, |_| std::mem::size_of::<GPURgbF32ImageHeader>());
        buffer_size += self.metal_rough_map_data.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<f32>());
        buffer_size += std::mem::size_of::<[f32; 16]>();  // trans_mat
        return buffer_size;
    }

    pub fn get_raw_buffer(&self) -> Vec<f32> {
        let mut buffer: Vec<f32> = Vec::new();
        buffer.extend_from_slice(bytemuck::cast_slice(&self.positions));
        buffer.extend_from_slice(bytemuck::cast_slice(&self.norms));
        buffer.extend_from_slice(bytemuck::cast_slice(&self.triangles));
        buffer.extend_from_slice(bytemuck::cast_slice(&self.rgb_info_factor));
        self.rgb_info_coords.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        self.norm_info_scale.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(&[*v])));
        self.norm_info_coords.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        self.tangents.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        buffer.extend_from_slice(bytemuck::cast_slice(&[self.metal_rough_metal]));
        buffer.extend_from_slice(bytemuck::cast_slice(&[self.metal_rough_rough]));
        self.metal_rough_coords.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        self.texture_header.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(&[*v])));
        self.texture_data.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        self.normal_map_header.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(&[*v])));
        self.normal_map_data.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(v)));
        self.metal_rough_map_header.as_ref().map(|v| buffer.extend_from_slice(bytemuck::cast_slice(&[*v])));
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
    pub _padding: [u32; 2],
    pub trans_mat: [f32; 16],
}

impl GPUMeshHeader {
    pub fn get_empty() -> Self {
        Self {
            length: 0,
            num_primitives: 0,
            _padding: [0; 2],
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
            let raw_data = GPUPrimitiveData {
                positions: mesh.poses[i].iter().map(|v| [v.x, v.y, v.z]).collect(),
                norms: mesh.norms[i].iter().map(|v| [v.x, v.y, v.z]).collect(),
                triangles: mesh.indices[i].iter().map(|v| [v[0] as u32, v[1] as u32, v[2] as u32]).collect(),
                rgb_info_factor: mesh.rgb_info[i].factor.into(),
                rgb_info_coords: mesh.rgb_info[i].coords.as_ref().map(|v| v.iter().map(|v| [v.x, v.y]).collect()),
                norm_info_scale: mesh.norm_info[i].as_ref().map(|v| v.scale),
                norm_info_coords: mesh.norm_info[i].as_ref().map(|v| v.coords.iter().map(|v| [v.x, v.y]).collect()),
                tangents: mesh.tangents[i].as_ref().map(|v| v.iter().map(|v| [v.x, v.y, v.z]).collect()),
                metal_rough_metal: mesh.metal_rough[i].metal,
                metal_rough_rough: mesh.metal_rough[i].rough,
                metal_rough_coords: mesh.metal_rough[i].coords.as_ref().map(|v| v.iter().map(|v| [v.x, v.y]).collect()),
                texture_header: mesh.textures[i].as_ref().map(|v| GPURgbF32ImageHeader { width: v.get_width(), height: v.get_height(), _padding: [0; 2] }),
                texture_data: mesh.textures[i].as_ref().map(|v| v.as_raw()),
                normal_map_header: mesh.normal_maps[i].as_ref().map(|v| GPURgbF32ImageHeader { width: v.get_width(), height: v.get_height(), _padding: [0; 2] }),
                normal_map_data: mesh.normal_maps[i].as_ref().map(|v| v.as_raw()),
                metal_rough_map_header: mesh.metal_rough_maps[i].as_ref().map(|v| GPURgbF32ImageHeader { width: v.get_width(), height: v.get_height(), _padding: [0; 2] }),
                metal_rough_map_data: mesh.metal_rough_maps[i].as_ref().map(|v| v.as_raw()),
            }.get_raw_buffer();
            let prim_header = GPUPrimitiveHeader {
                length: raw_data.len() as u32,
                position_count: mesh.poses[i].len() as u32,
                normal_count: mesh.norms[i].len() as u32,
                triangle_count: mesh.indices[i].len() as u32,
                rgb_info_coords_count: mesh.rgb_info[i].coords.as_ref().map_or(0, |v| v.len() as u32),
                has_norm_info: mesh.norm_info[i].is_some() as u32,
                norm_info_coords_count: mesh.norm_info[i].as_ref().map_or(0,|v| v.coords.len() as u32),
                tangents_count: mesh.tangents[i].as_ref().map_or(0,|v| v.len() as u32),
                has_texture: mesh.textures[i].is_some() as u32,
                has_normal_map: mesh.normal_maps[i].is_some() as u32,
                has_metal_rough_map: mesh.metal_rough_maps[i].is_some() as u32,
                _padding: 0,
            };
            total_length += raw_data.len() as u32;
            primitive_data.push(raw_data);
            primitive_headers.push(prim_header);
        }
        let mesh_header = GPUMeshHeader {
            length: total_length,
            num_primitives: num_primitives as u32,
            _padding: [0; 2],
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
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUVertexFromMesh {
    pub index: [u32; 2],
    pub mesh_index: u32,
    pub _padding: u32,
}

impl GPUVertexFromMesh {
    pub fn from_vertex_from_mesh(vertex_from_mesh: &VertexFromMesh) -> Self {
        Self {
            index: [vertex_from_mesh.index.0 as u32, vertex_from_mesh.index.1 as u32],
            mesh_index: vertex_from_mesh.mesh_index,
            _padding: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUNormFromMesh {
    pub index: [u32; 2],        // 8 bytes
    pub mesh_index: u32,        // 4 bytes
    pub _padding: u32,          // 4 bytes padding
    pub normal_transform: [f32; 12], // 48 bytes (3 rows of vec4), last element
}

impl GPUNormFromMesh {
    // Convert a 3x3 matrix to a 4x3 matrix
    // the last row is all 0
    fn matrix3_to_padded_array(matrix: &nalgebra::Matrix3<f32>) -> [f32; 12] {
        let mut array = [0.0; 12];
        for col in 0..3 {
            for row in 0..3 {
                array[col * 4 + row] = matrix[(row, col)];
            }
        }
        return array;
    }
    pub fn from_norm_from_mesh(norm_from_mesh: &NormFromMesh) -> Self {
        Self {
            index: [norm_from_mesh.index.0 as u32, norm_from_mesh.index.1 as u32],
            mesh_index: norm_from_mesh.mesh_index,
            _padding: 0,
            normal_transform: Self::matrix3_to_padded_array(&norm_from_mesh.normal_transform),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPURgbFromMesh {
    pub index: [u32; 2],
    pub mesh_index: u32,
    pub _padding: u32
}

impl GPURgbFromMesh {
    pub fn from_rgb_from_mesh(rgb_from_mesh: &RgbFromMesh) -> Self {
        Self {
            index: [rgb_from_mesh.index.0 as u32, rgb_from_mesh.index.1 as u32],
            mesh_index: rgb_from_mesh.mesh_index,
            _padding: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUDivertsRayFromMesh {
    pub index: [u32; 2],
    pub mesh_index: u32,
    pub _padding: u32,
}

impl GPUDivertsRayFromMesh {
    pub fn from_diverts_ray_from_mesh(diverts_ray_from_mesh: &DivertsRayFromMesh) -> Self {
        Self {
            index: [diverts_ray_from_mesh.index.0 as u32, diverts_ray_from_mesh.index.1 as u32],
            mesh_index: diverts_ray_from_mesh.mesh_index,
            _padding: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Deserialize, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUMeshTriangle {
    pub verts: GPUVertexFromMesh,
    pub norm: GPUNormFromMesh,
    pub rgb: GPURgbFromMesh,
    pub diverts_ray: GPUDivertsRayFromMesh,
    pub is_valid: u32,
    _padding: [u32; 3],
}

impl GPUMeshTriangle {
    pub fn get_empty() -> Self {
        Self {
            verts: GPUVertexFromMesh {
                index: [0; 2],
                mesh_index: 0,
                _padding: 0,
            },
            norm: GPUNormFromMesh {
                index: [0; 2],
                mesh_index: 0,
                _padding: 0,
                normal_transform: [0.0; 12],
            },
            rgb: GPURgbFromMesh {
                index: [0; 2],
                mesh_index: 0,
                _padding: 0,
            },
            diverts_ray: GPUDivertsRayFromMesh {
                index: [0; 2],
                mesh_index: 0,
                _padding: 0,
            },
            is_valid: 0,
            _padding: [0; 3],
        }
    }
    pub fn from_mesh_triangle(mesh_triangle: &MeshTriangle) -> Self {
        Self {
            verts: GPUVertexFromMesh::from_vertex_from_mesh(&mesh_triangle.verts),
            norm: GPUNormFromMesh::from_norm_from_mesh(&mesh_triangle.norm),
            rgb: GPURgbFromMesh::from_rgb_from_mesh(&mesh_triangle.rgb),
            diverts_ray: GPUDivertsRayFromMesh::from_diverts_ray_from_mesh(&mesh_triangle.diverts_ray),
            is_valid: 1,
            _padding: [0; 3],
        }
    }
}

assert_gpu_aligned!(GPUCamera);
assert_gpu_aligned!(GPURenderInfo);
assert_gpu_aligned!(GPUUniformDiffuseSpec);
assert_gpu_aligned!(GPUFreeTriangle);
assert_gpu_aligned!(GPUSphere);
assert_gpu_aligned!(GPUCubeMapFaceHeader);
assert_gpu_aligned!(GPURgbF32ImageHeader);
assert_gpu_aligned!(GPUMeshHeader);
assert_gpu_aligned!(GPUPrimitiveHeader);
assert_gpu_aligned!(GPUVertexFromMesh);
assert_gpu_aligned!(GPUNormFromMesh);
assert_gpu_aligned!(GPURgbFromMesh);
assert_gpu_aligned!(GPUDivertsRayFromMesh);
assert_gpu_aligned!(GPUMeshTriangle);

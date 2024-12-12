use crate::elements::mesh;
use crate::elements::sphere;
use crate::elements::distant_cube_map;
use crate::elements::triangle;

pub const GPU_NUM_MESH_BUFFERS: usize = 4;
pub type GPUElements = (Vec<sphere::Sphere>, Vec<distant_cube_map::DistantCubeMap>, Vec<triangle::FreeTriangle>, Vec<mesh::Mesh>);
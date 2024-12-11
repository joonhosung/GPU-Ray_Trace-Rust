use nalgebra::{Vector3, Vector2, Matrix4};
use crate::elements::Decomposable;
use crate::elements::Element;
use super::*;
use crate::material::*;

// so it begins .....


pub struct Mesh {
    // top layer of vec has each position as a single primitive
    pub poses: Vec<Vec<Vector3<f32>>>,
    pub norms: Vec<Vec<Vector3<f32>>>,
    pub indices: Vec<Vec<[usize; 3]>>, // each one represents a single triangle
    pub rgb_info: Vec<RgbInfo>,
    pub norm_info: Vec<Option<NormInfo>>,
    pub tangents: Vec<Option<Vec<Vector3<f32>>>>,
    pub metal_rough: Vec<PbrMetalRoughInfo>,

    pub textures: Vec<Option<UVRgb32FImage>>,
    pub normal_maps: Vec<Option<UVRgb32FImage>>,
    pub metal_rough_maps: Vec<Option<UVRgb32FImage>>,

    pub trans_mat: Matrix4<f32>,
}

impl Mesh {
    pub fn check_num_primitives(&self) {
        let num_primitives = self.poses.len();
        assert_eq!(num_primitives, self.norms.len());
        assert_eq!(num_primitives, self.indices.len());
        assert_eq!(num_primitives, self.rgb_info.len());
        assert_eq!(num_primitives, self.norm_info.len());
        assert_eq!(num_primitives, self.metal_rough.len());
        assert_eq!(num_primitives, self.textures.len());
        assert_eq!(num_primitives, self.normal_maps.len());
        assert_eq!(num_primitives, self.metal_rough_maps.len());
    }
}

pub struct PbrMetalRoughInfo {
    pub metal: f32,
    pub rough: f32,
    pub coords: Option<Vec<Vector2<f32>>>,
}

pub struct RgbInfo {
    pub factor: Vector3<f32>,
    pub coords: Option<Vec<Vector2<f32>>>,
}

pub struct NormInfo {
    pub scale: f32,
    pub coords: Vec<Vector2<f32>>,
}

impl Decomposable for Mesh {
    // the lifetime bound on this function was a solution that required my soul to find
    // allows me to create box of elements with a reference to the Mesh
    // that can exist for as long the Mesh does, skipping any useless Rc/Arc and crap
    fn decompose_to_elems<'e, 's>(&'s self, mesh_index: u32) -> Box<dyn Iterator<Item = Element<'e>> + 's> 
    where
        's : 'e,
    {
        Box::new(self.indices.iter().enumerate()
            .map(move |(p, idxs)| {
                (0..idxs.len()).map(
                    move |inner_idx| {
                        Box::new(MeshTriangle {
                            verts: VertexFromMesh {
                                index: (p, inner_idx),
                                mesh_index,
                                mesh: self,
                            },
                            norm: NormFromMesh::from_mesh_and_inner_idx(self, mesh_index, (p, inner_idx)),
    
                            // below needs to be updated when textures come!
                            diverts_ray: DivertsRayFromMesh{
                                index: (p, inner_idx),
                                mesh_index,
                                mesh: self,
                            },
                            rgb: RgbFromMesh{
                                index: (p, inner_idx),
                                mesh_index,
                                mesh: self,
                            },
                            type_name: "MeshTriangle".to_string(),
                        })} as Element<'e>)
            })
            .flatten())
    }
}
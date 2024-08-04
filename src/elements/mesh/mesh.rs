use nalgebra::{Vector3, Vector2};
use crate::elements::Decomposable;
use crate::elements::Element;
use super::*;
use crate::material::*;

// so it begins .....

pub struct Mesh {
    pub poses: Vec<Vector3<f32>>,
    pub norms: Vec<Vector3<f32>>,

    pub indices: Vec<[usize; 3]>, // each one represents a single triangle
    pub tex_coords: Vec<Vector2<f32>>,
    pub norm_coords: Vec<Vector2<f32>>,
    pub tangents: Option<Vec<Vector3<f32>>>,
    // all of the above likely need to be double wrapped by Vec instead of single
    // due to all above properties existing for any primitive under the mesh

    pub textures: Vec<UVRgb32FImage>, // indexed by primitive index
    pub normal_maps: Vec<UVRgb32FImage>, // indexed by primitive index
}

impl Decomposable for Mesh {
    // the lifetime bound on this function was a solution that required my soul to find
    // allows me to create box of elements with a reference to the Mesh
    // that can exist for as long the Mesh does, skipping any useless Rc/Arc and crap
    fn decompose_to_elems<'e, 's>(&'s self) -> Box<dyn Iterator<Item = Element<'e>> + 's> 
    where
        's : 'e,
    {
        Box::new((0..self.indices.len()).map(
                |inner_idx| {
                    Box::new(MeshTriangle {
                        verts: VertexFromMesh {
                            index: (0, inner_idx),
                            mesh: self,
                        },
                        norm: NormFromMesh::from_mesh_and_inner_idx(self, (0, inner_idx)),

                        // below needs to be updated when textures come!
                        mat: DiffuseSpecNoBaseMaterial{
                            divert_ray: DivertRayMethod::Spec,
                            emissive: None,
                        },
                        rgb: RgbFromMesh{
                            index: (0, inner_idx),
                            mesh: self,
                        },
                    })} as Element<'e>))
    }
}
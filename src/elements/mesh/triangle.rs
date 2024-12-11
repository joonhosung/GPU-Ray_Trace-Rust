use nalgebra::{Vector3, Vector2, Matrix3, Matrix3x2, Matrix2};
use crate::elements::triangle::{Triangle, GimmeNorm, GimmeRgb, DivertsRay};
use crate::ray::Ray;
use super::Mesh;
use std::ops::Index;
use std::iter::zip;
use crate::material::DynDiffSpec;

pub type MeshTriangle<'a> = Triangle<VertexFromMesh<'a>, NormFromMesh<'a>, RgbFromMesh<'a>, DivertsRayFromMesh<'a>>;

pub struct VertexFromMesh<'m> {
    pub index: (usize, usize),
    pub mesh_index: u32,
    pub mesh: &'m Mesh,
}

impl Index<usize> for VertexFromMesh<'_> {
    type Output = Vector3<f32>;

    fn index(&self, vert_idx: usize) -> &Vector3<f32> {
        let (prim_idx, inner_idx) = self.index;
        &self.mesh.poses[prim_idx][self.mesh.indices[prim_idx][inner_idx][vert_idx]]
    }
}

pub struct NormFromMesh<'m> {
    pub index: (usize, usize),
    pub normal_transform: Matrix3<f32>,
    pub mesh_index: u32,
    pub mesh: &'m Mesh,
}

//TODO: opt candidate
// All functions in this impl can be optimized to not thrash cache so much.
impl<'m> NormFromMesh<'m> {
    pub fn from_mesh_and_inner_idx(mesh: &'m Mesh, mesh_index: u32, full_idx: (usize, usize)) -> Self {
        NormFromMesh {
            index: full_idx,
            // norm_type: Self::generate_norm_type(mesh, full_idx),
            normal_transform: Self::generate_norm_type(mesh, full_idx),
            mesh_index,
            mesh,
        }
    }

    fn generate_norm_type(mesh: &Mesh, full_idx: (usize, usize)) -> Matrix3<f32> {
        // some help from https://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
        // to get the tangent to model space transform

        let (prim_idx, inner_idx) = full_idx;
        let indices = &mesh.indices[prim_idx][inner_idx];
        let trans_mat3 = mesh.trans_mat.try_inverse().expect("non invertible world?").transpose().fixed_resize::<3,3>(0.0); // special transform for normal vectors
        let face_norm = Self::get_face_norm(mesh, full_idx);

        match &mesh.norm_info[prim_idx] {
            Some(_) => {
                match &mesh.tangents[prim_idx] {
                    // _ => Uniform((trans_mat3 * face_norm).normalize()),
        
                    Some(tans) => {
                        // maybe we need to fix below calculation
                        // of tangent vector, might mess up tan -> mod transform for normal maps
                        let tan: Vector3<f32> = mesh.indices[prim_idx][inner_idx].iter()
                            .map(|i| tans[*i])
                            .sum();
                        let tan = tan.normalize();
                        let bitan = tan.cross(&face_norm);

                        let mut tang_to_mod: Matrix3<f32> = trans_mat3 * Matrix3::from_columns(&[tan.normalize(), bitan.normalize(), Vector3::zeros()]);
                        tang_to_mod.set_column(2, &face_norm);
                        for i in 0..3 {
                            tang_to_mod.set_column(i, &tang_to_mod.column(i).normalize());
                        }
        
                        tang_to_mod
                    },
                    // None => trans_mat3,
                    None => Self::norm_type_from_tex_coords(mesh, face_norm, (prim_idx, &indices), &trans_mat3),
                }
            },
            None => trans_mat3,
            // None => Self::norm_type_from_tex_coords(mesh, face_norm, (prim_idx, &indices), &trans_mat3),
        }
    }

    fn norm_type_from_tex_coords(mesh: &Mesh, face_norm: Vector3<f32>, full_idxs: (usize, &[usize; 3]), trans_mat3: &Matrix3<f32>) -> Matrix3<f32> {
        // use NormType::*;
        let (prim_idx, indices) = full_idxs;
        
        //TODO: opt candidate
        match &mesh.rgb_info[prim_idx].coords {
            Some(tex_coords) => {
                let t1 = tex_coords[indices[1]] - tex_coords[indices[0]];
                let t2 = tex_coords[indices[2]] - tex_coords[indices[0]];

                let tex_poses = Matrix2::from_columns(&[t1, t2]);
                match tex_poses.try_inverse() {
                    Some(inv_tex_poses) => {
                        let e1 = mesh.poses[prim_idx][indices[1]] - mesh.poses[prim_idx][indices[0]];
                        let e2 = mesh.poses[prim_idx][indices[2]] - mesh.poses[prim_idx][indices[0]];
                        
                        let mod_poses = Matrix3x2::from_columns(&[e1, e2]);

                        let incomplete = mod_poses * inv_tex_poses; // gives T and B as its columns

                        let mut tang_to_mod: Matrix3<f32> = incomplete.fixed_resize(0.0);
                        for i in 0..2 {
                            tang_to_mod.set_column(i, &tang_to_mod.column(i).normalize());
                        }
                        tang_to_mod = trans_mat3 * tang_to_mod;
                        tang_to_mod.set_column(2, &face_norm);
                        for i in 0..3 {
                            tang_to_mod.set_column(i, &tang_to_mod.column(i).normalize());
                        }

                        tang_to_mod
                    },
                    None => *trans_mat3,
                }
            },
            None => *trans_mat3,
        }
    }
    
    //TODO: opt candidate
    fn get_face_norm(mesh: &Mesh, full_idx: (usize, usize)) -> Vector3<f32> {
        let (prim_idx, inner_idx) = full_idx;
        let poses: Vec<Vector3<f32>> = mesh.indices[prim_idx][inner_idx].iter()
            .map(|i| mesh.poses[prim_idx][*i])
            .collect();
        let n = (poses[1] - poses[0]).cross(&(poses[2] - poses[0]));
        n.normalize()
    }
}

//TODO: opt candidate
impl GimmeNorm for NormFromMesh<'_> {
    fn get_norm(&self, barycentric: &(f32, f32)) -> Vector3<f32> {
        let (prim_idx, inner_idx) = self.index;
        // use NormType::*;
        match &self.mesh.norm_info[prim_idx] {
            Some(n_info) => {
                // let n_info = self.mesh.norm_info[prim_idx].as_ref().unwrap();
                let norm_coord = tex_coord_from_bary(self.mesh, &n_info.coords, barycentric, self.index);

                let norm = n_info.scale * self.normal_transform * self.mesh.normal_maps[prim_idx].as_ref().expect("no normal map???").get_pixel(norm_coord.x, norm_coord.y);
                norm.normalize()
            },
            None => { // just interpolate the normal vector from given
                let cum: Vector3<f32> = self.mesh.indices[prim_idx][inner_idx].iter()
                    .map(|i| self.normal_transform * self.mesh.norms[prim_idx][*i])
                    .sum();
                cum.normalize()
            }
        }
    }
}

pub struct RgbFromMesh<'m> {
    pub index: (usize, usize),
    pub mesh_index: u32,
    pub mesh: &'m Mesh,
}

//TODO: opt candidate
impl GimmeRgb for RgbFromMesh<'_> {
    fn get_rgb(&self, barycentric: &(f32, f32)) -> Vector3<f32> {
        let (prim_idx, _inner_idx) = self.index;
        match &self.mesh.rgb_info[prim_idx].coords {
            Some(tex_coords) => {
                let tex_coord = tex_coord_from_bary(self.mesh, &tex_coords, barycentric, self.index);
                self.mesh.rgb_info[prim_idx].factor.component_mul(&self.mesh.textures[prim_idx].as_ref().expect("no textures???").get_pixel(tex_coord.x, tex_coord.y))
            },
            None => self.mesh.rgb_info[prim_idx].factor,
        }
    }
}

pub struct DivertsRayFromMesh<'m> {
    pub index: (usize, usize),
    pub mesh_index: u32,
    pub mesh: &'m Mesh,
}

impl DivertsRay for DivertsRayFromMesh<'_> {
    type Seeding = (bool, f32); // (should_diff, roughness)

    //TODO: opt candidate
    fn divert_ray_seed(&self, ray: &Ray, norm: &Vector3<f32>, barycentric: &(f32, f32)) -> Self::Seeding {
        let (prim_idx, _inner_idx) = self.index;

        let (metalness, roughness) = match &self.mesh.metal_rough[prim_idx].coords {
            Some(coords) => {
                let mr_coord = tex_coord_from_bary(self.mesh, coords, barycentric, self.index);
                let mr_val = self.mesh.metal_rough_maps[prim_idx].as_ref().expect("no metal rough map???").get_pixel(mr_coord.x, mr_coord.y);
                (mr_val[2] * self.mesh.metal_rough[prim_idx].metal, mr_val[1] * self.mesh.metal_rough[prim_idx].rough)
            },
            None => (self.mesh.metal_rough[prim_idx].metal, self.mesh.metal_rough[prim_idx].rough),
        };
        
        const CUSTOM_ATTEN: f32 = 1.0; // attenuate metal because i think model didnt expect ray tracing!
        let r0 = 0.04 + (1.0 - 0.04) * metalness; // based on gltf definition of metalness for fresnel
        let reflectance = r0 + (1.0 - r0) * CUSTOM_ATTEN * (1.0 - (ray.d.dot(&norm)).abs().powf(5.0)); // schlick approximation

        (DynDiffSpec::should_diff(1.0 - reflectance), roughness)
    }

    fn divert_new_ray(&self, ray: &Ray, norm: &Vector3<f32>, o: &Vector3<f32>, seeding: &Self::Seeding) -> (Ray, f32) {
        let (should_diff, roughness) = *seeding;
        let (mut ray, p) = DynDiffSpec::gen_new_ray(ray, norm, o, should_diff);

        // we do roughness here, modify the ray
        let scatter: Vector3<f32> = {
            use rand::Rng;
            use nalgebra::vector;
            let u: f32 = crate::RNG.with_borrow_mut(|r| r.gen());
            let v: f32 = crate::RNG.with_borrow_mut(|r| r.gen());
            let w: f32 = crate::RNG.with_borrow_mut(|r| r.gen());
            roughness * vector![u,v,w].normalize()
        };

        ray.d = (ray.d + scatter).normalize();
        (ray, p)
    }
}

fn tex_coord_from_bary(mesh: &Mesh, coords: &Vec<Vector2<f32>>, barycentric: &(f32, f32), full_idx: (usize, usize)) -> Vector2<f32> {
    let (b1, b2) = *barycentric;
    let b0 = 1.0 - b2 - b1;
    let baryc: [f32; 3] = [b0, b1, b2];

    let (prim_idx, inner_idx) = full_idx;
    zip(mesh.indices[prim_idx][inner_idx].iter(), baryc.iter())
        .map(|(i, b)| coords[*i] * *b)
        .sum()
}

pub fn create_mesh_triangles_from_meshes(meshes: &Vec<Mesh>) -> Vec<MeshTriangle> {
    let mesh_triangles: Vec<MeshTriangle> = meshes.iter().enumerate().flat_map(|(mesh_idx, mesh)| {
        mesh.indices.iter().enumerate()
            .map(move |(p, idxs)| {
                (0..idxs.len()).map(move |inner_idx| {
                    let mesh_index = mesh_idx as u32;
                    MeshTriangle {
                        verts: VertexFromMesh {
                            index: (p, inner_idx),
                            mesh_index,
                            mesh,
                        },
                        norm: NormFromMesh::from_mesh_and_inner_idx(mesh, mesh_index, (p, inner_idx)),
                        diverts_ray: DivertsRayFromMesh {
                            index: (p, inner_idx),
                            mesh_index,
                            mesh,
                        },
                        rgb: RgbFromMesh {
                            index: (p, inner_idx),
                            mesh_index,
                            mesh,
                        },
                        type_name: "MeshTriangle".to_string(),
                    }
                })
            }).flatten()
    }).collect();
    return mesh_triangles;
}

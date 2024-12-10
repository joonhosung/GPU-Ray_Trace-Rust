use nalgebra::Vector3;
use crate::material::UniformDiffuseSpec;
use serde::Deserialize;
use crate::builder::Anim;

#[derive(Deserialize, Debug, Clone)]
pub struct FreeTriangle {
    pub verts: [Vector3<f32>; 3],
    pub norm: Vector3<f32>,

    pub rgb: Vector3<f32>,
    pub mat: UniformDiffuseSpec,
    pub animation: Option<Anim>,
}


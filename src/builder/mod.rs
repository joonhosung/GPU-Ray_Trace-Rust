use serde::Deserialize;
use serde_yaml;
use crate::render::RenderInfo;
use inner::*;
use nalgebra::Vector3;

mod inner;
mod pr;

#[derive(Deserialize, Debug)]
pub struct Scheme {
    pub render_info: RenderInfo,
    pub cam: pr::Cam,
    pub scene_members: VecInto<MemberTypes>,
}

#[derive(Debug, Deserialize)]
pub struct Anim {
    framerate: f32,
    keyframes: Vec<Keyframe>,
}

#[derive(Debug, Deserialize)]
pub struct Keyframe {
    translation: Vector3<f32>, // Same format as Model.translation
    ease_type: String, // TODO: Update to ease_type enum
    time: f32, //in seconds

}

impl Scheme {
    pub fn from_yml(contents: String) -> Scheme {
        let scheme: Scheme = serde_yaml::from_str(&contents).expect("dodnt parse!!");
        scheme.apply_corrections()
    }

    fn apply_corrections(mut self) -> Self {
        self.cam.up = self.cam.up.normalize();
        self
    }
}
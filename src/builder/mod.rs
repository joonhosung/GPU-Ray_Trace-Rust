use keyframe::EasingFunction;
use serde::Deserialize;
use serde_yaml;
use crate::render::RenderInfo;
use inner::*;
use nalgebra::Vector3;
// use keyframe::mint::Point3;
// use keyframe::{ease, functions};

pub mod inner;
mod pr;

#[derive(Deserialize, Debug, Clone)]
pub struct Scheme {
    pub render_info: RenderInfo,
    pub cam: pr::Cam,
    pub scene_members: VecInto<MemberTypes>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Anim {
    framerate: Option<f32>,
    keyframes: Vec<Keyframe>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Keyframe {
    translation: Vector3<f32>, // Same format as Model.translation
    ease_type: Option<String>, // TODO: Update to ease_type enum
    time: f32, //in seconds

}

impl Keyframe {
    pub fn get_ease_type (&self) -> Box<dyn EasingFunction + Send + Sync>{
        // Default option for ease_type is EastInOut. Same as if no easing function is given in the Keyframe crate
        match self.ease_type.clone().unwrap_or("EaseInOut".to_string()).as_str() {
            "EaseIn" => {Box::new(keyframe::functions::EaseIn)},
            "EaseInCubic" => {Box::new(keyframe::functions::EaseInCubic)},
            "EaseInOut" => {Box::new(keyframe::functions::EaseInOut)},
            "EaseInOutCubic" => {Box::new(keyframe::functions::EaseInOutCubic)},
            "EaseInOutQuad" => {Box::new(keyframe::functions::EaseInOutQuad)},
            "EaseInOutQuart" => {Box::new(keyframe::functions::EaseInOutQuart)},
            "EaseInOutQuint" => {Box::new(keyframe::functions::EaseInOutQuint)},
            "EaseInQuad" => {Box::new(keyframe::functions::EaseInQuad)},
            "EaseInQuart" => {Box::new(keyframe::functions::EaseInQuart)},
            "EaseInQuint" => {Box::new(keyframe::functions::EaseInQuint)},
            "EaseOut" => {Box::new(keyframe::functions::EaseOut)},
            "EaseOutCubic" => {Box::new(keyframe::functions::EaseOutCubic)},
            "EaseOutQuad" => {Box::new(keyframe::functions::EaseOutQuad)},
            "EaseOutQuart" => {Box::new(keyframe::functions::EaseOutQuart)},
            "EaseOutQuint" => {Box::new(keyframe::functions::EaseOutQuint)},
            "Hold" => {Box::new(keyframe::functions::Hold)},
            "Linear" => {Box::new(keyframe::functions::Linear)},
            "Step" => {Box::new(keyframe::functions::Step)},
            func => {panic!("Unsupported easing function: {func}")}
        }
    }
}


impl Scheme {
    pub fn from_yml(contents: String) -> Scheme {
        let scheme: Scheme = serde_yaml::from_str(&contents).expect("didn't parse!!");
        scheme.apply_corrections()
    }

    fn apply_corrections(mut self) -> Self {
        self.cam.up = self.cam.up.normalize();
        self
    }
}
use serde::Deserialize;
use super::radiance::RadianceInfo;
#[derive(Deserialize, Debug, Clone, Copy)]
pub struct RenderInfo {
    pub width: i32,
    pub height: i32,
    pub samps_per_pix: i32,
    pub rad_info: RadianceInfo,
    pub kd_tree_depth: usize,
    pub use_gpu: Option<bool>,
    pub animation: Option<bool>,
    pub framerate: Option<f32>,
}

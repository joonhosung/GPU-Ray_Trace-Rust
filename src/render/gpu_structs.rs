use serde::Deserialize;
use bytemuck;
use crate::scene::Cam;

use super::RenderInfo;

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
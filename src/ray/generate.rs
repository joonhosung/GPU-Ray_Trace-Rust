use crate::scene::Cam;
use nalgebra::Vector3;
use crate::render_target::RenderTarget;
use super::Ray;

pub struct RayCompute {
    x_cf: f32, y_cf: f32,
    right: Vector3<f32>,
    x_off: f32, y_off: f32,
}

impl RayCompute {
    pub fn new(render_target: &RenderTarget, cam: &Cam) -> Self {
        let canv_width = render_target.canv_width;
        let canv_height = render_target.canv_height;
        let x_cf = cam.screen_width / canv_width as f32;
        let y_cf = cam.screen_height / canv_height as f32;

        Self {
            x_cf, y_cf,
            right: cam.d.normalize().cross(&cam.up),
            x_off: (canv_width as f32) / 2.0,
            y_off: (canv_height as f32) / 2.0,
        }
    }
    pub fn pix_cam_to_rand_ray(&self, (x, y): (i32, i32), cam: &Cam) -> Ray { // randomly make over unit square of in-scene pixel
        let mut ray = self.pix_cam_raw_ray((x,y), cam);

        let up = &cam.up;
        let right = &self.right;

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let u: f32 = rng.gen::<f32>() - 0.5;
        let v: f32 = rng.gen::<f32>() - 0.5;

        ray.d = ray.d + right * u * self.x_cf + up * v * self.y_cf;
        ray.d = ray.d.normalize();
        ray
    }
    // pub fn pix_cam_to_ray(&self, (x, y): (i32, i32), cam: &Cam) -> Ray {
    //     let mut ray = self.pix_cam_raw_ray((x,y), cam);
    //     ray.d = ray.d.normalize();
    //     ray
    // }

    fn pix_cam_raw_ray(&self, (x, y): (i32, i32), cam: &Cam) -> Ray { 
        let up = &cam.up;
        let right = &self.right;
    
        let s_x: f32 = self.x_cf * (x as f32 - self.x_off);
        let s_y: f32 = self.y_cf * (y as f32 - self.y_off);
    
        let d = cam.d + s_x * right + s_y * up;

        Ray{d, o: cam.o}
    }
}
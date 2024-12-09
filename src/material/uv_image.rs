use image::{Pixel, Rgb32FImage};
use nalgebra::Vector3;

pub struct UVRgb32FImage (Rgb32FImage);

impl UVRgb32FImage {
    pub fn get_width(&self) -> u32 { self.0.width() }
    pub fn get_height(&self) -> u32 { self.0.height() }
    pub fn get_pixel(&self, u: f32, v: f32) -> Vector3<f32> {
        let face = &self.0;
        let width = face.width() as f32;
        let height = face.height() as f32;

        let rgb: Vec<f32> = face
            .get_pixel(
                (u * width).max(0.0).min(width-1.0).trunc() as u32,
                (v * height).max(0.0).min(height-1.0).trunc() as u32
                )
            .channels()
            .to_vec();
        let rgb: [f32; 3] = rgb.try_into().unwrap();
        rgb.into()
    }
    pub fn as_raw(&self) -> Vec<f32> {
        self.0.pixels().flat_map(|p| p.channels().to_vec()).collect()
    }
}

impl From<Rgb32FImage> for UVRgb32FImage {
    fn from(im: Rgb32FImage) -> Self { UVRgb32FImage(im) }
}
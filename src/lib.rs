use std::sync::Arc;
use egui::mutex::Mutex;
// use ray_trace_rust::ui_util::io_on_render_out;
pub use render::{RenderInfo, RenderTarget};
pub use builder::Scheme;
mod scene;
mod render;
mod ray;
mod material;
mod builder;
mod elements;
mod accel;
pub mod renderer;
pub mod ui_util;

pub type ArcMux<T> = Arc<Mutex<T>>;
pub type BufferMux = Arc<Mutex<Vec<u8>>>;

const EPS: f32 = 1e-4;

use std::cell::RefCell;
use rand::rngs::ThreadRng;

thread_local! {
    pub static RNG: RefCell<ThreadRng> = rand::thread_rng().into();
}
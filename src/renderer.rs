use std::thread;
use std::sync::Arc;
use egui::mutex::Mutex;
use crate::scene::Scene;
use std::sync::mpsc::{channel, Sender, Receiver};
// use ray_trace_rust::ui_util::io_on_render_out;
pub use crate::render::{RenderTarget, render_to_target};
pub use crate::builder::Scheme;
use crate::ui_util;


pub struct RenderOut {
    pub buf_q: Receiver<Vec<u8>>,
}

#[derive(Clone)]
pub struct Renderer {
    target: RenderTarget,
    sender: Sender<Vec<u8>>,

    scheme: Scheme,
}

impl Renderer {
    pub fn new(canv_width: i32, canv_height: i32, scheme: Scheme) -> (Self, RenderOut) {
        let buf: Vec<u8> = [0, 0, 0, 0].repeat((canv_width * canv_height).try_into().unwrap());
        let (tx, rx) = channel();
        let target = RenderTarget {
            buff_mux: Arc::new(Mutex::new(buf)),
            canv_width, canv_height,
        };
        (Self {
            target,
            sender: tx,
            scheme: scheme,
        }, RenderOut{buf_q: rx})
    }
    pub fn consume_and_do(self) {
        thread::spawn(move || {
            // let Scheme {
            //     cam, render_info, scene_members,
            //     ..
            // } = self.scheme;
            let renderer_inner = self.clone();
            let skene = Scene { cam: renderer_inner.scheme.cam.into(), members: renderer_inner.scheme.scene_members.into() };

            render_to_target(&renderer_inner.target, &skene, || self.update_output(), &renderer_inner.scheme.render_info);
        });
    }

    pub fn consume_and_do_anim(self, ui_mode: bool) {
        
        
        // let Scheme {
            //     cam, render_info, scene_members,
            //     ..
            // } = self.scheme.take().scheme;
            
        // let skene = Scene { cam: scheme.cam.into(), members: scheme.scene_members.into() };
        
        // Extract the locations of all scene members for each frame of the animation
        let mut frame_num = 0;
        for frame_scheme in self.extract_frames() {
            frame_num += 1;
            thread::spawn(move || {
                
                let (region_width, region_height) = (frame_scheme.render_info.width, frame_scheme.render_info.height);
                let region_width_inner = region_width.clone();
                let region_height_inner = region_height.clone();
                let ui_mode_inner = ui_mode.clone();

                let (buffer_renderer, render_out) = Renderer::new(region_width_inner, region_height_inner, frame_scheme.clone());

                // Send to normal renderer for each frame
                buffer_renderer.consume_and_do();

                // Render receiver
                ui_util::io_on_render_out(render_out, (region_width_inner, region_height_inner), ui_mode_inner, Some(format!("frame_{frame_num}.png")));
            }).join().unwrap();
        }
    }

    fn extract_frames(self) -> Vec<Scheme> {
        let mut schemes: Vec<Scheme> = Vec::new();

        schemes.push(self.scheme.clone());
        schemes.push(self.scheme.clone());
        schemes
    }

    fn update_output(&self) {
        self.sender.send(self.target.buff_mux.lock().clone()).expect("cannot send??");
    }
}
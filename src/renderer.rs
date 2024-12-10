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
    pub fn new(canv_width: i32, canv_height: i32, scheme: Scheme) -> (Self, Receiver<Vec<u8>>) {
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
        }, rx)
    }

    pub fn consume_and_do(self) {
        thread::spawn(move || {
            let renderer_inner = self.clone();
            let skene = Scene { cam: renderer_inner.scheme.cam.into(), members: renderer_inner.scheme.scene_members.into() };

            render_to_target(&self.target, &skene, || self.update_output(), &self.scheme.render_info);
        });
    }

    pub fn consume_and_do_anim(self, ui_mode: bool) {
        
        let (region_width, region_height, render_info) = (self.scheme.render_info.width, self.scheme.render_info.height, self.scheme.render_info);

        // After extracting the locations of the each frame, render them
        let mut frame_num = 0;
        let extracted_frames = self.extract_frames();
        println!("Extracted {} frames", extracted_frames.len());
        for frame_scene in extracted_frames {
            println!("Generating frame #{frame_num}");
            let (tx, render_out) = channel();
            thread::spawn(move || {
                
                let region_width_inner = region_width.clone();
                let region_height_inner = region_height.clone();
                // let ui_mode_inner = ui_mode.clone();
                
                // let (buffer_renderer, render_out) = Renderer::new(region_width_inner, region_height_inner, frame_scheme);
                
                let buf: Vec<u8> = [0, 0, 0, 0].repeat((region_width_inner * region_height_inner).try_into().unwrap());
                let target = RenderTarget {
                    buff_mux: Arc::new(Mutex::new(buf)),
                    canv_width: region_width_inner, 
                    canv_height: region_height_inner,
                };
                // let skene = Scene { cam: frame_scheme.cam.into(), members: frame_scheme.scene_members.into() };
                // Send to normal renderer for each frame
                // buffer_renderer.consume_and_do();
                
                render_to_target(&target, &frame_scene, || tx.send(target.buff_mux.lock().clone()).expect("cannot send??"), &render_info);
                println!("WITHIN THREAD: Finished outputting frame #{frame_num}");
                // Render receiver
            }).join().unwrap();
            
            ui_util::io_on_render_out(render_out, (region_width.clone(), region_height.clone()), ui_mode.clone(), Some(format!("anim_frames/frame_{frame_num}.png")));
            println!("Finished outputting frame #{frame_num}");
            frame_num += 1;
        }
    }

    // Extract the locations of all scene members for each frame of the animation
    fn extract_frames<'a>(self) -> Vec<Scene<'a>> {
        let mut scenes: Vec<Scene> = Vec::new();
        let updated_locations = self.scheme.clone();
        
        for member_frame in self.scheme.clone().scene_members.extract_anim(updated_locations.render_info.framerate.unwrap()) {
            // println!("Extracted frame: {member_frame:?}");
            let skene: Scene =  Scene { cam: updated_locations.clone().cam.into(), members: member_frame.into() };
            scenes.push(skene);
        }
        scenes
    }

    fn update_output(&self) {
        self.sender.send(self.target.buff_mux.lock().clone()).expect("cannot send??");
    }
}
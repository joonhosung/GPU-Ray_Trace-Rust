use ray_trace_rust::{Scheme, ui_util::io_on_render_out};
use ray_trace_rust::renderer::Renderer;
use std::fs::File;
use std::path::Path;
use std::io::prelude::*;
use std::env;

fn main() {
    env_logger::init();

    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {panic!("no yaml path specified");},
    };
    let ui_mode = match env::args().nth(2) {
        Some(u) => !(u == "no_ui"),
        None => true,
    };

    let paff = Path::new(&path);
    let mut file = File::open(&paff).expect("file boss???");
    let mut scheme_dat = String::new();
    file.read_to_string(&mut scheme_dat).unwrap();

    let scheme = Scheme::from_yml(scheme_dat);
    
    let (region_width, region_height) = (scheme.render_info.width, scheme.render_info.height);

    // let Scheme {
    //     cam, render_info, scene_members,
    //     ..
    // } = scheme.take().unwrap();

    
    // Normal rendering
    if !scheme.render_info.animation {
        let (buffer_renderer, render_out) = Renderer::new(region_width, region_height, scheme);
        buffer_renderer.consume_and_do();
        io_on_render_out(render_out, (region_width, region_height), ui_mode, None);
    } 
    
    // Animation rendering
    else { 
        let (buffer_renderer, _) = Renderer::new(region_width, region_height, scheme);
        buffer_renderer.consume_and_do_anim(ui_mode);

    }
    
}
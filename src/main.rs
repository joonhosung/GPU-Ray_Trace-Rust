use image::open;
use minimp4::Mp4Muxer;
use openh264::formats::YUVBuffer;
use ray_trace_rust::{Scheme, ui_util::io_on_render_out};
use ray_trace_rust::renderer::Renderer;
use std::fs::{self, File};
use std::path::Path;
use std::io::{prelude::*, Cursor, SeekFrom};
use std::env;
// use opencv::videoio::VideoWriter;
// use opencv::videoio::VideoCapture;
use openh264::encoder::{Encoder, EncoderConfig};

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
        let framerate: f32 = scheme.render_info.framerate.unwrap();
        
        // COMMENT FOR NO RENDERING!
        // let _ = fs::remove_dir_all("./anim_frames"); // Remove previous run if it exists
        // fs::create_dir_all("./anim_frames").expect("Couldn't create ./anim_frames directory...");
        // let (buffer_renderer, _) = Renderer::new(region_width, region_height, scheme);
        // buffer_renderer.consume_and_do_anim(ui_mode);



        // OpenH264 encoder config. Aim for maximum quality
        let encoder_config = EncoderConfig::new()
                                    .set_bitrate_bps(32 * region_height as u32 * region_width as u32 * framerate as u32) // Raw max bitrate: (RGBa size x #pixels x framerate)
                                    .max_frame_rate(framerate)
                                    .enable_skip_frame(false)
                                    .rate_control_mode(openh264::encoder::RateControlMode::Off);

        // OpenH264 encoder
        let mut encoder = Encoder::with_api_config(openh264::OpenH264API::from_source(), encoder_config).expect("??????");
        let mut vid_buf = Vec::new();

        let mut frames_dir: Vec<String> = fs::read_dir("./anim_frames").expect("frames not found??").map(|d| d.unwrap().file_name().into_string().unwrap()).collect();
        // println!("Sorted paths: {frames_dir:?}");

        frames_dir.sort_by(|a, b| Ord::cmp(&a.split(".").next().unwrap().parse::<usize>().unwrap() , &b.split(".").next().unwrap().parse::<usize>().unwrap()));

        // println!("Sorted paths after sort: {frames_dir:?}");

        for frame_path in frames_dir {
            // println!("Image Name: ./anim_frames/{frame_path}");
            let image = open(format!("./anim_frames/{frame_path}")).expect("Couldn't open image as RGBaU8?");

            let h264_slice = openh264::formats::RgbaSliceU8::new(image.as_rgba8().unwrap(), (region_width as usize, region_height as usize));
            let yuv_buffer = YUVBuffer::from_rgb_source(h264_slice);
            let bitstream = encoder.encode(&yuv_buffer).unwrap();
            bitstream.write_vec(&mut vid_buf);
        }

        let mut video_buffer = Cursor::new(Vec::new());
        let mut mp4_mux = Mp4Muxer::new(&mut video_buffer);

        mp4_mux.init_video(region_width, region_height, false, "animation");
        mp4_mux.write_video_with_fps(&vid_buf, framerate as u32);
        mp4_mux.close();
        
        video_buffer.seek(SeekFrom::Start(0)).unwrap();
        let mut video_bytes = Vec::new();
        video_buffer.read_to_end(&mut video_bytes).unwrap();

        std::fs::write("animation.mp4", &video_bytes).expect("Couldn't write to .mp4");
    }
    
}
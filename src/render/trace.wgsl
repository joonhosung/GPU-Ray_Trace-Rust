struct Camera {
    direction: vec4<f32>,
    origin: vec4<f32>,
    up: vec4<f32>,
    screen_dims: vec2<f32>,
    lens_radius: f32,
    padding: f32,
}

struct RenderInfo {
    width: u32,
    height: u32,
    samps_per_pix: u32,
    assured_depth: u32,
    max_threshold: f32,
    kd_tree_depth: u32,
    debug_single_ray: u32,
    dir_light_samp: u32,
}

fn get_pixel_index(x: u32, y: u32, width: u32) -> u32 {
    return 4 * (y * width + x);
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<uniform> render_info: RenderInfo;

// For better precision, each pixel is represented by 4 floats (RGBA)
@group(1) @binding(0)
var<storage, read_write> render_target: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Just to verify we can access the structs (these don't affect the output)
    let cam_dir = camera.direction;
    let info_width = render_info.width;

    let pixel_index = get_pixel_index(global_id.x, global_id.y, info_width);
    render_target[pixel_index] += 1.0;
    render_target[pixel_index + 1] += 1.0;
    render_target[pixel_index + 2] += 1.0;
    render_target[pixel_index + 3] += 1.0;
}

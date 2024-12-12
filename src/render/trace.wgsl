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

@group(2) @binding(0)
var<storage, read> mesh_chunk_0: array<f32>;

@group(2) @binding(1)
var<storage, read> mesh_chunk_1: array<f32>;

@group(2) @binding(2)
var<storage, read> mesh_chunk_2: array<f32>;

@group(2) @binding(3)
var<storage, read> mesh_chunk_3: array<f32>;

@group(2) @binding(4)
var<storage, read> mesh_triangles: array<f32>;

@group(3) @binding(0)
var<storage, read> spheres: array<f32>;

@group(3) @binding(1)
var<storage, read> cube_maps: array<f32>;

@group(3) @binding(2)
var<storage, read> free_triangles: array<f32>;


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Just to verify we can access the structs (these don't affect the output)
    let cam_dir = camera.direction;
    let info_width = render_info.width;

    let pixel_index = get_pixel_index(global_id.x, global_id.y, info_width);
    render_target[pixel_index] += 1.0;     // R
    render_target[pixel_index + 1] += 1.0; // G
    render_target[pixel_index + 2] += 1.0; // B
    render_target[pixel_index + 3] += 1.0; // A

    var seed = initRng();
    // Randomize pixels for run
    // for (var i: u32 = 0; i < arrayLength(&render_target); i++) {
    //     render_target[i] = get_random_f32(&seed);
    // }
    // render_target[pixel_index] = get_random_f32(&seed);
    // render_target[pixel_index+1] = get_random_f32(&seed);
    // render_target[pixel_index+2] = get_random_f32(&seed);
    // render_target[pixel_index+3] = get_random_f32(&seed);

// Shader algorithm for now:
    // For loop for samps_per_pix
    // for (i < samps_per_pix) 
    //     (do rendering)
    //    from this pixel:
    //         generate random ray based on generate.rs (function 1) (Jackson)
    //         with this random ray:
    //              get the first object hit based on intersect (function 2) (Jun Ho) (kd tree can help reduce this by a LOT)
    //        (Loop 1) if there is a hit, until the minimum assured bounces: (russian roullette filter)
    //                  See if we continue the ray. If continue ray: (function 3)
    //                       Return colour and generate new ray from the ray that just got hit (function 4)
    //                       With the new ray reflected off the hit object, Loop 1 again.
    //                         establish_dls_contrib doesn't get called at any scheme now. Don't implement??
    //                  If don't continue:
    //                       Just update the colour of the pixel
    // Update pixel using incoming_rgb from the last loop 1.

    // Struct 1: RAY
    // Two vectors called d & o (direction & origin)

}

// fn get_intersect(dir_vec, orig_vec)
// Returns hit index and ray length


fn get_random_f32(seed: ptr<function, u32>) -> f32 {
    // let seed = 88888888u;
    let newState = *seed * 747796405u + 2891336453u;
    *seed = newState;
    let word = ((newState >> ((newState >> 28u) + 4u)) ^ newState) * 277803737u;
    let x = (word >> 22u) ^ word;
    return f32(x) / f32(0xffffffffu);
}


// fn initRng(pixel: vec2<u32>, resolution: vec2<u32>, frame: u32) -> u32 {
fn initRng() -> u32 {
    // Adapted from https://github.com/boksajak/referencePT
    // let seed = dot(pixel, vec2<u32>(1u, resolution.x)) ^ jenkinsHash(frame);
    let seed = 88888888u ^ jenkinsHash(12345678u);
    return jenkinsHash(seed);
}

fn jenkinsHash(input: u32) -> u32 {
    var x = input;
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}

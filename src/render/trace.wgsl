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

struct UniformDiffuseSpec {
    emissive: vec3<f32>,
    has_emissive: u32,
    divert_ray_type: u32,
    diffp: f32,      // For DiffSpec
    n_out: f32,      // For Dielectric
    n_in: f32,       // For Dielectric
}

struct Sphere {
    center: vec4<f32>,
    coloring: vec4<f32>,
    material: UniformDiffuseSpec,
    radius: f32,
    padding: vec3<f32>,
}

struct FreeTriangle {
    vert1: vec4<f32>,
    vert2: vec4<f32>,
    vert3: vec4<f32>,
    norm: vec4<f32>,
    rgb: vec4<f32>,
    material: UniformDiffuseSpec,
}

struct CubeMapFaceHeader {
    width: u32,
    height: u32,
    uv_scale_x: f32,
    uv_scale_y: f32,
}

// Is this right? 6 arrays of data 
// struct CubeMapData {
//     headers: array<CubeMapFaceHeader, 6>,
//     data: array<array<f32>, 6>,
// }


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
var<storage, read> spheres: array<Sphere>;

@group(3) @binding(1)
var<storage, read> cube_maps: array<f32>;

@group(3) @binding(2)
var<storage, read> free_triangles: array<FreeTriangle>;


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

    // let == immutable
    // var == mutable!!

    // FIXME: Sanity for Jackson in the morning to try with camera ray generation sanity (rasterization)

    // PUT PIXEL GENRATION HERE
    let ray_dir = vec3<f32>(0f,0f,0f); // placeholder for ray vector
    let ray_orig = vec3<f32>(0f,0f,0f); 

    let ray_intersect = get_ray_intersect(ray_dir, ray_orig);

    render_target[pixel_index] = ray_intersect.colour.x;
    render_target[pixel_index+1] = ray_intersect.colour.y;
    render_target[pixel_index+2] = ray_intersect.colour.z;
    render_target[pixel_index+3] = ray_intersect.colour.w;
}

// Pseudo-enum for element types
const MESH = 0u;
const SPHERE = 1u;
const CUBEMAP = 2u;
const NONE = 3u;

struct Intersection {
    // Try to get colour information here too?
    colour: vec4<f32>,
    element_type: u32,
    element_idx: u32,
    has_bounce: bool,
    ray_distance: f32,
}

const MAXF = 0x1.fffffep+127f;

fn get_ray_intersect(ray_dir: vec3<f32>, ray_orig: vec3<f32>) -> Intersection {
    
    // Initialize intersect struct
    var intersect = Intersection(vec4<f32>(0f, 0f, 0f, 0f), NONE, 0u, false, 0f);
    var closest_intersect = MAXF;
    

    // Iterate through every sphere 
    for(var i = 0u; i < arrayLength(&spheres); i++) { 
        let got_dist = get_sphere_intersect(ray_dir, ray_orig, i);
        if got_dist != -1f {
            if got_dist < closest_intersect {
                closest_intersect = got_dist;
                
                intersect = Intersection(spheres[i].coloring, NONE, i, false, got_dist);
            }
        }
    }

    // Iterate through every free triangle
    // for(var i = 0u; i < arrayLength(&free_triangles); i++) {  
    
    // Iterate through every mesh triangle
    // for(var i = 0u; i < arrayLength(&meshes TODO: what's the best thing to iterate with??); i++) {  

    // If no hit get the cubemap background color
    if intersect.element_type == NONE {
        // Just grey for now. Add intersection later
        intersect = Intersection(vec4<f32>(0.5f, 0.5f, 0.5f, 1f), CUBEMAP, 0u, false, MAXF);
    }

    return intersect;
}
// Returns hit index and ray length

fn get_sphere_intersect(ray_dir: vec3<f32>, ray_orig: vec3<f32>, i: u32) -> f32 {
    let oc = ray_orig - spheres[i].center.xyz;
    let dir = dot(ray_dir, oc);
    let consts = dot(oc, oc) - (spheres[i].radius * spheres[i].radius);

    let discr = (dir * dir) - consts;

    // If the ray crosses the sphere, return the colour of the closer intersection
    if discr > 0.0 { 
        let offset = -dir;
        let thing = sqrt(discr);
        let intersect_dist_a = offset - thing;
        let intersect_dist_b = offset + thing;

        if (intersect_dist_a > 0.001f) && (intersect_dist_a < intersect_dist_b) {
            return intersect_dist_a;
        } else if (intersect_dist_b > 0.001f) && (intersect_dist_a > intersect_dist_b) {
            return intersect_dist_b;
        }
        
        // distance can't be negative
        return f32(-1.0); 
        // TODO: Should calculate how the ray is diverted
    }

    return f32(-1.0); 
}

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

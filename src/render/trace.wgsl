const MESH = 0u;
const SPHERE = 1u;
const CUBEMAP = 2u;
const NONE = 3u;
const MAXF = 0x1.fffffep+127f;

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
    radius: f32,
    is_valid: u32,
    padding: vec2<f32>,
    material: UniformDiffuseSpec,
}

struct FreeTriangle {
    vert1: vec4<f32>,
    vert2: vec4<f32>,
    vert3: vec4<f32>,
    norm: vec4<f32>,
    rgb: vec4<f32>,
    is_valid: u32,
    padding: vec3<f32>,
    material: UniformDiffuseSpec,
}

struct CubeMapFaceHeader {
    width: u32,
    height: u32,
    uv_scale_x: f32,
    uv_scale_y: f32,
}

struct VertexFromMesh {
    index: vec2<u32>,
    mesh_index: u32,
    padding: u32,
}

struct NormFromMesh {
    index: vec2<u32>,
    mesh_index: u32,
    padding: u32,
    normal_transform: mat3x4<f32>, // Last row is padded to all zeros
}

struct RgbFromMesh {
    index: vec2<u32>,
    mesh_index: u32,
    padding: u32,
}

struct DivertsRayFromMesh {
    index: vec2<u32>,
    mesh_index: u32,
    padding: u32,
}

struct MeshTriangle {
    verts: VertexFromMesh,
    norms: NormFromMesh,
    rgb: RgbFromMesh,
    diverts_ray: DivertsRayFromMesh,
    is_valid: u32,
    padding: vec3<f32>,
}

struct Ray {
    direction: vec3<f32>,
    origin: vec3<f32>,
}

struct RayCompute {
    x_coef: f32,
    y_coef: f32,
    right: vec3<f32>,
    x_offset: f32,
    y_offset: f32,
}

struct Intersection {
    // Try to get colour information here too?
    colour: vec4<f32>,
    element_type: u32,
    element_idx: u32,
    has_bounce: bool,
    ray_distance: f32,
}

// Is this right? 6 arrays of data 
// struct CubeMapData {
//     headers: array<CubeMapFaceHeader, 6>,
//     data: array<array<f32>, 6>,
// }


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
var<storage, read> mesh_triangles: array<MeshTriangle>;

@group(3) @binding(0)
var<storage, read> spheres: array<Sphere>;

@group(3) @binding(1)
var<storage, read> cube_maps: array<f32>;

@group(3) @binding(2)
var<storage, read> free_triangles: array<FreeTriangle>;


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_index = get_pixel_index(global_id.x, global_id.y, render_info.width);
    var seed = initRng();
    let ray_compute = create_ray_compute(vec2<u32>(render_info.width, render_info.height), camera);
    var sample_count = 0.0;
    // FIXME: Sanity for Jackson in the morning to try with camera ray generation sanity (rasterization)
    // PUT PIXEL GENRATION HERE
    for (var i = 0u; i < render_info.samps_per_pix; i += 1u) {
        let ray = pix_cam_to_rand_ray(ray_compute, vec2<u32>(global_id.x, global_id.y), camera, &seed);
        let ray_intersect = get_ray_intersect(ray);
        render_target[pixel_index] = (ray_intersect.colour.x + (render_target[pixel_index] * sample_count)) / (sample_count + 1.0);
        render_target[pixel_index + 1] = (ray_intersect.colour.y + (render_target[pixel_index + 1] * sample_count)) / (sample_count + 1.0);
        render_target[pixel_index + 2] = (ray_intersect.colour.z + (render_target[pixel_index + 2] * sample_count)) / (sample_count + 1.0);
        render_target[pixel_index + 3] = (ray_intersect.colour.w + (render_target[pixel_index + 3] * sample_count)) / (sample_count + 1.0);
        
        sample_count += 1.0;
    }

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
}

// Pseudo-enum for element types


fn get_pixel_index(x: u32, y: u32, width: u32) -> u32 {
    return 4 * (y * width + x);
}

fn create_ray_compute(canvas_dims: vec2<u32>, camera: Camera) -> RayCompute {
    let canvas_dims_f32 = vec2<f32>(f32(canvas_dims.x), f32(canvas_dims.y));
    let x_cf = camera.screen_dims.x / canvas_dims_f32.x;
    let y_cf = camera.screen_dims.y / canvas_dims_f32.y;

    return RayCompute(
        x_cf,
        y_cf,
        normalize(cross(normalize(camera.direction.xyz), camera.up.xyz)),
        camera.screen_dims.x / 2.0,
        camera.screen_dims.y / 2.0,
    );
}

fn pix_cam_to_rand_ray(compute: RayCompute, pixel: vec2<u32>, camera: Camera, rng: ptr<function, u32>) -> Ray {
    var ray = pix_cam_raw_ray(compute, pixel, camera, rng);

    // Random offset in [-0.5, 0.5]
    let u = get_random_f32(rng) - 0.5;
    let v = get_random_f32(rng) - 0.5;

    ray.direction = ray.direction + 
        compute.right * u * compute.x_coef + 
        camera.up.xyz * v * compute.y_coef;
    ray.direction = normalize(ray.direction);
    
    return ray;
}

fn pix_cam_raw_ray(compute: RayCompute, pixel: vec2<u32>, camera: Camera, rng: ptr<function, u32>) -> Ray {
    let s_x = compute.x_coef * (f32(pixel.x) - compute.x_offset);
    let s_y = compute.y_coef * (f32(pixel.y) - compute.y_offset);

    let direction = camera.direction.xyz + s_x * compute.right + s_y * camera.up.xyz;

    if (camera.lens_radius != 0.0) {
        // Random numbers in [0, 1]
        let u = get_random_f32(rng);
        let v = get_random_f32(rng);

        let r = sqrt(u);
        let theta = 2.0 * 3.1415926 * v;

        let x = (r - 0.5) * 2.0 * camera.lens_radius * cos(theta);
        let y = (r - 0.5) * 2.0 * camera.lens_radius * sin(theta);
        let offset = compute.right * x + camera.up.xyz * y;

        return Ray(
            direction - offset,
            offset + camera.origin.xyz,
        );
    }

    return Ray(direction, camera.origin.xyz);
}

fn get_ray_intersect_test(ray: Ray) -> Intersection {
    if (arrayLength(&spheres) == 8u) {
        return Intersection(vec4<f32>(1.0, 0.0, 0.0, 1.0), SPHERE, 0u, false, 1.0);
    }
    return Intersection(vec4<f32>(0.5f, 0.5f, 0.5f, 1f), CUBEMAP, 0u, false, MAXF);
}

fn get_ray_intersect(ray: Ray) -> Intersection {
    let ray_dir = ray.direction;
    let ray_orig = ray.origin;
    // Initialize intersect struct
    var intersect = Intersection(vec4<f32>(0f, 0f, 0f, 0f), NONE, 0u, false, 0f);
    var closest_intersect = MAXF;
    

    // Iterate through every sphere 
    if (contains_valid_spheres()) {
        for (var i = 0u; i < arrayLength(&spheres); i++) { 
            let got_dist = get_sphere_intersect(ray_dir, ray_orig, i);
            if got_dist != -1f {
                if got_dist < closest_intersect {
                    closest_intersect = got_dist;
                    
                    intersect = Intersection(spheres[i].coloring, SPHERE, i, false, got_dist);
                }
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

// 
// Utility functions
//
fn contains_valid_spheres() -> bool {
    if arrayLength(&spheres) == 1u && spheres[0].is_valid == 0u {
        return false;
    }
    return true;
}

fn contains_valid_free_triangles() -> bool {
    if arrayLength(&free_triangles) == 1u && free_triangles[0].is_valid == 0u {
        return false;
    }
    return true;
}

fn contains_valid_mesh_triangles() -> bool {
    if arrayLength(&mesh_triangles) == 1u && mesh_triangles[0].is_valid == 0u {
        return false;
    }
    return true;
}

fn num_cube_maps() -> u32 {
    return u32(cube_maps[0]);
}

fn num_meshes_in_chunk(chunk: u32) -> i32 {
    if chunk == 0 {
        return i32(mesh_chunk_0[0]);
    } else if chunk == 1 {
        return i32(mesh_chunk_1[0]);
    } else if chunk == 2 {
        return i32(mesh_chunk_2[0]);
    } else if chunk == 3 {
        return i32(mesh_chunk_3[0]);
    }
    return -1;
}

// Generate random float between 0 and 1
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

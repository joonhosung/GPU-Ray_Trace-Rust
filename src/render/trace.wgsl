const MESH = 0u;
const SPHERE = 1u;
const CUBEMAP = 2u;
const NONE = 3u;
const MAXF = 0x1.fffffep+127f;
const MIN_INTERSECT = 0.0001f;
const PI   = 3.1415926f;

// For UniformDiffuseSpec
const SPEC = 0u;
const DIFF = 1u;
const DIFFSPEC = 2u;
const DIELECTRIC = 3u;

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

struct HitInfo {
    emissive: vec3<f32>,
    pos: vec3<f32>,
    norm: vec3<f32>,
    refl_ray: RayRefl,
    has_emissive: bool,
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

struct RayRefl {
    ray: Ray,
    intensity: f32,
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
var<storage, read> cube_map_headers: array<CubeMapFaceHeader>;

@group(3) @binding(2)
var<storage, read> cube_map_faces: array<f32>;

@group(3) @binding(2)
var<storage, read> free_triangles: array<FreeTriangle>;


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_index = get_pixel_index(global_id.x, global_id.y, render_info.width);
    var seed = initRng();
    let ray_compute = create_ray_compute(vec2<u32>(render_info.width, render_info.height), camera);
    var sample_count = 0.0;
    var colour = vec3<f32>(0f);

    // FIXME: Sanity for Jackson in the morning to try with camera ray generation sanity (rasterization)
    // PUT PIXEL GENRATION HERE
    for (var i = 0u; i < render_info.samps_per_pix; i += 1u) {
        var intensity = 1f;
        var ray = pix_cam_to_rand_ray(ray_compute, vec2<u32>(global_id.x, global_id.y), camera, &seed);
        var hit_info: HitInfo;
        var ray_intersect: Intersection;
        var colour_inner = vec3<f32>(0f);
        var colour_intensity = vec3(1f);
        for (var bounce = 0f; bounce < 5f; bounce += 1f) {
            ray_intersect = get_ray_intersect(ray, &hit_info, &seed);
            
            // generate another ray to bounce off of 
            ray = hit_info.refl_ray.ray;
            
            if !ray_intersect.has_bounce || hit_info.has_emissive {
                colour_inner += (hit_info.emissive * colour_intensity ); //* intensity;
                break;
            };

            // if hit_info.has_emissive {
            //     colour_inner += hit_info.emissive;
            // }
            // colour_inner *= ray_intersect.colour.xyz;
            colour_intensity *= ray_intersect.colour.xyz;
            // colour_inner *= intensity; //* ray_intersect.colour.xyz * intensity;
            intensity *= hit_info.refl_ray.intensity;

        }
        
        colour = (colour_inner + (colour * sample_count) ) / (sample_count + 1.0);
        sample_count += 1.0;
    }
    // colour = normalize(colour);
    render_target[pixel_index]     = colour.x;//(colour.x + (render_target[pixel_index] * sample_count)) / (sample_count + 1.0);
    render_target[pixel_index + 1] = colour.y;//(colour.y + (render_target[pixel_index + 1] * sample_count)) / (sample_count + 1.0);
    render_target[pixel_index + 2] = colour.z;//(colour.z + (render_target[pixel_index + 2] * sample_count)) / (sample_count + 1.0);
    render_target[pixel_index + 3] = 1f; //colour.w;//(colour.w + (render_target[pixel_index + 3] * sample_count)) / (sample_count + 1.0);
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
        f32(canvas_dims.x) / 2.0,
        f32(canvas_dims.y) / 2.0,
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
        let theta = 2.0 * PI * v;

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

fn get_ray_intersect(ray: Ray, hit_info: ptr<function, HitInfo>, rng: ptr<function, u32>) -> Intersection {
    // Initialize intersect struct
    var intersect = Intersection(vec4<f32>(1f, 0f, 0f, 1f), NONE, 0u, false, 0f);
    var closest_intersect = MAXF;
    
    // Iterate through every sphere 
    if (contains_valid_spheres()) {
        for (var i = 0u; i < arrayLength(&spheres); i++) { 
            let got_dist = get_sphere_intersect(ray, i);
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
    if intersect.element_type == NONE && num_cube_map_faces() > 0u {
        intersect = Intersection(hit_info_distant_cube_map(ray), CUBEMAP, 0u, false, MAXF);
        *hit_info = HitInfo(intersect.colour.xyz, vec3(0f), vec3(0f), RayRefl(ray, 1), true);
        intersect.has_bounce = false;
    } else {
        *hit_info = get_hit_info(ray, intersect, rng);
        intersect.has_bounce = true;
    }

    return intersect;
}
// Returns hit index and ray length

fn get_sphere_intersect(ray: Ray, i: u32) -> f32 {
    let oc = ray.origin - spheres[i].center.xyz;
    let dir = dot(ray.direction, oc);
    let consts = dot(oc, oc) - (spheres[i].radius * spheres[i].radius);

    let discr = (dir * dir) - consts;

    // If the ray crosses the sphere, return the colour of the closer intersection
    if discr > 0.0 { 
        let offset = -dir;
        let thing = sqrt(discr);
        let intersect_dist_a = offset - thing;
        let intersect_dist_b = offset + thing;

        if (intersect_dist_a > MIN_INTERSECT) && (intersect_dist_a < intersect_dist_b) {
            return intersect_dist_a;
        } else if (intersect_dist_b > MIN_INTERSECT) && (intersect_dist_a > intersect_dist_b) {
            return intersect_dist_b;
        }
        
        // distance can't be negative
        return f32(-1.0); 
        // TODO: Should calculate how the ray is diverted
    }

    return f32(-1.0); 
}


fn get_hit_info(ray: Ray, intersect: Intersection, rng: ptr<function, u32>) -> HitInfo {
    var refl_ray = RayRefl(ray, 1f);
    var has_emissive = false;
    switch intersect.element_type {
        case MESH: {return HitInfo(vec3(0f), vec3(0f), vec3(0f), refl_ray, false);}

        case SPHERE: {
            let perfect_pos = ray.origin + ray.direction * intersect.ray_distance;
            let norm = normalize(perfect_pos - spheres[intersect.element_idx].center.xyz);

            let pos = perfect_pos + norm * MIN_INTERSECT;
            // var refl_ray: RayRefl;
            switch spheres[intersect.element_idx].material.divert_ray_type {
                case SPEC: {refl_ray = get_spec(ray, norm, pos);}
                case DIFF: {refl_ray = get_diff(ray, norm, pos, rng);}
                case DIFFSPEC: {
                    let u: bool = get_random_f32(rng) < spheres[intersect.element_idx].material.diffp * MAXF;
                    if u {refl_ray = get_diff(ray, norm, pos, rng);}
                    else {refl_ray = get_spec(ray, norm, pos);}
                }
                case DIELECTRIC: {refl_ray = get_refract(ray, norm, pos, spheres[intersect.element_idx].material.n_in, spheres[intersect.element_idx].material.n_out, rng);}
                default: {}
            }

            if spheres[intersect.element_idx].material.has_emissive > 0u {
                has_emissive = true;
            }

            return HitInfo(spheres[intersect.element_idx].material.emissive, pos, norm, refl_ray, has_emissive);
        }

        default: {return HitInfo(vec3(0f), vec3(0f), vec3(0f), refl_ray, false);}
    }
}

// Specular "mirror" reflection
fn get_spec(ray: Ray, norm: vec3<f32>, hit_point: vec3<f32>) -> RayRefl {
    let new_ray = Ray(normalize(ray.direction - norm * 2f * dot(ray.direction, norm)), ray.origin);

    return RayRefl(new_ray, 1f);
}

// Diffraction "rough" reflection
fn get_diff(ray: Ray, norm: vec3<f32>, hit_point: vec3<f32>, rng: ptr<function, u32>) -> RayRefl {
    let xd = normalize(ray.direction - norm * dot(ray.direction, norm));
    let yd = normalize(cross(norm, xd));

    let u = get_random_f32(rng);
    let v = get_random_f32(rng);

    let r = sqrt(u);
    let theta = 2f * PI * v;
    
    let x = r * cos(theta);
    let y = r * sin(theta);

    let d = normalize(xd * x + yd * y + norm * sqrt(max(1f - u, 0f)));

    return RayRefl(Ray(d, ray.origin), 1f);
}

// Refraction "prism effect"
fn get_refract(ray: Ray, norm: vec3<f32>, hit_point: vec3<f32>, n_in: f32, n_out: f32, rng: ptr<function, u32>) -> RayRefl {
    var c = dot(norm, ray.direction);
    var n1: f32;
    var n2: f32;
    var norm_refr: vec3<f32>;

    if c < 0f {
        n1 = n_out;
        n2 = n_in;
        c = -c;
        norm_refr = norm;
    } else {
        n1 = n_in;
        n2 = n_out;
        norm_refr = -norm;
    }

    let n_over = n1 / n2;
    let c22 = 1f - n_over * n_over * (1f - c * c);
    let spec = get_spec(ray, norm_refr, hit_point);

    if c22 < 0f {
        return spec;
    } else {
        let trns = n_over * ray.direction + norm_refr * (n_over * c - sqrt(c22));
        let r0 = pow((n1 - n2)/(n1 + n2), 2f);

        let re = r0 + (1f + r0) * pow(1 - dot(trns, norm), 5f);

        let u = get_random_f32(rng);

        if u < re {
            return spec;
        } else {
            return RayRefl(Ray(normalize(trns), hit_point), 1f - re);
        }
    }
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

///////////////////////////////
// Cube map functions
///////////////////////////////
fn num_cube_map_faces() -> u32 {
    if arrayLength(&cube_map_headers) == 1u && cube_map_headers[0].width == 0u {
        return 0u;
    }
    return arrayLength(&cube_map_headers);
}

fn get_cube_map_face_offset(face_index: u32) -> u32 {
    // need to skip the first element which is the length
    var offset = 0u;
    var curr_face_index = 0u;
    while curr_face_index < face_index {
        let header = cube_map_headers[curr_face_index];
        offset += (3u * header.width * header.height);
        curr_face_index += 1u;
    }
    return offset;
}

fn get_cube_map_face_pixel(face_id: u32, u: f32, v: f32) -> vec3<f32> {
    let header = cube_map_headers[face_id];
    let offset = get_cube_map_face_offset(face_id);
    let px = u32(trunc(clamp(u * f32(header.width), 0.0, f32(header.width - 1u))));
    let py = u32(trunc(clamp(v * f32(header.height), 0.0, f32(header.height - 1u))));

    let pixel_offset = offset + 3u * (px + py * header.width);
    return vec3<f32>(
        cube_map_faces[pixel_offset],
        cube_map_faces[pixel_offset + 1u],
        cube_map_faces[pixel_offset + 2u],
    );

}

fn sample_face(face_index: u32, u: f32, v: f32, fact: f32) -> vec3<f32> {
    let header = cube_map_headers[face_index];

    var scaled_u = u * header.uv_scale_x / fact;
    var scaled_v = v * header.uv_scale_y / fact;

    scaled_u = 0.5 * scaled_u + 0.5;
    scaled_v = 0.5 * scaled_v + 0.5;

    return get_cube_map_face_pixel(face_index, scaled_u, scaled_v);
}

fn hit_info_distant_cube_map(ray: Ray) -> vec4<f32> {
    let comps = abs(ray.direction.xyz);
    var face_index = 0u;
    var u: f32 = -1f;
    var v: f32 = -1f;
    var fact: f32 = -1f;
    let d = normalize(ray.direction.xyz);
    // Find the largest component and determine which face to sample
    // 0 -> neg_z, 1 -> pos_z, 2 -> neg_x, 3 -> pos_x, 4 -> neg_y, 5 -> pos_y
    if comps.x >= comps.y && comps.x >= comps.z {
        if d.x < 0.0 {
            u = d.z;
            v = d.y;
            fact = d.x;
            face_index = 2u; // neg_x
        } else {
            u = d.z;
            v = d.y;
            fact = d.x;
            face_index = 3u; // pos_x
        }
    } else if comps.y >= comps.x && comps.y >= comps.z {
        if d.y < 0.0 {
            u = d.x;
            v = d.z;
            fact = d.y;
            face_index = 4u; // neg_y
        } else {
            u = d.x;
            v = d.z;
            fact = d.y;
            face_index = 5u; // pos_y
        }
    } else if comps.z >= comps.x && comps.z >= comps.y {
        if d.z < 0.0 {
            u = d.x;
            v = d.y;
            fact = d.z;
            face_index = 0u; // neg_z
        } else {
            u = d.x;
            v = d.y;
            fact = d.z;
            face_index = 1u; // pos_z
        }
    }

    let rgb = sample_face(face_index, u, v, fact);
    let rgba = vec4<f32>(rgb, 0.0);
    return rgba;
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

const NONE = 0u;
const MESH = 1u;
const SPHERE = 2u;
const CUBEMAP = 3u;
const FREETRIANGLE = 4u;
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
    padding: vec3<f32>,
    is_valid: u32,
    material: UniformDiffuseSpec,
}

struct CubeMapFaceHeader {
    width: u32,
    height: u32,
    uv_scale_x: f32,
    uv_scale_y: f32,
}

struct MeshChunkHeader {
    num_meshes: u32,
    padding: vec3<u32>,
}

// 1 per mesh, a mesh can have multiple primitives
struct MeshHeader {
    length: u32,
    num_primitives: u32,
    padding: vec2<u32>,
    trans_mat: mat4x4<f32>,
}

// 1 per primitive
struct PrimitiveHeader {
    length: u32,
    position_count: u32,
    normal_count: u32,
    triangle_count: u32,
    rgb_info_coords_count: u32,
    has_norm_info: u32,
    norm_info_coords_count: u32,
    tangents_count: u32,
    has_texture: u32,
    has_normal_map: u32,
    has_metal_rough_map: u32,
    padding: u32,
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

struct TriangleHitResult {
    l: f32,
    barycentric: vec2<f32>,
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
var<storage, read> mesh_chunk_headers: array<MeshChunkHeader>;

@group(2) @binding(1)
var<storage, read> mesh_headers: array<MeshHeader>;

@group(2) @binding(2)
var<storage, read> primitive_headers: array<PrimitiveHeader>;

@group(2) @binding(3)
var<storage, read> mesh_data_chunk_0: array<f32>;

@group(2) @binding(4)
var<storage, read> mesh_data_chunk_1: array<f32>;

@group(2) @binding(5)
var<storage, read> mesh_data_chunk_2: array<f32>;

@group(2) @binding(6)
var<storage, read> mesh_data_chunk_3: array<f32>;

@group(2) @binding(7)
var<storage, read> mesh_triangles: array<MeshTriangle>;

@group(3) @binding(0)
var<storage, read> spheres: array<Sphere>;

@group(3) @binding(1)
var<storage, read> cube_map_headers: array<CubeMapFaceHeader>;

@group(3) @binding(2)
var<storage, read> cube_map_faces: array<f32>;

@group(3) @binding(3)
var<storage, read> free_triangles: array<FreeTriangle>;


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_index = get_pixel_index(global_id.x, global_id.y, render_info.width);
    
    let ray_compute = create_ray_compute(vec2<u32>(render_info.width, render_info.height), camera);
    var sample_count = 0.0;
    var colour = vec3<f32>(0f);

    var seed = initRng(global_id, render_info.samps_per_pix);

    for (var i = 0u; i < render_info.samps_per_pix; i += 1u) {
        seed = initRng(global_id, seed);
        
        var intensity = 1f;
        var ray = pix_cam_to_rand_ray(ray_compute, vec2<u32>(global_id.x, global_id.y), camera, &seed);
        var hit_info: HitInfo;
        var ray_intersect: Intersection;
        var colour_inner = vec3<f32>(0f);
        var colour_intensity = vec3(1f);
        var bounce = 0u; // bounce < render_info.assured_depth; bounce ++) {
        loop {
            ray_intersect = get_ray_intersect(ray, &hit_info, &seed);
            
            // generate another ray to bounce off of 
            ray = hit_info.refl_ray.ray;
            
            if !ray_intersect.has_bounce || hit_info.has_emissive {
                // colour_intensity = normalize(colour_intensity);
                colour_inner += hit_info.emissive * colour_intensity * intensity;
                colour_intensity *= ray_intersect.colour.xyz;
                if !ray_intersect.has_bounce {
                    break;
                }
            };

            // if hit_info.has_emissive {
            //     colour_inner += hit_info.emissive;
            // }
            // colour_inner *= ray_intersect.colour.xyz;
            
            colour_intensity *= ray_intersect.colour.xyz;
            
            if (bounce >= render_info.assured_depth) && (get_random_f32(&seed) > render_info.max_threshold){
                colour_intensity /= vec3(render_info.max_threshold);
                colour_inner +=  colour_intensity * intensity;//vec3(intensity) * ray_intersect.colour.xyz;//intensity * render_info.max_threshold;
                break;
            }
            
            intensity *= hit_info.refl_ray.intensity;

            bounce++;
        }
        
        colour = (colour_inner + (colour * sample_count)) / (sample_count + 1.0);
        sample_count += 1.0;
    }
    render_target[pixel_index]     = colour.x;
    render_target[pixel_index + 1] = colour.y;
    render_target[pixel_index + 2] = colour.z;
    render_target[pixel_index + 3] = 1f;
}

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
    var intersect = Intersection(vec4<f32>(0f, 0f, 0f, 1f), NONE, 0u, false, 0f);
    var closest_intersect = MAXF;
    
    // Iterate through every sphere 
    if (contains_valid_spheres()) {
        for (var i = 0u; i < arrayLength(&spheres); i++) { 
            let got_dist = get_sphere_intersect(ray, i);
            if got_dist != -1f && got_dist < closest_intersect {
                closest_intersect = got_dist;
                intersect = Intersection(spheres[i].coloring, SPHERE, i, false, got_dist);
            }
        }
    }

    // Iterate through every free triangle
    if (contains_valid_free_triangles()) {
        for (var i = 0u; i < arrayLength(&free_triangles); i++) {
            let hit_result = get_free_triangle_intersect(ray, i);
            if hit_result.l != -1f && hit_result.l < closest_intersect {
                closest_intersect = hit_result.l;
                intersect = Intersection(free_triangles[i].rgb, FREETRIANGLE, i, false, hit_result.l);
            }
        }
    }
    
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
                    let u: bool = get_random_f32(rng) < spheres[intersect.element_idx].material.diffp;
                    if u {refl_ray = get_diff(ray, norm, pos, rng);}
                    else {refl_ray = get_spec(ray, norm, pos);}
                }
                case DIELECTRIC: {refl_ray = get_refract(ray, norm, pos, spheres[intersect.element_idx].material.n_in, spheres[intersect.element_idx].material.n_out, rng);}
                default: {}
            }

            has_emissive = spheres[intersect.element_idx].material.has_emissive > 0u;
            return HitInfo(spheres[intersect.element_idx].material.emissive, pos, norm, refl_ray, has_emissive);
        }
        case FREETRIANGLE: {
            let triangle = free_triangles[intersect.element_idx];
            let norm = triangle.norm.xyz;
            let pos = ray.direction * intersect.ray_distance + ray.origin + norm * MIN_INTERSECT;

            switch triangle.material.divert_ray_type {
                case SPEC: {refl_ray = get_spec(ray, norm, pos);}
                case DIFF: {refl_ray = get_diff(ray, norm, pos, rng);}
                case DIFFSPEC: {
                    let u: bool = get_random_f32(rng) < triangle.material.diffp;
                    if u {refl_ray = get_diff(ray, norm, pos, rng);}
                    else {refl_ray = get_spec(ray, norm, pos);}
                }
                case DIELECTRIC: {refl_ray = get_refract(ray, norm, pos, triangle.material.n_in, triangle.material.n_out, rng);}
                default: {}
            }

            has_emissive = triangle.material.has_emissive > 0u;
            return HitInfo(triangle.material.emissive, pos, norm, refl_ray, has_emissive);
        }

        default: {return HitInfo(vec3(0f), vec3(0f), vec3(0f), refl_ray, false);}
    }
}

// Specular "mirror" reflection
fn get_spec(ray: Ray, norm: vec3<f32>, hit_point: vec3<f32>) -> RayRefl {
    // let new_ray = Ray(normalize(ray.direction - norm * 2f * dot(ray.direction, norm)), ray.origin);
    let new_ray = Ray(reflect(ray.direction, norm), hit_point);
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

    return RayRefl(Ray(d, hit_point), 1f);
}

// Refraction "prism effect"
fn get_refract(ray: Ray, norm: vec3<f32>, hit_point: vec3<f32>, n_in: f32, n_out: f32, rng: ptr<function, u32>) -> RayRefl {
    var c = dot(norm, ray.direction);
    var n1: f32;
    var n2: f32;
    var norm_refr: vec3<f32>;

    // refract(ray.direction)
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

///////////////////////////////
// Sphere functions
///////////////////////////////

fn contains_valid_spheres() -> bool {
    if arrayLength(&spheres) == 1u && spheres[0].is_valid == 0u {
        return false;
    }
    return true;
}

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


///////////////////////////////
// Free triangle functions
///////////////////////////////

fn contains_valid_free_triangles() -> bool {
    if arrayLength(&free_triangles) == 1u && free_triangles[0].is_valid == 0u {
        return false;
    }
    return true;
}

fn get_free_triangle_intersect(ray: Ray, i: u32) -> TriangleHitResult {
    let triangle = free_triangles[i];
    let e1 = triangle.vert2.xyz - triangle.vert1.xyz;
    let e2 = triangle.vert3.xyz - triangle.vert1.xyz;
    let ray_x_e2 = cross(ray.direction, e2);
    let det = dot(e1, ray_x_e2);
    let no_hit = TriangleHitResult(-1.0, vec2<f32>(-1.0, -1.0));
    if abs(det) < MIN_INTERSECT {
        return no_hit;
    }
    let inv_det = 1.0 / det;
    let rhs = ray.origin - triangle.vert1.xyz;
    let u = inv_det * dot(rhs, ray_x_e2);
    if u < 0.0 || u > 1.0 {
        return no_hit;
    }
    let rhs_x_e1 = cross(rhs, e1);
    let v = inv_det * dot(ray.direction, rhs_x_e1);
    if v < 0.0 || u + v > 1.0 {
        return no_hit;
    }
    let l = inv_det * dot(e2, rhs_x_e1);
    if l < MIN_INTERSECT {
        return no_hit;
    }
    let hit_result = TriangleHitResult(l, vec2<f32>(u, v));
    return hit_result;
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

///////////////////////////////
// Mesh functions
///////////////////////////////
fn num_meshes_in_chunk(chunk: u32) -> u32 {
    return mesh_chunk_headers[chunk].num_meshes;
}

fn contains_valid_mesh_triangles() -> bool {
    if arrayLength(&mesh_triangles) == 1u && mesh_triangles[0].is_valid == 0u {
        return false;
    }
    return true;
}


//////////////////////////////////
// Random Number Generation
/////////////////////////////////

// Shoutout to: https://github.com/boksajak/referencePT
fn initRng(global_id: vec3<u32>, in_seed: u32) -> u32 {
    let seed: u32 = dot(global_id.xy, vec2(render_info.width, render_info.height)) ^ jenkinsHash(in_seed);
    return jenkinsHash(seed);
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

fn jenkinsHash(input: u32) -> u32 {
    var x = input;
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}

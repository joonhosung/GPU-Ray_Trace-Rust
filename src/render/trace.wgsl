const NONE = 0u;
const SPHERE = 1u;
const CUBEMAP = 2u;
const FREETRIANGLE = 3u;
const MESHTRIANGLE = 4u;
const MAXF = 0x1.fffffep+127f;
const MIN_INTERSECT = 0.0001f;
const PI   = 3.1415926f;
const NUM_MESH_CHUNKS = 4u;
const CUSTOM_ATTEN = 1f;
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
    chunk_id: u32,
    data_offset: u32,
    padding: vec3<u32>,
    primitive_header_offset: u32,
    trans_mat: mat4x4<f32>,
}

// 1 per primitive
struct PrimitiveHeader {
    length: u32,
    // offset to the start of the primitive data in the mesh data sequence.
    // to access position, do chunk[mesh_header.data_offset + prim_header.mesh_data_offset + prim_header.position_offset]
    // to access normal, do chunk[mesh_header.data_offset + prim_header.mesh_data_offset + prim_header.normal_offset]
    // etc.
    // where mesh_header is the header of the mesh containing this primitive
    mesh_data_offset: u32,
    position_offset: u32,
    position_count: u32,

    normal_offset: u32,
    normal_count: u32,

    triangle_offset: u32,
    triangle_count: u32,

    rgb_info_factor_offset: u32,
    rgb_info_coords_offset: u32,
    rgb_info_coords_count: u32,

    norm_info_scale_offset: u32,
    norm_info_coords_offset: u32,
    has_norm_info: u32,
    norm_info_coords_count: u32,

    metal_rough_metal_offset: u32,
    metal_rough_rough_offset: u32,
    metal_rough_coords_offset: u32,
    metal_rough_coords_count: u32,

    texture_data_offset: u32,
    texture_data_width: u32,
    texture_data_height: u32,

    normal_map_data_offset: u32,
    normal_map_data_width: u32,
    normal_map_data_height: u32,

    metal_rough_map_data_offset: u32,
    metal_rough_map_data_width: u32,
    metal_rough_map_data_height: u32,
}

struct MeshTriangle {
    mesh_index: u32,
    prim_index: u32,
    inner_index: u32,
    is_valid: u32,
    normal_transform_c1: vec3<f32>,
    padding: f32,
    normal_transform_c2: vec3<f32>,
    padding2: f32,
    normal_transform_c3: vec3<f32>,
    padding3: f32,
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

struct Iter {
    padding: vec3<u32>,
    ation: u32,
}

struct DynDiffSpec {
    should_diff: bool,
    roughness: f32
}

struct Aabb {
    bounds: array<vec4<f32>, 3>,
    // paddding: vec2<u32>,
}

struct TreeNode {
    axis: u32, 
    split: f32, 
    low: u32, 
    high: u32, 
    is_leaf: u32, 
    leaf_mesh_index: u32, 
    leaf_mesh_size: u32, 
    padding: f32,
}

struct AabbEntryExit {
    entry_t: f32,
    exit_t: f32,
    intersects: bool,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<uniform> render_info: RenderInfo;

@group(1) @binding(0)
var<storage, read> mesh_chunk_headers: array<MeshChunkHeader>;

@group(1) @binding(1)
var<storage, read> mesh_headers: array<MeshHeader>;

@group(1) @binding(2)
var<storage, read> primitive_headers: array<PrimitiveHeader>;

@group(1) @binding(3)
var<storage, read> mesh_data_chunk_0: array<f32>;

@group(1) @binding(4)
var<storage, read> mesh_data_chunk_1: array<f32>;

@group(1) @binding(5)
var<storage, read> mesh_data_chunk_2: array<f32>;

@group(1) @binding(6)
var<storage, read> mesh_data_chunk_3: array<f32>;

@group(1) @binding(7)
var<storage, read> mesh_triangles: array<MeshTriangle>;

@group(2) @binding(0)
var<storage, read> spheres: array<Sphere>;

@group(2) @binding(1)
var<storage, read> cube_map_headers: array<CubeMapFaceHeader>;

@group(2) @binding(2)
var<storage, read> cube_map_faces: array<f32>;

@group(2) @binding(3)
var<storage, read> free_triangles: array<FreeTriangle>;

@group(2) @binding(4)
var<storage, read> leaf_node_triangles: array<u32>;

@group(2) @binding(5)
var<storage, read> tree_nodes: array<TreeNode>;

@group(2) @binding(6)
var<uniform> kd_tree: Aabb;

// For better precision, each pixel is represented by 4 floats (RGBA)
@group(3) @binding(0)
var<storage, read_write> render_target: array<f32>;

@group(3) @binding(1)
var<storage, read_write> iter: Iter;


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_index = get_pixel_index(global_id.x, global_id.y, render_info.width);
    
    let ray_compute = create_ray_compute(vec2<u32>(render_info.width, render_info.height), camera);
    var sample_count = 0.0;
    var colour = vec3<f32>(0f);

    var seed = initRng(global_id, iter.ation);

    for (var i = 0u; i < render_info.samps_per_pix; i += 1u) {
        iter.ation++;
        var intensity = 1f;
        var ray = pix_cam_to_rand_ray(ray_compute, vec2<u32>(global_id.x, global_id.y), camera, &seed);
        var hit_info: HitInfo;
        var ray_intersect: Intersection;
        var colour_inner = vec3<f32>(0f);
        var colour_intensity = vec3(1f);
        var bounce = 0u;
        loop {
            ray_intersect = get_ray_intersect(ray, &hit_info, &seed);
            
            // generate another ray to bounce off of 
            ray = hit_info.refl_ray.ray;
            
            if !ray_intersect.has_bounce || hit_info.has_emissive {
                colour_inner += hit_info.emissive * colour_intensity * intensity;
                colour_intensity *= ray_intersect.colour.xyz;
                if !ray_intersect.has_bounce {
                    break;
                }
            };
            
            colour_intensity *= ray_intersect.colour.xyz;
            
            if (bounce >= render_info.assured_depth) && (get_random_f32(&seed) > render_info.max_threshold){
                colour_intensity /= vec3(render_info.max_threshold);
                colour_inner +=  colour_intensity * intensity;
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
    var barycentric = vec2<f32>(0f, 0f);
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

    if (contains_valid_mesh_triangles()) {
        let entry_exit: AabbEntryExit = get_entry_exit(ray, kd_tree);

        // If we know the ray will go into the tree, search it
        // Still bottlenecked by the slowest ray, but the slowest ray is WAY faster
        if entry_exit.intersects {
            // let leaf_node: TreeNode = search_kd_tree(ray, entry_exit.x, entry_exit.y);                
            
            var closest_mesh_triangle = -1;
            var hit_result = TriangleHitResult(-1f, vec2<f32>(-1f, -1f));
            for (var i = 0u; i < leaf_node.leaf_mesh_size; i++) {
            // for (var i = 0u; i < arrayLength(&mesh_triangles); i++) {
            
                // Retrieve the MeshTriangle array indices using the leaf_node_triangles so we only iterate based on a limited number of triangles
                // hit_result = get_mesh_triangle_intersect(ray, i); 
                hit_result = get_mesh_triangle_intersect(ray, leaf_node_triangles[leaf_node.leaf_mesh_index+1]); 
                if hit_result.l != -1f && hit_result.l < closest_intersect {
                    closest_intersect = hit_result.l;
                    closest_mesh_triangle = i32(i);  
                }
            }
            if closest_mesh_triangle != -1 {
                let mesh_triangle = mesh_triangles[u32(closest_mesh_triangle)];
                let rgba = get_rgba_for_mesh_triangle(mesh_triangle, hit_result.barycentric);
                barycentric = hit_result.barycentric;
                intersect = Intersection(rgba, MESHTRIANGLE, u32(closest_mesh_triangle), false, hit_result.l);
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
        *hit_info = get_hit_info(ray, intersect, barycentric, rng);
        intersect.has_bounce = true;
    }

    return intersect;
}
// Returns hit index and ray length

fn get_hit_info(ray: Ray, intersect: Intersection, barycentric: vec2<f32>, rng: ptr<function, u32>) -> HitInfo {
    var refl_ray = RayRefl(ray, 1f);
    var has_emissive = false;
    switch intersect.element_type {
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
        case MESHTRIANGLE: {
            let mesh_triangle = mesh_triangles[intersect.element_idx];
            let norm = get_norm_for_mesh_triangle(mesh_triangle, barycentric);
            let pos = ray.direction * intersect.ray_distance + ray.origin + norm * MIN_INTERSECT;
            let dyn_diff_spec = get_diff_spec_and_roughness(mesh_triangle, ray, norm, barycentric, rng);
            if dyn_diff_spec.should_diff {
                refl_ray = get_diff(ray, norm, pos, rng);
            } else {
                refl_ray = get_spec(ray, norm, pos);
            }
            let uvw = vec3<f32>(
                get_random_f32(rng),
                get_random_f32(rng),
                get_random_f32(rng),
            );
            let scatter = dyn_diff_spec.roughness * normalize(uvw);
            refl_ray.ray.direction = normalize(refl_ray.ray.direction + scatter);
            return HitInfo(vec3(0f), pos, norm, refl_ray, false);
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
    }
    let trns = n_over * ray.direction + norm_refr * (n_over * c - sqrt(c22));
    let r0 = pow((n1 - n2)/(n1 + n2), 2f);

    let re = r0 + (1f + r0) * pow(1 - dot(trns, norm), 5f);

    let u = get_random_f32(rng);

    if u < re {
        return spec;
    } 
    return RayRefl(Ray(normalize(trns), hit_point), 1f - re);
}

///////////////////////////////
// Sphere functions
///////////////////////////////

fn contains_valid_spheres() -> bool {
    return spheres[0].is_valid == 1u;
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
// Triangle functions
///////////////////////////////

fn get_triangle_intersect(ray: Ray, vert1: vec3<f32>, vert2: vec3<f32>, vert3: vec3<f32>) -> TriangleHitResult {
    let e1 = vert2 - vert1;
    let e2 = vert3 - vert1;
    let ray_x_e2 = cross(ray.direction, e2);
    let det = dot(e1, ray_x_e2);
    let no_hit = TriangleHitResult(-1.0, vec2<f32>(-1.0, -1.0));
    if abs(det) < MIN_INTERSECT {
        return no_hit;
    }
    let inv_det = 1.0 / det;
    let rhs = ray.origin - vert1;
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
// Free triangle functions
///////////////////////////////

fn contains_valid_free_triangles() -> bool {
    return free_triangles[0].is_valid == 1u;
}

fn get_free_triangle_intersect(ray: Ray, i: u32) -> TriangleHitResult {
    let triangle = free_triangles[i];
    return get_triangle_intersect(ray, triangle.vert1.xyz, triangle.vert2.xyz, triangle.vert3.xyz);
}

///////////////////////////////
// KD Tree functions
///////////////////////////////
fn get_entry_exit(ray: Ray, aabb: Aabb) -> AabbEntryExit {
    var entry_exit_points: AabbEntryExit = AabbEntryExit( MAXF, -MAXF, false);
    for (var i=0; i<3; i++) {
        var ray_dir: f32;
        if abs(ray.direction[i]) < MIN_INTERSECT {
            if ray.direction[i] < 0 {ray_dir = -MIN_INTERSECT;}
            else {ray_dir = MIN_INTERSECT;}
        } else {ray_dir = ray.direction[i];}

        let min_t = min((aabb.bounds[i].x - ray.origin[i]) / ray.direction[i], (aabb.bounds[i].y - ray.origin[i]) / ray.direction[i]);
        let max_t = max((aabb.bounds[i].x - ray.origin[i]) / ray.direction[i], (aabb.bounds[i].y - ray.origin[i]) / ray.direction[i]);

        // MAXF, MAXF should be impossible. Use to denote no intersection found
        if (min_t >= max_t) || (max_t < 0.0) {
            entry_exit_points.intersects = false;
            return entry_exit_points;
        }
        else {
            entry_exit_points.entry_t = min(min_t, entry_exit_points.entry_t);
            entry_exit_points.exit_t = max(max_t, entry_exit_points.exit_t);
        }
    }
    entry_exit_points.intersects = true;
    return entry_exit_points; // return earliest entry point and latest exit point
}

fn search_kd_tree(ray: Ray, entry_t: f32, exit_t: f32) -> TreeNode {
    var cur_node = tree_nodes[0]; // Index 0 is always the head node
    var cur_entry = entry_t;
    var cur_exit = exit_t;
    var ray_dir = ray.direction;

    // Fix ray direction edge cases for each axis
    for (var i=0; i<3; i++) {
        if abs(ray.direction[i]) < MIN_INTERSECT {
            if ray.direction[i] < 0 {ray_dir[i] = -MIN_INTERSECT;}
            else {ray_dir[i] = MIN_INTERSECT;}
        } else {ray_dir[i] = ray.direction[i];}
    }

    // Search through the tree until we find a leaf node containing all the indices
    // while cur_node.is_leaf == 0u {
    //     let a = cur_node.axis;
    //     let t = (cur_node.split - ray.origin[a]) / ray_dir[a];

    // }

    return cur_node;
}

///////////////////////////////
// Mesh Triangle functions
///////////////////////////////
fn num_meshes_in_chunk(chunk: u32) -> u32 {
    return mesh_chunk_headers[chunk].num_meshes;
}

fn contains_valid_mesh_triangles() -> bool {
    return mesh_triangles[0].is_valid == 1u;
}

fn get_vertex_for_mesh_triangle(mesh_triangle: MeshTriangle, vert_id: u32) -> vec3<f32> {
    let mesh_id = mesh_triangle.mesh_index;
    let prim_id = mesh_triangle.prim_index;
    let inner_id = mesh_triangle.inner_index;
    let mesh_header = mesh_headers[mesh_id];
    let prim_header = primitive_headers[mesh_header.primitive_header_offset + prim_id];
    let data_offset = mesh_header.data_offset + prim_header.mesh_data_offset;
    let position_offset = data_offset + prim_header.position_offset;
    let triangle_offset = data_offset + prim_header.triangle_offset;
    if mesh_header.chunk_id == 0 {
        let position_index = mesh_data_chunk_0[triangle_offset + inner_id * 3 + vert_id];
        let base_index = position_offset + u32(position_index) * 3;
        return vec3<f32>(
            mesh_data_chunk_0[base_index],
            mesh_data_chunk_0[base_index + 1u],
            mesh_data_chunk_0[base_index + 2u],
        );
    }
    else if mesh_header.chunk_id == 1 {
        let position_index = mesh_data_chunk_1[triangle_offset + inner_id * 3 + vert_id];
        let base_index = position_offset + u32(position_index) * 3;
        return vec3<f32>(
            mesh_data_chunk_1[base_index],
            mesh_data_chunk_1[base_index + 1u],
            mesh_data_chunk_1[base_index + 2u],
        );
    }
    else if mesh_header.chunk_id == 2 {
        let position_index = mesh_data_chunk_2[triangle_offset + inner_id * 3 + vert_id];
        let base_index = position_offset + u32(position_index) * 3;
        return vec3<f32>(
            mesh_data_chunk_2[base_index],
            mesh_data_chunk_2[base_index + 1u],
            mesh_data_chunk_2[base_index + 2u],
        );
    }
    else if mesh_header.chunk_id == 3 {
        let position_index = mesh_data_chunk_3[triangle_offset + inner_id * 3 + vert_id];
        let base_index = position_offset + u32(position_index) * 3;
        return vec3<f32>(
            mesh_data_chunk_3[base_index],
            mesh_data_chunk_3[base_index + 1u],
            mesh_data_chunk_3[base_index + 2u],
        );
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn get_mesh_triangle_intersect(ray: Ray, i: u32) -> TriangleHitResult {
    let mesh_triangle = mesh_triangles[i];
    let vert1 = get_vertex_for_mesh_triangle(mesh_triangle, 0u);
    let vert2 = get_vertex_for_mesh_triangle(mesh_triangle, 1u);
    let vert3 = get_vertex_for_mesh_triangle(mesh_triangle, 2u);
    return get_triangle_intersect(ray, vert1, vert2, vert3);
}

fn tex_coord_from_bary(mesh_triangle: MeshTriangle, coords_offset: u32, barycentric: vec2<f32>) -> vec2<f32> {
    let mesh_id = mesh_triangle.mesh_index;
    let prim_id = mesh_triangle.prim_index;
    let inner_id = mesh_triangle.inner_index;
    let mesh_header = mesh_headers[mesh_id];
    let prim_header = primitive_headers[mesh_header.primitive_header_offset + prim_id];
    let data_offset = mesh_header.data_offset + prim_header.mesh_data_offset;
    let triangle_offset = data_offset + prim_header.triangle_offset;

    let b1 = barycentric.x;
    let b2 = barycentric.y;
    let b0 = 1.0 - b2 - b1;

    if mesh_header.chunk_id == 0 {
        let idx0 = mesh_data_chunk_0[triangle_offset + inner_id * 3];
        let idx1 = mesh_data_chunk_0[triangle_offset + inner_id * 3 + 1u];
        let idx2 = mesh_data_chunk_0[triangle_offset + inner_id * 3 + 2u];

        let uv0 = vec2<f32>(
            mesh_data_chunk_0[coords_offset + u32(idx0) * 2],
            mesh_data_chunk_0[coords_offset + u32(idx0) * 2 + 1u],
        );
        let uv1 = vec2<f32>(
            mesh_data_chunk_0[coords_offset + u32(idx1) * 2],
            mesh_data_chunk_0[coords_offset + u32(idx1) * 2 + 1u],
        );
        let uv2 = vec2<f32>(
            mesh_data_chunk_0[coords_offset + u32(idx2) * 2],
            mesh_data_chunk_0[coords_offset + u32(idx2) * 2 + 1u],
        );

        return uv0 * b0 + uv1 * b1 + uv2 * b2;
    }
    else if mesh_header.chunk_id == 1 {
        let idx0 = mesh_data_chunk_1[triangle_offset + inner_id * 3];
        let idx1 = mesh_data_chunk_1[triangle_offset + inner_id * 3 + 1u];
        let idx2 = mesh_data_chunk_1[triangle_offset + inner_id * 3 + 2u];

        let uv0 = vec2<f32>(
            mesh_data_chunk_1[coords_offset + u32(idx0) * 2],
            mesh_data_chunk_1[coords_offset + u32(idx0) * 2 + 1u],
        );
        let uv1 = vec2<f32>(
            mesh_data_chunk_1[coords_offset + u32(idx1) * 2],
            mesh_data_chunk_1[coords_offset + u32(idx1) * 2 + 1u],
        );
        let uv2 = vec2<f32>(
            mesh_data_chunk_1[coords_offset + u32(idx2) * 2],
            mesh_data_chunk_1[coords_offset + u32(idx2) * 2 + 1u],
        );

        return uv0 * b0 + uv1 * b1 + uv2 * b2;
    }
    else if mesh_header.chunk_id == 2 {
        let idx0 = mesh_data_chunk_2[triangle_offset + inner_id * 3];
        let idx1 = mesh_data_chunk_2[triangle_offset + inner_id * 3 + 1u];
        let idx2 = mesh_data_chunk_2[triangle_offset + inner_id * 3 + 2u];

        let uv0 = vec2<f32>(
            mesh_data_chunk_2[coords_offset + u32(idx0) * 2],
            mesh_data_chunk_2[coords_offset + u32(idx0) * 2 + 1u],
        );
        let uv1 = vec2<f32>(
            mesh_data_chunk_2[coords_offset + u32(idx1) * 2],
            mesh_data_chunk_2[coords_offset + u32(idx1) * 2 + 1u],
        );
        let uv2 = vec2<f32>(
            mesh_data_chunk_2[coords_offset + u32(idx2) * 2],
            mesh_data_chunk_2[coords_offset + u32(idx2) * 2 + 1u],
        );

        return uv0 * b0 + uv1 * b1 + uv2 * b2;
    }
    else if mesh_header.chunk_id == 3 {
        let idx0 = mesh_data_chunk_3[triangle_offset + inner_id * 3];
        let idx1 = mesh_data_chunk_3[triangle_offset + inner_id * 3 + 1u];
        let idx2 = mesh_data_chunk_3[triangle_offset + inner_id * 3 + 2u];

        let uv0 = vec2<f32>(
            mesh_data_chunk_3[coords_offset + u32(idx0) * 2],
            mesh_data_chunk_3[coords_offset + u32(idx0) * 2 + 1u],
        );
        let uv1 = vec2<f32>(
            mesh_data_chunk_3[coords_offset + u32(idx1) * 2],
            mesh_data_chunk_3[coords_offset + u32(idx1) * 2 + 1u],
        );
        let uv2 = vec2<f32>(
            mesh_data_chunk_3[coords_offset + u32(idx2) * 2],
            mesh_data_chunk_3[coords_offset + u32(idx2) * 2 + 1u],
        );

        return uv0 * b0 + uv1 * b1 + uv2 * b2;
    }
    return vec2<f32>(0.0, 0.0);
}

fn get_pixel_from_image(coord_from_bary: vec2<f32>, img_offset: u32, img_width: u32, img_height: u32, chunk_id: u32) -> vec3<f32> {
    let pixel_x = u32(trunc(clamp(coord_from_bary.x * f32(img_width), 0.0, f32(img_width - 1u))));
    let pixel_y = u32(trunc(clamp(coord_from_bary.y * f32(img_height), 0.0, f32(img_height - 1u))));
    let pixel_index = img_offset + 3u * (pixel_x + pixel_y * img_width);
    if chunk_id == 0 {
        return vec3<f32>(
            mesh_data_chunk_0[pixel_index],
            mesh_data_chunk_0[pixel_index + 1u],
            mesh_data_chunk_0[pixel_index + 2u],
        );
    }
    else if chunk_id == 1 {
        return vec3<f32>(
            mesh_data_chunk_1[pixel_index],
            mesh_data_chunk_1[pixel_index + 1u],
            mesh_data_chunk_1[pixel_index + 2u],
        );
    }
    else if chunk_id == 2 {
        return vec3<f32>(
            mesh_data_chunk_2[pixel_index],
            mesh_data_chunk_2[pixel_index + 1u],
            mesh_data_chunk_2[pixel_index + 2u],
        );
    }
    else if chunk_id == 3 {
        return vec3<f32>(
            mesh_data_chunk_3[pixel_index],
            mesh_data_chunk_3[pixel_index + 1u],
            mesh_data_chunk_3[pixel_index + 2u],
        );
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn get_rgb_factor_for_mesh_triangle(mesh_triangle: MeshTriangle) -> vec3<f32> {
    let mesh_id = mesh_triangle.mesh_index;
    let prim_id = mesh_triangle.prim_index;
    let inner_id = mesh_triangle.inner_index;
    let mesh_header = mesh_headers[mesh_id];
    let prim_header = primitive_headers[mesh_header.primitive_header_offset + prim_id];
    let data_offset = mesh_header.data_offset + prim_header.mesh_data_offset;
    let rgb_info_factor_offset = data_offset + prim_header.rgb_info_factor_offset;
    if mesh_header.chunk_id == 0 {
        return vec3<f32>(
            mesh_data_chunk_0[rgb_info_factor_offset],
            mesh_data_chunk_0[rgb_info_factor_offset + 1u],
            mesh_data_chunk_0[rgb_info_factor_offset + 2u],
        );
    }
    else if mesh_header.chunk_id == 1 {
        return vec3<f32>(
            mesh_data_chunk_1[rgb_info_factor_offset],
            mesh_data_chunk_1[rgb_info_factor_offset + 1u],
            mesh_data_chunk_1[rgb_info_factor_offset + 2u],
        );
    }
    else if mesh_header.chunk_id == 2 {
        return vec3<f32>(
            mesh_data_chunk_2[rgb_info_factor_offset],
            mesh_data_chunk_2[rgb_info_factor_offset + 1u],
            mesh_data_chunk_2[rgb_info_factor_offset + 2u],
        );
    }
    else if mesh_header.chunk_id == 3 {
        return vec3<f32>(
            mesh_data_chunk_3[rgb_info_factor_offset],
            mesh_data_chunk_3[rgb_info_factor_offset + 1u],
            mesh_data_chunk_3[rgb_info_factor_offset + 2u],
        );
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn get_rgba_for_mesh_triangle(mesh_triangle: MeshTriangle, barycentric: vec2<f32>) -> vec4<f32> {
    let mesh_id = mesh_triangle.mesh_index;
    let prim_id = mesh_triangle.prim_index;
    let inner_id = mesh_triangle.inner_index;
    let mesh_header = mesh_headers[mesh_id];
    let prim_header = primitive_headers[mesh_header.primitive_header_offset + prim_id];
    let rgb_info_factor = get_rgb_factor_for_mesh_triangle(mesh_triangle);
    if prim_header.rgb_info_coords_count == 0u {
        return vec4<f32>(rgb_info_factor, 0);
    }
    let data_offset = mesh_header.data_offset + prim_header.mesh_data_offset;
    let rgb_info_coords_offset = data_offset + prim_header.rgb_info_coords_offset;
    let tex_coord = tex_coord_from_bary(mesh_triangle, rgb_info_coords_offset, barycentric);
    let texture_offset = data_offset + prim_header.texture_data_offset;
    let texture_width = prim_header.texture_data_width;
    let texture_height = prim_header.texture_data_height;
    let pixel = get_pixel_from_image(tex_coord, texture_offset, texture_width, texture_height, mesh_header.chunk_id);
    let scaled_rgb = rgb_info_factor * pixel;
    return vec4<f32>(scaled_rgb, 0);
}

fn get_norm_from_norms(mesh_triangle: MeshTriangle, normal_transform: mat3x3<f32>) -> vec3<f32> {
    let mesh_id = mesh_triangle.mesh_index;
    let prim_id = mesh_triangle.prim_index;
    let inner_id = mesh_triangle.inner_index;
    let mesh_header = mesh_headers[mesh_id];
    let prim_header = primitive_headers[mesh_header.primitive_header_offset + prim_id];
    let data_offset = mesh_header.data_offset + prim_header.mesh_data_offset;
    let triangle_offset = data_offset + prim_header.triangle_offset;
    let normal_offset = data_offset + prim_header.normal_offset;
    if mesh_header.chunk_id == 0 {
        let norm_index1 = mesh_data_chunk_0[triangle_offset + inner_id * 3];
        let norm_index2 = mesh_data_chunk_0[triangle_offset + inner_id * 3 + 1u];
        let norm_index3 = mesh_data_chunk_0[triangle_offset + inner_id * 3 + 2u];
        let norm1 = vec3<f32>(
            mesh_data_chunk_0[normal_offset + u32(norm_index1) * 3],
            mesh_data_chunk_0[normal_offset + u32(norm_index1) * 3 + 1u],
            mesh_data_chunk_0[normal_offset + u32(norm_index1) * 3 + 2u],
        );
        let norm2 = vec3<f32>(
            mesh_data_chunk_0[normal_offset + u32(norm_index2) * 3],
            mesh_data_chunk_0[normal_offset + u32(norm_index2) * 3 + 1u],
            mesh_data_chunk_0[normal_offset + u32(norm_index2) * 3 + 2u],
        );
        let norm3 = vec3<f32>(
            mesh_data_chunk_0[normal_offset + u32(norm_index3) * 3],
            mesh_data_chunk_0[normal_offset + u32(norm_index3) * 3 + 1u],
            mesh_data_chunk_0[normal_offset + u32(norm_index3) * 3 + 2u],
        );

        let accum = normal_transform * (norm1 + norm2 + norm3);
        return normalize(accum);
    }
    else if mesh_header.chunk_id == 1 {
        let norm_index1 = mesh_data_chunk_1[triangle_offset + inner_id * 3];
        let norm_index2 = mesh_data_chunk_1[triangle_offset + inner_id * 3 + 1u];
        let norm_index3 = mesh_data_chunk_1[triangle_offset + inner_id * 3 + 2u];
        let norm1 = vec3<f32>(
            mesh_data_chunk_1[normal_offset + u32(norm_index1) * 3],
            mesh_data_chunk_1[normal_offset + u32(norm_index1) * 3 + 1u],
            mesh_data_chunk_1[normal_offset + u32(norm_index1) * 3 + 2u],
        );
        let norm2 = vec3<f32>(
            mesh_data_chunk_1[normal_offset + u32(norm_index2) * 3],
            mesh_data_chunk_1[normal_offset + u32(norm_index2) * 3 + 1u],
            mesh_data_chunk_1[normal_offset + u32(norm_index2) * 3 + 2u],
        );
        let norm3 = vec3<f32>(
            mesh_data_chunk_1[normal_offset + u32(norm_index3) * 3],
            mesh_data_chunk_1[normal_offset + u32(norm_index3) * 3 + 1u],
            mesh_data_chunk_1[normal_offset + u32(norm_index3) * 3 + 2u],
        );

        let accum = normal_transform * (norm1 + norm2 + norm3);
        return normalize(accum);
    }
    else if mesh_header.chunk_id == 2 {
        let norm_index1 = mesh_data_chunk_2[triangle_offset + inner_id * 3];
        let norm_index2 = mesh_data_chunk_2[triangle_offset + inner_id * 3 + 1u];
        let norm_index3 = mesh_data_chunk_2[triangle_offset + inner_id * 3 + 2u];
        let norm1 = vec3<f32>(
            mesh_data_chunk_2[normal_offset + u32(norm_index1) * 3],
            mesh_data_chunk_2[normal_offset + u32(norm_index1) * 3 + 1u],
            mesh_data_chunk_2[normal_offset + u32(norm_index1) * 3 + 2u],
        );
        let norm2 = vec3<f32>(
            mesh_data_chunk_2[normal_offset + u32(norm_index2) * 3],
            mesh_data_chunk_2[normal_offset + u32(norm_index2) * 3 + 1u],
            mesh_data_chunk_2[normal_offset + u32(norm_index2) * 3 + 2u],
        );
        let norm3 = vec3<f32>(
            mesh_data_chunk_2[normal_offset + u32(norm_index3) * 3],
            mesh_data_chunk_2[normal_offset + u32(norm_index3) * 3 + 1u],
            mesh_data_chunk_2[normal_offset + u32(norm_index3) * 3 + 2u],
        );
        let accum = normal_transform * (norm1 + norm2 + norm3);
        return normalize(accum);

    }
    else if mesh_header.chunk_id == 3 {
        let norm_index1 = mesh_data_chunk_3[triangle_offset + inner_id * 3];
        let norm_index2 = mesh_data_chunk_3[triangle_offset + inner_id * 3 + 1u];
        let norm_index3 = mesh_data_chunk_3[triangle_offset + inner_id * 3 + 2u];
        let norm1 = vec3<f32>(
            mesh_data_chunk_3[normal_offset + u32(norm_index1) * 3],
            mesh_data_chunk_3[normal_offset + u32(norm_index1) * 3 + 1u],
            mesh_data_chunk_3[normal_offset + u32(norm_index1) * 3 + 2u],
        );
        let norm2 = vec3<f32>(
            mesh_data_chunk_3[normal_offset + u32(norm_index2) * 3],
            mesh_data_chunk_3[normal_offset + u32(norm_index2) * 3 + 1u],
            mesh_data_chunk_3[normal_offset + u32(norm_index2) * 3 + 2u],
        );
        let norm3 = vec3<f32>(
            mesh_data_chunk_3[normal_offset + u32(norm_index3) * 3],
            mesh_data_chunk_3[normal_offset + u32(norm_index3) * 3 + 1u],
            mesh_data_chunk_3[normal_offset + u32(norm_index3) * 3 + 2u],
        );
        let accum = normal_transform * (norm1 + norm2 + norm3);
        return normalize(accum);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn get_norm_info_scale(norm_info_scale_offset: u32, chunk_id: u32) -> f32 {
    if chunk_id == 0 {
        return mesh_data_chunk_0[norm_info_scale_offset];
    }
    else if chunk_id == 1 {
        return mesh_data_chunk_1[norm_info_scale_offset];
    }
    else if chunk_id == 2 {
        return mesh_data_chunk_2[norm_info_scale_offset];
    }
    else if chunk_id == 3 {
        return mesh_data_chunk_3[norm_info_scale_offset];
    }
    return 0.0;
}

fn get_norm_for_mesh_triangle(mesh_triangle: MeshTriangle, barycentric: vec2<f32>) -> vec3<f32> {
    let normal_transform = mat3x3<f32>(
        mesh_triangle.normal_transform_c1,
        mesh_triangle.normal_transform_c2,
        mesh_triangle.normal_transform_c3,
    );
    let mesh_id = mesh_triangle.mesh_index;
    let prim_id = mesh_triangle.prim_index;
    let inner_id = mesh_triangle.inner_index;
    let mesh_header = mesh_headers[mesh_id];
    let prim_header = primitive_headers[mesh_header.primitive_header_offset + prim_id];
    if prim_header.has_norm_info == 0u {
        return get_norm_from_norms(mesh_triangle, normal_transform);
    }
    let data_offset = mesh_header.data_offset + prim_header.mesh_data_offset;
    let norm_info_scale_offset = data_offset + prim_header.norm_info_scale_offset;
    let norm_info_scale = get_norm_info_scale(norm_info_scale_offset, mesh_header.chunk_id);
    let norm_info_coords_offset = data_offset + prim_header.norm_info_coords_offset;
    let norm_coord = tex_coord_from_bary(mesh_triangle, norm_info_coords_offset, barycentric);
    let normal_map_offset = data_offset + prim_header.normal_map_data_offset;
    let normal_map_width = prim_header.normal_map_data_width;
    let normal_map_height = prim_header.normal_map_data_height;
    let pixel = get_pixel_from_image(norm_coord, normal_map_offset, normal_map_width, normal_map_height, mesh_header.chunk_id);
    let scaled_norm = norm_info_scale * normal_transform * pixel;
    return normalize(scaled_norm);
}

fn get_metal_rough_direct(mesh_triangle: MeshTriangle) -> vec2<f32> {
    let mesh_id = mesh_triangle.mesh_index;
    let prim_id = mesh_triangle.prim_index;
    let inner_id = mesh_triangle.inner_index;
    let mesh_header = mesh_headers[mesh_id];
    let prim_header = primitive_headers[mesh_header.primitive_header_offset + prim_id];
    let data_offset = mesh_header.data_offset + prim_header.mesh_data_offset;
    let metal_rough_metal_offset = data_offset + prim_header.metal_rough_metal_offset;
    let metal_rough_rough_offset = data_offset + prim_header.metal_rough_rough_offset;
    if mesh_header.chunk_id == 0 {
        return vec2<f32>(
            mesh_data_chunk_0[metal_rough_metal_offset],
            mesh_data_chunk_0[metal_rough_rough_offset],
        );
    }
    else if mesh_header.chunk_id == 1 {
        return vec2<f32>(
            mesh_data_chunk_1[metal_rough_metal_offset],
            mesh_data_chunk_1[metal_rough_rough_offset],
        );
    }
    else if mesh_header.chunk_id == 2 {
        return vec2<f32>(
            mesh_data_chunk_2[metal_rough_metal_offset],
            mesh_data_chunk_2[metal_rough_rough_offset],
        );
    }
    else if mesh_header.chunk_id == 3 {
        return vec2<f32>(
            mesh_data_chunk_3[metal_rough_metal_offset],
            mesh_data_chunk_3[metal_rough_rough_offset],
        );
    }
    return vec2<f32>(0.0, 0.0);
}

fn get_scaled_metal_rough(mesh_triangle: MeshTriangle, barycentric: vec2<f32>, metalness: f32, roughness: f32) -> vec2<f32> {
    let mesh_id = mesh_triangle.mesh_index;
    let prim_id = mesh_triangle.prim_index;
    let inner_id = mesh_triangle.inner_index;
    let mesh_header = mesh_headers[mesh_id];
    let prim_header = primitive_headers[mesh_header.primitive_header_offset + prim_id];
    let data_offset = mesh_header.data_offset + prim_header.mesh_data_offset;
    let metal_rough_coords_offset = data_offset + prim_header.metal_rough_coords_offset;
    let metal_rough_coord = tex_coord_from_bary(mesh_triangle, metal_rough_coords_offset, barycentric);
    let metal_rough_map_offset = data_offset + prim_header.metal_rough_map_data_offset;
    let metal_rough_map_width = prim_header.metal_rough_map_data_width;
    let metal_rough_map_height = prim_header.metal_rough_map_data_height;
    let pixel = get_pixel_from_image(metal_rough_coord, metal_rough_map_offset, metal_rough_map_width, metal_rough_map_height, mesh_header.chunk_id);
    return vec2<f32>(
        pixel.z * metalness,
        pixel.y * roughness,
    );
}

fn get_diff_spec_and_roughness(mesh_triangle: MeshTriangle, ray: Ray, norm: vec3<f32>, barycentric: vec2<f32>, rng: ptr<function, u32>) -> DynDiffSpec {
    let mesh_id = mesh_triangle.mesh_index;
    let prim_id = mesh_triangle.prim_index;
    let inner_id = mesh_triangle.inner_index;
    let mesh_header = mesh_headers[mesh_id];
    let prim_header = primitive_headers[mesh_header.primitive_header_offset + prim_id];
    let metal_rough = get_metal_rough_direct(mesh_triangle);
    var metalness = metal_rough.x;
    var roughness = metal_rough.y;
    if prim_header.metal_rough_coords_count != 0u {
        let scaled_metal_rough = get_scaled_metal_rough(mesh_triangle, barycentric, metalness, roughness);
        metalness = scaled_metal_rough.x;
        roughness = scaled_metal_rough.y;
    }
    let r0 = 0.04 + (1.0 - 0.04) * metalness; // based on gltf definition of metalness for fresnel
    let reflectance = r0 + (1.0 - r0) * CUSTOM_ATTEN * (1.0 - pow(abs(dot(ray.direction, norm)), 5.0));

    let diffp = 1.0 - reflectance;

    let u = get_random_f32(rng);
    let should_diff = (u < diffp);
    
    return DynDiffSpec(should_diff, roughness);
}

///////////////////////////////
// Cube map functions
///////////////////////////////
fn num_cube_map_faces() -> u32 {
    return select(arrayLength(&cube_map_headers), 0u, arrayLength(&cube_map_headers) == 1u && cube_map_headers[0].width == 0u);
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


//////////////////////////////////
// Random Number Generation
/////////////////////////////////

// Shoutout to: https://github.com/boksajak/referencePT
fn initRng(global_id: vec3<u32>, in_seed: u32) -> u32 {
    let seed: u32 = dot(global_id.xy, vec2(render_info.width, render_info.height)) ^ jenkinsHash(in_seed);
    return jenkinsHash(seed);
}

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

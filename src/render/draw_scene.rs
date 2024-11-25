use std::iter::zip;
use super::RenderTarget;
use crate::ray::RayCompute;
use crate::scene::Scene;
use crate::elements::{Renderable, Element};
use super::radiance::radiance;
use crate::accel::KdTree;
use crate::render::cpu_utils::RenderInfo;
use crate::render::gpu_utils::GPUState;
use crate::render::gpu_structs::{GPUCamera, GPURenderInfo};

/*
    Buffer design:
     1. render_target.chunk_to_pix => iterates index to x/y pixels. 
                                   => Not needed to put into buffer. Can iterate within the shader.
     
     2. kdtree => see how to organize into buffer?
               => Takes in renderables - can actually calculate this inside the shader?
               => If not allowed, we can add as buffer later after non-optimized shader written.
     
     3. ray => generated within shader, or should we pass in as a buffer?
     
     4. renderables => MAIN THING NEEDED FOR BUFFER!!!
                    => Storagebuffer
                    => Size 16 Vec, so it's already aligned. Should be plug & play as a buffer.
                    => Done, try

     5. render_info => parameters for rendering. Need to organize into buffer
                    => Struct made. Try to add to shader

     6. render_target.buff_mux => output of renderer. Already a buffer, so need to pass it into the shader to get to write to it
*/

pub fn render_to_target<F: Fn() -> ()>(render_target: &RenderTarget, scene: &Scene, update_hook: F, render_info: &RenderInfo) {
    if render_info.use_gpu {
        render_to_target_gpu(render_target, scene, update_hook, render_info);
    } else {
        render_to_target_cpu(render_target, scene, update_hook, render_info);
    }
}

fn print_gpu_results(results: &Vec<f32>) {
    println!("Got result from compute pipeline with length {}", results.len());   
    let is_zero = results.iter().all(|&x| x == 0.0);
    let is_one = results.iter().all(|&x| x == 1.0);
    println!("All results are zero: {}", is_zero);
    println!("All results are one: {}", is_one);
}

fn render_to_target_gpu<F : Fn() -> ()>(render_target: &RenderTarget, scene: &Scene, _update_hook: F, render_info: &RenderInfo) {
    let mut gpu_state = GPUState::new();
    let gpu_camera = GPUCamera::from_cam(&scene.cam);
    let gpu_render_info = GPURenderInfo::from_render_info(render_info);
    gpu_state.create_compute_pipeline(&gpu_camera, &gpu_render_info, &render_target);
    gpu_state.dispatch_compute_pipeline();
    gpu_state.submit_compute_pipeline();
    let results: Vec<f32> = gpu_state.block_and_get_single_result();
    print_gpu_results(&results);
}

fn render_to_target_cpu<F : Fn() -> ()>(render_target: &RenderTarget, scene: &Scene, update_hook: F, render_info: &RenderInfo) {
    use rayon::prelude::*;

    let ray_compute = RayCompute::new((&render_target.canv_width, &render_target.canv_height), &scene.cam);

    use std::time::Instant;
    let start = Instant::now();

    render_target.buff_mux.lock().fill(0);
    let mut sample_count: f32 = 0.0;
    let mut target: Vec<[f32; 3]> = [[0.0, 0.0, 0.0]].repeat((render_target.canv_width * render_target.canv_height).try_into().unwrap());

    // scene decomposing into renderables
    let (pure_elem_refs, decomposed_groups) = decompose_groups(&scene.members);
    let renderables: Vec<Renderable> = pure_elem_refs.into_iter().chain(decomposed_groups.iter().map(|e| e.as_ref())).collect();

    let unconditional: Vec<_> = renderables.iter().enumerate()
        .filter_map(|(i, r)| match r.give_aabb() {
            Some(_) => None,
            None => Some((i, *r)),
        })
        .collect();
    let elems_and_aabbs: Vec<_> = renderables.iter().enumerate()
        .filter_map(|(i, r)| r.give_aabb().map(|aabb| (i, *r, aabb)))
        .collect();
    let kdtree = KdTree::build(&elems_and_aabbs, &unconditional, render_info.kd_tree_depth);

    // let num_samples = 100000;
    for r_it in 0..render_info.samps_per_pix {
        target.par_iter_mut()
            .enumerate()
            .map(|(i, pix)| (render_target.chunk_to_pix(i.try_into().unwrap()), pix))
            .for_each(|((x, y), pix)| {
                let ray = ray_compute.pix_cam_to_rand_ray((x,y), &scene.cam);
                let (rgb, _) = radiance(&ray, &kdtree, &renderables, 0, &render_info.rad_info);
                let rgb: Vec<f32> = rgb.iter().copied().collect();

                zip(pix.iter_mut(), &rgb).for_each(|(p, r)| {
                    *p = (r + (*p * sample_count)) / (sample_count + 1.0);
                });
            });

        sample_count += 1.0;

        render_target.buff_mux.lock()
            .par_chunks_mut(4) // pixels have rgba values, so chunk by 4
            .zip(&target)
            .for_each(|(pix, tar)| {
                pix.copy_from_slice(&rgb_f_to_u8(tar));
                pix[3] = 255; // alpha value
            });

        update_hook();
        println!("render iteration {}: {:?}", r_it, start.elapsed());
    }

    let elapsed = start.elapsed();
    println!("elapsed {:?}", elapsed);
}


fn rgb_f_to_u8(f: &[f32]) -> [u8; 4] {
    let mut out: [u8; 4] = [0; 4];
    // 255.0 * (1.0 - 1.0 / (f * 10.0 + 1.0)) // this from smallpt
    zip(out.iter_mut(), f.iter()).for_each(|(e, f)| *e = (f.clamp(0.0, 1.0) * 255.0 + 0.5).trunc() as u8); // assume 0.0 -> 1.0 range
    out
}

use crate::scene::Member;
fn decompose_groups<'e>(members: &'e Vec<Member<'e>>) -> (Vec<Renderable<'e>>, Vec<Element<'e>>) {
    let mut pure_elem_refs: Vec<Renderable> = vec![];
    let mut group_iters: Vec<Box<dyn Iterator<Item = Element>>> = vec![];

    members.iter().for_each(|m| {
        use crate::scene::Member::*;
        match m {
            Elem(e) => { pure_elem_refs.push(e.as_ref()); },
            Grp(g) => { group_iters.push(g.decompose_to_elems()) },
        }
    });

    let decomposed: Vec<Element<'e>> = group_iters.into_iter().flatten().collect();

    (pure_elem_refs, decomposed)
}
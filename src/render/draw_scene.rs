use std::iter::zip;
use super::RenderTarget;
use crate::ray::RayCompute;
use crate::scene::Scene;
use crate::elements::{Renderable, Element};
use super::radiance::{radiance, RadianceInfo};
use crate::accel::KdTree;

use encase::{DynamicStorageBuffer, ShaderType, UniformBuffer, StorageBuffer};
use serde::Deserialize;

// RadianceInfo:
//  debug_single_ray: bool,
//  dir_light_samp: bool,
//  russ_roull_info: RussianRoullInfo,

#[derive(Deserialize, Debug, ShaderType)]
pub struct RenderInfo_gpu {
    pub width: i32,
    pub height: i32,
    pub samps_per_pix: i32,
    pub assured_depth: i32,
    pub max_threshold: f32,
    pub kd_tree_depth: u32,
}

#[derive(Deserialize, Debug)]
pub struct RenderInfo {
    pub width: i32,
    pub height: i32,
    pub samps_per_pix: i32,
    pub rad_info: RadianceInfo,
    pub kd_tree_depth: usize,
    pub use_gpu: bool,
}

// TODO: Optimization candidate
// Main renderer - need to go through this to see what can be reduced, parallelized, etc.
pub fn render_to_target<F : Fn() -> ()>(render_target: &RenderTarget, scene: &Scene, update_hook: F, render_info: &RenderInfo) {
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
    
    // TODO: original optimizations
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
    
    // Redo buffer to not need to clone
    // The example has separate buffers for the texture & normal map, but I think this should contain everything already
    // 4. renderables
    let mut renderables_buf = DynamicStorageBuffer::new_with_alignment(&mut renderables.clone(), 64);
    
    // 5. render_info
    let render_info_gpu = RenderInfo_gpu {
        width: render_info.width,
        height: render_info.height,
        samps_per_pix: render_info.samps_per_pix,
        assured_depth: render_info.rad_info.russ_roull_info.assured_depth,
        max_threshold: render_info.rad_info.russ_roull_info.max_thres,
        kd_tree_depth: render_info.kd_tree_depth as u32,
    };
    let mut render_info_buf = UniformBuffer::new(&render_info_gpu);

    // 6. render_target.buff_mux
    // I think this should work
    let mut render_output_buf = StorageBuffer::new(target);

    for r_it in 0..render_info.samps_per_pix {
        target.par_iter_mut()
            .enumerate()
            .map(|(i, pix)| (render_target.chunk_to_pix(i.try_into().unwrap()), pix))
            .for_each(|((x, y), pix)| {
                let ray = ray_compute.pix_cam_to_rand_ray((x,y), &scene.cam); // TODO: random ray - can we make less random to cover all pixels better?
                let (rgb, _) = radiance(&ray, &kdtree, &renderables, 0, &render_info.rad_info);
                let rgb: Vec<f32> = rgb.iter().copied().collect();
                                                          //pix         rgb
                zip(pix.iter_mut(), &rgb).for_each(|(p, r)| {
                    *p = (r + (*p * sample_count)) / (sample_count + 1.0);
                });
            });

        sample_count += 1.0;
        
        // Shader output should go here...
        // Requires a mutex for this --> performance bottleneck?
        render_target.buff_mux.lock()
            .par_chunks_mut(4) // pixels have rgba values, so chunk by 4
            .zip(&target)
            .for_each(|(pix, tar)| {
                pix.copy_from_slice(&rgb_f_to_u8(tar));
                pix[3] = 255; // alpha value
            });
        
        // self.sender.send(self.target.buff_mux) from Renderer struct in lib.rs
        update_hook();
        println!("render iteration {}: {:?}", r_it, start.elapsed());
    }

    let elapsed = start.elapsed();
    println!("elapsed {:?}", elapsed);
}


// TODO: this quantizes f32 to u8. Happens every iteration for every pixel. Is this necessary?
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
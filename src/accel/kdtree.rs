use crate::elements::Renderable;
use super::{Aabb, PlaneBounds};
use crate::ray::{Ray, closest_ray_hit, ClosestRayHit};
use nalgebra::Vector3;

// TODO: Make this GPU compatible
// pub struct GPUKdTree<'k> {
//     aabb: Aabb,
//     node: Node<'k>,
//     unconditional: &'k Vec<(usize, Renderable<'k>)>,
// }

// Cannot derive ShaderType -> make custom
pub struct KdTree<'k> {
    aabb: Aabb,
    node: Node<'k>,
    unconditional: &'k Vec<(usize, Renderable<'k>)>,
}

pub enum Node<'n> {
    Branch { axis: usize, split: f32, low: Box<Node<'n>>, high: Box<Node<'n>> },
    Leaf (Vec<(usize, Renderable<'n>)>),
}

impl<'k> KdTree<'k> {
    pub fn build(elems_and_aabbs: &Vec<(usize, Renderable<'k>, Aabb)>, unconditional: &'k Vec<(usize, Renderable<'k>)>, max_build_depth: usize) -> Self {
        let aabbs: Vec<&Aabb> = elems_and_aabbs.iter().map(|(_,_,aabb)| aabb).collect();

        let aabb = {
            let min_axes: Vec<f32> = (0..3).map(
                |a| (&aabbs).into_iter().map(|aabb| aabb.bounds[a].low)
                    .reduce(|pl, l| pl.min(l))
                    .unwrap()
                )
                .collect();
            let max_axes: Vec<f32> = (0..3).map(
                |a| (&aabbs).into_iter().map(|aabb| aabb.bounds[a].high)
                    .reduce(|ph, h| ph.max(h))
                    .unwrap()
                )
                .collect();
            Aabb {
                bounds: [
                    PlaneBounds {low: min_axes[0], high: max_axes[0]},
                    PlaneBounds {low: min_axes[1], high: max_axes[1]},
                    PlaneBounds {low: min_axes[2], high: max_axes[2]},
                ]
            }
        };

        KdTree {
            aabb,
            unconditional,
            node: node_from_elems(&elems_and_aabbs.iter().map(|(i, e, aabb)| (*i, *e, aabb)).collect(), 0, max_build_depth),
        }
    }

    pub fn closest_ray_hit(&self, ray: &Ray) -> ClosestRayHit {
        let enters_domain = self.aabb.get_entry_exit(ray);
        match enters_domain {
            None => closest_ray_hit(ray, self.unconditional.iter().map(|e| *e)),
            Some(((_, entry_t), (_, exit_t))) => self.stack_search(ray, entry_t, exit_t),
        }
    }
    
    fn stack_search(&self, ray: &Ray, entry_t: f32, exit_t: f32) -> ClosestRayHit {
        // adapted from https://dcgi.fel.cvut.cz/home/havran/ARTICLES/cgf2011.pdf 
        let mut stack: Vec<(&Node, f32, f32)> = vec![(&self.node, entry_t, exit_t)];
        
        use Node::*;
        while !stack.is_empty() {
            let (mut current_node, entry_t, mut exit_t) = stack.pop().unwrap();
            while let Branch {axis, split, low, high} = current_node {
                let a = *axis;
                let d = if ray.d[a].abs() < crate::EPS { 
                    if ray.d[a] < 0.0 { -crate::EPS } else { crate:: EPS}
                } else { ray.d[a] };
                let t = (split - ray.o[a]) / d; // apparently split is a point in the paper? lets see how it goes
                let (near, far) = if d > 0.0 {(low, high)} else {(high, low)};
                if t >= exit_t {
                    current_node = near;
                } else if t <= entry_t {
                    current_node = far;
                } else {
                    stack.push((far, t, exit_t));
                    current_node = near;
                    exit_t = t;
                }
            }
            
            if let Leaf(elems) = current_node {
                let (hit_results, idxo) = closest_ray_hit(ray, elems.into_iter().map(|e| *e));
                if let Some(hr_idx) = idxo { 
                    let (_elem_idx, hit_result) = &hit_results[hr_idx];
                    let hit_result = &hit_result.as_ref().unwrap();
                    if hit_result.l.0 <= (exit_t + crate::EPS) {
                        return (hit_results, idxo); // may need to handle case of primitive on the edge of the node volume
                    }
                }
            }
        }

        closest_ray_hit(ray, self.unconditional.iter().map(|e| *e))
    }
}

fn node_from_elems<'n>(elems_and_aabbs: &Vec<(usize, Renderable<'n>, &Aabb)>, depth: usize, max_depth: usize) -> Node<'n> {
    let axis = depth % 3;
    if depth > max_depth || elems_and_aabbs.len() <= 1 {
        Node::Leaf(elems_and_aabbs.iter().map(|(i, e, _)| (*i, *e)).collect())
    } else {
        let aabbs: Vec<&Aabb> = elems_and_aabbs.iter().map(|(_,_,aabb)| *aabb).collect();
        let split = (&aabbs).into_iter().map(|aabb| aabb.centroid()).sum::<Vector3<f32>>() / (aabbs.len() as f32);

        let (low, high): (Vec<(usize, Renderable, &Aabb)>, Vec<(usize, Renderable, &Aabb)>) = {
            let mut low: Vec<(usize, Renderable, &Aabb)> = vec![];
            let mut high: Vec<(usize, Renderable, &Aabb)> = vec![];
    
            elems_and_aabbs.iter().for_each(|(i, e, aabb)| {
                // this can handle case of element in both nodes
                if aabb.bounds[axis].high >= split[axis] {
                    high.push((*i, *e, aabb));
                }
                if aabb.bounds[axis].low <= split[axis] {
                    low.push((*i, *e, aabb));
                }
            });
            (low, high)
        };

        Node::Branch { 
            axis, 
            split: split[axis], 
            low: Box::new(node_from_elems(&low, depth + 1, max_depth)), 
            high: Box::new(node_from_elems(&high, depth + 1, max_depth))
        }
    }
}
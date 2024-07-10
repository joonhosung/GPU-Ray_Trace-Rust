use nalgebra::Vector3;
use crate::ray::{Ray, Hitable, HitResult};

pub struct Sphere {
    pub c: Vector3<f32>,
    pub r: f32,
}

impl Hitable<()> for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<HitResult<()>> {        
        // solve quadratic equation for sphere-ray intersection, from https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        let oc = ray.o - self.c;
        let dir = ray.d.dot(&oc);
        let consts = oc.dot(&oc) - self.r * self.r;

        let thing = dir * dir - consts;
        if thing > 0.0 {
            Some(HitResult{l: -1.0, intermed: ()})
        } else {
            None
        }
    }
}
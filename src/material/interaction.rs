use nalgebra::Vector3;
use crate::ray::Ray;
use rand::Rng;

// TODO: How the light interacts with the object. Lots of math here, optimization candidate
pub fn spec(ray: &Ray, norm: &Vector3<f32>, o: &Vector3<f32>) -> Ray {
    let d = (ray.d - norm * 2.0 * ray.d.dot(&norm)).normalize();
    Ray {d, o: o.clone()}
}

pub fn diff(ray: &Ray, norm: &Vector3<f32>, o: &Vector3<f32>) -> Ray {
    // cosine weighted hemisphere importance sampling based on https://www.mathematik.uni-marburg.de/~thormae/lectures/graphics1/code/ImportanceSampling/importance_sampling_notes.pdf
    let xd = (ray.d - norm * (ray.d.dot(&norm))).normalize();
    let yd = (norm.cross(&xd)).normalize();

    let u: f32 = crate::RNG.with_borrow_mut(|r| r.gen());
    let v: f32 = crate::RNG.with_borrow_mut(|r| r.gen());

    let r = u.sqrt();
    let thet = 2.0 * std::f32::consts::PI * v;

    let x = r * thet.cos();
    let y = r * thet.sin();
    let d = (xd * x + yd * y + norm * (1.0 - u).max(0.0).sqrt()).normalize();

    Ray {d, o: o.clone()}
}

pub fn refract(ray: &Ray, norm: &Vector3<f32>, o: &Vector3<f32>, n_out: &f32, n_in: &f32) -> (Ray, f32) {
    // adapt from scratchapixel and smallpt
    // this helped a bit: https://blog.demofox.org/2020/06/14/casual-shadertoy-path-tracing-3-fresnel-rough-refraction-absorption-orbit-camera/
    let c_ = norm.dot(&ray.d);
    let into: bool = c_ < 0.0;
    let (n1, n2, c1, norm_refr) = if into {
        (*n_out, *n_in, -c_, norm.clone())
    } else {
        (*n_in, *n_out, c_, -norm)
    };
    let n_over = n1 / n2;
    let c22 = 1.0 - n_over * n_over * (1.0 - c1 * c1);

    let total_internal: bool = c22 < 0.0;
    let refl = spec(ray, &norm_refr, o);
    if total_internal {
        (refl, 1.0)
    } else {
        let trns = n_over * ray.d + norm_refr * (n_over * c1 - c22.sqrt()); // derived from snells law
        let r0 = ((n1 - n2) / (n1 + n2)).powf(2.0);
        let c = 1.0 - if into { c1 } else { trns.dot(norm) };
        let re = r0 + (1.0 + r0) * c.powf(5.0); // schlick approximation for reflection coef in fresnel equation
        
        let u: f32 = crate::RNG.with_borrow_mut(|r| r.gen());

        if u < re {
            (refl, re)
        } else {
            (Ray {d: trns.normalize(), o: o.clone()}, 1.0 - re)
        }
    }
}
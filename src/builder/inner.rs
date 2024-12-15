use serde::Deserialize;
use crate::types::GPUElements;
use crate::elements::mesh;
use crate::elements::sphere::Sphere;
// use crate::elements::Element;
use crate::scene::Member;
use crate::elements::distant_cube_map;
use crate::elements::triangle;
use super::pr;
// use super::pr::Cam;
use keyframe::{Keyframe, AnimationSequence};
use nalgebra::Vector3;
// use keyframe::mint::Vector3;
use keyframe::mint::Point3;
use MemberTypes::*;


#[derive(Deserialize, Debug, Clone)]
pub struct VecInto<T>(Vec<T>); // wrapper st if elements have into one type to another, easily convert this vec into vec of another

impl From<VecInto<MemberTypes>> for Vec<Member<'_>> {
    fn from(mts: VecInto<MemberTypes>) -> Self {
        let mut members: Vec<Member<'_>> = vec![];
        // let mut group_iters: Vec<Box<dyn Iterator<Item = Element>>> = vec![];

        mts.0.into_iter().for_each(|m| {
            use MemberTypes::*;
            // use crate::scene::Member::*;
            match m {
                Sphere(s) => {
                    members.push(Member::Elem(Box::new(s)));
                },
                DistantCubeMap(prcs) => {
                    members.push(Member::Elem(
                        Box::new(distant_cube_map::DistantCubeMap {
                            neg_z: prcs.neg_z.into(),
                            pos_z: prcs.pos_z.into(),
                            neg_x: prcs.neg_x.into(),
                            pos_x: prcs.pos_x.into(),
                            neg_y: prcs.neg_y.into(),
                            pos_y: prcs.pos_y.into(),
                        })));
                },
                FreeTriangle(t) => {
                    members.push(Member::Elem(
                        Box::new(
                            triangle::FreeTriangle {
                                norm: t.norm.normalize().into(),
                                verts: t.verts,
                                rgb: t.rgb,
                                diverts_ray: t.mat,
                                type_name: "FreeTriangle".to_string(),
                            },
                    )));
                },
                Model(m) => {
                    members.extend(m.to_meshes().into_iter().map(|m| Member::Grp(Box::new(m))));
                },
            }
        });

        members
    }
}


impl VecInto<MemberTypes> {
    pub fn extract_concrete_types(self: VecInto<MemberTypes>) -> GPUElements {
        let mut spheres: Vec<Sphere> = vec![];
        let mut distant_cubemaps: Vec<distant_cube_map::DistantCubeMap> = vec![];
        let mut free_triangles: Vec<triangle::FreeTriangle> = vec![];
        let mut meshes: Vec<mesh::Mesh> = vec![];

        self.0.iter().for_each(|m| {
            match m {
                Sphere(s) => {
                    spheres.push(s.clone());
                },
                DistantCubeMap(prcs) => {
                    distant_cubemaps.push(
                        distant_cube_map::DistantCubeMap {
                            neg_z: prcs.neg_z.clone().into(),
                            pos_z: prcs.pos_z.clone().into(),
                            neg_x: prcs.neg_x.clone().into(),
                            pos_x: prcs.pos_x.clone().into(),
                            neg_y: prcs.neg_y.clone().into(),
                            pos_y: prcs.pos_y.clone().into(),
                        }
                    );
                },
                FreeTriangle(t) => {
                    free_triangles.push(
                        triangle::FreeTriangle {
                            norm: t.norm.normalize().into(),
                            verts: t.verts,
                            rgb: t.rgb,
                            diverts_ray: t.mat.clone(),
                            type_name: "FreeTriangle".to_string(),
                        },
                    )
                },
                Model(model) => {
                    meshes.extend(model.to_meshes().into_iter());
                },
            }
        });

        return (spheres, distant_cubemaps, free_triangles, meshes);
    
    }

    // Extract all the locations of the members for each frame
    pub fn extract_anim(self: VecInto<MemberTypes>, framerate: f32/*, cam: Cam*/) -> Vec<VecInto<MemberTypes>> {
        
        let max_time: f64 = self.get_last_timestamp() as f64;
        let time_per_frame: f64 = 1.0 / framerate as f64;
        let number_of_frames: usize = (max_time/time_per_frame) as usize;
        let mut frames: Vec<VecInto<MemberTypes>> = Vec::with_capacity(number_of_frames);
        
        for _ in 0..number_of_frames{
            frames.push(VecInto::<MemberTypes>{0: Vec::<MemberTypes>::new()});
        }

        println!("Extracting frames: \n\t Number of frames: {number_of_frames}\n\t Frame vec size: {}\n\t Time per frame {time_per_frame:.4?}\n\t Total time: {max_time}s", frames.len());
        self.0.iter().for_each(|m| {            
            match m {
                // Infer the locations of Sphere and Model translations for each frame
                Sphere(s) => {
                    match &s.animation {
                        Some(anim) => {
                            // let hi = anim.keyframes[0].translation.x;
                            

                            let mut sequence = AnimationSequence::<Point3<f32>>::new();
                            for frame in &anim.keyframes {
                                
                                let translation: Point3<f32> = Point3{x: frame.translation.x, y: frame.translation.y, z: frame.translation.z};
                                // s.clone().c
                                sequence.insert(Keyframe::new_dynamic(translation, frame.time, frame.get_ease_type()))
                                    .expect("Something happened while generating keyframe sequence for translation!!");
                            }

                            for i in 0..number_of_frames{
                                let mut frame_to_insert = s.clone();
                                let (x, y, z) = (sequence.now_strict().unwrap().x, sequence.now_strict().unwrap().y, sequence.now_strict().unwrap().z);
                                frame_to_insert.c = Vector3::new(x, y, z);
                                frames[i].0.push(MemberTypes::Sphere(frame_to_insert));
                                sequence.advance_by(time_per_frame);
                            }
                        }, 
                        None => {
                            frames.iter_mut().for_each(|frame| {
                                frame.0.push(MemberTypes::Sphere(s.clone()));
                            }); 
                        },
                    }
                }, 
                Model(m) => {
                    match &m.animation {
                        Some(anim) => {
                            // let hi = anim.keyframes[0].translation.x;
                            

                            let mut sequence_trans = AnimationSequence::<Point3<f32>>::new();
                            let mut sequence_angle = AnimationSequence::<Point3<f32>>::new();
                            
                            for frame in &anim.keyframes {
                                
                                let translation: Point3<f32> = Point3{x: frame.translation.x, y: frame.translation.y, z: frame.translation.z};
                                let angle: Point3<f32> = Point3{x: frame.euler_angles.unwrap().x, y: frame.euler_angles.unwrap().y, z: frame.euler_angles.unwrap().z};

                                // s.clone().c
                                sequence_trans.insert(Keyframe::new_dynamic(translation, frame.time, frame.get_ease_type()))
                                    .expect("Something happened while generating keyframe sequence for translation!!");
                                sequence_angle.insert(Keyframe::new_dynamic(angle, frame.time, frame.get_ease_type()))
                                    .expect("Something happened while generating keyframe sequence for angle!!");
                            }


                            for i in 0..number_of_frames{
                                let mut frame_to_insert = m.clone();
                                let (x, y, z) = (sequence_trans.now_strict().unwrap().x, sequence_trans.now_strict().unwrap().y, sequence_trans.now_strict().unwrap().z);
                                let (u, v, w) = (sequence_angle.now_strict().unwrap().x, sequence_angle.now_strict().unwrap().y, sequence_angle.now_strict().unwrap().z);
                                frame_to_insert.translation = Vector3::new(x, y, z);
                                frame_to_insert.euler_angles = [u, v, w];

                                frames[i].0.push(MemberTypes::Model(frame_to_insert));
                                sequence_trans.advance_by(time_per_frame);
                                sequence_angle.advance_by(time_per_frame);
                            } 
                        }, 
                        None => {
                            frames.iter_mut().for_each(|frame| {
                                frame.0.push(MemberTypes::Model(m.clone()));
                            }); 
                        },
                    }
                },

                // No animation for SkyBox or Triangles, but need to copy them into each frame's scene
                FreeTriangle(t) => {
                    frames.iter_mut().for_each(|frame| {
                        frame.0.push(MemberTypes::FreeTriangle(t.clone()));
                    });
                },

                DistantCubeMap(d) => {
                    frames.iter_mut().for_each(|frame| {
                        frame.0.push(MemberTypes::DistantCubeMap(d.clone()));
                    });
                }, 
            }
        });

        frames
    }

    fn get_last_timestamp(&self) -> f32 {
        let last_timestamp: f32 = self.clone().0.into_iter().map(|m| {
            use MemberTypes::*;
            let mut final_time: f32 = 0.0;
            match m {
                Sphere(s) => {
                    match s.animation {
                        Some(s) => {
                            final_time = s.keyframes.last().unwrap().time;
                            // for frame in s.keyframes {
                                // frame.time 
                            // }
                        }, 
                        None => {},
                    }
                }, 
                Model(m) => {
                    match m.animation {
                        Some(m) => {
                            final_time = m.keyframes.last().unwrap().time;
                        }, 
                        None => {},
                    }
                },

                _ => {}, // No animation for SkyBox or Triangles
            }
            final_time
        }).reduce(f32::max).unwrap();

        last_timestamp
    }
}


impl<A, B> From<VecInto<A>> for Vec<B> 
where
    B: From<A>
{
    fn from(val: VecInto<A>) -> Self {
        let VecInto(contents) = val;
        contents.into_iter().map(|t| t.into()).collect()
    }
}

#[derive(Deserialize, Debug, Clone)]
pub enum MemberTypes {
    Sphere(Sphere),
    DistantCubeMap(pr::DistantCubeMap),
    FreeTriangle(pr::FreeTriangle),

    Model(pr::Model),
}


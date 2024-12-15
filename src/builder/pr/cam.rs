use serde::Deserialize;
use nalgebra::{Matrix4, Vector3};
use crate::scene;

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct Cam {
    pub d: Vector3<f32>, // o -> center of screen, has distance
    pub o: Vector3<f32>,
    pub up: Vector3<f32>, // should be unit vector
    // in-scene dimensions, not view pixels
    pub screen_width: f32, 
    pub screen_height: f32,
    pub lens_r: Option<f32>,
    pub lookat: Option<Vector3<f32>>,
    pub follow: Option<usize>,
    view_eulers: [f32; 3], // Camera rotation based on up
}

impl From<Cam> for scene::Cam {
    fn from(c_: Cam) -> Self {
        let Cam {
            d, o, up, screen_width, screen_height, lens_r, lookat, follow, view_eulers
        } = c_;
        let [r, p, y] = view_eulers;
        let /*mut*/ rot;
        // let rot2;
        let /*mut*/ rot_use;
        
        

        // if let Some(lookat) = lookat {
        //     rot = look_at_lh(&(o).into(), &(lookat).into(), &up);//Vector3::<f32>::new(0.0,1.0,0.0).into());
        //     // // rot = Matrix4::<f32>::from_euler_angles(0.0, p, y);
        //     // let mut diff: Vector3<f32> = lookat - o;
        //     // diff = diff.normalize();
        //     // let v: Vector3<f32> = d.normalize().cross(&diff);
        //     // let identity = Matrix3::<f32>::new(
        //     //     1.0, 0.0,0.0,
        //     //     0.0, 1.0, 0.0,
        //     //     0.0, 0.0, 1.0);
                
        //     // let skew_mat = Matrix3::<f32>::new(
        //     //     0.0,-v.z, v.y,
        //     //     v.z, 0.0, -v.x,
        //     //     -v.y, v.x, 0.0);

        //     // rot2 = identity + skew_mat + (skew_mat.pow(2) / (1.0+diff.dot(&d)));
        //     // rot_use = rot2;
        //     println!("before inverse: {rot}");
        //     // rot = rot.try_inverse().unwrap();
        //     rot_use = rot.fixed_resize::<3,3>(0.0);
        //     // println!("Rot2  : {rot2}");
        //     println!("lookat: {rot}");
        //     println!("rot_resize: {rot_use}");
        //     // let cam_to_target = lookat - o;
        //     // println!("cam_to_target: {cam_to_target}");
        //     // let pitch = (d.yz().dot(&cam_to_target.yz())) / (d.yz().norm() * cam_to_target.yz().norm());
        //     // let yaw = (d.xz().dot(&cam_to_target.xz())) / (d.xz().norm() * cam_to_target.xz().norm());
            
        //     // println!("pitch: {pitch} | yaw: {yaw}");
        //     // let pitch_delta = ((d.xy().dot(&cam_to_target.xy())) / (d.xy().norm() * cam_to_target.xy().norm())).acos();
        //     // let yaw_delta = ((d.xz().dot(&cam_to_target.xz())) / (d.xz().norm() * cam_to_target.xz().norm())).acos();
        //     // println!("pitch_delta: {pitch_delta} | yaw_delta: {yaw_delta}");
        //     // rot_use = Matrix4::<f32>::from_euler_angles (pitch_delta, yaw_delta, 0.0).fixed_resize::<3,3>(0.0);
        // } else {
            rot = Matrix4::<f32>::from_euler_angles(r, p, y);
            rot_use = rot.fixed_resize::<3,3>(0.0);
        // }
        
        // let vector = Vector3::<f32>::new(0.0,1.0,0.0);
        
        println!("BEFORE d: {d:?} | up: {up:?}");
        let d = rot_use * d;
        let up = rot_use * up;

        // let o = o + rot.column(3).fixed_resize::<3,1>(0.0);
        println!("d: {d} | up: {up} | o: {o}");
        

        Self { d, o, up, screen_width, screen_height, lens_r }
    }   
}

impl Cam {
    // Update the view_eulers to look at a certain point
    fn look_at(self, pos: Vector3<f32>) {
        // self.view_eulers = 
    }
}
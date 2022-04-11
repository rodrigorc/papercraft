use crate::glr::{self, Rgba};
use crate::util_3d::*;

//////////////////////////////////////
// Uniforms and vertices

crate::uniform! {
    pub struct Uniforms3D {
        pub lights: [Vector3; 2],
        pub m: Matrix4,
        pub mnormal: Matrix3,
        pub tex: i32,
        pub line_top: i32,
    }
    pub struct Uniforms2D {
        pub m: Matrix3,
        pub tex: i32,
        pub frac_dash: f32,
        pub line_color: Rgba,
    }
    pub struct UniformQuad {
        pub color: Rgba,
    }
}

crate::attrib! {
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MVertex3D {
        pub pos: Vector3,
        pub normal: Vector3,
        pub uv: Vector2,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MVertex3DLine {
        pub pos: Vector3,
        pub color: Rgba,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MVertex2D {
        pub pos: Vector2,
        pub uv: Vector2,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MVertex2DColor {
        pub pos: Vector2,
        pub uv: Vector2,
        pub color: Rgba,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MVertex2DLine {
        pub pos: Vector2,
        pub line_dash: f32,
        pub width_left: f32,
        pub width_right: f32,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MVertexQuad {
        pub pos: Vector2,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MStatus3D {
        pub color: Rgba,
        pub top: u8,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MStatus2D {
        pub color: Rgba,
    }
}


pub const MSTATUS_UNSEL: MStatus3D = MStatus3D { color: Rgba::new(0.0, 0.0, 0.0, 0.0), top: 0 };
pub const MSTATUS_SEL: MStatus3D = MStatus3D { color: Rgba::new(0.0, 0.0, 1.0, 0.5), top: 1 };
pub const MSTATUS_HI: MStatus3D = MStatus3D { color: Rgba::new(1.0, 0.0, 0.0, 0.75), top: 1 };

pub fn program_from_source(shaders: &str) -> glr::Program {
    let split = shaders.find("###").unwrap();
    let vertex = &shaders[.. split];
    let frag = &shaders[split ..];
    let split_2 = frag.find('\n').unwrap();

    let mut frag = &frag[split_2 ..];

    let geom = if let Some(split) = frag.find("###") {
        let geom = &frag[split ..];
        frag = &frag[.. split];
        let split_2 = geom.find('\n').unwrap();
        Some(&geom[split_2 ..])
    } else {
        None
    };

    glr::Program::from_source(vertex, frag, geom).unwrap()
}

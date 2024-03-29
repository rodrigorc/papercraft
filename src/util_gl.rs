use crate::glr::{self, Rgba, UniformField};
use crate::paper::MaterialIndex;
use crate::util_3d::*;
use anyhow::{anyhow, Result};
use easy_imgui_window::easy_imgui_renderer::{attrib, uniform};

//////////////////////////////////////
// Uniforms and vertices

uniform! {
    pub struct Uniforms3D {
        pub lights: [Vector3; 2],
        pub m: Matrix4,
        pub mnormal: Matrix3,
        pub tex: i32,
        pub line_top: i32,
        pub texturize: i32,
    }
    pub struct Uniforms2D {
        pub m: Matrix3,
        pub tex: i32,
        pub frac_dash: f32,
        pub line_color: Rgba,
        pub texturize: i32,
        pub notex_color: Rgba,
    }
    pub struct UniformQuad {
        pub color: Rgba,
    }
}

attrib! {
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MVertex3D {
        pub pos: Vector3,
        pub normal: Vector3,
        pub uv: Vector2,
        pub mat: MaterialIndex,
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
        pub mat: MaterialIndex,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MVertex2DColor {
        pub pos: Vector2,
        pub uv: Vector2,
        pub mat: MaterialIndex,
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
    pub struct MStatus3D {
        pub color: Rgba,
        pub top: u8,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MStatus2D {
        pub color: Rgba,
    }
    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    pub struct MVertexText {
        pub pos: Vector2,
        pub uv: Vector2,
    }
}

pub const MSTATUS_UNSEL: MStatus3D = MStatus3D {
    color: Rgba::new(0.0, 0.0, 0.0, 0.0),
    top: 0,
};
pub const MSTATUS_SEL: MStatus3D = MStatus3D {
    color: Rgba::new(0.0, 0.0, 1.0, 0.5),
    top: 1,
};
pub const MSTATUS_HI: MStatus3D = MStatus3D {
    color: Rgba::new(1.0, 0.0, 0.0, 0.75),
    top: 1,
};

pub fn program_from_source(gl: &glr::GlContext, shaders: &str) -> Result<glr::Program> {
    let split = shaders
        .find("###")
        .ok_or_else(|| anyhow!("shader marker not found"))?;
    let vertex = &shaders[..split];
    let frag = &shaders[split..];
    let split_2 = frag
        .find('\n')
        .ok_or_else(|| anyhow!("shader marker not valid"))?;

    let mut frag = &frag[split_2..];

    let geom = if let Some(split) = frag.find("###") {
        let geom = &frag[split..];
        frag = &frag[..split];
        let split_2 = geom
            .find('\n')
            .ok_or_else(|| anyhow!("shader marker not valid"))?;
        Some(&geom[split_2..])
    } else {
        None
    };

    let prg = glr::Program::from_source(gl, vertex, frag, geom)?;
    Ok(prg)
}

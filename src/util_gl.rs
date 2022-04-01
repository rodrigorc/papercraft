use crate::util_3d::*;
use crate::glr;
use gl::types::*;

/*
use gtk::gdk;
impl Texture2dDataSink<(u8, u8, u8, u8)> for GdkPixbufDataSink {
    fn from_raw(data: std::borrow::Cow<'_, [(u8, u8, u8, u8)]>, width: u32, height: u32) -> Self
        where [(u8, u8, u8, u8)]: ToOwned
    {
        let data: &[(u8, u8, u8, u8)] = data.as_ref();
        let data: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, 4 * data.len()) };

        let pb = gdk_pixbuf::Pixbuf::new(gdk_pixbuf::Colorspace::Rgb, true, 8, width as i32, height as i32).unwrap();
        let stride = pb.rowstride() as usize;
        let byte_width = 4 * width as usize;

        unsafe {
            let pix = pb.pixels();
            let dsts = pix.chunks_mut(stride);
            let srcs = data.chunks(byte_width).rev();
            for (dst, src) in dsts.zip(srcs) {
                dst[.. byte_width].copy_from_slice(src);
            }
        }
        GdkPixbufDataSink(pb)
    }
}*/

//////////////////////////////////////
/// Uniforms and vertices

pub struct Uniforms3D {
    pub m: Matrix4,
    pub mnormal: Matrix3,
    pub lights: [Vector3; 2],
    pub texture: i32,
}

impl glr::UniformProvider for Uniforms3D {
    fn apply(&self, u: &glr::Uniform) {
        match u.name() {
            "m" => {
                unsafe {
                    gl::UniformMatrix4fv(u.location(), 1, gl::FALSE, &self.m[0][0]);
                }
            }
            "mnormal" => {
                unsafe {
                    gl::UniformMatrix3fv(u.location(), 1, gl::FALSE, &self.mnormal[0][0]);
                }
            }
            "lights[0]" => {
                unsafe {
                    gl::Uniform3fv(u.location(), 2, &self.lights[0][0]);
                }
            }
            "tex" => {
                unsafe {
                    gl::Uniform1i(u.location(), self.texture);
                }
            }
            _ => {}
        }
    }
}

pub struct Uniforms2D {
    pub m: Matrix3,
    pub texture: i32,
    pub frac_dash: f32,
}

impl glr::UniformProvider for Uniforms2D {
    fn apply(&self, u: &glr::Uniform) {
        match u.name() {
            "m" => {
                unsafe {
                    gl::UniformMatrix3fv(u.location(), 1, gl::FALSE, &self.m[0][0]);
                }
            }
            "frac_dash" => {
                unsafe {
                    gl::Uniform1f(u.location(), self.frac_dash);
                }
            }
            "tex" => {
                unsafe {
                    gl::Uniform1i(u.location(), 0); //TODO
                }
            }
            _ => {}
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MVertex3D {
    pub pos: Vector3,
    pub normal: Vector3,
    pub uv: Vector2,
}

impl glr::AttribProvider for MVertex3D {
    fn apply(a: &glr::Attribute) -> Option<(usize, GLenum, usize)> {
        match a.name() {
            "pos" => {
                Some((3, gl::FLOAT,  0))
            }
            "normal" => {
                Some((3, gl::FLOAT, 4 * 3))
            }
            "uv" => {
                Some((2, gl::FLOAT, 4 * 3 + 4 * 3))
            }
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MVertex3DLine {
    pub pos: Vector3,
    pub color: [f32; 4],
    pub top: u8,
}

impl glr::AttribProvider for MVertex3DLine {
    fn apply(a: &glr::Attribute) -> Option<(usize, GLenum, usize)> {
        match a.name() {
            "pos" => {
                Some((3, gl::FLOAT,  0))
            }
            "color" => {
                Some((4, gl::FLOAT,  3 * 4))
            }
            "top" => {
                Some((1, gl::BYTE,  3 * 4 + 4 * 4))
            }
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MVertex2D {
    pub pos: Vector2,
    pub uv: Vector2,
    pub color: [f32; 4],
}

impl glr::AttribProvider for MVertex2D {
    fn apply(a: &glr::Attribute) -> Option<(usize, GLenum, usize)> {
        match a.name() {
            "pos" => {
                Some((3, gl::FLOAT,  0))
            }
            "uv" => {
                Some((2, gl::FLOAT,  2 * 4))
            }
            "color" => {
                Some((4, gl::FLOAT,  4 * 4))
            }
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MVertexQuad {
    pub pos: [f32; 2],
}

impl glr::AttribProvider for MVertexQuad {
    fn apply(a: &glr::Attribute) -> Option<(usize, GLenum, usize)> {
        match a.name() {
            "pos" => {
                Some((2, gl::FLOAT,  0))
            }
            _ => None,
        }
    }
}


#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MStatus {
    pub status: [f32; 4],
}

pub const MSTATUS_UNSEL: MStatus = MStatus { status: [0.0, 0.0, 0.0, 0.0]};
pub const MSTATUS_SEL: MStatus = MStatus { status: [0.0, 0.0, 1.0, 0.5]};
pub const MSTATUS_HI: MStatus = MStatus { status: [1.0, 0.0, 0.0, 0.75]};

impl glr::AttribProvider for MStatus {
    fn apply(a: &glr::Attribute) -> Option<(usize, GLenum, usize)> {
        match a.name() {
            "status" => {
                Some((4, gl::FLOAT,  0))
            }
            _ => None,
        }
    }
}

pub fn program_from_source(shaders: &str) -> glr::Program {
    let split = shaders.find("###").unwrap();
    let vertex = &shaders[.. split];
    let frag = &shaders[split ..];
    let split_2 = frag.find('\n').unwrap();
    let frag = &frag[split_2 ..];

    glr::Program::from_source(vertex, frag).unwrap()
}
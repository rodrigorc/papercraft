use std::{rc::Rc, cell::Cell};
use cgmath::conv::{array4x4, array3x3, array3};
use glium::uniforms::AsUniformValue;
use gtk::gdk;
use crate::util_3d::*;

pub struct GdkGliumBackend {
    ctx: gdk::GLContext,
    size: Rc<Cell<(u32, u32)>>,
}

impl GdkGliumBackend {
    pub fn new(ctx: gdk::GLContext, size: Rc<Cell<(u32, u32)>>) -> GdkGliumBackend {
        GdkGliumBackend {
            ctx,
            size,
        }
    }
}

unsafe impl glium::backend::Backend for GdkGliumBackend {
    fn swap_buffers(&self) -> Result<(), glium::SwapBuffersError> {
        Ok(())
    }
    unsafe fn get_proc_address(&self, symbol: &str) -> *const core::ffi::c_void {
        gl_loader::get_proc_address(symbol) as _
    }
    fn get_framebuffer_dimensions(&self) -> (u32, u32) {
        //let w = self.ctx.window().unwrap();
        //(w.width() as u32, w.height() as u32)
        self.size.get()
    }
    fn is_current(&self) -> bool {
        gdk::GLContext::current().as_ref() == Some(&self.ctx)
    }
    unsafe fn make_current(&self) {
        self.ctx.make_current();
    }
}

pub struct PersistentVertexBuffer<V: glium::Vertex> {
    buffer: glium::VertexBuffer<V>,
    length: usize,
}

impl<V: glium::Vertex> PersistentVertexBuffer<V> {
    pub fn new(ctx: &impl glium::backend::Facade, initial_size: usize) -> PersistentVertexBuffer<V> {
        let buffer = glium::VertexBuffer::empty_persistent(ctx, initial_size).unwrap();
        PersistentVertexBuffer {
            buffer,
            length: 0,
        }
    }
    pub fn update(&mut self, data: &[V]) {
        if let Some(slice) = self.buffer.slice(0 .. data.len()) {
            self.length = data.len();
            if self.length > 0 {
                slice.write(data);
            }
        } else {
            // If the buffer is not big enough, remake it
            let ctx = self.buffer.get_context();
            self.buffer = glium::VertexBuffer::persistent(ctx, data).unwrap();
            self.length = data.len();
        }
    }
}

impl<'a, V: glium::Vertex> From<&'a PersistentVertexBuffer<V>> for glium::vertex::VerticesSource<'a> {
    fn from(buf: &'a PersistentVertexBuffer<V>) -> Self {
        buf.buffer.slice(0 .. buf.length).unwrap().into()
    }
}

pub struct PersistentIndexBuffer<V: glium::index::Index> {
    buffer: glium::IndexBuffer<V>,
    length: usize,
}

impl<V: glium::index::Index> PersistentIndexBuffer<V> {
    pub fn new(ctx: &impl glium::backend::Facade, prim: glium::index::PrimitiveType, initial_size: usize) -> PersistentIndexBuffer<V> {
        let buffer = glium::IndexBuffer::empty_persistent(ctx, prim, initial_size).unwrap();
        PersistentIndexBuffer {
            buffer,
            length: 0,
        }
    }
    pub fn update(&mut self, data: &[V]) {
        if let Some(slice) = self.buffer.slice(0 .. data.len()) {
            self.length = data.len();
            if self.length > 0 {
                slice.write(data);
            }
        } else {
            // If the buffer is not big enough, remake it
            let ctx = self.buffer.get_context();
            self.buffer = glium::IndexBuffer::persistent(ctx, self.buffer.get_primitives_type(), data).unwrap();
            self.length = data.len();
        }
    }
}

impl<'a, V: glium::index::Index> From<&'a PersistentIndexBuffer<V>> for glium::index::IndicesSource<'a> {
    fn from(buf: &'a PersistentIndexBuffer<V>) -> Self {
        buf.buffer.slice(0 .. buf.length).unwrap().into()
    }
}

pub struct GdkPixbufDataSink(pub gdk_pixbuf::Pixbuf);

impl glium::texture::Texture2dDataSink<(u8, u8, u8, u8)> for GdkPixbufDataSink {
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
}

//////////////////////////////////////
/// Uniforms and vertices

pub struct Uniforms3D<'a> {
    pub m: Matrix4,
    pub mnormal: Matrix3,
    pub lights: [Vector3; 2],
    pub texture: glium::uniforms::Sampler<'a, glium::Texture2d>,
    pub color: [f32; 4],
}

impl glium::uniforms::Uniforms for Uniforms3D<'_> {
    fn visit_values<'a, F: FnMut(&str, glium::uniforms::UniformValue<'a>)>(&'a self, mut visit: F) {
        use glium::uniforms::UniformValue::*;

        visit("m", Mat4(array4x4(self.m)));
        visit("mnormal", Mat3(array3x3(self.mnormal)));
        visit("lights[0]", Vec3(array3(self.lights[0])));
        visit("lights[1]", Vec3(array3(self.lights[1])));
        visit("tex", self.texture.as_uniform_value());
        visit("color", Vec4(self.color));
    }
}

pub struct Uniforms2D<'a> {
    pub m: Matrix3,
    pub texture: glium::uniforms::Sampler<'a, glium::Texture2d>,
    pub color: [f32; 4],
}

impl glium::uniforms::Uniforms for Uniforms2D<'_> {
    fn visit_values<'a, F: FnMut(&str, glium::uniforms::UniformValue<'a>)>(&'a self, mut visit: F) {
        use glium::uniforms::UniformValue::*;

        visit("m", Mat3(array3x3(self.m)));
        visit("tex", self.texture.as_uniform_value());
        visit("color", Vec4(self.color));
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MVertex3D {
    pub pos: Vector3,
    pub normal: Vector3,
    pub uv: Vector2,
}

impl glium::Vertex for MVertex3D {
    fn build_bindings() -> glium::VertexFormat {
        use std::borrow::Cow::Borrowed;
        Borrowed(
            &[
                (Borrowed("pos"), 0, glium::vertex::AttributeType::F32F32F32, false),
                (Borrowed("normal"), 4*3, glium::vertex::AttributeType::F32F32F32, false),
                (Borrowed("uv"), 4*3 + 4*3, glium::vertex::AttributeType::F32F32, false),
            ]
        )
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MVertex2D {
    pub pos: Vector2,
    pub uv: Vector2,
}

impl glium::Vertex for MVertex2D {
    fn build_bindings() -> glium::VertexFormat {
        use std::borrow::Cow::Borrowed;
        Borrowed(
            &[
                (Borrowed("pos"), 0, glium::vertex::AttributeType::F32F32, false),
                (Borrowed("uv"), 4*2, glium::vertex::AttributeType::F32F32, false),
            ]
        )
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MVertexQuad {
    pub pos: [f32; 2],
}

impl glium::Vertex for MVertexQuad {
    fn build_bindings() -> glium::VertexFormat {
        use std::borrow::Cow::Borrowed;
        Borrowed(
            &[
                (Borrowed("pos"), 0, glium::vertex::AttributeType::F32F32, false),
            ]
        )
    }
}

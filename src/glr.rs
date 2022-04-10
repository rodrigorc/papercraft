#![allow(dead_code)]

use std::{ffi::CString, cell::Cell, marker::PhantomData};

use gl::types::*;
use smallvec::SmallVec;

#[derive(Debug)]
pub struct Error;

pub type Result<T> = std::result::Result<T, Error>;

pub struct Texture {
    id: u32,
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.id);
        }
    }
}

impl Texture {
    pub fn generate() -> Result<Texture> {
        unsafe {
            let mut id = 0;
            gl::GenTextures(1, &mut id);
            Ok(Texture { id })
        }
    }
    pub fn id(&self) -> u32 {
        self.id
    }
}


pub struct EnablerVertexAttribArray(GLuint);

impl EnablerVertexAttribArray {
    fn enable(id: GLuint) -> EnablerVertexAttribArray {
        unsafe {
            gl::EnableVertexAttribArray(id);
        }
         EnablerVertexAttribArray(id)
    }
}

impl Drop for EnablerVertexAttribArray {
    fn drop(&mut self) {
        unsafe {
            gl::DisableVertexAttribArray(self.0);
        }
    }
}

pub struct PushViewport([i32; 4]);

impl PushViewport {
    pub fn push(x: i32, y: i32, width: i32, height: i32) -> PushViewport {
        unsafe {
            let mut prev = [0; 4];
            gl::GetIntegerv(gl::VIEWPORT, prev.as_mut_ptr());
            gl::Viewport(x, y, width, height);
            PushViewport(prev)
        }
    }
}
impl Drop for PushViewport {
    fn drop(&mut self) {
        unsafe {
            gl::Viewport(self.0[0], self.0[1], self.0[2], self.0[3]);
        }
    }
}

pub struct Program {
    id: u32,
    uniforms: Vec<Uniform>,
    attribs: Vec<Attribute>,
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.id);
        }
    }
}

impl Program {
    pub fn from_source(vertex: &str, fragment: &str, geometry: Option<&str>) -> Result<Program> {
        unsafe {
            let vsh = Shader::compile(gl::VERTEX_SHADER, vertex)?;
            let fsh = Shader::compile(gl::FRAGMENT_SHADER, fragment)?;
            let gsh = match geometry {
                Some(source) => Some(Shader::compile(gl::GEOMETRY_SHADER, source)?),
                None => None,
            };
            let id = gl::CreateProgram();
            if id == 0 {
                return Err(Error);
            }
            let mut prg = Program {
                id,
                uniforms: Vec::new(),
                attribs: Vec::new(),
            };
            gl::AttachShader(prg.id, vsh.id);
            gl::AttachShader(prg.id, fsh.id);
            if let Some(id) = gsh {
                gl::AttachShader(prg.id, id.id);
            }
            gl::LinkProgram(prg.id);

            let mut st = 0;
            gl::GetProgramiv(prg.id, gl::LINK_STATUS, &mut st);
            if st == gl::FALSE as  GLint {
                let mut buf = [0; 1024];
                let mut len_msg = 0;
                gl::GetProgramInfoLog(prg.id, buf.len() as _, &mut len_msg, buf.as_mut_ptr() as *mut i8);
                let bmsg = &buf[0..len_msg as usize];
                let msg = String::from_utf8_lossy(bmsg);
                eprintln!("{msg}");
                return Err(Error);
            }

            let mut nu = 0;
            gl::GetProgramiv(prg.id, gl::ACTIVE_UNIFORMS, &mut nu);
            prg.uniforms = Vec::with_capacity(nu as usize);
            for u in 0..nu {
                let mut name = [0; 64];
                let mut len_name = 0;
                let mut size = 0;
                let mut type_ = 0;
                gl::GetActiveUniform(prg.id, u as u32, name.len() as i32, &mut len_name, &mut size, &mut type_, name.as_mut_ptr() as *mut i8);
                if len_name == 0 {
                    continue;
                }
                let name = CString::new(&name[0..len_name as usize]).unwrap();
                let location = gl::GetUniformLocation(prg.id, name.as_ptr());
                let name = name.into_string().unwrap();

                let u = Uniform {
                    name,
                    location,
                    _size: size,
                    _type: type_,
                };
                prg.uniforms.push(u);
            }
            let mut na = 0;
            gl::GetProgramiv(prg.id, gl::ACTIVE_ATTRIBUTES, &mut na);
            prg.attribs = Vec::with_capacity(na as usize);
            for a in 0..na {
                let mut name = [0; 64];
                let mut len_name = 0;
                let mut size = 0;
                let mut type_ = 0;
                gl::GetActiveAttrib(prg.id, a as u32, name.len() as i32, &mut len_name, &mut size, &mut type_, name.as_mut_ptr() as *mut i8);
                if len_name == 0 {
                    continue;
                }
                let name = CString::new(&name[0..len_name as usize]).unwrap();
                let location = gl::GetAttribLocation(prg.id, name.as_ptr());
                let name = name.into_string().unwrap();

                let a = Attribute {
                    name,
                    location,
                    _size: size,
                    _type: type_,
                };
                prg.attribs.push(a);

            }

            Ok(prg)
        }
    }

    pub fn attrib_by_name(&self, name: &str) -> Option<&Attribute> {
        self.attribs.iter().find(|a| a.name == name)
    }
    pub fn draw<U, AS>(&self, uniforms: &U, attribs: AS, primitive: GLenum)
        where
            U: UniformProvider,
            AS: AttribProviderList,
    {
        unsafe {
            gl::UseProgram(self.id);

            for u in &self.uniforms {
                uniforms.apply(u);
            }

            let _bufs = attribs.bind(self);
            gl::DrawArrays(primitive, 0, attribs.len() as i32);
            //dbg!(gl::GetError());
        }
    }
}

struct Shader {
    id: u32,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.id);
        }
    }
}

impl Shader {
    fn compile(ty: GLenum, source: &str) -> Result<Shader> {
        unsafe {
            let id = gl::CreateShader(ty);
            if id == 0 {
                dbg!(gl::GetError());
                return Err(Error);
            }
            let sh = Shader{id};
            let mut lines = Vec::new();
            let mut lens = Vec::new();

            for line in source.split_inclusive('\n') {
                lines.push(line.as_ptr() as *const i8);
                lens.push(line.len() as i32);
            }
            gl::ShaderSource(sh.id, lines.len() as i32, lines.as_ptr(), lens.as_ptr());
            gl::CompileShader(sh.id);
            let mut st = 0;
            gl::GetShaderiv(sh.id, gl::COMPILE_STATUS, &mut st);
            if st == gl::FALSE as GLint {
                //TODO: get errors
                let mut buf = [0u8; 1024];
                let mut len_msg = 0;
                gl::GetShaderInfoLog(sh.id, buf.len() as _, &mut len_msg, buf.as_mut_ptr() as *mut i8);
                let bmsg = &buf[0..len_msg as usize];
                let msg = String::from_utf8_lossy(bmsg);
                eprintln!("{msg}");
                return Err(Error);
            }
            Ok(sh)
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Rgba {
        Rgba { r, g, b, a }
    }
}

#[derive(Debug)]
pub struct Uniform {
    name: String,
    location: GLint,
    _size: GLint,
    _type: GLenum,
}

impl Uniform {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn location(&self) -> GLint {
        self.location
    }
}

#[derive(Debug)]
pub struct Attribute {
    name: String,
    location: GLint,
    _size: GLint,
    _type: GLenum,
}

impl Attribute {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn location(&self) -> GLint {
        self.location
    }
}

pub trait UniformProvider {
    fn apply(&self, u: &Uniform);
}

/// # Safety
///
/// This trait returns offsets from Self that will be used to index the raw memory of a
/// VertexAttribBuffer. Better implemented using the `attrib!` macro.
pub unsafe trait AttribProvider: Copy {
    fn apply(a: &Attribute) -> Option<(usize, GLenum, usize)>;
}

pub trait AttribProviderList {
    type KeepType;
    fn len(&self) -> usize;
    fn bind(&self, p: &Program) -> Self::KeepType;
}

// This is quite inefficient, but easy to use
#[cfg(xxx)]
impl<A: AttribProvider> AttribProviderList for &[A] {
    type KeepType = (Buffer, SmallVec<[EnablerVertexAttribArray; 8]>);

    fn len(&self) -> usize {
        <[A]>::len(self)
    }
    fn bind(&self, p: &Program) -> (Buffer, SmallVec<[EnablerVertexAttribArray; 8]>) {
        let buf = Buffer::generate().unwrap();
        let mut vas = SmallVec::new();
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, buf.id());
            gl::BufferData(gl::ARRAY_BUFFER, (std::mem::size_of::<A>() * self.len()) as isize, self.as_ptr() as *const A as *const _, gl::STATIC_DRAW);
            for a in &p.attribs {
                if let Some((size, ty, offs)) = A::apply(a) {
                    let loc = a.location() as u32;
                    vas.push(EnablerVertexAttribArray::enable(loc));
                    gl::VertexAttribPointer(loc, size as i32, ty, gl::FALSE, std::mem::size_of::<A>() as i32, offs as *const _);
                }
            }
        }
        (buf, vas)
    }
}

/// # Safety
///
/// Returned information will be used to index the raw memory of a VertexAttribBuffer. Returning
/// wrong information will cause seg faults.
pub unsafe trait AttribField {
    fn detail() -> (usize, GLenum);
}

unsafe impl AttribField for f32 {
    fn detail() -> (usize, GLenum) {
        (1, gl::FLOAT)
    }
}
unsafe impl AttribField for u8 {
    fn detail() -> (usize, GLenum) {
        (1, gl::BYTE)
    }
}
unsafe impl AttribField for u32 {
    fn detail() -> (usize, GLenum) {
        (1, gl::UNSIGNED_INT)
    }
}
unsafe impl AttribField for i32 {
    fn detail() -> (usize, GLenum) {
        (1, gl::INT)
    }
}
unsafe impl AttribField for Rgba {
    fn detail() -> (usize, GLenum) {
        (4, gl::FLOAT)
    }
}
unsafe impl<F: AttribField, const N: usize> AttribField for [F; N] {
    fn detail() -> (usize, GLenum) {
        let (d, t) = F::detail();
        (N * d, t)
    }
}
unsafe impl<F: AttribField> AttribField for cgmath::Vector2<F> {
    fn detail() -> (usize, GLenum) {
        let (d, t) = F::detail();
        (2 * d, t)
    }
}
unsafe impl<F: AttribField> AttribField for cgmath::Vector3<F> {
    fn detail() -> (usize, GLenum) {
        let (d, t) = F::detail();
        (3 * d, t)
    }
}

#[macro_export]
macro_rules! attrib {
    (
        $(
            $(#[$a:meta])* $v:vis struct $name:ident {
                $(
                    $fv:vis $f:ident : $ft:ty
                ),*
                $(,)?
            }
        )*
    ) => {
        $(
            $(#[$a])* $v struct $name {
                $(
                    $fv $f: $ft ,
                )*
            }
            unsafe impl crate::glr::AttribProvider for $name {
                fn apply(a: &crate::glr::Attribute) -> Option<(usize, gl::types::GLenum, usize)> {
                    let name = a.name();
                    $(
                        if name == stringify!($f) {
                            let (n, t) = <$ft as crate::glr::AttribField>::detail();
                            return Some((n, t, memoffset::offset_of!($name, $f)));
                        }
                    )*
                    None
                }
            }
        )*
    }
}

pub unsafe trait UniformField {
    fn apply(&self, count: i32, location: GLint);
}

unsafe impl UniformField for cgmath::Matrix4<f32> {
    fn apply(&self, count: i32, location: GLint) {
        unsafe {
            gl::UniformMatrix4fv(location, count, gl::FALSE, &self[0][0]);
        }
    }
}

unsafe impl UniformField for cgmath::Matrix3<f32> {
    fn apply(&self, count: i32, location: GLint) {
        unsafe {
            gl::UniformMatrix3fv(location, count, gl::FALSE, &self[0][0]);
        }
    }
}

unsafe impl UniformField for cgmath::Vector3<f32> {
    fn apply(&self, count: i32, location: GLint) {
        unsafe {
            gl::Uniform3fv(location, count, &self[0]);
        }
    }
}

unsafe impl UniformField for i32 {
    fn apply(&self, count: i32, location: GLint) {
        unsafe {
            gl::Uniform1iv(location, count, self);
        }
    }
}

unsafe impl UniformField for f32 {
    fn apply(&self, count: i32, location: GLint) {
        unsafe {
            gl::Uniform1fv(location, count, self);
        }
    }
}

unsafe impl UniformField for Rgba {
    fn apply(&self, count: i32, location: GLint) {
        unsafe {
            gl::Uniform4fv(location, count, &self.r);
        }
    }
}

unsafe impl<T: UniformField, const N: usize> UniformField for [T; N] {
    fn apply(&self, count: i32, location: GLint) {
        T::apply(&self[0], count * N as i32, location);
    }
}


#[macro_export]
macro_rules! uniform {
    (
        $(
            $(#[$a:meta])* $v:vis struct $name:ident {
                $(
                    $fv:vis $f:ident : $ft:tt
                ),*
                $(,)?
            }
        )*
    ) => {
        $(
            $(#[$a])* $v struct $name {
                $(
                    $fv $f: $ft ,
                )*
            }
            impl crate::glr::UniformProvider for $name {
                fn apply(&self, u: &crate::glr::Uniform) {
                    let name = u.name();
                    $(
                        if name == crate::uniform!{ @NAME $f: $ft }  {
                            <$ft as crate::glr::UniformField>::apply(&self.$f, 1, u.location());
                            return;
                        }
                    )*
                    dbg!(name);
                }
            }
        )*
    };
    (@NAME $f:ident : [ $ft:ty; $n:literal ]) => { concat!(stringify!($f), "[0]") };
    (@NAME $f:ident : $ft:ty) => { stringify!($f) };
}

impl<A0: AttribProviderList, A1: AttribProviderList> AttribProviderList for (A0, A1) {
    type KeepType = (A0::KeepType, A1::KeepType);
    fn len(&self) -> usize {
        self.0.len().min(self.1.len())
    }
    fn bind(&self, p: &Program) -> (A0::KeepType, A1::KeepType) {
        let k0 = self.0.bind(p);
        let k1 = self.1.bind(p);
        (k0, k1)
    }
}

pub struct DynamicVertexArray<A> {
    data: Vec<A>,
    buf: Buffer,
    buf_len: Cell<usize>,
    dirty: Cell<bool>,
}

impl<A: AttribProvider> DynamicVertexArray<A> {
    pub fn new() -> Self {
        Self::from(Vec::new())
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn set(&mut self, data: impl Into<Vec<A>>) {
        self.dirty.set(true);
        self.data = data.into();
    }
    pub fn data(&self) -> &[A] {
        &self.data[..]
    }
    pub fn sub(&self, range: std::ops::Range<usize>) -> DynamicVertexArraySub<'_, A> {
        DynamicVertexArraySub {
            array: self,
            range,
        }
    }
    pub fn bind_buffer(&self) {
        if self.data.is_empty() {
            return;
        }
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.buf.id());
            if self.dirty.get() {
                if self.data.len() > self.buf_len.get() {
                    gl::BufferData(gl::ARRAY_BUFFER,
                        (std::mem::size_of::<A>() * self.data.len()) as isize,
                        self.data.as_ptr() as *const A as *const _,
                        gl::DYNAMIC_DRAW
                    );
                    self.buf_len.set(self.data.len());
                } else {
                    gl::BufferSubData(gl::ARRAY_BUFFER,
                        0,
                        (std::mem::size_of::<A>() * self.data.len()) as isize,
                        self.data.as_ptr() as *const A as *const _
                    );
                }
                self.dirty.set(false);
            }
        }
    }
}

impl<A: AttribProvider > From<Vec<A>> for DynamicVertexArray<A> {
    fn from(data: Vec<A>) -> Self {
        DynamicVertexArray {
            data,
            buf: Buffer::generate().unwrap(),
            buf_len: Cell::new(0),
            dirty: Cell::new(true),
        }
    }
}

impl<A: AttribProvider> std::ops::Index<usize> for DynamicVertexArray<A> {
    type Output = A;

    fn index(&self, index: usize) -> &A {
        &self.data[index]
    }
}

impl<A: AttribProvider> std::ops::IndexMut<usize> for DynamicVertexArray<A> {
    fn index_mut(&mut self, index: usize) -> &mut A {
        self.dirty.set(true);
        &mut self.data[index]
    }
}

impl<A: AttribProvider> AttribProviderList for &DynamicVertexArray<A> {
    type KeepType = SmallVec<[EnablerVertexAttribArray; 8]>;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn bind(&self, p: &Program) -> SmallVec<[EnablerVertexAttribArray; 8]> {
        let mut vas = SmallVec::new();
        unsafe {
            self.bind_buffer();
            for a in &p.attribs {
                if let Some((size, ty, offs)) = A::apply(a) {
                    let loc = a.location() as u32;
                    vas.push(EnablerVertexAttribArray::enable(loc));
                    gl::VertexAttribPointer(loc, size as i32, ty, gl::FALSE, std::mem::size_of::<A>() as i32, offs as *const _);
                }
            }
        }
        vas
    }
}

pub struct DynamicVertexArraySub<'a, A> {
    array: &'a DynamicVertexArray<A>,
    range: std::ops::Range<usize>,
}

impl<A: AttribProvider> AttribProviderList for DynamicVertexArraySub<'_, A> {
    type KeepType = SmallVec<[EnablerVertexAttribArray; 8]>;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn bind(&self, p: &Program) -> Self::KeepType {
        let mut vas = SmallVec::new();
        unsafe {
            self.array.bind_buffer();
            for a in &p.attribs {
                if let Some((size, ty, offs)) = A::apply(a) {
                    let loc = a.location() as u32;
                    vas.push(EnablerVertexAttribArray::enable(loc));
                    let offs = offs + std::mem::size_of::<A>() * self.range.start;
                    gl::VertexAttribPointer(loc, size as i32, ty, gl::FALSE, std::mem::size_of::<A>() as i32, offs as *const _);
                }
            }
        }
        vas

    }
}

pub struct Buffer {
    id: u32,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.id);
        }
    }
}

impl Buffer {
    pub fn generate() -> Result<Buffer> {
        unsafe {
            let mut id = 0;
            gl::GenBuffers(1, &mut id);
            Ok(Buffer { id })
        }
    }
    pub fn id(&self) -> u32 {
        self.id
    }
}

pub struct VertexArray {
    id: u32,
}

impl Drop for VertexArray {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, &self.id);
        }
    }
}

impl VertexArray {
    pub fn generate() -> Result<VertexArray> {
        unsafe {
            let mut id = 0;
            gl::GenVertexArrays(1, &mut id);
            Ok(VertexArray { id })
        }
    }
    pub fn id(&self) -> u32 {
        self.id
    }
}

pub struct Renderbuffer {
    id: u32,
}

impl Drop for Renderbuffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteRenderbuffers(1, &self.id);
        }
    }
}

impl Renderbuffer {
    pub fn generate() -> Result<Renderbuffer> {
        unsafe {
            let mut id = 0;
            gl::GenRenderbuffers(1, &mut id);
            Ok(Renderbuffer { id })
        }
    }
    pub fn id(&self) -> u32 {
        self.id
    }
}

pub struct BinderRenderbuffer(());

impl BinderRenderbuffer {
    pub fn bind(rb: &Renderbuffer) -> BinderRenderbuffer {
        unsafe {
            gl::BindRenderbuffer(gl::RENDERBUFFER, rb.id);
        }
        BinderRenderbuffer(())
    }
    pub fn target(&self) -> GLenum {
        gl::RENDERBUFFER
    }
    pub fn rebind(&self, rb: &Renderbuffer) {
        unsafe {
            gl::BindRenderbuffer(gl::RENDERBUFFER, rb.id);
        }
    }
}
impl Drop for BinderRenderbuffer {
    fn drop(&mut self) {
        unsafe {
            gl::BindRenderbuffer(gl::RENDERBUFFER, 0);
        }
    }
}


pub struct Framebuffer {
    id: u32,
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteFramebuffers(1, &self.id);
        }
    }
}

impl Framebuffer {
    pub fn generate() -> Result<Framebuffer> {
        unsafe {
            let mut id = 0;
            gl::GenFramebuffers(1, &mut id);
            Ok(Framebuffer { id })
        }
    }
    #[allow(unused)]
    pub fn id(&self) -> u32 {
        self.id
    }
}


pub trait BinderFBOTarget {
    const TARGET: GLenum;
    const GET_BINDING: GLenum;
}

pub struct BinderFramebuffer<TGT: BinderFBOTarget>(u32, PhantomData<TGT>);

impl<TGT: BinderFBOTarget> BinderFramebuffer<TGT> {
    pub fn new() -> Self {
        let mut id = 0;
        unsafe {
            gl::GetIntegerv(TGT::GET_BINDING, &mut id);
        }
        BinderFramebuffer(id as u32, PhantomData)
    }
    pub fn target(&self) -> GLenum {
        TGT::TARGET
    }
    pub fn bind(rb: &Framebuffer) -> Self {
        unsafe {
            gl::BindFramebuffer(TGT::TARGET, rb.id);
        }
        BinderFramebuffer(0, PhantomData)
    }
    pub fn rebind(&self, rb: &Framebuffer) {
        unsafe {
            gl::BindFramebuffer(TGT::TARGET, rb.id);
        }
    }
}

impl<TGT: BinderFBOTarget> Drop for BinderFramebuffer<TGT> {
    fn drop(&mut self) {
        unsafe {
            gl::BindFramebuffer(TGT::TARGET, self.0);
        }
    }
}

pub struct BinderFBODraw;

impl BinderFBOTarget for BinderFBODraw {
    const TARGET: GLenum = gl::DRAW_FRAMEBUFFER;
    const GET_BINDING: GLenum = gl::DRAW_FRAMEBUFFER_BINDING;
}

pub type BinderDrawFramebuffer = BinderFramebuffer<BinderFBODraw>;

pub struct BinderFBORead;

impl BinderFBOTarget for BinderFBORead {
    const TARGET: GLenum = gl::READ_FRAMEBUFFER;
    const GET_BINDING: GLenum = gl::READ_FRAMEBUFFER_BINDING;
}

pub type BinderReadFramebuffer = BinderFramebuffer<BinderFBORead>;

pub fn available_multisamples(target: GLenum, internalformat: GLenum) -> Vec<GLint> {
    unsafe {
        let mut num = 0;
        gl::GetInternalformativ(target, internalformat, gl::NUM_SAMPLE_COUNTS, 1, &mut num);
        let mut samples = vec![0; num as usize];
        gl::GetInternalformativ(target, internalformat, gl::SAMPLES, samples.len() as i32, samples.as_mut_ptr());
        samples
    }
}

pub fn try_renderbuffer_storage_multisample(target: GLenum, internalformat: GLenum, width: i32, height: i32) -> Option<GLint> {
    let all_samples = available_multisamples(target, internalformat);
    unsafe {
        for samples in all_samples {
            // purge the gl error
            gl::GetError();
            gl::RenderbufferStorageMultisample(target, samples, internalformat, width, height);
            if gl::GetError() == 0 {
                eprintln!("Multisamples = {samples}");
                return Some(samples);
            }
            eprintln!("Multisamples {samples} failed");
        }
    }
    None
}
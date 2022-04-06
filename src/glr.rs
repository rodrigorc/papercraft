use std::{ffi::CString, cell::Cell};

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
    pub fn from_source(vertex: &str, fragment: &str) -> Result<Program> {
        unsafe {
            let vsh = Shader::compile(gl::VERTEX_SHADER, vertex)?;
            let fsh = Shader::compile(gl::FRAGMENT_SHADER, fragment)?;

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
unsafe impl AttribField for crate::Rgba {
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
                fn apply(a: &glr::Attribute) -> Option<(usize, gl::types::GLenum, usize)> {
                    let name = a.name();
                    $(
                        if name == stringify!($f) {
                            let (n, t) = <$ft as glr::AttribField>::detail();
                            return Some((n, t, memoffset::offset_of!($name, $f)));
                        }
                    )*
                    None
                }
            }
        )*
    }
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
    dirty: Cell<bool>,
}

impl<A: AttribProvider> DynamicVertexArray<A> {
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.buf.id());
            if self.dirty.get() {
                gl::BufferData(gl::ARRAY_BUFFER, (std::mem::size_of::<A>() * self.len()) as isize, self.data.as_ptr() as *const A as *const _, gl::DYNAMIC_DRAW);
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


use crate::{
    paper::{Matrix4, Vector2, Vector3},
    util_3d::{Matrix3, Point3},
};

use anyhow::{Result, anyhow, bail};
use cgmath::{EuclideanSpace, InnerSpace, Matrix, SquareMatrix, Zero};
use easy_imgui_window::easy_imgui_renderer::glow;
use image::{DynamicImage, ImageReader};
use std::{
    borrow::Cow,
    fs, io,
    path::{Path, PathBuf},
};

fn get_u32(data: &[u8], offs: usize) -> Option<u32> {
    let bs = data.get(offs..offs + 4)?;
    Some(u32::from_le_bytes(bs.try_into().unwrap()))
}

pub struct Gltf<'a> {
    dir: PathBuf,
    header: Header<'a>,
    buffers: Vec<Cow<'a, [u8]>>,
}

impl<'a> Gltf<'a> {
    pub fn parse(data: &'a [u8], file_name: &Path) -> Result<Gltf<'a>> {
        let dir = file_name.parent().map(PathBuf::from).unwrap_or_default();
        if data.get(0..4) == Some(b"glTF") {
            Self::parse_glb(data, dir)
        } else {
            Self::parse_gltf(data, dir)
        }
    }

    fn parse_gltf(data: &'a [u8], dir: PathBuf) -> Result<Gltf<'a>> {
        let header: Header = serde_json::from_slice(data)?;
        let gltf = Gltf::new(header, dir, None)?;
        Ok(gltf)
    }

    fn parse_glb(data: &'a [u8], dir: PathBuf) -> Result<Gltf<'a>> {
        let version = get_u32(data, 4).ok_or(anyhow!("read error"))?;
        if version != 2 {
            bail!("only glTF version 2 is supported");
        }
        let size = get_u32(data, 8).ok_or(anyhow!("read error"))? as usize;
        if size > data.len() {
            bail!("glTF truncated");
        }
        let mut data = &data[12..size];
        let mut header = None;
        let mut bin_buffer = None;
        while !data.is_empty() {
            let sz = get_u32(data, 0).ok_or(anyhow!("read error"))? as usize;
            let ty = data.get(4..8).ok_or(anyhow!("read error"))?;
            let bs = data.get(8..8 + sz).ok_or(anyhow!("read error"))?;
            data = &data[8 + sz..];
            match ty {
                b"JSON" => {
                    let hdr: Header = serde_json::from_slice(bs)?;
                    if header.is_some() {
                        bail!("double glTF header");
                    }
                    header = Some(hdr);
                }
                b"BIN\0" => {
                    if bin_buffer.is_some() {
                        bail!("double glTF binary");
                    }
                    bin_buffer = Some(bs);
                }
                _ => {
                    log::warn!("unknown tag {}", String::from_utf8_lossy(ty));
                }
            }
        }
        let header = header.ok_or(anyhow!("missing glTF header"))?;

        let gltf = Gltf::new(header, dir, bin_buffer)?;
        Ok(gltf)
    }

    fn new(header: Header<'a>, dir: PathBuf, bin_buffer: Option<&'a [u8]>) -> Result<Gltf<'a>> {
        let version = header
            .asset
            .min_version
            .or(header.asset.version)
            .ok_or(anyhow!("missing version in glTF header"))?;
        log::info!("glTF version: {version}");
        let i_ver = version.split('.').next().unwrap().parse::<u32>()?;
        if i_ver != 2 {
            bail!("Unknown glTF major version {}", version);
        }
        if let Some(copyright) = header.asset.copyright {
            log::info!("glTF copyright: {copyright}");
        }
        if let Some(generator) = header.asset.generator {
            log::info!("glTF generator: {generator}");
        }
        //log::debug!("{header:?}");
        let mut gltf = Gltf {
            dir,
            header,
            buffers: Vec::new(),
        };
        gltf.load_buffers(bin_buffer)?;
        Ok(gltf)
    }
    fn load_uri(&self, uri: &str) -> Result<Vec<u8>> {
        if let Some(data) = uri.strip_prefix("data:") {
            return decode_data_uri(data);
        }

        let file_name = uri_decode(uri)?;
        let bs = fs::read(self.dir.join(file_name))?;
        Ok(bs)
    }

    fn load_buffers(&mut self, mut bin_buffer: Option<&'a [u8]>) -> Result<()> {
        for buf in &self.header.buffers {
            if let Some(uri) = &buf.uri {
                let mut bs = self.load_uri(uri)?;
                bs.truncate(buf.byte_length);
                if bs.len() != buf.byte_length {
                    bail!("incorrect buffer length");
                }
                self.buffers.push(bs.into());
            } else {
                let Some(buf2) = bin_buffer.take() else {
                    bail!("missing gltf BIN");
                };
                let buf2 = buf2
                    .get(0..buf.byte_length)
                    .ok_or(anyhow!("binary buffer too short"))?;
                self.buffers.push(buf2.into());
            }
        }
        Ok(())
    }

    pub fn load_images(&self) -> Result<Vec<(String, DynamicImage)>> {
        let mut images = Vec::new();
        for (i, img) in self.header.images.iter().enumerate() {
            let data;
            let bs = match img.binary {
                Binary::BufferView(bv) => {
                    let (bs, _) = self
                        .buffer_view(bv)
                        .ok_or(anyhow!("missing bufferView {bv}"))?;
                    bs
                }
                Binary::Uri(ref uri) => {
                    data = self.load_uri(uri)?;
                    &data[..]
                }
            };
            let bs = io::Cursor::new(bs);
            let mut rdr = ImageReader::new(bs);
            match img.mime_type {
                Some("image/png") => {
                    rdr.set_format(image::ImageFormat::Png);
                }
                Some("image/jpeg") => {
                    rdr.set_format(image::ImageFormat::Jpeg);
                }
                _ => {
                    rdr = rdr.with_guessed_format()?;
                }
            }
            let pixbuf = rdr.decode()?;
            let name = img.name.map_or_else(|| format!("tex_{i}"), String::from);
            images.push((name, pixbuf));
        }
        Ok(images)
    }

    fn access<T: AccessorType>(&self, idx: usize) -> Option<Access<'_, T>> {
        let accessor = self.header.accessors.get(idx)?;
        if accessor.ty != T::TYPE {
            return None;
        }
        let (buffer, stride) = self.buffer_view(accessor.buffer_view)?;
        let buffer = buffer.get(accessor.byte_offset..)?;
        Some(Access {
            _pd: std::marker::PhantomData,
            component_type: accessor.component_type,
            count: accessor.count,
            buffer,
            stride,
        })
    }

    fn buffer_view(&self, idx: usize) -> Option<(&[u8], Option<usize>)> {
        let buffer_view = self.header.buffer_views.get(idx)?;
        let buffer = self.buffers.get(buffer_view.buffer)?;
        let stride = buffer_view.byte_stride;
        let bs = buffer
            .get(buffer_view.byte_offset..buffer_view.byte_offset + buffer_view.byte_length)?;
        Some((bs, stride))
    }

    pub fn process_scene<F>(&self, mut f_emit_face: F) -> Result<()>
    where
        F: FnMut(Option<usize>, [Vector3; 3], Option<[Vector3; 3]>, [Vector2; 3]),
    {
        let scene = self
            .header
            .scenes
            .get(self.header.scene)
            .ok_or(anyhow!("missing scene {}", self.header.scene))?;
        self.process_nodes(&scene.nodes, Matrix4::identity(), &mut f_emit_face)?;
        Ok(())
    }

    fn process_nodes<F>(
        &self,
        i_nodes: &[usize],
        mx_parent: Matrix4,
        f_emit_face: &mut F,
    ) -> Result<()>
    where
        F: FnMut(Option<usize>, [Vector3; 3], Option<[Vector3; 3]>, [Vector2; 3]),
    {
        for &i_node in i_nodes {
            let node = self
                .header
                .nodes
                .get(i_node)
                .ok_or(anyhow!("missing node {}", i_node))?;

            let mx = mx_parent * node.transform.matrix();
            let mx_normal = mx.invert().unwrap_or_else(Matrix4::identity).transpose();
            let mx_normal = crate::util_3d::Matrix3::from_cols(
                mx_normal.x.truncate(),
                mx_normal.y.truncate(),
                mx_normal.z.truncate(),
            );

            //let _rot = node.rotation;
            if let Some(i_mesh) = node.mesh {
                let mesh = self
                    .header
                    .meshes
                    .get(i_mesh)
                    .ok_or(anyhow!("missing mesh {}", i_mesh))?;
                for pri in &mesh.primitives {
                    let Some(a_pos) = pri.attributes.position else {
                        continue;
                    };
                    let mode = pri.mode.unwrap_or(4);
                    if mode != 4 {
                        log::warn!("unsupported primitive model {mode}");
                        continue;
                    }
                    let pos = self
                        .access::<Vector3>(a_pos)
                        .ok_or(anyhow!("missing accessor {a_pos}"))?;
                    let norm = pri
                        .attributes
                        .normal
                        .and_then(|a_normal| self.access::<Vector3>(a_normal));
                    let texcoord = pri
                        .attributes
                        .texcoord_0
                        .and_then(|a_texcoord| self.access::<Vector2>(a_texcoord));

                    let tex = pri.material.and_then(|i_material| {
                        let mat = self.header.materials.get(i_material)?;
                        let i_tex = mat
                            .pbr_metallic_roughness
                            .as_ref()?
                            .base_color_texture
                            .as_ref()?
                            .index;
                        let i_tex = self.header.textures.get(i_tex)?.source;
                        Some(i_tex)
                    });

                    let mut emit_face = |indices: [usize; 3]| -> Result<()> {
                        use cgmath::Transform;

                        let ps = indices.map(|index| {
                            let p = pos.get(index).unwrap_or(Vector3::zero());
                            mx.transform_point(cgmath::Point3::from_vec(p)).to_vec()
                        });
                        let ns = norm.as_ref().map(|norm| {
                            indices.map(|index| {
                                let n = norm.get(index).unwrap_or(Vector3::zero());
                                <Matrix3 as Transform<Point3>>::transform_vector(&mx_normal, n)
                                    .normalize()
                            })
                        });
                        let uv = texcoord.as_ref().map_or_else(
                            || [Vector2::zero(); 3],
                            |texcoord| {
                                indices.map(|index| texcoord.get(index).unwrap_or(Vector2::zero()))
                            },
                        );
                        f_emit_face(tex, ps, ns, uv);

                        Ok(())
                    };

                    if let Some(a_indices) = pri.indices {
                        let indices = self
                            .access::<u32>(a_indices)
                            .ok_or(anyhow!("missing accessor {a_indices}"))?;
                        for i in 0..indices.count / 3 {
                            let ii0 = 3 * i;
                            let i0 = indices.get(ii0).ok_or(anyhow!("missing index"))? as usize;
                            let i1 = indices.get(ii0 + 1).ok_or(anyhow!("missing index"))? as usize;
                            let i2 = indices.get(ii0 + 2).ok_or(anyhow!("missing index"))? as usize;
                            emit_face([i0, i1, i2])?;
                        }
                    } else {
                        for i in 0..pos.count / 3 {
                            let i0 = 3 * i;
                            emit_face([i0, i0 + 1, i0 + 2])?;
                        }
                    }
                }
            }
            self.process_nodes(&node.children, mx, f_emit_face)?;
        }
        Ok(())
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Header<'a> {
    #[serde(borrow)]
    asset: Asset<'a>,
    #[serde(default)]
    scene: usize,
    scenes: Vec<Scene>,
    nodes: Vec<Node>,
    meshes: Vec<Mesh>,
    buffers: Vec<Buffer>,
    buffer_views: Vec<BufferView>,
    accessors: Vec<Accessor<'a>>,
    #[serde(default)]
    images: Vec<Image<'a>>,
    #[serde(default)]
    materials: Vec<Material>,
    #[serde(default)]
    textures: Vec<Texture>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Asset<'a> {
    generator: Option<&'a str>,
    copyright: Option<&'a str>,
    version: Option<&'a str>,
    min_version: Option<&'a str>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Scene {
    #[serde(default)]
    nodes: Vec<usize>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Node {
    #[serde(default)]
    mesh: Option<usize>,
    #[serde(default)]
    children: Vec<usize>,
    #[serde(flatten)]
    transform: Transform,
}

#[derive(Debug, serde::Deserialize)]
enum Transform {
    #[serde(untagged)]
    Matrix { matrix: [f32; 16] },
    #[serde(untagged)]
    Trs {
        translation: Option<[f32; 3]>,
        rotation: Option<[f32; 4]>,
        scale: Option<[f32; 3]>,
    },
}
impl Transform {
    fn matrix(&self) -> Matrix4 {
        match self {
            Transform::Matrix { matrix } => *<&Matrix4>::from(matrix),
            Transform::Trs {
                translation,
                rotation,
                scale,
            } => {
                let mut m = Matrix4::identity();
                if let Some(t) = translation {
                    let t = Vector3::from(*t);
                    let t = Matrix4::from_translation(t);
                    // m is identity here, no need to multiply
                    m = t;
                }
                if let Some(r) = rotation {
                    let r = cgmath::Quaternion::from(*r);
                    let r = Matrix4::from(r);
                    m = m * r;
                }
                if let Some(s) = scale {
                    let s = Matrix4::from_nonuniform_scale(s[0], s[1], s[2]);
                    m = m * s;
                }
                m
            }
        }
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Mesh {
    #[serde(default)]
    primitives: Vec<Primitive>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Primitive {
    pub attributes: Attributes,
    pub indices: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<u32>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct Attributes {
    pub position: Option<usize>,
    pub normal: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texcoord_0: Option<usize>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Buffer {
    pub byte_length: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BufferView {
    pub buffer: usize,
    pub byte_length: usize,
    #[serde(default)]
    pub byte_offset: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_stride: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<u32>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Accessor<'a> {
    #[serde(rename = "type")]
    pub ty: &'a str,
    pub buffer_view: usize,
    #[serde(default)]
    pub byte_offset: usize,
    pub component_type: u32,
    pub count: usize,

    #[serde(default, flatten, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<BoundingBox>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct BoundingBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Image<'a> {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<&'a str>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<&'a str>,
    #[serde(flatten)]
    pub binary: Binary,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Binary {
    BufferView(usize),
    Uri(String),
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Texture {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<usize>,
    pub source: usize,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Material {
    pbr_metallic_roughness: Option<MetallicRoughness>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct MetallicRoughness {
    base_color_texture: Option<TextureRef>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct TextureRef {
    index: usize,
}

trait Scalar: Default + Copy {
    fn get(data: &[u8], component_type: u32, offs: usize) -> Option<Self>;
}

impl Scalar for u32 {
    fn get(data: &[u8], component_type: u32, idx: usize) -> Option<Self> {
        match component_type {
            glow::BYTE | glow::UNSIGNED_BYTE => {
                let x = *data.get(idx)?;
                Some(u32::from(x))
            }
            glow::SHORT | glow::UNSIGNED_SHORT => {
                let bs = data.get(2 * idx..2 * (idx + 1))?;
                let x = u16::from_le_bytes(bs.try_into().unwrap());
                Some(u32::from(x))
            }
            glow::UNSIGNED_INT => {
                let bs = data.get(4 * idx..4 * (idx + 1))?;
                let x = u32::from_le_bytes(bs.try_into().unwrap());
                Some(x)
            }
            _ => {
                log::warn!("Unknown component type 0x{component_type:X}");
                None
            }
        }
    }
}

impl Scalar for f32 {
    fn get(data: &[u8], component_type: u32, idx: usize) -> Option<Self> {
        #[allow(clippy::single_match_else)]
        match component_type {
            glow::FLOAT => {
                let bs = data.get(4 * idx..4 * (idx + 1))?;
                let x = u32::from_le_bytes(bs.try_into().unwrap());
                Some(f32::from_bits(x))
            }
            _ => {
                log::warn!("Unknown component type 0x{component_type:X}");
                None
            }
        }
    }
}

trait AccessorType {
    const TYPE: &'static str;
    const N: usize;
    type Inner: Scalar;
    fn new(d: &[Self::Inner]) -> Self;
}

impl AccessorType for u32 {
    const TYPE: &'static str = "SCALAR";
    const N: usize = 1;
    type Inner = u32;
    fn new(d: &[Self::Inner]) -> Self {
        d[0]
    }
}

impl AccessorType for Vector2 {
    const TYPE: &'static str = "VEC2";
    const N: usize = 2;
    type Inner = f32;
    fn new(d: &[Self::Inner]) -> Self {
        Vector2::new(d[0], d[1])
    }
}

impl AccessorType for Vector3 {
    const TYPE: &'static str = "VEC3";
    const N: usize = 3;
    type Inner = f32;
    fn new(d: &[Self::Inner]) -> Self {
        Vector3::new(d[0], d[1], d[2])
    }
}

struct Access<'a, T> {
    _pd: std::marker::PhantomData<T>,
    component_type: u32,
    count: usize,
    buffer: &'a [u8],
    stride: Option<usize>,
}

impl<T: AccessorType> Access<'_, T> {
    fn get(&self, idx: usize) -> Option<T> {
        let mut d = [T::Inner::default(); 3];
        for (i, r) in d.iter_mut().enumerate().take(T::N) {
            *r = self.get_scalar(idx, i)?;
        }
        Some(T::new(&d[..T::N]))
    }
    fn get_scalar(&self, idx: usize, comp: usize) -> Option<T::Inner> {
        let (buffer, offs) = match self.stride {
            None => (self.buffer, T::N * idx + comp),
            Some(stride) => (self.buffer.get(stride * idx..)?, comp),
        };
        Scalar::get(buffer, self.component_type, offs)
    }
}

fn uri_decode(uri: &str) -> Result<String> {
    let mut bres = Vec::new();
    let buri = uri.as_bytes();
    let mut i = 0;
    while i < buri.len() {
        let mut b = buri[i];
        if b == b'%' {
            let n = buri
                .get(i + 1..i + 3)
                .ok_or(anyhow!("invalid encoded uri {uri}"))?;
            b = u8::from_str_radix(std::str::from_utf8(n)?, 16)?;
            i += 3;
        } else {
            i += 1;
        }
        bres.push(b);
    }
    let res = String::from_utf8(bres)?;
    Ok(res)
}

fn decode_data_uri(mut data: &str) -> Result<Vec<u8>> {
    use base64::prelude::*;

    if let Some(semicolon) = data.find(';') {
        // let _mime = &data[..semicolon];
        data = &data[semicolon + 1..];
    }
    if let Some(comma) = data.find(',') {
        let encoder = &data[..comma];
        data = &data[comma + 1..];
        if encoder != "base64" {
            bail!("unknown data-uri encoder");
        }
    }
    let res = BASE64_STANDARD.decode(data)?;
    Ok(res)
}

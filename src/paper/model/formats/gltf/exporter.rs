use anyhow::Result;
use easy_imgui_window::easy_imgui_renderer::glow;
use serde_json::json;
use std::{
    borrow::Cow,
    io::Write,
    path::{Path, PathBuf},
};

use super::{
    GLTF_SCALE,
    data::{Accessor, Attributes, Binary, Buffer, BufferView, Image, Primitive, Texture},
};
use crate::{
    paper::{Papercraft, VertexIndex, formats::gltf::data::BoundingBox},
    util_3d::{self, Vector2, Vector3},
};

fn v_write_vec3(w: &mut Vec<u8>, v: Vector3) {
    w.extend_from_slice(&v.x.to_le_bytes());
    w.extend_from_slice(&v.y.to_le_bytes());
    w.extend_from_slice(&v.z.to_le_bytes());
}

fn v_write_vec2(w: &mut Vec<u8>, v: Vector2) {
    w.extend_from_slice(&v.x.to_le_bytes());
    w.extend_from_slice(&v.y.to_le_bytes());
}

fn v_write_index_as_u16(w: &mut Vec<u8>, x: VertexIndex) {
    let x = usize::from(x) as u16;
    w.extend_from_slice(&x.to_le_bytes());
}

fn v_write_index_as_u32(w: &mut Vec<u8>, x: VertexIndex) {
    let x = usize::from(x) as u32;
    w.extend_from_slice(&x.to_le_bytes());
}

fn write_u32(mut w: impl Write, x: u32) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}

struct Pieces {
    header: serde_json::Value,
    buffer: PiecesBuffer,
}

enum PiecesBuffer {
    Embedded(Vec<u8>),
    External(Vec<(PathBuf, Vec<u8>)>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GltfFormat {
    Binary,
    Text,
}

fn export_pieces(papercraft: &Papercraft, name: &str, gltf_format: GltfFormat) -> Pieces {
    let model = papercraft.model();

    let (min_filter, mag_filter) = if papercraft.options().tex_filter {
        (glow::LINEAR_MIPMAP_LINEAR, glow::LINEAR)
    } else {
        (glow::NEAREST, glow::NEAREST)
    };

    // As many meshes as textures/materials
    let mut primitives = Vec::with_capacity(model.num_textures());
    let mut materials = Vec::with_capacity(model.num_textures());
    let mut textures = Vec::with_capacity(model.num_textures());
    let mut images = Vec::with_capacity(model.num_textures());

    let mut extra_buffers = Vec::new();

    // Accessors for vertices, normals and uvs are shared between primitives.
    // Primitives only have indices in exclusivity.
    let mut accessors = Vec::with_capacity(3 + model.num_textures());

    let mut buffer_views = Vec::new();
    let mut buffers = Vec::new();

    let mut faces_by_mat = Vec::new();
    faces_by_mat.resize_with(model.num_textures(), Vec::new);
    for (_, face) in model.faces() {
        faces_by_mat[usize::from(face.material)].push(face);
    }

    let mut buffer: Vec<u8> = Vec::new();

    // The size of a vertex
    let vertex_stride = 2 * size_of::<Vector3>() + size_of::<Vector2>();

    // Position (0)
    buffer_views.push(BufferView {
        buffer: 0,
        byte_length: vertex_stride * model.num_vertices(),
        byte_offset: 0,
        byte_stride: Some(vertex_stride),
        target: Some(glow::ARRAY_BUFFER),
    });
    let (bb_min, bb_max) =
        util_3d::bounding_box_3d(papercraft.model().vertices().map(|(_, v)| v.pos()));

    accessors.push(Accessor {
        ty: "VEC3",
        buffer_view: 0,
        byte_offset: 0,
        component_type: glow::FLOAT,
        count: model.num_vertices(),
        bbox: Some(BoundingBox {
            min: (bb_min / GLTF_SCALE).into(),
            max: (bb_max / GLTF_SCALE).into(),
        }),
    });

    // Normal (1)
    buffer_views.push(BufferView {
        buffer: 0,
        byte_length: vertex_stride * model.num_vertices(),
        byte_offset: size_of::<Vector3>(),
        byte_stride: Some(vertex_stride),
        target: Some(glow::ARRAY_BUFFER),
    });
    accessors.push(Accessor {
        ty: "VEC3",
        buffer_view: 1,
        byte_offset: 0,
        component_type: glow::FLOAT,
        count: model.num_vertices(),
        bbox: None,
    });

    // UV (2)
    buffer_views.push(BufferView {
        buffer: 0,
        byte_length: vertex_stride * model.num_vertices(),
        byte_offset: 2 * size_of::<Vector3>(),
        byte_stride: Some(vertex_stride),
        target: Some(glow::ARRAY_BUFFER),
    });
    accessors.push(Accessor {
        ty: "VEC2",
        buffer_view: 2,
        byte_offset: 0,
        component_type: glow::FLOAT,
        count: model.num_vertices(),
        bbox: None,
    });

    for v in &model.vertices {
        v_write_vec3(&mut buffer, v.pos / GLTF_SCALE); // millimeters to meters
        v_write_vec3(&mut buffer, v.normal);
        v_write_vec2(&mut buffer, v.uv);
    }

    for (tex, faces) in model.textures().zip(&faces_by_mat) {
        if faces.is_empty() {
            continue;
        }

        // Indices
        let num_indices = 3 * faces.len();
        let use_32_bit_indices = num_indices > 65000;

        buffer_views.push(BufferView {
            buffer: 0,
            byte_length: num_indices
                * if use_32_bit_indices {
                    size_of::<u32>()
                } else {
                    size_of::<u16>()
                },
            byte_offset: buffer.len(),
            byte_stride: None,
            target: Some(glow::ELEMENT_ARRAY_BUFFER),
        });
        accessors.push(Accessor {
            ty: "SCALAR",
            buffer_view: buffer_views.len() - 1,
            byte_offset: 0,
            component_type: if use_32_bit_indices {
                glow::UNSIGNED_INT
            } else {
                glow::UNSIGNED_SHORT
            },
            count: num_indices,
            bbox: None,
        });

        let write_index = if use_32_bit_indices {
            v_write_index_as_u32
        } else {
            v_write_index_as_u16
        };
        for &v in faces.iter().flat_map(|f| f.vertices.iter()) {
            write_index(&mut buffer, v);
        }

        let material;

        if let Some(pixbuf) = tex.pixbuf.as_ref() {
            let format =
                image::ImageFormat::from_path(&tex.file_name).unwrap_or(image::ImageFormat::Png);

            let binary = match gltf_format {
                GltfFormat::Binary => {
                    let mut bv_image = BufferView {
                        buffer: 0,
                        byte_length: 0, // computed later
                        byte_offset: buffer.len(),
                        byte_stride: None,
                        target: None,
                    };

                    let mut w = std::io::Cursor::new(&mut buffer);
                    w.set_position(w.get_ref().len() as u64);
                    // Writing to a Cursor<Vec<u8>> can't fail
                    pixbuf.write_to(&mut w, format).unwrap();

                    bv_image.byte_length = buffer.len() - bv_image.byte_offset;

                    buffer_views.push(bv_image);

                    Binary::BufferView(buffer_views.len() - 1)
                }
                GltfFormat::Text => {
                    let mut name = PathBuf::from(&tex.file_name);
                    if let &[ext, ..] = format.extensions_str() {
                        name.set_extension(ext);
                    }
                    //pixbuf.save_with_format(name, format);
                    let mut b = Vec::new();
                    let mut w = std::io::Cursor::new(&mut b);
                    // Writing to a Cursor<Vec<u8>> can't fail
                    pixbuf.write_to(&mut w, format).unwrap();

                    let binary = Binary::Uri(name.to_string_lossy().into_owned());
                    extra_buffers.push((name, b));
                    binary
                }
            };

            images.push(Image {
                name: Some(&tex.file_name),
                mime_type: Some(format.to_mime_type()),
                binary,
            });

            textures.push(Texture {
                sampler: Some(0),
                source: images.len() - 1,
            });

            // The data::Material struct is to simple for exporting
            materials.push(json!({
                "name": tex.file_name,
                "doubleSided": true,
                "pbrMetallicRoughness": {
                    "baseColorTexture": { "index": textures.len() - 1 },
                    "metallicFactor": 0
                },
            }));
            material = Some(materials.len() - 1);
        } else {
            material = None;
        }

        primitives.push(Primitive {
            attributes: Attributes {
                position: Some(0),
                normal: Some(1),
                texcoord_0: Some(2),
            },
            indices: Some(accessors.len() - 1),
            material,
            mode: None,
        });
    }

    buffer.resize(buffer.len().next_multiple_of(4), 0);

    let pieces_buffer = match gltf_format {
        GltfFormat::Binary => {
            buffers.push(Buffer {
                byte_length: buffer.len(),
                uri: None,
            });
            PiecesBuffer::Embedded(buffer)
        }
        GltfFormat::Text => {
            let mut bin_name = PathBuf::from(&name);
            bin_name.set_extension("bin");
            buffers.push(Buffer {
                byte_length: buffer.len(),
                uri: Some(bin_name.to_string_lossy().into_owned()),
            });
            extra_buffers.push((bin_name, buffer));
            PiecesBuffer::External(extra_buffers)
        }
    };

    let header = json!({
        "asset": { "generator": format!("Papercraft {}", env!("CARGO_PKG_VERSION")), "version": "2.0" },
        "scene": 0,
        "scenes": [
            {
                "name": "Papercraft",
                "nodes": [0]
            }
        ],
        "nodes": [
            {
                "mesh": 0,
                "name": name,
            }
        ],
        "samplers": [ { "minFilter": min_filter, "magFilter": mag_filter } ],
        "meshes": [
            {
                "name": name,
                "primitives": primitives,
            }
        ],
        "materials": materials,
        "textures": textures,
        "images": images,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": buffers,
    });

    Pieces {
        header,
        buffer: pieces_buffer,
    }
}

pub fn export(papercraft: &Papercraft, file_name: &Path, gltf_format: GltfFormat) -> Result<()> {
    let name = file_name
        .file_stem()
        .map(|s| Cow::Owned(s.display().to_string()))
        .unwrap_or(Cow::Borrowed("model"));

    let Pieces { header, buffer } = export_pieces(papercraft, &name, gltf_format);

    match buffer {
        PiecesBuffer::Embedded(buffer) => {
            let mut header = header.to_string().into_bytes();
            header.resize(header.len().next_multiple_of(4), b' ');

            let f = std::fs::File::create(file_name)?;
            let mut f = std::io::BufWriter::new(f);
            // signature
            f.write_all(b"glTF")?;
            // version
            write_u32(&mut f, 2)?;
            // total len
            let total_len = 12 + 8 + header.len() + 8 + buffer.len();
            write_u32(&mut f, total_len as u32)?;

            write_u32(&mut f, header.len() as u32)?;
            f.write_all(b"JSON")?;
            f.write_all(&header)?;

            write_u32(&mut f, buffer.len() as u32)?;
            f.write_all(b"BIN\0")?;
            f.write_all(&buffer)?;
            f.flush()?;
        }
        PiecesBuffer::External(buffers) => {
            std::fs::write(file_name, serde_json::to_vec_pretty(&header).unwrap())?;
            let dir = match file_name.parent() {
                Some(p) => p,
                None => Path::new("."),
            };
            for (buf_name, data) in buffers {
                let file_name = dir.join(&buf_name);
                std::fs::write(file_name, data)?;
            }
        }
    }

    Ok(())
}

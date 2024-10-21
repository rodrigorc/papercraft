// Pepakura PDO format is documented, by reverse engeneering, I presume, at:
// * https://github.com/dpethes/pdo-tools.git/doc/pdo_spec_draft.txt
// Many thanks to "dpethes" for the work!

// I don't use many of the values, but I'll keep them there for reference
#![allow(dead_code)]

use anyhow::anyhow;
use std::cell::Cell;

use super::super::*;
type Vector2 = cgmath::Vector2<f32>;
type Vector3 = cgmath::Vector3<f32>;
use cgmath::Rad;

#[derive(Debug)]
pub struct Pdo {
    objs: Vec<Object>,
    mats: Vec<Material>,
    unfold: Option<Unfold>,
    settings: Settings,
}

impl Pdo {
    pub fn from_reader<R: Read>(mut rdr: R) -> Result<Pdo> {
        let mut reader = Reader {
            rdr: &mut rdr,
            version: 0,
            _mbcs: false,
            shift: 0,
        };
        reader.read_pdo()
    }
    pub fn objects(&self) -> &[Object] {
        &self.objs
    }
    pub fn materials(&self) -> &[Material] {
        &self.mats
    }
    pub fn unfold(&self) -> Option<&Unfold> {
        self.unfold.as_ref()
    }
    pub fn settings(&self) -> &Settings {
        &self.settings
    }
}

struct Reader<'r, R> {
    rdr: &'r mut R,
    version: u8,
    _mbcs: bool,
    shift: u32,
}

#[derive(Debug)]
pub struct BoundingBox {
    pub v0: Vector2,
    pub v1: Vector2,
}

#[derive(Debug)]
pub struct Object {
    pub name: String,
    pub visible: bool,
    pub vertices: Vec<Vertex>,
    pub faces: Vec<Face>,
    pub edges: Vec<Edge>,
}

#[derive(Debug)]
pub struct Vertex {
    pub v: Vector3,
}

#[derive(Debug)]
pub struct Face {
    pub mat_index: u32,
    pub part_index: u32,
    pub normal: Vector3,
    pub verts: Vec<VertInFace>,
}

#[derive(Debug)]
pub struct VertInFace {
    pub i_v: u32,
    pub pos2d: Vector2,
    pub uv: Vector2,
    pub flap: Option<Flap>,
}

#[derive(Debug)]
pub struct Flap {
    pub width: f32,
    pub angle1: Rad<f32>,
    pub angle2: Rad<f32>,
}

#[derive(Debug)]
pub struct Edge {
    pub i_f1: u32,
    pub i_f2: Option<u32>,
    pub i_v1: u32,
    pub i_v2: u32,
    pub connected: bool,
}

#[derive(Debug)]
pub struct Material {
    pub name: String,
    pub texture: Option<Texture>,
}

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub data: Cell<Vec<u8>>,
}
impl std::fmt::Debug for Texture {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("Texture")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("data", &"...")
            .finish()
    }
}

#[derive(Debug)]
pub struct Unfold {
    pub scale: f32,
    pub padding: bool,
    pub bb: BoundingBox,
    pub parts: Vec<Part>,
}

#[derive(Debug)]
pub struct Settings {
    pub margin_side: u32,
    pub margin_top: u32,
    pub page_size: Vector2,
    pub fold_line_hide_angle: Option<u32>, //degrees (180 = flat)
}

#[derive(Debug)]
pub struct Part {
    pub i_obj: u32,
    pub bb: BoundingBox,
    pub name: String,
    pub lines: Vec<Line>,
}

#[derive(Debug)]
pub struct Line {
    pub hidden: bool,
    pub type_: u32,        //0: Cut, 1: mountain, 2: valley, >=3: invisible
    pub first: (u32, u32), // (i_face, i_vertex)
    pub second: Option<(u32, u32)>,
}

impl<R: Read> Reader<'_, R> {
    fn read_bounding_box(&mut self) -> Result<BoundingBox> {
        let v0 = read_vector2_f64(self.rdr)?;
        let v1 = read_vector2_f64(self.rdr)?;
        Ok(BoundingBox { v0, v1 })
    }
    fn read_string(&mut self) -> Result<String> {
        let len = read_u32(self.rdr)?;
        let mut res = vec![0; len as usize];
        self.rdr.read_exact(&mut res)?;
        //TODO mbcs?
        res.pop();
        for c in &mut res {
            *c = c.wrapping_sub(self.shift as u8);
        }
        Ok(String::from_utf8_lossy(&res).into_owned())
    }
    fn read_texture(&mut self) -> Result<Texture> {
        let width = read_u32(self.rdr)?;
        let height = read_u32(self.rdr)?;
        let size = read_u32(self.rdr)?;
        let mut zdata = vec![0; size as usize];
        self.rdr.read_exact(&mut zdata)?;
        let mut z = flate2::bufread::ZlibDecoder::new(&zdata[..]);
        let mut data = Vec::with_capacity(width as usize * height as usize * 3);
        z.read_to_end(&mut data)?;
        Ok(Texture {
            width,
            height,
            data: Cell::new(data),
        })
    }
    fn read_face(&mut self) -> Result<Face> {
        let mat_index = read_u32(self.rdr)?;
        let part_index = read_u32(self.rdr)?;
        let normal = read_vector3_f64(self.rdr)?;
        let _coord = read_f64(self.rdr)?;
        let n_verts = read_u32(self.rdr)?;
        let mut verts = Vec::with_capacity(n_verts as usize);
        for _ in 0..n_verts {
            let i_v = read_u32(self.rdr)?;
            let pos2d = read_vector2_f64(self.rdr)?;
            let uv = read_vector2_f64(self.rdr)?;
            let flap = read_bool(self.rdr)?;
            let width = read_f64(self.rdr)?;
            let a1 = read_f64(self.rdr)?;
            let a2 = read_f64(self.rdr)?;
            let flap = flap.then_some(Flap {
                width: width as f32,
                angle1: Rad(a1 as f32),
                angle2: Rad(a2 as f32),
            });

            let mut _fold_info = [0; 24];
            self.rdr.read_exact(&mut _fold_info)?;
            verts.push(VertInFace {
                i_v,
                pos2d,
                uv,
                flap,
            });
        }
        Ok(Face {
            mat_index,
            part_index,
            normal,
            verts,
        })
    }
    fn read_object(&mut self) -> Result<Object> {
        let name = self.read_string()?;
        let visible = read_bool(self.rdr)?;
        let n_vertices = read_u32(self.rdr)?;
        let mut vertices = Vec::with_capacity(n_vertices as usize);
        for _ in 0..n_vertices {
            let v = read_vector3_f64(self.rdr)?;
            vertices.push(Vertex { v });
        }
        let n_faces = read_u32(self.rdr)?;
        let mut faces = Vec::with_capacity(n_faces as usize);
        for _ in 0..n_faces {
            let face = self.read_face()?;
            faces.push(face);
        }
        let n_edges = read_u32(self.rdr)?;
        let mut edges = Vec::with_capacity(n_edges as usize);
        for _ in 0..n_edges {
            let i_f1 = read_u32(self.rdr)?;
            let i_f2 = read_u32(self.rdr)?;
            let i_f2 = if i_f2 == u32::MAX { None } else { Some(i_f2) };
            let i_v1 = read_u32(self.rdr)?;
            let i_v2 = read_u32(self.rdr)?;
            let connected = read_u16(self.rdr)? != 0;
            let _nf = read_u32(self.rdr)?;
            edges.push(Edge {
                i_f1,
                i_f2,
                i_v1,
                i_v2,
                connected,
            });
        }
        Ok(Object {
            name,
            visible,
            vertices,
            faces,
            edges,
        })
    }
    fn read_material(&mut self) -> Result<Material> {
        let name = self.read_string()?;
        for _ in 0..16 {
            let _color3d = read_f32(self.rdr)?;
        }
        for _ in 0..4 {
            let _color2d = read_f32(self.rdr)?;
        }
        let textured = read_bool(self.rdr)?;
        let texture = if textured {
            let tex = self.read_texture()?;
            Some(tex)
        } else {
            None
        };
        Ok(Material { name, texture })
    }
    fn read_part(&mut self) -> Result<Part> {
        let i_obj = read_u32(self.rdr)?;
        let bb = self.read_bounding_box()?;
        let name = if self.version >= 5 {
            self.read_string()?
        } else {
            String::new()
        };
        let n_lines = read_u32(self.rdr)?;
        let mut lines = Vec::with_capacity(n_lines as usize);
        for _ in 0..n_lines {
            let line = self.read_line()?;
            lines.push(line);
        }
        Ok(Part {
            i_obj,
            bb,
            name,
            lines,
        })
    }
    fn read_line(&mut self) -> Result<Line> {
        let hidden = read_bool(self.rdr)?;
        let type_ = read_u32(self.rdr)?;
        let _unk = read_u8(self.rdr)?;
        let i_f = read_u32(self.rdr)?;
        let i_v = read_u32(self.rdr)?;
        let first = (i_f, i_v);
        let second = read_bool(self.rdr)?;
        let second = if second {
            let face2_idx = read_u32(self.rdr)?;
            let vertex2_idx = read_u32(self.rdr)?;
            Some((face2_idx, vertex2_idx))
        } else {
            None
        };
        Ok(Line {
            hidden,
            type_,
            first,
            second,
        })
    }
    fn read_pdo(&mut self) -> Result<Pdo> {
        const SIGNATURE: &[u8] = b"version 3\n";
        for s in SIGNATURE {
            let c = read_u8(self.rdr)?;
            if c != *s {
                return Err(anyhow!("signature error"));
            }
        }
        self.version = read_u32(self.rdr)? as u8;
        log::debug!("Version: {}", self.version);
        let mbcs = read_u32(self.rdr)?;
        log::debug!("MBCS: {}", mbcs);
        let _unk = read_u32(self.rdr)?;
        if self.version >= 5 {
            let designer = self.read_string()?;
            log::debug!("Designer: {}", designer);
            self.shift = read_u32(self.rdr)?;
            log::debug!("Shift: {}", self.shift);
        }
        let locale = self.read_string()?;
        log::debug!("Locale: {}", locale);
        let codepage = self.read_string()?;
        log::debug!("Codepage: {}", codepage);

        let texlock = read_u32(self.rdr)?;
        log::debug!("Texlock: {}", texlock);
        if self.version >= 6 {
            let show_startup_notes = read_bool(self.rdr)?;
            log::debug!("ShowStartupNotes: {}", show_startup_notes);
            let password_flag = read_bool(self.rdr)?;
            log::debug!("PasswordFlag: {}", password_flag);
        }
        let key = self.read_string()?;
        log::debug!("Key: {}", key);
        if self.version >= 6 {
            let v6_lock = read_u32(self.rdr)?;
            log::debug!("V6Lock: {}", v6_lock);
            for _ in 0..v6_lock {
                let _ = read_u64(self.rdr)?;
            }
        } else if self.version == 5 {
            let show_startup_notes = read_bool(self.rdr)?;
            log::debug!("ShowStartupNotes: {}", show_startup_notes);
            let password_flag = read_bool(self.rdr)?;
            log::debug!("PasswordFlags: {}", password_flag);
        }
        let assembled_height = read_f64(self.rdr)?;
        log::debug!("AssembledHeight: {}", assembled_height);
        let origin = read_vector3_f64(self.rdr)?;
        log::debug!("Origin: {:?}", origin);

        let n_objects = read_u32(self.rdr)?;
        let mut objs = Vec::with_capacity(n_objects as usize);
        for _ in 0..n_objects {
            let obj = self.read_object()?;
            objs.push(obj);
        }
        let n_materials = read_u32(self.rdr)?;
        let mut mats = Vec::with_capacity(n_materials as usize);
        for _ in 0..n_materials {
            let mat = self.read_material()?;
            mats.push(mat);
        }

        let has_unfold = read_bool(self.rdr)?;
        let unfold = if has_unfold {
            let scale = read_f64(self.rdr)? as f32;
            let padding = read_bool(self.rdr)?;
            let bb = self.read_bounding_box()?;
            let n_parts = read_u32(self.rdr)?;
            let mut parts = Vec::with_capacity(n_parts as usize);
            for _ in 0..n_parts {
                let part = self.read_part()?;
                parts.push(part);
            }
            let n_texts = read_u32(self.rdr)?;
            for _ in 0..n_texts {
                let _bb = self.read_bounding_box()?;
                let _line_spacing = read_f64(self.rdr)?;
                let _color = read_u32(self.rdr)?;
                let _font_size = read_u32(self.rdr)?;
                let _font_name = self.read_string()?;
                let n_lines = read_u32(self.rdr)?;
                for _ in 0..n_lines {
                    let _text = self.read_string()?;
                }
            }
            let n_images = read_u32(self.rdr)?;
            for _ in 0..n_images {
                let _bb = self.read_bounding_box()?;
                let _tex = self.read_texture()?;
            }
            let n_images2 = read_u32(self.rdr)?;
            for _ in 0..n_images2 {
                let _bb = self.read_bounding_box()?;
                let _tex = self.read_texture()?;
            }
            Some(Unfold {
                scale,
                padding,
                bb,
                parts,
            })
        } else {
            None
        };
        if self.version >= 6 && unfold.as_ref().is_some_and(|u| !u.parts.is_empty()) {
            let n_unk = read_u32(self.rdr)?;
            for _ in 0..n_unk {
                let n_parts = read_u32(self.rdr)?;
                for _ in 0..n_parts {
                    let _ = read_u32(self.rdr)?;
                }
            }
        }
        // settings
        let _show_flaps = read_bool(self.rdr)?;
        let _show_edge_id = read_bool(self.rdr)?;
        let _edge_id_pos = read_bool(self.rdr)?;
        let _face_mat = read_bool(self.rdr)?;
        let hide_almost_flat = read_bool(self.rdr)?;
        let fold_line_hide_angle = read_u32(self.rdr)?;
        let _draw_white_dot = read_bool(self.rdr)?;
        for _ in 0..4 {
            let _mountain_style = read_u32(self.rdr)?;
        }
        let page_type = read_u32(self.rdr)?;
        let mut page_size = match page_type {
            0  /*A4*/=> Vector2::new(210.0, 297.0),
            1  /*A3*/=> Vector2::new(297.0, 420.0),
            2  /*A2*/=> Vector2::new(420.0, 594.0),
            3  /*A1*/=> Vector2::new(594.0, 841.0),
            4  /*B5*/=> Vector2::new(176.0, 250.0),
            5  /*B4*/=> Vector2::new(250.0, 353.0),
            6  /*B3*/=> Vector2::new(353.0, 500.0),
            7  /*B2*/=> Vector2::new(500.0, 707.0),
            8  /*B1*/=> Vector2::new(707.0, 1000.0),
            9  /*letter*/=> Vector2::new(215.9, 279.4),
            10 /*legal*/=> Vector2::new(215.9, 355.6),
            11 => {
                let width = read_f64(self.rdr)?;
                let height = read_f64(self.rdr)?;
                Vector2::new(width as f32, height as f32)
            }
            _ /* unk */=> Vector2::new(210.0, 297.0),
        };
        let orientation = read_u32(self.rdr)?;
        if orientation != 0 && page_size.y > page_size.x {
            //landscape
            std::mem::swap(&mut page_size.x, &mut page_size.y);
        }
        let margin_side = read_u32(self.rdr)?;
        let margin_top = read_u32(self.rdr)?;
        for _ in 0..12 {
            let _fold_pattern = read_f64(self.rdr)?;
        }
        let _outline_padding = read_bool(self.rdr)?;
        let _scale_factor = read_f64(self.rdr)?;
        if self.version >= 5 {
            let _author = self.read_string()?;
            let _comment = self.read_string()?;
        }
        let settings = Settings {
            margin_side,
            margin_top,
            page_size,
            fold_line_hide_angle: hide_almost_flat.then_some(fold_line_hide_angle),
        };
        // eof!
        let eof = read_u32(self.rdr)?;
        assert!(eof == 9999);
        Ok(Pdo {
            objs,
            mats,
            unfold,
            settings,
        })
    }
}

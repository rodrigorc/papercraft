use std::io::Read;
use anyhow::{anyhow, Result};
type Vector2 = cgmath::Vector2<f32>;
type Vector3 = cgmath::Vector3<f32>;

#[derive(Debug)]
pub struct Pdo {
    objs: Vec<Object>,
    mats: Vec<Material>,
    unfold: Option<Unfold>,
    settings: Settings,
}

impl Pdo {
    pub fn from_reader<R: Read>(rdr: R) -> Result<Pdo> {
        let mut reader = Reader {
            rdr,
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

struct Reader<R> {
    rdr: R,
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
    pub flap: bool,
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
    pub data: Vec<u8>,
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
    pub type_: u32, //0: Cut, 1: mountain, 2: valley, >=3: invisible
    pub first: (u32, u32), // (i_face, i_vertex)
    pub second: Option<(u32, u32)>,
}

impl <R: Read> Reader<R> {
    fn read_u8(&mut self) -> Result<u8> {
        let mut x = [0; 1];
        self.rdr.read_exact(&mut x)?;
        Ok(x[0])
    }
    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }
    fn read_u16(&mut self) -> Result<u16> {
        let mut x = [0; 2];
        self.rdr.read_exact(&mut x)?;
        Ok(u16::from_le_bytes(x))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let mut x = [0; 4];
        self.rdr.read_exact(&mut x)?;
        Ok(u32::from_le_bytes(x))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let mut x = [0; 8];
        self.rdr.read_exact(&mut x)?;
        Ok(u64::from_le_bytes(x))
    }
    fn read_f32(&mut self) -> Result<f32> {
        let mut x = [0; 4];
        self.rdr.read_exact(&mut x)?;
        Ok(f32::from_le_bytes(x))
    }
    fn read_f64(&mut self) -> Result<f64> {
        let mut x = [0; 8];
        self.rdr.read_exact(&mut x)?;
        Ok(f64::from_le_bytes(x))
    }
    fn read_vector2(&mut self) -> Result<Vector2> {
        let x = self.read_f64()? as f32;
        let y = self.read_f64()? as f32;
        Ok(Vector2::new(x, y))
    }
    fn read_vector3(&mut self) -> Result<Vector3> {
        let x = self.read_f64()? as f32;
        let y = self.read_f64()? as f32;
        let z = self.read_f64()? as f32;
        Ok(Vector3::new(x, y, z))
    }
    fn read_bounding_box(&mut self) -> Result<BoundingBox> {
        let v0 = self.read_vector2()?;
        let v1 = self.read_vector2()?;
        Ok(BoundingBox { v0, v1 })
    }
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()?;
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
        let width = self.read_u32()?;
        let height = self.read_u32()?;
        let size = self.read_u32()?;
        let mut zdata = vec![0; size as usize];
        self.rdr.read_exact(&mut zdata)?;
        let mut z = flate2::bufread::ZlibDecoder::new(&zdata[..]);
        let mut data = Vec::with_capacity(width as usize * height as usize * 3);
        z.read_to_end(&mut data)?;
        Ok(Texture {
            width,
            height,
            data,
        })
    }
    fn read_face(&mut self) -> Result<Face> {
        let mat_index = self.read_u32()?;
        let part_index = self.read_u32()?;
        let normal = self.read_vector3()?;
        let _coord = self.read_f64()?;
        let n_verts = self.read_u32()?;
        let mut verts = Vec::with_capacity(n_verts as usize);
        for _ in 0 .. n_verts {
            let i_v = self.read_u32()?;
            let pos2d = self.read_vector2()?;
            let uv = self.read_vector2()?;
            let flap = self.read_bool()?;
            let _h = self.read_f64()?;
            let _a1 = self.read_f64()?;
            let _a2 = self.read_f64()?;

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
        let visible = self.read_bool()?;
        let n_vertices = self.read_u32()?;
        let mut vertices = Vec::with_capacity(n_vertices as usize);
        for _ in 0 .. n_vertices {
            let v = self.read_vector3()?;
            vertices.push(Vertex { v });
        }
        let n_faces = self.read_u32()?;
        let mut faces = Vec::with_capacity(n_faces as usize);
        for _ in 0 .. n_faces {
            let face = self.read_face()?;
            faces.push(face);
        }
        let n_edges = self.read_u32()?;
        let mut edges = Vec::with_capacity(n_edges as usize);
        for _ in 0 .. n_edges {
            let i_f1 = self.read_u32()?;
            let i_f2 = self.read_u32()?;
            let i_f2 = if i_f2 == u32::MAX { None } else { Some(i_f2) };
            let i_v1 = self.read_u32()?;
            let i_v2 = self.read_u32()?;
            let connected = self.read_u16()? != 0;
            let _nf = self.read_u32()?;
            edges.push(Edge {
                i_f1, i_f2,
                i_v1, i_v2,
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
        for _ in 0 .. 16 {
            let _color3d = self.read_f32()?;
        }
        for _ in 0 .. 4 {
            let _color2d = self.read_f32()?;
        }
        let textured = self.read_bool()?;
        let texture = if textured {
            let tex = self.read_texture()?;
            Some(tex)
        } else {
            None
        };
        Ok(Material {
            name,
            texture,
        })
    }
    fn read_part(&mut self) -> Result<Part> {
        let i_obj = self.read_u32()?;
        let bb = self.read_bounding_box()?;
        let name = if self.version >= 5 {
            self.read_string()?
        } else {
            String::new()
        };
        let n_lines = self.read_u32()?;
        let mut lines = Vec::with_capacity(n_lines as usize);
        for _ in 0 .. n_lines {
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
        let hidden = self.read_bool()?;
        let type_ = self.read_u32()?;
        let _unk = self.read_u8()?;
        let i_f = self.read_u32()?;
        let i_v = self.read_u32()?;
        let first = (i_f, i_v);
        let second = self.read_bool()?;
        let second = if second {
            let face2_idx = self.read_u32()?;
            let vertex2_idx = self.read_u32()?;
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
            let c = self.read_u8()?;
            if c != *s {
                return Err(anyhow!("signature error"));
            }
        }
        self.version = self.read_u32()? as u8;
        dbg!(self.version);
        let mbcs = self.read_u32()?;
        dbg!(mbcs);
        let _unk = self.read_u32()?;
        if self.version >= 5 {
            let designer = self.read_string()?;
            dbg!(designer);
            self.shift = self.read_u32()?;
            dbg!(self.shift);
        }
        let locale = self.read_string()?;
        dbg!(locale);
        let codepage = self.read_string()?;
        dbg!(codepage);

        let texlock = self.read_u32()?;
        dbg!(texlock);
        if self.version >= 6 {
            let show_startup_notes = self.read_bool()?;
            dbg!(show_startup_notes);
            let password_flag = self.read_bool()?;
            dbg!(password_flag);
        }
        let key = self.read_string()?;
        dbg!(key);
        if self.version >= 6 {
            let v6_lock = self.read_u32()?;
            dbg!(v6_lock);
            for _ in 0 .. v6_lock {
                let _ = self.read_u64()?;
            }
        } else if self.version == 5 {
            let show_startup_notes = self.read_bool()?;
            dbg!(show_startup_notes);
            let password_flag = self.read_bool()?;
            dbg!(password_flag);
        }
        let assembled_height = self.read_f64()?;
        dbg!(assembled_height);
        let origin = self.read_vector3()?;
        dbg!(origin);

        let n_objects = self.read_u32()?;
        let mut objs = Vec::with_capacity(n_objects as usize);
        for _ in 0 .. n_objects {
            let obj = self.read_object()?;
            objs.push(obj);
        }
        let n_materials = self.read_u32()?;
        let mut mats = Vec::with_capacity(n_materials as usize);
        for _ in 0 .. n_materials {
            let mat = self.read_material()?;
            mats.push(mat);
        }

        let has_unfold = self.read_bool()?;
        let unfold =  if has_unfold {
            let scale = self.read_f64()? as f32;
            let padding = self.read_bool()?;
            let bb = self.read_bounding_box()?;
            let n_parts = self.read_u32()?;
            let mut parts = Vec::with_capacity(n_parts as usize);
            for _ in 0 .. n_parts {
                let part = self.read_part()?;
                parts.push(part);
            }
            let n_texts = self.read_u32()?;
            for _ in 0 .. n_texts {
                let _bb = self.read_bounding_box()?;
                let _line_spacing = self.read_f64()?;
                let _color = self.read_u32()?;
                let _font_size = self.read_u32()?;
                let _font_name = self.read_string()?;
                let n_lines = self.read_u32()?;
                for _ in 0 .. n_lines {
                    let _text = self.read_string()?;
                }
            }
            let n_images = self.read_u32()?;
            for _ in 0 .. n_images {
                let _bb = self.read_bounding_box()?;
                let _tex = self.read_texture()?;
            }
            let n_images2 = self.read_u32()?;
            for _ in 0 .. n_images2 {
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
        if self.version >= 6 && !unfold.as_ref().is_some_and(|u| !u.parts.is_empty()) {
            let n_unk = self.read_u32()?;
            for _ in 0 .. n_unk {
                let n_parts = self.read_u32()?;
                for _ in 0 .. n_parts {
                    let _ = self.read_u32()?;
                }
            }
        }
        // settings
        let _show_flaps = self.read_bool()?;
        let _show_edge_id = self.read_bool()?;
        let _edge_id_pos = self.read_bool()?;
        let _face_mat = self.read_bool()?;
        let hide_almost_flat = self.read_bool()?;
        let fold_line_hide_angle = self.read_u32()?;
        let _draw_white_dot = self.read_bool()?;
        for _ in 0 .. 4 {
            let _mountain_style = self.read_u32()?;
        }
        let page_type = self.read_u32()?;
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
                let width = self.read_f64()?;
                let height = self.read_f64()?;
                Vector2::new(width as f32, height as f32)
            }
            _ /* unk */=> Vector2::new(210.0, 297.0),
        };
        let orientation = self.read_u32()?;
        if orientation != 0 && page_size.y > page_size.x { //landscape
            std::mem::swap(&mut page_size.x, &mut page_size.y);
        }
        let margin_side = self.read_u32()?;
        let margin_top = self.read_u32()?;
        for _ in 0 .. 12 {
            let _fold_pattern = self.read_f64()?;
        }
        let _outline_padding = self.read_bool()?;
        let _scale_factor = self.read_f64()?;
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
        let eof = self.read_u32()?;
        assert!(eof == 9999);
        Ok(Pdo {
            objs,
            mats,
            unfold,
            settings,
        })
    }
}

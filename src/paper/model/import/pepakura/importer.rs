use std::io::BufRead;

use super::data;
use super::super::*;
use cgmath::InnerSpace;
use image::{DynamicImage, ImageBuffer};

pub struct PepakuraImporter {
    pdo: data::Pdo,
    //VertexIndex -> (obj_id, face_id, vert_in_face)
    vertex_map: Vec<(u32, u32, u32)>,
    options: Option<PaperOptions>,
}

impl PepakuraImporter {
    pub fn new<R: BufRead>(f: R) -> Result<Self> {
        let pdo = data::Pdo::from_reader(f)?;

        let vertex_map: Vec<(u32, u32, u32)> = pdo
            .objects()
            .iter()
            .enumerate()
            .flat_map(|(i_o, obj)| {
                obj.faces
                    .iter()
                    .enumerate()
                    .flat_map(move |(i_f, f)| {
                        (0 .. f.verts.len()).map(move |i_vf| (i_o as u32, i_f as u32, i_vf as u32))
                    })
            })
            .collect();

        Ok(PepakuraImporter {
            pdo,
            vertex_map,
            options: None,
        })
    }
}
impl Importer for PepakuraImporter {
    // (obj_id, vertex_id)
    type VertexId = (u32, u32);
    // (obj_id, face_id)
    type FaceId = (u32, u32);

    fn build_vertices(&self) -> (bool, Vec<Vertex>) {
        let vs = self.vertex_map
            .iter()
            .map(|&(i_o, i_f, i_vf)| {
                let obj = &self.pdo.objects()[i_o as usize];
                let f = &obj.faces[i_f as usize];
                let v_f = &f.verts[i_vf as usize];
                let v = &obj.vertices[v_f.i_v as usize];

                Vertex {
                    pos: v.v,
                    normal: f.normal,
                    uv: v_f.uv,
                }
            })
            .collect();
        (true, vs)
    }
    fn vertex_map(&self, i_v: VertexIndex) -> Self::VertexId {
        let (i_o, i_f, i_vf) = self.vertex_map[usize::from(i_v)];
        let i_v = self.pdo.objects()[i_o as usize].faces[i_f as usize].verts[i_vf as usize].i_v;
        (i_o, i_v)
    }
    fn face_count(&self) -> usize {
        self.pdo.objects()
            .iter()
            .map(|o| o.faces.len())
            .sum()
    }
    fn for_each_face(&self, mut f: impl FnMut(Self::FaceId, &[VertexIndex], MaterialIndex)) {
        for (obj_id, obj) in self.pdo.objects().iter().enumerate() {
            let obj_id = obj_id as u32;
            for (face_id, face) in obj.faces.iter().enumerate() {
                let face_id = face_id as u32;
                let verts: Vec<VertexIndex> = (0 .. face.verts.len())
                    .map(|v_f| {
                        let id = (obj_id, face_id, v_f as u32);
                        let i = self.vertex_map.iter().position(|x| x == &id).unwrap();
                        VertexIndex::from(i)
                    })
                    .collect();
                // We will add a default material at the end of the textures, so map any out-of bounds to that
                let mat_index = face.mat_index.min(self.pdo.materials().len() as u32);
                let mat = MaterialIndex::from(mat_index as usize);
                f((obj_id, face_id), &verts, mat)
            }
        }
    }
    fn build_textures(&mut self) -> Vec<Texture> {
        let mut textures: Vec<_> = self.pdo.materials_mut()
            .iter_mut()
            .map(|mat| {
                let pixbuf = mat.texture.take().and_then(|t| {
                    let img = ImageBuffer::from_raw(t.width, t.height, t.data);
                    img.map(|b| DynamicImage::ImageRgb8(b))
                });
                Texture {
                    file_name: mat.name.clone() + ".png",
                    pixbuf,
                }
            })
            .collect();
        textures.push(Texture::default());
        textures
    }
    fn compute_edge_status(&self, edge_id: (Self::VertexId, Self::VertexId)) -> Option<EdgeStatus> {
        let ((obj_id, v0_id), (_, v1_id)) = edge_id;
        let vv = (v0_id, v1_id);
        let obj = &self.pdo.objects()[obj_id as usize];
        let Some(edge) = obj.edges
            .iter()
            .find(|&e| vv == (e.i_v1, e.i_v2) || vv == (e.i_v2, e.i_v1))
            else { return None };
        if edge.connected {
            Some(EdgeStatus::Joined)
        } else {
            let v_f = obj.faces[edge.i_f1 as usize].verts.iter().find(|v_f| v_f.i_v == edge.i_v1).unwrap();
            if v_f.flap {
                Some(EdgeStatus::Cut(TabSide::True))
            } else {
                None
            }
        }
    }
    fn relocate_islands<'a>(&mut self, model: &Model, islands: impl Iterator<Item=&'a mut Island>) -> bool {
        let Some(unfold) = self.pdo.unfold() else { return false; };

        let margin = Vector2::new(self.pdo.settings().margin_side as f32, self.pdo.settings().margin_top as f32);
        let page_size = self.pdo.settings().page_size;
        let area_size = page_size - 2.0 * margin;

        let mut options = PaperOptions::default();
        options.scale = unfold.scale;
        options.page_size = (page_size.x, page_size.y);
        options.margin = (margin.y, margin.x, margin.x, margin.y);

        let mut n_cols = 0;
        let mut max_page = (0, 0);
        for island in islands {
            let face = &model[island.root_face()];
            let [i_v0, i_v1, _] = face.index_vertices();
            let (ip_obj, ip_face, ip_v0) = self.vertex_map[usize::from(i_v0)];
            let (_, _, ip_v1) = self.vertex_map[usize::from(i_v1)];
            let p_face = &self.pdo.objects()[ip_obj as usize].faces[ip_face as usize];
            let vf0 = p_face.verts[ip_v0 as usize].pos2d;
            let vf1 = p_face.verts[ip_v1 as usize].pos2d;
            let i_part = p_face.part_index;

            let normal = model.face_plane(face);
            let pv0 = normal.project(&model[i_v0].pos(), options.scale);
            let pv1 = normal.project(&model[i_v1].pos(), options.scale);

            let part = &unfold.parts[i_part as usize];

            let rot = (pv1 - pv0).angle(vf1 - vf0);
            let loc = vf0 - pv0 + part.bb.v0;

            let mut col = loc.x.div_euclid(area_size.x) as i32;
            let mut row = loc.y.div_euclid(area_size.y) as i32;
            let loc = Vector2::new(loc.x.rem_euclid(area_size.x), loc.y.rem_euclid(area_size.y));
            let loc = loc + margin;

            // Some models use negative pages to hide pieces
            if col < 0 || row < 0 {
                col = -1;
                row = 0;
            } else {
                let row = row as u32;
                let col = col as u32;
                n_cols = n_cols.max(col);
                if row > max_page.0 || (row == max_page.0 && col > max_page.1) {
                    max_page = (row, col);
                }
            }

            let loc = options.page_to_global(PageOffset { row, col, offset: loc });
            island.reset_transformation(island.root_face(), rot, loc);
        }
        // 0-based
        options.page_cols = n_cols + 1;
        options.pages = max_page.0 * options.page_cols + max_page.1 + 1;

        self.options = Some(options);
        true
    }
    fn build_options(&mut self) -> Option<PaperOptions> {
        self.options
            .take()
            .map(|mut options| {
                if let Some(a) = self.pdo.settings().fold_line_hide_angle {
                    options.hidden_line_angle = (180 - a) as f32;
                }
                options
            })
    }
}
use std::{
    io::{Read, Seek, Write},
    path::Path,
};

use super::*;
use anyhow::Result;
use model::import::Importer;

impl Papercraft {
    pub fn save<W: Write + Seek>(&self, w: W) -> Result<()> {
        let mut zip = zip::ZipWriter::new(w);
        let options = zip::write::FileOptions::default();

        zip.start_file("model.json", options)?;
        serde_json::to_writer(&mut zip, self)?;

        for tex in self.model.textures() {
            if let Some(pixbuf) = tex.pixbuf() {
                let file_name = tex.file_name();
                zip.start_file(&format!("tex/{file_name}"), options)?;
                let mut data = Vec::new();
                let format =
                    image::ImageFormat::from_path(file_name).unwrap_or(image::ImageFormat::Png);
                pixbuf.write_to(&mut std::io::Cursor::new(&mut data), format)?;
                zip.write_all(&data[..])?;
            }
        }
        zip.finish()?;
        Ok(())
    }

    pub fn load<R: Read + Seek>(r: R) -> Result<Papercraft> {
        let mut zip = zip::ZipArchive::new(r)?;
        let mut zmodel = zip.by_name("model.json")?;
        let mut papercraft: Papercraft = serde_json::from_reader(&mut zmodel)?;
        drop(zmodel);

        papercraft.model.reload_textures(|file_name| {
            let mut ztex = zip.by_name(&format!("tex/{file_name}"))?;
            let mut data = Vec::new();
            ztex.read_to_end(&mut data)?;
            let img = image::io::Reader::new(std::io::Cursor::new(&data))
                .with_guessed_format()?
                .decode()?;
            Ok(img)
        })?;

        papercraft.recompute_edge_ids();
        Ok(papercraft)
    }
    fn recompute_edge_ids(&mut self) {
        let mut next_edge_id = 0;
        let mut edge_ids: Vec<Option<EdgeId>> = vec![None; self.model.num_edges()];

        let mut edge_collection: Vec<_> = self
            .model
            .edges()
            .zip(&self.edges)
            .zip(&mut edge_ids)
            .map(|(((_, edge), edge_status), edge_id)| {
                let (p0, p1) = self.model.edge_pos(edge);
                let c = (p0 + p1) / 2.0;
                (c, edge, edge_status, edge_id)
            })
            .collect();

        edge_collection.sort_by(|(ca, _, _, _), (cb, _, _, _)| {
            ca.y.total_cmp(&cb.y)
                .then_with(|| ca.z.total_cmp(&cb.z))
                .then_with(|| ca.x.total_cmp(&cb.x))
        });

        for (_, edge, edge_status, edge_id) in edge_collection {
            match (edge.faces(), edge_status) {
                // edges from tessellations or rims don't have ids
                (_, EdgeStatus::Hidden) | ((_, None), _) => {}
                _ => {
                    next_edge_id += 1;
                    *edge_id = Some(EdgeId::new(next_edge_id));
                }
            }
        }
        self.edge_ids = edge_ids;
    }

    pub fn import<I: Importer>(mut importer: I) -> Papercraft {
        let (model, face_map, edge_map) = Model::from_importer(&mut importer);

        let edges: Vec<_> = edge_map
            .iter()
            .enumerate()
            .map(|(i_edge, edge_id)| {
                let i_edge = EdgeIndex::from(i_edge);
                let edge = &model[i_edge];
                match edge.faces() {
                    // Edge from tessellation of a n-gon
                    (fa, Some(fb)) if face_map[usize::from(fa)] == face_map[usize::from(fb)] => {
                        EdgeStatus::Hidden
                    }
                    // Rim
                    (_, None) => {
                        // edges in the rim (without adjacent face) usually do not have a flap, but if it does,
                        // the FlapSide is false, no matter what the loader says
                        match importer.compute_edge_status(*edge_id) {
                            Some(EdgeStatus::Cut(FlapSide::False | FlapSide::True)) => {
                                EdgeStatus::Cut(FlapSide::False)
                            }
                            _ => EdgeStatus::Cut(FlapSide::Hidden),
                        }
                    }
                    // Normal edge
                    _ => importer
                        .compute_edge_status(*edge_id)
                        .unwrap_or(EdgeStatus::Cut(FlapSide::False)),
                }
            })
            .collect();

        let mut pending_faces: FxHashSet<FaceIndex> =
            model.faces().map(|(i_face, _face)| i_face).collect();

        let mut islands = SlotMap::with_key();
        while let Some(root) = pending_faces.iter().copied().next() {
            pending_faces.remove(&root);

            traverse_faces_ex(
                &model,
                root,
                (),
                NoMatrixTraverseFace(&edges),
                |i_face, _, _| {
                    pending_faces.remove(&i_face);
                    ControlFlow::Continue(())
                },
            );

            let island = Island {
                root,
                loc: Vector2::zero(),
                rot: Rad::zero(),
                mx: Matrix3::one(),
                name: String::new(),
            };
            islands.insert(island);
        }

        let need_packing = !importer.relocate_islands(&model, islands.values_mut());

        let mut need_fix_options = false;
        let mut options = importer.build_options().unwrap_or_else(|| {
            need_fix_options = true;
            PaperOptions::default()
        });
        if !model.has_textures() {
            options.texture = false;
        }

        let mut papercraft = Papercraft {
            model,
            options,
            edges,
            islands,
            memo: Memoization::default(),
            edge_ids: Vec::new(),
        };
        if need_fix_options {
            let (v_min, v_max) = crate::util_3d::bounding_box_3d(
                papercraft.model().vertices().map(|(_, v)| v.pos()),
            );
            let size = (v_max.x - v_min.x)
                .max(v_max.y - v_min.y)
                .max(v_max.z - v_min.z);
            let paper_size = papercraft
                .options
                .page_size
                .0
                .max(papercraft.options.page_size.1);
            // Scale to half the paper size, that looks handy.
            let scale = paper_size / size / 2.0;
            let scale = if scale > 1.0 {
                scale.round()
            } else {
                // All in one line just for show
                (((1.0 / (1.0 / scale).round()) * 100.0).round() / 100.0).max(0.01)
            };
            papercraft.options.scale = scale;
        }
        if need_packing {
            let num_pages = papercraft.pack_islands();
            papercraft.options.pages = num_pages;
        }
        papercraft.recompute_edge_ids();
        papercraft
    }

    pub fn export_waveobj(&self, file_name: &Path) -> Result<()> {
        use std::collections::hash_map::Entry;
        use std::io::prelude::*;

        // f32 cannot be used as a hash index because it does not implement Eq nor Hash, something to do with precision and ambiguous representations and NaNs...
        // But we never operate with the values in model, so same f32 values should always have the same bit pattern, and we can use that bit-pattern as the hash index.
        fn index_vector2(v: &Vector2) -> (u32, u32) {
            (v.x.to_bits(), v.y.to_bits())
        }
        fn index_vector3(v: &Vector3) -> (u32, u32, u32) {
            (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
        }

        let title = file_name
            .file_stem()
            .map(|s| s.to_string_lossy())
            .unwrap_or(std::borrow::Cow::Borrowed("object"));
        let title: String = title
            .as_ref()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let mtl_name = file_name.with_extension("mtl");
        let has_textures = self.model.has_textures();

        let f = std::fs::File::create(file_name)?;
        let mut f = std::io::BufWriter::new(f);

        // Vertex identity is important for properly exporting the mesh:
        // Two Papercraft vertices are considered the same if they appear on opposite sides of an
        // edge.

        #[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
        struct Vid(u32);

        // Maps VertexIndex into the position of next_id
        let mut idx: Vec<Option<usize>> = vec![None; self.model.num_vertices()];
        // Unique ids of the vertices
        let mut next_id: Vec<Vid> = Vec::new();

        fn mark_equal(
            next_id: &mut Vec<Vid>,
            idx: &mut [Option<usize>],
            a: VertexIndex,
            b: VertexIndex,
        ) {
            let a = usize::from(a);
            let b = usize::from(b);
            match (idx[a], idx[b]) {
                (None, None) => {
                    // id and its position are the same when new
                    let id = next_id.len();
                    next_id.push(Vid(id as u32));
                    idx[a] = Some(id);
                    idx[b] = Some(id);
                }
                (Some(x), None) => {
                    idx[b] = Some(x);
                }
                (None, Some(x)) => {
                    idx[a] = Some(x);
                }
                (Some(x), Some(y)) => {
                    next_id[x] = next_id[y];
                }
            };
        }
        for (i_face, face) in self.model.faces() {
            for (i_v0, i_v1, i_edge) in face.vertices_with_edges() {
                let edge = &self.model[i_edge];
                let (i_fa, Some(i_fb)) = edge.faces() else {
                    continue;
                };
                if i_fa != i_face {
                    continue;
                }
                let face_b = &self.model[i_fb];
                let (mut i_v0b, mut i_v1b) = face_b.vertices_of_edge(i_edge).unwrap();

                // Is it a normal or inverted edge?
                // Check for normal edge: First compare the ids, they are sometimes the same
                if i_v0 != i_v1b && i_v1 != i_v0b &&
                    // Then compare the exact values, those are most likely the same
                    self.model[i_v0].pos() != self.model[i_v1b].pos()
                {
                    // Here it is probably an inverted edge, make sure by computing the distance
                    let d2_normal = self.model[i_v0].pos().distance2(self.model[i_v1b].pos());
                    let d2_inverted = self.model[i_v0].pos().distance2(self.model[i_v0b].pos());
                    if d2_inverted < d2_normal {
                        std::mem::swap(&mut i_v0b, &mut i_v1b);
                    }
                }
                mark_equal(&mut next_id, &mut idx, i_v0, i_v1b);
                mark_equal(&mut next_id, &mut idx, i_v1, i_v0b);
            }
        }

        let mut fx = FxHashMap::default();

        // Deduplicate the ids, make them contiguous and get the position
        let mut vertex_pos = Vec::new();
        let vertex_map: Vec<_> = idx
            .iter()
            .enumerate()
            .map(|(i_v, id)| {
                let vid = next_id[id.unwrap()];
                let id = *fx.entry(vid).or_insert_with(|| {
                    vertex_pos.push(self.model[VertexIndex::from(i_v)].pos());
                    vertex_pos.len() as u32
                });
                id
            })
            .collect();

        if has_textures {
            writeln!(f, "mtllib {}", mtl_name.display())?;
        }
        writeln!(f, "o {title}")?;
        for pos in &vertex_pos {
            writeln!(f, "v {} {} {}", pos[0], pos[1], pos[2])?;
        }

        // Normals and UVs are deduplicated by value, they don't affect the geometry
        let mut index_vn = FxHashMap::default();
        let mut index_vt = FxHashMap::default();

        for (_, v) in self.model.vertices() {
            let n = v.normal();
            let id = index_vn.len() + 1;
            let e = index_vn.entry(index_vector3(&n));
            if let Entry::Vacant(vacant) = e {
                writeln!(f, "vn {} {} {}", n[0], n[1], n[2])?;
                vacant.insert(id);
            }
        }
        for (_, v) in self.model.vertices() {
            let uv = v.uv();
            let id = index_vt.len() + 1;
            let e = index_vt.entry(index_vector2(&uv));
            if let Entry::Vacant(vacant) = e {
                writeln!(f, "vt {} {}", uv[0], 1.0 - uv[1])?;
                vacant.insert(id);
            }
        }
        writeln!(f, "s 0")?;

        let by_mat = if has_textures {
            let mut by_mat = vec![Vec::new(); self.model.num_textures()];
            for (i_face, face) in self.model.faces() {
                by_mat[usize::from(face.material())].push(i_face);
            }
            by_mat
        } else {
            vec![(0..self.model.num_faces()).map(FaceIndex::from).collect()]
        };

        // We iterate over the triangles, but export the flat-face, we have to skip duplicated
        // triangles. If one flat-face uses two different materials, that will not be properly
        // exported.
        let mut done_faces = vec![false; self.model.num_faces()];
        for (i_mat, face_by_mat) in by_mat.iter().enumerate() {
            if face_by_mat.is_empty() {
                continue;
            }
            if has_textures {
                writeln!(f, "usemtl Material.{i_mat:03}")?;
            }
            for &i_face in face_by_mat {
                if done_faces[usize::from(i_face)] {
                    continue;
                }
                //In model, faces are all triangles, group them by flatness
                let flat_face = self.get_flat_faces(i_face);
                let mut flat_contour: Vec<_> = flat_face
                    .iter()
                    .map(|&f| {
                        done_faces[usize::from(f)] = true;
                        f
                    })
                    .flat_map(|f| self.model()[f].vertices_with_edges())
                    .filter_map(|(i_v0, i_v1, e)| {
                        if self.edge_status(e) == EdgeStatus::Hidden {
                            None
                        } else {
                            Some((i_v0, i_v1))
                        }
                    })
                    .collect();
                write!(f, "f")?;
                let mut next = Some(flat_contour.len() - 1);
                while let Some(pos) = next {
                    let vertex = flat_contour.remove(pos);
                    let (i_v0, i_v1) = vertex;
                    let v0 = vertex_map[usize::from(i_v0)];
                    let vx = &self.model()[i_v0];
                    let t = index_vt[&index_vector2(&vx.uv())];
                    let n = index_vn[&index_vector3(&vx.normal())];
                    write!(f, " {v0}/{t}/{n}")?;
                    next = flat_contour.iter().position(|(i_x0, _)| i_v1 == *i_x0);
                }
                writeln!(f)?;
            }
        }
        drop(f);

        if has_textures {
            let fm = std::fs::File::create(mtl_name)?;
            let mut fm = std::io::BufWriter::new(fm);

            let dir = file_name.parent();
            for (i_mat, (tex, faces)) in self.model().textures().zip(&by_mat).enumerate() {
                if faces.is_empty() {
                    continue;
                }
                writeln!(fm, "newmtl Material.{i_mat:03}")?;

                if let Some(pixbuf) = tex.pixbuf() {
                    let bmp_name = tex.file_name();
                    writeln!(fm, "map_Kd {bmp_name}")?;

                    let path = Path::new(bmp_name);
                    let mut full_path_buf;
                    let full_path = if let Some(dir) = dir {
                        full_path_buf = dir.to_owned();
                        full_path_buf.push(path);
                        &full_path_buf
                    } else {
                        path
                    };
                    pixbuf.save(full_path)?;
                }
            }
            drop(fm);
        }
        Ok(())
    }
}

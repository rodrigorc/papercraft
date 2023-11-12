use std::{io::{Read, Seek, Write}, path::Path, hash::Hash};

use fxhash::{FxHashMap, FxHashSet};
use cgmath::{One, Rad, Zero};
use slotmap::SlotMap;
use anyhow::Result;

use super::*;

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
                let format = image::ImageFormat::from_path(file_name).unwrap_or(image::ImageFormat::Png);
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

        let mut edge_collection: Vec<_> = self.model.edges()
            .zip(&self.edges)
            .zip(&mut edge_ids)
            .map(|(((_, edge), edge_status), edge_id)| {
                let (p0, p1) = self.model.edge_pos(edge);
                let c = (p0 + p1) / 2.0;
                (c, edge, edge_status, edge_id)
            })
            .collect();

        edge_collection.sort_by(|(ca, _, _, _), (cb, _, _, _)| {
            ca.y.total_cmp(&cb.y).then_with(|| ca.z.total_cmp(&cb.z)).then_with(|| ca.x.total_cmp(&cb.x))
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

    pub fn import<I: Importer>(file_name: &Path) -> Result<Papercraft> {
        let mut importer = I::import(file_name)?;
        let (model, face_map, edge_map) = Model::from_importer(&mut importer);

        let edges: Vec<_> = edge_map.iter().enumerate()
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
                        // edges in the rim (without adjacent face) usually do not have a tab, but if it does,
                        // the TabSide is false, no matter what the loader says
                        match importer.compute_edge_status(*edge_id) {
                            Some(EdgeStatus::Cut(TabSide::False | TabSide::True)) => EdgeStatus::Cut(TabSide::False),
                            _ => EdgeStatus::Cut(TabSide::Hidden),
                        }
                    }
                    // Normal edge
                    _ => {
                        importer.compute_edge_status(*edge_id)
                            .unwrap_or(EdgeStatus::Cut(TabSide::False))
                    }
                }
            })
            .collect();

        let mut pending_faces: FxHashSet<FaceIndex> = model.faces().map(|(i_face, _face)| i_face).collect();

        let mut islands = SlotMap::with_key();
        while let Some(root) = pending_faces.iter().copied().next() {
            pending_faces.remove(&root);

            traverse_faces_ex(&model, root, (), NoMatrixTraverseFace(&model, &edges),
                |i_face, _, _| {
                    pending_faces.remove(&i_face);
                    ControlFlow::Continue(())
                }
            );

            let island = Island {
                root,
                loc: Vector2::zero(),
                rot: Rad::zero(),
                mx: Matrix3::one(),
            };
            islands.insert(island);
        }

        let need_packing = !importer.relocate_islands(&model, islands.values_mut());

        let mut need_fix_options = false;
        let options = importer.build_options()
            .unwrap_or_else(|| {
                need_fix_options = true;
                PaperOptions::default()
            });

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
                papercraft.model()
                    .vertices()
                    .map(|(_, v)| v.pos())
            );
            let size = (v_max.x - v_min.x).max(v_max.y - v_min.y).max(v_max.z - v_min.z);
            let paper_size = papercraft.options.page_size.0.max(papercraft.options.page_size.1);
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

        Ok(papercraft)
    }

    pub fn export_waveobj(&self, file_name: &Path) -> Result<()> {
        use std::io::prelude::*;
        use std::collections::hash_map::Entry;

        let title = file_name.file_stem().map(|s| s.to_string_lossy()).unwrap_or(std::borrow::Cow::Borrowed("object"));
        let title: String = title.as_ref().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_'}).collect();
        let mtl_name = file_name.with_extension("mtl");

        let f = std::fs::File::create(file_name)?;
        let mut f = std::io::BufWriter::new(f);

        let mut index_v = FxHashMap::default();
        let mut index_vn = FxHashMap::default();
        let mut index_vt = FxHashMap::default();

        //TODO: vertex de-duplication properly
        // f32 cannot be used as a hash index because it does not implement Eq nor Hash, something to do with precision and ambiguous representations and NaNs...
        // But we never operate with the values in model, so same f32 values should always have the same bit pattern, and we can use that bit-pattern as the hash index.

        trait Indexable {
            type Res: Eq + Hash;
            fn indexable(&self) -> Self::Res;
        }
        impl Indexable for f32 {
            type Res = u32;
            fn indexable(&self) -> Self::Res {
                self.to_bits()
            }
        }
        impl Indexable for Vector2 {
            type Res = (u32, u32);
            fn indexable(&self) -> Self::Res {
                (self.x.to_bits(), self.y.to_bits())
            }
        }
        impl Indexable for Vector3 {
            type Res = (u32, u32, u32);
            fn indexable(&self) -> Self::Res {
                (self.x.to_bits(), self.y.to_bits(), self.z.to_bits())
            }
        }
        writeln!(f, "mtllib {}", mtl_name.display())?;
        writeln!(f, "o {title}")?;
        for (_, v) in self.model.vertices() {
            let pos = v.pos();
            let id = index_v.len() + 1;
            let e = index_v.entry(pos.indexable());
            if let Entry::Vacant(vacant) = e {
                writeln!(f, "v {} {} {}", pos[0], pos[1], pos[2])?;
                vacant.insert(id);
            }
        }
        for (_, v) in self.model.vertices() {
            let n = v.normal();
            let id = index_vn.len() + 1;
            let e = index_vn.entry(n.indexable());
            if let Entry::Vacant(vacant) = e {
                writeln!(f, "vn {} {} {}", n[0], n[1], n[2])?;
                vacant.insert(id);
            }
        }
        for (_, v) in self.model.vertices() {
            let uv = v.uv();
            let id = index_vt.len() + 1;
            let e = index_vt.entry(uv.indexable());
            if let Entry::Vacant(vacant) = e {
                writeln!(f, "vt {} {}", uv[0], 1.0 - uv[1])?;
                vacant.insert(id);
            }
        }
        writeln!(f, "s 1")?;
        let mut by_mat = vec![Vec::new(); self.model.num_textures()];
        for (i_face, face) in self.model.faces() {
            by_mat[usize::from(face.material())].push(i_face);
        }

        let mut done_faces = FxHashSet::default();
        for (i_mat, face_by_mat) in by_mat.iter().enumerate() {
            writeln!(f, "usemtl Material.{i_mat:03}")?;
            for &i_face in face_by_mat {
                if !done_faces.insert(i_face) {
                    continue;
                }
                //In model, faces are all triangles, group them by flatness
                let flat_face = self.get_flat_faces(i_face);
                let mut flat_contour: Vec<_> = flat_face
                    .iter()
                    .map(|f| {
                        done_faces.insert(*f);
                        f
                    })
                    .flat_map(|&f| self.model()[f].vertices_with_edges())
                    .filter_map(|(v0, v1, e)| {
                        if self.edge_status(e) == EdgeStatus::Hidden {
                            None
                        } else {
                            Some((v0, v1))
                        }
                    })
                    .collect();
                write!(f, "f")?;
                let mut next = Some(flat_contour.len() - 1);
                while let Some(pos) = next {
                    let vertex = flat_contour.remove(pos);
                    let (v0, v1) = vertex;
                    let vx = &self.model()[v0];
                    let v = index_v[&vx.pos().indexable()];
                    let t = index_vt[&vx.uv().indexable()];
                    let n = index_vn[&vx.normal().indexable()];
                    write!(f, " {v}/{t}/{n}")?;
                    next = flat_contour.iter().position(|(x0, _x1)| *x0 == v1);
                }
                writeln!(f)?;
            }
        }
        drop(f);

        let fm = std::fs::File::create(mtl_name)?;
        let mut fm = std::io::BufWriter::new(fm);

        let dir = file_name.parent();

        for (i_mat, tex) in self.model().textures().enumerate() {
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

        Ok(())
    }
}


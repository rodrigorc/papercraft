use std::{io::{Read, Seek, Write}, path::Path, hash::Hash};

use fxhash::{FxHashMap, FxHashSet};
use cgmath::{One, Rad, Zero};
use slotmap::SlotMap;
use crate::{waveobj, pepakura};
use anyhow::{Result, anyhow, Context};

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

    pub fn import_waveobj(file_name: &Path) -> Result<Papercraft> {
        let f = std::fs::File::open(file_name)?;
        let f = std::io::BufReader::new(f);
        let (matlib, obj) = waveobj::Model::from_reader(f)?;
        let matlib = match matlib {
            Some(matlib) => {
                Some(waveobj::solve_find_matlib_file(matlib.as_ref(), file_name)
                    .ok_or_else(|| anyhow!("{} matlib not found", matlib))?)
            }
            None => None,
        };
        let mut texture_map = FxHashMap::default();

        if let Some(matlib) = matlib {
            // Textures are read from the .mtl file
            let err_mtl = || format!("Error reading matlib file {}", matlib.display());
            let f = std::fs::File::open(&matlib)
                .with_context(err_mtl)?;
            let f = std::io::BufReader::new(f);

            for lib in waveobj::Material::from_reader(f)
                .with_context(err_mtl)?
            {
                if let Some(map) = lib.map() {
                    let err_map = || format!("Error reading texture file {map}");
                    if let Some(map) = waveobj::solve_find_matlib_file(map.as_ref(), &matlib) {
                        let img = image::io::Reader::open(&map)
                            .with_context(err_map)?
                            .with_guessed_format()
                            .with_context(err_map)?
                            .decode()
                            .with_context(err_map)?;
                        let map_name = map.file_name().and_then(|f| f.to_str())
                            .ok_or_else(|| anyhow!("Invalid texture name"))?;
                        texture_map.insert(lib.name().to_owned(), (map_name.to_owned(), img));
                    } else {
                        return Err(anyhow!("{} texture from {} matlib not found", map, matlib.display()));
                    }
                }
            }
        }
        let (model, facemap) = Model::from_waveobj(&obj, texture_map);

        let mut edges = vec![EdgeStatus::Cut(TabSide::False); model.num_edges()];

        for (i_edge, edge_status) in edges.iter_mut().enumerate() {
            let i_edge = EdgeIndex::from(i_edge);
            let edge = &model[i_edge];
            match edge.faces() {
                // Edge from tessellation of a n-gon
                (fa, Some(fb)) if facemap[&fa] == facemap[&fb] => {
                    *edge_status = EdgeStatus::Hidden;
                }
                // Rim
                (_, None) => {
                    // edges in the rim (without adjacent face) do not have a tab
                    *edge_status = EdgeStatus::Cut(TabSide::Hidden)
                }
                // Normal edge
                _ => {}
            }
        }

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

        let mut papercraft = Papercraft {
            model,
            options: PaperOptions::default(),
            edges,
            islands,
            memo: Memoization::default(),
            edge_ids: Vec::new(),
        };
        papercraft.recompute_edge_ids();

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
        let num_pages = papercraft.pack_islands();
        papercraft.options.pages = num_pages;
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

    pub fn import_pepakura(file_name: &Path) -> Result<Papercraft> {
        let f = std::fs::File::open(file_name)?;
        let f = std::io::BufReader::new(f);
        let pdo = pepakura::Pdo::from_reader(f)?;
        //dbg!(&pdo);

        let (model, facemap, edgemap, vertexmap) = Model::from_pepakura(&pdo);

        let mut edges = vec![EdgeStatus::Cut(TabSide::False); model.num_edges()];

        for (i_edge, edge_status) in edges.iter_mut().enumerate() {
            let i_edge = EdgeIndex::from(i_edge);
            let edge = &model[i_edge];
            match edge.faces() {
                // Edge from tessellation of a n-gon
                (fa, Some(fb)) if facemap[&fa] == facemap[&fb] => {
                    *edge_status = EdgeStatus::Hidden;
                }
                // Rim
                (_, None) => {
                    // edges in the rim (without adjacent face) do not have a tab
                    *edge_status = EdgeStatus::Cut(TabSide::Hidden)
                }
                // Normal edge
                _ => {}
            }
        }

        for (i_o, obj) in pdo.objects().iter().enumerate() {
            for edge in obj.edges.iter() {
                let edge_pos = edgemap.iter()
                    .position(|&((o, v0), (_, v1))| {
                        if i_o as u32 != o { return false; }
                        (v0, v1) == (edge.i_v1, edge.i_v2) || (v1, v0) == (edge.i_v1, edge.i_v2)
                    });
                let Some(i_edge) = edge_pos else { continue; };
                if edge.connected {
                    edges[i_edge] = EdgeStatus::Joined;
                } else {
                    let v_f = obj.faces[edge.i_f1 as usize].verts.iter().find(|v_f| v_f.i_v == edge.i_v1).unwrap();
                    edges[i_edge] = EdgeStatus::Cut(if v_f.flap { TabSide::True } else { TabSide::False });
                    //TODO Cut(?)
                }
            }
        }

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
                rot: Rad(0.0),
                mx: Matrix3::one(),
            };
            islands.insert(island);
        }

        let mut options = PaperOptions::default();

        if let Some(unfold) = pdo.unfold() {
            let margin = Vector2::new(pdo.settings().margin_side as f32, pdo.settings().margin_top as f32);
            let page_size = pdo.settings().page_size;
            let area_size = page_size - 2.0 * margin;

            options.scale = unfold.scale;
            options.page_size = (page_size.x, page_size.y);
            options.margin = (margin.y, margin.x, margin.x, margin.y);

            let mut n_cols = 0;
            let mut max_page = (0, 0);
            for island in islands.values_mut() {
                let face = &model[island.root];
                let [i_v0, i_v1, _] = face.index_vertices();
                let (ip_obj, ip_face, ip_v0) = vertexmap[usize::from(i_v0)];
                let (_, _, ip_v1) = vertexmap[usize::from(i_v1)];
                let p_face = &pdo.objects()[ip_obj as usize].faces[ip_face as usize];
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
                island.reset_transformation(island.root, rot, loc);
            }
            // 0-based
            options.page_cols = n_cols + 1;
            options.pages = max_page.0 * options.page_cols + max_page.1 + 1;
        }
        options.hidden_line_angle = pdo.settings().fold_line_hide_angle
            .map(|a| (180 - a) as f32)
            .unwrap_or(0.0);

        let mut papercraft = Papercraft {
            model,
            options,
            edges,
            islands,
            memo: Memoization::default(),
            edge_ids: Vec::new(),
        };

        if pdo.unfold().is_none() {
            papercraft.pack_islands();
        }
        papercraft.recompute_edge_ids();
        Ok(papercraft)
    }
}


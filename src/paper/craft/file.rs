use std::{collections::{HashMap, HashSet}, io::{Read, Seek, Write}, path::Path, ffi::OsStr, hash::Hash};

use cgmath::{One, Rad, Zero};
use gdk_pixbuf::traits::PixbufLoaderExt;
use slotmap::SlotMap;
use crate::waveobj;
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
                let format = gdk_format_from_extension(Path::new(file_name).extension());
                let data = pixbuf.save_to_bufferv(format, &[])?;
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

            let pbl = gdk_pixbuf::PixbufLoader::new();
            pbl.write(&data)?;
            pbl.close()?;
            let img = pbl.pixbuf().ok_or_else(|| anyhow!("Invalid texture image"))?;
            Ok(img)
        })?;
        Ok(papercraft)
    }

    pub fn import_waveobj(file_name: &Path) -> Result<Papercraft> {
        let f = std::fs::File::open(file_name)?;
        let f = std::io::BufReader::new(f);
        let (matlib, obj) = waveobj::Model::from_reader(f)?;

        let mut texture_map = HashMap::new();

        // Textures are read from the .mtl file
        let err_mtl = || format!("Error reading matlib file {matlib}");
        let f = std::fs::File::open(&matlib)
            .with_context(err_mtl)?;
        let f = std::io::BufReader::new(f);

        for lib in waveobj::Material::from_reader(f)
            .with_context(err_mtl)?
        {
            if let Some(map) = lib.map() {
                let err_map = || format!("Error reading texture file {map}");
                let pbl = gdk_pixbuf::PixbufLoader::new();

                let data = std::fs::read(map)
                    .with_context(err_map)?;
                pbl.write(&data)
                    .with_context(err_map)?;
                pbl.close()
                    .with_context(err_map)?;
                let img = pbl.pixbuf().ok_or_else(|| anyhow!(err_map()))?;

                let map_name = Path::new(map).file_name().and_then(|f| f.to_str())
                    .ok_or_else(|| anyhow!("Invalid texture name"))?;
                texture_map.insert(lib.name().to_owned(), (map_name.to_owned(), img));
            }
        }
        let (model, facemap) = Model::from_waveobj(&obj, texture_map);

        let mut edges = vec![EdgeStatus::Cut(false); model.num_edges()];

        for (i_edge, edge_status) in edges.iter_mut().enumerate() {
            let i_edge = EdgeIndex::from(i_edge);
            let edge = &model[i_edge];
            match edge.faces() {
                (fa, Some(fb)) if facemap[&fa] == facemap[&fb] => {
                    *edge_status = EdgeStatus::Hidden;
                }
                _ => {}
            }
        }

        let mut pending_faces: HashSet<FaceIndex> = model.faces().map(|(i_face, _face)| i_face).collect();

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
        };
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

        let mut index_v = HashMap::new();
        let mut index_vn = HashMap::new();
        let mut index_vt = HashMap::new();

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

        let mut done_faces = HashSet::new();
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
                    write!(f, " {}/{}/{}", v, t, n)?;
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
                    full_path_buf.push(&path);
                    &full_path_buf
                } else {
                    path
                };
                let format = gdk_format_from_extension(path.extension());
                let data = pixbuf.save_to_bufferv(format, &[])?;
                std::fs::write(&full_path, &data)?;
            }
        }
        drop(fm);

        Ok(())

    }

}

fn gdk_format_from_extension(ext: Option<&OsStr>) -> &str {
    //TODO: use gdk_pixbuf_format_get_extensions?
    let ext = match ext.and_then(|s| s.to_str()) {
        None => return "png",
        Some(e) => e.to_ascii_lowercase(),
    };
    match ext.as_str() {
        "png" => "png",
        "jpg" | "jpeg" | "jfif" => "jpeg",
        _ => "png",
    }
}

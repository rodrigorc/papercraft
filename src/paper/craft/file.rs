use std::{collections::{HashMap, HashSet}, io::{Read, Seek, Write}, path::Path, ffi::OsStr};

use cgmath::{One, EuclideanSpace, Transform, Rad, Zero};
use gdk_pixbuf::traits::PixbufLoaderExt;
use slotmap::SlotMap;
use crate::{waveobj, util_3d};

use super::*;

impl Papercraft {
    pub fn save<W: Write + Seek>(&self, w: W) -> std::io::Result<()> {
        let mut zip = zip::ZipWriter::new(w);
        let options = zip::write::FileOptions::default();

        zip.start_file("model.json", options)?;
        serde_json::to_writer(&mut zip, self)?;

        for tex in self.model.textures() {
            if let Some(pixbuf) = tex.pixbuf() {
                let file_name = tex.file_name();
                zip.start_file(&format!("tex/{file_name}"), options)?;
                let format = gdk_format_from_extension(Path::new(file_name).extension());
                let data = pixbuf.save_to_bufferv(&format, &[]).unwrap();
                zip.write_all(&data[..])?;
            }
        }
        zip.finish()?;
        Ok(())
    }

    pub fn load<R: Read + Seek>(r: R) -> std::io::Result<Papercraft> {
        let mut zip = zip::ZipArchive::new(r)?;
        let mut zmodel = zip.by_name("model.json")?;
        let mut papercraft: Papercraft = serde_json::from_reader(&mut zmodel)?;
        drop(zmodel);

        papercraft.model.reload_textures(|file_name| {
            let mut ztex = zip.by_name(&format!("tex/{file_name}")).ok()?;
            let mut data = Vec::new();
            ztex.read_to_end(&mut data).ok()?;

            let pbl = gdk_pixbuf::PixbufLoader::new();
            pbl.write(&data).ok().unwrap();
            pbl.close().ok().unwrap();
            let img = pbl.pixbuf().unwrap();
            Some(img)
        });
        Ok(papercraft)
    }

    pub fn import_waveobj(file_name: impl AsRef<Path>) -> Papercraft {
        let f = std::fs::File::open(file_name).unwrap();
        let f = std::io::BufReader::new(f);
        let (matlib, obj) = waveobj::Model::from_reader(f).unwrap();

        let mut texture_map = HashMap::new();

        // Textures are read from the .mtl file
        let f = std::fs::File::open(matlib).unwrap();
        let f = std::io::BufReader::new(f);

        for lib in waveobj::Material::from_reader(f).unwrap()  {
            if let Some(map) = lib.map() {
                let pbl = gdk_pixbuf::PixbufLoader::new();
                let data = std::fs::read(map).unwrap();
                pbl.write(&data).ok().unwrap();
                pbl.close().ok().unwrap();
                let img = pbl.pixbuf().unwrap();
                //dbg!(img.width(), img.height(), img.rowstride(), img.bits_per_sample(), img.n_channels());
                let map_name = Path::new(map).file_name().unwrap().to_str().unwrap();
                texture_map.insert(lib.name().to_owned(), (map_name.to_owned(), img));
            }
        }
        let (mut model, facemap) = Model::from_waveobj(&obj, texture_map);

        // Compute the bounding box, then move to the center and scale to a standard size
        let (v_min, v_max) = util_3d::bounding_box_3d(
            model
                .vertices()
                .map(|v| v.pos())
        );
        let size = (v_max.x - v_min.x).max(v_max.y - v_min.y).max(v_max.z - v_min.z);
        let mscale = Matrix4::from_scale(1.0 / size);
        let center = (v_min + v_max) / 2.0;
        let mcenter = Matrix4::from_translation(-center);
        let m = mscale * mcenter;

        model.transform_vertices(|pos, _normal| {
            //only scale and translate, no need to touch normals
            *pos = m.transform_point(Point3::from_vec(*pos)).to_vec();
        });

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
            options: Options::default(),
            edges,
            islands,
        };
        papercraft.pack_islands();
        papercraft
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
use std::{collections::{HashMap, HashSet}, ops::ControlFlow, io::{Read, Seek, Write}, path::Path};

use cgmath::{One, EuclideanSpace, Transform, Rad, Zero};
use gdk_pixbuf::traits::PixbufLoaderExt;
use slotmap::SlotMap;

use super::*;

impl Papercraft {
    pub fn from_model(model: Model, facemap: &HashMap<FaceIndex, u32>) -> Papercraft {
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

        let mut row_height = 0.0f32;
        let mut pos_x = 0.0;
        let mut pos_y = 0.0;

        let mut pending_faces: HashSet<FaceIndex> = model.faces().map(|(i_face, _face)| i_face).collect();
        let scale = 100.0;

        let mut islands = SlotMap::with_key();
        while let Some(root) = pending_faces.iter().copied().next() {
            pending_faces.remove(&root);

            //Compute the bounding box of the flat face, since Self is not yet build, we have to use the traverse_faces_ex() version directly
            let mut vx = Vec::new();
            traverse_faces_ex(&model, root, Matrix3::one(), craft::NormalTraverseFace(&model, &edges, scale),
                |i_face, face, mx| {
                    pending_faces.remove(&i_face);
                    let normal = face.plane(&model, scale);
                    vx.extend(face.index_vertices().map(|v| {
                        mx.transform_point(Point2::from_vec(normal.project(&model[v].pos()))).to_vec()
                    }));
                    ControlFlow::Continue(())
                }
            );

            let bbox = bounding_box_2d(vx);
            let pos = Vector2::new(pos_x - bbox.0.x, pos_y - bbox.0.y);
            pos_x += bbox.1.x - bbox.0.x + 5.0;
            row_height = row_height.max(bbox.1.y - bbox.0.y);

            if pos_x > 210.0 {
                pos_y += row_height + 5.0;
                row_height = 0.0;
                pos_x = 0.0;
            }

            let mut island = Island {
                root,
                loc: pos,
                rot: Rad::zero(),
                mx: Matrix3::one(),
            };
            island.recompute_matrix();
            islands.insert(island);
        }

        Papercraft {
            model,
            scale,
            edges,
            islands,
        }
    }

    pub fn save<W: Write + Seek>(&self, w: W) -> std::io::Result<()> {
        let mut zip = zip::ZipWriter::new(w);
        let options = zip::write::FileOptions::default();

        zip.start_file("model.json", options)?;
        serde_json::to_writer(&mut zip, self)?;

        for tex in self.model.textures() {
            if let Some(pixbuf) = tex.pixbuf() {
                let file_name = tex.file_name();
                zip.start_file(&format!("tex/{file_name}"), options)?;
                let ext = Path::new(file_name).extension().and_then(|s| s.to_str()).unwrap_or("png").to_ascii_lowercase();
                let data = pixbuf.save_to_bufferv(&ext, &[]).unwrap();
                zip.write_all(&mut &data[..])?;
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
}
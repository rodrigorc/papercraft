use std::io::{Read, Seek, Write};

use super::*;
use anyhow::Result;
use formats::Importer;

impl Papercraft {
    pub fn save<W: Write + Seek>(&self, w: W, thumbnail: Option<image::RgbaImage>) -> Result<()> {
        let mut zip = zip::ZipWriter::new(w);
        let options = zip::write::SimpleFileOptions::default();

        zip.start_file("model.json", options)?;
        serde_json::to_writer(&mut zip, self)?;

        for tex in self.model.textures() {
            if let Some(pixbuf) = tex.pixbuf() {
                let file_name = tex.file_name();
                zip.start_file(format!("tex/{file_name}"), options)?;
                let mut data = Vec::new();
                let format =
                    image::ImageFormat::from_path(file_name).unwrap_or(image::ImageFormat::Png);
                pixbuf.write_to(&mut std::io::Cursor::new(&mut data), format)?;
                zip.write_all(&data[..])?;
            }
        }

        if let Some(thumbnail) = thumbnail {
            zip.start_file("thumb.png", options)?;
            let mut data = Vec::new();
            thumbnail.write_to(
                &mut std::io::Cursor::new(&mut data),
                image::ImageFormat::Png,
            )?;
            zip.write_all(&data[..])?;
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
            let Ok(mut ztex) = zip.by_name(&format!("tex/{file_name}")) else {
                return Ok(None);
            };
            let mut data = Vec::new();
            ztex.read_to_end(&mut data)?;
            let img = image::ImageReader::new(std::io::Cursor::new(&data))
                .with_guessed_format()?
                .decode()?;
            Ok(Some(img))
        })?;

        papercraft.post_create();
        Ok(papercraft)
    }
    pub fn post_create(&mut self) {
        self.sanitize();
        self.recompute_edge_ids();
    }
    pub fn sanitize(&mut self) {
        // Fix islands that are not acyclic graphs
        struct SanitizeTraverse<'a, 'b>(&'a Papercraft, &'b mut FxHashSet<EdgeIndex>);

        impl TraverseFacePolicy for SanitizeTraverse<'_, '_> {
            type State = ();
            fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
                match self.0.edges[usize::from(i_edge)] {
                    RealEdgeStatus::Cut(_) => false,
                    RealEdgeStatus::Joined | RealEdgeStatus::Hidden => true,
                }
            }
            fn duplicated_face(
                &mut self,
                _i_face: FaceIndex,
                i_edge: EdgeIndex,
                _i_next_face: FaceIndex,
            ) {
                self.1.insert(i_edge);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for (_i_island, island) in &self.islands {
                let mut faults = FxHashSet::default();
                let _ = traverse_faces_ex(
                    &self.model,
                    island.root,
                    (),
                    SanitizeTraverse(self, &mut faults),
                    |_, _, _| ControlFlow::Continue(()),
                );
                if !faults.is_empty() {
                    for i_edge in faults {
                        log::warn!("Splitting looping edge {i_edge:?}");
                        self.edges[usize::from(i_edge)] = RealEdgeStatus::Cut(FlapSide::False);
                    }
                    changed = true;
                }
            }
        }

        // Fix any face that appears in two islands
        let mut changed = true;
        while changed {
            changed = false;
            for (i_island, island) in &self.islands {
                let i_owner = self.island_by_face(island.root);
                if i_owner != i_island {
                    log::warn!(
                        "Removing face in two pieces: {:?} was in {:?} and {:?}",
                        island.root,
                        i_island,
                        i_owner
                    );
                    self.islands.remove(i_island);
                    self.memo = Memoization::default();
                    changed = true;
                    break; // restart loop
                }
            }
        }

        // Add islands for any orphan face
        let mut changed = true;
        while changed {
            changed = false;
            let mut all_faces: FxHashSet<FaceIndex> = self.model.faces().map(|(i, _)| i).collect();
            for (_i_island, island) in &self.islands {
                let _ = self.traverse_faces_no_matrix(island, |i_face| {
                    all_faces.remove(&i_face);
                    ControlFlow::Continue(())
                });
            }
            // Create just one island, just in case it has some connected
            if let Some(&root) = all_faces.iter().next() {
                log::warn!("Creating missing island for face {root:?}");
                // Any coordinates are good enough, we are on emergency mode
                self.islands.insert(Island {
                    root,
                    loc: Vector2::zero(),
                    rot: Rad::zero(),
                    mx: Matrix3::one(),
                    name: String::new(),
                });
                self.memo = Memoization::default();
                changed = true;
            }
        }

        for (i_edge, edge) in self.model.edges() {
            let (_, f1) = edge.faces();
            // Rim edges can't have any status
            if f1.is_none() {
                match &mut self.edges[usize::from(i_edge)] {
                    RealEdgeStatus::Cut(FlapSide::Hidden)
                    | RealEdgeStatus::Cut(FlapSide::False) => (), // ok
                    x => {
                        log::warn!("Fix rim edge {i_edge:?}");
                        *x = RealEdgeStatus::Cut(FlapSide::Hidden);
                    }
                }
            }
        }
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

        edge_collection.sort_by_key(|(c, _, _, _)| (TotalF32(c.y), TotalF32(c.z), TotalF32(c.x)));

        for (_, edge, edge_status, edge_id) in edge_collection {
            match (edge.faces(), edge_status) {
                // edges from tessellations or rims don't have ids
                (_, RealEdgeStatus::Hidden) | ((_, None), _) => {}
                _ => {
                    next_edge_id += 1;
                    *edge_id = Some(EdgeId::new(next_edge_id));
                }
            }
        }
        self.edge_ids = edge_ids;
    }

    pub fn import<I: Importer>(mut importer: I) -> Papercraft {
        let ImportedModule {
            model,
            face_map,
            edge_map,
        } = Model::from_importer(&mut importer);

        let edges: Vec<_> = edge_map
            .iter()
            .enumerate()
            .map(|(i_edge, edge_id)| {
                let i_edge = EdgeIndex::from(i_edge);
                let edge = &model[i_edge];
                match edge.faces() {
                    // Edge from tessellation of a n-gon
                    (fa, Some(fb)) if face_map[usize::from(fa)] == face_map[usize::from(fb)] => {
                        RealEdgeStatus::Hidden
                    }
                    // Rim
                    (_, None) => {
                        // edges in the rim (without adjacent face) usually do not have a flap, but if it does,
                        // the FlapSide is false, no matter what the loader says
                        match importer.compute_edge_status(*edge_id) {
                            Some(RealEdgeStatus::Cut(FlapSide::False | FlapSide::True)) => {
                                RealEdgeStatus::Cut(FlapSide::False)
                            }
                            _ => RealEdgeStatus::Cut(FlapSide::Hidden),
                        }
                    }
                    // Normal edge
                    _ => importer.compute_edge_status(*edge_id).unwrap_or_else(|| {
                        // If the importer doesn't compute the edge_status, try the heuristics, a small angle is Hidden, a big angle is Cut.
                        if edge.angle().0.abs() < Rad::from(Deg(1.0)).0 {
                            RealEdgeStatus::Hidden
                        } else {
                            RealEdgeStatus::Cut(FlapSide::False)
                        }
                    }),
                }
            })
            .collect();

        let mut pending_faces: FxHashSet<FaceIndex> =
            model.faces().map(|(i_face, _face)| i_face).collect();

        let mut islands = SlotMap::with_key();
        while let Some(root) = pending_faces.iter().copied().next() {
            pending_faces.remove(&root);

            let _ = traverse_faces_ex(
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
        papercraft.post_create();
        papercraft
    }
}

use super::*;

fn compute_edge_map(new: &Papercraft, old: &Papercraft) -> FxHashMap<EdgeIndex, (EdgeIndex, bool)> {
    use rayon::prelude::*;

    let model = &new.model;
    let omodel = &old.model;
    let n_edges = model.num_edges();

    // Brute force algorithm, probably it could be made much smarter, but it
    // is easier to invoke the Rayon superpowers.
    // It is not a function called so frequently.
    (0..n_edges)
        .into_par_iter()
        .map(EdgeIndex::from)
        .filter_map(|i_new| {
            let e_new = &model[i_new];
            let (np0, np1) = model.edge_pos(e_new);
            let distance = |e_old: &Edge| {
                let (op0, op1) = omodel.edge_pos(e_old);
                let da = op0.distance2(np0) + op1.distance2(np1);
                let db = op0.distance2(np1) + op1.distance2(np0);
                (da, db)
            };

            // f32 is not Eq so min_by_key cannot be used directly
            let best = omodel.edges().min_by(|&(_, e_old_1), &(_, e_old_2)| {
                let (da1, db1) = distance(e_old_1);
                let (da2, db2) = distance(e_old_2);
                let d1 = da1.min(db1);
                let d2 = da2.min(db2);
                d1.total_cmp(&d2)
            });

            let (i_old, e_old) = best?;
            let (da, db) = distance(e_old);
            let crossed = da > db;
            Some((i_new, (i_old, crossed)))
        })
        .collect()
}

type IslandFaceMap = FxHashMap<IslandKey, FxHashSet<FaceIndex>>;

fn compute_island_to_faces_map(pc: &Papercraft) -> IslandFaceMap {
    let mut new_faces_map = FxHashMap::default();
    for (i_new, island) in pc.islands() {
        let mut faces = FxHashSet::default();
        pc.traverse_faces_no_matrix(island, |f| {
            faces.insert(f);
            ControlFlow::Continue(())
        });
        new_faces_map.insert(i_new, faces);
    }
    new_faces_map
}

fn compute_island_map(
    new: &Papercraft,
    old: &Papercraft,
    new_map: &IslandFaceMap,
    old_map: &IslandFaceMap,
) -> FxHashMap<IslandKey, IslandKey> {
    let mut map = FxHashMap::default();
    for (i_island, _) in new.islands() {
        let new_faces = &new_map[&i_island];
        let best = old.islands().max_by_key(|&(i_oisland, _)| {
            let old_faces = &old_map[&i_oisland];
            new_faces.intersection(old_faces).count()
        });
        if let Some((i_oisland, _)) = best {
            map.insert(i_island, i_oisland);
        }
    }
    map
}

impl Papercraft {
    pub fn update_from_obj(&mut self, old_obj: &Papercraft) {
        self.options = old_obj.options.clone();
        // Options are changed, discard memo
        self.memo = Memoization::default();

        // Check which edges are nearest, checking the distance between their vertices
        let eno_map = compute_edge_map(self, old_obj);
        let eon_map = compute_edge_map(old_obj, self);

        // If the best match for A is B and for B is A, then it is a match
        let mut real_edge_map = FxHashMap::default();
        let mut edge_status_map = FxHashMap::default();
        for (i_edge, _) in self.model.edges() {
            // If an edge matches in both directions, then it is a match
            let (o, o_cross) = match eno_map.get(&i_edge) {
                None => continue,
                Some(o) => *o,
            };
            let (i, _) = match eon_map.get(&o) {
                None => continue,
                Some(i) => *i,
            };
            if i_edge != i {
                continue;
            }
            real_edge_map.insert(i, o);

            let o_status = old_obj.edge_status(o);
            let i_status = self.edge_status(i);
            if i_status != EdgeStatus::Hidden && o_status != EdgeStatus::Hidden {
                edge_status_map.insert(i_edge, (o_status, o_cross));
            }
        }

        //Apply the old status to the new model
        for (i_edge, (status, crossed)) in edge_status_map {
            // Is it a rim?
            if self.model[i_edge].faces().1.is_none() {
                // Rims can't be crossed, so ignore that
                self.edges[usize::from(i_edge)] = match status {
                    EdgeStatus::Cut(FlapSide::Hidden) | EdgeStatus::Cut(FlapSide::False) => status,
                    _ => EdgeStatus::Cut(FlapSide::Hidden),
                };
            } else {
                match status {
                    EdgeStatus::Hidden => { /* should not happen, because they are filtered above */
                    }
                    EdgeStatus::Joined => {
                        self.edge_join(i_edge, None);
                    }
                    EdgeStatus::Cut(c) => {
                        self.edge_cut(i_edge, None);
                        if let EdgeStatus::Cut(_) = self.edge_status(i_edge) {
                            let c = if crossed { c.opposite() } else { c };
                            self.edges[usize::from(i_edge)] = EdgeStatus::Cut(c);
                        }
                    }
                }
            }
        }

        self.memo = Memoization::default();

        // Match the faces: two faces are equivalent if their 3 edges match
        let mut oi_real_face_map = FxHashMap::default();
        let mut rotations = Vec::new();
        for (i_face, face) in self.model.faces() {
            let i_edges = face.index_edges();
            let o_edges = i_edges.map(|i| real_edge_map.get(&i));
            let o_edges = match o_edges {
                [Some(a), Some(b), Some(c)] => [*a, *b, *c],
                _ => continue,
            };

            // Instead of looking for the o_face everywhere, look just on the faces of one
            // of the o_edges, they are at most 2 faces.
            let o_faces = old_obj.model[o_edges[0]].faces();
            let o_faces = std::iter::once(o_faces.0).chain(o_faces.1);
            //let o_edges_set = o_edges.into_iter().collect::<FxHashSet<_>>();
            for o_face in o_faces {
                let oface = &old_obj.model[o_face];
                let real_edges = oface.index_edges();
                if real_edges == o_edges {
                    //no rotation
                } else if real_edges[0] == o_edges[1]
                    && real_edges[1] == o_edges[2]
                    && real_edges[2] == o_edges[0]
                {
                    rotations.push((i_face, -1));
                } else if real_edges[0] == o_edges[2]
                    && real_edges[1] == o_edges[0]
                    && real_edges[2] == o_edges[1]
                {
                    rotations.push((i_face, 1));
                } else {
                    // no match
                    continue;
                };
                // match!
                oi_real_face_map.insert(o_face, i_face);
            }
        }
        // Fix the order of edges inside the faces
        for (o_face, rotation) in rotations {
            self.model.rotate_face_vertices(o_face, rotation);
        }
        self.memo = Memoization::default();

        // Match the islands: A maps to B if B is the target island with most common faces.
        let new_islands = compute_island_to_faces_map(self);
        let mut old_islands = compute_island_to_faces_map(old_obj);

        // old_islands uses FaceIndex from the old model, while new_islands uses FaceIndex from the new model, so they cannot compare directly.
        // Here we switch old_island to use the FaceIndex from the new model, if available.
        for (_, o_faces) in old_islands.iter_mut() {
            let i_faces = o_faces
                .iter()
                .filter_map(|o| oi_real_face_map.get(o))
                .copied()
                .collect();
            *o_faces = i_faces;
        }

        // If island A maps to B and B maps to A, then it is a match.
        let mut real_island_map = FxHashMap::default();
        let ino_map = compute_island_map(self, old_obj, &new_islands, &old_islands);
        let ion_map = compute_island_map(old_obj, self, &old_islands, &new_islands);

        for (i_island, _) in self.islands() {
            let o = match ino_map.get(&i_island) {
                None => continue,
                Some(o) => *o,
            };
            let i = match ion_map.get(&o) {
                None => continue,
                Some(i) => *i,
            };
            if i != i_island {
                continue;
            }
            real_island_map.insert(i, o);
        }

        // Apply the same transformation to corresponding islands
        let mut new_island_pos: FxHashMap<IslandKey, _> =
            self.islands.iter().map(|(i, _)| (i, None)).collect();

        for (i, island) in &self.islands {
            let oisland = real_island_map
                .get(&i)
                .map(|o| old_obj.island_by_key(*o).unwrap());

            // If the island is not matches, use any face that is matched as a second best option
            let oisland = oisland.or_else(|| {
                let new_faces = new_islands.get(&i).unwrap();
                for i_f in new_faces {
                    if let Some((o, _)) = oi_real_face_map.iter().find(|&(_, i)| i == i_f) {
                        let oo = old_obj.island_by_face(*o);
                        return old_obj.island_by_key(oo);
                    }
                }
                None
            });

            let Some(oisland) = oisland else { continue };

            // If the root face has a direct map, use that; if not...
            let (iroot, rot, loc) = match oi_real_face_map.get(&oisland.root_face()) {
                Some(&i_f) if self.contains_face(island, i_f) => {
                    (i_f, oisland.rotation(), oisland.location())
                }
                _ => {
                    // Look for any other face that is in both islands, and if found,
                    // do some math to fit in the equivalent place
                    let mut res = None;
                    old_obj.traverse_faces(oisland, |o_f, _f, m| {
                        match oi_real_face_map.get(&o_f) {
                            Some(&i_f) if self.contains_face(island, i_f) => {
                                res = Some((i_f, Rad::atan2(m.x.y, m.x.x), m.z.truncate()));
                                ControlFlow::Break(())
                            }
                            _ => ControlFlow::Continue(()),
                        }
                    });
                    if res.is_none() {
                        println!("ffff");
                    }
                    res.unwrap_or((island.root_face(), oisland.rotation(), oisland.location()))
                }
            };
            new_island_pos.insert(i, Some((iroot, rot, loc)));
        }
        for (i_island, maybe_pos) in new_island_pos {
            let island = self.islands.get_mut(i_island).unwrap();
            match maybe_pos {
                Some((iroot, rot, loc)) => {
                    island.reset_transformation(iroot, rot, loc);
                }
                None => {
                    // If the island doesn't have a mapping, dump it into the page -1.
                    let mut page_offs = self.options.global_to_page(island.loc);
                    page_offs.row = 0;
                    page_offs.col = -1;
                    let loc = self.options.page_to_global(page_offs);
                    island.reset_transformation(island.root_face(), island.rotation(), loc);
                }
            }
        }
        self.memo = Memoization::default();

        // Mixing two sane things may create something insane, fix it now
        self.sanitize();
    }
}

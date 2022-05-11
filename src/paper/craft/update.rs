use super::*;

fn compute_edge_map(new: &Papercraft, old: &Papercraft) -> HashMap<EdgeIndex, (EdgeIndex, bool)> {
    let mut map = HashMap::new();
    for (i_new, e_new) in new.model.edges() {
        let np0 = new.model[e_new.v0()].pos();
        let np1 = new.model[e_new.v1()].pos();
        let distance = |e_old: &Edge| {
            let op0 = old.model[e_old.v0()].pos();
            let op1 = old.model[e_old.v1()].pos();
            let da = op0.distance2(np0) + op1.distance2(np1);
            let db = op0.distance2(np1) + op1.distance2(np0);
            (da, db)
        };
        // f32 is not Eq so min_by_key cannot be used directly
        let best = old.model.edges().min_by(|&(_, e_old_1), &(_, e_old_2)| {
            let (da1, db1) = distance(e_old_1);
            let (da2, db2) = distance(e_old_2);
            let d1 = da1.min(db1);
            let d2 = da2.min(db2);
            d1.partial_cmp(&d2).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some((i_old, e_old)) = best {
            let (da, db) = distance(e_old);
            let crossed = da > db;
            map.insert(i_new, (i_old, crossed));
        }
    }
    map
}

type IslandFaceMap = HashMap<IslandKey, HashSet<FaceIndex>>;

fn compute_island_to_faces_map(pc: &Papercraft) -> IslandFaceMap {
    let mut new_faces_map = HashMap::new();
    for (i_new, island) in pc.islands() {
        let mut faces = HashSet::new();
        pc.traverse_faces_no_matrix(island, |f| {
            faces.insert(f);
            ControlFlow::Continue(())
        });
        new_faces_map.insert(i_new, faces);
    }
    new_faces_map
}

fn compute_island_map(new: &Papercraft, old: &Papercraft, new_map: &IslandFaceMap, old_map: &IslandFaceMap) -> HashMap<IslandKey, IslandKey> {
    let mut map = HashMap::new();
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

        // Check which edges are nearest, checking the distance between their vertices
        let eno_map = compute_edge_map(self, old_obj);
        let eon_map = compute_edge_map(old_obj, self);

        // If the best match for A is B and for B is A, then it is a match
        let mut real_edge_map = HashMap::new();
        let mut edge_status_map = HashMap::new();
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
            if i_status != EdgeStatus::Hidden &&
               o_status != EdgeStatus::Hidden {
                edge_status_map.insert(i_edge, (o_status, o_cross));
            }
        }

        //Apply the old status to the new model
        for (i_edge, (status, crossed)) in edge_status_map {
            match status {
                EdgeStatus::Hidden => { /* should not happen */},
                EdgeStatus::Joined => {
                    self.edge_join(i_edge, None);
                }
                EdgeStatus::Cut(c) => {
                    // This  should not be needed, because when a model is imported all edges are cut by default, but just in case
                    self.edge_cut(i_edge, None);
                    if let EdgeStatus::Cut(new_c) = self.edge_status(i_edge) {
                        if !crossed && c != new_c || crossed && c == new_c {
                            self.edge_toggle_tab(i_edge);
                        }
                    }
                }
            }
        }

        // Match the faces: two faces are equivalent if their 3 edges match
        let mut oi_real_face_map = HashMap::new();
        for (i_face, face) in self.model.faces() {
            let i_edges = face.index_edges();
            let o_edges = i_edges.map(|i| real_edge_map.get(&i));
            let o_edges = match o_edges {
                [Some(a), Some(b), Some(c)] => [*a, *b, *c],
                _ => continue,
            };

            let o_faces = old_obj.model[o_edges[0]].faces();
            let o_edges_set = o_edges.into_iter().collect::<HashSet<_>>();
            let o_faces = std::iter::once(o_faces.0).chain(o_faces.1);
            for o_face in o_faces {
                let oface = &old_obj.model[o_face];
                let real_edges = oface.index_edges();
                if real_edges.into_iter().collect::<HashSet<_>>() == o_edges_set {
                    oi_real_face_map.insert(o_face, i_face);
                    break;
                }
            }
        }

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

        // If face A maps to B and B maps to A, then it is a match.
        let mut real_island_map = HashMap::new();
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
        for (&i, &o) in &real_island_map {
            let island = self.island_by_key(i).unwrap();
            let oisland = old_obj.island_by_key(o).unwrap();
            // If the root face has a direct map, use that; if not, keep the same root
            let iroot = match oi_real_face_map.get(&oisland.root_face()) {
                Some(&f) if self.contains_face(island, f) => f,
                _ => island.root_face(),
            };

            let island = self.island_by_key_mut(i).unwrap();
            island.reset_transformation(iroot, oisland.rotation(), oisland.location());
        }
    }
}
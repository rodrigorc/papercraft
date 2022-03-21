use std::{collections::{HashSet, HashMap}, ops::ControlFlow};

use cgmath::{prelude::*, Transform, EuclideanSpace, InnerSpace, Rad};
use slotmap::{SlotMap, new_key_type};

use crate::{paper::*, util_3d::*};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum EdgeStatus {
    Joined,
    Cut,
    Hidden,
}
new_key_type! {
    pub struct IslandKey;
}

pub struct Papercraft {
    edges: Vec<EdgeStatus>, //parallel to EdgeIndex
    islands: SlotMap<IslandKey, Island>,
}

impl Papercraft {
    pub fn new(model: &Model, facemap: &HashMap<FaceIndex, u32>) -> Papercraft {
        let mut edges = vec![EdgeStatus::Cut; model.num_edges()];

        for (i_edge, edge_status) in edges.iter_mut().enumerate() {
            let i_edge = EdgeIndex::from(i_edge);
            let edge = &model[i_edge];
            let mut faces = edge.faces();
            let first = faces.next().unwrap();
            let original = facemap[&first];
            let flat = faces.all(|f| facemap[&f] == original);
            if flat {
                *edge_status = EdgeStatus::Hidden;
            }
        }

        let mut row_height = 0.0f32;
        let mut pos_x = 0.0;
        let mut pos_y = 0.0;

        let mut pending_faces: HashSet<FaceIndex> = model.faces().map(|(i_face, _face)| i_face).collect();

        let mut islands = SlotMap::with_key();
        while let Some(root) = pending_faces.iter().copied().next() {
            pending_faces.remove(&root);
            let mut vx = Vec::new();
            Self::traverse_faces_ex(model, root, Matrix3::one(), |i_face, face, mx| {
                pending_faces.remove(&i_face);
                let normal = face.normal(model);
                vx.extend(face.index_vertices().map(|v| {
                    mx.transform_point(Point2::from_vec(normal.project(&model[v].pos()))).to_vec()
                }));
                ControlFlow::Continue(())
            },
            |i_edge| {
                edges[usize::from(i_edge)] != EdgeStatus::Cut
            });

            let bbox = bounding_box_2d(vx);
            let pos = Vector2::new(pos_x - bbox.0.x, pos_y - bbox.0.y);
            pos_x += bbox.1.x - bbox.0.x + 0.05;
            row_height = row_height.max(bbox.1.y - bbox.0.y);

            if pos_x > 2.0 {
                pos_y += row_height + 0.05;
                row_height = 0.0;
                pos_x = 0.0;
            }

            let island = Island {
                mx: Matrix3::from_translation(pos),
                root,
            };
            islands.insert(island);
        }

        Papercraft {
            edges,
            islands,
        }
    }

    pub fn islands(&self) -> impl Iterator<Item = (IslandKey, &Island)> + '_ {
        self.islands.iter()
    }

    pub fn island_by_face(&self, model: &Model, i_face: FaceIndex) -> IslandKey {
        for (i_island, island) in &self.islands {
            if self.contains_face(model, &island, i_face) {
                return i_island;
            }
        }
        panic!("Island not found");
    }
    pub fn island_by_key(&self, key: IslandKey) -> Option<&Island> {
        self.islands.get(key)
    }
    pub fn island_by_key_mut(&mut self, key: IslandKey) -> Option<&mut Island> {
        self.islands.get_mut(key)
    }

    pub fn edge_status(&self, edge: EdgeIndex) -> EdgeStatus {
        self.edges[usize::from(edge)]
    }

    pub fn edge_toggle(&mut self, model: &Model, i_edge: EdgeIndex, priority_face: Option<FaceIndex>) {
        let edge = &model[i_edge];
        let faces: Vec<_> = edge.faces().collect();

        let (i_face_a, i_face_b) = match &faces[..] {
            &[a, b] => (a, b),
            _ => return,
        };

        let edge_status = self.edges[usize::from(i_edge)];

        match edge_status {
            EdgeStatus::Joined => {
                //one of the edge faces will be the root of the new island, but we do not know which one, yet
                let i_island = self.island_by_face(model, i_face_a);

                self.edges[usize::from(i_edge)] = EdgeStatus::Cut;

                let mut data_found = None;
                self.traverse_faces(model, &self.islands[i_island],
                    |i_face, _, fmx| {
                        if i_face == i_face_a {
                            data_found = Some((*fmx, i_face_b, i_face_a));
                        } else if i_face == i_face_b {
                            data_found = Some((*fmx, i_face_a, i_face_b));
                        }
                        ControlFlow::Continue(())
                    }
                );
                let (face_mx, new_root, i_face_old) = data_found.unwrap();

                let medge = model.face_to_face_edge_matrix(edge, &model[i_face_old], &model[new_root]);
                let mx = face_mx * medge;

                let mut new_island = Island {
                    mx,
                    root: new_root,
                };

                //Compute the offset
                let sign = if edge.face_sign(new_root) { 1.0 } else { -1.0 };
                let new_root = &model[new_root];
                let new_root_plane = new_root.normal(model);
                let v0 = new_root_plane.project(&model[edge.v0()].pos());
                let v1 = new_root_plane.project(&model[edge.v1()].pos());
                let v0 = mx.transform_point(Point2::from_vec(v0)).to_vec();
                let v1 = mx.transform_point(Point2::from_vec(v1)).to_vec();
                let v = (v1 - v0).normalize_to(0.05);

                //priority_face makes no sense when doing a split, so pass None here unconditionally
                if self.compare_islands(model, &self.islands[i_island], &new_island, None) {
                    let offs = Matrix3::from_translation(-sign * Vector2::new(-v.y, v.x));
                    let island = &mut self.islands[i_island];
                    island.mx = offs * island.mx;
                } else {
                    let offs = Matrix3::from_translation(sign * Vector2::new(-v.y, v.x));
                    new_island.mx = offs * new_island.mx;
                }
                self.islands.insert(new_island);
            }
            EdgeStatus::Cut => {
                let i_island_b = self.island_by_face(model, i_face_b);
                if self.contains_face(model, &self.islands[i_island_b], i_face_a) {
                    // Same island on both sides, nothing to do
                } else {
                    // Join both islands
                    let mut island_b = self.islands.remove(i_island_b).unwrap();
                    let i_island_a = self.island_by_face(model, i_face_a);

                    // Keep position of a or b?
                    if self.compare_islands(model, &self.islands[i_island_a], &island_b, priority_face) {
                        std::mem::swap(&mut self.islands[i_island_a], &mut island_b);
                    }

                    self.edges[usize::from(i_edge)] = EdgeStatus::Joined;
                }
            }
            EdgeStatus::Hidden => {}
        };
    }

    fn compare_islands(&self, model: &Model, a: &Island, b: &Island, priority_face: Option<FaceIndex>) -> bool {
        if let Some(f) = priority_face {
            if self.contains_face(model, a, f) {
                return false;
            }
            if self.contains_face(model, b, f) {
                return true;
            }
        }
        let (mut weight_a, mut weight_b) = (0, 0);
        self.traverse_faces(model, a, |_,_,_| { weight_a += 1; ControlFlow::Continue(()) });
        self.traverse_faces(model, b, |_,_,_| { weight_b += 1; ControlFlow::Continue(()) });
        weight_b > weight_a
    }

    pub fn contains_face(&self, model: &Model, island: &Island, face: FaceIndex) -> bool {
        let mut found = false;
        self.traverse_faces(model, island,
            |i_face, _, _|
                if i_face == face {
                    found = true;
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            );
        found
    }

    fn traverse_faces_ex<F, E>(model: &Model, root: FaceIndex, mx_root: Matrix3, mut visit_face: F, mut cross_edge: E) -> ControlFlow<()>
        where F: FnMut(FaceIndex, &Face, &Matrix3) -> ControlFlow<()>,
              E: FnMut(EdgeIndex) -> bool,
    {
        let mut visited_faces = HashSet::new();
        let mut stack = vec![(root, mx_root)];
        visited_faces.insert(root);

        while let Some((i_face, m)) = stack.pop() {
            let face = &model[i_face];
            visit_face(i_face, face, &m)?;
            for i_edge in face.index_edges() {
                if !cross_edge(i_edge) {
                    continue;
                }
                let edge = &model[i_edge];
                for i_next_face in edge.faces() {
                    if visited_faces.contains(&i_next_face) {
                        continue;
                    }

                    let next_face = &model[i_next_face];
                    let medge = model.face_to_face_edge_matrix(edge, face, next_face);

                    stack.push((i_next_face, m * medge));
                    visited_faces.insert(i_next_face);
                }
            }
        };
        ControlFlow::Continue(())
    }
    pub fn traverse_faces<F>(&self, model: &Model, island: &Island, visit_face: F) -> ControlFlow<()>
        where F: FnMut(FaceIndex, &Face, &Matrix3) -> ControlFlow<()>
    {
        Self::traverse_faces_ex(model, island.root_face(), island.matrix(),
            visit_face,
            |i_edge| {
                match self.edge_status(i_edge) {
                    EdgeStatus::Cut => false,
                    EdgeStatus::Joined |
                    EdgeStatus::Hidden => true,
                }
            })
    }
}

pub struct Island {
    mx: Matrix3,
    root: FaceIndex,
}

impl Island {
    pub fn root_face(&self) -> FaceIndex {
        self.root
    }
    pub fn matrix(&self) -> Matrix3 {
        self.mx
    }
    pub fn translate(&mut self, delta: Vector2) {
        self.mx = Matrix3::from_translation(delta) * self.mx;
    }
    pub fn rotate(&mut self, angle: impl Into<Rad<f32>>) {
        self.mx = self.mx * Matrix3::from(cgmath::Matrix2::from_angle(angle));
    }
}

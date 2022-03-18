use std::{collections::HashSet, ops::ControlFlow};

use cgmath::{Transform, EuclideanSpace, InnerSpace};

use crate::{paper::*, util_3d::*};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum EdgeStatus {
    Joined,
    Cut,
}

pub struct Papercraft {
    edges: Vec<EdgeStatus>, //parallel to EdgeIndex
    islands: Vec<Island>,
}

impl Papercraft {
    pub fn new(model: &Model) -> Papercraft {
        let edges = vec![EdgeStatus::Cut; model.num_edges()];
        let mut row_height = 0.0f32;
        let mut pos_x = 0.0;
        let mut pos_y = 0.0;
        let islands = model.faces()
            .map(|(i_face, _)| {
                let face = &model[i_face];
                let bbox = model.bounding_box(face);
                let pos = Vector2::new(pos_x - bbox.0.x, pos_y - bbox.0.y);
                pos_x += bbox.1.x - bbox.0.x + 0.05;
                row_height = row_height.max(bbox.1.y - bbox.0.y);

                if pos_x > 2.0 {
                    pos_y += row_height + 0.05;
                    row_height = 0.0;
                    pos_x = 0.0;
                }
                Island {
                    mx: Matrix3::from_translation(pos),
                    root: i_face,
                }
            })
            .collect();
        Papercraft {
            edges,
            islands,
        }
    }

    pub fn islands(&self) -> impl Iterator<Item = &Island> + '_ {
        self.islands.iter()
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
                let i_island = self.islands.iter().position(|i| self.contains_face(model, i, i_face_a)).unwrap();

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
                let new_root_plane = new_root.normal();
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
                self.islands.push(new_island);
            }
            EdgeStatus::Cut => {
                let i_island_b = self.islands.iter().position(|i| self.contains_face(model, i, i_face_b)).unwrap();
                if self.contains_face(model, &self.islands[i_island_b], i_face_a) {
                    // Same island on both sides, nothing to do
                } else {
                    // Join both islands
                    let mut island_b = self.islands.remove(i_island_b);
                    let i_island_a = self.islands.iter().position(|i| self.contains_face(model, i, i_face_a)).unwrap();

                    // Keep position of a or b?
                    if self.compare_islands(model, &self.islands[i_island_a], &island_b, priority_face) {
                        std::mem::swap(&mut self.islands[i_island_a], &mut island_b);
                    }

                    self.edges[usize::from(i_edge)] = EdgeStatus::Joined;
                }
            }
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

    fn contains_face(&self, model: &Model, island: &Island, face: FaceIndex) -> bool {
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

    pub fn traverse_faces<F>(&self, model: &Model, island: &Island, mut visit_face: F) -> ControlFlow<()>
        where F: FnMut(FaceIndex, &Face, &Matrix3) -> ControlFlow<()>
    {
        let root = island.root_face();
        let mut visited_faces = HashSet::new();
        let mut stack = vec![(root, island.matrix())];
        visited_faces.insert(root);

        while let Some((i_face, m)) = stack.pop() {
            let face = &model[i_face];
            visit_face(i_face, face, &m)?;
            for i_edge in face.index_edges() {
                if self.edge_status(i_edge) != EdgeStatus::Joined {
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
}

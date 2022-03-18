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
                let face = model.face_by_index(i_face);
                let bbox = model.bounding_box(&face);
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
                    faces: vec![i_face],
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
        self.edges[u32::from(edge) as usize]
    }

    pub fn edge_toggle(&mut self, model: &Model, i_edge: EdgeIndex, priority_face: Option<FaceIndex>) {
        let edge = model.edge_by_index(i_edge);
        let faces: Vec<_> = edge.faces().collect();

        let (i_face_a, i_face_b) = match &faces[..] {
            &[a, b] => (a, b),
            _ => return,
        };

        let edge_status = self.edges[u32::from(i_edge) as usize];

        match edge_status {
            EdgeStatus::Joined => {
                self.edges[u32::from(i_edge) as usize] = EdgeStatus::Cut;
                //one of the edge faces will be the root of the new island
                let island = self.islands.iter_mut().find(|i| i.contains_face(i_face_a)).unwrap();
                let mx = island.mx;
                let root = island.root_face();
                let mut faces_keep = Vec::new();
                let mut face_mx = None;
                crate::paper::traverse_faces(model, root, &mx,
                    |i_face, _, fmx| {
                        faces_keep.push(i_face);
                        if i_face == i_face_a || i_face == i_face_b {
                            face_mx = Some(*fmx);
                        }
                    },
                    |i_edge| self.edges[u32::from(i_edge) as usize] == EdgeStatus::Joined
                );

                let (new_root, i_face_old) = if faces_keep.contains(&i_face_a) {
                    (i_face_b, i_face_a)
                } else {
                    (i_face_a, i_face_b)
                };
                island.faces = faces_keep;

                let mut faces_new = Vec::new();
                crate::paper::traverse_faces(model, new_root, &mx,
                    |i_face,_,_| faces_new.push(i_face),
                    |i_edge| self.edges[u32::from(i_edge) as usize] == EdgeStatus::Joined
                );
                let medge = model.face_to_face_edge_matrix(edge, model.face_by_index(i_face_old), model.face_by_index(new_root));
                let mx = face_mx.unwrap() * medge;

                //Compute the offset
                let sign = if edge.face_sign(new_root) { 1.0 } else { -1.0 };
                let new_root = model.face_by_index(new_root);
                let new_root_plane = new_root.normal();
                let v0 = new_root_plane.project(&model.vertex_by_index(edge.v0()).pos());
                let v1 = new_root_plane.project(&model.vertex_by_index(edge.v1()).pos());
                let v0 = mx.transform_point(Point2::from_vec(v0)).to_vec();
                let v1 = mx.transform_point(Point2::from_vec(v1)).to_vec();
                let v = (v1 - v0).normalize_to(0.05);

                let mut new_island = Island {
                    mx,
                    faces: faces_new,
                };

                if Self::compare_islands(&island, &new_island, priority_face) {
                    let offs = Matrix3::from_translation(-sign * Vector2::new(-v.y, v.x));
                    island.mx = offs * island.mx;
                } else {
                    let offs = Matrix3::from_translation(sign * Vector2::new(-v.y, v.x));
                    new_island.mx = offs * new_island.mx;
                }
                self.islands.push(new_island);
            }
            EdgeStatus::Cut => {
                let pos_b = self.islands.iter().position(|i| i.contains_face(i_face_b)).unwrap();
                if self.islands[pos_b].contains_face(i_face_a) {
                    // Same island on both sides
                } else {
                    // Join both islands
                    self.edges[u32::from(i_edge) as usize] = EdgeStatus::Joined;

                    let mut island_b = self.islands.remove(pos_b);
                    let island_a = self.islands.iter_mut().find(|i| i.contains_face(i_face_a)).unwrap();

                    // Keep position of a or b?
                    if Self::compare_islands(island_a, &island_b, priority_face) {
                        std::mem::swap(island_a, &mut island_b);
                    }
                    island_a.faces.extend(island_b.faces);
                }
            }
        };
    }

    fn compare_islands(a: &Island, b: &Island, priority_face: Option<FaceIndex>) -> bool {
        if let Some(f) = priority_face {
            if a.contains_face(f) {
                return false;
            }
            if b.contains_face(f) {
                return true;
            }
        }
        let weight_a = a.faces.len();
        let weight_b = b.faces.len();
        weight_b > weight_a
    }

    fn _island_by_face(&self, i_face: FaceIndex) -> &Island {
        for i in &self.islands {
            if i.faces.contains(&i_face) {
                return i;
            }
        }
        panic!("island not found for face");
    }

}

pub struct Island {
    mx: Matrix3,
    faces: Vec<FaceIndex>,
}

impl Island {
    pub fn root_face(&self) -> FaceIndex {
        self.faces[0]
    }
    pub fn contains_face(&self, i_face: FaceIndex) -> bool {
        self.faces.contains(&i_face)
    }
    pub fn matrix(&self) -> Matrix3 {
        self.mx
    }
    pub fn _faces(&self) -> impl Iterator<Item = FaceIndex> + '_ {
        self.faces.iter().copied()
    }
}

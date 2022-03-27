use std::{collections::{HashSet, HashMap}, ops::ControlFlow};

use cgmath::{prelude::*, Transform, EuclideanSpace, InnerSpace, Rad};
use slotmap::{SlotMap, new_key_type};
use serde::{Serialize, Deserialize};


use super::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum EdgeStatus {
    Joined,
    Cut,
    Hidden,
}
new_key_type! {
    pub struct IslandKey;
}

#[derive(Serialize, Deserialize)]
pub struct Papercraft {
    model: Model,
    edges: Vec<EdgeStatus>, //parallel to EdgeIndex
    #[serde(with="super::ser::slot_map")]
    islands: SlotMap<IslandKey, Island>,
}

impl Papercraft {
    pub fn new(model: Model, facemap: &HashMap<FaceIndex, u32>) -> Papercraft {
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

            //Compute the bounding box of the flat face, since Self is not yet build, we have to use the traverse_faces_ex() version directly
            let mut vx = Vec::new();
            traverse_faces_ex(&model, root, Matrix3::one(), NormalTraverseFace(&model, &edges),
                |i_face, face, mx| {
                    pending_faces.remove(&i_face);
                    let normal = face.plane(&model);
                    vx.extend(face.index_vertices().map(|v| {
                        mx.transform_point(Point2::from_vec(normal.project(&model[v].pos()))).to_vec()
                    }));
                    ControlFlow::Continue(())
                }
            );

            let bbox = bounding_box_2d(vx);
            let pos = Vector2::new(pos_x - bbox.0.x, pos_y - bbox.0.y);
            pos_x += bbox.1.x - bbox.0.x + 0.05;
            row_height = row_height.max(bbox.1.y - bbox.0.y);

            if pos_x > 2.0 {
                pos_y += row_height + 0.05;
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
            edges,
            islands,
        }
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn islands(&self) -> impl Iterator<Item = (IslandKey, &Island)> + '_ {
        self.islands.iter()
    }

    pub fn island_by_face(&self, i_face: FaceIndex) -> IslandKey {
        for (i_island, island) in &self.islands {
            if self.contains_face(island, i_face) {
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

    pub fn edge_toggle(&mut self, i_edge: EdgeIndex, priority_face: Option<FaceIndex>) {
        let edge = &self.model[i_edge];
        let faces: Vec<_> = edge.faces().collect();

        let (i_face_a, i_face_b) = match &faces[..] {
            &[a, b] => (a, b),
            _ => return,
        };

        let edge_status = self.edges[usize::from(i_edge)];

        match edge_status {
            EdgeStatus::Joined => {
                //one of the edge faces will be the root of the new island, but we do not know which one, yet
                let i_island = self.island_by_face(i_face_a);

                self.edges[usize::from(i_edge)] = EdgeStatus::Cut;

                let mut data_found = None;
                self.traverse_faces(&self.islands[i_island],
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

                let medge = self.model.face_to_face_edge_matrix(edge, &self.model[i_face_old], &self.model[new_root]);
                let mx = face_mx * medge;

                let mut new_island = Island {
                    root: new_root,
                    loc: dbg!(Vector2::new(mx[2][0], mx[2][1])),
                    rot: Rad(mx[0][1].atan2(mx[0][0])),
                    mx: Matrix3::one(),
                };
                new_island.recompute_matrix();

                //Compute the offset
                let sign = if edge.face_sign(new_root) { 1.0 } else { -1.0 };
                let new_root = &self.model[new_root];
                let new_root_plane = new_root.plane(&self.model);
                let v0 = new_root_plane.project(&self.model[edge.v0()].pos());
                let v1 = new_root_plane.project(&self.model[edge.v1()].pos());
                let v0 = mx.transform_point(Point2::from_vec(v0)).to_vec();
                let v1 = mx.transform_point(Point2::from_vec(v1)).to_vec();
                let v = (v1 - v0).normalize_to(0.05);

                //priority_face makes no sense when doing a split, so pass None here unconditionally
                if self.compare_islands(&self.islands[i_island], &new_island, None) {
                    let island = &mut self.islands[i_island];
                    island.translate(-sign * Vector2::new(-v.y, v.x));
                } else {
                    new_island.translate(sign * Vector2::new(-v.y, v.x));
                }
                self.islands.insert(new_island);
            }
            EdgeStatus::Cut => {
                let i_island_b = self.island_by_face(i_face_b);
                if self.contains_face(&self.islands[i_island_b], i_face_a) {
                    // Same island on both sides, nothing to do
                } else {
                    // Join both islands
                    let mut island_b = self.islands.remove(i_island_b).unwrap();
                    let i_island_a = self.island_by_face(i_face_a);

                    // Keep position of a or b?
                    if self.compare_islands(&self.islands[i_island_a], &island_b, priority_face) {
                        std::mem::swap(&mut self.islands[i_island_a], &mut island_b);
                    }

                    self.edges[usize::from(i_edge)] = EdgeStatus::Joined;
                }
            }
            EdgeStatus::Hidden => {}
        };
    }

    fn compare_islands(&self, a: &Island, b: &Island, priority_face: Option<FaceIndex>) -> bool {
        if let Some(f) = priority_face {
            if self.contains_face(a, f) {
                return false;
            }
            if self.contains_face(b, f) {
                return true;
            }
        }
        let (mut weight_a, mut weight_b) = (0, 0);
        self.traverse_faces_no_matrix(a, |_| { weight_a += 1; ControlFlow::Continue(()) });
        self.traverse_faces_no_matrix(b, |_| { weight_b += 1; ControlFlow::Continue(()) });
        weight_b > weight_a
    }

    pub fn contains_face(&self, island: &Island, face: FaceIndex) -> bool {
        let mut found = false;
        self.traverse_faces_no_matrix(island,
            |i_face|
                if i_face == face {
                    found = true;
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            );
        found
    }

    pub fn get_flat_faces(&self, i_face: FaceIndex) -> HashSet<FaceIndex> {
        let mut res = HashSet::new();
        traverse_faces_ex(&self.model, i_face, (), FlatTraverseFace(self),
            |i_next_face, _, _| {
                res.insert(i_next_face);
                ControlFlow::Continue(())
            }
        );
        res
    }
    pub fn get_flat_faces_with_matrix(&self, i_face: FaceIndex, mx: Matrix3) -> HashMap<FaceIndex, Matrix3> {
        let mut res = HashMap::new();
        traverse_faces_ex(&self.model, i_face, mx, FlatTraverseFaceWithMatrix(self),
            |i_next_face, _, mx| {
                res.insert(i_next_face, *mx);
                ControlFlow::Continue(())
            }
        );
        res
    }
    pub fn flat_face_angles(&self, i_face_b: FaceIndex, i_edge: EdgeIndex) -> (Rad<f32>, Rad<f32>) {
        let flat_face = self.get_flat_faces_with_matrix(i_face_b, Matrix3::one());
        let flat_contour: Vec<_> = flat_face
            .iter()
            .flat_map(|(&f, _m)| self.model()[f].vertices_with_edges().map(move |(v0,v1,e)| (f, v0, v1, e)))
            .filter(|&(_f, _v0, _v1, e)| self.edge_status(e) != EdgeStatus::Hidden)
            .collect();
        let (_f, i_v0_b, i_v1_b, _edge_b) = flat_contour
            .iter()
            .copied()
            .filter(|&(_f, _v0, _v1, e)| e == i_edge)
            .next().unwrap();
        let x0 = flat_contour
            .iter()
            .copied()
            .filter(|&(_f, _v0, v1, _e)| i_v0_b == v1)
            .next().unwrap();
        let x1 = flat_contour
            .iter()
            .copied()
            .filter(|&(_f, v0, _v1, _e)| i_v1_b == v0)
            .next().unwrap();

        let pps = [(x0.0, x0.1), (x0.0, x0.2), (x1.0, x1.1), (x1.0, x1.2)]
            .map(|(f, v)| {
                let face = &self.model()[f];
                let lpos = face.plane(self.model()).project(&self.model()[v].pos());
                flat_face[&f].transform_point(Point2::from_vec(lpos)).to_vec()
            });
        let e0 = pps[1] - pps[0];
        let e1 = pps[2] - pps[1];
        let e2 = pps[3] - pps[2];
        let a0 = e1.angle(e0);
        let a1 = e2.angle(e1);
        let a0 = Rad::turn_div_2() - a0;
        let a1 = Rad::turn_div_2() - a1;
        (a0, a1)
    }

    pub fn traverse_faces<F>(&self, island: &Island, visit_face: F) -> ControlFlow<()>
        where F: FnMut(FaceIndex, &Face, &Matrix3) -> ControlFlow<()>
    {
        traverse_faces_ex(&self.model, island.root_face(), island.matrix(), NormalTraverseFace(&self.model, &self.edges), visit_face)
    }
    pub fn traverse_faces_no_matrix<F>(&self, island: &Island, mut visit_face: F) -> ControlFlow<()>
        where F: FnMut(FaceIndex) -> ControlFlow<()>
    {
        traverse_faces_ex(&self.model, island.root_face(), (), NoMatrixTraverseFace(&self.model, &self.edges), |i, _, ()| visit_face(i))
    }
}

fn traverse_faces_ex<F, TP>(model: &Model, root: FaceIndex, initial_state: TP::State, policy: TP, mut visit_face: F) -> ControlFlow<()>
where F: FnMut(FaceIndex, &Face, &TP::State) -> ControlFlow<()>,
      TP: TraverseFacePolicy,
{
    let mut visited_faces = HashSet::new();
    let mut stack = vec![(root, initial_state)];
    visited_faces.insert(root);

    while let Some((i_face, m)) = stack.pop() {
        let face = &model[i_face];
        visit_face(i_face, face, &m)?;
        for i_edge in face.index_edges() {
            if !policy.cross_edge(i_edge) {
                continue;
            }
            let edge = &model[i_edge];
            for i_next_face in edge.faces() {
                if visited_faces.contains(&i_next_face) {
                    continue;
                }
                let next_state = policy.next_state(&m, edge, face, i_next_face);
                stack.push((i_next_face, next_state));
                visited_faces.insert(i_next_face);
            }
        }
    };
    ControlFlow::Continue(())
}

trait TraverseFacePolicy {
    type State;
    fn cross_edge(&self, i_edge: EdgeIndex) -> bool;
    fn next_state(&self, st: &Self::State, edge: &Edge, face: &Face, i_next_face: FaceIndex) -> Self::State;
}
struct NormalTraverseFace<'a>(&'a Model, &'a [EdgeStatus]);

impl TraverseFacePolicy for NormalTraverseFace<'_> {
    type State = Matrix3;

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.1[usize::from(i_edge)] {
            EdgeStatus::Cut => false,
            EdgeStatus::Joined |
            EdgeStatus::Hidden => true,
        }
    }

    fn next_state(&self, st: &Self::State, edge: &Edge, face: &Face, i_next_face: FaceIndex) -> Self::State {
        let next_face = &self.0[i_next_face];
        let medge = self.0.face_to_face_edge_matrix(edge, face, next_face);
        st * medge
    }
}

struct NoMatrixTraverseFace<'a>(&'a Model, &'a [EdgeStatus]);

impl TraverseFacePolicy for NoMatrixTraverseFace<'_> {
    type State = ();

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.1[usize::from(i_edge)] {
            EdgeStatus::Cut => false,
            EdgeStatus::Joined |
            EdgeStatus::Hidden => true,
        }
    }

    fn next_state(&self, _st: &Self::State, _edge: &Edge, _face: &Face, _i_next_face: FaceIndex) -> Self::State {
    }
}

struct FlatTraverseFace<'a>(&'a Papercraft);

impl TraverseFacePolicy for FlatTraverseFace<'_> {
    type State = ();

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.0.edge_status(i_edge) {
            EdgeStatus::Joined |
            EdgeStatus::Cut => false,
            EdgeStatus::Hidden => true,
        }
    }

    fn next_state(&self, _st: &Self::State, _edge: &Edge, _face: &Face, _i_next_face: FaceIndex) -> Self::State {
    }
}

struct FlatTraverseFaceWithMatrix<'a>(&'a Papercraft);


impl TraverseFacePolicy for FlatTraverseFaceWithMatrix<'_> {
    type State = Matrix3;

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.0.edge_status(i_edge) {
            EdgeStatus::Joined |
            EdgeStatus::Cut => false,
            EdgeStatus::Hidden => true,
        }
    }

    fn next_state(&self, st: &Self::State, edge: &Edge, face: &Face, i_next_face: FaceIndex) -> Self::State {
        let next_face = &self.0.model[i_next_face];
        let medge = self.0.model.face_to_face_edge_matrix(edge, face, next_face);
        st * medge
    }
}


#[derive(Debug)]
pub struct Island {
    root: FaceIndex,

    rot: Rad<f32>,
    loc: Vector2,
    mx: Matrix3,
}

impl Island {
    pub fn root_face(&self) -> FaceIndex {
        self.root
    }
    pub fn matrix(&self) -> Matrix3 {
        self.mx
    }
    pub fn translate(&mut self, delta: Vector2) {
        self.loc += delta;
        self.recompute_matrix();
    }
    pub fn rotate(&mut self, angle: impl Into<Rad<f32>>) {
        self.rot = (self.rot + angle.into()).normalize();
        self.recompute_matrix();
    }
    fn recompute_matrix(&mut self) {
        let r = Matrix3::from(cgmath::Matrix2::from_angle(self.rot));
        let t = Matrix3::from_translation(self.loc);
        self.mx = t * r;
    }
}

impl Serialize for EdgeStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        let is = match self {
            EdgeStatus::Hidden => 0,
            EdgeStatus::Joined => 1,
            EdgeStatus::Cut => 2,
        };
        serializer.serialize_i32(is)
    }
}
impl<'de> Deserialize<'de> for EdgeStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        let d = u32::deserialize(deserializer)?;
        let res = match d {
            0 => EdgeStatus::Hidden,
            1 => EdgeStatus::Joined,
            2 => EdgeStatus::Cut,
            _ => return Err(serde::de::Error::missing_field("invalid edge status")),
        };
        Ok(res)
    }
}

impl Serialize for Island {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        let mut map = serializer.serialize_struct("Island", 4)?;
        map.serialize_field("root", &usize::from(self.root))?;
        map.serialize_field("x", &self.loc.x)?;
        map.serialize_field("y", &self.loc.y)?;
        map.serialize_field("r", &self.rot.0)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for Island {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        #[derive(Deserialize)]
        struct Def { root: usize, x: f32, y: f32, r: f32 }
        let d = Def::deserialize(deserializer)?;
        let mut island = Island {
            root: FaceIndex::from(d.root),
            loc: Vector2::new(d.x, d.y),
            rot: Rad(d.r),
            mx: Matrix3::one(),
        };
        island.recompute_matrix();
        Ok(island)
}
}

use std::collections::{HashSet, HashMap};

use cgmath::InnerSpace;

use crate::waveobj;
use crate::util_3d::{self, Vector2, Vector3, Matrix2, Matrix3};

// We use u32 where usize should be use to save some memory in 64-bit systems, and because OpenGL likes 32-bit types in its buffers.
// 32-bit indices should be enough for everybody ;-)

#[derive(Debug)]
pub struct Model {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VertexIndex(u32);

//unsafe: VertexIndex is a transparent u32
unsafe impl glium::index::Index for VertexIndex {
    fn get_type() -> glium::index::IndexType {
        glium::index::IndexType::U32
    }
}

impl From<VertexIndex> for usize {
    fn from(idx: VertexIndex) -> usize {
        idx.0 as usize
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct EdgeIndex(u32);

impl From<EdgeIndex> for usize {
    fn from(idx: EdgeIndex) -> usize {
        idx.0 as usize
    }
}
impl From<usize> for EdgeIndex {
    fn from(idx: usize) -> EdgeIndex {
        EdgeIndex(idx as u32)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FaceIndex(u32);

impl From<FaceIndex> for usize {
    fn from(idx: FaceIndex) -> usize {
        idx.0 as usize
    }
}

#[derive(Debug)]
pub struct Face {
    vertices: [VertexIndex; 3],
    edges: [EdgeIndex; 3],
}

#[derive(Debug)]
pub struct Edge {
    v0: VertexIndex,
    v1: VertexIndex,
    faces: Vec<(FaceIndex, bool)>,
}

#[derive(Debug)]
pub struct Vertex {
    pos: Vector3,
    normal: Vector3,
    uv: Vector2,
}

impl Model {
    pub fn from_waveobj(obj: &waveobj::Model) -> (Model, HashMap<FaceIndex, u32>) {
        // Remove duplicated vertices by adding them into a set
        let all_vertices: HashSet<waveobj::FaceVertex> =
            obj.faces()
                .iter()
                .flat_map(|f| f.vertices())
                .copied()
                .collect();

        //Fix the order into a vector
        let all_vertices = Vec::from_iter(all_vertices);

        // TODO: iterate all_vertices only once
        let idx_vertices: HashMap<waveobj::FaceVertex, u32> =
            all_vertices
                .iter()
                .enumerate()
                .map(|(i, v)| (*v, i as u32))
                .collect();
        let vertices: Vec<Vertex> =
            all_vertices
                .iter()
                .map(|fv| {
                    let uv = Vector2::from(*obj.texcoord_by_index(fv.t()));
                    Vertex {
                        pos: Vector3::from(*obj.vertex_by_index(fv.v())),
                        normal: Vector3::from(*obj.normal_by_index(fv.n())),
                        uv: Vector2::new(uv.x, 1.0 - uv.y),
                    }
                })
                .collect();

        let mut faces: Vec<Face> = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();
        //TODO: index idx_edges?
        let mut idx_edges = Vec::new();
        let mut facemap: HashMap<FaceIndex, u32> = HashMap::new();

        for (index, face) in obj.faces().iter().enumerate() {
            let face_verts: Vec<_> = face
                .vertices()
                .iter()
                .map(|idx| VertexIndex(idx_vertices[idx]))
                .collect();
            let face_verts_orig: Vec<_> = face
                .vertices()
                .iter()
                .map(|idx| idx.v())
                .collect();

            let to_tess: Vec<_> = face_verts
                .iter()
                .map(|v| vertices[usize::from(*v)].pos)
                .collect();
            let (tris, _) = util_3d::tessellate(&to_tess);

            for tri in tris {
                let i_face = FaceIndex(faces.len() as u32);
                facemap.insert(i_face, index as u32);

                let mut face_vertices = [VertexIndex(0); 3];
                let mut face_edges = [EdgeIndex(0); 3];

                for i in 0 .. 3 {
                    face_vertices[i] = face_verts[tri[i]];
                    let v0 = face_verts_orig[tri[i]];
                    let v1 = face_verts_orig[tri[(i + 1) % 3]];
                    if let Some(i_edge) = idx_edges.iter().position(|&(p0, p1)| (p0 == v0 && p1 == v1) || (p0 == v1 && p1 == v0)) {
                        face_edges[i] = EdgeIndex(i_edge as u32);
                        edges[i_edge].faces.push((i_face, true)); //TODO sign?
                    } else {
                        face_edges[i] = EdgeIndex(idx_edges.len() as u32);
                        idx_edges.push((v0, v1));
                        edges.push(Edge {
                            v0: VertexIndex(*idx_vertices.iter().find(|&(f, _)| f.v() == v0).unwrap().1),
                            v1: VertexIndex(*idx_vertices.iter().find(|&(f, _)| f.v() == v1).unwrap().1),
                            faces: vec![(i_face, false)],
                        })
                    }
                }
                faces.push(Face {
                    vertices: face_vertices,
                    edges: face_edges,
                });
            }
        }

        let model = Model {
            vertices,
            edges,
            faces,
        };
        (model, facemap)
    }
    // F gets (pos, normal)
    pub fn transform_vertices<F>(&mut self, mut f: F)
        where F: FnMut(&mut Vector3, &mut Vector3)
    {
        self.vertices.iter_mut().for_each(|v| f(&mut v.pos, &mut v.normal));
    }
    pub fn vertices(&self) -> impl Iterator<Item = &Vertex> {
        self.vertices.iter()
    }
    pub fn faces(&self) -> impl Iterator<Item = (FaceIndex, &Face)> + '_ {
        self.faces
            .iter()
            .enumerate()
            .map(|(i, f)| (FaceIndex(i as u32), f))
    }
    pub fn edges(&self) -> impl Iterator<Item = (EdgeIndex, &Edge)> + '_ {
        self.edges
            .iter()
            .enumerate()
            .map(|(i, e)| (EdgeIndex(i as u32), e))
    }
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
    pub fn face_to_face_edge_matrix(&self, edge: &Edge, face_a: &Face, face_b: &Face) -> Matrix3 {
        let v0 = self[edge.v0()].pos();
        let v1 = self[edge.v1()].pos();
        let plane_a = face_a.normal(self);
        let plane_b = face_b.normal(self);
        let a0 = plane_a.project(&v0);
        let b0 = plane_b.project(&v0);
        let a1 = plane_a.project(&v1);
        let b1 = plane_b.project(&v1);
        let mabt0 = Matrix3::from_translation(-b0);
        let mabr = Matrix3::from(Matrix2::from_angle((b1 - b0).angle(a1 - a0)));
        let mabt1 = Matrix3::from_translation(a0);
        mabt1 * mabr * mabt0
    }
}

impl std::ops::Index<VertexIndex> for Model {
    type Output = Vertex;

    fn index(&self, index: VertexIndex) -> &Vertex {
        &self.vertices[index.0 as usize]
    }
}

impl std::ops::Index<FaceIndex> for Model {
    type Output = Face;

    fn index(&self, index: FaceIndex) -> &Face {
        &self.faces[index.0 as usize]
    }
}

impl std::ops::Index<EdgeIndex> for Model {
    type Output = Edge;

    fn index(&self, index: EdgeIndex) -> &Edge {
        &self.edges[index.0 as usize]
    }
}

impl Face {
    pub fn index_vertices(&self) -> [VertexIndex; 3] {
        self.vertices
    }
    pub fn index_edges(&self) -> [EdgeIndex; 3] {
        self.edges
    }
    pub fn normal(&self, model: &Model) -> util_3d::Plane {
        util_3d::Plane::from_tri([
            model[self.vertices[0]].pos(),
            model[self.vertices[1]].pos(),
            model[self.vertices[2]].pos(),
        ])
    }
    pub fn edges_with_vertices(&self) -> impl Iterator<Item = (VertexIndex, VertexIndex, EdgeIndex)> + '_ {
        self.edges
            .iter()
            .copied()
            .enumerate()
            .map(|(i, e)| {
                let v0 = self.vertices[i];
                let v1 = self.vertices[(i + 1) % self.vertices.len()];
                (v0, v1, e)
            })
    }
}

impl Vertex {
    pub fn pos(&self) -> Vector3 {
        self.pos
    }
    pub fn normal(&self) -> Vector3 {
        self.normal
    }
    pub fn uv(&self) -> Vector2 {
        self.uv
    }
}

impl Edge {
    pub fn v0(&self) -> VertexIndex {
        self.v0
    }
    pub fn v1(&self) -> VertexIndex {
        self.v1
    }
    pub fn faces(&self) -> impl Iterator<Item = FaceIndex> + '_ {
        self.faces.iter().map(|(f, _)| *f)
    }
    pub fn face_sign(&self, face: FaceIndex) -> bool {
        self.faces.iter().find(|(f, _)| *f == face).unwrap().1
    }
}


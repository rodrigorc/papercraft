#![allow(dead_code)]

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct EdgeIndex(u32);

impl From<EdgeIndex> for u32 {
    fn from(idx: EdgeIndex) -> u32 {
        idx.0
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FaceIndex(u32);

impl From<FaceIndex> for u32 {
    fn from(idx: FaceIndex) -> u32 {
        idx.0
    }
}

#[derive(Debug)]
pub struct Face {
    vertices: Vec<VertexIndex>,
    edges: Vec<EdgeIndex>,
    tris: Vec<[u32; 3]>, //result of tesselation, indices in self.vertices
    plane: util_3d::Plane,
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
    pub fn from_waveobj(obj: &waveobj::Model) -> Model {
        // Remove duplicated vertices by adding them into a set
        let all_vertices: HashSet<_> =
            obj.faces()
                .iter()
                .flat_map(|f| f.vertices())
                .copied()
                .collect();

        //Fix the order into a vector
        let all_vertices = Vec::from_iter(all_vertices);

        // TODO: iterate all_vertices only once
        let idx_vertices: HashMap<_, _> =
            all_vertices
                .iter()
                .enumerate()
                .map(|(i, v)| (*v, i as u32))
                .collect();
        let vertices: Vec<_> =
            all_vertices
                .iter()
                .map(|fv| Vertex {
                    pos: Vector3::from(*obj.vertex_by_index(fv.v())),
                    normal: Vector3::from(*obj.normal_by_index(fv.n())),
                    uv: Vector2::from(*obj.texcoord_by_index(fv.t())),
                })
                .collect();

        let mut faces: Vec<Face> = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();
        //TODO: index idx_edges?
        let mut idx_edges = Vec::new();

        for (i_face, face) in obj.faces().iter().enumerate() {
            let i_face = FaceIndex(i_face as u32);
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

            let mut face_edges = Vec::with_capacity(face_verts_orig.len());
            for (i0, &v0) in face_verts_orig.iter().enumerate() {
                let v1 = face_verts_orig[(i0 + 1) % face_verts_orig.len()];

                if let Some(i_edge) = idx_edges.iter().position(|&(p0, p1)| (p0 == v0 && p1 == v1) || (p0 == v1 && p1 == v0)) {
                    face_edges.push(EdgeIndex(i_edge as u32));
                    edges[i_edge].faces.push((i_face, true)); //TODO sign?
                } else {
                    face_edges.push(EdgeIndex(idx_edges.len() as u32));
                    idx_edges.push((v0, v1));
                    edges.push(Edge {
                        v0: VertexIndex(*idx_vertices.iter().find(|&(f, _)| f.v() == v0).unwrap().1),
                        v1: VertexIndex(*idx_vertices.iter().find(|&(f, _)| f.v() == v1).unwrap().1),
                        faces: vec![(i_face, false)],
                    })
                }
            }
            faces.push(Face {
                vertices: face_verts,
                edges: face_edges,
                tris: Vec::new(),
                plane: util_3d::Plane::default(),
            });
        }

        Model {
            vertices,
            edges,
            faces,
        }
    }
    // F gets (pos, normal)
    pub fn transform_vertices<F>(&mut self, mut f: F)
        where F: FnMut(&mut Vector3, &mut Vector3)
    {
        self.vertices.iter_mut().for_each(|v| f(&mut v.pos, &mut v.normal));
    }
    pub fn tessellate_faces(&mut self) {
        for face in &mut self.faces {
            let to_tess: Vec<_> = face.vertices
                .iter()
                .map(|v| self.vertices[v.0 as usize].pos)
                .collect();
            let (tris, plane) = util_3d::tessellate(&to_tess);
            face.tris =
                tris
                    .into_iter()
                    .map(|tri| tri.map(|x| x as u32))
                    .collect();
            face.plane = plane;
        }
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
    pub fn face_by_index(&self, idx: FaceIndex) -> &Face {
        &self.faces[idx.0 as usize]
    }
    pub fn vertex_by_index(&self, idx: VertexIndex) -> &Vertex {
        &self.vertices[idx.0 as usize]
    }
    pub fn edge_by_index(&self, idx: EdgeIndex) -> &Edge {
        &self.edges[idx.0 as usize]
    }
    pub fn faces_by_edge(&self, edge: EdgeIndex) -> [(FaceIndex, &Face); 2] {
        let mut res = Vec::with_capacity(2);
        for (iface, face) in self.faces() {
            if face.edges.contains(&edge) {
                res.push((iface, face));
                if res.len() == 2 {
                    return res.try_into().unwrap();
                }
            }
        }
        //TODO: do not panic
        panic!("unconnected edge")
    }
    pub fn face_to_face_edge_matrix(&self, edge: &Edge, face_a: &Face, face_b: &Face) -> Matrix3 {
        let v0 = self.vertex_by_index(edge.v0()).pos();
        let v1 = self.vertex_by_index(edge.v1()).pos();
        let a0 = face_a.normal().project(&v0);
        let b0 = face_b.normal().project(&v0);
        let a1 = face_a.normal().project(&v1);
        let b1 = face_b.normal().project(&v1);
        let mabt0 = Matrix3::from_translation(-b0);
        let mabr = Matrix3::from(Matrix2::from_angle((b1 - b0).angle(a1 - a0)));
        let mabt1 = Matrix3::from_translation(a0);
        let medge = mabt1 * mabr * mabt0;
        medge
    }

    pub fn bounding_box(&self, face: &Face) -> (Vector2, Vector2) {
        let mut min = Vector2::new(1000.0, 1000.0);
        let mut max = Vector2::new(-1000.0, -1000.0);
        let normal = face.normal();
        for v in face.index_vertices() {
            let vertex = self.vertex_by_index(v);
            let v = normal.project(&vertex.pos());
            min.x = min.x.min(v.x);
            min.y = min.y.min(v.y);
            max.x = max.x.max(v.x);
            max.y = max.y.max(v.y);
        }
        (min, max)
    }
}

impl Face {
    pub fn index_vertices(&self) -> impl Iterator<Item = VertexIndex> + '_ {
        self.vertices.iter().copied()
    }
    pub fn index_edges(&self) -> impl Iterator<Item = EdgeIndex> + '_ {
        self.edges.iter().copied()
    }
    pub fn index_triangles(&self) -> impl Iterator<Item = [VertexIndex; 3]> + '_ {
        self.tris
            .iter()
            .map(|tri| tri.map(|v| self.vertices[v as usize]))
    }
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }
    pub fn num_triangles(&self) -> usize {
        self.tris.len()
    }
    pub fn normal(&self) -> &util_3d::Plane {
        &self.plane
    }
}

impl Vertex {
    pub fn pos(&self) -> Vector3 {
        self.pos
    }
    pub fn pos_mut(&mut self) -> &mut Vector3 {
        &mut self.pos
    }
    pub fn normal(&self) -> Vector3 {
        self.normal
    }
    pub fn uv(&self) -> Vector2 {
        self.uv
    }
    pub fn uv_inv(&self) -> Vector2 {
        Vector2::new(self.uv.x, 1.0 - self.uv.y)
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
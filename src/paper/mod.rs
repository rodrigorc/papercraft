#![allow(dead_code)]

use crate::waveobj;

#[derive(Debug)]
pub struct Model {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
}

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct VertexIndex(u32);

//unsafe: VertexIndex is a transparent u32
unsafe impl glium::index::Index for VertexIndex {
    fn get_type() -> glium::index::IndexType {
        glium::index::IndexType::U32
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct EdgeIndex(u32);

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct FaceIndex(u32);

#[derive(Debug)]
pub struct Face {
    vertices: Vec<VertexIndex>,
    edges: Vec<EdgeIndex>,
    tris: Vec<[usize; 3]>, //indices in self.vertices
}

#[derive(Debug)]
pub struct Edge {
    v0: VertexIndex,
    v1: VertexIndex,
}

#[derive(Debug)]
pub struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

impl Model {
    pub fn from_waveobj(obj: &waveobj::Model) -> Model {
        let vertices: Vec<_> = obj.vertices()
            .iter()
            .map(|v| Vertex {
                pos: *v.pos(),
                normal: *v.normal(),
                uv: *v.uv(),
            })
            .collect();

        let mut faces = Vec::new();
        let mut edges = Vec::new();
        for face in obj.faces() {
            let face_verts: Vec<_> = face.indices().iter().map(|idx| VertexIndex(*idx)).collect();
            let mut face_edges = Vec::with_capacity(face_verts.len());
            for (i0, &v0) in face_verts.iter().enumerate() {
                let v1 = face_verts[(i0 + 1) % face_verts.len()];
                face_edges.push(EdgeIndex(edges.len() as u32));
                edges.push(Edge {
                    v0,
                    v1,
                });
            }

            let to_tess: Vec<_> = face_verts
                .iter()
                .map(|v| {
                    cgmath::Vector3::from(vertices[v.0 as usize].pos)
                })
                .collect();
            let tris = crate::util_3d::tessellate(&to_tess);
            faces.push(Face {
                vertices: face_verts,
                edges: face_edges,
                tris,
            });
        }

        Model {
            vertices,
            edges,
            faces,
        }
    }
    pub fn transform_vertices<F>(&mut self, f: F)
        where F: FnMut(&mut Vertex)
    {
        self.vertices.iter_mut().for_each(f);
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
    pub fn face_by_index(&self, idx: FaceIndex) -> &Face {
        &self.faces[idx.0 as usize]
    }
    pub fn vertex_by_index(&self, idx: VertexIndex) -> &Vertex {
        &self.vertices[idx.0 as usize]
    }
    pub fn edge_by_index(&self, idx: EdgeIndex) -> &Edge {
        &self.edges[idx.0 as usize]
    }
}

impl Face {
    pub fn index_vertices(&self) -> &[VertexIndex] {
        &self.vertices
    }
    pub fn index_triangles(&self) -> impl Iterator<Item = [VertexIndex; 3]> + '_ {
        self.tris
            .iter()
            .map(|tri| tri.map(|v| self.vertices[v]))
    }
}

impl Vertex {
    pub fn pos(&self) -> &[f32; 3] {
        &self.pos
    }
    pub fn pos_mut(&mut self) -> &mut [f32; 3] {
        &mut self.pos
    }
    pub fn normal(&self) -> &[f32; 3] {
        &self.normal
    }
    pub fn uv(&self) -> &[f32; 2] {
        &self.uv
    }
}

impl Edge {
    pub fn v0(&self) -> VertexIndex {
        self.v0
    }
    pub fn v1(&self) -> VertexIndex {
        self.v1
    }
}

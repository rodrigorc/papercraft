#![allow(dead_code)]
use crate::waveobj;

#[derive(Debug)]
pub struct Model {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
}

#[derive(Debug, Copy, Clone)]
pub struct VertexIndex(pub u32);

#[derive(Debug, Copy, Clone)]
pub struct EdgeIndex(pub u32);

#[derive(Debug)]
pub struct Face {
    vertices: Vec<VertexIndex>,
    edges: Vec<EdgeIndex>,
    tris: Vec<(usize, usize, usize)>, //indices in self.vertices
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

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }
    pub fn vertices_mut(&mut self) -> &mut [Vertex] {
        &mut self.vertices
    }

    pub fn faces(&self) -> &[Face] {
        &self.faces
    }

    pub fn vertex_by_index(&self, idx: VertexIndex) -> &Vertex {
        &self.vertices[idx.0 as usize]
    }
}

impl Face {
    pub fn index_vertices(&self) -> &[VertexIndex] {
        &self.vertices
    }
    pub fn index_triangles(&self) -> impl Iterator<Item = (VertexIndex, VertexIndex, VertexIndex)> + '_ {
        self.tris
            .iter()
            .map(|&(a, b, c)| {
                (self.vertices[a], self.vertices[b], self.vertices[c])
            })
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

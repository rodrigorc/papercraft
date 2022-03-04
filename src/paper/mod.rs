#![allow(dead_code)]

use crate::waveobj;
use crate::util_3d;

// We use u32 where usize should be use to save some memory in 64-bit systems, and because OpenGL likes 32-bit types in its buffers.
// 32-bit indices should be enough for everybody ;-)

#[derive(Debug)]
pub struct Model {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct VertexIndex(u32);

//unsafe: VertexIndex is a transparent u32
unsafe impl glium::index::Index for VertexIndex {
    fn get_type() -> glium::index::IndexType {
        glium::index::IndexType::U32
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct EdgeIndex(u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct FaceIndex(u32);

#[derive(Debug)]
pub struct Face {
    vertices: Vec<VertexIndex>,
    edges: Vec<EdgeIndex>,
    tris: Vec<[u32; 3]>, //result of tesselation, indices in self.vertices
    normal: util_3d::Plane,
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
            let (tris, normal) = util_3d::tessellate(&to_tess);
            let tris =
                tris
                    .into_iter()
                    .map(|tri| tri.map(|x| x as u32))
                    .collect();

            faces.push(Face {
                vertices: face_verts,
                edges: face_edges,
                tris,
                normal,
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
        where F: FnMut(&mut [f32; 3], &mut [f32; 3])
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
    pub fn index_vertices(&self) -> impl Iterator<Item = VertexIndex> + '_ {
        self.vertices.iter().copied()
    }
    pub fn index_triangles(&self) -> impl Iterator<Item = [VertexIndex; 3]> + '_ {
        self.tris
            .iter()
            .map(|tri| tri.map(|v| self.vertices[v as usize]))
    }
    pub fn normal(&self) -> &util_3d::Plane {
        &self.normal
    }
}

impl Vertex {
    pub fn pos(&self) -> [f32; 3] {
        self.pos
    }
    pub fn pos_mut(&mut self) -> &mut [f32; 3] {
        &mut self.pos
    }
    pub fn normal(&self) -> [f32; 3] {
        self.normal
    }
    pub fn uv(&self) -> [f32; 2] {
        self.uv
    }
    pub fn uv_inv(&self) -> [f32; 2] {
        [self.uv[0], 1.0 - self.uv[1]]
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

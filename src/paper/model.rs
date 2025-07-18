use anyhow::{Context, Result};
use cgmath::{Angle, InnerSpace, Rad};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::marker::PhantomData;

use crate::paper::import::Importer;
use crate::util_3d::{self, Vector2, Vector3};

use super::{BodyTraverse, EdgeStatus, Island, PaperOptions, traverse_faces_ex};

pub mod import;

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Texture {
    file_name: String,
    #[serde(skip)]
    pixbuf: Option<DynamicImage>,
}

impl Texture {
    pub fn file_name(&self) -> &str {
        &self.file_name
    }
    pub fn pixbuf(&self) -> Option<&DynamicImage> {
        self.pixbuf.as_ref()
    }
}

#[derive(Debug)]
pub struct Model {
    textures: Vec<Texture>,
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
    multi_body: bool,
}

use maybe_owned::MaybeOwned;

#[derive(Serialize, Deserialize)]
#[serde(rename = "Model")]
struct ModelSer<'s> {
    textures: MaybeOwned<'s, Vec<Texture>>,
    #[serde(rename = "vs")]
    vertices: MaybeOwned<'s, Vec<Vertex>>,
    #[serde(rename = "es")]
    edges: MaybeOwned<'s, Vec<Edge>>,
    #[serde(rename = "fs")]
    faces: MaybeOwned<'s, Vec<Face>>,
}

// Hack to pass a serialization context to the Edges, it will be removed, eventually
thread_local! {
    static SER_MODEL: Cell<Option<*const Model>> = const { Cell::new(None) };
}
struct SetSerModel<'a> {
    old: Option<*const Model>,
    _pd: PhantomData<&'a Model>,
}
impl SetSerModel<'_> {
    fn new(m: &Model) -> SetSerModel<'_> {
        let old = SER_MODEL.replace(Some(m));
        SetSerModel {
            old,
            _pd: PhantomData,
        }
    }
}
impl Drop for SetSerModel<'_> {
    fn drop(&mut self) {
        SER_MODEL.set(self.old);
    }
}

impl Serialize for Model {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use MaybeOwned::Borrowed;
        let _ctx = SetSerModel::new(self);

        let se = ModelSer {
            textures: Borrowed(&self.textures),
            vertices: Borrowed(&self.vertices),
            edges: Borrowed(&self.edges),
            faces: Borrowed(&self.faces),
        };
        se.serialize(ser)
    }
}
impl<'de> Deserialize<'de> for Model {
    fn deserialize<D>(des: D) -> Result<Model, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use MaybeOwned::Owned;
        let ModelSer {
            textures: Owned(textures),
            vertices: Owned(vertices),
            edges: Owned(edges),
            faces: Owned(faces),
        } = ModelSer::deserialize(des)?
        else {
            unreachable!()
        };
        let mut model = Model {
            textures,
            vertices,
            edges,
            faces,
            multi_body: false,
        };
        model.post_create();
        Ok(model)
    }
}

// We use u32 where usize should be use to save some memory in 64-bit systems, and because OpenGL likes 32-bit types in its buffers.
// 32-bit indices should be enough for everybody ;-)
macro_rules! index_type {
    ($vis:vis $name:ident : $inner:ty) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
        #[repr(transparent)]
        #[serde(transparent)]
        $vis struct $name($inner);

        impl From<$name> for usize {
            fn from(idx: $name) -> usize {
                idx.0 as usize
            }
        }

        impl From<usize> for $name {
            fn from(idx: usize) -> $name {
                $name(idx as $inner)
            }
        }

        unsafe impl crate::glr::AttribField for $name {
            fn detail() -> (usize, u32) {
                <$inner>::detail()
            }
        }
    }
}

index_type!(pub MaterialIndex: u32);
index_type!(pub VertexIndex: u32);
index_type!(pub EdgeIndex: u32);
index_type!(pub FaceIndex: u32);
index_type!(pub BodyIndex: u32);

#[derive(Debug, Serialize, Deserialize)]
pub struct Face {
    #[serde(rename = "m")]
    material: MaterialIndex,
    #[serde(rename = "vs")]
    vertices: [VertexIndex; 3],
    #[serde(rename = "es")]
    edges: [EdgeIndex; 3],
}

// Beware! The vertices that form the edge in `f0` and those in `f1` may be different, because of
// the UV.
// If you want the proper VertexIndex from the POV of a face, use `Face::vertices_with_edges()`.
// If you just want the position of the edge limits use `Model::edge_pos()`.
#[derive(Debug, Deserialize)]
pub struct Edge {
    f0: FaceIndex,
    f1: Option<FaceIndex>,
    #[serde(skip, default = "default_angle")]
    angle: Rad<f32>,
}

fn default_angle() -> Rad<f32> {
    Rad::full_turn() / 2.0 //180 degree
}

impl Serialize for Edge {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        // (v0,v1) are not used, they are there for compatibility with old Papercraft
        // versions.
        let model = unsafe { &*SER_MODEL.get().unwrap() };
        let i_edge = model.edge_index(self);
        let (v0, v1) = model[self.f0].vertices_of_edge(i_edge).unwrap();

        let mut x = ser.serialize_struct("Edge", 4)?;
        x.serialize_field("f0", &self.f0)?;
        x.serialize_field("f1", &self.f1)?;
        x.serialize_field("v0", &v0)?;
        x.serialize_field("v1", &v1)?;
        x.end()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Vertex {
    #[serde(rename = "p", with = "super::ser::vector3")]
    pos: Vector3,
    #[serde(rename = "n", with = "super::ser::vector3")]
    normal: Vector3,
    #[serde(rename = "t", with = "super::ser::vector2")]
    uv: Vector2,
}

// To check if two faces originate from the same face in an imported model
#[derive(Debug, PartialEq, Eq)]
pub struct FaceSource(u32);

pub struct ImportedModule<I: Importer> {
    pub model: Model,
    pub face_map: Vec<FaceSource>,
    pub edge_map: Vec<(I::VertexId, I::VertexId)>,
}

impl Model {
    pub fn empty() -> Model {
        Model {
            textures: Vec::new(),
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            multi_body: false,
        }
    }

    pub fn from_importer<I: Importer>(obj: &mut I) -> ImportedModule<I> {
        let (has_normals, mut vertices) = obj.build_vertices();

        let num_faces = obj.face_count();
        let mut faces: Vec<Face> = Vec::with_capacity(num_faces);
        let mut face_map: Vec<FaceSource> = Vec::with_capacity(num_faces);
        let mut edges: Vec<Edge> = Vec::with_capacity(num_faces * 3 / 2);
        let mut edge_map: Vec<(I::VertexId, I::VertexId)> = Vec::with_capacity(num_faces * 3 / 2);

        let mut face_source_id = 0;
        'face_loop: for (face_verts, face_mat) in obj.faces() {
            let face_verts = face_verts.as_ref();
            face_source_id += 1;
            let to_tess: Vec<_> = face_verts
                .iter()
                .map(|v| vertices[usize::from(*v)].pos)
                .collect();
            let (tris, _) = util_3d::tessellate(&to_tess);
            for tri in tris {
                let i_face = FaceIndex::from(faces.len());

                // Some faces may be degenerate and have to be skipped, and we must not modify the model structure,
                // so be sure it is correct before we accept it.
                enum EdgeCreation<I: Importer> {
                    Existing(usize),
                    New(Edge, (I::VertexId, I::VertexId)),
                }
                // dummy values, will be filled later
                let mut face_edges = [
                    EdgeCreation::<I>::Existing(0),
                    EdgeCreation::Existing(0),
                    EdgeCreation::Existing(0),
                ];
                let mut face_vertices = [VertexIndex::from(0); 3];

                for ((i, face_edge), face_vertex) in
                    (0..3).zip(&mut face_edges).zip(&mut face_vertices)
                {
                    let v0 = face_verts[tri[i]];
                    *face_vertex = v0;
                    let v1 = face_verts[tri[(i + 1) % 3]];
                    let v0 = obj.vertex_map(v0);
                    let v1 = obj.vertex_map(v1);
                    let mut i_edge_candidate = edge_map
                        .iter()
                        .position(|(p0, p1)| (p0, p1) == (&v0, &v1) || (p0, p1) == (&v1, &v0));

                    if let Some(i_edge) = i_edge_candidate {
                        if edges[i_edge].f1.is_some() {
                            // Maximum 2 faces per edge, additional faces will clone the edge and be disconnected
                            println!("Warning: three-faced edge #{i_edge}");
                            i_edge_candidate = None;
                        } else if edge_map[i_edge] != (v1, v0) {
                            // The found edge should be inverted: (v1,v0), unless you are doing a Moebius strip or something weird.
                            // This is mostly harmless, though.
                            println!("Warning: inverted edge #{i_edge}: {v0:?}-{v1:?}");
                        }
                    }

                    *face_edge = match i_edge_candidate {
                        Some(i_edge) => EdgeCreation::Existing(i_edge),
                        None => EdgeCreation::New(
                            Edge {
                                f0: i_face,
                                f1: None,
                                angle: default_angle(),
                            },
                            (v0, v1),
                        ),
                    }
                }

                // If the face uses the same egde twice, it is invalid
                match face_edges {
                    [EdgeCreation::Existing(a), EdgeCreation::Existing(b), _]
                    | [EdgeCreation::Existing(a), _, EdgeCreation::Existing(b)]
                    | [_, EdgeCreation::Existing(a), EdgeCreation::Existing(b)]
                        if a == b =>
                    {
                        continue 'face_loop;
                    }
                    _ => {}
                }

                let edges = face_edges.map(|face_edge| {
                    let e = match face_edge {
                        EdgeCreation::New(edge, idxs) => {
                            edge_map.push(idxs);
                            let e = edges.len();
                            edges.push(edge);
                            e
                        }
                        EdgeCreation::Existing(e) => {
                            edges[e].f1 = Some(i_face);
                            e
                        }
                    };
                    EdgeIndex::from(e)
                });

                if !has_normals {
                    let v0 = vertices[usize::from(face_vertices[0])].pos;
                    let v1 = vertices[usize::from(face_vertices[1])].pos;
                    let v2 = vertices[usize::from(face_vertices[2])].pos;
                    let l0 = v1 - v0;
                    let l1 = v2 - v0;
                    let normal = l0.cross(l1).normalize();
                    vertices[usize::from(face_vertices[0])].normal += normal;
                    vertices[usize::from(face_vertices[1])].normal += normal;
                    vertices[usize::from(face_vertices[2])].normal += normal;
                }

                face_map.push(FaceSource(face_source_id));
                faces.push(Face {
                    material: face_mat,
                    vertices: face_vertices,
                    edges,
                });
            }
        }

        let textures = obj.build_textures();
        let mut model = Model {
            textures,
            vertices,
            edges,
            faces,
            multi_body: false,
        };
        model.post_create();
        ImportedModule {
            model,
            face_map,
            edge_map,
        }
    }
    fn post_create(&mut self) {
        // Compute edge angles
        for i_edge in 0..self.edges.len() {
            let i_edge = EdgeIndex::from(i_edge);
            let edge = &self[i_edge];
            let angle = match edge.faces() {
                (fa, Some(fb)) => {
                    let fa = &self[fa];
                    let fb = &self[fb];
                    let na = self.face_plane(fa).normal();
                    let nb = self.face_plane(fb).normal();

                    let i_va = fa.opposite_edge(i_edge);
                    let pos_va = &self[i_va].pos();
                    let i_vb = fb.opposite_edge(i_edge);
                    let pos_vb = &self[i_vb].pos();

                    let sign = na.dot(pos_vb - pos_va).signum();
                    Rad(sign * nb.angle(na).0)
                }
                _ => default_angle(),
            };
            self.edges[usize::from(i_edge)].angle = angle;
        }

        let first_face = self.faces().next();
        if let Some((i_face, _)) = first_face {
            let mut count = 0;
            let _ = traverse_faces_ex(self, i_face, (), BodyTraverse, |_, _, _| {
                count += 1;
                std::ops::ControlFlow::Continue(())
            });
            self.multi_body = count < self.num_faces();
            if self.multi_body {
                log::info!("Multi body model");
            }
        }
    }
    pub fn vertices(&self) -> impl Iterator<Item = (VertexIndex, &Vertex)> {
        self.vertices
            .iter()
            .enumerate()
            .map(|(i, v)| (VertexIndex::from(i), v))
    }
    pub fn faces(&self) -> impl Iterator<Item = (FaceIndex, &Face)> + '_ {
        self.faces
            .iter()
            .enumerate()
            .map(|(i, f)| (FaceIndex::from(i), f))
    }
    pub fn edges(&self) -> impl Iterator<Item = (EdgeIndex, &Edge)> + '_ {
        self.edges
            .iter()
            .enumerate()
            .map(|(i, e)| (EdgeIndex::from(i), e))
    }
    // These are a bit hacky...
    pub fn edge_index(&self, e: &Edge) -> EdgeIndex {
        let e = e as *const Edge as usize;
        let s = self.edges.as_ptr() as usize;
        EdgeIndex::from((e - s) / std::mem::size_of::<Edge>())
    }
    pub fn face_index(&self, f: &Face) -> FaceIndex {
        let e = f as *const Face as usize;
        let s = self.faces.as_ptr() as usize;
        FaceIndex::from((e - s) / std::mem::size_of::<Face>())
    }
    pub fn edge_pos(&self, e: &Edge) -> (Vector3, Vector3) {
        let i_edge = self.edge_index(e);
        let (v0, v1) = self[e.f0].vertices_of_edge(i_edge).unwrap();
        (self[v0].pos, self[v1].pos)
    }
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
    pub fn num_textures(&self) -> usize {
        self.textures.len()
    }
    pub fn has_textures(&self) -> bool {
        self.textures.iter().any(|t| t.pixbuf.is_some())
    }
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
    pub fn multi_body(&self) -> bool {
        self.multi_body
    }
    pub fn textures(&self) -> impl Iterator<Item = &Texture> + '_ {
        self.textures.iter()
    }
    pub fn reload_textures<F: FnMut(&str) -> Result<Option<DynamicImage>>>(
        &mut self,
        mut f: F,
    ) -> Result<()> {
        for tex in &mut self.textures {
            tex.pixbuf = if tex.file_name.is_empty() {
                None
            } else {
                f(&tex.file_name).with_context(|| tex.file_name.clone())?
            };
        }
        Ok(())
    }
    pub fn rotate_face_vertices(&mut self, i_face: FaceIndex, rotation: i32) {
        let face = &mut self.faces[usize::from(i_face)];
        match rotation.cmp(&0) {
            std::cmp::Ordering::Less => {
                face.vertices.rotate_left(1);
                face.edges.rotate_left(1);
            }
            std::cmp::Ordering::Greater => {
                face.vertices.rotate_right(1);
                face.edges.rotate_right(1);
            }
            std::cmp::Ordering::Equal => {}
        }
    }
    pub fn face_plane(&self, face: &Face) -> util_3d::Plane {
        util_3d::Plane::from_tri([
            self[face.vertices[0]].pos(),
            self[face.vertices[1]].pos(),
            self[face.vertices[2]].pos(),
        ])
    }
    pub fn face_area(&self, i_face: FaceIndex) -> f32 {
        let face = &self[i_face];
        // Area in 3D space should be almost equal to the area in 2D space,
        // except for very non planar n-gons, but if that is the case blame the user.
        let [a, b, c] = face.index_vertices().map(|iv| self[iv].pos());
        let ab = b - a;
        let ac = c - a;
        ab.cross(ac).magnitude() / 2.0
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
    pub fn material(&self) -> MaterialIndex {
        self.material
    }
    pub fn vertices_with_edges(
        &self,
    ) -> impl Iterator<Item = (VertexIndex, VertexIndex, EdgeIndex)> + '_ {
        self.edges.iter().copied().enumerate().map(|(i, e)| {
            let v0 = self.vertices[i];
            let v1 = self.vertices[(i + 1) % self.vertices.len()];
            (v0, v1, e)
        })
    }
    pub fn vertices_of_edge(&self, i_edge: EdgeIndex) -> Option<(VertexIndex, VertexIndex)> {
        let i = self.edges.iter().position(|i_e| *i_e == i_edge)?;
        let v0 = self.vertices[i];
        let v1 = self.vertices[(i + 1) % self.vertices.len()];
        Some((v0, v1))
    }
    pub fn opposite_edge(&self, i_edge: EdgeIndex) -> VertexIndex {
        let i = self.edges.iter().position(|e| *e == i_edge).unwrap();
        self.vertices[(i + 2) % 3]
    }
    pub fn next_edge(&self, i_edge: EdgeIndex) -> (EdgeIndex, VertexIndex) {
        let i = self.edges.iter().position(|e| *e == i_edge).unwrap();
        let i = (i + 1) % 3;
        (self.edges[i], self.vertices[i])
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
    // Do not make (v0,v1) public, they are prone to error
    pub fn faces(&self) -> (FaceIndex, Option<FaceIndex>) {
        (self.f0, self.f1)
    }
    pub fn face_sign(&self, i_face: FaceIndex) -> bool {
        if self.f0 == i_face {
            false
        } else if self.f1 == Some(i_face) {
            true
        } else {
            // Model is inconsistent
            panic!();
        }
    }
    pub fn face_by_sign(&self, face_sign: bool) -> Option<FaceIndex> {
        if face_sign { self.f1 } else { Some(self.f0) }
    }
    pub fn angle(&self) -> Rad<f32> {
        self.angle
    }
}

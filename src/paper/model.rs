use std::marker::PhantomData;
use std::cell::Cell;
use std::path::Path;
use fxhash::{FxHashMap, FxHashSet};
use cgmath::{InnerSpace, Rad, Angle, Zero};
use image::{DynamicImage, ImageBuffer};
use serde::{Serialize, Deserialize};
use anyhow::{anyhow, Result, Context};

use crate::{waveobj, pepakura};
use crate::util_3d::{self, Vector2, Vector3, TransparentType};

use super::{EdgeStatus, TabSide, Island, PaperOptions, PageOffset};

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

#[derive(Debug, Deserialize)]
pub struct Model {
    textures: Vec<Texture>,
    #[serde(rename="vs")]
    vertices: Vec<Vertex>,
    #[serde(rename="es")]
    edges: Vec<Edge>,
    #[serde(rename="fs")]
    faces: Vec<Face>,
}

// Hack to pass a serialization context to the Edges, it will be removed, eventually
thread_local! {
    static SER_MODEL: Cell<Option<*const Model>> = Cell::new(None);
}
struct SetSerModel<'a> {
    old: Option<*const Model>,
    _pd: PhantomData<&'a Model>,
}
impl SetSerModel<'_> {
    fn new(m: &Model) -> SetSerModel {
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
        where S: serde::Serializer
    {
        let _ctx = SetSerModel::new(self);

        use serde::ser::SerializeStruct;
        let mut x = ser.serialize_struct("Model", 4)?;
        x.serialize_field("textures", &self.textures)?;
        x.serialize_field("vs", &self.vertices)?;
        x.serialize_field("es", &self.edges)?;
        x.serialize_field("fs", &self.faces)?;
        x.end()
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

        impl TransparentType for $name {
            type Inner = $inner;
        }
    }
}

index_type!(pub MaterialIndex: u32);
index_type!(pub VertexIndex: u32);
index_type!(pub EdgeIndex: u32);
index_type!(pub FaceIndex: u32);

#[derive(Debug, Serialize, Deserialize)]
pub struct Face {
    #[serde(rename="m")]
    material: MaterialIndex,
    #[serde(rename="vs")]
    vertices: [VertexIndex; 3],
    #[serde(rename="es")]
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
}

impl Serialize for Edge {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        use serde::ser::SerializeStruct;

        // (v0,v1) are not used, they are there for compatibility with old Papercraft
        // versions.
        let model = unsafe { &*SER_MODEL.get().unwrap() };
        let i_edge = model.edge_index(self);
        let (v0, v1, _) = model[self.f0].vertices_with_edges().find(|&(_, _, e)| e == i_edge).unwrap();

        let mut x = ser.serialize_struct("Edge", 4)?;
        x.serialize_field("f0", &self.f0)?;
        x.serialize_field("f1", &self.f1)?;
        x.serialize_field("v0", &v0)?;
        x.serialize_field("v1", &v1)?;
        x.end()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Vertex {
    #[serde(rename="p", with="super::ser::vector3")]
    pos: Vector3,
    #[serde(rename="n", with="super::ser::vector3")]
    normal: Vector3,
    #[serde(rename="t", with="super::ser::vector2")]
    uv: Vector2,
}

impl Model {
    pub fn empty() -> Model {
        Model {
            textures: Vec::new(),
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
        }
    }

    pub fn from_importer<I: Importer>(obj: &mut I) -> (Model, Vec<I::FaceId>, Vec<(I::VertexId, I::VertexId)>) {
        let (has_normals, mut vertices) = obj.build_vertices();

        let num_faces = obj.face_count();
        let mut faces: Vec<Face> = Vec::with_capacity(num_faces);
        let mut face_map: Vec<I::FaceId> = Vec::with_capacity(num_faces);
        let mut edges: Vec<Edge> = Vec::with_capacity(num_faces * 3 / 2);
        let mut edge_map: Vec<(I::VertexId, I::VertexId)> = Vec::with_capacity(num_faces * 3 / 2);

        obj.for_each_face(|face_id, face_verts, face_mat| {
            let to_tess: Vec<_> = face_verts
                .iter()
                .map(|v| vertices[usize::from(*v)].pos)
                .collect();
            let (tris, _) = util_3d::tessellate(&to_tess);
            for tri in tris {
                let i_face = FaceIndex::from(faces.len());

                // Some faces may be degenerate and have to be skipped, and we must not modify the model structure, so be sure it is correct before we accept it
                enum EdgeCreation<I:Importer> {
                    Existing(usize),
                    New(Edge, (I::VertexId, I::VertexId)),
                }
                // dummy values, will be filled later
                let mut face_edges = [EdgeCreation::<I>::Existing(0), EdgeCreation::Existing(0), EdgeCreation::Existing(0)];
                let mut face_vertices = [VertexIndex::from(0); 3];

                for ((i, face_edge), face_vertex) in (0 .. 3).zip(&mut face_edges).zip(&mut face_vertices) {
                    *face_vertex = face_verts[tri[i]];
                    //let v0 = face_verts_orig[tri[i]];
                    //let v1 = face_verts_orig[tri[(i + 1) % 3]];
                    let v0 = face_verts[tri[i]];
                    let v1 = face_verts[tri[(i + 1) % 3]];
                    let v0 = obj.vertex_map(v0);
                    let v1 = obj.vertex_map(v1);
                    let mut i_edge_candidate = edge_map.iter().position(|(p0, p1)| (p0, p1) == (&v0, &v1) || (p0, p1) == (&v1, &v0));

                    if let Some(i_edge) = i_edge_candidate {
                        if edges[i_edge].f1.is_some() {
                            // Maximum 2 faces per edge, additional faces will clone the edge and be disconnected
                            println!("Warning: three-faced edge #{i_edge}");
                            i_edge_candidate = None;
                        } else if edge_map[i_edge] != (v1, v0) {
                            // The found edge should be inverted: (v1,v0), unless you are doing a Moebius strip or something weird. This is mostly harmless, though.
                            println!("Warning: inverted edge #{i_edge}: {v0:?}-{v1:?}");
                        }
                    }

                    *face_edge = match i_edge_candidate {
                        Some(i_edge) => {
                            EdgeCreation::Existing(i_edge)
                        }
                        None => {
                            EdgeCreation::New(Edge {
                                f0: i_face,
                                f1: None,
                            }, (v0, v1))
                        }
                    }
                }

                // If the face uses the same egde twice, it is invalid
                match face_edges {
                    [EdgeCreation::Existing(a), EdgeCreation::Existing(b), _] |
                    [EdgeCreation::Existing(a), _, EdgeCreation::Existing(b)] |
                    [_, EdgeCreation::Existing(a), EdgeCreation::Existing(b)]
                        if a == b =>
                    {
                        return;
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

                face_map.push(face_id);
                faces.push(Face {
                    material: face_mat,
                    vertices: face_vertices,
                    edges,
                });
            }
        });

        let textures = obj.build_textures();
        let model = Model {
            textures,
            vertices,
            edges,
            faces,
        };
        (model, face_map, edge_map)
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
        let (v0, v1, _) = self[e.f0].vertices_with_edges().find(|&(_, _, e)| e == i_edge).unwrap();
        (self[v0].pos, self[v1].pos)
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
    pub fn textures(&self) -> impl Iterator<Item = &Texture> + '_ {
        self.textures.iter()
    }
    pub fn reload_textures<F: FnMut(&str) -> Result<DynamicImage>>(&mut self, mut f: F) -> Result<()> {
        for tex in &mut self.textures {
            tex.pixbuf = if tex.file_name.is_empty() {
                None
            } else {
                let img = f(&tex.file_name)?;
                Some(img)
            };
        }
        Ok(())
    }
    pub fn face_plane(&self, face: &Face) -> util_3d::Plane {
        util_3d::Plane::from_tri([
            self[face.vertices[0]].pos(),
            self[face.vertices[1]].pos(),
            self[face.vertices[2]].pos(),
        ])
    }
    pub fn edge_angle(&self, i_edge: EdgeIndex) -> Rad<f32> {
        let edge = &self[i_edge];
        match edge.faces() {
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
            _ => Rad::full_turn() / 2.0, //180 degrees
        }
    }
    pub fn face_area(&self, i_face: FaceIndex) -> f32 {
        let face = &self[i_face];
        // Area in 3D space should be almost equal to the area in 2D space,
        // except for very non planar n-gons, but if that is the case blame the user.
        let [a, b, c] = face
            .index_vertices()
            .map(|iv| self[iv].pos());
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
    pub fn vertices_with_edges(&self) -> impl Iterator<Item = (VertexIndex, VertexIndex, EdgeIndex)> + '_ {
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
    pub fn opposite_edge(&self, i_edge: EdgeIndex) -> VertexIndex {
        let i = self.edges.iter().position(|e| *e == i_edge).unwrap();
        self.vertices[(i + 2) % 3]
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
        } else if self.f1.map_or(false, |f| f == i_face) {
            true
        } else {
            // Model is inconsistent
            panic!();
        }
    }
}

pub trait Importer: Sized {
    type VertexId: Copy + Eq + std::fmt::Debug;
    type FaceId: Copy + Eq + std::fmt::Debug;

    fn import(file_name: &Path) -> Result<Self>;
    fn vertex_map(&self, i_v: VertexIndex) -> Self::VertexId;
    // return (has_normals, vertices)
    fn build_vertices(&self) -> (bool, Vec<Vertex>);
    fn face_count(&self) -> usize;
    fn for_each_face(&self, f: impl FnMut(Self::FaceId, &[VertexIndex], MaterialIndex));
    // Returns at least 1 texture, maybe default
    fn build_textures(&mut self) -> Vec<Texture>;

    fn compute_edge_status(&self, _edge_id: (Self::VertexId, Self::VertexId)) -> Option<EdgeStatus> { None }
    fn relocate_islands<'a>(&mut self, _model: &Model, _islands: impl Iterator<Item=&'a mut Island>) -> bool { false }
    fn build_options(&mut self) -> Option<PaperOptions> { None }
}
pub struct WaveObjImporter {
    obj: waveobj::Model,
    texture_map: FxHashMap<String, (String, DynamicImage)>,
    // VertexIndex -> FaceVertex
    all_vertices: Vec<waveobj::FaceVertex>,
}

impl Importer for WaveObjImporter {
    type VertexId = u32;
    type FaceId = u32;

    fn import(file_name: &Path) -> Result<Self> {
        let f = std::fs::File::open(file_name)?;
        let f = std::io::BufReader::new(f);
        let (matlib, obj) = waveobj::Model::from_reader(f)?;
        let matlib = match matlib {
            Some(matlib) => {
                Some(waveobj::solve_find_matlib_file(matlib.as_ref(), file_name)
                    .ok_or_else(|| anyhow!("{} matlib not found", matlib))?)
            }
            None => None,
        };
        let mut texture_map = FxHashMap::default();

        if let Some(matlib) = matlib {
            // Textures are read from the .mtl file
            let err_mtl = || format!("Error reading matlib file {}", matlib.display());
            let f = std::fs::File::open(&matlib)
                .with_context(err_mtl)?;
            let f = std::io::BufReader::new(f);

            for lib in waveobj::Material::from_reader(f)
                .with_context(err_mtl)?
            {
                if let Some(map) = lib.map() {
                    let err_map = || format!("Error reading texture file {map}");
                    if let Some(map) = waveobj::solve_find_matlib_file(map.as_ref(), &matlib) {
                        let img = image::io::Reader::open(&map)
                            .with_context(err_map)?
                            .with_guessed_format()
                            .with_context(err_map)?
                            .decode()
                            .with_context(err_map)?;
                        let map_name = map.file_name().and_then(|f| f.to_str())
                            .ok_or_else(|| anyhow!("Invalid texture name"))?;
                        texture_map.insert(lib.name().to_owned(), (map_name.to_owned(), img));
                    } else {
                        return Err(anyhow!("{} texture from {} matlib not found", map, matlib.display()));
                    }
                }
            }
        }

        // Remove duplicated vertices by adding them into a set
        let all_vertices: FxHashSet<waveobj::FaceVertex> =
            obj.faces()
                .iter()
                .flat_map(|f| f.vertices())
                .copied()
                .collect();
        //Fix the order into a vector, indexed by VertexIndex
        let all_vertices = Vec::from_iter(all_vertices);

        Ok(WaveObjImporter {
            obj,
            texture_map,
            all_vertices,
        })
    }
    fn vertex_map(&self, i_v: VertexIndex) -> Self::VertexId {
        self.all_vertices[usize::from(i_v)].v()
    }
    fn build_vertices(&self) -> (bool, Vec<Vertex>) {
        let mut has_normals = true;
        let vs = self.all_vertices
            .iter()
            .map(|fv| {
                let uv = if let Some(t) = fv.t() {
                    Vector2::from(*self.obj.texcoord_by_index(t))
                } else {
                    // If there is no texture coordinates there will be no textures so this value does not matter.
                    // A zero is easier to work with than an Option<Vector2>.
                    Vector2::zero()
                };
                let normal = if let Some(n) = fv.n() {
                    Vector3::from(*self.obj.normal_by_index(n))
                } else {
                    has_normals = false;
                    Vector3::zero()
                };
                Vertex {
                    pos: Vector3::from(*self.obj.vertex_by_index(fv.v())),
                    normal,
                    uv: Vector2::new(uv.x, 1.0 - uv.y),
                }
            })
            .collect();
        (has_normals, vs)
    }
    fn face_count(&self) -> usize {
        self.obj.faces().len()
    }
    fn for_each_face(&self, mut f: impl FnMut(Self::FaceId, &[VertexIndex], MaterialIndex)) {
        for (face_id, face) in self.obj.faces().iter().enumerate() {
            let verts: Vec<_> = face
                .vertices()
                .iter()
                .map(|fv| VertexIndex::from(self.all_vertices.iter().position(|v| v == fv).unwrap()))
                .collect();
            let mat = MaterialIndex::from(face.material());
            f(face_id as u32, &verts, mat)
        }
    }
    fn build_textures(&mut self) -> Vec<Texture> {
        let mut textures: Vec<_> = self.obj.materials().map(|s| {
            let tex = self.texture_map.remove(s);
            let (file_name, pixbuf) = match tex {
                Some((n, p)) => (n, Some(p)),
                None => (String::new(), None)
            };

            Texture {
                file_name,
                pixbuf,
            }
        }).collect();
        if textures.is_empty() {
            textures.push(Texture::default());
        }
        textures
    }
}

////////////
pub struct PepakuraImporter {
    pdo: pepakura::Pdo,
    //VertexIndex -> (obj_id, face_id, vert_in_face)
    vertex_map: Vec<(u32, u32, u32)>,
    options: Option<PaperOptions>,
}

impl Importer for PepakuraImporter {
    // (obj_id, vertex_id)
    type VertexId = (u32, u32);
    // (obj_id, face_id)
    type FaceId = (u32, u32);

    fn import(file_name: &Path) -> Result<Self> {
        let f = std::fs::File::open(file_name)?;
        let f = std::io::BufReader::new(f);
        let pdo = pepakura::Pdo::from_reader(f)?;

        let vertex_map: Vec<(u32, u32, u32)> = pdo
            .objects()
            .iter()
            .enumerate()
            .flat_map(|(i_o, obj)| {
                obj.faces
                    .iter()
                    .enumerate()
                    .flat_map(move |(i_f, f)| {
                        (0 .. f.verts.len()).map(move |i_vf| (i_o as u32, i_f as u32, i_vf as u32))
                    })
            })
            .collect();

        Ok(PepakuraImporter {
            pdo,
            vertex_map,
            options: None,
        })
    }
    fn build_vertices(&self) -> (bool, Vec<Vertex>) {
        let vs = self.vertex_map
            .iter()
            .map(|&(i_o, i_f, i_vf)| {
                let obj = &self.pdo.objects()[i_o as usize];
                let f = &obj.faces[i_f as usize];
                let v_f = &f.verts[i_vf as usize];
                let v = &obj.vertices[v_f.i_v as usize];

                Vertex {
                    pos: v.v,
                    normal: f.normal,
                    uv: v_f.uv,
                }
            })
            .collect();
        (true, vs)
    }
    fn vertex_map(&self, i_v: VertexIndex) -> Self::VertexId {
        let (i_o, i_f, i_vf) = self.vertex_map[usize::from(i_v)];
        let i_v = self.pdo.objects()[i_o as usize].faces[i_f as usize].verts[i_vf as usize].i_v;
        (i_o, i_v)
    }
    fn face_count(&self) -> usize {
        self.pdo.objects()
            .iter()
            .map(|o| o.faces.len())
            .sum()
    }
    fn for_each_face(&self, mut f: impl FnMut(Self::FaceId, &[VertexIndex], MaterialIndex)) {
        for (obj_id, obj) in self.pdo.objects().iter().enumerate() {
            let obj_id = obj_id as u32;
            for (face_id, face) in obj.faces.iter().enumerate() {
                let face_id = face_id as u32;
                let verts: Vec<VertexIndex> = (0 .. face.verts.len())
                    .map(|v_f| {
                        let id = (obj_id, face_id, v_f as u32);
                        let i = self.vertex_map.iter().position(|x| x == &id).unwrap();
                        VertexIndex::from(i)
                    })
                    .collect();
                // We will add a default material at the end of the textures, so map any out-of bounds to that
                let mat_index = face.mat_index.min(self.pdo.materials().len() as u32);
                let mat = MaterialIndex(mat_index);
                f((obj_id, face_id), &verts, mat)
            }
        }
    }
    fn build_textures(&mut self) -> Vec<Texture> {
        let mut textures: Vec<_> = self.pdo.materials_mut()
            .iter_mut()
            .map(|mat| {
                let pixbuf = mat.texture.take().and_then(|t| {
                    let img = ImageBuffer::from_raw(t.width, t.height, t.data);
                    img.map(|b| DynamicImage::ImageRgb8(b))
                });
                Texture {
                    file_name: mat.name.clone() + ".png",
                    pixbuf,
                }
            })
            .collect();
        textures.push(Texture::default());
        textures
    }
    fn compute_edge_status(&self, edge_id: (Self::VertexId, Self::VertexId)) -> Option<EdgeStatus> {
        let ((obj_id, v0_id), (_, v1_id)) = edge_id;
        let vv = (v0_id, v1_id);
        let obj = &self.pdo.objects()[obj_id as usize];
        let Some(edge) = obj.edges
            .iter()
            .find(|&e| vv == (e.i_v1, e.i_v2) || vv == (e.i_v2, e.i_v1))
            else { return None };
        if edge.connected {
            Some(EdgeStatus::Joined)
        } else {
            let v_f = obj.faces[edge.i_f1 as usize].verts.iter().find(|v_f| v_f.i_v == edge.i_v1).unwrap();
            if v_f.flap {
                Some(EdgeStatus::Cut(TabSide::True))
            } else {
                None
            }
        }
    }
    fn relocate_islands<'a>(&mut self, model: &Model, islands: impl Iterator<Item=&'a mut Island>) -> bool {
        let Some(unfold) = self.pdo.unfold() else { return false; };

        let margin = Vector2::new(self.pdo.settings().margin_side as f32, self.pdo.settings().margin_top as f32);
        let page_size = self.pdo.settings().page_size;
        let area_size = page_size - 2.0 * margin;

        let mut options = PaperOptions::default();
        options.scale = unfold.scale;
        options.page_size = (page_size.x, page_size.y);
        options.margin = (margin.y, margin.x, margin.x, margin.y);

        let mut n_cols = 0;
        let mut max_page = (0, 0);
        for island in islands {
            let face = &model[island.root_face()];
            let [i_v0, i_v1, _] = face.index_vertices();
            let (ip_obj, ip_face, ip_v0) = self.vertex_map[usize::from(i_v0)];
            let (_, _, ip_v1) = self.vertex_map[usize::from(i_v1)];
            let p_face = &self.pdo.objects()[ip_obj as usize].faces[ip_face as usize];
            let vf0 = p_face.verts[ip_v0 as usize].pos2d;
            let vf1 = p_face.verts[ip_v1 as usize].pos2d;
            let i_part = p_face.part_index;

            let normal = model.face_plane(face);
            let pv0 = normal.project(&model[i_v0].pos(), options.scale);
            let pv1 = normal.project(&model[i_v1].pos(), options.scale);

            let part = &unfold.parts[i_part as usize];

            let rot = (pv1 - pv0).angle(vf1 - vf0);
            let loc = vf0 - pv0 + part.bb.v0;

            let mut col = loc.x.div_euclid(area_size.x) as i32;
            let mut row = loc.y.div_euclid(area_size.y) as i32;
            let loc = Vector2::new(loc.x.rem_euclid(area_size.x), loc.y.rem_euclid(area_size.y));
            let loc = loc + margin;

            // Some models use negative pages to hide pieces
            if col < 0 || row < 0 {
                col = -1;
                row = 0;
            } else {
                let row = row as u32;
                let col = col as u32;
                n_cols = n_cols.max(col);
                if row > max_page.0 || (row == max_page.0 && col > max_page.1) {
                    max_page = (row, col);
                }
            }

            let loc = options.page_to_global(PageOffset { row, col, offset: loc });
            island.reset_transformation(island.root_face(), rot, loc);
        }
        // 0-based
        options.page_cols = n_cols + 1;
        options.pages = max_page.0 * options.page_cols + max_page.1 + 1;

        self.options = Some(options);
        true
    }
    fn build_options(&mut self) -> Option<PaperOptions> {
        self.options
            .take()
            .map(|mut options| {
                if let Some(a) = self.pdo.settings().fold_line_hide_angle {
                    options.hidden_line_angle = (180 - a) as f32;
                }
                options
            })
    }
}
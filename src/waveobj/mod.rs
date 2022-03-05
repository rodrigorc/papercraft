#![allow(dead_code)]

use std::{io::BufRead, collections::BTreeMap};
use anyhow::{anyhow, Result};

#[derive(Clone, Debug)]
pub struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct Face {
    idx: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Model {
    name: String,
    material: Option<String>,
    vs: Vec<Vertex>,
    faces: Vec<Face>,
}

impl Model {
    //Returns (matlibs, models)
    pub fn from_reader<R: BufRead>(r: R) -> Result<(Vec<String>, Vec<Model>)> {
        let syn_error = || anyhow!("invalid obj syntax");

        let mut objs = Vec::new();

        let mut material_libs = Vec::new();
        #[derive(Default)]
        struct ModelData {
            name: Option<String>,
            material: Option<String>,
            vertices: Vec<Vertex>,
            faces: Vec<Face>,
            pos: Vec<[f32; 3]>,
            normals: Vec<[f32; 3]>,
            uvs: Vec<[f32; 2]>,
            idx_map: BTreeMap<(usize, usize, usize), usize>,
        }
        impl ModelData {
            fn build(&mut self) -> Option<Model> {
                self.pos.clear();
                self.normals.clear();
                self.uvs.clear();
                self.idx_map.clear();

                //Build the model even if empty, to always reset all self fields
                let m = Model {
                    name: self.name.take().unwrap_or_default(),
                    material: self.material.take(),
                    vs: std::mem::take(&mut self.vertices),
                    faces: std::mem::take(&mut self.faces),
                };
                if m.faces.is_empty() {
                    None
                } else {
                    Some(m)
                }
            }
        }
        let mut data = ModelData::default();


        for line in r.lines() {
            let line = line?;
            let line = line.trim();
            //skip empty and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut words = line.split(' ');
            let first = words.next().ok_or_else(syn_error)?;
            match first {
                "o" => {
                    objs.extend(data.build());

                    let name = words.next().ok_or_else(syn_error)?;
                    data.name = Some(String::from(name));
                }
                "v" => {
                    let x: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let y: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let z: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    data.pos.push([x, y, z]);
                }
                "vt" => {
                    let u: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let v: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    data.uvs.push([u, v]);
                }
                "vn" => {
                    let x: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let y: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let z: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    data.normals.push([x, y, z]);
                }
                "f" => {
                    let mut idx_face = Vec::new();
                    for fv in words {
                        let mut vals = fv.split('/');
                        let v = vals.next().ok_or_else(syn_error)?.parse::<usize>()? - 1;
                        let t = vals.next().ok_or_else(syn_error)?.parse::<usize>()? - 1;
                        let n = vals.next().ok_or_else(syn_error)?.parse::<usize>()? - 1;

                        use std::collections::btree_map::Entry::*;

                        let idx_entry = data.idx_map.entry((v, t, n));
                        let idx = match idx_entry {
                            Occupied(e) => {
                                *e.get()
                            }
                            Vacant(_) => {
                                data.vertices.push(Vertex {
                                    pos: *data.pos.get(v).ok_or_else(syn_error)?,
                                    normal: *data.normals.get(n).ok_or_else(syn_error)?,
                                    uv: *data.uvs.get(t).ok_or_else(syn_error)?,
                                });
                                data.vertices.len() - 1
                            }
                        };
                        idx_face.push(u32::try_from(idx)?);
                    }
                    data.faces.push(Face {
                        idx: idx_face
                    })
                }
                "mtllib" => {
                    let lib = words.next().ok_or_else(syn_error)?;
                    material_libs.push(String::from(lib));
                }
                "usemtl" => {
                    let mtl = words.next().ok_or_else(syn_error)?;
                    data.material = Some(String::from(mtl));
                }
                "s" => { /* smoothing is ignored */}
                p => {
                    println!("{}??", p);
                }
            }
        }

        objs.extend(data.build());

        Ok((material_libs, objs))
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn material(&self) -> Option<&str> {
        self.material.as_deref()
    }
    pub fn faces(&self) -> &[Face] {
        &self.faces
    }
    pub fn vertices(&self) -> &[Vertex] {
        &self.vs
    }
    pub fn vertex_by_index(&self, idx: u32) -> &Vertex {
        &self.vs[idx as usize]
    }
}

impl Face {
    pub fn indices(&self) -> &[u32] {
        &self.idx
    }
}

impl Vertex {
    pub fn pos(&self) -> &[f32; 3] {
        &self.pos
    }
    pub fn normal(&self) -> &[f32; 3] {
        &self.normal
    }
    pub fn uv(&self) -> &[f32; 2] {
        &self.uv
    }
}

#[derive(Clone, Debug)]
pub struct Material {
    name: String,
    map: Option<String>,
}

impl Material {
    pub fn from_reader<R: BufRead>(r: R) -> Result<Vec<Material>> {
        let syn_error = || anyhow!("invalid mtl syntax");

        let mut mats = Vec::new();

        #[derive(Default)]
        struct MaterialData {
            name: Option<String>,
            map: Option<String>,
        }

        impl MaterialData {
            fn build(&mut self) -> Option<Material> {
                if self.name.is_none() {
                    *self = MaterialData::default();
                    return None;
                }

                let m = Material {
                    name: self.name.take().unwrap(),
                    map: self.map.take(),
                };

                Some(m)
            }
        }
        let mut data = MaterialData::default();

        for line in r.lines() {
            let line = line?;
            let line = line.trim();
            //skip empty and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut words = line.split(' ');
            let first = words.next().ok_or_else(syn_error)?;
            match first {
                "newmtl" => {
                    mats.extend(data.build());

                    let name = words.next().ok_or_else(syn_error)?;
                    data.name = Some(String::from(name));
                }
                "map_Kd" => {
                    let map = words.next().ok_or_else(syn_error)?;
                    data.map = Some(String::from(map));
                }
                p => {
                    println!("{}??", p);
                }
           }
        }
        mats.extend(data.build());
        Ok(mats)
    }
    pub fn map(&self) -> Option<&str> {
        self.map.as_deref()
    }
    pub fn name(&self) -> &str {
        &self.name
    }
}
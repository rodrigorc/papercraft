use std::io::BufRead;
use anyhow::{anyhow, Result};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct FaceVertex {
    v: u32,
    t: u32,
    n: u32,
}

#[derive(Clone, Debug)]
pub struct Face {
    verts: Vec<FaceVertex>,
}

#[derive(Clone, Debug)]
pub struct Model {
    #[allow(dead_code)]
    name: String,
    material: Option<String>,
    vs0: Vec<[f32; 3]>,
    ns: Vec<[f32; 3]>,
    ts: Vec<[f32; 2]>,
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
            faces: Vec<Face>,
            pos: Vec<[f32; 3]>,
            normals: Vec<[f32; 3]>,
            uvs: Vec<[f32; 2]>,
            //idx_map: BTreeMap<(usize, usize, usize), usize>,
        }
        impl ModelData {
            fn build(&mut self) -> Option<Model> {
                //self.pos.clear();
                //self.normals.clear();
                //self.uvs.clear();
                //self.idx_map.clear();

                //Build the model even if empty, to always reset all self fields
                let m = Model {
                    name: self.name.take().unwrap_or_default(),
                    material: self.material.take(),
                    vs0: std::mem::take(&mut self.pos),
                    ns: std::mem::take(&mut self.normals),
                    ts: std::mem::take(&mut self.uvs),
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
                    let mut verts = Vec::new();
                    for fv in words {
                        let mut vals = fv.split('/');
                        let v = vals.next().ok_or_else(syn_error)?.parse::<usize>()? - 1;
                        let t = vals.next().ok_or_else(syn_error)?.parse::<usize>()? - 1;
                        let n = vals.next().ok_or_else(syn_error)?.parse::<usize>()? - 1;

                        let v = FaceVertex {
                            v: v as u32,
                            t: t as u32,
                            n: n as u32
                        };
                        verts.push(v);

                    }
                    data.faces.push(Face {
                        verts
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
    #[allow(dead_code)]
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn material(&self) -> Option<&str> {
        self.material.as_deref()
    }
    pub fn faces(&self) -> &[Face] {
        &self.faces
    }
    pub fn vertex_by_index(&self, idx: u32) -> &[f32; 3] {
        &self.vs0[idx as usize]
    }
    pub fn normal_by_index(&self, idx: u32) -> &[f32; 3] {
        &self.ns[idx as usize]
    }
    pub fn texcoord_by_index(&self, idx: u32) -> &[f32; 2] {
        &self.ts[idx as usize]
    }
}

impl Face {
    pub fn vertices(&self) -> &[FaceVertex] {
        &self.verts
    }
}

impl FaceVertex {
    pub fn v(&self) -> u32 {
        self.v
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn t(&self) -> u32 {
        self.t
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
use std::{io::BufRead, path::{Path, PathBuf}};
use anyhow::{anyhow, Result};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct FaceVertex {
    v: u32,
    t: Option<u32>,
    n: u32,
}

#[derive(Clone, Debug)]
pub struct Face {
    material: usize,
    verts: Vec<FaceVertex>,
}

#[derive(Clone, Debug)]
pub struct Model {
    materials: Vec<String>,
    vs: Vec<[f32; 3]>,
    ns: Vec<[f32; 3]>,
    ts: Vec<[f32; 2]>,
    faces: Vec<Face>,
}

impl Model {
    //Returns (matlib, model)
    pub fn from_reader<R: BufRead>(r: R) -> Result<(String, Model)> {
        let syn_error = || anyhow!("invalid obj syntax");

        let mut material_lib = String::new();
        let mut current_material: usize = 0;
        let mut data = Model {
            materials: Vec::new(),
            vs: Vec::new(),
            ns: Vec::new(),
            ts: Vec::new(),
            faces: Vec::new(),
        };

        for line in r.lines() {
            let line = line?;
            let line = line.trim();
            //skip empty and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut words = line.split_whitespace();
            let first = words.next().ok_or_else(syn_error)?;
            match first {
                "o" => {
                    // We combine all the objects into one.
                    // Fortunately the numbering of vertices and faces is global to the file not to the object, so nothing to do here.
                }
                "v" => {
                    let x: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let y: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let z: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    data.vs.push([x, y, z]);
                }
                "vt" => {
                    let u: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let v: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    data.ts.push([u, v]);
                }
                "vn" => {
                    let x: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let y: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    let z: f32 = words.next().ok_or_else(syn_error)?.parse()?;
                    data.ns.push([x, y, z]);
                }
                "f" => {
                    let mut verts = Vec::new();
                    for fv in words {
                        let mut vals = fv.split('/');
                        let v = vals.next().ok_or_else(syn_error)?.parse::<usize>()? - 1;
                        let t = vals.next().ok_or_else(syn_error)?.parse::<usize>().ok().map(|x| x - 1);
                        let n = vals.next().ok_or_else(syn_error)?.parse::<usize>()? - 1;
                        if v >= data.vs.len() {
                            return Err(anyhow!("vertex index out of range"));
                        }
                        if matches!(t, Some(t) if t >= data.ts.len()) {
                            return Err(anyhow!("texture index out of range"));
                        }
                        if n >= data.ns.len() {
                            return Err(anyhow!("normal index out of range"));
                        }
                        let v = FaceVertex {
                            v: v as u32,
                            t: t.map(|t| t as u32),
                            n: n as u32
                        };
                        verts.push(v);
                    }
                    data.faces.push(Face {
                        material: current_material,
                        verts
                    })
                }
                "mtllib" => {
                    let lib = words.next().ok_or_else(syn_error)?;
                    material_lib = lib.to_owned();
                }
                "usemtl" => {
                    let mtl = words.next().ok_or_else(syn_error)?;
                    if let Some(p) = data.materials.iter().position(|m| m == mtl) {
                        current_material = p;
                    } else {
                        current_material = data.materials.len();
                        data.materials.push(String::from(mtl));
                    }
                }
                "s" => { /* smoothing is ignored */}
                p => {
                    println!("{}??", p);
                }
            }
        }

        Ok((material_lib, data))
    }
    pub fn materials(&self) -> impl Iterator<Item = &str> + '_ {
        self.materials.iter().map(|s| &s[..])
    }
    pub fn faces(&self) -> &[Face] {
        &self.faces
    }
    pub fn vertex_by_index(&self, idx: u32) -> &[f32; 3] {
        &self.vs[idx as usize]
    }
    pub fn normal_by_index(&self, idx: u32) -> &[f32; 3] {
        &self.ns[idx as usize]
    }
    pub fn texcoord_by_index(&self, idx: u32) -> &[f32; 2] {
        &self.ts[idx as usize]
    }
}

impl Face {
    pub fn material(&self) -> usize {
        self.material
    }
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
    pub fn t(&self) -> Option<u32> {
        self.t
    }
}

pub fn solve_find_matlib_file(mtl: &Path, obj: &Path) -> Option<PathBuf> {
    let obj_dir = match obj.parent() {
        None => ".".into(),
        Some(d) => d.to_owned(),
    };
    if mtl.is_relative() {
        // First find the mtl in the same directory as the obj, using the local path
        let mut dir = obj_dir.clone();
        dir.push(mtl);
        if dir.exists() {
            return Some(dir);
        }
        // Then without the mtl path
        dir = obj_dir.clone();
        dir.push(mtl.file_name().unwrap());
        if dir.exists() {
            return Some(dir);
        }
    } else {
        // If mtl is absolute, first try the real file
        if mtl.exists() {
            return Some(mtl.to_owned());
        }
        // Then try the same name in a local path
        let mut dir = obj_dir.clone();
        dir.push(mtl.file_name().unwrap());
        if dir.exists() {
            return Some(dir);
        }
    }
    None
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
            let mut words = line.split_whitespace();
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
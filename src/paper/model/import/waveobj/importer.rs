use super::super::*;
use super::data;
use cgmath::Zero;
use fxhash::{FxHashMap, FxHashSet};
use image::DynamicImage;


pub struct WaveObjImporter {
    obj: data::Model,
    texture_map: FxHashMap<String, (String, DynamicImage)>,
    // VertexIndex -> FaceVertex
    all_vertices: Vec<data::FaceVertex>,
}

impl WaveObjImporter {
    pub fn new<R: BufRead>(f: R, file_name: &Path) -> Result<Self> {
        let (matlib, obj) = data::Model::from_reader(f)?;
        let matlib = match matlib {
            Some(matlib) => {
                Some(data::solve_find_matlib_file(matlib.as_ref(), file_name)
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

            for lib in data::Material::from_reader(f)
                .with_context(err_mtl)?
            {
                if let Some(map) = lib.map() {
                    let err_map = || format!("Error reading texture file {map}");
                    if let Some(map) = data::solve_find_matlib_file(map.as_ref(), &matlib) {
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
        let all_vertices: FxHashSet<data::FaceVertex> =
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
}

impl Importer for WaveObjImporter {
    type VertexId = u32;
    type FaceId = u32;

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

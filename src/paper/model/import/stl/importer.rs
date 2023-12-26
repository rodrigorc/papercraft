use super::data;
use super::super::*;
use cgmath::Zero;

pub struct StlImporter {
    stl: data::Stl,
}

impl StlImporter {
    pub fn new<R: BufRead>(f: R) -> Result<StlImporter> {
        let stl = data::Stl::new(f)?;

        Ok(StlImporter {
            stl,
        })
    }
}

impl Importer for StlImporter {
    // STL doesn't have vertex identity, we consider them the same if they are bitwise identical
    type VertexId = [u32; 3];

    fn build_vertices(&self) -> (bool, Vec<Vertex>) {
        // The VertexIndex is {3*nface, +1, +2}
        let mut has_normals = false;
        let vxs = self.stl.triangles()
            .iter()
            .flat_map(|tri| {
                if !has_normals && tri.normal != Vector3::zero() {
                    has_normals = true;
                }
                tri.vertices.iter().map(|v| {
                    Vertex {
                        // swizzle the coordinates to make the model look at the front
                        pos: Vector3::new(v.x, v.z, -v.y),
                        normal: tri.normal,
                        uv: Vector2::zero(),
                    }
                })
            })
            .collect();
        (has_normals, vxs)
    }

    fn vertex_map(&self, i_v: VertexIndex) -> Self::VertexId {
        //self.vx_map[usize::from(i_v)]
        let i_v = usize::from(i_v);
        let v: [f32; 3] = self.stl.triangles()[i_v / 3].vertices[i_v % 3].into();
        v.map(|f| f.to_bits())
    }

    fn face_count(&self) -> usize {
        self.stl.triangles().len()
    }

    fn faces<'s>(&'s self) -> impl Iterator<Item = (impl AsRef<[VertexIndex]>, MaterialIndex)> + 's {
        (0 .. self.stl.triangles().len() as u32).map(|i_face| {
            let i_v0 = 3 * i_face;
            ([VertexIndex(i_v0), VertexIndex(i_v0 + 1), VertexIndex(i_v0 + 2)], MaterialIndex(0))
        })
    }

    fn build_textures(&self) -> Vec<Texture> {
        vec![Texture::default()]
    }
}

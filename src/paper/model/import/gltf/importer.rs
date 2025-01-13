use anyhow::Result;
use image::DynamicImage;
use std::cell::Cell;
use std::io::BufRead;

use super::super::*;
use super::data::*;

pub struct GltfImporter {
    // Cell to avoid cloning
    images: Cell<Vec<(String, DynamicImage)>>,
    // 3 vertices per face
    vertices: Vec<Vertex>,
    // 1 tex_id per face
    textures: Vec<u32>,
    has_normals: bool,
}

impl GltfImporter {
    pub fn new<R: BufRead>(mut f: R, file_name: &Path) -> Result<GltfImporter> {
        let mut data = Vec::new();
        f.read_to_end(&mut data)?;

        let gltf = Gltf::parse(&data, file_name)?;
        Self::new_inner(gltf)
    }

    fn new_inner(gltf: Gltf) -> Result<GltfImporter> {
        let images = gltf.load_images()?;
        let mut vertices = Vec::new();
        let mut textures = Vec::new();
        let mut has_normals = true;
        gltf.process_scene(|tex, vs, ns, uv| {
            for i in 0..3 {
                vertices.push(Vertex {
                    pos: vs[i],
                    normal: ns.map(|n| n[i]).unwrap_or_else(|| {
                        has_normals = false;
                        Vector3::new(0.0, 0.0, 0.0)
                    }),
                    uv: uv[i],
                });
            }
            // texture 0 is the no-tex
            textures.push(tex.map(|t| t + 1).unwrap_or(0))
        })?;

        Ok(GltfImporter {
            images: Cell::new(images),
            vertices,
            has_normals,
            textures,
        })
    }
}

impl Importer for GltfImporter {
    type VertexId = [u32; 3];

    fn vertex_map(&self, i_v: VertexIndex) -> Self::VertexId {
        let i_v = usize::from(i_v);
        let v: [f32; 3] = self.vertices[i_v].pos.into();
        v.map(|f| f.to_bits())
    }

    fn build_vertices(&self) -> (bool, Vec<Vertex>) {
        let vs = self.vertices.clone();
        (self.has_normals, vs)
    }

    fn face_count(&self) -> usize {
        self.vertices.len() / 3
    }

    fn faces(&self) -> impl Iterator<Item = (impl AsRef<[VertexIndex]>, MaterialIndex)> + '_ {
        (0..self.vertices.len() / 3).map(|i| {
            let n = 3 * i as u32;
            (
                [VertexIndex(n), VertexIndex(n + 1), VertexIndex(n + 2)],
                MaterialIndex(self.textures[i]),
            )
        })
    }

    fn build_textures(&self) -> Vec<Texture> {
        let mut texs = vec![Texture::default()];
        for (file_name, pixbuf) in self.images.take() {
            texs.push(Texture {
                file_name,
                pixbuf: Some(pixbuf),
            })
        }
        texs
    }
}

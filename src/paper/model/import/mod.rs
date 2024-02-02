use anyhow::{anyhow, bail, Context, Result};
use std::io::{BufRead, Read};
use std::panic::catch_unwind;
use std::path::Path;

use super::{EdgeStatus, Island, MaterialIndex, Model, PaperOptions, Texture, Vertex, VertexIndex};
use crate::paper::{FlapSide, PageOffset, Papercraft};
use crate::util_3d::{Vector2, Vector3};

pub mod pepakura;
pub mod stl;
pub mod waveobj;

fn read_u8(rdr: &mut impl Read) -> Result<u8> {
    let mut x = [0; 1];
    rdr.read_exact(&mut x)?;
    Ok(x[0])
}
fn read_bool(rdr: &mut impl Read) -> Result<bool> {
    Ok(read_u8(rdr)? != 0)
}
fn read_u16(rdr: &mut impl Read) -> Result<u16> {
    let mut x = [0; 2];
    rdr.read_exact(&mut x)?;
    Ok(u16::from_le_bytes(x))
}
fn read_u32(rdr: &mut impl Read) -> Result<u32> {
    let mut x = [0; 4];
    rdr.read_exact(&mut x)?;
    Ok(u32::from_le_bytes(x))
}
fn read_u64(rdr: &mut impl Read) -> Result<u64> {
    let mut x = [0; 8];
    rdr.read_exact(&mut x)?;
    Ok(u64::from_le_bytes(x))
}
fn read_f32(rdr: &mut impl Read) -> Result<f32> {
    let mut x = [0; 4];
    rdr.read_exact(&mut x)?;
    Ok(f32::from_le_bytes(x))
}
fn read_f64(rdr: &mut impl Read) -> Result<f64> {
    let mut x = [0; 8];
    rdr.read_exact(&mut x)?;
    Ok(f64::from_le_bytes(x))
}
fn read_vector2_f64(rdr: &mut impl Read) -> Result<Vector2> {
    let x = read_f64(rdr)? as f32;
    let y = read_f64(rdr)? as f32;
    Ok(Vector2::new(x, y))
}
fn read_vector3_f64(rdr: &mut impl Read) -> Result<Vector3> {
    let x = read_f64(rdr)? as f32;
    let y = read_f64(rdr)? as f32;
    let z = read_f64(rdr)? as f32;
    Ok(Vector3::new(x, y, z))
}
fn _read_vector2_f32(rdr: &mut impl Read) -> Result<Vector2> {
    let x = read_f32(rdr)? as f32;
    let y = read_f32(rdr)? as f32;
    Ok(Vector2::new(x, y))
}
fn read_vector3_f32(rdr: &mut impl Read) -> Result<Vector3> {
    let x = read_f32(rdr)? as f32;
    let y = read_f32(rdr)? as f32;
    let z = read_f32(rdr)? as f32;
    Ok(Vector3::new(x, y, z))
}

pub trait Importer: Sized {
    type VertexId: Copy + Eq + std::fmt::Debug;

    fn vertex_map(&self, i_v: VertexIndex) -> Self::VertexId;
    // return (has_normals, vertices)
    fn build_vertices(&self) -> (bool, Vec<Vertex>);
    fn face_count(&self) -> usize;

    fn faces<'s>(&'s self)
        -> impl Iterator<Item = (impl AsRef<[VertexIndex]>, MaterialIndex)> + 's;

    // Returns at least 1 texture, maybe default.
    // As a risky optimization, it can consume the texture data, call only once
    fn build_textures(&self) -> Vec<Texture>;

    // Optional functions
    fn compute_edge_status(
        &self,
        _edge_id: (Self::VertexId, Self::VertexId),
    ) -> Option<EdgeStatus> {
        None
    }
    fn relocate_islands<'a>(
        &self,
        _model: &Model,
        _islands: impl Iterator<Item = &'a mut Island>,
    ) -> bool {
        false
    }
    fn build_options(&self) -> Option<PaperOptions> {
        None
    }
}

// Returns (model, is_native_format)
pub fn import_model_file(file_name: &Path) -> Result<(Papercraft, bool)> {
    // Models have a lot of indices and unwraps, a corrupted file could easily panic
    match catch_unwind(|| import_model_file_priv(file_name)) {
        Ok(res) => res,
        Err(err) => {
            if let Some(msg) = err.downcast_ref::<&str>() {
                bail!(
                    "Panic importing the model '{}'!\n{}",
                    file_name.display(),
                    msg
                );
            } else {
                bail!("Panic importing the model '{}'!", file_name.display());
            }
        }
    }
}
pub fn import_model_file_priv(file_name: &Path) -> Result<(Papercraft, bool)> {
    let ext = match file_name.extension() {
        None => String::new(),
        Some(ext) => {
            let mut ext = ext.to_string_lossy().into_owned();
            ext.make_ascii_lowercase();
            ext
        }
    };

    let f = std::fs::File::open(file_name)
        .with_context(|| format!("Error opening file {}", file_name.display()))?;
    let f = std::io::BufReader::new(f);
    let mut is_native = false;

    let papercraft = match ext.as_str() {
        "craft" => {
            is_native = true;
            Papercraft::load(f)
                .with_context(|| format!("Error reading Papercraft file {}", file_name.display()))?
        }
        "pdo" => {
            let importer = pepakura::PepakuraImporter::new(f)
                .with_context(|| format!("Error reading Pepakura file {}", file_name.display()))?;
            Papercraft::import(importer)
        }
        "stl" => {
            let importer = stl::StlImporter::new(f)
                .with_context(|| format!("Error reading STL file {}", file_name.display()))?;
            Papercraft::import(importer)
        }
        "mtl" => {
            anyhow::bail!(
                "MTL are material files for OBJ models. Try opening the OBJ file instead."
            );
        }
        // unknown extensions are tried as obj, that was the default previously
        "obj" | _ => {
            let importer = waveobj::WaveObjImporter::new(f, file_name)
                .with_context(|| format!("Error reading Wavefront file {}", file_name.display()))?;
            Papercraft::import(importer)
        }
    };
    Ok((papercraft, is_native))
}

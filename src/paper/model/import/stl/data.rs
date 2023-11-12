use std::io::BufRead;
use anyhow::{Result, bail};
use cgmath::Zero;

use crate::paper::import::*;

#[derive(Debug)]
pub struct Stl {
    tris: Vec<Triangle>,
}

#[derive(Debug)]
pub struct Triangle {
    pub normal: Vector3,
    pub vertices: [Vector3; 3],
}

impl Stl {
    pub fn new<R: BufRead>(mut f: R) -> Result<Stl> {
        let mut hdr = [0; 80];
        f.read_exact(&mut hdr[.. 5])?;
        if &hdr[.. 5] == b"solid" {
            Self::new_text(f)
        } else {
            f.read_exact(&mut hdr[5 .. ])?;
            Self::new_binary(f)
        }
    }
    fn new_binary<R: BufRead>(mut f: R) -> Result<Stl> {
        let rdr = &mut f;
        let n_tris = read_u32(rdr)?;
        let mut tris = Vec::with_capacity(n_tris as usize);
        for _ in 0 .. n_tris {
            let normal = read_vector3_f32(rdr)?;
            let v0 = read_vector3_f32(rdr)?;
            let v1 = read_vector3_f32(rdr)?;
            let v2 = read_vector3_f32(rdr)?;
            let _attr = read_u16(rdr)?;
            tris.push(Triangle {
                normal,
                vertices: [v0, v1, v2],
            })
        }

        Ok(Stl {
            tris,
        })
   }
   fn new_text<R: BufRead>(mut f: R) -> Result<Stl> {
        let mut line = String::new();
        // "{solid} NAME"
        f.read_line(&mut line)?;
        let mut tris = Vec::new();
        loop {
            // "facet normal nx ny nz"
            line.clear();
            f.read_line(&mut line)?;
            let mut words = line.split_ascii_whitespace();
            let w = words.next();
            match w {
                Some("endsolid") => break,
                Some("facet") => {}
                _ => { bail!(r#"expected "facet""#); }
            }
            let w = words.next();
            if w != Some("normal") {
                bail!(r#"expected "normal""#);
            }
            let Some(nx) = words.next() else { bail!(r#"expected number"#) };
            let Some(ny) = words.next() else { bail!(r#"expected number"#) };
            let Some(nz) = words.next() else { bail!(r#"expected number"#) };
            let normal = Vector3::new(nx.parse()?, ny.parse()?, nz.parse()?);

            // do not bother parsing lines without values, they can only result in error
            // "outer loop"
            line.clear();
            f.read_line(&mut line)?;

            // 3 * ("vertex x y z")
            let mut vertices = [Vector3::zero(); 3];
            for vert in &mut vertices {
                line.clear();
                f.read_line(&mut line)?;
                let mut words = line.split_ascii_whitespace();
                let w = words.next();
                match w {
                    Some("vertex") => {}
                    _ => { bail!(r#"expected "vertex""#); }
                }
                let Some(vx) = words.next() else { bail!(r#"expected number"#) };
                let Some(vy) = words.next() else { bail!(r#"expected number"#) };
                let Some(vz) = words.next() else { bail!(r#"expected number"#) };
                *vert = Vector3::new(vx.parse()?, vy.parse()?, vz.parse()?);
            }

            // "end loop"
            line.clear();
            f.read_line(&mut line)?;
            // "end facet"
            line.clear();
            f.read_line(&mut line)?;
            tris.push(Triangle {
                normal,
                vertices,
            })
        }
        Ok(Stl {
            tris,
        })
   }
   pub fn triangles(&self) -> &[Triangle] {
        &self.tris
   }
}
use std::path::Path;

use crate::{
    paper::{EdgeStatus, FaceIndex, Papercraft, VertexIndex},
    util_3d::{Vector2, Vector3},
};
use anyhow::Result;
use cgmath::MetricSpace;
use fxhash::FxHashMap;

pub fn export(papercraft: &Papercraft, file_name: &Path) -> Result<()> {
    use std::collections::hash_map::Entry;
    use std::io::prelude::*;

    let model = papercraft.model();

    // f32 cannot be used as a hash index because it does not implement Eq nor Hash, something to do with precision and ambiguous representations and NaNs...
    // But we never operate with the values in model, so same f32 values should always have the same bit pattern, and we can use that bit-pattern as the hash index.
    fn index_vector2(v: &Vector2) -> (u32, u32) {
        (v.x.to_bits(), v.y.to_bits())
    }
    fn index_vector3(v: &Vector3) -> (u32, u32, u32) {
        (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
    }

    let title = file_name
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or(std::borrow::Cow::Borrowed("object"));
    let title: String = title
        .as_ref()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let mtl_name = file_name.with_extension("mtl");
    let has_textures = model.has_textures();

    let f = std::fs::File::create(file_name)?;
    let mut f = std::io::BufWriter::new(f);

    // Vertex identity is important for properly exporting the mesh:
    // Two Papercraft vertices are considered the same if they appear on opposite sides of an
    // edge.

    #[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
    struct Vid(u32);

    // Maps VertexIndex into the position of next_id
    let mut idx: Vec<Option<usize>> = vec![None; model.num_vertices()];
    // Unique ids of the vertices
    let mut next_id: Vec<Vid> = Vec::new();

    fn mark_equal(
        next_id: &mut Vec<Vid>,
        idx: &mut [Option<usize>],
        a: VertexIndex,
        b: VertexIndex,
    ) {
        let a = usize::from(a);
        let b = usize::from(b);
        match (idx[a], idx[b]) {
            (None, None) => {
                // id and its position are the same when new
                let id = next_id.len();
                next_id.push(Vid(id as u32));
                idx[a] = Some(id);
                idx[b] = Some(id);
            }
            (Some(x), None) => {
                idx[b] = Some(x);
            }
            (None, Some(x)) => {
                idx[a] = Some(x);
            }
            (Some(x), Some(y)) => {
                next_id[x] = next_id[y];
            }
        };
    }
    for (i_face, face) in model.faces() {
        for (i_v0, i_v1, i_edge) in face.vertices_with_edges() {
            let edge = &model[i_edge];

            match edge.faces() {
                (i_fa, Some(i_fb)) if i_fa == i_face => {
                    // Normal edge, main face
                    let face_b = &model[i_fb];
                    let (mut i_v0b, mut i_v1b) = face_b.vertices_of_edge(i_edge).unwrap();

                    // Is it a normal or inverted edge?
                    // Check for normal edge: First compare the ids, they are sometimes the same
                    if i_v0 != i_v1b && i_v1 != i_v0b &&
                        // Then compare the exact values, those are most likely the same
                        model[i_v0].pos() != model[i_v1b].pos()
                    {
                        // Here it is probably an inverted edge, make sure by computing the distance
                        let d2_normal = model[i_v0].pos().distance2(model[i_v1b].pos());
                        let d2_inverted = model[i_v0].pos().distance2(model[i_v0b].pos());
                        if d2_inverted < d2_normal {
                            std::mem::swap(&mut i_v0b, &mut i_v1b);
                        }
                    }
                    mark_equal(&mut next_id, &mut idx, i_v0, i_v1b);
                    mark_equal(&mut next_id, &mut idx, i_v1, i_v0b);
                }
                (_, None) => {
                    // An edge rim, no `mark_equal()` call because there is no face_b.
                    // Assign the idx if not already.
                    for i in [usize::from(i_v0), usize::from(i_v1)] {
                        if idx[i].is_none() {
                            let id = next_id.len();
                            next_id.push(Vid(id as u32));
                            idx[i] = Some(id);
                        }
                    }
                }
                _ => {
                    // Normal edge, but from face_b point of view, do nothing; face_a will take care
                }
            }
        }
    }

    let mut fx = FxHashMap::default();

    // Deduplicate the ids, make them contiguous and get the position
    let mut vertex_pos = Vec::new();
    let vertex_map: Vec<_> = idx
        .iter()
        .enumerate()
        .map(|(i_v, id)| {
            let vid = next_id[id.unwrap()];
            *fx.entry(vid).or_insert_with(|| {
                vertex_pos.push(model[VertexIndex::from(i_v)].pos());
                vertex_pos.len() as u32
            })
        })
        .collect();

    if has_textures {
        writeln!(f, "mtllib {}", mtl_name.display())?;
    }
    writeln!(f, "o {title}")?;
    for pos in &vertex_pos {
        writeln!(f, "v {} {} {}", pos[0], pos[1], pos[2])?;
    }

    // Normals and UVs are deduplicated by value, they don't affect the geometry
    let mut index_vn = FxHashMap::default();
    let mut index_vt = FxHashMap::default();

    for (_, v) in model.vertices() {
        let n = v.normal();
        let id = index_vn.len() + 1;
        let e = index_vn.entry(index_vector3(&n));
        if let Entry::Vacant(vacant) = e {
            writeln!(f, "vn {} {} {}", n[0], n[1], n[2])?;
            vacant.insert(id);
        }
    }
    for (_, v) in model.vertices() {
        let uv = v.uv();
        let id = index_vt.len() + 1;
        let e = index_vt.entry(index_vector2(&uv));
        if let Entry::Vacant(vacant) = e {
            writeln!(f, "vt {} {}", uv[0], 1.0 - uv[1])?;
            vacant.insert(id);
        }
    }
    writeln!(f, "s 0")?;

    let by_mat = if has_textures {
        let mut by_mat = vec![Vec::new(); model.num_textures()];
        for (i_face, face) in model.faces() {
            by_mat[usize::from(face.material())].push(i_face);
        }
        by_mat
    } else {
        vec![(0..model.num_faces()).map(FaceIndex::from).collect()]
    };

    // We iterate over the triangles, but export the flat-face, we have to skip duplicated
    // triangles. If one flat-face uses two different materials, that will not be properly
    // exported.
    let mut done_faces = vec![false; model.num_faces()];
    for (i_mat, face_by_mat) in by_mat.iter().enumerate() {
        if face_by_mat.is_empty() {
            continue;
        }
        if has_textures {
            writeln!(f, "usemtl Material.{i_mat:03}")?;
        }
        for &i_face in face_by_mat {
            if done_faces[usize::from(i_face)] {
                continue;
            }
            //In model, faces are all triangles, group them by flatness
            let flat_face = papercraft.get_flat_faces(i_face);
            let mut flat_contour: Vec<_> = flat_face
                .iter()
                .map(|&f| {
                    done_faces[usize::from(f)] = true;
                    f
                })
                .flat_map(|f| model[f].vertices_with_edges())
                .filter_map(|(i_v0, i_v1, e)| {
                    if papercraft.edge_status(e) == EdgeStatus::Hidden {
                        None
                    } else {
                        Some((i_v0, i_v1))
                    }
                })
                .collect();
            write!(f, "f")?;
            let mut next = Some(flat_contour.len() - 1);
            while let Some(pos) = next {
                let vertex = flat_contour.remove(pos);
                let (i_v0, i_v1) = vertex;
                let v0 = vertex_map[usize::from(i_v0)];
                let vx = &model[i_v0];
                let t = index_vt[&index_vector2(&vx.uv())];
                let n = index_vn[&index_vector3(&vx.normal())];
                write!(f, " {v0}/{t}/{n}")?;
                next = flat_contour.iter().position(|(i_x0, _)| i_v1 == *i_x0);
            }
            writeln!(f)?;
        }
    }
    drop(f);

    if has_textures {
        let fm = std::fs::File::create(mtl_name)?;
        let mut fm = std::io::BufWriter::new(fm);

        let dir = file_name.parent();
        for (i_mat, (tex, faces)) in model.textures().zip(&by_mat).enumerate() {
            if faces.is_empty() {
                continue;
            }
            writeln!(fm, "newmtl Material.{i_mat:03}")?;

            if let Some(pixbuf) = tex.pixbuf() {
                let bmp_name = tex.file_name();
                writeln!(fm, "map_Kd {bmp_name}")?;

                let path = Path::new(bmp_name);
                let mut full_path_buf;
                let full_path = if let Some(dir) = dir {
                    full_path_buf = dir.to_owned();
                    full_path_buf.push(path);
                    &full_path_buf
                } else {
                    path
                };
                pixbuf.save(full_path)?;
            }
        }
        drop(fm);
    }
    Ok(())
}

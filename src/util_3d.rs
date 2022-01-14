use cgmath::{Vector3, Zero, InnerSpace, Vector2, Rad};
use std::f32::consts::PI;

// Each returned tuple is a triangle of indices into the original vector
pub fn tessellate(ps: &[Vector3<f32>]) -> Vec<(usize, usize, usize)> {
    if ps.len() < 3 {
        return Vec::new();
    }

    if ps.len() == 3 {
        return vec![(0, 1, 2)];
    }

    let mut res = Vec::with_capacity(ps.len() - 2);

    // Compute the face plane
    let mut normal = Vector3::zero();
    for i in 0 .. ps.len() {
        let a = ps[i];
        let b = ps[(i + 1) % ps.len()];
        normal += a.cross(b);
    }

    let normal = normal.normalize();
    let plane_x = (ps[1] - ps[0]).normalize();
    let plane_y = plane_x.cross(normal);
    let plane_o = ps[0];

    // Project every vertex into this plane
    let mut ps = ps
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let p = p - plane_o;
            let x = p.dot(plane_x);
            let y = p.dot(plane_y);
            (idx, Vector2::new(x, y))
        })
        .collect::<Vec<_>>();

    // Tessellate the 2D polygon using the "ear" method
    while ps.len() >= 3 {
        let mut min_angle = None;

        for i in 0 .. ps.len() {
            let (_, a) = ps[i];
            let (_, b) = ps[(i + 1) % ps.len()];
            let (_, c) = ps[(i + 2) % ps.len()];
            let angle = (c - b).angle(b - a);

            // Find the vertex with the minimum inner angle
            let inner_angle = Rad(PI) - angle;

            if min_angle.map(|(_, a)| inner_angle < a).unwrap_or(true) {
                // If this point is not an ear, discard it
                if !ps.iter().enumerate().any(|(i_other, (_, p_other))| {
                    i_other != i &&
                    i_other != (i + 1) % ps.len() &&
                    i_other != (i + 2) % ps.len() &&
                    point_in_triangle(*p_other, a, b, c)
                }) {
                    min_angle = Some((i, inner_angle));
                }
            }
        }
        // min_angle should never be None, but just in case
        let i = min_angle.map(|(i, _)| i).unwrap_or(0);

        let tri = (i, (i + 1) % ps.len(), (i + 2) % ps.len());
        res.push((ps[tri.0].0, ps[tri.1].0, ps[tri.2].0));
        ps.remove(tri.1);
    }

    res
}

fn point_in_triangle(p: Vector2<f32>, p0: Vector2<f32>, p1: Vector2<f32>, p2: Vector2<f32>) -> bool {
    let s = (p0.x - p2.x) * (p.y - p2.y) - (p0.y - p2.y) * (p.x - p2.x);
    let t = (p1.x - p0.x) * (p.y - p0.y) - (p1.y - p0.y) * (p.x - p0.x);

    if (s < 0.0) != (t < 0.0) && s != 0.0 && t != 0.0 {
        false
    } else {
        let d = (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x);
        d == 0.0 || (d < 0.0) == (s + t <= 0.0)
    }
}

pub fn bounding_box(vs: impl IntoIterator<Item=Vector3<f32>>) -> (Vector3<f32>, Vector3<f32>) {
    let mut vs = vs.into_iter();
    let (mut a, mut b) = match vs.next() {
        Some(v) => (v, v),
        None => return (Vector3::zero(), Vector3::zero()),
    };
    for v in vs {
        a.x = a.x.min(v.x);
        a.y = a.y.min(v.y);
        a.z = a.z.min(v.z);
        b.x = b.x.max(v.x);
        b.y = b.y.max(v.y);
        b.z = b.z.max(v.z);
    }
    (a, b)
}

pub fn ray_crosses_face(ray: (Vector3<f32>, Vector3<f32>), vs: &[Vector3<f32>; 3]) -> Option<f32> {
    // Möller-Trumbore algorithm

    let v0v1 = vs[1] - vs[0];
    let v0v2 = vs[2] - vs[0];
    let dir = ray.1 - ray.0;
    let pvec = dir.cross(v0v2);
    let det = v0v1.dot(pvec);

    //backface culling?
    /*if (det < 0.0001) {
        return None;
    }*/

    // ray and triangle are parallel if det is close to 0
    if det.abs() < std::f32::EPSILON {
        return None;
    }

    let inv_det = 1.0 / det;

    let tvec = ray.0 - vs[0];
    let u = tvec.dot(pvec) * inv_det;
    if u < 0.0 || u > 1.0 {
        return None;
    }

    let qvec = tvec.cross(v0v1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = v0v2.dot(qvec) * inv_det;

    Some(t)
}
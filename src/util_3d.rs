use std::f32::consts::PI;
use cgmath::{Zero, InnerSpace, Transform, One, Rad};

pub type Vector2 = cgmath::Vector2<f32>;
pub type Vector3 = cgmath::Vector3<f32>;
pub type Point2 = cgmath::Point2<f32>;
pub type Point3 = cgmath::Point3<f32>;
pub type Quaternion = cgmath::Quaternion<f32>;
pub type Matrix2 = cgmath::Matrix2<f32>;
pub type Matrix3 = cgmath::Matrix3<f32>;
pub type Matrix4 = cgmath::Matrix4<f32>;

#[derive(Debug)]
pub struct Plane {
    origin: Vector3,
    base_x: Vector3,
    base_y: Vector3,
}

impl Default for Plane {
    fn default() -> Plane{
        Plane {
            origin: Vector3::zero(),
            base_x: Vector3::new(1.0, 0.0, 0.0),
            base_y: Vector3::new(0.0, 1.0, 0.0),
        }
    }
}

impl Plane {
    pub fn project(&self, p: &Vector3) -> Vector2 {
        let p = p - self.origin;
        let x = p.dot(self.base_x);
        let y = p.dot(self.base_y);
        Vector2::new(x, y)
    }
}

// Each returned tuple is a triangle of indices into the original vector
pub fn tessellate(ps: &[Vector3]) -> (Vec<[usize; 3]>, Plane) {
    if ps.len() < 3 {
        return (Vec::new(), Plane::default());
    }

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

    let plane = Plane {
        origin: plane_o,
        base_x: plane_x,
        base_y: plane_y,
    };

    if ps.len() == 3 {
        return (vec![[0, 1, 2]], plane);
    }

    let mut res = Vec::with_capacity(ps.len() - 2);

    // Project every vertex into this plane
    let mut ps = ps
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let p2 = plane.project(p);
            (idx, p2)
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
        res.push([ps[tri.0].0, ps[tri.1].0, ps[tri.2].0]);
        ps.remove(tri.1);
    }

    (res, plane)
}

pub fn point_in_triangle(p: Vector2, p0: Vector2, p1: Vector2, p2: Vector2) -> bool {
    let s = (p0.x - p2.x) * (p.y - p2.y) - (p0.y - p2.y) * (p.x - p2.x);
    let t = (p1.x - p0.x) * (p.y - p0.y) - (p1.y - p0.y) * (p.x - p0.x);

    if (s < 0.0) != (t < 0.0) && s != 0.0 && t != 0.0 {
        false
    } else {
        let d = (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x);
        d == 0.0 || (d < 0.0) == (s + t <= 0.0)
    }
}

pub fn bounding_box(vs: impl IntoIterator<Item=Vector3>) -> (Vector3, Vector3) {
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

pub fn ray_crosses_face(ray: (Vector3, Vector3), vs: &[Vector3; 3]) -> Option<f32> {
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
    if !(0.0..=1.0).contains(&u) {
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

// Returns (offset0, offset1, distance2)
pub fn line_line_distance(line0: (Vector3, Vector3), line1: (Vector3, Vector3)) -> (f32, f32, f32) {
    let diff = line0.0 - line1.0;
    let line0d = line0.1 - line0.0;
    let line1d = line1.1 - line1.0;
    let len0 = line0d.magnitude();
    let len1 = line1d.magnitude();
    let line0d = line0d / len0;
    let line1d = line1d / len1;
    let a01 = -line0d.dot(line1d);
    let b0 = line0d.dot(diff);
    let c = diff.magnitude2();
    let det = 1.0 - a01 * a01;
    let l0_closest;
    let l1_closest;
    let distance2;
    if det.abs() > std::f32::EPSILON {
        //not parallel
        let b1 = -line1d.dot(diff);
        let inv_det = 1.0 / det;
        l0_closest = (a01 * b1 - b0) * inv_det;
        l1_closest = (a01 * b0 - b1) * inv_det;
        distance2 = l0_closest * (l0_closest + a01 * l1_closest + 2.0 * b0) +
                            l1_closest * (a01 * l0_closest + l1_closest + 2.0 * b1) + c;
    } else {
        //almost parallel
        l0_closest = -b0;
        l1_closest = 0.0;
        distance2 = b0 * l0_closest + c;
    }
    (l0_closest / len0, l1_closest / len1, distance2.abs())
}

pub fn line_segment_distance(line0: (Vector3, Vector3), line1: (Vector3, Vector3)) -> (f32, f32, f32) {
    let (l0_closest, mut l1_closest, mut distance2) = line_line_distance(line0, line1);
    if l1_closest < 0.0 {
        l1_closest = 0.0;
        let p = line0.0 + (line0.1 - line0.0) * l0_closest;
        distance2 = (line1.0 - p).magnitude2();
    } else if l1_closest > 1.0 {
        l1_closest = 1.0;
        let p = line0.0 + (line0.1 - line0.0) * l0_closest;
        distance2 = (line1.1 - p).magnitude2();
    }
    (l0_closest, l1_closest, distance2)
}

// Returns (offset, distance)
pub fn point_line_distance(p: Vector2, line: (Vector2, Vector2)) -> (f32, f32) {
    let vline = line.1 - line.0;
    let line_len = vline.magnitude();
    let vline = vline / line_len;
    let vline_perp = Vector2::new(vline.y, -vline.x);
    let p = p - line.0;
    let o = p.dot(vline) / line_len;
    let d = p.dot(vline_perp).abs();
    (o, d)
}

pub fn point_segment_distance(p: Vector2, line: (Vector2, Vector2)) -> (f32, f32) {
    let (o, d) = point_line_distance(p, line);
    if o < 0.0 {
        (0.0, 10000.0)//(0.0, (p - line.0).magnitude())
    } else if o > 1.0 {
        (1.0, 10000.0)//(1.0, (p - line.1).magnitude())
    } else {
        (o, d)
    }
}

pub fn ortho2d(width: f32, height: f32) -> Matrix3 {
    let right = width as f32 / 2.0;
    let left = -right;
    let top = -height as f32 / 2.0;
    let bottom = -top;
    Matrix3::new(
        2.0 / (right - left), 0.0, 0.0,
        0.0, 2.0 / (top - bottom), 0.0,
        0.0, 0.0, 1.0
    )
}

//Computes a 2D matrix that converts from `a` to [(1,0), (0,0), (0,1)]
pub fn basis_2d_matrix(a: [Vector2; 3]) -> Matrix3 {

    let mt = Matrix3::from_translation(-a[1]);
    let angle = (a[0] - a[1]).angle(Vector2::new(1.0, 0.0));
    let mr = Matrix2::from_angle(angle);
    let len = (a[0] - a[1]).magnitude();
    let ms = Matrix3::from_scale(1.0 / len);

    let m = ms * Matrix3::from(mr) * mt;

    let a2 = m.transform_point(cgmath::Point2::new(a[2].x, a[2].y));
    let ms2 = Matrix3::from_nonuniform_scale(1.0, 1.0 / a2.y);

    let mut shear = Matrix3::one();
    shear[1][0] = -a2.x;

    shear * ms2 * m
}

#[cfg(test)]
mod tests {
    use cgmath::SquareMatrix;

    use super::*;
    #[test]
    fn test1() {
        let a = [Vector2::new(3.2, 4.1), Vector2::new(8.0, 2.4), Vector2::new(5.1, 4.0)];
        let b = [Vector2::new(4.2, 1.1), Vector2::new(2.0, 3.4), Vector2::new(4.1, -2.0)];
        let ma = basis_2d_matrix(a);
        let mb = basis_2d_matrix(b);

        let mm = mb.invert().unwrap() * ma;

        for (va, vb) in a.iter().zip(b.iter()) {
            let pa = mm.transform_point(cgmath::Point2::new(va.x, va.y));
            dbg!(pa - vb);
        }

    }
}
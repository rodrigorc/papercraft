use std::f32::consts::PI;
use cgmath::{Zero, InnerSpace, MetricSpace, Rad};

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
    pub fn project(&self, p: &Vector3, scale: f32) -> Vector2 {
        let p = p - self.origin;
        let x = p.dot(self.base_x);
        let y = p.dot(self.base_y);
        scale * Vector2::new(x, y)
    }
    pub fn from_tri(tri: [Vector3; 3]) -> Plane {
        let v0 = tri[1] - tri[0];
        let v1 = tri[2] - tri[0];
        let normal = v0.cross(v1);
        Plane {
            origin: tri[0],
            base_x: v0.normalize(),
            base_y: v0.cross(normal).normalize(),
        }
    }
    pub fn normal(&self) -> Vector3 {
        self.base_x.cross(self.base_y)
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
            let p2 = plane.project(p, 1.0);
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
                    point_in_triangle(*p_other, [a, b, c])
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

pub fn point_in_triangle(p: Vector2, tri: [Vector2; 3]) -> bool {
    let [p0, p1, p2] = tri;
    let s = (p0.x - p2.x) * (p.y - p2.y) - (p0.y - p2.y) * (p.x - p2.x);
    let t = (p1.x - p0.x) * (p.y - p0.y) - (p1.y - p0.y) * (p.x - p0.x);

    if (s < 0.0) != (t < 0.0) && s != 0.0 && t != 0.0 {
        false
    } else {
        let d = (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x);
        d == 0.0 || (d < 0.0) == (s + t <= 0.0)
    }
}

pub fn bounding_box_3d(vs: impl IntoIterator<Item=Vector3>) -> (Vector3, Vector3) {
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

pub fn bounding_box_2d(vs: impl IntoIterator<Item=Vector2>) -> (Vector2, Vector2) {
    let mut vs = vs.into_iter();
    let (mut a, mut b) = match vs.next() {
        Some(v) => (v, v),
        None => return (Vector2::zero(), Vector2::zero()),
    };
    for v in vs {
        a.x = a.x.min(v.x);
        a.y = a.y.min(v.y);
        b.x = b.x.max(v.x);
        b.y = b.y.max(v.y);
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
        distance2 = p.distance2(line1.0);
    } else if l1_closest > 1.0 {
        l1_closest = 1.0;
        let p = line0.0 + (line0.1 - line0.0) * l0_closest;
        distance2 = p.distance2(line1.1);
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

pub fn point_line_side(p: Vector2, line: (Vector2, Vector2)) -> bool {
    let v1 = line.1 - line.0;
    let v2 = p - line.0;
    (v1.x * v2.y - v1.y  * v2.x) >= 0.0
}

// (offset, distance)
pub fn point_segment_distance(p: Vector2, line: (Vector2, Vector2)) -> (f32, f32) {
    let (o, d) = point_line_distance(p, line);
    if o < 0.0 {
        (0.0, p.distance(line.0))
    } else if o > 1.0 {
        (1.0, p.distance(line.1))
    } else {
        (o, d)
    }
}

// (intersection, offset_1, offset_2)
pub fn line_line_intersection(line_1: (Vector2, Vector2), line_2: (Vector2, Vector2)) -> (Vector2, f32, f32) {
    let s1 = line_1.1 - line_1.0;
    let s2 = line_2.1 - line_2.0;
    let d = line_1.0 - line_2.0;

    let dd = s1.x * s2.y - s2.x * s1.y;
    if dd.abs() < 0.000001 {
        return (line_1.0, f32::MAX, f32::MAX);
    }

    let s = ( s2.x * d.y - s2.y * d.x) / dd;
    let t = ( s1.x * d.y - s1.y * d.x) / dd;

    ((line_1.0 + s * s1), s, t)
}

pub fn ortho2d(width: f32, height: f32) -> Matrix3 {
    let right = width / 2.0;
    let left = -right;
    let top = -height / 2.0;
    let bottom = -top;
    Matrix3::new(
        2.0 / (right - left), 0.0, 0.0,
        0.0, 2.0 / (top - bottom), 0.0,
        0.0, 0.0, 1.0
    )
}

// Like ortho2d but aligned to the (0,0)
pub fn ortho2d_zero(width: f32, height: f32) -> Matrix3 {
    Matrix3::new(
        2.0 / width, 0.0, 0.0,
        0.0, -2.0 / height, 0.0,
        -1.0, -1.0, 1.0
    )
}


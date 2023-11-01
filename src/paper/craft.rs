use std::ops::ControlFlow;
use std::cell::RefCell;

use fxhash::{FxHashMap, FxHashSet};
use cgmath::{prelude::*, Transform, EuclideanSpace, InnerSpace, Rad, Deg};
use slotmap::{SlotMap, new_key_type};
use serde::{Serialize, Deserialize};
use crate::util_3d;


use super::*;
mod file;
mod update;

// Which side of a cut will the tab be drawn, compare with face_sign
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TabSide {
    False,
    True,
    Hidden,
}

pub enum EdgeToggleTabAction {
    Toggle,
    Hide,
    Set(TabSide),
}

impl TabSide {
    pub fn apply(self, action: EdgeToggleTabAction, rim: bool) -> TabSide {
        use TabSide::*;
        match (self, action) {
            (_, EdgeToggleTabAction::Set(next)) => next,
            // toggle
            (False, EdgeToggleTabAction::Toggle) => if rim { Hidden } else { True },
            (True, EdgeToggleTabAction::Toggle) => False,
            (Hidden, EdgeToggleTabAction::Toggle) => False,

            // hide
            (False, EdgeToggleTabAction::Hide) => Hidden,
            (True, EdgeToggleTabAction::Hide) => Hidden,
            (Hidden, EdgeToggleTabAction::Hide) => False,
        }
    }
    pub fn tab_visible(self, face_sign: bool) -> bool {
        match (self, face_sign) {
            (TabSide::False, false) | (TabSide::True, true) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum EdgeStatus {
    Hidden,
    Joined,
    Cut(TabSide),
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub enum TabStyle {
    #[default]
    Textured,
    HalfTextured,
    White,
    None,
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub enum FoldStyle {
    #[default]
    Full,
    FullAndOut,
    Out,
    In,
    InAndOut,
    None,
}

new_key_type! {
    pub struct IslandKey;
}

#[derive(Debug, Copy, Clone)]
pub struct JoinResult {
    pub i_edge: EdgeIndex,
    pub i_island: IslandKey,
    pub prev_root: FaceIndex,
    pub prev_rot: Rad<f32>,
    pub prev_loc: Vector2,
}

fn my_true() -> bool { true }
fn default_fold_line_width() -> f32 { 0.1 }
fn my_zero() -> f32 { 0.0 }

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaperOptions {
    pub scale: f32,
    pub page_size: (f32, f32),
    pub resolution: u32, //dpi
    pub pages: u32,
    pub page_cols: u32,
    pub margin: (f32, f32, f32, f32), //top, left, right, bottom
    #[serde(default="my_true")]
    pub texture: bool,
    #[serde(default="my_true")]
    pub tex_filter: bool,
    #[serde(default)]
    pub tab_style: TabStyle,
    #[serde(default)]
    pub fold_style: FoldStyle,
    pub tab_width: f32,
    pub tab_angle: f32, //degrees
    pub fold_line_len: f32, //only for folds in & out
    #[serde(default)]
    pub shadow_tab_alpha: f32, //0.0 - 1.0
    #[serde(default="default_fold_line_width")]
    pub fold_line_width: f32, //only for folds in & out
    #[serde(default)]
    pub hidden_line_angle: f32, //degrees
    #[serde(default="my_true")]
    pub show_self_promotion: bool,
    #[serde(default="my_true")]
    pub show_page_number: bool,
    #[serde(default="my_zero")]
    pub edge_id_font_size: f32, //0.0 to disable
}

impl Default for PaperOptions {
    fn default() -> Self {
        PaperOptions {
            scale: 1.0,
            page_size: (210.0, 297.0),
            resolution: 300,
            pages: 1,
            page_cols: 2,
            margin: (10.0, 10.0, 10.0, 10.0),
            texture: true,
            tex_filter: true,
            tab_style: TabStyle::Textured,
            fold_style: FoldStyle::Full,
            tab_width: 5.0,
            tab_angle: 45.0,
            fold_line_len: 4.0,
            shadow_tab_alpha: 0.0,
            fold_line_width: default_fold_line_width(),
            hidden_line_angle: 0.0,
            show_self_promotion: true,
            show_page_number: true,
            edge_id_font_size: 0.0,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PageOffset {
    pub page: u32,
    pub offset: Vector2,
}

const PAGE_SEP: f32 = 10.0; // Currently not configurable
                            //
impl PaperOptions {
    pub fn page_position(&self, page: u32) -> Vector2 {
        let page_cols = self.page_cols;
        let page_size = Vector2::from(self.page_size);
        let row = page / page_cols;
        let col = page % page_cols;
        Vector2::new((col as f32) * (page_size.x + PAGE_SEP), (row as f32) * (page_size.y + PAGE_SEP))
    }
    pub fn is_in_page_fn(&self, page: u32) -> impl Fn(Vector2) -> (bool, Vector2) {
        let page_pos_0 = self.page_position(page);
        let page_size = Vector2::from(self.page_size);
        move |p: Vector2| {
            let r = p - page_pos_0;
            let is_in = r.x >= 0.0 && r.y >= 0.0 && r.x < page_size.x && r.y < page_size.y;
            (is_in, r)
        }
    }
    pub fn global_to_page(&self, pos: Vector2) -> PageOffset {
        let page_cols = self.page_cols;
        let page_size = Vector2::from(self.page_size);
        let col = ((pos.x / (page_size.x + PAGE_SEP)) as i32).clamp(0, page_cols as i32) as u32;
        let row = ((pos.y / (page_size.y + PAGE_SEP)) as i32).max(0) as u32;

        let page = row * page_cols + col;
        let zero_pos = self.page_position(page);
        let offset = pos - zero_pos;
        PageOffset {
            page,
            offset,
        }
    }
    pub fn page_to_global(&self, po: PageOffset) -> Vector2 {
        let zero_pos = self.page_position(po.page);
        zero_pos + po.offset
    }
    pub fn is_inside_canvas(&self, pos: Vector2) -> bool {
        let page_cols = self.page_cols;
        let page_rows = (self.pages + self.page_cols - 1) / self.page_cols;
        let page_size = Vector2::from(self.page_size);

        #[allow(clippy::if_same_then_else, clippy::needless_bool)]
        if pos.x < -(page_size.x + PAGE_SEP) {
            false
        } else if pos.y < -(page_size.y + PAGE_SEP) {
            false
        } else if pos.x > (page_cols + 1) as f32 * (page_size.x + PAGE_SEP) {
            false
        } else if pos.y > (page_rows + 1) as f32 * (page_size.y + PAGE_SEP) {
            false
        } else {
            true
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Papercraft {
    model: Model,
    #[serde(default)] //TODO: default not actually needed
    options: PaperOptions,
    edges: Vec<EdgeStatus>, //parallel to EdgeIndex
    #[serde(with="super::ser::slot_map")]
    islands: SlotMap<IslandKey, Island>,

    #[serde(skip)]
    memo: Memoization,
}

#[derive(Copy, Clone, Default)]
pub struct TabGeom {
    pub tan_0: f32,
    pub tan_1: f32,
    pub width: f32,
    pub triangular: bool,
}

#[derive(Default)]
struct Memoization {
    flat_face_tab_dimensions: RefCell<FxHashMap<(FaceIndex, EdgeIndex), TabGeom>>,
    face_to_face_edge_matrix: RefCell<FxHashMap<(EdgeIndex, FaceIndex, FaceIndex), Matrix3>>,
}

impl Papercraft {
    pub fn empty() -> Papercraft {
        Papercraft {
            model: Model::empty(),
            options: PaperOptions::default(),
            edges: Vec::new(),
            islands: SlotMap::with_key(),
            memo: Memoization::default(),
        }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn options(&self) -> &PaperOptions {
        &self.options
    }
    // Returns the old options
    pub fn set_options(&mut self, mut options: PaperOptions) -> PaperOptions{
        let scale = options.scale / self.options.scale;
        // Compute positions relative to the nearest page
        let page_pos: FxHashMap<_, _> = self.islands
            .iter()
            .map(|(i_island, island)| {
                let po = self.options.global_to_page(island.location());
                (i_island, po)
            })
            .collect();

        // Apply the new options
        std::mem::swap(&mut self.options, &mut options);
        // Invalidate the memoized values that may depend on any option
        self.memo = Memoization::default();

        // Apply the new positions
        for (i_island, mut po) in page_pos {
            po.offset *= scale;
            let loc = self.options.page_to_global(po);
            if let Some(island) = self.island_by_key_mut(i_island) {
                island.loc = loc;
                island.recompute_matrix();
            }
        }

        options
    }
    pub fn islands(&self) -> impl Iterator<Item = (IslandKey, &Island)> + '_ {
        self.islands.iter()
    }
    pub fn num_islands(&self) -> usize {
        self.islands.len()
    }
    pub fn island_bounding_box_angle(&self, island: &Island, angle: Rad<f32>) -> (Vector2, Vector2) {
        let mx = island.matrix() * Matrix3::from(Matrix2::from_angle(angle));
        let mut vx = Vec::new();
        traverse_faces_ex(&self.model, island.root_face(),
            mx,
            NormalTraverseFace(self),
            |_, face, mx| {
                let vs = face.index_vertices().map(|v| {
                    let normal = self.model.face_plane(face);
                    mx.transform_point(Point2::from_vec(normal.project(&self.model[v].pos(), self.options.scale))).to_vec()
                });
                vx.extend(vs);
                ControlFlow::Continue(())
            }
        );

        let (a, b) = util_3d::bounding_box_2d(vx);
        let m = self.options.tab_width;
        let mm = Vector2::new(m, m);
        (a - mm, b + mm)
    }
    pub fn island_best_bounding_box(&self, island: &Island) -> (Rad<f32>, (Vector2, Vector2)) {

        const TRIES: i32 = 60;

        fn bbox_weight(bb: (Vector2, Vector2)) -> f32 {
            let d = bb.1 - bb.0;
            d.y
        }

        let delta_a = Rad::full_turn() / TRIES as f32;

        let mut best_angle = Rad::zero();
        let mut best_bb = self.island_bounding_box_angle(island, best_angle);
        let mut best_width = bbox_weight(best_bb);

        let mut angle2 = delta_a;
        for _ in 1 .. TRIES {
            let bb2 = self.island_bounding_box_angle(island, angle2);
            let width2 = bbox_weight(bb2);

            if width2 < best_width {
                best_width = width2;
                best_angle = angle2;
                best_bb = bb2;
            }
            angle2 += delta_a;
        }
        (best_angle, best_bb)
    }

    pub fn island_by_face(&self, i_face: FaceIndex) -> IslandKey {
        for (i_island, island) in &self.islands {
            if self.contains_face(island, i_face) {
                return i_island;
            }
        }
        panic!("Island not found");
    }
    pub fn island_by_root(&self, i_face: FaceIndex) -> Option<IslandKey> {
        for (i_island, island) in &self.islands {
            if island.root == i_face {
                return Some(i_island);
            }
        }
        None
    }
    // Islands come and go, so this kay may not exist.
    pub fn island_by_key(&self, key: IslandKey) -> Option<&Island> {
        self.islands.get(key)
    }
    pub fn island_by_key_mut(&mut self, key: IslandKey) -> Option<&mut Island> {
        self.islands.get_mut(key)
    }

    pub fn edge_status(&self, edge: EdgeIndex) -> EdgeStatus {
        self.edges[usize::from(edge)]
    }

    pub fn edge_toggle_tab(&mut self, i_edge: EdgeIndex, action: EdgeToggleTabAction) -> Option<TabSide> {
        let rim = matches!(self.model()[i_edge].faces(), (_, None));
        if let EdgeStatus::Cut(ref mut x) = self.edges[usize::from(i_edge)] {
            Some(std::mem::replace(x, x.apply(action, rim)))
        } else {
            None
        }
    }

    pub fn edge_cut(&mut self, i_edge: EdgeIndex, offset: Option<f32>) {
        match self.edges[usize::from(i_edge)] {
            EdgeStatus::Joined => {}
            _ => { return; }
        }
        let edge = &self.model[i_edge];
        let (i_face_a, i_face_b) = match edge.faces() {
            (fa, Some(fb)) => (fa, fb),
            _ => { return; }
        };

        //one of the edge faces will be the root of the new island, but we do not know which one, yet
        let i_island = self.island_by_face(i_face_a);

        self.edges[usize::from(i_edge)] = EdgeStatus::Cut(TabSide::False);

        let mut data_found = None;
        self.traverse_faces(&self.islands[i_island],
            |i_face, _, fmx| {
                if i_face == i_face_a {
                    data_found = Some((*fmx, i_face_b, i_face_a));
                } else if i_face == i_face_b {
                    data_found = Some((*fmx, i_face_a, i_face_b));
                }
                ControlFlow::Continue(())
            }
        );
        let (face_mx, new_root, i_face_old) = data_found.unwrap();

        let medge = self.face_to_face_edge_matrix(edge, &self.model[i_face_old], &self.model[new_root]);
        let mx = face_mx * medge;

        let mut new_island = Island {
            root: new_root,
            loc: Vector2::new(mx[2][0], mx[2][1]),
            rot: Rad(mx[0][1].atan2(mx[0][0])),
            mx: Matrix3::one(),
        };
        new_island.recompute_matrix();

        //Compute the offset
        if let Some(offset_on_cut) = offset {
            let sign = if edge.face_sign(new_root) { 1.0 } else { -1.0 };
            let new_root = &self.model[new_root];
            let new_root_plane = self.model.face_plane(new_root);
            let v0 = new_root_plane.project(&self.model[edge.v0()].pos(), self.options.scale);
            let v1 = new_root_plane.project(&self.model[edge.v1()].pos(), self.options.scale);
            let v0 = mx.transform_point(Point2::from_vec(v0)).to_vec();
            let v1 = mx.transform_point(Point2::from_vec(v1)).to_vec();
            let v = (v1 - v0).normalize_to(offset_on_cut);

            //priority_face makes no sense when doing a split, so pass None here unconditionally
            if self.compare_islands(&self.islands[i_island], &new_island, None) {
                let island = &mut self.islands[i_island];
                island.translate(-sign * Vector2::new(-v.y, v.x));
            } else {
                new_island.translate(sign * Vector2::new(-v.y, v.x));
            }
        }
        self.islands.insert(new_island);
    }

    //Retuns a map from the island that disappears into the extra join data.
    pub fn edge_join(&mut self, i_edge: EdgeIndex, priority_face: Option<FaceIndex>) -> FxHashMap<IslandKey, JoinResult> {
        let mut renames = FxHashMap::default();
        match self.edges[usize::from(i_edge)] {
            EdgeStatus::Cut(_) => {}
            _ => { return renames; }
        }
        let edge = &self.model[i_edge];
        let (i_face_a, i_face_b) = match edge.faces() {
            (fa, Some(fb)) => (fa, fb),
            _ => { return renames; }
        };

        let i_island_b = self.island_by_face(i_face_b);
        if self.contains_face(&self.islands[i_island_b], i_face_a) {
            // Same island on both sides, nothing to do
            return renames;
        }

        // Join both islands
        let mut island_b = self.islands.remove(i_island_b).unwrap();
        let i_island_a = self.island_by_face(i_face_a);

        // Keep position of a or b?
        if self.compare_islands(&self.islands[i_island_a], &island_b, priority_face) {
            std::mem::swap(&mut self.islands[i_island_a], &mut island_b);
        }
        renames.insert(i_island_b, JoinResult {
            i_edge,
            i_island: i_island_a,
            prev_root: island_b.root_face(),
            prev_rot: island_b.rotation(),
            prev_loc: island_b.location(),
        });
        self.edges[usize::from(i_edge)] = EdgeStatus::Joined;
        renames
    }

    fn compare_islands(&self, a: &Island, b: &Island, priority_face: Option<FaceIndex>) -> bool {
        if let Some(f) = priority_face {
            if self.contains_face(a, f) {
                return false;
            }
            if self.contains_face(b, f) {
                return true;
            }
        }
        let weight_a = self.island_face_count(a);
        let weight_b = self.island_face_count(b);
        if weight_b > weight_a {
            return true;
        }
        if weight_b < weight_a {
            return false;
        }
        usize::from(a.root) > usize::from(b.root)
    }
    pub fn contains_face(&self, island: &Island, face: FaceIndex) -> bool {
        let mut found = false;
        self.traverse_faces_no_matrix(island,
            |i_face|
                if i_face == face {
                    found = true;
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            );
        found
    }
    pub fn island_face_count(&self, island: &Island) -> u32 {
        let mut count = 0;
        self.traverse_faces_no_matrix(island, |_| { count += 1; ControlFlow::Continue(()) });
        count
    }
    pub fn get_flat_faces(&self, i_face: FaceIndex) -> FxHashSet<FaceIndex> {
        let mut res = FxHashSet::default();
        traverse_faces_ex(&self.model, i_face, (), FlatTraverseFace(self),
            |i_next_face, _, _| {
                res.insert(i_next_face);
                ControlFlow::Continue(())
            }
        );
        res
    }
    fn get_flat_faces_with_matrix(&self, i_face: FaceIndex) -> FxHashMap<FaceIndex, Matrix3> {
        let mut res = FxHashMap::default();
        traverse_faces_ex(&self.model, i_face, Matrix3::one(), FlatTraverseFaceWithMatrix(self),
            |i_next_face, _, mx| {
                res.insert(i_next_face, *mx);
                ControlFlow::Continue(())
            }
        );
        res
    }
    pub fn face_to_face_edge_matrix(&self, edge: &Edge, face_a: &Face, face_b: &Face) -> Matrix3 {
        // Try to use a memoized value
        let mut memo = self.memo.face_to_face_edge_matrix.borrow_mut();
        use std::collections::hash_map::Entry::*;

        let i_edge = self.model.edge_index(edge);
        let i_face_a = self.model.face_index(face_a);
        let i_face_b = self.model.face_index(face_b);
        match memo.entry((i_edge, i_face_a, i_face_b)) {
            Occupied(o) => return *o.get(),
            Vacant(v) => {
                let value = self.face_to_face_edge_matrix_internal(edge, face_a, face_b);
                *v.insert(value)
            }
        }
    }
    fn face_to_face_edge_matrix_internal(&self, edge: &Edge, face_a: &Face, face_b: &Face) -> Matrix3 {
        let v0 = self.model[edge.v0()].pos();
        let v1 = self.model[edge.v1()].pos();
        let plane_a = self.model.face_plane(face_a);
        let plane_b = self.model.face_plane(face_b);
        let scale = self.options.scale;
        let a0 = plane_a.project(&v0, scale);
        let b0 = plane_b.project(&v0, scale);
        let a1 = plane_a.project(&v1, scale);
        let b1 = plane_b.project(&v1, scale);
        let mabt0 = Matrix3::from_translation(-b0);
        let mabr = Matrix3::from(Matrix2::from_angle((b1 - b0).angle(a1 - a0)));
        let mabt1 = Matrix3::from_translation(a0);
        mabt1 * mabr * mabt0
    }
    // Returns the max. angles of the tab sides, actually their cotangent, and the max. width.
    // Ideally it should return all the tab metrics
    pub fn flat_face_tab_dimensions(&self, i_face_b: FaceIndex, i_edge: EdgeIndex) -> TabGeom {
        // Try to use a memoized value
        let mut memo = self.memo.flat_face_tab_dimensions.borrow_mut();
        use std::collections::hash_map::Entry::*;
        let res = match memo.entry((i_face_b, i_edge)) {
            Occupied(o) => *o.get(),
            Vacant(v) => {
                let value = self.flat_face_tab_dimensions_internal(i_face_b, i_edge);
                *v.insert(value)
            }
        };
        res
    }
    fn flat_face_tab_dimensions_internal(&self, i_face_b: FaceIndex, i_edge: EdgeIndex) -> TabGeom {
        #[derive(Debug)]
        struct EData {
            i_edge: EdgeIndex,
            i_v0: VertexIndex,
            i_v1: VertexIndex,
            p0: Vector2,
            p1: Vector2,
        }
        // Compute the flat-face_b contour
        let scale = self.options.scale;
        let flat_face = self.get_flat_faces_with_matrix(i_face_b);
        let flat_contour: Vec<EData> = flat_face
            .iter()
            .flat_map(|(f, _m)| {
                let face = &self.model()[*f];
                face.vertices_with_edges()
                      .filter_map(|(i_v0, i_v1, i_edge)| {
                          if self.edge_status(i_edge) == EdgeStatus::Hidden {
                              return None;
                          }
                          let plane = self.model.face_plane(face);

                          let p0 = plane.project(&self.model()[i_v0].pos(), scale);
                          let p0 = flat_face[f].transform_point(Point2::from_vec(p0)).to_vec();
                          let p1 = plane.project(&self.model()[i_v1].pos(), scale);
                          let p1 = flat_face[f].transform_point(Point2::from_vec(p1)).to_vec();
                          Some(EData { i_edge, i_v0, i_v1, p0, p1 })
                      })
            })
            .collect();
        // The selected edge data
        let the_edge = flat_contour
            .iter()
            .find(|d| d.i_edge == i_edge)
            .unwrap();

        // Adjacent edges data
        let d0 = flat_contour
            .iter()
            .find(|d| the_edge.i_v0 == d.i_v1)
            .unwrap();
        let d1 = flat_contour
            .iter()
            .find(|d| the_edge.i_v1 == d.i_v0)
            .unwrap();

        // ** Compute max tab angles **
        let e0 = d0.p1 - d0.p0;
        let e1 = d1.p0 - d0.p1;
        let e2 = d1.p1 - d1.p0;
        let a0 = e1.angle(e0);
        let a1 = e2.angle(e1);
        let a0 = Rad::turn_div_2() - a0;
        let a1 = Rad::turn_div_2() - a1;

        let tab_angle = Rad::from(Deg(self.options.tab_angle));
        let a0 = Rad(a0.0.min(tab_angle.0));
        let a1 = Rad(a1.0.min(tab_angle.0));

        // ** Compute max tab width **
        //
        // Maximum width is computed with a tab angle of 90°, that is not exact, but the difference
        // should be quite small and always to the safe side.
        // We look in the flat_contour of the face B for the nearest point to the selected edge, that falls inside
        // an imaginary tab of 90° angle and infinite width.
        // This nearest point must be either a vertex or the intersection of an edge with the sides of the tab.

        let base = (d0.p1, d1.p0);

        // Normalize the perpendicular to make easier to compute the distance later, then it is just the offset returned by util_3d.
        let n = Vector2::new(e1.y, -e1.x).normalize();

        // (a0,a1) are the biggest angles that fit, and usually smaller angles will result in
        // smaller width too, but in some edge cases a smaller angle will result in a larger width,
        // and the user will probably prefer the biggest tab area, so we try with some extra angles and see
        // what happens

        let compute_width = |a0: Rad<f32>, a1: Rad<f32>| -> f32 {
            let mut minimum_width = self.options.tab_width;
            let (tab_sin_0, tab_cos_0) = a0.sin_cos();
            let normal_0 = Vector2::new(n.x * tab_sin_0 - n.y * tab_cos_0, n.x * tab_cos_0 + n.y * tab_sin_0);
            //90° is the original normal so switch the sin/cos in these rotations
            let (tab_sin_1, tab_cos_1) = a1.sin_cos();
            let normal_1 = Vector2::new(n.x * tab_sin_1 + n.y * tab_cos_1, -n.x * tab_cos_1 + n.y * tab_sin_1);

            let side_0 = (base.0, base.0 + normal_0);
            let side_1 = (base.1, base.1 + normal_1);

            // Collisions are checked with a [0..1] f32 interval, but we extend it by a little bit to
            // make for precision losses.
            const EPSILON: f32 = 1e-4;
            const ZERO: f32 = -EPSILON;
            const ONE: f32 = 1.0 + EPSILON;

            for other in &flat_contour {
                // The selected edge and its adjacent edges don't need to be considered, because we adjust the angle of the real
                // tab to avoid crossing those.
                if other.i_edge == i_edge || other.i_edge == d0.i_edge || other.i_edge == d1.i_edge {
                    continue;
                }

                // Check the intersections with the edges of the imaginary tab:
                for (tab_sin, side) in [(tab_sin_0, side_0), (tab_sin_1, side_1)] {
                    let (_, o1, o2) = util_3d::line_line_intersection((other.p0, other.p1), side);
                    if (ZERO..=ONE).contains(&o1) && ZERO <= o2 {
                        minimum_width = minimum_width.min(o2 * tab_sin);
                    }
                }

                // Check the vertices of the other edge.
                // We can skip the vertices shared with the adjacent edges to the base
                // And since the contour is contiguous every edge appears twice, one as p0 and another as p1,
                // so we can check just one of them per edge
                if other.i_v0 != d1.i_v1
                    && util_3d::point_line_side(other.p0, side_0)
                        && !util_3d::point_line_side(other.p0, side_1)
                        && !util_3d::point_line_side(other.p0, base)
                        {
                            let (seg_0_off, seg_0_dist) = util_3d::point_line_distance(other.p0, base);
                            if (ZERO..=ONE).contains(&seg_0_off) {
                                minimum_width = minimum_width.min(seg_0_dist);
                            }
                        }
            }
            minimum_width
        };
        let base_len = e1.magnitude();
        let tab_area = |a0: Rad<f32>, a1: Rad<f32>, width: f32| -> f32 {
            if a0.0 <= 0.0 || a1.0 <= 0.0 {
                return 0.0;
            }
            let tan_0 = a0.cot();
            let tan_1 = a1.cot();
            let base2 = base_len - (tan_0 + tan_1) * width;
            if base2 > 0.0 {
                // trapezium
                width * (base_len + base2) / 2.0
            } else {
                // triangle
                let tri_width = base_len / (tan_0 + tan_1);
                tri_width * base_len / 2.0
            }
        };

        let mut a0 = a0;
        let mut a1 = a1;
        let mut width = compute_width(a0, a1);
        let mut area = tab_area(a0, a1, width);
        let (mut doing0, mut doing1) = (true, true);
        let step = Rad::from(Deg(5.0));
        while doing0 || doing1 {
            if doing0 {
                let a0x = a0 - step;
                if a0x > Rad(0.0) {
                    let w = compute_width(a0x, a1);
                    let a = tab_area(a0x, a1, w);
                    if a > area {
                        width = w;
                        area = a;
                        a0 = a0x;
                        doing1 = true;
                    } else {
                        doing0 = false;
                    }
                } else {
                    doing0 = false;
                }
            }
            if doing1 {
                let a1x = a1 - step;
                if a1x > Rad(0.0) {
                    let w = compute_width(a0, a1x);
                    let a = tab_area(a0, a1x, w);
                    if a > area {
                        width = w;
                        area = a;
                        a1 = a1x;
                        doing0 = true;
                    } else {
                        doing1 = false;
                    }
                } else {
                    doing1 = false;
                }
            }
        }
        if a0.0 <= 0.0 || a1.0 <= 0.0 {
            // Invalid tab
            TabGeom::default()
        } else {
            let tan_0 = a0.cot();
            let tan_1 = a1.cot();
            let base2 = base_len - (tan_0 + tan_1) * width;
            let triangular = base2 <= 0.0;
            if triangular {
                width = base_len / (tan_0 + tan_1);
            }
            TabGeom {
                tan_0,
                tan_1,
                width,
                triangular,
            }
        }
    }
    pub fn flat_face_rim_tab_dimensions(&self, i_face: FaceIndex, i_edge: EdgeIndex) -> TabGeom {
        let tab_angle = Rad::from(Deg(self.options.tab_angle));
        let tan = tab_angle.cot();
        let face = &self.model[i_face];
        let plane = self.model.face_plane(face);
        let edge = &self.model[i_edge];
        let v0 = self.model[edge.v0()].pos();
        let v1 = self.model[edge.v1()].pos();
        let scale = self.options.scale;
        let p0 = plane.project(&v0, scale);
        let p1 = plane.project(&v1, scale);
        let base_len = (p1 - p0).magnitude();
        let mut width = self.options.tab_width;
        let base2 = base_len - 2.0 * tan * width;
        let triangular = base2 <= 0.0;
        if triangular {
            width = base_len / (2.0 * tan);
        }
        TabGeom {
            tan_0: tan,
            tan_1: tan,
            width,
            triangular,
        }
    }

    pub fn traverse_faces<F>(&self, island: &Island, visit_face: F) -> ControlFlow<()>
        where F: FnMut(FaceIndex, &Face, &Matrix3) -> ControlFlow<()>
    {
        traverse_faces_ex(&self.model, island.root_face(), island.matrix(), NormalTraverseFace(self), visit_face)
    }
    pub fn traverse_faces_no_matrix<F>(&self, island: &Island, mut visit_face: F) -> ControlFlow<()>
        where F: FnMut(FaceIndex) -> ControlFlow<()>
    {
        traverse_faces_ex(&self.model, island.root_face(), (), NoMatrixTraverseFace(&self.model, &self.edges), |i, _, ()| visit_face(i))
    }
    pub fn try_join_strip(&mut self, i_edge: EdgeIndex) -> FxHashMap<IslandKey, JoinResult> {
        let mut renames = FxHashMap::default();
        let mut i_edges = vec![i_edge];
        while let Some(i_edge) = i_edges.pop() {
            // First try to join the edge, if it fails skip.
            let (i_face_a, i_face_b) = match self.model[i_edge].faces() {
                (a, Some(b)) => (a, b),
                _ => continue,
            };
            if !matches!(self.edge_status(i_edge), EdgeStatus::Cut(_)) {
                continue;
            }

            // Compute the number of faces before joining them
            let n_faces_a = self.island_face_count(self.island_by_key(self.island_by_face(i_face_a)).unwrap());
            let n_faces_b = self.island_face_count(self.island_by_key(self.island_by_face(i_face_b)).unwrap());
            if n_faces_a != 2 && n_faces_b != 2 {
                continue;
            }

            let r = self.edge_join(i_edge, None);
            if r.is_empty() {
                continue;
            }
            renames.extend(r);

            // Move to the opposite edge of both faces
            for (i_face, n_faces) in [(i_face_a, n_faces_a), (i_face_b, n_faces_b)] {
                // face strips must be made by isolated quads: 4 flat edges and 2 faces
                let edges: Vec<_> = self.get_flat_faces(i_face)
                    .into_iter()
                    .flat_map(|f| self.model[f].index_edges())
                    .filter(|&e| self.edge_status(e) != EdgeStatus::Hidden)
                    .collect();

                if n_faces != 2 {
                    continue;
                }
                if edges.len() != 4 {
                    continue;
                }
                // Get the opposite edge, if any
                let opposite = edges.iter().copied().find(|&i_e| {
                    if i_e == i_edge {
                        return false;
                    }
                    let edge = &self.model[i_edge];
                    let e = &self.model[i_e];
                    if e.v0() == edge.v0() || e.v0() == edge.v1() || e.v1() == edge.v0() || e.v1() == edge.v1() {
                        return false;
                    }
                    true
                });
                i_edges.extend(opposite);
            }
        }
        renames
    }

    pub fn pack_islands(&mut self) -> u32 {
        let mut row_height = 0.0f32;
        let mut pos_x = 0.0;
        let mut pos_y = 0.0;
        let mut num_in_row = 0;

        let mut page = 0;
        let page_margin = Vector2::new(self.options.margin.1, self.options.margin.0);
        let page_size = Vector2::new(
            self.options.page_size.0 - self.options.margin.1 - self.options.margin.2,
            self.options.page_size.1 - self.options.margin.0 - self.options.margin.3,
        );
        let mut zero = self.options().page_position(page) + page_margin;

        // The island position cannot be updated while iterating
        let mut positions = slotmap::SecondaryMap::<IslandKey, (Rad<f32>, Vector2)>::new();

        let mut ordered_islands: Vec<_> = self.islands
            .iter()
            .map(|(i_island, island)| {
                let (angle, bbox) = self.island_best_bounding_box(island);
                (i_island, angle, bbox)
            })
            .collect();
        ordered_islands.sort_by_key(|(_, _, bbox)| {
            let w = bbox.1.x - bbox.0.x;
            let h = bbox.1.y - bbox.0.y;
            -(w * h) as i64
        });

        for (i_island, angle, bbox) in ordered_islands {
            let mut next_pos_x = pos_x + bbox.1.x - bbox.0.x;
            if next_pos_x > page_size.x && num_in_row > 0 {
                next_pos_x -= pos_x;
                pos_x = 0.0;
                pos_y += row_height;
                row_height = 0.0;
                num_in_row = 0;
                if pos_y > page_size.y {
                    pos_y = 0.0;
                    page += 1;
                    zero = self.options().page_position(page) + page_margin;
                }
            }
            let pos = Vector2::new(pos_x - bbox.0.x, pos_y - bbox.0.y);
            pos_x = next_pos_x;
            row_height = row_height.max(bbox.1.y - bbox.0.y);
            num_in_row += 1;

            positions.insert(i_island, (angle, zero + pos));
        }
        for (i_island, (angle, pos)) in positions {
            let island = self.island_by_key_mut(i_island).unwrap();
            island.loc += pos;
            island.rot += angle;
            island.recompute_matrix();
        }
        page + 1
    }
}

fn traverse_faces_ex<F, TP>(model: &Model, root: FaceIndex, initial_state: TP::State, policy: TP, mut visit_face: F) -> ControlFlow<()>
where F: FnMut(FaceIndex, &Face, &TP::State) -> ControlFlow<()>,
      TP: TraverseFacePolicy,
{
    let mut visited_faces = FxHashSet::default();
    let mut stack = vec![(root, initial_state)];
    visited_faces.insert(root);

    while let Some((i_face, m)) = stack.pop() {
        let face = &model[i_face];
        visit_face(i_face, face, &m)?;
        for i_edge in face.index_edges() {
            if !policy.cross_edge(i_edge) {
                continue;
            }
            let edge = &model[i_edge];
            let (fa, fb) = edge.faces();
            for i_next_face in std::iter::once(fa).chain(fb) {
                if visited_faces.contains(&i_next_face) {
                    continue;
                }
                let next_state = policy.next_state(&m, edge, face, i_next_face);
                stack.push((i_next_face, next_state));
                visited_faces.insert(i_next_face);
            }
        }
    };
    ControlFlow::Continue(())
}

trait TraverseFacePolicy {
    type State;
    fn cross_edge(&self, i_edge: EdgeIndex) -> bool;
    fn next_state(&self, st: &Self::State, edge: &Edge, face: &Face, i_next_face: FaceIndex) -> Self::State;
}
struct NormalTraverseFace<'a>(&'a Papercraft);

impl TraverseFacePolicy for NormalTraverseFace<'_> {
    type State = Matrix3;

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.0.edges[usize::from(i_edge)] {
            EdgeStatus::Cut(_) => false,
            EdgeStatus::Joined |
            EdgeStatus::Hidden => true,
        }
    }

    fn next_state(&self, st: &Self::State, edge: &Edge, face: &Face, i_next_face: FaceIndex) -> Self::State {
        let next_face = &self.0.model[i_next_face];
        let medge = self.0.face_to_face_edge_matrix(edge, face, next_face);
        st * medge
    }
}

struct NoMatrixTraverseFace<'a>(&'a Model, &'a [EdgeStatus]);

impl TraverseFacePolicy for NoMatrixTraverseFace<'_> {
    type State = ();

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.1[usize::from(i_edge)] {
            EdgeStatus::Cut(_) => false,
            EdgeStatus::Joined |
            EdgeStatus::Hidden => true,
        }
    }

    fn next_state(&self, _st: &Self::State, _edge: &Edge, _face: &Face, _i_next_face: FaceIndex) -> Self::State {
    }
}

struct FlatTraverseFace<'a>(&'a Papercraft);

impl TraverseFacePolicy for FlatTraverseFace<'_> {
    type State = ();

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.0.edge_status(i_edge) {
            EdgeStatus::Joined |
            EdgeStatus::Cut(_) => false,
            EdgeStatus::Hidden => true,
        }
    }

    fn next_state(&self, _st: &Self::State, _edge: &Edge, _face: &Face, _i_next_face: FaceIndex) -> Self::State {
    }
}

struct FlatTraverseFaceWithMatrix<'a>(&'a Papercraft);


impl TraverseFacePolicy for FlatTraverseFaceWithMatrix<'_> {
    type State = Matrix3;

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.0.edge_status(i_edge) {
            EdgeStatus::Joined |
            EdgeStatus::Cut(_) => false,
            EdgeStatus::Hidden => true,
        }
    }

    fn next_state(&self, st: &Self::State, edge: &Edge, face: &Face, i_next_face: FaceIndex) -> Self::State {
        let next_face = &self.0.model[i_next_face];
        let medge = self.0.face_to_face_edge_matrix(edge, face, next_face);
        st * medge
    }
}


#[derive(Debug)]
pub struct Island {
    root: FaceIndex,

    rot: Rad<f32>,
    loc: Vector2,
    mx: Matrix3,
}

impl Island {
    pub fn root_face(&self) -> FaceIndex {
        self.root
    }
    pub fn rotation(&self) -> Rad<f32> {
        self.rot
    }
    pub fn location(&self) -> Vector2 {
        self.loc
    }
    pub fn matrix(&self) -> Matrix3 {
        self.mx
    }
    pub fn reset_transformation(&mut self, root_face: FaceIndex, rot: Rad<f32>, loc: Vector2) {
        //WARNING: root_face should be already of this island
        self.root = root_face;
        self.rot = rot;
        self.loc = loc;
        self.recompute_matrix();
    }
    pub fn translate(&mut self, delta: Vector2) {
        self.loc += delta;
        self.recompute_matrix();
    }
    pub fn rotate(&mut self, angle: impl Into<Rad<f32>>, center: Vector2) {
        let angle = angle.into();
        self.rot = (self.rot + angle).normalize();
        self.loc = center + Matrix2::from_angle(angle) * (self.loc - center);

        self.recompute_matrix();
    }
    fn recompute_matrix(&mut self) {
        let r = Matrix3::from(cgmath::Matrix2::from_angle(self.rot));
        let t = Matrix3::from_translation(self.loc);
        self.mx = t * r;
    }
}

impl Serialize for EdgeStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        let is = match self {
            EdgeStatus::Hidden => 0,
            EdgeStatus::Joined => 1,
            EdgeStatus::Cut(TabSide::False) => 2,
            EdgeStatus::Cut(TabSide::True) => 3,
            EdgeStatus::Cut(TabSide::Hidden) => 4,
        };
        serializer.serialize_i32(is)
    }
}
impl<'de> Deserialize<'de> for EdgeStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        let d = u32::deserialize(deserializer)?;
        let res = match d {
            0 => EdgeStatus::Hidden,
            1 => EdgeStatus::Joined,
            2 => EdgeStatus::Cut(TabSide::False),
            3 => EdgeStatus::Cut(TabSide::True),
            4 => EdgeStatus::Cut(TabSide::Hidden),
            _ => return Err(serde::de::Error::missing_field("invalid edge status")),
        };
        Ok(res)
    }
}

impl Serialize for TabStyle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        let is = match self {
            TabStyle::Textured => 0,
            TabStyle::HalfTextured => 1,
            TabStyle::White => 2,
            TabStyle::None => 3,
        };
        serializer.serialize_i32(is)
    }
}
impl<'de> Deserialize<'de> for TabStyle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        let d = u32::deserialize(deserializer)?;
        let res = match d {
            0 => TabStyle::Textured,
            1 => TabStyle::HalfTextured,
            2 => TabStyle::White,
            3 => TabStyle::None,
            _ => return Err(serde::de::Error::missing_field("invalid tab_style value")),
        };
        Ok(res)
    }
}

impl Serialize for FoldStyle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        let is = match self {
            FoldStyle::Full => 0,
            FoldStyle::FullAndOut => 1,
            FoldStyle::Out => 2,
            FoldStyle::In => 3,
            FoldStyle::InAndOut => 4,
            FoldStyle::None => 5,
        };
        serializer.serialize_i32(is)
    }
}
impl<'de> Deserialize<'de> for FoldStyle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        let d = u32::deserialize(deserializer)?;
        let res = match d {
            0 => FoldStyle::Full,
            1 => FoldStyle::FullAndOut,
            2 => FoldStyle::Out,
            3 => FoldStyle::In,
            4 => FoldStyle::InAndOut,
            5 => FoldStyle::None,
            _ => return Err(serde::de::Error::missing_field("invalid fold_style value")),
        };
        Ok(res)
    }
}

impl Serialize for Island {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        let mut map = serializer.serialize_struct("Island", 4)?;
        map.serialize_field("root", &usize::from(self.root))?;
        map.serialize_field("x", &self.loc.x)?;
        map.serialize_field("y", &self.loc.y)?;
        map.serialize_field("r", &self.rot.0)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for Island {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        #[derive(Deserialize)]
        struct Def { root: usize, x: f32, y: f32, r: f32 }
        let d = Def::deserialize(deserializer)?;
        let mut island = Island {
            root: FaceIndex::from(d.root),
            loc: Vector2::new(d.x, d.y),
            rot: Rad(d.r),
            mx: Matrix3::one(),
        };
        island.recompute_matrix();
        Ok(island)
}
}

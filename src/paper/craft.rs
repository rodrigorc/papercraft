use std::num::NonZeroU32;
use std::ops::ControlFlow;
use std::{cell::RefCell, rc::Rc};

use crate::util_3d;
use crate::util_gl::MLine3DStatus;
use cgmath::{Deg, Rad, prelude::*};
use easy_imgui_window::easy_imgui::Color;
use easy_imgui_window::easy_imgui_renderer::easy_imgui_opengl::Rgba;
use fxhash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use slotmap::{SlotMap, new_key_type};

use super::*;
mod file;
mod update;

// Which side of a cut will the flap be drawn, compare with face_sign
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum FlapSide {
    False,
    True,
    Hidden,
}

pub enum EdgeToggleFlapAction {
    Toggle,
    Hide,
    Set(FlapSide),
}

impl FlapSide {
    pub fn apply(self, action: EdgeToggleFlapAction, rim: bool) -> FlapSide {
        use FlapSide::*;
        match (self, action) {
            (_, EdgeToggleFlapAction::Set(next)) => next,
            // toggle
            (False, EdgeToggleFlapAction::Toggle) => {
                if rim {
                    Hidden
                } else {
                    True
                }
            }
            (True, EdgeToggleFlapAction::Toggle) => False,
            (Hidden, EdgeToggleFlapAction::Toggle) => False,

            // hide
            (False, EdgeToggleFlapAction::Hide) => Hidden,
            (True, EdgeToggleFlapAction::Hide) => Hidden,
            (Hidden, EdgeToggleFlapAction::Hide) => False,
        }
    }
    pub fn flap_visible(self, face_sign: bool) -> bool {
        matches!(
            (self, face_sign),
            (FlapSide::False, false) | (FlapSide::True, true)
        )
    }
    pub fn opposite(self) -> FlapSide {
        match self {
            FlapSide::False => FlapSide::True,
            FlapSide::True => FlapSide::False,
            FlapSide::Hidden => FlapSide::Hidden,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum EdgeStatus {
    Hidden,
    Joined,
    Cut(FlapSide),
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub enum FlapStyle {
    Textured,
    #[default]
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

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub enum EdgeIdPosition {
    None,
    #[default]
    Outside,
    Inside,
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

fn my_true() -> bool {
    true
}
fn default_fold_line_color() -> MyColor {
    MyColor(Color::BLACK)
}
fn default_fold_line_width() -> f32 {
    0.1
}

fn default_cut_line_color() -> MyColor {
    MyColor(Color::BLACK)
}
fn default_cut_line_width() -> f32 {
    0.1
}

fn default_tab_line_color() -> MyColor {
    MyColor(Color::BLACK)
}
fn default_tab_line_width() -> f32 {
    0.2
}

fn default_edge_id_font_size() -> f32 {
    8.0
}

fn default_line3d_normal() -> LineConfig {
    LineConfig {
        thick: 1.0,
        color: Color::BLACK,
    }
}

fn default_line3d_rim() -> LineConfig {
    LineConfig {
        thick: 1.0,
        color: Color::YELLOW,
    }
}

fn default_line3d_rim_tab() -> LineConfig {
    LineConfig {
        thick: 5.0,
        color: Color::new(0.75, 0.75, 0.0, 1.0),
    }
}

fn default_line3d_cut() -> LineConfig {
    LineConfig {
        thick: 3.0,
        color: Color::WHITE,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MyColor(pub Color);

impl MyColor {
    pub fn to_rgba(&self) -> Rgba {
        Rgba::new(self.0.r, self.0.g, self.0.b, self.0.a)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LineConfig {
    pub thick: f32,
    pub color: Color,
}

impl LineConfig {
    pub fn to_3dstatus(&self, def: &MLine3DStatus) -> MLine3DStatus {
        MLine3DStatus {
            thick: self.thick / 2.0,
            color: Rgba::new(self.color.r, self.color.g, self.color.b, self.color.a),
            ..*def
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PaperOptions {
    pub scale: f32,
    pub page_size: (f32, f32),
    pub resolution: u32, //dpi
    pub pages: u32,
    pub page_cols: u32,
    pub margin: (f32, f32, f32, f32), //top, left, right, bottom
    #[serde(default = "my_true")]
    pub texture: bool,
    #[serde(default = "my_true")]
    pub tex_filter: bool,
    #[serde(default, rename = "tab_style")]
    pub flap_style: FlapStyle,
    #[serde(default)]
    pub fold_style: FoldStyle,
    #[serde(rename = "tab_width")]
    pub flap_width: f32,
    #[serde(rename = "tab_angle")]
    pub flap_angle: f32, //degrees
    pub fold_line_len: f32, //only for folds in & out
    #[serde(default, rename = "shadow_tab_alpha")]
    pub shadow_flap_alpha: f32, //0.0 - 1.0
    // Do not use LineConfig for compatibility with older models
    #[serde(default = "default_fold_line_color")]
    pub fold_line_color: MyColor,
    #[serde(default = "default_fold_line_width")]
    pub fold_line_width: f32, //only for folds in & out
    #[serde(default = "default_cut_line_color")]
    pub cut_line_color: MyColor,
    #[serde(default = "default_cut_line_width")]
    pub cut_line_width: f32, //for cuts without tab
    #[serde(default = "default_tab_line_color")]
    pub tab_line_color: MyColor,
    #[serde(default = "default_tab_line_width")]
    pub tab_line_width: f32, //for cuts with tab
    #[serde(default)]
    pub hidden_line_angle: f32, //degrees
    #[serde(default = "my_true")]
    pub show_self_promotion: bool,
    #[serde(default = "my_true")]
    pub show_page_number: bool,
    #[serde(default = "default_edge_id_font_size")]
    pub edge_id_font_size: f32,
    #[serde(default)]
    pub edge_id_position: EdgeIdPosition,
    #[serde(default)]
    pub island_name_only: bool,
    #[serde(default = "default_line3d_normal")]
    pub line3d_normal: LineConfig,
    #[serde(default = "default_line3d_rim")]
    pub line3d_rim: LineConfig,
    #[serde(default = "default_line3d_rim_tab")]
    pub line3d_rim_tab: LineConfig,
    #[serde(default = "default_line3d_cut")]
    pub line3d_cut: LineConfig,
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
            flap_style: FlapStyle::default(),
            fold_style: FoldStyle::default(),
            flap_width: 5.0,
            flap_angle: 45.0,
            fold_line_len: 4.0,
            shadow_flap_alpha: 0.0,
            fold_line_color: default_fold_line_color(),
            fold_line_width: default_fold_line_width(),
            cut_line_color: default_cut_line_color(),
            cut_line_width: default_cut_line_width(),
            tab_line_color: default_tab_line_color(),
            tab_line_width: default_tab_line_width(),
            hidden_line_angle: 0.0,
            show_self_promotion: true,
            show_page_number: true,
            edge_id_font_size: default_edge_id_font_size(),
            edge_id_position: EdgeIdPosition::default(),
            island_name_only: false,
            line3d_normal: default_line3d_normal(),
            line3d_rim: default_line3d_rim(),
            line3d_rim_tab: default_line3d_rim_tab(),
            line3d_cut: default_line3d_cut(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PageOffset {
    pub row: i32,
    pub col: i32,
    pub offset: Vector2,
}

const PAGE_SEP: f32 = 10.0; // Currently not configurable
//
impl PaperOptions {
    pub fn page_position(&self, page: u32) -> Vector2 {
        let page_cols = self.page_cols;
        let row = page / page_cols;
        let col = page % page_cols;
        self.row_col_position(row as i32, col as i32)
    }
    fn row_col_position(&self, row: i32, col: i32) -> Vector2 {
        Vector2::new(
            (col as f32) * (self.page_size.0 + PAGE_SEP),
            (row as f32) * (self.page_size.1 + PAGE_SEP),
        )
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
        let col = ((pos.x / (page_size.x + PAGE_SEP)) as i32).clamp(0, page_cols as i32);
        let row = ((pos.y / (page_size.y + PAGE_SEP)) as i32).max(0);

        let zero_pos = self.row_col_position(row, col);
        let offset = pos - zero_pos;
        PageOffset { row, col, offset }
    }
    pub fn page_to_global(&self, po: PageOffset) -> Vector2 {
        let zero_pos = self.row_col_position(po.row, po.col);
        zero_pos + po.offset
    }
    pub fn is_inside_canvas(&self, pos: Vector2) -> bool {
        let page_cols = self.page_cols;
        let page_rows = self.pages.div_ceil(self.page_cols);
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
    #[serde(with = "super::ser::slot_map")]
    islands: SlotMap<IslandKey, Island>,

    #[serde(skip)]
    memo: Memoization,
    #[serde(skip)]
    edge_ids: Vec<Option<EdgeId>>, //parallel to EdgeIndex
}

/// The printable edge id, not to be confused with EdgeIndex
#[derive(Copy, Clone, Debug)]
pub struct EdgeId(NonZeroU32);

impl EdgeId {
    fn new(id: u32) -> EdgeId {
        EdgeId(NonZeroU32::new(id).unwrap())
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.get().fmt(fmt)
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct FlapGeom {
    pub tan_0: f32,
    pub tan_1: f32,
    pub width: f32,
    pub triangular: bool,
}

#[derive(Debug, Clone)]
pub struct FlapEdgeData {
    i_edge: EdgeIndex,
    face_sign: bool,
    // Do not assume that these coordinates are the real ones in the paper, the movement of the island is not tracked.
    p0: Vector2,
    p1: Vector2,
}

impl FlapEdgeData {
    pub fn i_edge(&self) -> EdgeIndex {
        self.i_edge
    }
    pub fn face_sign(&self) -> bool {
        self.face_sign
    }
    // Do not expose p0/p1 for now, because they have an arbitraty transformation
}

type FlatFaceFlapDimensions = FxHashMap<(FaceIndex, EdgeIndex), FlapGeom>;

#[derive(Default)]
struct Memoization {
    // This depends on the options, because of the scale, but not on the islands,
    // because it is computed as if both faces are joined.
    face_to_face_edge_matrix: RefCell<FxHashMap<(EdgeIndex, FaceIndex, FaceIndex), Matrix3>>,

    // This depends on the islands, but not on the options
    // Indexed by FaceIndex
    island_by_face: RefCell<Vec<IslandKey>>,

    // This depends on the options and the islands
    flat_face_flap_dimensions: RefCell<FxHashMap<IslandKey, FlatFaceFlapDimensions>>,

    // This depends on the islands, but not on the options
    island_perimeters: RefCell<FxHashMap<IslandKey, Rc<[FlapEdgeData]>>>,
}

impl Memoization {
    fn invalidate_options(&self) {
        self.face_to_face_edge_matrix.borrow_mut().clear();
        self.flat_face_flap_dimensions.borrow_mut().clear();
        self.island_perimeters.borrow_mut().clear();
    }
    fn invalidate_islands(&self, islands: &[IslandKey]) {
        self.island_by_face.borrow_mut().clear();

        let mut flat_face_flap_dimensions = self.flat_face_flap_dimensions.borrow_mut();
        let mut island_perimeters = self.island_perimeters.borrow_mut();
        for island in islands {
            flat_face_flap_dimensions.remove(island);
            island_perimeters.remove(island);
        }
    }
}

type OrderedContour = Vec<(EdgeIndex, bool)>;

impl Papercraft {
    pub fn empty() -> Papercraft {
        Papercraft {
            model: Model::empty(),
            options: PaperOptions::default(),
            edges: Vec::new(),
            islands: SlotMap::with_key(),
            memo: Memoization::default(),
            edge_ids: Vec::new(),
        }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn options(&self) -> &PaperOptions {
        &self.options
    }
    // Returns the old options
    pub fn set_options(
        &mut self,
        mut options: PaperOptions,
        relocate_pieces: bool,
    ) -> PaperOptions {
        let scale = options.scale / self.options.scale;
        // Compute positions relative to the nearest page
        let page_pos: Option<FxHashMap<_, _>> = if relocate_pieces {
            Some(
                self.islands
                    .iter()
                    .map(|(i_island, island)| {
                        let mut po = self.options.global_to_page(island.location());
                        po.offset *= scale;
                        (i_island, po)
                    })
                    .collect(),
            )
        } else {
            None
        };

        // Apply the new options
        std::mem::swap(&mut self.options, &mut options);
        // Invalidate the memoized values that may depend on any option
        self.memo.invalidate_options();

        // Apply the new positions
        if let Some(page_pos) = page_pos {
            for (i_island, po) in page_pos {
                let loc = self.options.page_to_global(po);
                if let Some(island) = self.island_by_key_mut(i_island) {
                    island.loc = loc;
                    island.recompute_matrix();
                }
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
    pub fn island_bounding_box_angle(
        &self,
        island: &Island,
        angle: Rad<f32>,
    ) -> (Vector2, Vector2) {
        let mx = island.matrix() * Matrix3::from(Matrix2::from_angle(angle));
        let mut vx = Vec::new();
        let _ = traverse_faces_ex(
            &self.model,
            island.root_face(),
            mx,
            NormalTraverseFace(self),
            |_, face, mx| {
                let vs = face.index_vertices().map(|v| {
                    let normal = self.model.face_plane(face);
                    mx.transform_point(Point2::from_vec(
                        normal.project(&self.model[v].pos(), self.options.scale),
                    ))
                    .to_vec()
                });
                vx.extend(vs);
                ControlFlow::Continue(())
            },
        );

        let (a, b) = util_3d::bounding_box_2d(vx);
        let m = self.options.flap_width;
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
        for _ in 1..TRIES {
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
        // Try to use a memoized value
        let mut memo = self.memo.island_by_face.borrow_mut();
        if memo.is_empty() {
            self.rebuild_island_by_face(&mut memo);
        }
        memo[usize::from(i_face)]
    }
    fn rebuild_island_by_face(&self, memo: &mut Vec<IslandKey>) {
        // This could be optimized and updated every time an island is split/joined,
        // instead of creating it from scratch every time, but is it worth it?
        memo.resize(self.model.num_faces(), IslandKey::default());
        for (i_island, island) in self.islands() {
            let _ = self.traverse_faces_no_matrix(island, |i_face| {
                memo[usize::from(i_face)] = i_island;
                ControlFlow::Continue(())
            });
        }
    }
    // Islands come and go, so this key may not exist.
    pub fn island_by_key(&self, key: IslandKey) -> Option<&Island> {
        self.islands.get(key)
    }
    pub fn island_by_key_mut(&mut self, key: IslandKey) -> Option<&mut Island> {
        self.islands.get_mut(key)
    }
    pub fn rebuild_island_names(&mut self) {
        // To get somewhat predictable names try to sort the islands before naming them.
        // For now, sort them by number of faces.
        let mut islands: Vec<_> = self
            .islands
            .iter()
            .map(|(i_island, island)| (i_island, self.island_area(island)))
            .collect();
        islands.sort_by(|(_, n1), (_, n2)| n2.total_cmp(n1));

        // A, B, ... Z, AA, ... AZ, BA, .... ZZ, AAA, AAB, ...
        fn next_name(name: &mut Vec<u8>) {
            for ch in name.iter_mut().rev() {
                if *ch < b'Z' {
                    *ch += 1;
                    return;
                }
                *ch = b'A';
            }
            // The new 'A' is at the beginning, but here they are all 'A's, so it
            // doesn't matter.
            name.push(b'A');
        }

        let mut island_name = Vec::new();
        for (i_island, _) in &islands {
            next_name(&mut island_name);
            self.islands[*i_island].name = String::from_utf8(island_name.clone()).unwrap();
        }
    }

    pub fn edge_status(&self, edge: EdgeIndex) -> EdgeStatus {
        self.edges[usize::from(edge)]
    }
    pub fn edge_id(&self, edge: EdgeIndex) -> Option<EdgeId> {
        if self.options.edge_id_font_size <= 0.0
            || self.options.edge_id_position == EdgeIdPosition::None
        {
            return None;
        }
        self.edge_ids[usize::from(edge)]
    }

    pub fn edge_toggle_flap(
        &mut self,
        i_edge: EdgeIndex,
        action: EdgeToggleFlapAction,
    ) -> Option<FlapSide> {
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
            _ => {
                return;
            }
        }
        let edge = &self.model[i_edge];
        let (i_face_a, i_face_b) = match edge.faces() {
            (fa, Some(fb)) => (fa, fb),
            _ => {
                return;
            }
        };

        //one of the edge faces will be the root of the new island, but we do not know which one, yet
        let i_island = self.island_by_face(i_face_a);

        self.edges[usize::from(i_edge)] = EdgeStatus::Cut(FlapSide::False);

        let mut data_found = None;
        let _ = self.traverse_faces(&self.islands[i_island], |i_face, _, fmx| {
            if i_face == i_face_a {
                data_found = Some((*fmx, i_face_b, i_face_a));
            } else if i_face == i_face_b {
                data_found = Some((*fmx, i_face_a, i_face_b));
            }
            ControlFlow::Continue(())
        });
        let (face_mx, new_root, i_face_old) = data_found.unwrap();

        let medge =
            self.face_to_face_edge_matrix(edge, &self.model[i_face_old], &self.model[new_root]);
        let mx = face_mx * medge;

        let mut new_island = Island {
            root: new_root,
            loc: Vector2::new(mx[2][0], mx[2][1]),
            rot: Rad(mx[0][1].atan2(mx[0][0])),
            mx: Matrix3::one(),
            name: String::new(),
        };
        new_island.recompute_matrix();

        //Compute the offset
        if let Some(offset_on_cut) = offset {
            let sign = if edge.face_sign(new_root) { 1.0 } else { -1.0 };
            let new_root = &self.model[new_root];
            let new_root_plane = self.model.face_plane(new_root);
            let (v0, v1) = self.model.edge_pos(edge);
            let v0 = new_root_plane.project(&v0, self.options.scale);
            let v1 = new_root_plane.project(&v1, self.options.scale);
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
        let i_new_island = self.islands.insert(new_island);
        self.memo.invalidate_islands(&[i_island, i_new_island]);
    }

    //Retuns a map from the island that disappears into the extra join data.
    pub fn edge_join(
        &mut self,
        i_edge: EdgeIndex,
        priority_face: Option<FaceIndex>,
    ) -> FxHashMap<IslandKey, JoinResult> {
        let mut renames = FxHashMap::default();
        match self.edges[usize::from(i_edge)] {
            EdgeStatus::Cut(_) => {}
            _ => {
                return renames;
            }
        }
        let edge = &self.model[i_edge];
        let (i_face_a, i_face_b) = match edge.faces() {
            (fa, Some(fb)) => (fa, fb),
            _ => {
                return renames;
            }
        };

        let i_island_b = self.island_by_face(i_face_b);
        if self.contains_face(&self.islands[i_island_b], i_face_a) {
            // Same island on both sides, nothing to do
            return renames;
        }

        // Join both islands
        let i_island_a = self.island_by_face(i_face_a);
        self.memo.invalidate_islands(&[i_island_a, i_island_b]);
        let mut island_b = self.islands.remove(i_island_b).unwrap();

        // Keep position of a or b?
        if self.compare_islands(&self.islands[i_island_a], &island_b, priority_face) {
            std::mem::swap(&mut self.islands[i_island_a], &mut island_b);
        }
        renames.insert(
            i_island_b,
            JoinResult {
                i_edge,
                i_island: i_island_a,
                prev_root: island_b.root_face(),
                prev_rot: island_b.rotation(),
                prev_loc: island_b.location(),
            },
        );
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
        let _ = self.traverse_faces_no_matrix(island, |i_face| {
            if i_face == face {
                found = true;
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        found
    }
    pub fn island_face_count(&self, island: &Island) -> u32 {
        let mut count = 0;
        let _ = self.traverse_faces_no_matrix(island, |_| {
            count += 1;
            ControlFlow::Continue(())
        });
        count
    }
    pub fn island_area(&self, island: &Island) -> f32 {
        let mut area = 0.0;
        let _ = self.traverse_faces_no_matrix(island, |face| {
            area += self.model().face_area(face);
            ControlFlow::Continue(())
        });
        area
    }
    pub fn get_flat_faces(&self, i_face: FaceIndex) -> FxHashSet<FaceIndex> {
        let mut res = FxHashSet::default();
        let _ = traverse_faces_ex(
            &self.model,
            i_face,
            (),
            FlatTraverseFace(self),
            |i_next_face, _, _| {
                res.insert(i_next_face);
                ControlFlow::Continue(())
            },
        );
        res
    }
    fn get_flat_faces_with_matrix(&self, i_face: FaceIndex) -> FxHashMap<FaceIndex, Matrix3> {
        let mut res = FxHashMap::default();
        let _ = traverse_faces_ex(
            &self.model,
            i_face,
            Matrix3::one(),
            FlatTraverseFaceWithMatrix(self),
            |i_next_face, _, mx| {
                res.insert(i_next_face, *mx);
                ControlFlow::Continue(())
            },
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
            Occupied(o) => *o.get(),
            Vacant(v) => {
                let value = self.face_to_face_edge_matrix_internal(edge, face_a, face_b);
                *v.insert(value)
            }
        }
    }
    fn face_to_face_edge_matrix_internal(
        &self,
        edge: &Edge,
        face_a: &Face,
        face_b: &Face,
    ) -> Matrix3 {
        let (v0, v1) = self.model.edge_pos(edge);
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
    // Returns the max. angles of the flap sides, actually their cotangent, and the max. width.
    // Ideally it should return all the flap metrics
    pub fn flat_face_flap_dimensions(
        &self,
        i_face_a: FaceIndex,
        i_face_b: Option<FaceIndex>,
        i_edge: EdgeIndex,
    ) -> FlapGeom {
        // Try to use a memoized value
        let mut memo = self.memo.flat_face_flap_dimensions.borrow_mut();
        use std::collections::hash_map::Entry::*;
        let i_island = self.island_by_face(i_face_a);
        let island_data = memo.entry(i_island).or_default();
        match island_data.entry((i_face_a, i_edge)) {
            Occupied(o) => *o.get(),
            Vacant(v) => {
                let value = self.flat_face_flap_dimensions_internal(i_face_a, i_face_b, i_edge);
                *v.insert(value)
            }
        }
    }
    pub fn island_edges(&self, island: &Island) -> FxHashSet<EdgeIndex> {
        let mut res = FxHashSet::default();
        let _ = self.traverse_faces_no_matrix(island, |i_f| {
            res.extend(
                self.model[i_f]
                    .index_edges()
                    .iter()
                    .filter(|&&i| matches!(self.edge_status(i), EdgeStatus::Cut(_))),
            );
            ControlFlow::Continue(())
        });
        res
    }
    fn flat_face_flap_dimensions_internal(
        &self,
        i_face_a: FaceIndex,
        i_face_b: Option<FaceIndex>,
        i_edge: EdgeIndex,
    ) -> FlapGeom {
        // Compute the flat-face_b contour
        let scale = self.options.scale;
        let flap_angle = Rad::from(Deg(self.options.flap_angle));
        let mut a0 = flap_angle;
        let mut a1 = flap_angle;

        let SelfCollisionPerimeter {
            perimeter,
            perimeter_egde_base,
            angle_0: (angle_0, perimeter_egde_0),
            angle_1: (angle_1, perimeter_egde_1),
        } = self.self_collision_perimeter(i_edge, i_face_a);
        if perimeter.is_empty() {
            // should not happen
            return FlapGeom::default();
        }

        a0 = Rad(a0.0.min(angle_0.0));
        a1 = Rad(a1.0.min(angle_1.0));

        let base = &perimeter[perimeter_egde_base];
        let base = (base.p0, base.p1);
        let base_vec = base.1 - base.0;

        let flat_contour = if let Some(i_face_b) = i_face_b {
            let flat_face = self.get_flat_faces_with_matrix(i_face_b);

            // flat_perimeter will be the ordered perimeter of the flat extension of i_face_b, but with the coordinates contiguous to i_face_a
            let contour = self.flat_face_contour(i_face_b);
            let mut base_index = None;
            let mut flat_perimeter: Vec<FlapEdgeData> = contour
                .into_iter()
                .enumerate()
                .map(|(i, (i_e, s))| {
                    let edge = &self.model[i_e];
                    let i_face = edge.face_by_sign(s).unwrap();
                    let face = &self.model[i_face];
                    let (i_v0, i_v1) = face.vertices_of_edge(i_e).unwrap();
                    let plane = self.model.face_plane(face);
                    let mx = flat_face[&i_face];
                    let p0 = plane.project(&self.model()[i_v0].pos(), scale);
                    let p1 = plane.project(&self.model()[i_v1].pos(), scale);
                    let p0 = mx.transform_point(Point2::from_vec(p0)).to_vec();
                    let p1 = mx.transform_point(Point2::from_vec(p1)).to_vec();
                    if i_e == i_edge && i_face == i_face_b {
                        base_index = Some(i);
                    }
                    FlapEdgeData {
                        i_edge: i_e,
                        face_sign: s,
                        p0,
                        p1,
                    }
                })
                .collect();

            let base_index = base_index.unwrap();
            let d0 =
                &flat_perimeter[(base_index + flat_perimeter.len() - 1) % flat_perimeter.len()];
            let d1 = &flat_perimeter[(base_index + 1) % flat_perimeter.len()];

            // ** Compute max flap angles **
            let e0 = d0.p1 - d0.p0;
            let e1 = d1.p0 - d0.p1;
            let e2 = d1.p1 - d1.p0;
            let angle0 = Rad::turn_div_2() - e1.angle(e0);
            let angle1 = Rad::turn_div_2() - e2.angle(e1);

            a0 = Rad(a0.0.min(angle0.0));
            a1 = Rad(a1.0.min(angle1.0));

            // Convert the flat_contour to island coordinates, we know that this edge should match, inverted
            let mx = Matrix3::from_translation(base.0)
                * Matrix3::from_angle_z((d0.p1 - d1.p0).angle(base_vec))
                * Matrix3::from_translation(-d1.p0);

            for edata in &mut flat_perimeter {
                edata.p0 = mx.transform_point(Point2::from_vec(edata.p0)).to_vec();
                edata.p1 = mx.transform_point(Point2::from_vec(edata.p1)).to_vec();
            }

            Some((flat_perimeter, base_index))
        } else {
            // It is a rim edge, no corresponding flat face
            None
        };

        // ** Compute max flap width **
        //
        // Maximum width is computed with a flap angle of 90°, that is not exact, but the difference
        // should be quite small and always to the safe side.
        // We look in the flat_contour of the face B for the nearest point to the selected edge, that falls inside
        // an imaginary flap of 90° angle and infinite width.
        // This nearest point must be either a vertex or the intersection of an edge with the sides of the flap.

        // Normalize the perpendicular to make easier to compute the distance later, then it is just the offset returned by util_3d.
        let n = Vector2::new(-base_vec.y, base_vec.x).normalize();
        let base_len = base_vec.magnitude();

        // (a0,a1) are the biggest angles that fit, and usually smaller angles will result in
        // smaller width too, but in some edge cases a smaller angle will result in a larger width,
        // and the user will probably prefer the biggest flap area, so we try with some extra angles and see
        // what happens

        let compute_width = |a0: Rad<f32>, a1: Rad<f32>| -> f32 {
            let mut minimum_width = self.options.flap_width;
            let (flap_sin_0, flap_cos_0) = a0.sin_cos();
            let normal_0 = Vector2::new(
                n.x * flap_sin_0 - n.y * flap_cos_0,
                n.x * flap_cos_0 + n.y * flap_sin_0,
            );
            //90° is the original normal so switch the sin/cos in these rotations
            let (flap_sin_1, flap_cos_1) = a1.sin_cos();
            let normal_1 = Vector2::new(
                n.x * flap_sin_1 + n.y * flap_cos_1,
                -n.x * flap_cos_1 + n.y * flap_sin_1,
            );

            // base is inverted here, because the flap goes to the outside, so side_0 is in the second base point, side_1 is in the first one.
            let side_0 = (base.1, base.1 + normal_0);
            let side_1 = (base.0, base.0 + normal_1);

            // Collisions are checked with a [0..1] f32 interval, but we extend it by a little bit to
            // make for precision losses.
            // Ideally, we would want to make the (0..1) range a bit longer if the angle of `other` to the next edge
            // is >180° and shorter if the angle is <180°, but doesn't seem to be worth it. Making it always a bit longer
            // works for a more conservative solution.
            const EPSILON: f32 = 1e-4;
            const ZERO: f32 = -EPSILON;
            const ONE: f32 = 1.0 + EPSILON;

            // Compute the lines that will limit this flap
            let from_flat_contour = flat_contour.iter().flat_map(|(flat_contour, base_index)| {
                flat_contour
                    .iter()
                    .enumerate()
                    .filter_map(|(index, other)| {
                        // The selected edge and its adjacent edges don't need to be considered, because we adjust the angle of the real
                        // flap to avoid crossing those.
                        let prev = (*base_index + flat_contour.len() - 1) % flat_contour.len();
                        let next = (*base_index + 1) % flat_contour.len();
                        if index == *base_index || index == prev || index == next {
                            return None;
                        }
                        let next_2 = (*base_index + 2) % flat_contour.len();
                        Some((other, index != next_2))
                    })
            });

            let from_perimeter = perimeter.iter().enumerate().filter_map(|(i_other, other)| {
                if i_other == perimeter_egde_0
                    || i_other == perimeter_egde_base
                    || i_other == perimeter_egde_1
                {
                    return None;
                }
                let check_base = perimeter_egde_1 != usize::MAX
                    && other.p0.distance2(perimeter[perimeter_egde_1].p1) > 1e-4;
                Some((other, check_base))
            });

            for (other, check_base) in from_flat_contour.chain(from_perimeter) {
                // Check the intersections with the edges of the imaginary flap:
                for (flap_sin, side) in [(flap_sin_0, side_0), (flap_sin_1, side_1)] {
                    let (_, o1, o2) = util_3d::line_line_intersection((other.p0, other.p1), side);
                    if (ZERO..=ONE).contains(&o1) && ZERO <= o2 {
                        minimum_width = minimum_width.min(o2 * flap_sin);
                    }
                }

                // Check the vertices of the other edge.
                // We can skip the vertices shared with the adjacent edges to the base
                // And since the contour is contiguous every point appears twice, one as p0 and another as p1,
                // so we can check just one of them per edge
                if check_base
                    && util_3d::point_line_side(other.p0, side_0)
                    && !util_3d::point_line_side(other.p0, side_1)
                    && util_3d::point_line_side(other.p0, base)
                {
                    let (_seg_0_off, seg_0_dist) = util_3d::point_line_distance(other.p0, base);
                    minimum_width = minimum_width.min(seg_0_dist);
                }
            }
            minimum_width
        };
        let flap_area = |a0: Rad<f32>, a1: Rad<f32>, width: f32| -> f32 {
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

        let mut width = compute_width(a0, a1);
        let mut area = flap_area(a0, a1, width);
        let (mut doing0, mut doing1) = (true, true);
        let step = Rad::from(Deg(5.0));
        while doing0 || doing1 {
            if doing0 {
                let a0x = a0 - step;
                if a0x > Rad(0.0) {
                    let w = compute_width(a0x, a1);
                    let a = flap_area(a0x, a1, w);
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
                    let a = flap_area(a0, a1x, w);
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
            // Invalid flap
            FlapGeom::default()
        } else {
            let tan_0 = a0.cot();
            let tan_1 = a1.cot();
            let base2 = base_len - (tan_0 + tan_1) * width;
            let triangular = base2 <= 0.0;
            if triangular {
                width = base_len / (tan_0 + tan_1);
            }
            FlapGeom {
                tan_0,
                tan_1,
                width,
                triangular,
            }
        }
    }

    pub fn traverse_faces<F>(&self, island: &Island, visit_face: F) -> ControlFlow<()>
    where
        F: FnMut(FaceIndex, &Face, &Matrix3) -> ControlFlow<()>,
    {
        traverse_faces_ex(
            &self.model,
            island.root_face(),
            island.matrix(),
            NormalTraverseFace(self),
            visit_face,
        )
    }
    pub fn traverse_faces_no_matrix<F>(&self, island: &Island, mut visit_face: F) -> ControlFlow<()>
    where
        F: FnMut(FaceIndex) -> ControlFlow<()>,
    {
        traverse_faces_ex(
            &self.model,
            island.root_face(),
            (),
            NoMatrixTraverseFace(&self.edges),
            |i, _, ()| visit_face(i),
        )
    }
    pub fn try_join_strip(&mut self, i_edge: EdgeIndex) -> FxHashMap<IslandKey, JoinResult> {
        let mut renames = FxHashMap::default();
        let mut i_edges = vec![i_edge];
        while let Some(i_edge) = i_edges.pop() {
            // First try to join the edge, if it fails skip.
            let (i_face_a, i_face_b) = match self.model[i_edge].faces() {
                (a, Some(b)) => (a, b),
                // Rims cannot be joined
                _ => continue,
            };
            // Only cuts can be joined
            if !matches!(self.edge_status(i_edge), EdgeStatus::Cut(_)) {
                continue;
            }

            // Compute the number of faces before joining them, a strip is made of quads,
            // and each square has 2 triangles. At least one of them must be a quad.
            let n_faces_a =
                self.island_face_count(self.island_by_key(self.island_by_face(i_face_a)).unwrap());
            let n_faces_b =
                self.island_face_count(self.island_by_key(self.island_by_face(i_face_b)).unwrap());
            if n_faces_a != 2 && n_faces_b != 2 {
                continue;
            }

            let r = self.edge_join(i_edge, None);
            // Join failed?
            if r.is_empty() {
                continue;
            }
            renames.extend(r);

            // Move to the opposite edge of both faces
            for (i_face, n_faces) in [(i_face_a, n_faces_a), (i_face_b, n_faces_b)] {
                // face strips must be made by isolated quads: 4 flat edges and 2 faces
                let edges: Vec<_> = self
                    .get_flat_faces(i_face)
                    .into_iter()
                    .flat_map(|f| self.model[f].vertices_with_edges())
                    .filter(|(_, _, e)| self.edge_status(*e) != EdgeStatus::Hidden)
                    .collect();

                if n_faces != 2 {
                    continue;
                }
                if edges.len() != 4 {
                    continue;
                }
                let Some(&(this_v0, this_v1, _)) = edges.iter().find(|(_, _, i_e)| *i_e == i_edge)
                else {
                    continue;
                };
                // Get the opposite edge, if any
                let Some(&(_, _, opposite)) = edges.iter().find(|&&(i_v0, i_v1, i_e)| {
                    if i_e == i_edge {
                        return false;
                    }
                    if i_v0 == this_v0 || i_v0 == this_v1 || i_v1 == this_v0 || i_v1 == this_v1 {
                        return false;
                    }
                    true
                }) else {
                    continue;
                };
                i_edges.push(opposite);
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

        let mut ordered_islands: Vec<_> = self
            .islands
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
    // Returns the ((face, area), total_area)
    pub fn get_biggest_flat_face(&self, island: &Island) -> (Vec<(FaceIndex, f32)>, f32) {
        let mut biggest_face = None;
        let mut visited = FxHashSet::<FaceIndex>::default();
        let _ = self.traverse_faces_no_matrix(island, |i_face| {
            if !visited.contains(&i_face) {
                let flat_face = self.get_flat_faces(i_face);
                visited.extend(&flat_face);
                // Compute the area of the flat-face
                let with_area: Vec<_> = flat_face
                    .iter()
                    .copied()
                    .map(|i_face| (i_face, self.model().face_area(i_face)))
                    .collect();
                let total_area: f32 = with_area.iter().map(|(_, a)| a).sum();
                if !biggest_face
                    .as_ref()
                    .is_some_and(|(_, prev_area)| *prev_area > total_area)
                {
                    biggest_face = Some((with_area, total_area));
                }
            }
            ControlFlow::Continue(())
        });
        biggest_face.unwrap()
    }

    // Returns the island perimeter in paper size, but, beware! with an arbitrary position
    pub fn island_perimeter(&self, island_key: IslandKey) -> Rc<[FlapEdgeData]> {
        let mut memo = self.memo.island_perimeters.borrow_mut();
        use std::collections::hash_map::Entry::*;
        match memo.entry(island_key) {
            Occupied(o) => Rc::clone(o.get()),
            Vacant(v) => {
                let value = self.island_perimeter_internal(island_key);
                Rc::clone(v.insert(value.into()))
            }
        }
    }

    fn island_perimeter_internal(&self, island_key: IslandKey) -> Vec<FlapEdgeData> {
        let island = self.island_by_key(island_key).unwrap();
        let scale = self.options.scale;

        let mut mxs = FxHashMap::default();
        let _ = self.traverse_faces(island, |i_face, _, mx| {
            mxs.insert(i_face, *mx);
            ControlFlow::Continue(())
        });

        self.island_contour(island_key)
            .into_iter()
            .map(|(i_edge, face_sign)| {
                let edge = &self.model[i_edge];
                let i_face = edge.face_by_sign(face_sign).unwrap();
                let face = &self.model[i_face];
                let (i_v0, i_v1) = face.vertices_of_edge(i_edge).unwrap();
                let plane = self.model.face_plane(face);
                let mx = &mxs[&i_face];
                let tr = |v: VertexIndex| {
                    let v = &self.model[v];
                    let v = plane.project(&v.pos(), scale);
                    mx.transform_point(Point2::from_vec(v)).to_vec()
                };
                FlapEdgeData {
                    i_edge,
                    face_sign,
                    p0: tr(i_v0),
                    p1: tr(i_v1),
                }
            })
            .collect()
    }

    // Given an edge, compute the island perimeter, and the angles that it forms with its adjacent edges
    fn self_collision_perimeter(
        &self,
        i_edge: EdgeIndex,
        i_face_a: FaceIndex,
    ) -> SelfCollisionPerimeter {
        let island_key = self.island_by_face(i_face_a);
        let perimeter = self.island_perimeter(island_key);
        let edge = &self.model[i_edge];
        let face_sign = edge.face_sign(i_face_a);
        let base_on_paper = perimeter
            .iter()
            .position(|e| i_edge == e.i_edge && face_sign == e.face_sign);

        let Some(perimeter_egde_base) = base_on_paper else {
            // should not happen
            return SelfCollisionPerimeter::default();
        };

        let base = &perimeter[perimeter_egde_base];
        let base = (base.p0, base.p1);

        // Get the order of the contour edges
        let Some(base_pos) = perimeter
            .iter()
            .position(|flap| flap.i_edge == i_edge && flap.face_sign == edge.face_sign(i_face_a))
        else {
            // should not happen
            return SelfCollisionPerimeter::default();
        };
        let prev_index = (base_pos + perimeter.len() - 1) % perimeter.len();
        let next_index = (base_pos + 1) % perimeter.len();
        let prev_flap = &perimeter[prev_index];
        let next_flap = &perimeter[next_index];

        let angle_0 = Rad::turn_div_2() - (base.1 - base.0).angle(next_flap.p1 - next_flap.p0);
        let angle_1 = Rad::turn_div_2() - (prev_flap.p1 - prev_flap.p0).angle(base.1 - base.0);

        SelfCollisionPerimeter {
            perimeter,
            perimeter_egde_base,
            angle_0: (angle_0, next_index),
            angle_1: (angle_1, prev_index),
        }
    }

    fn island_contour(&self, i_island: IslandKey) -> OrderedContour {
        let island = self.island_by_key(i_island).unwrap();
        let mut first = None;
        let _ = traverse_faces_ex(
            &self.model,
            island.root_face(),
            (),
            NoMatrixTraverseFace(&self.edges),
            |i_face, face, _| {
                for i_edge in face.index_edges() {
                    if let EdgeStatus::Cut(_) = self.edge_status(i_edge) {
                        first = Some((i_edge, i_face));
                        return ControlFlow::Break(());
                    }
                }
                ControlFlow::Continue(())
            },
        );

        let Some((i_edge, i_face)) = first else {
            return OrderedContour::default();
        };
        self.extend_contour(i_face, i_edge, |i_e| {
            matches!(self.edge_status(i_e), EdgeStatus::Cut(_))
        })
    }

    pub fn flat_face_contour(&self, i_face: FaceIndex) -> OrderedContour {
        let face = &self.model[i_face];
        let first = face
            .index_edges()
            .into_iter()
            .find(|&i_e| self.edge_status(i_e) != EdgeStatus::Hidden);

        let Some(i_edge) = first else {
            return OrderedContour::default();
        };
        self.extend_contour(i_face, i_edge, |i_e| {
            self.edge_status(i_e) != EdgeStatus::Hidden
        })
    }

    fn extend_contour(
        &self,
        mut i_face: FaceIndex,
        mut i_edge: EdgeIndex,
        is_contour: impl Fn(EdgeIndex) -> bool,
    ) -> OrderedContour {
        let mut contour = Vec::new();
        let first_edge = (i_edge, self.model[i_edge].face_sign(i_face));
        loop {
            let face = &self.model[i_face];
            let (i_edge_next, _) = face.next_edge(i_edge);
            i_edge = i_edge_next;
            let edge = &self.model[i_edge];
            if is_contour(i_edge) {
                let next_edge = (i_edge, edge.face_sign(i_face));
                contour.push((next_edge.0, next_edge.1));
                if next_edge == first_edge {
                    break;
                }
            } else {
                let faces = edge.faces();
                i_face = if faces.0 == i_face {
                    match faces.1 {
                        Some(f) => f,
                        None => {
                            // Should not happen!
                            log::error!("Broken contour!");
                            break;
                        }
                    }
                } else if Some(i_face) == faces.1 {
                    faces.0
                } else {
                    // Should not happen!
                    log::error!("Broken contour!");
                    break;
                }
            }
        }
        contour
    }
}

struct SelfCollisionPerimeter {
    perimeter: Rc<[FlapEdgeData]>,
    perimeter_egde_base: usize,
    angle_0: (Rad<f32>, usize),
    angle_1: (Rad<f32>, usize),
}

impl Default for SelfCollisionPerimeter {
    fn default() -> Self {
        SelfCollisionPerimeter {
            perimeter: Vec::new().into(),
            perimeter_egde_base: usize::MAX,
            angle_0: (Rad::turn_div_2(), usize::MAX),
            angle_1: (Rad::turn_div_2(), usize::MAX),
        }
    }
}

pub fn traverse_faces_ex<F, TP>(
    model: &Model,
    root: FaceIndex,
    initial_state: TP::State,
    mut policy: TP,
    mut visit_face: F,
) -> ControlFlow<()>
where
    F: FnMut(FaceIndex, &Face, &TP::State) -> ControlFlow<()>,
    TP: TraverseFacePolicy,
{
    let mut visited_faces = FxHashSet::default();
    let mut stack = vec![(root, root, initial_state)];
    visited_faces.insert(root);

    while let Some((i_parent, i_face, m)) = stack.pop() {
        let face = &model[i_face];
        visit_face(i_face, face, &m)?;
        for i_edge in face.index_edges() {
            if !policy.cross_edge(i_edge) {
                continue;
            }
            let edge = &model[i_edge];
            let (fa, fb) = edge.faces();
            for i_next_face in std::iter::once(fa).chain(fb) {
                if i_next_face == i_parent || i_next_face == i_face {
                    continue;
                }
                if visited_faces.insert(i_next_face) {
                    let next_state = policy.next_state(&m, edge, face, i_next_face);
                    stack.push((i_face, i_next_face, next_state));
                } else {
                    policy.duplicated_face(i_face, i_edge, i_next_face);
                }
            }
        }
    }
    ControlFlow::Continue(())
}

pub trait TraverseFacePolicy {
    type State: Copy;
    fn cross_edge(&self, i_edge: EdgeIndex) -> bool;
    fn duplicated_face(&mut self, _i_face: FaceIndex, _i_edge: EdgeIndex, _i_next_face: FaceIndex) {
    }
    fn next_state(
        &self,
        st: &Self::State,
        _edge: &Edge,
        _face: &Face,
        _i_next_face: FaceIndex,
    ) -> Self::State {
        *st
    }
}

struct NormalTraverseFace<'a>(&'a Papercraft);

impl TraverseFacePolicy for NormalTraverseFace<'_> {
    type State = Matrix3;

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.0.edges[usize::from(i_edge)] {
            EdgeStatus::Cut(_) => false,
            EdgeStatus::Joined | EdgeStatus::Hidden => true,
        }
    }
    fn next_state(
        &self,
        st: &Self::State,
        edge: &Edge,
        face: &Face,
        i_next_face: FaceIndex,
    ) -> Self::State {
        let next_face = &self.0.model[i_next_face];
        let medge = self.0.face_to_face_edge_matrix(edge, face, next_face);
        st * medge
    }
}

struct NoMatrixTraverseFace<'a>(&'a [EdgeStatus]);

impl TraverseFacePolicy for NoMatrixTraverseFace<'_> {
    type State = ();

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.0[usize::from(i_edge)] {
            EdgeStatus::Cut(_) => false,
            EdgeStatus::Joined | EdgeStatus::Hidden => true,
        }
    }
}

struct FlatTraverseFace<'a>(&'a Papercraft);

impl TraverseFacePolicy for FlatTraverseFace<'_> {
    type State = ();

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.0.edge_status(i_edge) {
            EdgeStatus::Joined | EdgeStatus::Cut(_) => false,
            EdgeStatus::Hidden => true,
        }
    }
}

struct FlatTraverseFaceWithMatrix<'a>(&'a Papercraft);

impl TraverseFacePolicy for FlatTraverseFaceWithMatrix<'_> {
    type State = Matrix3;

    fn cross_edge(&self, i_edge: EdgeIndex) -> bool {
        match self.0.edge_status(i_edge) {
            EdgeStatus::Joined | EdgeStatus::Cut(_) => false,
            EdgeStatus::Hidden => true,
        }
    }

    fn next_state(
        &self,
        st: &Self::State,
        edge: &Edge,
        face: &Face,
        i_next_face: FaceIndex,
    ) -> Self::State {
        let next_face = &self.0.model[i_next_face];
        let medge = self.0.face_to_face_edge_matrix(edge, face, next_face);
        st * medge
    }
}

pub struct BodyTraverse;

impl TraverseFacePolicy for BodyTraverse {
    type State = ();

    fn cross_edge(&self, _i_edge: EdgeIndex) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct Island {
    root: FaceIndex,

    rot: Rad<f32>,
    loc: Vector2,
    mx: Matrix3,
    name: String,
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
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Serialize for EdgeStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let is = match self {
            EdgeStatus::Hidden => 0,
            EdgeStatus::Joined => 1,
            EdgeStatus::Cut(FlapSide::False) => 2,
            EdgeStatus::Cut(FlapSide::True) => 3,
            EdgeStatus::Cut(FlapSide::Hidden) => 4,
        };
        serializer.serialize_i32(is)
    }
}
impl<'de> Deserialize<'de> for EdgeStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let d = u32::deserialize(deserializer)?;
        let res = match d {
            0 => EdgeStatus::Hidden,
            1 => EdgeStatus::Joined,
            2 => EdgeStatus::Cut(FlapSide::False),
            3 => EdgeStatus::Cut(FlapSide::True),
            4 => EdgeStatus::Cut(FlapSide::Hidden),
            _ => return Err(serde::de::Error::missing_field("invalid edge status")),
        };
        Ok(res)
    }
}

impl Serialize for FlapStyle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let is = match self {
            FlapStyle::Textured => 0,
            FlapStyle::HalfTextured => 1,
            FlapStyle::White => 2,
            FlapStyle::None => 3,
        };
        serializer.serialize_i32(is)
    }
}
impl<'de> Deserialize<'de> for FlapStyle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let d = u32::deserialize(deserializer)?;
        let res = match d {
            0 => FlapStyle::Textured,
            1 => FlapStyle::HalfTextured,
            2 => FlapStyle::White,
            3 => FlapStyle::None,
            _ => return Err(serde::de::Error::missing_field("invalid tab_style value")),
        };
        Ok(res)
    }
}

impl Serialize for FoldStyle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
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
    where
        D: serde::Deserializer<'de>,
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

impl Serialize for EdgeIdPosition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let is = match self {
            EdgeIdPosition::None => 0,
            EdgeIdPosition::Outside => 1,
            EdgeIdPosition::Inside => -1,
        };
        serializer.serialize_i32(is)
    }
}
impl<'de> Deserialize<'de> for EdgeIdPosition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let d = i32::deserialize(deserializer)?;
        let res = match d {
            1 => EdgeIdPosition::Outside,
            -1 => EdgeIdPosition::Inside,
            _ => {
                return Err(serde::de::Error::missing_field(
                    "invalid edge_id_position value",
                ));
            }
        };
        Ok(res)
    }
}

impl Serialize for Island {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
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
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Def {
            root: usize,
            x: f32,
            y: f32,
            r: f32,
        }
        let d = Def::deserialize(deserializer)?;
        let mut island = Island {
            root: FaceIndex::from(d.root),
            loc: Vector2::new(d.x, d.y),
            rot: Rad(d.r),
            mx: Matrix3::one(),
            name: String::new(),
        };
        island.recompute_matrix();
        Ok(island)
    }
}

impl Serialize for LineConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_struct("line", 4)?;
        map.serialize_field("thick", &self.thick)?;
        map.serialize_field("r", &self.color.r)?;
        map.serialize_field("g", &self.color.g)?;
        map.serialize_field("b", &self.color.b)?;
        map.serialize_field("a", &self.color.a)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for LineConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Def {
            thick: f32,
            r: f32,
            g: f32,
            b: f32,
            a: f32,
        }
        let d = Def::deserialize(deserializer)?;
        Ok(LineConfig {
            thick: d.thick,
            color: Color::new(d.r, d.g, d.b, d.a),
        })
    }
}

impl Serialize for MyColor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_struct("color", 4)?;
        map.serialize_field("r", &self.0.r)?;
        map.serialize_field("g", &self.0.g)?;
        map.serialize_field("b", &self.0.b)?;
        map.serialize_field("a", &self.0.a)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for MyColor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Def {
            r: f32,
            g: f32,
            b: f32,
            a: f32,
        }
        let d = Def::deserialize(deserializer)?;
        Ok(MyColor(Color::new(d.r, d.g, d.b, d.a)))
    }
}

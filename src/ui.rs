use std::{collections::HashMap, ops::Range};
/* Everything in this crate is public so that it can be freely used from main.rs */
use std::ops::ControlFlow;

use cgmath::{prelude::*, Deg, Rad};
use easy_imgui_window::easy_imgui_renderer::{
    glow::{self, HasContext},
    glr::GlContext,
};
use fxhash::{FxHashMap, FxHashSet};
use image::DynamicImage;

use crate::paper::{
    EdgeId, EdgeIdPosition, EdgeIndex, EdgeStatus, EdgeToggleFlapAction, Face, FaceIndex, FlapGeom,
    FlapSide, FlapStyle, FoldStyle, IslandKey, JoinResult, MaterialIndex, Model, PaperOptions,
    Papercraft,
};
use crate::util_3d::{
    self, Matrix2, Matrix3, Matrix4, Point2, Point3, Quaternion, Vector2, Vector3,
};
use crate::util_gl::{
    MLine3DStatus, MStatus, MVertex2D, MVertex2DColor, MVertex2DLine, MVertex3D, MVertex3DLine,
    MVertexText, MLINE3D_CUT, MLINE3D_HIDDEN, MLINE3D_NORMAL, MSTATUS_HI, MSTATUS_SEL,
    MSTATUS_UNSEL,
};
use crate::{
    glr::{self, Rgba},
    PrintableText, TextAlign,
};
use crate::{TextBuilder, FONT_SIZE};
use tr::tr;

// In millimeters, these are not configurable, but they should be cut out, so they should not be visible anyways
const FLAP_LINE_WIDTH: f32 = 0.2;
const BORDER_LINE_WIDTH: f32 = 0.1;

// In pixels
const LINE_SEL_WIDTH: f32 = 5.0;

pub struct GLObjects {
    pub textures: Option<glr::Texture>,

    //GL objects that are rebuilt with the model

    // Attributes common for 3D/2D
    pub vertices: glr::DynamicVertexArray<MVertex3D>,
    pub vertices_sel: glr::DynamicVertexArray<MStatus>,

    // For 3D
    pub scene_vertices_edge: glr::DynamicVertexArray<MVertex3DLine>,
    pub scene_vertices_edge_status: glr::DynamicVertexArray<MLine3DStatus>,
    // Maps an edge_index into the scene_vertices_edge_status
    pub edge_map: FxHashMap<EdgeIndex, usize>,

    // For 2D
    pub paper_vertices: glr::DynamicVertexArray<MVertex2D>,
    pub paper_vertices_edge_cut: glr::DynamicVertexArray<MVertex2DLine>,
    pub paper_vertices_edge_crease: glr::DynamicVertexArray<MVertex2DLine>,
    pub paper_vertices_flap: glr::DynamicVertexArray<MVertex2DColor>,
    pub paper_vertices_flap_edge: glr::DynamicVertexArray<MVertex2DLine>,
    pub paper_vertices_edge_sel: glr::DynamicVertexArray<MVertex2DLine>,
    pub paper_vertices_shadow_flap: glr::DynamicVertexArray<MVertex2DColor>,
    pub paper_text: glr::DynamicVertexArray<MVertexText>,

    // 2D background
    pub paper_vertices_page: glr::DynamicVertexArray<MVertex2DColor>,
    pub paper_vertices_margin: glr::DynamicVertexArray<MVertex2DLine>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MouseMode {
    Face,
    Edge,
    Flap,
    ReadOnly,
}

pub fn color_edge(mode: MouseMode) -> Rgba {
    match mode {
        MouseMode::Edge => Rgba::new(0.5, 0.5, 1.0, 1.0),
        MouseMode::Flap => Rgba::new(0.0, 0.5, 0.0, 1.0),
        MouseMode::Face | MouseMode::ReadOnly => Rgba::new(0.5, 0.5, 0.5, 1.0),
    }
}

//UndoItem cannot store IslandKey, because they are dynamic, use the root of the island instead
#[derive(Debug)]
pub enum UndoAction {
    IslandMove {
        i_root: FaceIndex,
        prev_rot: Rad<f32>,
        prev_loc: Vector2,
    },
    FlapToggle {
        i_edge: EdgeIndex,
        flap_side: FlapSide,
    },
    EdgeCut {
        i_edge: EdgeIndex,
    },
    EdgeJoin {
        join_result: JoinResult,
    },
    DocConfig {
        options: PaperOptions,
        island_pos: FxHashMap<FaceIndex, (Rad<f32>, Vector2)>,
    },
    Modified,
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Debug)]
    pub struct RebuildFlags: u32 {
        const PAGES = 0x0001;
        const PAPER = 0x0002;
        const SCENE_EDGE = 0x0004;
        const SELECTION = 0x0008;
        const PAPER_REDRAW = 0x0010;
        const SCENE_REDRAW = 0x0020;
        const ISLANDS = 0x0040;

        const ANY_REDRAW_PAPER = Self::PAGES.bits() | Self::PAPER.bits() | Self::SELECTION.bits() | Self::PAPER_REDRAW.bits() | Self::ISLANDS.bits();
        const ANY_REDRAW_SCENE = Self::SCENE_EDGE.bits() | Self::SELECTION.bits() | Self::SCENE_REDRAW.bits();
    }
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Debug)]
    pub struct SetSelectionFlags: u32 {
        // CLICKED means not hovering
        const CLICKED = 0x0001;
        // RELEASE should be combined with CLICKED
        const RELEASED = 0x0002;
        const ADD_TO_SEL = 0x0004;
        const ALT_PRESSED = 0x0008;
    }
}

//Objects that are recreated when a new model is loaded
pub struct PapercraftContext {
    // The model
    papercraft: Papercraft,
    gl_objs: GLObjects,

    undo_stack: Vec<Vec<UndoAction>>,
    pub modified: bool,

    // State
    selected_face: Option<FaceIndex>,
    selected_edges: Option<FxHashSet<EdgeIndex>>,
    selected_islands: Vec<IslandKey>,
    // Contains the UndoActions if these islands are to be moved, the actual grabbed islands are selected_islands
    grabbed_island: Option<Vec<UndoAction>>,
    last_cursor_pos: Vector2,
    rotation_center: Option<Vector2>,
    // The selection rectangle, while dragging it, plus the pre-selected islands.
    // The boolean says if the selection is to be added or removed
    pre_selection: Option<(Vector2, Vector2, FxHashSet<IslandKey>, bool)>,
    // The island that has just been selected
    just_selected: Option<IslandKey>,

    pub ui: UiSettings,
}

#[derive(Clone)]
pub struct UiSettings {
    pub mode: MouseMode,
    pub trans_scene: Transformation3D,
    pub trans_paper: TransformationPaper,

    // These shouldn't really be here but in main.rs
    pub show_textures: bool,
    pub show_flaps: bool,
    pub show_3d_lines: bool,
    pub xray_selection: bool,
    pub show_texts: bool,
    pub highlight_overlaps: bool,
    pub draw_paper: bool,
}

#[derive(Clone)]
pub struct Transformation3D {
    pub location: Vector3,
    pub rotation: Quaternion,
    pub scale: f32,

    pub obj: Matrix4,
    pub persp: Matrix4,
    pub persp_inv: Matrix4,
    pub view: Matrix4,
    pub view_inv: Matrix4,
    pub mnormal: Matrix3,
}

impl Transformation3D {
    fn new(
        obj: Matrix4,
        location: Vector3,
        rotation: Quaternion,
        scale: f32,
        persp: Matrix4,
    ) -> Transformation3D {
        let mut tr = Transformation3D {
            location,
            rotation,
            scale,
            obj,
            persp,
            persp_inv: persp.invert().unwrap(),
            view: Matrix4::one(),
            view_inv: Matrix4::one(),
            mnormal: Matrix3::one(),
        };
        tr.recompute_obj();
        tr
    }
    pub fn recompute_obj(&mut self) {
        let r = Matrix3::from(self.rotation);
        let t = Matrix4::from_translation(self.location);
        let s = Matrix4::from_scale(self.scale);

        self.view = t * Matrix4::from(r) * s * self.obj;
        self.view_inv = self.view.invert().unwrap();
        self.mnormal = r; //should be inverse of transpose
    }

    pub fn set_ratio(&mut self, ratio: f32) {
        let f = self.persp[1][1];
        self.persp[0][0] = f / ratio;
        self.persp_inv = self.persp.invert().unwrap();
    }
}

#[derive(Clone)]
pub struct TransformationPaper {
    pub ortho: Matrix3,
    pub mx: Matrix3,
}

impl TransformationPaper {
    /// Converts coordinates from window to canvas
    pub fn paper_click(&self, size: Vector2, pos: Vector2) -> Vector2 {
        let x = (pos.x / size.x) * 2.0 - 1.0;
        let y = -((pos.y / size.y) * 2.0 - 1.0);
        let click = Point2::new(x, y);

        let mx = self.ortho * self.mx;
        let mx_inv = mx.invert().unwrap();
        mx_inv.transform_point(click).to_vec()
    }
    /// Converts coordinates from canvas to window
    pub fn paper_unclick(&self, size: Vector2, pos: Vector2) -> Vector2 {
        let mx = self.ortho * self.mx;
        let pos = mx.transform_point(Point2::from_vec(pos)).to_vec();

        let x = (pos.x + 1.0) / 2.0 * size.x;
        let y = (-pos.y + 1.0) / 2.0 * size.y;
        Vector2::new(x, y)
    }
}

fn default_transformations(
    obj: Matrix4,
    sz_scene: Vector2,
    sz_paper: Vector2,
    ops: &PaperOptions,
) -> (Transformation3D, TransformationPaper) {
    let page = Vector2::from(ops.page_size);
    let persp = cgmath::perspective(Deg(60.0), 1.0, 1.0, 100.0);
    let mut trans_scene = Transformation3D::new(
        obj,
        Vector3::new(0.0, 0.0, -30.0),
        Quaternion::one(),
        20.0,
        persp,
    );
    let ratio = sz_scene.x / sz_scene.y;
    trans_scene.set_ratio(ratio);

    let trans_paper = {
        let mt = Matrix3::from_translation(Vector2::new(-page.x / 2.0, -page.y / 2.0));
        let ms = Matrix3::from_scale(1.0);
        let ortho = util_3d::ortho2d(sz_paper.x, sz_paper.y);
        TransformationPaper { ortho, mx: ms * mt }
    };
    (trans_scene, trans_paper)
}

unsafe fn set_texture_filter(gl: &GlContext, tex_filter: bool) {
    if tex_filter {
        gl.tex_parameter_i32(
            glow::TEXTURE_2D_ARRAY,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR_MIPMAP_LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D_ARRAY,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
    } else {
        gl.tex_parameter_i32(
            glow::TEXTURE_2D_ARRAY,
            glow::TEXTURE_MIN_FILTER,
            glow::NEAREST as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D_ARRAY,
            glow::TEXTURE_MAG_FILTER,
            glow::NEAREST as i32,
        );
    }
}

#[derive(Debug)]
pub enum ClickResult {
    None,
    Face(FaceIndex),
    Edge(EdgeIndex, Option<FaceIndex>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EdgeDrawKind {
    Mountain,
    Valley,
}

pub struct PaperDrawFaceArgs {
    vertices: Vec<MVertex2D>,
    vertices_edge_cut: Vec<Line2D>,
    vertices_edge_crease: Vec<Line2D>,
    vertices_flap: Vec<MVertex2DColor>,
    vertices_flap_edge: Vec<Line2D>,
    vertices_shadow_flap: Vec<MVertex2DColor>,
    vertices_text: Vec<MVertexText>,
}

// Complements PaperDrawFaceArgs for printable operations
#[derive(Default)]
pub struct PaperDrawFaceArgsExtra {
    // For each line in vertices_edge_crease says which kind of line
    crease_kind: Vec<EdgeDrawKind>,
    cut_index: Vec<CutIndex>,
}

impl PaperDrawFaceArgs {
    fn new(model: &Model) -> PaperDrawFaceArgs {
        PaperDrawFaceArgs {
            vertices: vec![MVertex2D::default(); 3 * model.num_faces()],
            vertices_edge_cut: Vec::new(),
            vertices_edge_crease: Vec::new(),
            vertices_flap: Vec::new(),
            vertices_flap_edge: Vec::new(),
            vertices_shadow_flap: Vec::new(),
            vertices_text: Vec::new(),
        }
    }

    pub fn iter_cut(&self) -> impl Iterator<Item = (Vector2, Vector2)> + '_ {
        self.iter_cut_ex((0, 0)..self.cut_last_index())
    }
    pub fn cut_last_index(&self) -> (usize, usize) {
        (self.vertices_flap_edge.len(), self.vertices_edge_cut.len())
    }
    pub fn iter_cut_ex(
        &self,
        range: Range<(usize, usize)>,
    ) -> impl Iterator<Item = (Vector2, Vector2)> + '_ {
        self.vertices_flap_edge[range.start.0..range.end.0]
            .iter()
            .chain(&self.vertices_edge_cut[range.start.1..range.end.1])
            .map(|s| (s.p0, s.p1))
    }
    pub fn iter_crease<'a>(
        &'a self,
        extra: &'a PaperDrawFaceArgsExtra,
        kind: EdgeDrawKind,
    ) -> impl Iterator<Item = (Vector2, Vector2)> + 'a {
        self.vertices_edge_crease
            .iter()
            .zip(extra.crease_kind.iter())
            .filter_map(move |(line, ek)| (*ek == kind).then_some(line))
            .map(|s| (s.p0, s.p1))
    }
    pub fn vertices_for_face(&self, i_face: FaceIndex) -> [Vector2; 3] {
        let i0 = 3 * usize::from(i_face);
        [
            self.vertices[i0].pos_2d,
            self.vertices[i0 + 1].pos_2d,
            self.vertices[i0 + 2].pos_2d,
        ]
    }
}

impl PaperDrawFaceArgsExtra {
    pub fn cut_indices(&self) -> &[CutIndex] {
        &self.cut_index
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CutIndex {
    pub center: Vector2,
    pub dir: Vector2,
    pub angle: Rad<f32>,
    pub i_face_b: FaceIndex,
    pub id: EdgeId,
    pub voffs: f32,
}

impl CutIndex {
    /// This struct stores the position and content of an edge-id
    /// (a, b): coordinates of the edge on paper
    /// n_flap: if there is a flap, a vector normal to the edge with the length of the flap width
    /// i_face_b: the face index of the _other _ face (_this_ face is not needed)
    /// id: the edge id
    /// options: the PaperOptions
    fn new(
        a: Vector2,
        b: Vector2,
        n_flap: Option<Vector2>,
        i_face_b: FaceIndex,
        id: EdgeId,
        options: &PaperOptions,
    ) -> CutIndex {
        let mut center = (a + b) / 2.0;

        // Where does the edge-id go?
        let voffs = match (options.edge_id_position, n_flap) {
            // inside the face
            (EdgeIdPosition::Inside, None) => -0.2,
            // in the flap, next to the face
            (EdgeIdPosition::Inside, Some(_)) => 1.0,
            // outside the face
            (EdgeIdPosition::Outside, None) => 1.0,
            // outside the flap
            (EdgeIdPosition::Outside, Some(n)) => {
                center += n;
                1.0
            }
            // should not happen, if pos is None it is filtered before getting here
            (EdgeIdPosition::None, _) => 0.0,
        };
        let dir = (b - a).normalize();
        //let normal = Vector2::new(-dir.y, dir.x);
        //let pos = center + normal * options.edge_id_font_size * factor;
        let angle = -dir.angle(Vector2::new(1.0, 0.0));
        CutIndex {
            center,
            dir,
            angle,
            i_face_b,
            id,
            voffs,
        }
    }
    pub fn pos(&self, font_size: f32) -> Vector2 {
        let normal = Vector2::new(-self.dir.y, self.dir.x);
        self.center + font_size * self.voffs * normal
    }
}

pub enum UndoResult {
    False,
    Model,
    ModelAndOptions(PaperOptions),
}

pub struct Rectangle {
    pub a: Vector2,
    pub b: Vector2,
}

impl Rectangle {
    pub fn new(mut a: Vector2, mut b: Vector2) -> Rectangle {
        if a.x > b.x {
            std::mem::swap(&mut a.x, &mut b.x);
        }
        if a.y > b.y {
            std::mem::swap(&mut a.y, &mut b.y);
        }
        Rectangle { a, b }
    }

    pub fn is_null(&self) -> bool {
        self.a == self.b
    }

    pub fn contains(&self, p: cgmath::Vector2<f32>) -> bool {
        (self.a.x..self.b.x).contains(&p.x) && (self.a.y..self.b.y).contains(&p.y)
    }
}

impl PapercraftContext {
    pub fn papercraft(&self) -> &Papercraft {
        &self.papercraft
    }
    pub fn gl_objs(&self) -> &GLObjects {
        &self.gl_objs
    }
    pub fn set_papercraft_options(&mut self, options: PaperOptions, push_undo_action: bool) {
        let island_pos = push_undo_action.then(|| {
            self.papercraft()
                .islands()
                .map(|(_, island)| (island.root_face(), (island.rotation(), island.location())))
                .collect()
        });

        self.ui.show_textures = options.texture;
        if let Some(tex) = &self.gl_objs.textures {
            unsafe {
                tex.gl().active_texture(glow::TEXTURE0);
                tex.gl()
                    .bind_texture(glow::TEXTURE_2D_ARRAY, Some(tex.id()));
                set_texture_filter(tex.gl(), options.tex_filter);
            }
        }
        let old_options = self
            .papercraft
            .set_options(options, /* relocate_pieces */ push_undo_action);

        if let Some(island_pos) = island_pos {
            self.push_undo_action(vec![UndoAction::DocConfig {
                options: old_options,
                island_pos,
            }]);
        }
    }
    pub fn from_papercraft(
        papercraft: Papercraft,
        gl: &GlContext,
    ) -> anyhow::Result<PapercraftContext> {
        // Compute the bounding box, then move to the center and scale to a standard size
        let (v_min, v_max) =
            util_3d::bounding_box_3d(papercraft.model().vertices().map(|(_, v)| v.pos()));
        let size = (v_max.x - v_min.x)
            .max(v_max.y - v_min.y)
            .max(v_max.z - v_min.z);
        let mscale = Matrix4::from_scale(if size == 0.0 { 1.0 } else { 1.0 / size });
        let center = (v_min + v_max) / 2.0;
        let mcenter = Matrix4::from_translation(-center);
        let obj = mscale * mcenter;

        let sz_dummy = Vector2::new(1.0, 1.0);
        let (trans_scene, trans_paper) =
            default_transformations(obj, sz_dummy, sz_dummy, papercraft.options());
        let show_textures = papercraft.options().texture;
        let gl_objs = GLObjects::new(&papercraft, gl)?;

        Ok(PapercraftContext {
            papercraft,
            undo_stack: Vec::new(),
            modified: false,
            gl_objs,
            selected_face: None,
            selected_edges: None,
            selected_islands: Vec::new(),
            grabbed_island: None,
            last_cursor_pos: Vector2::zero(),
            rotation_center: None,
            pre_selection: None,
            just_selected: None,
            ui: UiSettings {
                mode: MouseMode::Face,
                trans_scene,
                trans_paper,
                show_textures,
                show_flaps: true,
                show_3d_lines: true,
                xray_selection: true,
                show_texts: false,
                highlight_overlaps: false,
                draw_paper: true,
            },
        })
    }

    pub fn pre_render(&mut self, rebuild: RebuildFlags, text_builder: &impl TextBuilder) {
        if rebuild.contains(RebuildFlags::ISLANDS) && self.ui.show_texts {
            self.papercraft.rebuild_island_names();
        }
        if rebuild.contains(RebuildFlags::PAGES) {
            self.pages_rebuild();
        }
        if rebuild.contains(RebuildFlags::PAPER) {
            self.paper_rebuild(text_builder);
        }
        if rebuild.contains(RebuildFlags::SELECTION) {
            self.selection_rebuild();
        }
        if rebuild.intersects(RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION) {
            self.scene_edge_rebuild();
        }
    }

    pub fn reset_views(&mut self, sz_scene: Vector2, sz_paper: Vector2) {
        (self.ui.trans_scene, self.ui.trans_paper) = default_transformations(
            self.ui.trans_scene.obj,
            sz_scene,
            sz_paper,
            self.papercraft.options(),
        );
    }

    fn paper_draw_face(
        &self,
        face: &Face,
        i_face: FaceIndex,
        m: &Matrix3,
        args: &mut PaperDrawFaceArgs,
        mut flap_cache: Option<&mut Vec<(FaceIndex, FlapVertices)>>,
        mut extra: Option<&mut PaperDrawFaceArgsExtra>,
    ) {
        let i0 = 3 * usize::from(i_face);
        let options = self.papercraft.options();
        let scale = options.scale;
        let flap_style = options.flap_style;
        let fold_line_width = options.fold_line_width;

        for (i, i_v) in face.index_vertices().into_iter().enumerate() {
            let v = &self.papercraft.model()[i_v];
            let p = self
                .papercraft
                .model()
                .face_plane(face)
                .project(&v.pos(), scale);
            let pos_2d = m.transform_point(Point2::from_vec(p)).to_vec();

            args.vertices[i0 + i] = MVertex2D { pos_2d };
        }

        for (i_v0, i_v1, i_edge) in face.vertices_with_edges() {
            let edge = &self.papercraft.model()[i_edge];
            let edge_status = self.papercraft.edge_status(i_edge);
            let edge_id = self.papercraft.edge_id(i_edge);

            // `draw_flap`` references the adjacent face, and tells if it has to be drawn
            #[derive(Copy, Clone)]
            enum DrawFlap {
                Visible(Option<FaceIndex>),
                Invisible(Option<FaceIndex>),
            }
            impl DrawFlap {
                fn is_visible(&self) -> bool {
                    matches!(self, DrawFlap::Visible(_))
                }
                fn face(self) -> Option<FaceIndex> {
                    match self {
                        DrawFlap::Visible(x) | DrawFlap::Invisible(x) => x,
                    }
                }
            }
            let draw_flap = match edge_status {
                EdgeStatus::Hidden => {
                    // hidden edges are never drawn
                    continue;
                }
                EdgeStatus::Cut(c) => {
                    // cut edges are always drawn, the flap depends on the value of c and the face_sign
                    let maybe_i_face_b = match edge.faces() {
                        (fa, Some(fb)) if i_face == fb => Some(fa),
                        (fa, Some(fb)) if i_face == fa => Some(fb),
                        // Rim edge, no face b
                        (_, None) => None,
                        // should not happen
                        _ => continue,
                    };
                    if flap_style == FlapStyle::None {
                        // User doesn't want flaps
                        DrawFlap::Invisible(maybe_i_face_b)
                    } else if !c.flap_visible(edge.face_sign(i_face)) {
                        // The flap is in the other face
                        DrawFlap::Invisible(maybe_i_face_b)
                    } else {
                        DrawFlap::Visible(maybe_i_face_b)
                    }
                }
                EdgeStatus::Joined => {
                    // joined edges are drawn from one side only, no matter which one
                    if !edge.face_sign(i_face) {
                        continue;
                    }
                    // but never with a flap or a flap-id
                    DrawFlap::Invisible(None)
                }
            };

            let plane = self.papercraft.model().face_plane(face);
            let v0 = &self.papercraft.model()[i_v0];
            let p0 = plane.project(&v0.pos(), scale);
            let pos0 = m.transform_point(Point2::from_vec(p0)).to_vec();

            let v1 = &self.papercraft.model()[i_v1];
            let p1 = plane.project(&v1.pos(), scale);
            let pos1 = m.transform_point(Point2::from_vec(p1)).to_vec();

            //Dotted lines are drawn for negative 3d angles (valleys) if the edge is joined or
            //cut with a flap
            let crease_kind = if edge_status == EdgeStatus::Joined || draw_flap.is_visible() {
                let angle_3d = edge.angle();
                if edge_status == EdgeStatus::Joined
                    && Rad(angle_3d.0.abs()) < Rad::from(Deg(options.hidden_line_angle))
                {
                    continue;
                }
                let kind = if angle_3d.0.is_sign_negative() {
                    EdgeDrawKind::Valley
                } else {
                    EdgeDrawKind::Mountain
                };
                Some(kind)
            } else {
                None
            };

            let v = pos1 - pos0;
            let (width_left, width_right) = if edge_status == EdgeStatus::Joined {
                (fold_line_width / 2.0, fold_line_width / 2.0)
            } else if draw_flap.is_visible() {
                (fold_line_width, 0.0)
            } else {
                (BORDER_LINE_WIDTH, 0.0)
            };

            let v_len = v.magnitude();

            let fold_factor = options.fold_line_len / v_len;
            if let Some(crease_kind) = crease_kind {
                let visible_line = match options.fold_style {
                    FoldStyle::Full => (Some(0.0), None),
                    FoldStyle::FullAndOut => (Some(fold_factor), None),
                    FoldStyle::Out => (Some(fold_factor), Some(0.0)),
                    FoldStyle::In => (Some(0.0), Some(fold_factor)),
                    FoldStyle::InAndOut => (Some(fold_factor), Some(fold_factor)),
                    FoldStyle::None => (None, None),
                };
                match visible_line {
                    (None, None) | (None, Some(_)) => {}
                    (Some(f), None) => {
                        let vn = v * f;
                        args.vertices_edge_crease.push(Line2D {
                            p0: pos0 - vn,
                            p1: pos1 + vn,
                            dash0: 0.0,
                            dash1: if crease_kind == EdgeDrawKind::Valley {
                                v_len * (1.0 + 2.0 * f)
                            } else {
                                0.0
                            },
                            width_left,
                            width_right,
                        });
                        if let Some(extra) = extra.as_mut() {
                            extra.crease_kind.push(crease_kind);
                        }
                    }
                    (Some(f_a), Some(f_b)) => {
                        let vn_a = v * f_a;
                        let vn_b = v * f_b;
                        let line_a = Line2D {
                            p0: pos0 - vn_a,
                            p1: pos0 + vn_b,
                            dash0: 0.0,
                            dash1: if crease_kind == EdgeDrawKind::Valley {
                                v_len * (f_a + f_b)
                            } else {
                                0.0
                            },
                            width_left,
                            width_right,
                        };
                        let line_b = Line2D {
                            p0: pos1 - vn_b,
                            p1: pos1 + vn_a,
                            dash0: 0.0,
                            dash1: line_a.dash1,
                            width_left,
                            width_right,
                        };
                        args.vertices_edge_crease.push(line_a);
                        args.vertices_edge_crease.push(line_b);
                        // two lines
                        if let Some(extra) = extra.as_mut() {
                            extra
                                .crease_kind
                                .extend_from_slice(&[crease_kind, crease_kind]);
                        }
                    }
                };
            } else {
                // Weird cuts are drawn differently:
                // * cuts with hidden flaps
                // * rims
                let line_dash = if edge_status == EdgeStatus::Cut(FlapSide::Hidden) {
                    5.0
                } else {
                    0.0
                };
                let mut line = Line2D {
                    p0: pos0,
                    p1: pos1,
                    dash0: 0.0,
                    dash1: 0.0,
                    width_left,
                    width_right,
                };
                line.set_dash(line_dash);
                args.vertices_edge_cut.push(line);

                if let (Some(extra), Some(edge_id), Some(i_face_b)) =
                    (extra.as_mut(), edge_id, draw_flap.face())
                {
                    extra
                        .cut_index
                        .push(CutIndex::new(pos0, pos1, None, i_face_b, edge_id, options));
                }
            }

            // Draw the flap?
            if let DrawFlap::Visible(maybe_i_face_b) = draw_flap {
                let flap_geom =
                    self.papercraft
                        .flat_face_flap_dimensions(i_face, maybe_i_face_b, i_edge);
                let FlapGeom {
                    tan_0,
                    tan_1,
                    width,
                    triangular,
                } = flap_geom;

                let vn = v * (width / v_len);
                let v_0 = vn * tan_0;
                let v_1 = vn * tan_1;
                let n = Vector2::new(-vn.y, vn.x);
                let mut p = [pos0, pos0 + n + v_1, pos1 + n - v_0, pos1];
                if triangular {
                    //The unneeded vertex is [2]
                    p[2] = p[3];
                    let mut line_0 = Line2D {
                        p0: p[0],
                        p1: p[1],
                        dash0: 0.0,
                        dash1: 0.0,
                        width_left: FLAP_LINE_WIDTH,
                        width_right: 0.0,
                    };
                    let mut line_1 = Line2D {
                        p0: p[1],
                        p1: p[2],
                        dash0: 0.0,
                        dash1: 0.0,
                        width_left: FLAP_LINE_WIDTH,
                        width_right: 0.0,
                    };
                    // Weird flaps are drawn differently:
                    // * forced flap in a rim
                    if edge.faces().1.is_none() {
                        line_0.set_dash(5.0);
                        line_1.set_dash(5.0);
                    }
                    args.vertices_flap_edge.push(line_0);
                    args.vertices_flap_edge.push(line_1);
                } else {
                    let mut line_0 = Line2D {
                        p0: p[0],
                        p1: p[1],
                        dash0: 0.0,
                        dash1: 0.0,
                        width_left: FLAP_LINE_WIDTH,
                        width_right: 0.0,
                    };
                    let mut line_1 = Line2D {
                        p0: p[1],
                        p1: p[2],
                        dash0: 0.0,
                        dash1: 0.0,
                        width_left: FLAP_LINE_WIDTH,
                        width_right: 0.0,
                    };
                    let mut line_2 = Line2D {
                        p0: p[2],
                        p1: p[3],
                        dash0: 0.0,
                        dash1: 0.0,
                        width_left: FLAP_LINE_WIDTH,
                        width_right: 0.0,
                    };
                    if edge.faces().1.is_none() {
                        line_0.set_dash(5.0);
                        line_1.set_dash(5.0);
                        line_2.set_dash(5.0);
                    }
                    args.vertices_flap_edge.push(line_0);
                    args.vertices_flap_edge.push(line_1);
                    args.vertices_flap_edge.push(line_2);
                };
                if let (Some(extra), Some(edge_id), Some(i_face_b)) =
                    (extra.as_mut(), edge_id, maybe_i_face_b)
                {
                    extra.cut_index.push(CutIndex::new(
                        pos0,
                        pos1,
                        Some(n),
                        i_face_b,
                        edge_id,
                        options,
                    ));
                }

                // Get material and geometry from adjacent face, if any
                let geom_b; //Option<(mx_b_inv, i_face_b)>
                let mat;
                let uvs;

                // helper function for the two cases below
                let compute_uvs = |face_b: &Face, mx_b: &Matrix3| -> Vec<Vector2> {
                    if flap_style == FlapStyle::White {
                        vec![Vector2::zero(); 4]
                    } else {
                        //Now we have to compute the texture coordinates of `p` in the adjacent face
                        let plane_b = self.papercraft.model().face_plane(face_b);
                        let vs_b = face_b.index_vertices().map(|v| {
                            let v = &self.papercraft.model()[v];
                            let p = plane_b.project(&v.pos(), scale);
                            (v, p)
                        });
                        // mx_basis converts from edge-relative coordinates to local face_b, where position of the tri vertices are [(0,0), (1,0), (0,1)]
                        let mx_basis =
                            Matrix2::from_cols(vs_b[1].1 - vs_b[0].1, vs_b[2].1 - vs_b[0].1);
                        // mxx does both convertions at once, inverted
                        let mxx = (mx_b * Matrix3::from(mx_basis)).invert().unwrap();

                        p.iter()
                            .map(|px| {
                                //vlocal is in edge-relative coordinates, that can be used to interpolate between UVs
                                let vlocal = mxx.transform_point(Point2::from_vec(*px)).to_vec();
                                let uv0 = vs_b[0].0.uv();
                                let uv1 = vs_b[1].0.uv();
                                let uv2 = vs_b[2].0.uv();
                                uv0 + vlocal.x * (uv1 - uv0) + vlocal.y * (uv2 - uv0)
                            })
                            .collect()
                    }
                };
                match maybe_i_face_b {
                    Some(i_face_b) => {
                        let face_b = &self.papercraft.model()[i_face_b];
                        let mx_b = m * self.papercraft.face_to_face_edge_matrix(edge, face, face_b);
                        let mx_b_inv = mx_b.invert().unwrap();
                        // mx_b_inv converts from paper to local face_b coordinates
                        geom_b = Some((mx_b_inv, i_face_b));
                        mat = face_b.material();
                        uvs = compute_uvs(face_b, &mx_b);
                    }
                    None => {
                        // There is no adjacent face to copy the texture from, so use the current
                        // face but mirrored.
                        // N shadow flaps.
                        geom_b = None;
                        mat = face.material();
                        uvs = compute_uvs(face, m);
                    }
                }
                let (root_alpha, tip_alpha) = match flap_style {
                    FlapStyle::Textured => (0.0, 0.0),
                    FlapStyle::HalfTextured => (0.0, 1.0),
                    FlapStyle::White => (1.0, 1.0),
                    FlapStyle::None => (0.0, 0.0), //should not happen
                };
                let root_color = Rgba::new(1.0, 1.0, 1.0, root_alpha);
                let tip_color = Rgba::new(1.0, 1.0, 1.0, tip_alpha);
                if triangular {
                    args.vertices_flap.push(MVertex2DColor {
                        pos_2d: p[0],
                        uv: uvs[0],
                        mat,
                        color: root_color,
                    });
                    args.vertices_flap.push(MVertex2DColor {
                        pos_2d: p[1],
                        uv: uvs[1],
                        mat,
                        color: tip_color,
                    });
                    args.vertices_flap.push(MVertex2DColor {
                        pos_2d: p[2],
                        uv: uvs[2],
                        mat,
                        color: root_color,
                    });
                } else {
                    let pp = [
                        MVertex2DColor {
                            pos_2d: p[0],
                            uv: uvs[0],
                            mat,
                            color: root_color,
                        },
                        MVertex2DColor {
                            pos_2d: p[1],
                            uv: uvs[1],
                            mat,
                            color: tip_color,
                        },
                        MVertex2DColor {
                            pos_2d: p[2],
                            uv: uvs[2],
                            mat,
                            color: tip_color,
                        },
                        MVertex2DColor {
                            pos_2d: p[3],
                            uv: uvs[3],
                            mat,
                            color: root_color,
                        },
                    ];
                    args.vertices_flap
                        .extend_from_slice(&[pp[0], pp[2], pp[1], pp[0], pp[3], pp[2]]);
                }
                if let (Some(flaps), Some((mx_b_inv, i_face_b))) = (&mut flap_cache, geom_b) {
                    let mut flap_vs = if triangular {
                        FlapVertices::Tri([p[0], p[1], p[2]])
                    } else {
                        FlapVertices::Quad([p[0], p[2], p[1], p[0], p[3], p[2]])
                    };
                    // Undo the mx_b transformation becase the shadow will be drawn over another
                    // face, the right matrix will be applied afterwards.
                    for sp in flap_vs.iter_mut() {
                        *sp = mx_b_inv.transform_point(Point2::from_vec(*sp)).to_vec();
                    }
                    flaps.push((i_face_b, flap_vs));
                }
            }
        }
    }

    fn paper_rebuild(&mut self, text_builder: &impl TextBuilder) {
        let options = self.papercraft.options();
        let mut args = PaperDrawFaceArgs::new(self.papercraft.model());
        let mut edge_id_info =
            if options.edge_id_position != EdgeIdPosition::None && self.ui.show_texts {
                Some((
                    PaperDrawFaceArgsExtra::default(),
                    slotmap::SecondaryMap::new(),
                ))
            } else {
                None
            };

        // Shadow flaps have to be drawn the the face adjacent to the one being drawn, but we do not
        // now its coordinates yet.
        // So we store the flap vertices and the face matrixes in temporary storage and draw the
        // shadow flaps later.
        let shadow_flap_alpha = options.shadow_flap_alpha;
        let mut shadow_cache = if shadow_flap_alpha > 0.0 {
            Some((HashMap::new(), Vec::new()))
        } else {
            None
        };
        for (i_island, island) in self.papercraft.islands() {
            let cut_before = args.cut_last_index();
            self.papercraft.traverse_faces(island, |i_face, face, mx| {
                if let Some((mx_face, _)) = &mut shadow_cache {
                    mx_face.insert(i_face, *mx);
                }
                self.paper_draw_face(
                    face,
                    i_face,
                    mx,
                    &mut args,
                    shadow_cache.as_mut().map(|(_, t)| t),
                    edge_id_info.as_mut().map(|(extra, _)| extra),
                );
                ControlFlow::Continue(())
            });
            if let Some((_, cuts)) = &mut edge_id_info {
                let cut_next = args.cut_last_index();
                cuts.insert(i_island, cut_before..cut_next);
            }
        }

        if let Some((mx_face, flap_cache)) = &shadow_cache {
            let uv = Vector2::zero();
            let mat = MaterialIndex::from(0);
            let color = Rgba::new(0.0, 0.0, 0.0, shadow_flap_alpha);
            for (i_face_b, ps) in flap_cache {
                let Some(mx) = mx_face.get(i_face_b) else {
                    continue; // should not happen
                };
                args.vertices_shadow_flap.extend(ps.iter().map(|p| {
                    let pos_2d = mx.transform_point(Point2::from_vec(*p)).to_vec();
                    MVertex2DColor {
                        pos_2d,
                        uv,
                        mat,
                        color,
                    }
                }));
            }
        }

        // Draw the EdgeId?
        if let Some((extra, cut_by_island)) = edge_id_info {
            let edge_id_font_size = options.edge_id_font_size * 25.4 / 72.0; // pt to mm

            // Edge ids
            for cut_idx in extra.cut_indices() {
                let i_island_b = self.papercraft().island_by_face(cut_idx.i_face_b);
                let ii = self
                    .papercraft()
                    .island_by_key(i_island_b)
                    .map(|island_b| island_b.name())
                    .unwrap_or("?");
                let text = format!("{}:{}", ii, cut_idx.id);
                let pos = cut_idx.pos(text_builder.font_text_line_scale() * edge_id_font_size);
                let t = PrintableText {
                    size: edge_id_font_size,
                    pos,
                    angle: cut_idx.angle,
                    align: TextAlign::Center,
                    text,
                };
                text_builder.make_text(&t, &mut args.vertices_text);
            }

            for (i_island, island) in self.papercraft().islands() {
                // Island ids
                let pos = match options.edge_id_position {
                    // On top (None should not happen)
                    EdgeIdPosition::None | EdgeIdPosition::Outside => {
                        let cut_range = cut_by_island[i_island].clone();
                        let top = args
                            .iter_cut_ex(cut_range)
                            .min_by(|a, b| a.0.y.total_cmp(&b.0.y))
                            .unwrap()
                            .0;
                        top - Vector2::new(0.0, edge_id_font_size)
                    }
                    // In the middle
                    EdgeIdPosition::Inside => {
                        let (flat_face, total_area) =
                            self.papercraft().get_biggest_flat_face(island);
                        // Compute the center of mass of the flat-face, that will be the
                        // weighted mean of the centers of masses of each single face.
                        let center: Vector2 = flat_face
                            .iter()
                            .map(|(i_face, area)| {
                                let vv: Vector2 = args.vertices_for_face(*i_face).into_iter().sum();
                                vv * *area
                            })
                            .sum();
                        // Don't forget to divide the center of each triangle by 3!
                        let center = center / total_area / 3.0;
                        center + Vector2::new(0.0, edge_id_font_size)
                    }
                };
                if let Some(island) = self.papercraft().island_by_key(i_island) {
                    let t = PrintableText {
                        size: 2.0 * edge_id_font_size,
                        pos,
                        angle: Rad(0.0),
                        align: TextAlign::Center,
                        text: String::from(island.name()),
                    };
                    text_builder.make_text(&t, &mut args.vertices_text);
                }
            }
        }

        //TODO PrintableTexts duplicated here and in generate_pages???
        if options.show_self_promotion || options.show_page_number {
            let (_margin_top, margin_left, margin_right, margin_bottom) = options.margin;
            let page_size_mm = Vector2::from(options.page_size);
            let page_count = options.pages;
            for page in 0..page_count {
                let page_pos = options.page_position(page);
                if options.show_self_promotion {
                    let x = margin_left;
                    let y = (page_size_mm.y - margin_bottom + FONT_SIZE)
                        .min(page_size_mm.y - FONT_SIZE);
                    let text = signature();
                    let t = PrintableText {
                        size: FONT_SIZE,
                        pos: page_pos + Vector2::new(x, y),
                        angle: Rad(0.0),
                        align: TextAlign::Near,
                        text,
                    };
                    text_builder.make_text(&t, &mut args.vertices_text);
                }
                if options.show_page_number {
                    let x = page_size_mm.x - margin_right;
                    let y = (page_size_mm.y - margin_bottom + FONT_SIZE)
                        .min(page_size_mm.y - FONT_SIZE);
                    let text = tr!("Page {}/{}", page + 1, page_count);
                    let t = PrintableText {
                        size: FONT_SIZE,
                        pos: page_pos + Vector2::new(x, y),
                        angle: Rad(0.0),
                        align: TextAlign::Far,
                        text,
                    };
                    text_builder.make_text(&t, &mut args.vertices_text);
                }
            }
        }

        self.gl_objs.paper_vertices.set(args.vertices);
        //self.gl_objs.paper_vertices_edge_cut.set(args.vertices_edge_cut);
        build_vertices_for_lines_2d(
            self.gl_objs.paper_vertices_edge_cut.data_mut(),
            &args.vertices_edge_cut,
            Rgba::new(0.0, 0.0, 0.0, 1.0),
        );
        build_vertices_for_lines_2d(
            self.gl_objs.paper_vertices_edge_crease.data_mut(),
            &args.vertices_edge_crease,
            Rgba::new(0.0, 0.0, 0.0, 1.0),
        );
        self.gl_objs.paper_vertices_flap.set(args.vertices_flap);
        build_vertices_for_lines_2d(
            self.gl_objs.paper_vertices_flap_edge.data_mut(),
            &args.vertices_flap_edge,
            Rgba::new(0.0, 0.0, 0.0, 1.0),
        );
        self.gl_objs
            .paper_vertices_shadow_flap
            .set(args.vertices_shadow_flap);
        self.gl_objs.paper_text.set(args.vertices_text);
    }

    fn pages_rebuild(&mut self) {
        let color = Rgba::new(1.0, 1.0, 1.0, 1.0);
        let mat = MaterialIndex::from(0);
        let mut page_vertices = Vec::new();
        let mut margin_vertices = Vec::new();
        let margin_line_width = 0.5;

        let page_size = Vector2::from(self.papercraft.options().page_size);
        let margin = self.papercraft.options().margin;
        let page_count = self.papercraft.options().pages;

        for page in 0..page_count {
            let page_pos = self.papercraft.options().page_position(page);

            let page_0 = MVertex2DColor {
                pos_2d: page_pos,
                uv: Vector2::zero(),
                mat,
                color,
            };
            let page_2 = MVertex2DColor {
                pos_2d: page_pos + page_size,
                uv: Vector2::zero(),
                mat,
                color,
            };
            let page_1 = MVertex2DColor {
                pos_2d: Vector2::new(page_2.pos_2d.x, page_0.pos_2d.y),
                uv: Vector2::zero(),
                mat,
                color,
            };
            let page_3 = MVertex2DColor {
                pos_2d: Vector2::new(page_0.pos_2d.x, page_2.pos_2d.y),
                uv: Vector2::zero(),
                mat,
                color,
            };
            page_vertices.extend_from_slice(&[page_0, page_2, page_1, page_0, page_3, page_2]);

            let mpos_0 = page_0.pos_2d + Vector2::new(margin.1, margin.0);
            let mpos_1 = page_3.pos_2d + Vector2::new(margin.1, -margin.3);
            let mpos_2 = page_2.pos_2d + Vector2::new(-margin.2, -margin.3);
            let mpos_3 = page_1.pos_2d + Vector2::new(-margin.2, margin.0);

            margin_vertices.push(Line2D {
                p0: mpos_0,
                p1: mpos_1,
                dash0: 0.0,
                dash1: page_size.y / 10.0,
                width_left: margin_line_width,
                width_right: 0.0,
            });
            margin_vertices.push(Line2D {
                p0: mpos_1,
                p1: mpos_2,
                dash0: 0.0,
                dash1: page_size.x / 10.0,
                width_left: margin_line_width,
                width_right: 0.0,
            });
            margin_vertices.push(Line2D {
                p0: mpos_2,
                p1: mpos_3,
                dash0: 0.0,
                dash1: page_size.y / 10.0,
                width_left: margin_line_width,
                width_right: 0.0,
            });
            margin_vertices.push(Line2D {
                p0: mpos_3,
                p1: mpos_0,
                dash0: 0.0,
                dash1: page_size.x / 10.0,
                width_left: margin_line_width,
                width_right: 0.0,
            });
        }
        self.gl_objs.paper_vertices_page.set(page_vertices);
        build_vertices_for_lines_2d(
            self.gl_objs.paper_vertices_margin.data_mut(),
            &margin_vertices,
            Rgba::new(0.5, 0.5, 0.5, 1.0),
        );
    }

    fn scene_edge_rebuild(&mut self) {
        let scene_vertices_edge_status = self.gl_objs.scene_vertices_edge_status.data_mut();

        for (&i_edge, &edge_idx) in &self.gl_objs.edge_map {
            let status = self.papercraft.edge_status(i_edge);
            let edge = &self.papercraft.model()[i_edge];

            let mut edge_status = if self
                .selected_edges
                .as_ref()
                .map(|sel| sel.contains(&i_edge))
                .unwrap_or(false)
            {
                MLine3DStatus {
                    color: color_edge(self.ui.mode),
                    thick: 5.0 / 2.0,
                    top: self.ui.xray_selection as u8,
                }
            } else if self.ui.show_3d_lines {
                match status {
                    EdgeStatus::Hidden => MLINE3D_HIDDEN,
                    EdgeStatus::Joined => {
                        let angle_3d = edge.angle();
                        if Rad(angle_3d.0.abs())
                            < Rad::from(Deg(self.papercraft.options().hidden_line_angle))
                        {
                            MLINE3D_HIDDEN
                        } else {
                            MLINE3D_NORMAL
                        }
                    }
                    EdgeStatus::Cut(_) => MLINE3D_CUT,
                }
            } else {
                MLINE3D_HIDDEN
            };

            // Edges of a top face are also top, or they would be hidden
            let (i_fa, i_fb) = edge.faces();
            if self.gl_objs.vertices_sel[3 * usize::from(i_fa)].top != 0 {
                edge_status.top = 1;
            } else if let Some(i_fb) = i_fb {
                if self.gl_objs.vertices_sel[3 * usize::from(i_fb)].top != 0 {
                    edge_status.top = 1;
                }
            }

            // The 3d line status is duplicated x6
            scene_vertices_edge_status[edge_idx..][..6].fill(edge_status);
        }
    }

    fn selection_rebuild(&mut self) {
        let n = self.gl_objs.vertices_sel.len();
        for i in 0..n {
            self.gl_objs.vertices_sel[i] = MSTATUS_UNSEL;
        }
        let top = self.ui.xray_selection as u8;

        // Pre-selected islands are drawn as if selected
        let empty = FxHashSet::default();
        let (pre_add, pre_remove) = match self.pre_selection.as_ref() {
            None => (&empty, &empty),
            Some((_, _, sel, true)) => (sel, &empty),
            Some((_, _, sel, false)) => (&empty, sel),
        };

        for &sel_island in self.selected_islands.iter().chain(pre_add) {
            if pre_remove.contains(&sel_island) {
                continue;
            }
            if let Some(island) = self.papercraft.island_by_key(sel_island) {
                self.papercraft.traverse_faces_no_matrix(island, |i_face| {
                    let pos = 3 * usize::from(i_face);
                    for i in pos..pos + 3 {
                        self.gl_objs.vertices_sel[i] = MStatus {
                            color: MSTATUS_SEL.color,
                            top,
                        };
                    }
                    ControlFlow::Continue(())
                });
            }
        }

        if let Some(i_sel_face) = self.selected_face {
            for i_face in self.papercraft.get_flat_faces(i_sel_face) {
                let pos = 3 * usize::from(i_face);
                for i in pos..pos + 3 {
                    self.gl_objs.vertices_sel[i] = MStatus {
                        color: MSTATUS_HI.color,
                        top,
                    };
                }
            }
        }

        if let Some(i_sel_edges) = self.selected_edges.as_ref() {
            let mut edge_sel_2d = Vec::with_capacity(i_sel_edges.len());
            for &i_sel_edge in i_sel_edges {
                let edge = &self.papercraft.model()[i_sel_edge];
                let (i_face_a, i_face_b) = edge.faces();

                // Returns the 2D vertices of i_sel_edge that belong to face i_face
                let get_vx = |i_face: FaceIndex| {
                    let face_a = &self.papercraft.model()[i_face];
                    let idx_face = 3 * usize::from(i_face);
                    let idx_edge = face_a
                        .index_edges()
                        .iter()
                        .position(|&e| e == i_sel_edge)
                        .unwrap();
                    let v0 = &self.gl_objs.paper_vertices[idx_face + idx_edge];
                    let v1 = &self.gl_objs.paper_vertices[idx_face + (idx_edge + 1) % 3];
                    (v0, v1)
                };

                let line_width = LINE_SEL_WIDTH / 2.0 / self.ui.trans_paper.mx[0][0];

                let (v0, v1) = get_vx(i_face_a);
                let idx_2d = edge_sel_2d.len();
                edge_sel_2d.push(Line2D {
                    p0: v0.pos_2d,
                    p1: v1.pos_2d,
                    dash0: 0.0,
                    dash1: 0.0,
                    width_left: line_width,
                    width_right: line_width,
                });
                if let Some(i_face_b) = i_face_b {
                    let (vb0, vb1) = get_vx(i_face_b);
                    edge_sel_2d.push(Line2D {
                        p0: vb0.pos_2d,
                        p1: vb1.pos_2d,
                        dash0: 0.0,
                        dash1: 0.0,
                        width_left: line_width,
                        width_right: line_width,
                    });
                    let mut link_line = Line2D {
                        p0: (edge_sel_2d[idx_2d].p0 + edge_sel_2d[idx_2d].p1) / 2.0,
                        p1: (edge_sel_2d[idx_2d + 1].p0 + edge_sel_2d[idx_2d + 1].p1) / 2.0,
                        dash0: 0.0,
                        dash1: 0.0,
                        width_left: line_width,
                        width_right: line_width,
                    };
                    link_line.dash1 = link_line.p0.distance(link_line.p1);
                    edge_sel_2d.push(link_line);
                } else {
                    // If there is no face_b it is a rim, highlight it specially
                    // This line_dash will create a 4.5 repetition pattern (- - - -)
                    edge_sel_2d[idx_2d].set_dash(5.0);
                }
            }
            build_vertices_for_lines_2d(
                self.gl_objs.paper_vertices_edge_sel.data_mut(),
                &edge_sel_2d,
                color_edge(self.ui.mode),
            );
        }
    }

    #[must_use]
    pub fn set_selection(
        &mut self,
        selection: ClickResult,
        flags: SetSelectionFlags,
    ) -> RebuildFlags {
        let mut island_changed = false;

        let just_selected = self.just_selected.take();
        let (new_edges, new_face) = match selection {
            ClickResult::None => {
                if flags.contains(SetSelectionFlags::CLICKED)
                    && !flags.contains(SetSelectionFlags::ADD_TO_SEL)
                    && !self.selected_islands.is_empty()
                {
                    self.selected_islands.clear();
                    island_changed = true;
                }
                (None, None)
            }
            ClickResult::Face(i_face) => {
                let i_island = self.papercraft.island_by_face(i_face);
                if flags.contains(SetSelectionFlags::CLICKED) {
                    if flags.contains(SetSelectionFlags::ADD_TO_SEL) {
                        if let Some(n) = self.selected_islands.iter().position(|i| *i == i_island) {
                            if flags.contains(SetSelectionFlags::RELEASED)
                                && just_selected != Some(i_island)
                            {
                                self.selected_islands.swap_remove(n);
                                island_changed = true;
                            }
                        } else {
                            self.selected_islands.push(i_island);
                            self.just_selected = Some(i_island);
                            island_changed = true;
                        }
                    } else {
                        // If the clicked island is already selected do nothing on click, but select on release.
                        // This makes it easier to move several islands, because the user doesn't need to hold Ctrl to move them.
                        if !self.selected_islands.contains(&i_island)
                            || flags.contains(SetSelectionFlags::RELEASED)
                        {
                            self.selected_islands = vec![i_island];
                            self.just_selected = Some(i_island);
                            island_changed = true;
                        }
                    }
                }
                let edges = if flags.contains(SetSelectionFlags::ALT_PRESSED) {
                    let island = self.papercraft.island_by_key(i_island).unwrap();
                    Some(self.papercraft.island_edges(island))
                } else {
                    None
                };
                (edges, Some(i_face))
            }
            ClickResult::Edge(i_edge, maybe_i_face) => {
                let edges = match (flags.contains(SetSelectionFlags::ALT_PRESSED), maybe_i_face) {
                    (true, Some(i_face)) => {
                        let i_island = self.papercraft.island_by_face(i_face);
                        let island = self.papercraft.island_by_key(i_island).unwrap();
                        self.papercraft.island_edges(island)
                    }
                    _ => {
                        let mut set = FxHashSet::default();
                        set.insert(i_edge);
                        set
                    }
                };
                (Some(edges), None)
            }
        };
        let rebuild =
            if island_changed || self.selected_edges != new_edges || self.selected_face != new_face
            {
                RebuildFlags::SELECTION
            } else {
                RebuildFlags::empty()
            };
        self.selected_edges = new_edges;
        self.selected_face = new_face;
        rebuild
    }

    #[must_use]
    pub fn edge_toggle_cut(
        &mut self,
        i_edge: EdgeIndex,
        priority_face: Option<FaceIndex>,
    ) -> Option<Vec<UndoAction>> {
        match self.papercraft.edge_status(i_edge) {
            EdgeStatus::Hidden => None,
            EdgeStatus::Joined => {
                let offset = self.papercraft.options().flap_width * 2.0;
                self.papercraft.edge_cut(i_edge, Some(offset));
                Some(vec![UndoAction::EdgeCut { i_edge }])
            }
            EdgeStatus::Cut(_) => {
                let renames = self.papercraft.edge_join(i_edge, priority_face);
                if renames.is_empty() {
                    return None;
                }
                let undo_actions = renames
                    .values()
                    .map(|join_result| UndoAction::EdgeJoin {
                        join_result: *join_result,
                    })
                    .collect();
                self.islands_renamed(&renames);
                Some(undo_actions)
            }
        }
    }

    #[must_use]
    pub fn try_join_strip(&mut self, i_edge: EdgeIndex) -> Option<Vec<UndoAction>> {
        let renames = self.papercraft.try_join_strip(i_edge);
        if renames.is_empty() {
            return None;
        }

        let undo_actions = renames
            .values()
            .map(|join_result| UndoAction::EdgeJoin {
                join_result: *join_result,
            })
            .collect();
        self.islands_renamed(&renames);
        Some(undo_actions)
    }

    fn islands_renamed(&mut self, renames: &FxHashMap<IslandKey, JoinResult>) {
        for x in &mut self.selected_islands {
            while let Some(jr) = renames.get(x) {
                *x = jr.i_island;
            }
        }
    }

    pub fn scene_analyze_click(&self, mode: MouseMode, size: Vector2, pos: Vector2) -> ClickResult {
        let x = (pos.x / size.x) * 2.0 - 1.0;
        let y = -((pos.y / size.y) * 2.0 - 1.0);
        let click = Point3::new(x, y, 1.0);
        let height = size.y * self.ui.trans_scene.obj[1][1];

        let click_camera = self.ui.trans_scene.persp_inv.transform_point(click);
        let click_obj = self.ui.trans_scene.view_inv.transform_point(click_camera);
        let camera_obj = self
            .ui
            .trans_scene
            .view_inv
            .transform_point(Point3::new(0.0, 0.0, 0.0));

        let ray = (camera_obj.to_vec(), click_obj.to_vec());

        //Faces has to be checked both in Edge and Face mode, because Edges can be hidden by a face.
        let mut hit_face = None;
        for (iface, face) in self.papercraft.model().faces() {
            let tri = face
                .index_vertices()
                .map(|v| self.papercraft.model()[v].pos());
            let maybe_new_hit = util_3d::ray_crosses_face(ray, &tri);
            if let Some(new_hit) = maybe_new_hit {
                hit_face = match (hit_face, new_hit) {
                    (Some((_, p)), x) if p > x && x > 0.0 => Some((iface, x)),
                    (None, x) if x > 0.0 => Some((iface, x)),
                    (old, _) => old,
                };
            }
        }

        if mode == MouseMode::Face {
            return match hit_face {
                None => ClickResult::None,
                Some((f, _)) => ClickResult::Face(f),
            };
        }

        let mut hit_edge = None;
        for (i_edge, edge) in self.papercraft.model().edges() {
            match (self.papercraft.edge_status(i_edge), mode) {
                (EdgeStatus::Hidden, _) => continue,
                (EdgeStatus::Joined, MouseMode::Flap) => continue,
                _ => (),
            }
            let (v1, v2) = self.papercraft.model().edge_pos(edge);
            let (ray_hit, _line_hit, new_dist) = util_3d::line_segment_distance(ray, (v1, v2));

            // Behind the screen, it is not a hit
            if ray_hit <= 0.0001 {
                continue;
            }

            // new_dist is originally the distance in real-world space, but the user is using the screen, so scale accordingly
            let new_dist = new_dist / ray_hit * height;

            // If this egde is from the ray further that the best one, it is worse and ignored
            match hit_edge {
                Some((_, _, p)) if p < new_dist => {
                    continue;
                }
                _ => {}
            }

            // Too far from the edge
            if new_dist > 5.0 {
                continue;
            }

            // If there is a face 99% nearer this edge, it is hidden, probably, so it does not count
            match hit_face {
                Some((_, p)) if p < 0.99 * ray_hit => {
                    continue;
                }
                _ => {}
            }

            hit_edge = Some((i_edge, ray_hit, new_dist));
        }

        // Edge has priority
        match (hit_edge, hit_face) {
            (Some((e, _, _)), Some((f, _))) => ClickResult::Edge(e, Some(f)),
            (Some((e, _, _)), None) => ClickResult::Edge(e, None),
            (None, Some((f, _))) => ClickResult::Face(f),
            (None, None) => ClickResult::None,
        }
    }

    pub fn paper_analyze_click(&self, mode: MouseMode, size: Vector2, pos: Vector2) -> ClickResult {
        let click = self.ui.trans_paper.paper_click(size, pos);
        let mx = self.ui.trans_paper.ortho * self.ui.trans_paper.mx;
        let scale = self.papercraft.options().scale;

        let mut hit_edge = None;
        let mut hit_face = None;

        for (_i_island, island) in self
            .papercraft
            .islands()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            self.papercraft.traverse_faces(island, |i_face, face, fmx| {
                let plane = self.papercraft.model().face_plane(face);

                let tri = face.index_vertices();
                let tri = tri.map(|v| {
                    let v3 = self.papercraft.model()[v].pos();
                    let v2 = plane.project(&v3, scale);
                    fmx.transform_point(Point2::from_vec(v2)).to_vec()
                });
                if hit_face.is_none() && util_3d::point_in_triangle(click, tri) {
                    hit_face = Some(i_face);
                }
                match mode {
                    MouseMode::Face => {}
                    MouseMode::Edge | MouseMode::Flap | MouseMode::ReadOnly => {
                        for i_edge in face.index_edges() {
                            match (self.papercraft.edge_status(i_edge), mode) {
                                (EdgeStatus::Hidden, _) => continue,
                                (EdgeStatus::Joined, MouseMode::Flap) => continue,
                                _ => (),
                            }
                            let edge = &self.papercraft.model()[i_edge];
                            let (v0, v1) = self.papercraft.model().edge_pos(edge);
                            let v0 = plane.project(&v0, scale);
                            let v1 = plane.project(&v1, scale);
                            let v0 = fmx.transform_point(Point2::from_vec(v0)).to_vec();
                            let v1 = fmx.transform_point(Point2::from_vec(v1)).to_vec();

                            let (_o, d) = util_3d::point_segment_distance(click, (v0, v1));
                            let d = <Matrix3 as Transform<Point2>>::transform_vector(
                                &mx,
                                Vector2::new(d, 0.0),
                            )
                            .magnitude();
                            if d > 0.02 {
                                //too far?
                                continue;
                            }
                            match &hit_edge {
                                None => {
                                    hit_edge = Some((d, i_edge, i_face));
                                }
                                &Some((d_prev, _, _)) if d < d_prev => {
                                    hit_edge = Some((d, i_edge, i_face));
                                }
                                _ => {}
                            }
                        }
                    }
                }
                ControlFlow::Continue(())
            });
        }

        // Edge has priority
        match (hit_edge, hit_face) {
            (Some((_d, i_edge, i_face)), _) => ClickResult::Edge(i_edge, Some(i_face)),
            (None, Some(i_face)) => ClickResult::Face(i_face),
            (None, None) => ClickResult::None,
        }
    }

    #[must_use]
    pub fn scene_zoom(&mut self, _size: Vector2, _pos: Vector2, zoom: f32) -> RebuildFlags {
        self.ui.trans_scene.scale *= zoom;
        self.ui.trans_scene.recompute_obj();
        RebuildFlags::SCENE_REDRAW
    }
    #[must_use]
    pub fn scene_hover_event(
        &mut self,
        size: Vector2,
        pos: Vector2,
        alt_pressed: bool,
    ) -> RebuildFlags {
        self.last_cursor_pos = pos;
        let selection = self.scene_analyze_click(self.ui.mode, size, pos);
        let flags = if alt_pressed {
            SetSelectionFlags::ALT_PRESSED
        } else {
            SetSelectionFlags::empty()
        };
        self.set_selection(selection, flags)
    }
    #[must_use]
    pub fn scene_button1_click_event(&mut self, _size: Vector2, pos: Vector2) -> RebuildFlags {
        let delta = pos - self.last_cursor_pos;
        self.last_cursor_pos = pos;
        // Rotate, half angles
        let ang = delta / 200.0 / 2.0;
        let cosy = ang.x.cos();
        let siny = ang.x.sin();
        let cosx = ang.y.cos();
        let sinx = ang.y.sin();
        let roty = Quaternion::new(cosy, 0.0, siny, 0.0);
        let rotx = Quaternion::new(cosx, sinx, 0.0, 0.0);

        self.ui.trans_scene.rotation = (roty * rotx * self.ui.trans_scene.rotation).normalize();
        self.ui.trans_scene.recompute_obj();
        RebuildFlags::SCENE_REDRAW
    }
    #[must_use]
    pub fn scene_button2_click_event(&mut self, _size: Vector2, pos: Vector2) -> RebuildFlags {
        let delta = pos - self.last_cursor_pos;
        self.last_cursor_pos = pos;
        // Translate
        let delta = delta / 50.0;
        self.ui.trans_scene.location += Vector3::new(delta.x, -delta.y, 0.0);
        self.ui.trans_scene.recompute_obj();
        RebuildFlags::SCENE_REDRAW
    }
    #[must_use]
    pub fn scene_button1_dblclick_event(&mut self, size: Vector2, pos: Vector2) -> RebuildFlags {
        let selection = self.scene_analyze_click(MouseMode::Face, size, pos);
        let ClickResult::Face(i_face) = selection else {
            return RebuildFlags::empty();
        };
        // Compute the average of all the faces flat with the selected one, and move it to the center of the paper.
        // Some vertices are counted twice, but they tend to be in diagonally opposed so the compensate, and it is an approximation anyways.
        let mut center = Vector2::zero();
        let mut n = 0.0;
        for i_face in self.papercraft.get_flat_faces(i_face) {
            let idx = 3 * usize::from(i_face);
            for i in idx..idx + 3 {
                center += self.gl_objs.paper_vertices[i].pos_2d;
                n += 1.0;
            }
        }
        center /= n;
        self.ui.trans_paper.mx[2][0] = -center.x * self.ui.trans_paper.mx[0][0];
        self.ui.trans_paper.mx[2][1] = -center.y * self.ui.trans_paper.mx[1][1];
        RebuildFlags::SCENE_REDRAW
    }

    // These 2 functions are common for {scene,paper}_button1_release_event
    #[must_use]
    fn do_edge_action(
        &mut self,
        i_edge: EdgeIndex,
        i_face: Option<FaceIndex>,
        shift_action: bool,
    ) -> RebuildFlags {
        let undo = if shift_action {
            self.try_join_strip(i_edge)
        } else {
            self.edge_toggle_cut(i_edge, i_face)
        };
        if let Some(undo) = undo {
            self.push_undo_action(undo);
        }
        RebuildFlags::PAPER
            | RebuildFlags::SCENE_EDGE
            | RebuildFlags::SELECTION
            | RebuildFlags::ISLANDS
    }
    #[must_use]
    fn do_flap_action(&mut self, i_edge: EdgeIndex, shift_action: bool) -> RebuildFlags {
        let action = if shift_action {
            EdgeToggleFlapAction::Hide
        } else {
            EdgeToggleFlapAction::Toggle
        };
        if let Some(flap_side) = self.papercraft.edge_toggle_flap(i_edge, action) {
            self.push_undo_action(vec![UndoAction::FlapToggle { i_edge, flap_side }]);
            RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION
        } else {
            RebuildFlags::empty()
        }
    }

    #[must_use]
    pub fn scene_button1_release_event(
        &mut self,
        size: Vector2,
        pos: Vector2,
        shift_action: bool,
        add_to_sel: bool,
    ) -> RebuildFlags {
        let selection = self.scene_analyze_click(self.ui.mode, size, pos);
        let flags = if add_to_sel {
            SetSelectionFlags::ADD_TO_SEL
        } else {
            SetSelectionFlags::empty()
        };
        let flags = flags | SetSelectionFlags::CLICKED | SetSelectionFlags::RELEASED;
        match (self.ui.mode, selection) {
            (MouseMode::Edge, ClickResult::Edge(i_edge, i_face)) => {
                self.do_edge_action(i_edge, i_face, shift_action)
            }
            (MouseMode::Flap, ClickResult::Edge(i_edge, _)) => {
                self.do_flap_action(i_edge, shift_action)
            }
            (_, ClickResult::Face(f)) | (MouseMode::ReadOnly, ClickResult::Edge(_, Some(f))) => {
                self.set_selection(ClickResult::Face(f), flags)
            }
            (_, ClickResult::None) => self.set_selection(ClickResult::None, flags),
            _ => RebuildFlags::empty(),
        }
    }
    #[must_use]
    pub fn paper_button2_event(&mut self, _size: Vector2, pos: Vector2) -> RebuildFlags {
        let delta = pos - self.last_cursor_pos;
        self.last_cursor_pos = pos;
        // Translate
        self.ui.trans_paper.mx = Matrix3::from_translation(delta) * self.ui.trans_paper.mx;
        RebuildFlags::PAPER_REDRAW
    }
    #[must_use]
    pub fn paper_button1_grab_event(
        &mut self,
        size: Vector2,
        pos: Vector2,
        rotating: bool,
        dragging: bool,
    ) -> RebuildFlags {
        let delta = pos - self.last_cursor_pos;
        self.last_cursor_pos = pos;

        // Check if any island is to be moved
        match (
            self.selected_islands.is_empty(),
            self.grabbed_island.as_mut(),
        ) {
            (false, Some(undo)) => {
                // Keep grabbed_island as Some(empty), grabbed but already pushed into undo_actions
                if !dragging {
                    return RebuildFlags::empty();
                }
                let undo = std::mem::take(undo);
                self.push_undo_action(undo);
            }
            _ => {
                // The same key for rotating (shift) is used for removing the selection.
                let adding = !rotating;

                // No selection, check if we should build a pre-selection rectangle
                let click = self.ui.trans_paper.paper_click(size, pos);

                let (next, flags) = match self.pre_selection.take() {
                    // First pre-selection click, just save it
                    None => (
                        (click, click, FxHashSet::default(), adding),
                        RebuildFlags::empty(),
                    ),
                    // If not dragging yet keep a zero size rectangle
                    Some(p) if !dragging => (p, RebuildFlags::empty()),
                    // If dragging do the full pre-selection stuff
                    Some((orig, _, pre_sel, pre_adding)) => {
                        let mut next_sel = FxHashSet::default();
                        let rect = Rectangle::new(orig, click);

                        // An island is in the selection box if any of its vertices is inside the
                        // rectangle.
                        for (island_key, island) in self.papercraft.islands() {
                            self.papercraft.traverse_faces_no_matrix(island, |i_face| {
                                let idx = 3 * usize::from(i_face);
                                for i in idx..idx + 3 {
                                    let pos = self.gl_objs.paper_vertices[i].pos_2d;
                                    if rect.contains(pos) {
                                        next_sel.insert(island_key);
                                        return ControlFlow::Break(());
                                    }
                                }
                                ControlFlow::Continue(())
                            });
                        }
                        let flags = if pre_sel != next_sel || pre_adding != adding {
                            RebuildFlags::SELECTION
                        } else {
                            RebuildFlags::empty()
                        };
                        ((orig, click, next_sel, adding), flags)
                    }
                };
                self.pre_selection = Some(next);
                return flags;
            }
        }

        if rotating {
            // Rotate island
            let center = *self.rotation_center.get_or_insert(pos);
            //Rotating when the pointer is very near to the center or rotation the angle could go crazy, so disable it
            if center.distance2(pos) > 10.0_f32.powi(2) {
                let pcenter = self.ui.trans_paper.paper_click(size, center);
                let ppos_prev = self.ui.trans_paper.paper_click(size, pos - delta);
                let ppos = self.ui.trans_paper.paper_click(size, pos);
                let angle = (ppos_prev - pcenter).angle(ppos - pcenter);
                for &i_island in &self.selected_islands {
                    if let Some(island) = self.papercraft.island_by_key_mut(i_island) {
                        island.rotate(angle, pcenter);
                    }
                }
            }
        } else {
            // Move island
            let delta_scaled = <Matrix3 as Transform<Point2>>::inverse_transform_vector(
                &self.ui.trans_paper.mx,
                delta,
            )
            .unwrap();

            let mut going_outside = false;
            for &i_island in &self.selected_islands {
                let location = if let Some(island) = self.papercraft.island_by_key_mut(i_island) {
                    island.translate(delta_scaled);
                    island.location()
                } else {
                    continue;
                };
                going_outside |= !self.papercraft.options().is_inside_canvas(location);
            }

            // When moving an island the center of rotation is preserved as the original clicked point
            if let Some(c) = &mut self.rotation_center {
                *c += delta;
            }

            if !going_outside {
                'scroll: {
                    //If the mouse is outside of the canvas, do as if it were inside, so it can be scrolled in the next tick
                    let delta = if pos.x < 5.0 {
                        Vector2::new((-pos.x).clamp(5.0, 25.0), 0.0)
                    } else if pos.x > size.x - 5.0 {
                        Vector2::new(-(pos.x - size.x).clamp(5.0, 25.0), 0.0)
                    } else if pos.y < 5.0 {
                        Vector2::new(0.0, (-pos.y).clamp(5.0, 25.0))
                    } else if pos.y > size.y - 5.0 {
                        Vector2::new(0.0, -(pos.y - size.y).clamp(5.0, 25.0))
                    } else {
                        break 'scroll;
                    };
                    let delta = delta / 2.0;
                    self.last_cursor_pos += delta;
                    self.ui.trans_paper.mx =
                        Matrix3::from_translation(delta) * self.ui.trans_paper.mx;
                }
            }
        }
        RebuildFlags::PAPER
    }

    #[must_use]
    pub fn paper_button1_drag_complete_event(&mut self) -> RebuildFlags {
        if let Some((_, _, mut sel, adding)) = self.pre_selection.take() {
            if adding {
                // Use the sel hash-map to remove duplicates
                sel.extend(self.selected_islands.iter().copied());
                self.selected_islands = Vec::from_iter(sel);
            } else {
                // Remove the pre-selected islands
                let prev: FxHashSet<_> = self.selected_islands.iter().copied().collect();
                self.selected_islands = Vec::from_iter(prev.difference(&sel).copied());
            }
            RebuildFlags::SELECTION
        } else {
            RebuildFlags::empty()
        }
    }

    #[must_use]
    pub fn paper_button1_click_event(
        &mut self,
        size: Vector2,
        pos: Vector2,
        shift_action: bool,
        add_to_sel: bool,
        modifiable: bool,
    ) -> RebuildFlags {
        let selection = self.paper_analyze_click(self.ui.mode, size, pos);
        let flags = if add_to_sel {
            SetSelectionFlags::ADD_TO_SEL
        } else {
            SetSelectionFlags::empty()
        };
        let flags = flags | SetSelectionFlags::CLICKED;

        match (self.ui.mode, selection) {
            (MouseMode::Edge, ClickResult::Edge(i_edge, i_face)) => {
                self.grabbed_island = None;
                self.do_edge_action(i_edge, i_face, shift_action)
            }
            (MouseMode::Flap, ClickResult::Edge(i_edge, _)) => {
                self.do_flap_action(i_edge, shift_action)
            }
            (_, ClickResult::Face(f)) | (MouseMode::ReadOnly, ClickResult::Edge(_, Some(f))) => {
                let flags = self.set_selection(ClickResult::Face(f), flags);
                if modifiable {
                    let undo_action: Vec<_> = self
                        .selected_islands
                        .iter()
                        .map(|&i_island| {
                            let island = self.papercraft.island_by_key(i_island).unwrap();
                            UndoAction::IslandMove {
                                i_root: island.root_face(),
                                prev_rot: island.rotation(),
                                prev_loc: island.location(),
                            }
                        })
                        .collect();
                    self.grabbed_island
                        .get_or_insert_with(Vec::new)
                        .extend(undo_action);
                }
                flags
            }
            (_, ClickResult::None) => {
                self.grabbed_island = None;
                self.set_selection(ClickResult::None, flags)
            }
            _ => RebuildFlags::empty(),
        }
    }
    #[must_use]
    pub fn paper_button1_release_event(
        &mut self,
        size: Vector2,
        pos: Vector2,
        add_to_sel: bool,
    ) -> RebuildFlags {
        self.pre_selection = None;
        let selection = self.paper_analyze_click(self.ui.mode, size, pos);
        let flags = if add_to_sel {
            SetSelectionFlags::ADD_TO_SEL
        } else {
            SetSelectionFlags::empty()
        };
        let flags = flags | SetSelectionFlags::CLICKED | SetSelectionFlags::RELEASED;
        match (self.ui.mode, selection) {
            (_, ClickResult::Face(f)) | (MouseMode::ReadOnly, ClickResult::Edge(_, Some(f))) => {
                self.set_selection(ClickResult::Face(f), flags)
            }
            _ => RebuildFlags::empty(),
        }
    }
    #[must_use]
    pub fn paper_zoom(&mut self, size: Vector2, pos: Vector2, zoom: f32) -> RebuildFlags {
        let pos = pos - size / 2.0;
        let tr = Matrix3::from_translation(pos)
            * Matrix3::from_scale(zoom)
            * Matrix3::from_translation(-pos);
        self.ui.trans_paper.mx = tr * self.ui.trans_paper.mx;
        // If there is a rotation center keep it at the same relative point
        if let Some(c) = &mut self.rotation_center {
            *c = pos + zoom * (*c - pos);
        }
        RebuildFlags::PAPER_REDRAW | RebuildFlags::SELECTION
    }
    #[must_use]
    pub fn paper_hover_event(
        &mut self,
        size: Vector2,
        pos: Vector2,
        alt_pressed: bool,
    ) -> RebuildFlags {
        self.pre_selection = None;
        self.last_cursor_pos = pos;
        let selection = self.paper_analyze_click(self.ui.mode, size, pos);
        self.rotation_center = None;
        self.grabbed_island = None;
        let flags = if alt_pressed {
            SetSelectionFlags::ALT_PRESSED
        } else {
            SetSelectionFlags::empty()
        };
        self.set_selection(selection, flags)
    }

    pub fn pre_selection_rectangle(&self) -> Option<Rectangle> {
        self.pre_selection
            .as_ref()
            .map(|(a, b, _, _)| Rectangle::new(*a, *b))
    }

    #[must_use]
    pub fn pack_islands(&mut self) -> Vec<UndoAction> {
        let undo_actions = self
            .papercraft
            .islands()
            .map(|(_, island)| UndoAction::IslandMove {
                i_root: island.root_face(),
                prev_rot: island.rotation(),
                prev_loc: island.location(),
            })
            .collect();
        self.papercraft.pack_islands();
        undo_actions
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }
    pub fn undo_action(&mut self) -> UndoResult {
        //Do not undo while grabbing or the stack will be messed up
        if self.grabbed_island.is_some() {
            return UndoResult::False;
        }

        let action_pack = match self.undo_stack.pop() {
            None => return UndoResult::False,
            Some(a) => a,
        };

        let mut res = UndoResult::Model;

        for action in action_pack.into_iter().rev() {
            match action {
                UndoAction::IslandMove {
                    i_root,
                    prev_rot,
                    prev_loc,
                } => {
                    let i_island = self.papercraft.island_by_face(i_root);
                    let island = self.papercraft.island_by_key_mut(i_island).unwrap();
                    island.reset_transformation(i_root, prev_rot, prev_loc);
                }
                UndoAction::FlapToggle { i_edge, flap_side } => {
                    self.papercraft
                        .edge_toggle_flap(i_edge, EdgeToggleFlapAction::Set(flap_side));
                }
                UndoAction::EdgeCut { i_edge } => {
                    self.papercraft.edge_join(i_edge, None);
                }
                UndoAction::EdgeJoin { join_result } => {
                    self.papercraft.edge_cut(join_result.i_edge, None);
                    let i_prev_island = self.papercraft.island_by_face(join_result.prev_root);
                    let island = self.papercraft.island_by_key_mut(i_prev_island).unwrap();

                    island.reset_transformation(
                        join_result.prev_root,
                        join_result.prev_rot,
                        join_result.prev_loc,
                    );
                }
                UndoAction::DocConfig {
                    options,
                    island_pos,
                } => {
                    for (i_root_face, (rot, loc)) in island_pos {
                        let i_island = self.papercraft.island_by_face(i_root_face);
                        let island = self.papercraft.island_by_key_mut(i_island).unwrap();
                        island.reset_transformation(i_root_face, rot, loc);
                    }
                    res = UndoResult::ModelAndOptions(options);
                }
                UndoAction::Modified => {
                    self.modified = false;
                }
            }
        }
        res
    }
    pub fn push_undo_action(&mut self, mut action: Vec<UndoAction>) {
        if action.is_empty() {
            return;
        }
        if !self.modified {
            action.push(UndoAction::Modified);
            self.modified = true;
        }
        self.undo_stack.push(action);
    }
    pub fn has_selected_edge(&self) -> bool {
        self.selected_edges.is_some()
    }

    pub fn lines_by_island(&self) -> Vec<(IslandKey, (PaperDrawFaceArgs, PaperDrawFaceArgsExtra))> {
        self.papercraft
            .islands()
            .map(|(id, island)| {
                let mut args = PaperDrawFaceArgs::new(self.papercraft.model());
                let mut extra = PaperDrawFaceArgsExtra::default();
                self.papercraft.traverse_faces(island, |i_face, face, mx| {
                    self.paper_draw_face(face, i_face, mx, &mut args, None, Some(&mut extra));
                    ControlFlow::Continue(())
                });
                (id, (args, extra))
            })
            .collect()
    }

    pub fn prepare_thumbnail(&mut self, sz: Vector2) -> ThumbnailData {
        let ui_settings = self.ui.clone();

        let (mut trans_scene, _) =
            default_transformations(self.ui.trans_scene.obj, sz, sz, self.papercraft.options());

        trans_scene.rotation =
            Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0).normalize(), Deg(20.0))
                * Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0).normalize(), -Deg(40.0));
        trans_scene.recompute_obj();

        self.ui.trans_scene = trans_scene;
        self.ui.show_3d_lines = false;
        self.ui.xray_selection = false;
        self.ui.show_textures = self.papercraft().model().has_textures();
        let selected_face = self.selected_face.take();
        let selected_edges = self.selected_edges.take();
        let selected_islands = std::mem::take(&mut self.selected_islands);
        ThumbnailData {
            ui_settings,
            selected_face,
            selected_edges,
            selected_islands,
        }
    }
    pub fn restore_thumbnail(&mut self, td: ThumbnailData) {
        self.ui = td.ui_settings;
        self.selected_face = td.selected_face;
        self.selected_edges = td.selected_edges;
        self.selected_islands = td.selected_islands;
    }
}

pub struct ThumbnailData {
    ui_settings: UiSettings,
    selected_face: Option<FaceIndex>,
    selected_edges: Option<FxHashSet<EdgeIndex>>,
    selected_islands: Vec<IslandKey>,
}

impl GLObjects {
    fn new(papercraft: &Papercraft, gl: &GlContext) -> anyhow::Result<GLObjects> {
        let model = papercraft.model();
        let images = model.textures().map(|tex| tex.pixbuf()).collect::<Vec<_>>();

        let sizes = images
            .iter()
            .filter_map(|i| i.as_ref())
            .map(|i| (i.width(), i.height()));
        let max_width = sizes.clone().map(|(w, _)| w).max();
        let max_height = sizes.map(|(_, h)| h).max();

        let textures = match max_width.zip(max_height) {
            None => None,
            Some((width, height)) => {
                let mut blank = None;
                unsafe {
                    let textures = glr::Texture::generate(gl)?;
                    gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(textures.id()));
                    gl.tex_image_3d(
                        glow::TEXTURE_2D_ARRAY,
                        0,
                        glow::RGBA8 as i32,
                        width as i32,
                        height as i32,
                        images.len() as i32,
                        0,
                        glow::RGB,
                        glow::UNSIGNED_BYTE,
                        None,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D_ARRAY,
                        glow::TEXTURE_WRAP_S,
                        glow::REPEAT as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D_ARRAY,
                        glow::TEXTURE_WRAP_T,
                        glow::REPEAT as i32,
                    );
                    set_texture_filter(gl, papercraft.options().tex_filter);

                    for (layer, image) in images.iter().enumerate() {
                        if let Some(image) = image {
                            let scaled_image;
                            let image = if width == image.width() && height == image.height() {
                                image
                            } else {
                                let scaled = image::imageops::resize(
                                    *image,
                                    width,
                                    height,
                                    image::imageops::FilterType::Triangle,
                                );
                                scaled_image = DynamicImage::ImageRgba8(scaled);
                                &scaled_image
                            };
                            let bytes = image.as_bytes();
                            let (format, type_) = match image {
                                DynamicImage::ImageLuma8(_) => (glow::RED, glow::UNSIGNED_BYTE),
                                DynamicImage::ImageLumaA8(_) => (glow::RG, glow::UNSIGNED_BYTE),
                                DynamicImage::ImageRgb8(_) => (glow::RGB, glow::UNSIGNED_BYTE),
                                DynamicImage::ImageRgba8(_) => (glow::RGBA, glow::UNSIGNED_BYTE),
                                DynamicImage::ImageLuma16(_) => (glow::RED, glow::UNSIGNED_SHORT),
                                DynamicImage::ImageLumaA16(_) => (glow::RG, glow::UNSIGNED_SHORT),
                                DynamicImage::ImageRgb16(_) => (glow::RGB, glow::UNSIGNED_SHORT),
                                DynamicImage::ImageRgba16(_) => (glow::RGBA, glow::UNSIGNED_SHORT),
                                DynamicImage::ImageRgb32F(_) => (glow::RGB, glow::FLOAT),
                                DynamicImage::ImageRgba32F(_) => (glow::RGBA, glow::FLOAT),
                                _ => (glow::RED, glow::UNSIGNED_BYTE), //probably wrong but will not read out of bounds
                            };
                            gl.tex_sub_image_3d(
                                glow::TEXTURE_2D_ARRAY,
                                0,
                                0,
                                0,
                                layer as i32,
                                width as i32,
                                height as i32,
                                1,
                                format,
                                type_,
                                glow::PixelUnpackData::Slice(bytes),
                            );
                        } else {
                            let blank = blank.get_or_insert_with(|| {
                                vec![0x80u8; 3 * width as usize * height as usize]
                            });
                            gl.tex_sub_image_3d(
                                glow::TEXTURE_2D_ARRAY,
                                0,
                                0,
                                0,
                                layer as i32,
                                width as i32,
                                height as i32,
                                1,
                                glow::RGB,
                                glow::UNSIGNED_BYTE,
                                glow::PixelUnpackData::Slice(blank),
                            );
                        }
                    }
                    gl.generate_mipmap(glow::TEXTURE_2D_ARRAY);
                    Some(textures)
                }
            }
        };
        let mut vertices = Vec::with_capacity(3 * model.num_faces());
        for (_, face) in model.faces() {
            for i_v in face.index_vertices() {
                let v = &model[i_v];
                vertices.push(MVertex3D {
                    pos_3d: v.pos(),
                    normal: v.normal(),
                    uv: v.uv(),
                    mat: face.material(),
                });
            }
        }
        let mut edges = Vec::new();
        let mut edge_status = Vec::new();
        let mut edge_map = FxHashMap::default();
        for (i_edge, edge) in model.edges() {
            let status = papercraft.edge_status(i_edge);
            if status == EdgeStatus::Hidden {
                continue;
            }
            let (p0, p1) = model.edge_pos(edge);
            let line = p1 - p0;
            let vs = [
                MVertex3DLine {
                    pos_3d: p0,
                    pos_b: p1,
                },
                MVertex3DLine {
                    pos_3d: p0,
                    pos_b: p0 - line,
                },
                MVertex3DLine {
                    pos_3d: p1,
                    pos_b: p1 + line,
                },
                MVertex3DLine {
                    pos_3d: p1,
                    pos_b: p0,
                },
            ];
            edge_map.insert(i_edge, edges.len());
            edges.extend_from_slice(&[vs[0], vs[1], vs[2], vs[2], vs[1], vs[3]]);
            edge_status.extend_from_slice(&[MLINE3D_HIDDEN; 6]);
        }

        let vertices = glr::DynamicVertexArray::from_data(gl, vertices)?;
        let vertices_sel =
            glr::DynamicVertexArray::from_data(gl, vec![MSTATUS_UNSEL; 3 * model.num_faces()])?;

        let scene_vertices_edge = glr::DynamicVertexArray::from_data(gl, edges)?;
        let scene_vertices_edge_status = glr::DynamicVertexArray::from_data(gl, edge_status)?;

        let paper_vertices = glr::DynamicVertexArray::new(gl)?;
        let paper_vertices_edge_cut = glr::DynamicVertexArray::new(gl)?;
        let paper_vertices_edge_crease = glr::DynamicVertexArray::new(gl)?;
        let paper_vertices_flap = glr::DynamicVertexArray::new(gl)?;
        let paper_vertices_flap_edge = glr::DynamicVertexArray::new(gl)?;
        let paper_vertices_edge_sel = glr::DynamicVertexArray::new(gl)?;
        let paper_vertices_shadow_flap = glr::DynamicVertexArray::new(gl)?;

        let paper_vertices_page = glr::DynamicVertexArray::new(gl)?;
        let paper_vertices_margin = glr::DynamicVertexArray::new(gl)?;

        let paper_text = glr::DynamicVertexArray::new(gl)?;

        Ok(GLObjects {
            textures,

            vertices,
            vertices_sel,

            scene_vertices_edge,
            scene_vertices_edge_status,
            edge_map,

            paper_vertices,
            paper_vertices_edge_cut,
            paper_vertices_edge_crease,
            paper_vertices_flap,
            paper_vertices_flap_edge,
            paper_vertices_edge_sel,
            paper_vertices_shadow_flap,

            paper_vertices_page,
            paper_vertices_margin,
            paper_text,
        })
    }
}

pub fn signature() -> String {
    tr!("Created with Papercraft. https://github.com/rodrigorc/papercraft")
}

enum FlapVertices {
    Tri([Vector2; 3]),
    Quad([Vector2; 6]),
}

impl std::ops::Deref for FlapVertices {
    type Target = [Vector2];

    fn deref(&self) -> &Self::Target {
        match self {
            FlapVertices::Tri(s) => s,
            FlapVertices::Quad(s) => s,
        }
    }
}
impl std::ops::DerefMut for FlapVertices {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            FlapVertices::Tri(s) => s,
            FlapVertices::Quad(s) => s,
        }
    }
}

struct Line2D {
    pub p0: Vector2,
    pub p1: Vector2,
    pub dash0: f32,
    pub dash1: f32,
    pub width_left: f32,
    pub width_right: f32,
}

impl Line2D {
    fn set_dash(&mut self, size: f32) {
        if size == 0.0 {
            self.dash1 = self.dash0;
        } else {
            let line_dash = (self.p0.distance(self.p1) / size).round() + 0.5;
            self.dash1 = self.dash0 + line_dash;
        }
    }
}
// Given a list of lines build a triangle-strip geometry
fn build_vertices_for_lines_2d(vs: &mut Vec<MVertex2DLine>, lines: &[Line2D], color: Rgba) {
    // 2 tris per line
    vs.clear();
    vs.reserve(6 * lines.len());

    for line in lines {
        let mut p0 = line.p0;
        let mut p1 = line.p1;
        let mut v = (p1 - p0).normalize();
        let n = Vector2::new(v.y, -v.x);
        v *= (line.width_right + line.width_left) / 2.0;
        p0 -= v;
        p1 += v;

        let v0 = MVertex2DLine {
            pos_2d: p0 - n * line.width_left,
            color,
            line_dash: line.dash0,
        };
        let v1 = MVertex2DLine {
            pos_2d: p0 + n * line.width_right,
            color,
            line_dash: line.dash0,
        };
        let v2 = MVertex2DLine {
            pos_2d: p1 + n * line.width_right,
            color,
            line_dash: line.dash1,
        };
        let v3 = MVertex2DLine {
            pos_2d: p1 - n * line.width_left,
            color,
            line_dash: line.dash1,
        };
        vs.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
    }
}

use cgmath::{
    prelude::*,
    Deg, Rad,
};
use glium::{
    draw_parameters::PolygonOffset,
};
use gtk::{
    prelude::*,
    gdk::{self, EventMask},
};

use std::{collections::HashMap, cell::Cell, ops::ControlFlow};
use std::rc::Rc;
use std::cell::RefCell;

mod waveobj;
mod paper;
mod util_3d;
mod util_gl;

use paper::Papercraft;

use util_3d::{Matrix3, Matrix4, Quaternion, Vector2, Point2, Point3, Vector3};
use util_gl::{GdkGliumBackend, Uniforms2D, Uniforms3D, MVertex3D, MVertex2D, MVertexQuad, DynamicVertexBuffer, MStatus, MSTATUS_UNSEL, MSTATUS_SEL, MSTATUS_HI, MVertex3DLine};

use crate::util_3d::Matrix2;

fn main() {
    std::env::set_var("GTK_CSD", "0");
    gtk::init().expect("gtk::init");

    let f = std::fs::File::open("pikachu.obj").unwrap();
    let f = std::io::BufReader::new(f);
    let (matlibs, models) = waveobj::Model::from_reader(f).unwrap();

    // For now read only the first model from the file
    let obj = models.get(0).unwrap();
    let material = obj.material().map(String::from);
    let mut texture_images = HashMap::new();
    texture_images.insert(String::new(), None);

    // Other textures are read from the .mtl file
    for lib in matlibs {
        let f = std::fs::File::open(lib).unwrap();
        let f = std::io::BufReader::new(f);

        for lib in waveobj::Material::from_reader(f).unwrap()  {
            if let Some(map) = lib.map() {
                let pbl = gdk_pixbuf::PixbufLoader::new();
                let data = std::fs::read(dbg!(map)).unwrap();
                pbl.write(&data).ok().unwrap();
                pbl.close().ok().unwrap();
                let img = pbl.pixbuf().unwrap();
                //dbg!(img.width(), img.height(), img.rowstride(), img.bits_per_sample(), img.n_channels());
                texture_images.insert(lib.name().to_owned(), Some(img));
                //textures.insert(name, tex);
            }
        }
    }

    let (mut model, facemap) = paper::Model::from_waveobj(obj);

    // Compute the bounding box, then move to the center and scale to a standard size
    let (v_min, v_max) = util_3d::bounding_box_3d(
        model
            .vertices()
            .map(|v| v.pos())
    );
    let size = (v_max.x - v_min.x).max(v_max.y - v_min.y).max(v_max.z - v_min.z);
    let mscale = Matrix4::from_scale(1.0 / size);
    let center = (v_min + v_max) / 2.0;
    let mcenter = Matrix4::from_translation(-center);
    let m = mscale * mcenter;

    model.transform_vertices(|pos, _normal| {
        //only scale and translate, no need to touch normals
        *pos = m.transform_point(Point3::from_vec(*pos)).to_vec();
    });

    let persp = cgmath::perspective(Deg(60.0), 1.0, 1.0, 100.0);
    let trans_scene = Transformation3D::new(
        Vector3::new(0.0, 0.0, -30.0),
        Quaternion::one(),
         20.0,
         persp
    );
    let trans_paper = {
        let mt = Matrix3::from_translation(Vector2::new(-1.0, -1.5));
        let ms = Matrix3::from_scale(200.0);
        TransformationPaper {
            ortho: util_3d::ortho2d(1.0, 1.0),
            //mx: mt * ms * mr,
            mx: ms * mt,
        }
    };

    let _papercraft = Papercraft::new(model, &facemap);
    let papercraft: Papercraft = {
        let f = std::fs::File::open("a.json").unwrap();
        let f = std::io::BufReader::new(f);
        serde_json::from_reader(f).unwrap()
    };

    let wscene = gtk::GLArea::new();
    let wpaper = gtk::GLArea::new();

    let ctx = MyContext {
        wscene: wscene.clone(),
        wpaper: wpaper.clone(),
        gl_scene: None,
        gl_paper: None,
        gl_paper_size: Rc::new(Cell::new((1,1))),
        gl_objs: None,

        papercraft,
        texture_images,
        material,
        selected_face: None,
        selected_edge: None,
        grabbed_island: None,

        last_cursor_pos: (0.0, 0.0),

        trans_scene,
        trans_paper,
    };

    let ctx: Rc<RefCell<MyContext>> = Rc::new(RefCell::new(ctx));

    let w = gtk::Window::new(gtk::WindowType::Toplevel);
    w.set_default_size(800, 600);
    w.connect_destroy(move |_| {
        gtk::main_quit();
    });
    gl_loader::init_gl();

    wscene.set_events(EventMask::BUTTON_PRESS_MASK | EventMask::BUTTON_MOTION_MASK | EventMask::POINTER_MOTION_MASK | EventMask::SCROLL_MASK);
    wscene.set_has_depth_buffer(true);

    wscene.connect_button_press_event({
        let ctx = ctx.clone();
        move |w, ev| {
            w.grab_focus();
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            ctx.last_cursor_pos = ev.position();

            if ev.button() == 1 && ev.event_type() == gdk::EventType::ButtonPress {
                let selection = ctx.analyze_click(ev.position());
                if let ClickResult::Edge(i_edge, priority_face) = selection {
                    ctx.papercraft.edge_toggle(i_edge, priority_face);
                    ctx.paper_build();
                    ctx.scene_edge_build();
                    ctx.wscene.queue_render();
                    ctx.wpaper.queue_render();
                }
             }
            Inhibit(false)
        }
    });
    wscene.connect_scroll_event({
        let ctx = ctx.clone();
        move |_w, ev|  {
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            let dz = match ev.direction() {
                gdk::ScrollDirection::Up => 1.1,
                gdk::ScrollDirection::Down => 1.0 / 1.1,
                _ => 1.0,
            };
            ctx.trans_scene.scale *= dz;
            ctx.trans_scene.recompute_obj();
            ctx.wscene.queue_render();
            Inhibit(true)
        }
    });
    wscene.connect_motion_notify_event({
        let ctx = ctx.clone();
        move |_w, ev|  {
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            let pos = ev.position();
            let dx = (pos.0 - ctx.last_cursor_pos.0)  as f32;
            let dy = (pos.1 - ctx.last_cursor_pos.1) as f32;
            ctx.last_cursor_pos = pos;

            if ev.state().contains(gdk::ModifierType::BUTTON3_MASK) {
                // half angles
                let ang_x = dx / 200.0 / 2.0;
                let ang_y = dy / 200.0 / 2.0;
                let cosy = ang_x.cos();
                let siny = ang_x.sin();
                let cosx = ang_y.cos();
                let sinx = ang_y.sin();
                let roty = Quaternion::new(cosy, 0.0, siny, 0.0);
                let rotx = Quaternion::new(cosx, sinx, 0.0, 0.0);

                ctx.trans_scene.rotation = (roty * rotx * ctx.trans_scene.rotation).normalize();
                ctx.trans_scene.recompute_obj();
                ctx.wscene.queue_render();
            } else if ev.state().contains(gdk::ModifierType::BUTTON2_MASK) {
                let dx = dx / 50.0;
                let dy = -dy / 50.0;

                ctx.trans_scene.location += Vector3::new(dx, dy, 0.0);
                ctx.trans_scene.recompute_obj();
                ctx.wscene.queue_render();
            } else {
                let selection = ctx.analyze_click(ev.position());
                ctx.set_selection(selection);
            }
            Inhibit(true)
        }
    });
    wscene.connect_realize({
        let ctx = ctx.clone();
        move |w| scene_realize(w, &mut *ctx.borrow_mut())
    });
    wscene.connect_unrealize({
        let ctx = ctx.clone();
        move |_w| {
            ctx.borrow_mut().gl_scene = None;
        }
    });
    wscene.connect_render({
        let ctx = ctx.clone();
        move |_w, _gl| {
            ctx.borrow().scene_render();
            gtk::Inhibit(false)
        }
    });
    wscene.connect_resize({
        let ctx = ctx.clone();
        move |_w, width, height| {
            if height <= 0 || width <= 0 {
                return;
            }
            let ratio = width as f32 / height as f32;
            ctx.borrow_mut().trans_scene.set_ratio(ratio);
        }
    });


    wpaper.set_events(EventMask::BUTTON_PRESS_MASK | EventMask::BUTTON_MOTION_MASK | EventMask::POINTER_MOTION_MASK | EventMask::SCROLL_MASK);
    wpaper.set_has_stencil_buffer(true);
    wpaper.connect_realize({
        let ctx = ctx.clone();
        move |w| paper_realize(w, &mut *ctx.borrow_mut())
    });
    wscene.connect_unrealize({
        let ctx = ctx.clone();
        move |_w| {
            ctx.borrow_mut().gl_paper = None;
        }
    });
    wpaper.connect_render({
        let ctx = ctx.clone();
        move |_w, _gl| {
            ctx.borrow().paper_render();
            Inhibit(true)
        }
    });
    wpaper.connect_resize({
        let ctx = ctx.clone();
        move |_w, width, height| {
            if height <= 0 || width <= 0 {
                return;
            }
            ctx.borrow_mut().trans_paper.ortho = util_3d::ortho2d(width as f32, height as f32);
        }
    });

    wpaper.connect_button_press_event({
        let ctx = ctx.clone();
        move |w, ev|  {
            w.grab_focus();
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            ctx.last_cursor_pos = ev.position();
            if ev.button() == 1 && ev.event_type() == gdk::EventType::ButtonPress {
                let selection = ctx.analyze_click_paper(ev.position());
                if let ClickResult::Face(i_face) = selection {
                    let i_island = ctx.papercraft.island_by_face(i_face);
                    ctx.grabbed_island = Some(i_island);
                } else {
                    ctx.grabbed_island = None;
                    if let ClickResult::Edge(i_edge, priority_face) = selection {
                        ctx.papercraft.edge_toggle(i_edge, priority_face);
                        ctx.paper_build();
                        ctx.scene_edge_build();
                        ctx.wscene.queue_render();
                        ctx.wpaper.queue_render();
                    }
                }
            }
            Inhibit(true)
        }
    });
    wpaper.connect_motion_notify_event({
        let ctx = ctx.clone();
        move |_w, ev| {
            let mut ctx = ctx.borrow_mut();
            let pos = ev.position();
            let delta = Vector2::new((pos.0 - ctx.last_cursor_pos.0)  as f32,(pos.1 - ctx.last_cursor_pos.1) as f32);
            ctx.last_cursor_pos = pos;
            if ev.state().contains(gdk::ModifierType::BUTTON2_MASK) {
                ctx.trans_paper.mx = Matrix3::from_translation(delta) * ctx.trans_paper.mx;
                ctx.wpaper.queue_render();
            } else if ev.state().contains(gdk::ModifierType::BUTTON1_MASK) {
                if let Some(i_island) = ctx.grabbed_island {
                    let delta_scaled = <Matrix3 as Transform<Point2>>::inverse_transform_vector(&ctx.trans_paper.mx, delta).unwrap();
                    if let Some(island) = ctx.papercraft.island_by_key_mut(i_island) {
                        if ev.state().contains(gdk::ModifierType::SHIFT_MASK) {
                            // Rotate island
                            island.rotate(Deg(delta.y));
                        } else {
                            // Move island
                            island.translate(delta_scaled);
                        }
                        ctx.paper_build();
                        ctx.wpaper.queue_render();
                    }
                }
            } else {
                let selection = ctx.analyze_click_paper(ev.position());
                ctx.set_selection(selection);
            }
            Inhibit(true)
        }
    });
    wpaper.connect_scroll_event({
        let ctx = ctx.clone();
        move |_w, ev|  {
            let mut ctx = ctx.borrow_mut();
            let dz = match ev.direction() {
                gdk::ScrollDirection::Up => 1.1,
                gdk::ScrollDirection::Down => 1.0 / 1.1,
                _ => 1.0,
            };
            ctx.trans_paper.mx = Matrix3::from_scale(dz) * ctx.trans_paper.mx;
            ctx.wpaper.queue_render();
            Inhibit(true)
        }
    });

    let hbin = gtk::Paned::new(gtk::Orientation::Horizontal);
    hbin.pack1(&wscene, true, true);
    hbin.pack2(&wpaper, true, true);
    w.add(&hbin);

    w.show_all();
    gtk::main();

    let ctx = ctx.borrow();
    {
        let f = std::fs::File::create("a.json").unwrap();
        let f = std::io::BufWriter::new(f);
        serde_json::to_writer(f, &ctx.papercraft).unwrap()
    };
}

fn scene_realize(w: &gtk::GLArea, ctx: &mut MyContext) {
    w.attach_buffers();
    let backend = GdkGliumBackend::new(w.context().unwrap(), Rc::new(Cell::new((1,1))));
    let glctx = unsafe { glium::backend::Context::new(backend, false, glium::debug::DebugCallbackBehavior::Ignore).unwrap() };
    ctx.build_gl_objects(&glctx);
    ctx.gl_scene = Some(glctx);
}

fn paper_realize(w: &gtk::GLArea, ctx: &mut MyContext) {
    w.attach_buffers();
    let backend = GdkGliumBackend::new(w.context().unwrap(), Rc::clone(&ctx.gl_paper_size));
    let glctx = unsafe { glium::backend::Context::new(backend, false, glium::debug::DebugCallbackBehavior::Ignore).unwrap() };
    ctx.build_gl_objects(&glctx);
    ctx.gl_paper = Some(glctx);
}

struct GLObjects {
    prg_scene_solid: glium::Program,
    prg_scene_line: glium::Program,
    prg_paper_solid: glium::Program,
    prg_paper_line: glium::Program,
    #[allow(dead_code)]
    prg_quad: glium::Program,

    textures: HashMap<String, glium::Texture2d>,

    vertex_buf: glium::VertexBuffer<MVertex3D>,
    vertex_buf_sel: glium::VertexBuffer<MStatus>,
    vertex_edges_buf: DynamicVertexBuffer<MVertex3DLine>,

    paper_vertex_buf: DynamicVertexBuffer<MVertex2D>,
    paper_vertex_edge_buf: DynamicVertexBuffer<MVertex2D>,
    paper_vertex_edge_sel_buf: DynamicVertexBuffer<MVertex2D>,
    paper_vertex_tab_buf: DynamicVertexBuffer<MVertex2D>,
    paper_vertex_tab_edge_buf: DynamicVertexBuffer<MVertex2D>,

    #[allow(dead_code)]
    quad_vertex_buf: glium::VertexBuffer<MVertexQuad>,
}

// This contains GL objects that are object specific
struct MyContext {
    wscene: gtk::GLArea,
    wpaper: gtk::GLArea,

    gl_scene: Option<Rc<glium::backend::Context>>,
    gl_paper: Option<Rc<glium::backend::Context>>,
    gl_paper_size: Rc<Cell<(u32, u32)>>,
    gl_objs: Option<GLObjects>,

    // The model
    papercraft: Papercraft,
    texture_images: HashMap<String, Option<gdk_pixbuf::Pixbuf>>,

    // State
    material: Option<String>,
    selected_face: Option<paper::FaceIndex>,
    selected_edge: Option<paper::EdgeIndex>,
    grabbed_island: Option<paper::IslandKey>,

    last_cursor_pos: (f64, f64),


    trans_scene: Transformation3D,
    trans_paper: TransformationPaper,
}

struct Transformation3D {
    location: Vector3,
    rotation: Quaternion,
    scale: f32,

    persp: Matrix4,
    persp_inv: Matrix4,
    obj: Matrix4,
    obj_inv: Matrix4,
    mnormal: Matrix3,
}

impl Transformation3D {
    fn new(location: Vector3, rotation: Quaternion, scale: f32, persp: Matrix4) -> Transformation3D {
        let mut tr = Transformation3D {
            location,
            rotation,
            scale,
            persp,
            persp_inv: persp.invert().unwrap(),
            obj: Matrix4::one(),
            obj_inv: Matrix4::one(),
            mnormal: Matrix3::one(),
        };
        tr.recompute_obj();
        tr
    }
    fn recompute_obj(&mut self) {
        let r = Matrix3::from(self.rotation);
        let t = Matrix4::from_translation(self.location);
        let s = Matrix4::from_scale(self.scale);

        self.obj = t * Matrix4::from(r) * s;
        self.obj_inv = self.obj.invert().unwrap();
        self.mnormal = r; //should be inverse of transpose
    }

    fn set_ratio(&mut self, ratio: f32) {
        let f = self.persp[1][1];
        self.persp[0][0] = f / ratio;
        self.persp_inv = self.persp.invert().unwrap();
    }
}

struct TransformationPaper {
    ortho: Matrix3,
    mx: Matrix3,
}

#[derive(Debug)]
enum ClickResult {
    None,
    Face(paper::FaceIndex),
    Edge(paper::EdgeIndex, Option<paper::FaceIndex>),
}

#[derive(Default)]
struct PaperDrawFaceArgs {
    vertices: Vec<MVertex2D>,
    vertices_edge: Vec<MVertex2D>,
    vertices_edge_sel: Vec<MVertex2D>,
    vertices_tab_buf: Vec<MVertex2D>,
    vertices_tab_edge_buf: Vec<MVertex2D>,
}

impl MyContext {
    fn analyze_click(&self, (x, y): (f64, f64)) -> ClickResult {
        let rect = self.wscene.allocation();
        let x = (x as f32 / rect.width() as f32) * 2.0 - 1.0;
        let y = -((y as f32 / rect.height() as f32) * 2.0 - 1.0);
        let click = Point3::new(x as f32, y as f32, 1.0);
        let height = rect.height() as f32;


        let click_camera = self.trans_scene.persp_inv.transform_point(click);
        let click_obj = self.trans_scene.obj_inv.transform_point(click_camera);
        let camera_obj = self.trans_scene.obj_inv.transform_point(Point3::new(0.0, 0.0, 0.0));

        let ray = (camera_obj.to_vec(), click_obj.to_vec());

        let mut hit_face = None;
        for (iface, face) in self.papercraft.model().faces() {
            let tri = face.index_vertices().map(|v| self.papercraft.model()[v].pos());
            let maybe_new_hit = util_3d::ray_crosses_face(ray, &tri);
            if let Some(new_hit) = maybe_new_hit {
                hit_face = match (hit_face, new_hit) {
                    (Some((_, p)), x) if p > x && x > 0.0 => Some((iface, x)),
                    (None, x) if x > 0.0 => Some((iface, x)),
                    (old, _) => old
                };
            }
        }

        let mut hit_edge = None;
        for (i_edge, edge) in self.papercraft.model().edges() {
            if self.papercraft.edge_status(i_edge) == paper::EdgeStatus::Hidden {
                continue;
            }
            let v1 = self.papercraft.model()[edge.v0()].pos();
            let v2 = self.papercraft.model()[edge.v1()].pos();
            let (ray_hit, _line_hit, new_dist) = util_3d::line_segment_distance(ray, (v1, v2));

            // Behind the screen, it is not a hit
            if ray_hit <= 0.0001 {
                continue;
            }

            // new_dist is originally the distance in real-world space, but the user is using the screen, so scale accordingly
            let new_dist = new_dist / ray_hit * height;

            // If this egde is from the ray further that the best one, it is worse and ignored
            match hit_edge {
                Some((_, _, p)) if p < new_dist => { continue; }
                _ => {}
            }

            // Too far from the edge
            if new_dist > 0.1 {
                continue;
            }

            // If there is a face 99% nearer this edge, it is hidden, probably, so it does not count
            match hit_face {
                Some((_, p)) if p < 0.99 * ray_hit => { continue; }
                _ => {}
            }

            hit_edge = Some((i_edge, ray_hit, new_dist));
        }

        match (hit_face, hit_edge) {
            (_, Some((e, _, _))) => ClickResult::Edge(e, None),
            (Some((f, _)), None) => ClickResult::Face(f),
            (None, None) => ClickResult::None,
        }
    }

    fn analyze_click_paper(&self, (x, y): (f64, f64)) -> ClickResult {
        let rect = self.wpaper.allocation();
        let x = (x as f32 / rect.width() as f32) * 2.0 - 1.0;
        let y = -((y as f32 / rect.height() as f32) * 2.0 - 1.0);
        let click = Point2::new(x as f32, y as f32);

        let mx = self.trans_paper.ortho * self.trans_paper.mx;
        let mx_inv = mx.invert().unwrap();
        let click = mx_inv.transform_point(click).to_vec();

        let mut edge_sel = None;
        let mut face_sel = None;

        for (_i_island, island) in self.papercraft.islands() {
            self.papercraft.traverse_faces(island,
                |i_face, face, fmx| {
                    let normal = face.plane(self.papercraft.model());
                    let tri = face.index_vertices();
                    let tri = tri.map(|v| {
                        let v3 = self.papercraft.model()[v].pos();
                        let v2 = normal.project(&v3);
                        fmx.transform_point(Point2::from_vec(v2)).to_vec()
                    });
                    if face_sel.is_none() && util_3d::point_in_triangle(click, tri[0], tri[1], tri[2]) {
                        face_sel = Some(i_face);
                    }

                    for i_edge in face.index_edges() {
                        if self.papercraft.edge_status(i_edge) == paper::EdgeStatus::Hidden {
                            continue;
                        }
                        let edge = &self.papercraft.model()[i_edge];
                        let v0 = self.papercraft.model()[edge.v0()].pos();
                        let v0 = normal.project(&v0);
                        let v0 = fmx.transform_point(Point2::from_vec(v0)).to_vec();
                        let v1 = self.papercraft.model()[edge.v1()].pos();
                        let v1 = normal.project(&v1);
                        let v1 = fmx.transform_point(Point2::from_vec(v1)).to_vec();

                        let (_o, d) = util_3d::point_segment_distance(click, (v0, v1));
                        let d = <Matrix3 as Transform<Point2>>::transform_vector(&mx, Vector2::new(d, 0.0)).magnitude();
                        if d > 0.02 { //too far?
                            continue;
                        }
                        match &edge_sel {
                            None => {
                                edge_sel = Some((d, i_edge, i_face));
                            }
                            &Some((d_prev, _, _)) if d < d_prev => {
                                edge_sel = Some((d, i_edge, i_face));
                            }
                            _ => {}
                        }
                    }
                    ControlFlow::Continue(())
                }
            );
        }
        //Edge selection has priority
        match (edge_sel, face_sel) {
            (Some((_d, i_edge, i_face)), _) => ClickResult::Edge(i_edge, Some(i_face)),
            (None, Some(i_face)) => ClickResult::Face(i_face),
            (None, None) => ClickResult::None,
        }
    }
    fn set_selection(&mut self, selection: ClickResult) {
        match selection {
            ClickResult::None => {
                let is_same = self.selected_edge.is_none() && self.selected_face.is_none();
                if is_same {
                    return;
                }
                self.selected_edge = None;
                self.selected_face = None;

                self.clear_scene_face_selection();
            }
            ClickResult::Face(i_face) => {
                let is_same = self.selected_edge.is_none() && self.selected_face == Some(i_face);
                if is_same {
                    return;
                }
                self.selected_edge = None;
                self.selected_face = Some(i_face);

                self.clear_scene_face_selection();
                if let Some(gl_objs) = &mut self.gl_objs {
                    //let mut idxs = Vec::new();
                    let island = self.papercraft.island_by_face(i_face);
                    let island = self.papercraft.island_by_key(island).unwrap();

                    let mut vertex_buf_sel = gl_objs.vertex_buf_sel.as_mut_slice().map_write();
                    self.papercraft.traverse_faces(island, |i_face_2, _, _,| {
                        let pos = 3 * usize::from(i_face_2);
                        for i in pos .. pos + 3 {
                            vertex_buf_sel.set(i, MSTATUS_SEL);
                        }
                        ControlFlow::Continue(())
                    });
                    self.papercraft.traverse_faces_flat(i_face, |i_face_2, _, _,| {
                        let pos = 3 * usize::from(i_face_2);
                        for i in pos .. pos + 3 {
                            vertex_buf_sel.set(i, MSTATUS_HI);
                        }
                        ControlFlow::Continue(())
                    });
                }
            }
            ClickResult::Edge(i_edge, _) => {
                let is_same = self.selected_edge == Some(i_edge) && self.selected_face.is_none();
                if is_same {
                    return;
                }
                self.selected_edge = Some(i_edge);
                self.selected_face = None;
                self.clear_scene_face_selection();
            }
        }
        self.paper_build();
        self.scene_edge_build();
        self.wscene.queue_render();
        self.wpaper.queue_render();
    }

    fn clear_scene_face_selection(&mut self) -> Option<()> {
        let gl_objs = &mut self.gl_objs.as_mut()?;
        let mut vertex_buf_sel = gl_objs.vertex_buf_sel.as_mut_slice().map_write();
        let n = vertex_buf_sel.len();
        for i in 0..n {
            vertex_buf_sel.set(i, MSTATUS_UNSEL);
        }
        Some(())
    }
    fn build_gl_objects(&mut self, gl: &Rc<glium::backend::Context>) {
        if self.gl_objs.is_some() {
            return;
        }
        let textures = self.texture_images
            .iter()
            .map(|(name, pixbuf)| {
                let texture = match pixbuf {
                    None => {
                        // Empty texture is just a single white texel
                        let empty = glium::Texture2d::empty(gl, 1, 1).unwrap();
                        empty.write(glium::Rect{ left: 0, bottom: 0, width: 1, height: 1 }, vec![vec![(0xffu8, 0xffu8, 0xffu8, 0xffu8)]]);
                        empty
                    }
                    Some(img) => {
                        let bytes = img.read_pixel_bytes().unwrap();
                        let raw =  glium::texture::RawImage2d {
                            data: std::borrow::Cow::Borrowed(&bytes),
                            width: img.width() as u32,
                            height: img.height() as u32,
                            format: match img.n_channels() {
                                4 => glium::texture::ClientFormat::U8U8U8U8,
                                3 => glium::texture::ClientFormat::U8U8U8,
                                2 => glium::texture::ClientFormat::U8U8,
                                _ => glium::texture::ClientFormat::U8,
                            },
                        };
                        //dbg!(img.width(), img.height(), img.rowstride(), img.bits_per_sample(), img.n_channels());
                        glium::Texture2d::new(gl,  raw).unwrap()
                    }
                };
                (name.clone(), texture)
            })
            .collect();

        let prg_scene_solid = util_gl::program_from_source(gl, include_str!("shaders/scene_solid.glsl"));
        let prg_scene_line = util_gl::program_from_source(gl, include_str!("shaders/scene_line.glsl"));
        let prg_paper_solid = util_gl::program_from_source(gl, include_str!("shaders/paper_solid.glsl"));
        let prg_paper_line = util_gl::program_from_source(gl, include_str!("shaders/paper_line.glsl"));
        let prg_quad = util_gl::program_from_source(gl, include_str!("shaders/quad.glsl"));

        let mut vertices = Vec::with_capacity(self.papercraft.model().num_faces() * 3);
        for (_, face) in self.papercraft.model().faces() {
            for i_v in face.index_vertices() {
                let v = &self.papercraft.model()[i_v];
                vertices.push(MVertex3D {
                    pos: v.pos(),
                    normal: v.normal(),
                    uv: v.uv(),
                });
            }
        }

        let vertex_buf = glium::VertexBuffer::immutable(gl, &vertices).unwrap(); // 1 value per vertex
        let vertex_buf_sel = glium::VertexBuffer::persistent(gl, &vec![MSTATUS_UNSEL; vertices.len()]).unwrap(); // 1 value per vertex
        let vertex_edges_buf = DynamicVertexBuffer::new(gl, self.papercraft.model().num_edges() * 2); // one line per edge

        let paper_vertex_buf = DynamicVertexBuffer::new(gl, self.papercraft.model().num_faces() * 3); // 1 tri per face
        let paper_vertex_edge_buf = DynamicVertexBuffer::new(gl, self.papercraft.model().num_edges() * 2 * 2); // 2 lines per cut edge
        let paper_vertex_edge_sel_buf = DynamicVertexBuffer::new(gl, 6); // 3 lines
        let paper_vertex_tab_buf = DynamicVertexBuffer::new(gl, self.papercraft.model().num_edges() * 2 * 3); // 2 tris per edge
        let paper_vertex_tab_edge_buf = DynamicVertexBuffer::new(gl, self.papercraft.model().num_edges() * 3 * 2); // 3 lines per edge

        let quad_vertex_buf = glium::VertexBuffer::immutable(gl,
            &[
                MVertexQuad { pos: [-1.0, -1.0] },
                MVertexQuad { pos: [ 3.0, -1.0] },
                MVertexQuad { pos: [-1.0,  3.0] },
            ]).unwrap();

        let gl_objs = GLObjects {
            prg_scene_solid,
            prg_scene_line,
            prg_paper_solid,
            prg_paper_line,
            prg_quad,
            textures,
            vertex_buf,
            vertex_buf_sel,
            vertex_edges_buf,
            paper_vertex_buf,
            paper_vertex_edge_buf,
            paper_vertex_edge_sel_buf,
            paper_vertex_tab_buf,
            paper_vertex_tab_edge_buf,

            quad_vertex_buf,
        };

        self.gl_objs = Some(gl_objs);

        self.scene_edge_build();
        self.paper_build();
    }

    fn paper_draw_face(&self, face: &paper::Face, i_face: paper::FaceIndex, m: &Matrix3, selected: bool, hi: bool, args: &mut PaperDrawFaceArgs) {
        for i_v in face.index_vertices() {
            let v = &self.papercraft.model()[i_v];
            let p = face.plane(self.papercraft.model()).project(&v.pos());
            let pos = m.transform_point(Point2::from_vec(p)).to_vec();
            args.vertices.push(MVertex2D {
                pos,
                uv: v.uv(),
                color: if hi { [1.0, 0.0, 0.0, 0.75] } else if selected { [0.0, 0.0, 1.0, 0.5] } else { [0.0, 0.0, 0.0, 0.0] },
            });
        }

        for (v0, v1, i_edge) in face.vertices_with_edges() {
            let edge = &self.papercraft.model()[i_edge];
            let edge_status = self.papercraft.edge_status(i_edge);
            let draw = match edge_status {
                paper::EdgeStatus::Hidden => false,
                paper::EdgeStatus::Cut => true,
                paper::EdgeStatus::Joined => edge.face_sign(i_face),
            };
            let plane = face.plane(self.papercraft.model());
            let selected_edge = self.selected_edge == Some(i_edge);
            let v0 = &self.papercraft.model()[v0];
            let p0 = plane.project(&v0.pos());
            let pos0 = m.transform_point(Point2::from_vec(p0)).to_vec();

            let v1 = &self.papercraft.model()[v1];
            let p1 = plane.project(&v1.pos());
            let pos1 = m.transform_point(Point2::from_vec(p1)).to_vec();

            if draw || selected_edge {
                let angle_3d = self.papercraft.model().edge_angle(i_edge);

                args.vertices_edge.push(MVertex2D {
                    pos: pos0,
                    uv: Vector2::zero(),
                    color: [0.0, 0.0, 0.0, 1.0],
                });
                args.vertices_edge.push(MVertex2D {
                    pos: pos1,
                    uv: Vector2::new(if angle_3d < Rad(0.0) { (pos1 - pos0).magnitude() * 100.0 } else { 0.0 }, 0.0),
                    color: [0.0, 0.0, 0.0, 1.0],
                });

                if selected_edge {
                    args.vertices_edge_sel.push(MVertex2D {
                        pos: pos0,
                        uv: Vector2::zero(),
                        color: [0.5, 0.5, 1.0, 1.0],
                    });
                    args.vertices_edge_sel.push(MVertex2D {
                        pos: pos1,
                        uv: Vector2::zero(),
                        color: [0.5, 0.5, 1.0, 1.0],
                    });
                    }
            }

            if edge_status == paper::EdgeStatus::Cut && edge.face_sign(i_face) {
                const TAB: f32 = 0.02;
                let v = pos1 - pos0;

                let v_len = v.magnitude();
                let short_len = v_len - 2.0 * TAB;
                let tab = if short_len < 0.0 {
                    v_len / 2.0
                } else {
                    TAB
                };
                let v = v * (tab / v_len);
                let n = Vector2::new(-v.y, v.x);
                let mut p = [

                    MVertex2D {
                        pos: pos0,
                        uv: Vector2::zero(),
                        color: [0.0, 0.0, 0.0, 1.0],
                    },
                    MVertex2D {
                        pos: pos0 + n + v,
                        uv: Vector2::zero(),
                        color: [0.0, 0.0, 0.0, 1.0],
                    },
                    MVertex2D {
                        pos: pos1 + n - v,
                        uv: Vector2::zero(),
                        color: [0.0, 0.0, 0.0, 1.0],
                    },
                    MVertex2D {
                        pos: pos1,
                        uv: Vector2::zero(),
                        color: [0.0, 0.0, 0.0, 1.0],
                    },
                ];
                args.vertices_tab_edge_buf.extend([p[0], p[1], p[1], p[2], p[2], p[3]]);

                //Now we have to compute the texture coordinates of `p` in the adjacent face
                let i_face_b = edge.faces().filter(|f| *f != i_face).next().unwrap();
                let face_b = &self.papercraft.model()[i_face_b];
                let plane_b = face_b.plane(self.papercraft.model());
                let vs_b = face_b.index_vertices().map(|v| {
                    let v = &self.papercraft.model()[v];
                    let p = plane_b.project(&v.pos());
                    (v, p)
                });
                let mx_b = m * self.papercraft.model().face_to_face_edge_matrix(edge, face, face_b);
                let mx_b_inv = mx_b.invert().unwrap();
                let mx_basis = Matrix2::from_cols(vs_b[1].1 - vs_b[0].1, vs_b[2].1 - vs_b[0].1).invert().unwrap();

                // mx_b_inv converts from paper to local face_b coordinates
                // mx_basis converts from local face_b to edge-relative coordinates, where position of the tri vertices are [(0,0), (1,0), (0,1)]
                // mxx do both convertions at once
                let mxx = Matrix3::from(mx_basis) * mx_b_inv;
                let uv_b = p.map(|px| {
                    //vlocal is in edge-relative coordinates, that can be used to interpolate between UVs
                    let vlocal = mxx.transform_point(Point2::from_vec(px.pos)).to_vec();
                    let uv0 = vs_b[0].0.uv();
                    let uv1 = vs_b[1].0.uv();
                    let uv2 = vs_b[2].0.uv();
                    uv0 + vlocal.x * (uv1 - uv0) + vlocal.y * (uv2 - uv0)
                });

                p[0].uv = uv_b[0];
                p[0].color = [1.0, 1.0, 1.0, 0.0];
                p[1].uv = uv_b[1];
                p[1].color = [1.0, 1.0, 1.0, 1.0];
                p[2].uv = uv_b[2];
                p[2].color = [1.0, 1.0, 1.0, 1.0];
                p[3].uv = uv_b[3];
                p[3].color = [1.0, 1.0, 1.0, 0.0];
                args.vertices_tab_buf.extend([p[0], p[2], p[1], p[0], p[3], p[2]]);
            }
        }
    }

    fn paper_build(&mut self) {
        //Maps VertexIndex in the model to index in vertices
        let mut args = PaperDrawFaceArgs::default();

        for (_, island) in self.papercraft.islands() {
            let selected = if let Some(sel) = self.selected_face {
                self.papercraft.contains_face(island, sel)
            } else {
                false
            };
            self.papercraft.traverse_faces(island,
                |i_face, face, mx| {
                    let hi = if let Some(sel) = self.selected_face {
                        self.papercraft.are_flat_faces(i_face, sel)
                    } else {
                        false
                    };
                    self.paper_draw_face(face, i_face, mx, selected, hi, &mut args);
                    ControlFlow::Continue(())
                }
            );
        }

        if args.vertices_edge_sel.len() == 4 {
            for i in 0 .. 2 {
                let v0 = &args.vertices_edge_sel[2*i];
                let v1 = &args.vertices_edge_sel[2*i+1];
                let vm = MVertex2D {
                    pos: (v0.pos + v1.pos) / 2.0,
                    .. *v0
                };
                args.vertices_edge_sel.push(vm);
            }
        }

        if let Some(gl_objs) = &mut self.gl_objs {
            gl_objs.paper_vertex_buf.update(&args.vertices);
            gl_objs.paper_vertex_edge_buf.update(&args.vertices_edge);
            gl_objs.paper_vertex_edge_sel_buf.update(&args.vertices_edge_sel);
            gl_objs.paper_vertex_tab_buf.update(&args.vertices_tab_buf);
            gl_objs.paper_vertex_tab_edge_buf.update(&args.vertices_tab_edge_buf);
        }
    }

    fn scene_render(&self) {
        let rect = self.wscene.allocation();

        let mut frm = glium::Frame::new(Rc::clone(self.gl_scene.as_ref().unwrap()), (rect.width() as u32, rect.height() as u32));
        let gl_objs = self.gl_objs.as_ref().unwrap();

        use glium::Surface;

        frm.clear_color_and_depth((0.2, 0.2, 0.4, 1.0), 1.0);

        let light0 = Vector3::new(-0.5, -0.4, -0.8).normalize() * 0.55;
        let light1 = Vector3::new(0.8, 0.2, 0.4).normalize() * 0.25;

        let mat_name = self.material.as_deref().unwrap_or("");
        let texture = gl_objs.textures.get(mat_name)
            .unwrap_or_else(|| gl_objs.textures.get("").unwrap());

        let u = Uniforms3D {
            m: self.trans_scene.persp * self.trans_scene.obj,
            mnormal: self.trans_scene.mnormal, // should be transpose of inverse
            lights: [light0, light1],
            texture: texture.sampled(),
        };

        let mut dp = glium::DrawParameters {
            viewport: Some(glium::Rect { left: 0, bottom: 0, width: rect.width() as u32, height: rect.height() as u32}),
            blend: glium::Blend::alpha_blending(),
            depth: glium::Depth {
                test: glium::DepthTest::IfLessOrEqual,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };

        // Draw the textured polys
        dp.polygon_offset = PolygonOffset {
            line: true,
            fill: true,
            factor: 1.0,
            units: 1.0,
            .. PolygonOffset::default()
        };
        frm.draw((&gl_objs.vertex_buf, &gl_objs.vertex_buf_sel), &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList), &gl_objs.prg_scene_solid, &u, &dp).unwrap();

        // Draw the lines:
        dp.line_width = Some(3.0);
        dp.smooth = Some(glium::Smooth::Nicest);
        frm.draw(&gl_objs.vertex_edges_buf, &glium::index::NoIndices(glium::index::PrimitiveType::LinesList), &gl_objs.prg_scene_line, &u, &dp).unwrap();

        frm.finish().unwrap();
    }

    fn paper_render(&self) {
        let rect = self.wpaper.allocation();
        use glium::Surface;

        let mut frm = glium::Frame::new(Rc::clone(self.gl_paper.as_ref().unwrap()), (rect.width() as u32, rect.height() as u32));
        let gl_objs = self.gl_objs.as_ref().unwrap();

        frm.clear_all((0.7, 0.7, 0.7, 1.0), 1.0, 0);

        let mat_name = self.material.as_deref().unwrap_or("");
        let texture = gl_objs.textures.get(mat_name)
            .unwrap_or_else(|| gl_objs.textures.get("").unwrap());

        let mut u = Uniforms2D {
            m: self.trans_paper.ortho * self.trans_paper.mx,
            texture: texture.sampled(),
            frac_dash: 0.5,
        };

        let mut dp = glium::DrawParameters {
            viewport: Some(glium::Rect { left: 0, bottom: 0, width: rect.width() as u32, height: rect.height() as u32}),
            blend: glium::Blend::alpha_blending(),
            stencil: glium::draw_parameters::Stencil {
                depth_pass_operation_counter_clockwise: glium::StencilOperation::Increment,
                .. Default::default()
            },
            .. Default::default()
        };

        // Textured faces
        frm.draw(&gl_objs.paper_vertex_buf, &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList), &gl_objs.prg_paper_solid, &u, &dp).unwrap();

        // Solid Tabs
        frm.draw(&gl_objs.paper_vertex_tab_buf, &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList), &gl_objs.prg_paper_solid, &u, &dp).unwrap();

        // Line Tabs
        dp.line_width = Some(1.0);
        dp.smooth = Some(glium::Smooth::Nicest);
        frm.draw(&gl_objs.paper_vertex_tab_edge_buf, &glium::index::NoIndices(glium::index::PrimitiveType::LinesList), &gl_objs.prg_paper_line, &u, &dp).unwrap();

        // Creases
        dp.stencil.depth_pass_operation_counter_clockwise = glium::StencilOperation::Keep;
        frm.draw(&gl_objs.paper_vertex_edge_buf, &glium::index::NoIndices(glium::index::PrimitiveType::LinesList), &gl_objs.prg_paper_line, &u, &dp).unwrap();

        // - Selected edge
        if self.selected_edge.is_some() {
            dp.line_width = Some(5.0);
            u.frac_dash = 0.5;
            frm.draw(&gl_objs.paper_vertex_edge_sel_buf, &glium::index::NoIndices(glium::index::PrimitiveType::LinesList), &gl_objs.prg_paper_line, &u, &dp).unwrap();

        }
        // Overlaps
        #[cfg(xxx)]
        {
            u.color = [1.0, 1.0, 1.0, 0.75];
            dp.stencil = glium::draw_parameters::Stencil {
                test_counter_clockwise: glium::StencilTest::IfEqual { mask: 0xff },
                reference_value_counter_clockwise: 1,
                write_mask_counter_clockwise: 0,
                .. Default::default()
            };
            frm.draw(&gl_objs.quad_vertex_buf, &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList), &gl_objs.prg_quad, &u, &dp).unwrap();

            u.color = [1.0, 0.0, 0.0, 0.75];
            dp.stencil = glium::draw_parameters::Stencil {
                test_counter_clockwise: glium::StencilTest::IfLess { mask: 0xff },
                reference_value_counter_clockwise: 1,
                write_mask_counter_clockwise: 0,
                .. Default::default()
            };
            frm.draw(&gl_objs.quad_vertex_buf, &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList), &gl_objs.prg_quad, &u, &dp).unwrap();
        }

        frm.finish().unwrap();

        #[cfg(xxx)]
        {
            ctx.gl_paper_size.set((rect.width() as u32, rect.height() as u32));
            let rb = glium::framebuffer::RenderBuffer::new(&gl, glium::texture::UncompressedFloatFormat::U8U8U8U8, rect.width() as u32, rect.height() as u32).unwrap();
            let mut frm = glium::framebuffer::SimpleFrameBuffer::new(&gl, &rb).unwrap();

            frm.clear_color_and_depth((0.7, 0.7, 0.7, 1.0), 1.0);

            let mat_name = ctx.material.as_deref().unwrap_or("");
            let (texture, _) = ctx.textures.get(mat_name)
                .unwrap_or_else(|| ctx.textures.get("").unwrap());

            let u = MyUniforms2D {
                m: ctx.trans_paper.ortho * ctx.trans_paper.mx,
                texture: texture.sampled(),
            };

            // Draw the textured polys
            let dp = glium::DrawParameters {
                viewport: Some(glium::Rect { left: 0, bottom: 0, width: rect.width() as u32, height: rect.height() as u32}),
                blend: glium::Blend::alpha_blending(),
                .. Default::default()
            };

            frm.draw(&ctx.paper_vertex_buf, &ctx.paper_indices_solid_buf, &ctx.prg_solid_paper, &u, &dp).unwrap();

            let GdkPixbufDataSink(pb) = gl.read_front_buffer().unwrap();
            pb.savev("test.png", "png", &[]).unwrap();
        }
    }

    fn scene_edge_build(&mut self) {
        if let Some(gl_objs) = &mut self.gl_objs {
            let mut edges = Vec::new();
            for (i_edge, edge) in self.papercraft.model().edges() {
                let selected = self.selected_edge == Some(i_edge);
                let status = self.papercraft.edge_status(i_edge);
                if status == paper::EdgeStatus::Hidden {
                    continue;
                }
                let cut = self.papercraft.edge_status(i_edge) == paper::EdgeStatus::Cut;
                let color = match (selected, cut) {
                    (true, false) => [0.0, 0.0, 1.0, 1.0],
                    (true, true) => [0.5, 0.5, 1.0, 1.0],
                    (false, false) => [0.0, 0.0, 0.0, 1.0],
                    (false, true) => [1.0, 1.0, 1.0, 1.0],
                };
                let p0 = self.papercraft.model()[edge.v0()].pos();
                let p1 = self.papercraft.model()[edge.v1()].pos();
                edges.push(MVertex3DLine { pos: p0, color, top: selected as u8 });
                edges.push(MVertex3DLine { pos: p1, color, top: selected as u8 });
            }
            gl_objs.vertex_edges_buf.update(&edges);
        }
    }
}


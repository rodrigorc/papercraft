#![allow(dead_code)]

use cgmath::{
    prelude::*,
    Deg,
};
use glium::{
    draw_parameters::PolygonOffset,
};
use gtk::{
    prelude::*,
    gdk::{self, EventMask},
};

use std::{collections::{HashMap, HashSet}, cell::Cell};
use std::rc::Rc;
use std::cell::RefCell;

mod waveobj;
mod paper;
mod util_3d;
mod util_gl;

use util_3d::{Matrix2, Matrix3, Matrix4, Quaternion, Vector2, Point2, Point3, Vector3};
use util_gl::{GdkGliumBackend, Uniforms2D, Uniforms3D, MVertex3D, MVertex2D, MVertexQuad, PersistentIndexBuffer, PersistentVertexBuffer};

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
                let data = std::fs::read(map).unwrap();
                pbl.write(&data).ok().unwrap();
                pbl.close().ok().unwrap();
                let img = pbl.pixbuf().unwrap();
                dbg!(img.width(), img.height(), img.rowstride(), img.bits_per_sample(), img.n_channels());
                texture_images.insert(lib.name().to_owned(), Some(img));
                //textures.insert(name, tex);
            }
        }
    }

    let mut model = paper::Model::from_waveobj(obj);

    // Compute the bounding box, then move to the center and scale to a standard size
    let (v_min, v_max) = util_3d::bounding_box(
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
    model.tessellate_faces();

    let persp = cgmath::perspective(Deg(60.0), 1.0, 1.0, 100.0);
    let trans_scene = Transformation3D::new(
        Vector3::new(0.0, 0.0, -30.0),
        Quaternion::one(),
         20.0,
         persp
    );
    let trans_paper = {
        //let mr = Matrix3::from(Matrix2::from_angle(Deg(30.0)));
        //let mt = Matrix3::from_translation(Vector2::new(0.0, 0.0));
        let ms = Matrix3::from_scale(200.0);
        TransformationPaper {
            ortho: util_3d::ortho2d(1.0, 1.0),
            //mx: mt * ms * mr,
            mx: ms,
        }
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

        model,
        texture_images,
        material,
        selected_face: None,
        selected_edge: None,

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

    wscene.set_events(EventMask::BUTTON_PRESS_MASK | EventMask::BUTTON_MOTION_MASK | EventMask::SCROLL_MASK);
    wscene.set_has_depth_buffer(true);

    wscene.connect_button_press_event({
        let ctx = ctx.clone();
        move |w, ev| {
            w.grab_focus();
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            ctx.last_cursor_pos = ev.position();

            if ev.button() == 1 {
                let rect = w.allocation();
                let (x, y) = ev.position();
                let x = (x as f32 / rect.width() as f32) * 2.0 - 1.0;
                let y = -((y as f32 / rect.height() as f32) * 2.0 - 1.0);
                let click = Point3::new(x as f32, y as f32, 1.0);

                let selection = ctx.analyze_click(click, rect.height() as f32);
                match selection {
                    ClickResult::None => {
                        ctx.selected_edge = None;
                        ctx.selected_face = None;
                    }
                    ClickResult::Face(iface) => {
                        let face = ctx.model.face_by_index(iface);
                        let idxs: Vec<_> = face.index_triangles()
                            .flatten()
                            .collect();
                        ctx.gl_objs.as_mut().map(|gl_objs| gl_objs.indices_face_sel.update(&idxs));
                        ctx.selected_face = Some(iface);
                        ctx.selected_edge = None;
                    }
                    ClickResult::Edge(iedge) => {
                        let edge = ctx.model.edge_by_index(iedge);
                        let idxs = [edge.v0(), edge.v1()];
                        ctx.gl_objs.as_mut().map(|gl_objs| gl_objs.indices_edge_sel.update(&idxs));
                        ctx.selected_edge = Some(iedge);
                        ctx.selected_face = None;
                    }
                }
                ctx.paper_build();
                ctx.wscene.queue_render();
                ctx.wpaper.queue_render();
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


    //let paper = gtk::DrawingArea::new();
    wpaper.set_events(EventMask::BUTTON_PRESS_MASK | EventMask::BUTTON_MOTION_MASK | EventMask::SCROLL_MASK);
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
            ctx.borrow_mut().last_cursor_pos = ev.position();
            Inhibit(true)
        }
    });
    wpaper.connect_motion_notify_event({
        let ctx = ctx.clone();
        move |_w, ev| {
            let mut ctx = ctx.borrow_mut();
            let pos = ev.position();
            let dx = (pos.0 - ctx.last_cursor_pos.0)  as f32;
            let dy = (pos.1 - ctx.last_cursor_pos.1) as f32;
            ctx.last_cursor_pos = pos;

            if ev.state().contains(gdk::ModifierType::BUTTON2_MASK) {
                ctx.trans_paper.mx = Matrix3::from_translation(Vector2::new(dx, dy)) * ctx.trans_paper.mx;
                ctx.wpaper.queue_render();
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
    prg_solid: glium::Program,
    prg_line: glium::Program,
    prg_solid_paper: glium::Program,
    prg_line_paper: glium::Program,
    prg_quad: glium::Program,

    textures: HashMap<String, glium::Texture2d>,

    vertex_buf: glium::VertexBuffer<MVertex3D>,
    indices_solid_buf: glium::IndexBuffer<paper::VertexIndex>,
    indices_edges_buf: glium::IndexBuffer<paper::VertexIndex>,
    indices_face_sel: PersistentIndexBuffer<paper::VertexIndex>,
    indices_edge_sel: PersistentIndexBuffer<paper::VertexIndex>,

    paper_vertex_buf: PersistentVertexBuffer<MVertex2D>,
    paper_indices_solid_buf: PersistentIndexBuffer<u32>,
    paper_indices_edge_buf: PersistentIndexBuffer<u32>,

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
    model: paper::Model,
    texture_images: HashMap<String, Option<gdk_pixbuf::Pixbuf>>,

    // State
    material: Option<String>,
    selected_face: Option<paper::FaceIndex>,
    selected_edge: Option<paper::EdgeIndex>,

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

enum ClickResult {
    None,
    Face(paper::FaceIndex),
    Edge(paper::EdgeIndex),
}

impl MyContext {
    fn analyze_click(&self, click: Point3, height: f32) -> ClickResult {
        let click_camera = self.trans_scene.persp_inv.transform_point(click);
        let click_obj = self.trans_scene.obj_inv.transform_point(click_camera);
        let camera_obj = self.trans_scene.obj_inv.transform_point(Point3::new(0.0, 0.0, 0.0));

        let ray = (camera_obj.to_vec(), click_obj.to_vec());

        let mut hit_face = None;
        for (iface, face) in self.model.faces() {
            for tri in face.index_triangles() {
                let tri = tri.map(|v| self.model.vertex_by_index(v).pos());
                let maybe_new_hit = util_3d::ray_crosses_face(ray, &tri);
                if let Some(new_hit) = maybe_new_hit {
                    dbg!(new_hit);
                    hit_face = match (hit_face, new_hit) {
                        (Some((_, p)), x) if p > x && x > 0.0 => Some((iface, x)),
                        (None, x) if x > 0.0 => Some((iface, x)),
                        (old, _) => old
                    };
                    break;
                }
            }
        }

        dbg!(hit_face);
        /*self.selected_face = hit_face.map(|(iface, _distance)| {
            let face = self.model.face_by_index(iface);
            let idxs: Vec<_> = face.index_triangles()
                .flatten()
                .collect();
                self.indices_face_sel.update(&idxs);
            iface
        });*/

        let mut hit_edge = None;
        for (iedge, edge) in self.model.edges() {
            let v1 = self.model.vertex_by_index(edge.v0()).pos();
            let v2 = self.model.vertex_by_index(edge.v1()).pos();
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

            hit_edge = Some((iedge, ray_hit, new_dist));
        }
        dbg!(hit_edge);

        match (hit_face, hit_edge) {
            (_, Some((e, _, _))) => ClickResult::Edge(e),
            (Some((f, _)), None) => ClickResult::Face(f),
            (None, None) => ClickResult::None,
        }
        /*self.selected_edge = hit_edge.map(|(iedge, _, _)| {
            let edge = self.model.edge_by_index(iedge);
            let idxs = [edge.v0(), edge.v1()];
            self.indices_edge_sel.update(&idxs);
            iedge
        });*/
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
                        dbg!(img.width(), img.height(), img.rowstride(), img.bits_per_sample(), img.n_channels());
                        glium::Texture2d::new(gl,  raw).unwrap()
                    }
                };
                (name.clone(), texture)
            })
            .collect();



        let vert_3d = include_str!("shaders/3d.vert");
        let vert_2d = include_str!("shaders/2d.vert");
        let frag_solid = include_str!("shaders/solid.frag");
        let frag_line = include_str!("shaders/line.frag");
        let vert_quad = include_str!("shaders/quad.vert");
        let frag_quad = include_str!("shaders/quad.frag");

        let prg_solid = glium::Program::from_source(gl, vert_3d, frag_solid, None).unwrap();
        let prg_line = glium::Program::from_source(gl, vert_3d, frag_line, None).unwrap();

        let prg_solid_paper = glium::Program::from_source(gl, vert_2d, frag_solid, None).unwrap();
        let prg_line_paper = glium::Program::from_source(gl, vert_2d, frag_line, None).unwrap();
        let prg_quad = glium::Program::from_source(gl, vert_quad, frag_quad, None).unwrap();

        let vertices: Vec<MVertex3D> = self.model.vertices()
            .map(|v| {
                MVertex3D {
                    pos: v.pos(),
                    normal: v.normal(),
                    uv: v.uv_inv(),
                }
            }).collect();

        let mut indices_solid = Vec::new();
        let mut indices_edges = Vec::new();
        for (_, face) in self.model.faces() {
            indices_solid.extend(face.index_triangles().flatten());
        }
        for (_, edge) in self.model.edges() {
            indices_edges.push(edge.v0());
            indices_edges.push(edge.v1());
        }


        let vertex_buf = glium::VertexBuffer::immutable(gl, &vertices).unwrap();
        let indices_solid_buf = glium::IndexBuffer::immutable(gl, glium::index::PrimitiveType::TrianglesList, &indices_solid).unwrap();
        let indices_edges_buf = glium::IndexBuffer::immutable(gl, glium::index::PrimitiveType::LinesList, &indices_edges).unwrap();

        let indices_face_sel = PersistentIndexBuffer::new(gl, glium::index::PrimitiveType::TrianglesList, 16);
        let indices_edge_sel = PersistentIndexBuffer::new(gl, glium::index::PrimitiveType::LinesList, 16);

        let paper_vertex_buf = PersistentVertexBuffer::new(gl, 0);
        let paper_indices_solid_buf = PersistentIndexBuffer::new(gl, glium::index::PrimitiveType::TrianglesList, 16);
        let paper_indices_edge_buf = PersistentIndexBuffer::new(gl, glium::index::PrimitiveType::LinesList, 16);

        let quad_vertex_buf = glium::VertexBuffer::immutable(gl,
            &[
                MVertexQuad { pos: [-1.0, -1.0] },
                MVertexQuad { pos: [ 3.0, -1.0] },
                MVertexQuad { pos: [-1.0,  3.0] },
            ]).unwrap();

        let gl_objs = GLObjects {
            prg_solid,
            prg_line,
            prg_solid_paper,
            prg_line_paper,
            prg_quad,
            textures,
            vertex_buf,
            indices_solid_buf,
            indices_edges_buf,
            indices_face_sel,
            indices_edge_sel,
            paper_vertex_buf,
            paper_indices_solid_buf,
            paper_indices_edge_buf,
            quad_vertex_buf,
        };

        self.gl_objs = Some(gl_objs);
    }

    fn paper_edge_matrix(&self, edge: &paper::Edge, face_a: &paper::Face, face_b: &paper::Face) -> cgmath::Matrix3<f32> {
        let v0 = self.model.vertex_by_index(edge.v0()).pos();
        let v1 = self.model.vertex_by_index(edge.v1()).pos();
        let a0 = face_a.normal().project(&v0);
        let b0 = face_b.normal().project(&v0);
        let a1 = face_a.normal().project(&v1);
        let b1 = face_b.normal().project(&v1);
        let mabt0 = Matrix3::from_translation(-b0);
        let mabr = Matrix3::from(Matrix2::from_angle((b1 - b0).angle(a1 - a0)));
        let mabt1 = Matrix3::from_translation(a0);
        let medge = mabt1 * mabr * mabt0;
        medge
    }

    fn paper_draw_face(&self, face: &paper::Face, i_face: paper::FaceIndex, m: &Matrix3, vertices: &mut Vec<MVertex2D>, indices_solid: &mut Vec<u32>, indices_edge: &mut Vec<u32>, vertex_map: &mut HashMap<(paper::FaceIndex, paper::VertexIndex), u32>) {
        for tri in face.index_triangles() {
            for i_v in tri {
                let i = vertex_map
                    .entry((i_face, i_v))
                    .or_insert_with(|| {
                        let v = self.model.vertex_by_index(i_v);
                        let p2 = face.normal().project(&v.pos());
                        let pos = m.transform_point(Point2::from_vec(p2)).to_vec();
                        let idx = vertices.len();
                        vertices.push(MVertex2D {
                            pos,
                            uv: v.uv_inv(),
                        });
                        idx as u32
                    });
                indices_solid.push(*i);
            }
        }

        let mut vs = face.index_vertices();
        let first = vs.next().unwrap();
        let first = *vertex_map.get(&(i_face, first)).unwrap();
        indices_edge.push(first);

        for v in face.index_vertices() {
            let v = *vertex_map.get(&(i_face, v)).unwrap();
            indices_edge.push(v);
            indices_edge.push(v);
        }
        indices_edge.push(first);
    }

    fn paper_build(&mut self) {
        if let Some(i_face) = self.selected_face {
            let mut visited_faces = HashSet::new();

            //Maps VertexIndex in the model to index in vertices
            let mut vertex_map = HashMap::new();
            let mut vertices = Vec::new();
            let mut indices_solid = Vec::new();
            let mut indices_edge = Vec::new();

            let mut stack = Vec::new();
            stack.push((i_face, Matrix3::identity()));
            visited_faces.insert(i_face);

            loop {
                let (i_face, m) = match stack.pop() {
                    Some(x) => x,
                    None => break,
                };

                let face = self.model.face_by_index(i_face);
                self.paper_draw_face(face, i_face, &m, &mut vertices, &mut indices_solid, &mut indices_edge, &mut vertex_map);
                for i_edge in face.index_edges() {
                    let edge = self.model.edge_by_index(i_edge);
                    for i_next_face in edge.faces() {
                        if visited_faces.contains(&i_next_face) {
                            continue;
                        }

                        let next_face = self.model.face_by_index(i_next_face);
                        let medge = self.paper_edge_matrix(edge, face, next_face);

                        stack.push((i_next_face, m * medge));
                        visited_faces.insert(i_next_face);
                    }
                }
            }

            self.gl_objs.as_mut().map(|gl_objs| {
                gl_objs.paper_vertex_buf.update(&vertices);
                gl_objs.paper_indices_solid_buf.update(&indices_solid);
                gl_objs.paper_indices_edge_buf.update(&indices_edge);
            });
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

        let mut u = Uniforms3D {
            m: self.trans_scene.persp * self.trans_scene.obj,
            mnormal: self.trans_scene.mnormal, // should be transpose of inverse
            lights: [light0, light1],
            texture: texture.sampled(),
        };

        // Draw the textured polys
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

        //dp.color_mask = (false, false, false, false);
        dp.polygon_offset = PolygonOffset {
            line: true,
            fill: true,
            factor: 1.0,
            units: 1.0,
            .. PolygonOffset::default()
        };
        frm.draw(&gl_objs.vertex_buf, &gl_objs.indices_solid_buf, &gl_objs.prg_solid, &u, &dp).unwrap();

        if self.selected_face.is_some() {
            u.texture = gl_objs.textures.get("").unwrap().sampled();
            frm.draw(&gl_objs.vertex_buf, &gl_objs.indices_face_sel, &gl_objs.prg_solid, &u, &dp).unwrap();
        }

        // Draw the lines:

        //dp.color_mask = (true, true, true, true);
        //dp.polygon_offset = PolygonOffset::default();
        dp.line_width = Some(1.0);
        dp.smooth = Some(glium::Smooth::Nicest);
        frm.draw(&gl_objs.vertex_buf, &gl_objs.indices_edges_buf, &gl_objs.prg_line, &u, &dp).unwrap();

        dp.depth.test = glium::DepthTest::Overwrite;
        if self.selected_edge.is_some() {
            dp.line_width = Some(3.0);
            frm.draw(&gl_objs.vertex_buf, &gl_objs.indices_edge_sel, &gl_objs.prg_line, &u, &dp).unwrap();
        }

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

        let u = Uniforms2D {
            m: self.trans_paper.ortho * self.trans_paper.mx,
            texture: texture.sampled(),
        };

        // Draw the textured polys
        let mut dp = glium::DrawParameters {
            viewport: Some(glium::Rect { left: 0, bottom: 0, width: rect.width() as u32, height: rect.height() as u32}),
            blend: glium::Blend::alpha_blending(),
            stencil: glium::draw_parameters::Stencil {
                depth_pass_operation_counter_clockwise: glium::StencilOperation::Increment,
                .. Default::default()
            },
            .. Default::default()
        };

        frm.draw(&gl_objs.paper_vertex_buf, &gl_objs.paper_indices_solid_buf, &gl_objs.prg_solid_paper, &u, &dp).unwrap();

        dp.line_width = Some(3.0);
        frm.draw(&gl_objs.paper_vertex_buf, &gl_objs.paper_indices_edge_buf, &gl_objs.prg_line_paper, &u, &dp).unwrap();

        dp.stencil = glium::draw_parameters::Stencil {
            test_counter_clockwise: glium::StencilTest::IfLess { mask: 0xff },
            reference_value_counter_clockwise: 1,
            write_mask_counter_clockwise: 0,
            .. Default::default()
        };
        frm.draw(&gl_objs.quad_vertex_buf, &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList), &gl_objs.prg_quad, &u, &dp).unwrap();

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
}


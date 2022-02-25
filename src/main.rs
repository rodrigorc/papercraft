#![allow(dead_code)]

use cgmath::prelude::*;
use cgmath::conv::{array4x4, array3x3, array3};
use glium::draw_parameters::PolygonOffset;
use glium::uniforms::AsUniformValue;
use gtk::prelude::*;
use gtk::gdk::{self, EventMask};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

mod waveobj;
mod paper;
mod util_3d;

fn main() {
    std::env::set_var("GTK_CSD", "0");
    gtk::init().expect("gtk::init");

    let w = gtk::Window::new(gtk::WindowType::Toplevel);
    w.set_default_size(800, 600);
    w.connect_destroy(move |_| {
        gtk::main_quit();
    });
    gl_loader::init_gl();
    let gl = gtk::GLArea::new();
    gl.set_has_depth_buffer(true);
    let ctx: Rc<RefCell<Option<MyContext>>> = Rc::new(RefCell::new(None));

    gl.set_events(EventMask::BUTTON_PRESS_MASK | EventMask::BUTTON_MOTION_MASK | EventMask::SCROLL_MASK);
    gl.connect_button_press_event({
        let ctx = ctx.clone();
        move |w, ev|  {
            w.grab_focus();
            if let Some(ctx) = ctx.borrow_mut().as_mut() {
                ctx.last_cursor_pos = ev.position();

                if ev.button() == 1 {
                    let rect = w.allocation();
                    let (x, y) = ev.position();
                    let x = (x as f32 / rect.width() as f32) * 2.0 - 1.0;
                    let y = -((y as f32 / rect.height() as f32) * 2.0 - 1.0);
                    let click = cgmath::Point3::new(x as f32, y as f32, 1.0);

                    let click_camera = ctx.trans.persp_inv.transform_point(click);
                    let click_obj = ctx.trans.obj_inv.transform_point(click_camera);
                    let camera_obj = ctx.trans.obj_inv.transform_point(cgmath::Point3::new(0.0, 0.0, 0.0));

                    let ray = (camera_obj.to_vec(), click_obj.to_vec());

                    let mut hit_face = None;
                    for (iface, face) in ctx.model.faces() {
                        for tri in face.index_triangles() {
                            let tri = tri.map(|v| ctx.model.vertex_by_index(v).pos().into());
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
                    ctx.selected_face = hit_face.map(|(iface, _distance)| {
                        let face = ctx.model.face_by_index(iface);
                        let idxs: Vec<_> = face.index_triangles()
                            .flatten()
                            .collect();
                            ctx.indices_face_sel.update(&idxs);
                        iface
                    });

                    let mut hit_edge = None;
                    for (iedge, edge) in ctx.model.edges() {
                        let v1 = ctx.model.vertex_by_index(edge.v0()).pos().into();
                        let v2 = ctx.model.vertex_by_index(edge.v1()).pos().into();
                        let (ray_hit, _line_hit, new_dist) = util_3d::line_segment_distance(ray, (v1, v2));

                        // Behind the screen, it is not a hit
                        if ray_hit <= 0.0001 {
                            continue;
                        }

                        // new_dist is originally the distance in real-world space, but the user is using the screen, so scale accordingly
                        let new_dist = new_dist / ray_hit * (rect.height() as f32);

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
                        ctx.selected_face = None;
                    }
                    dbg!(hit_edge);
                    ctx.selected_edge = hit_edge.map(|(iedge, _, _)| {
                        let edge = ctx.model.edge_by_index(iedge);
                        let idxs = [edge.v0(), edge.v1()];
                        ctx.indices_edge_sel.update(&idxs);
                        iedge
                    });
                    w.queue_render();
                }
            }
            Inhibit(false)
        }
    });
    gl.connect_scroll_event({
        let ctx = ctx.clone();
        let gl = gl.clone();
        move |_w, ev|  {
            if let Some(ctx) = ctx.borrow_mut().as_mut() {
                let dz = match ev.direction() {
                    gdk::ScrollDirection::Up => 1.1,
                    gdk::ScrollDirection::Down => 1.0 / 1.1,
                    _ => 1.0,
                };
                ctx.trans.scale *= dz;
                ctx.trans.recompute_obj();
                gl.queue_render();
            }
            Inhibit(true)
        }
    });
    gl.connect_motion_notify_event({
        let ctx = ctx.clone();
        let gl = gl.clone();
        move |_w, ev|  {
            if let Some(ctx) = ctx.borrow_mut().as_mut() {
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
                    let roty = cgmath::Quaternion::new(cosy, 0.0, siny, 0.0);
                    let rotx = cgmath::Quaternion::new(cosx, sinx, 0.0, 0.0);

                    ctx.trans.rotation = (roty * rotx * ctx.trans.rotation).normalize();
                    ctx.trans.recompute_obj();
                    gl.queue_render();
                } else if ev.state().contains(gdk::ModifierType::BUTTON2_MASK) {
                    let dx = dx / 50.0;
                    let dy = -dy / 50.0;

                    ctx.trans.location = ctx.trans.location + cgmath::Vector3::new(dx, dy, 0.0);
                    ctx.trans.recompute_obj();
                    gl.queue_render();
                }
            }
            Inhibit(true)
        }
    });

    gl.connect_realize({
        let ctx = ctx.clone();
        move |w| gl_realize(w, &ctx)
    });
    gl.connect_unrealize({
        let ctx = ctx.clone();
        move |w| gl_unrealize(w, &ctx)
    });
    gl.connect_render({
        let ctx = ctx.clone();
        move |w, gl| gl_render(w, gl, &ctx)
    });
    gl.connect_resize({
        let ctx = ctx.clone();
        move |_w, width, height| {
            if let Some(ctx) = ctx.borrow_mut().as_mut() {
                let ratio = width as f32 / height as f32;
                ctx.trans.set_ratio(ratio);
            }
        }
    });
    w.add(&gl);

    /*
    glib::timeout_add_local(std::time::Duration::from_millis(50), {
        let ctx = ctx.clone();
        let gl = gl.clone();
        move || {
            if let Some(ctx) = ctx.borrow_mut().as_mut() {
                gl.queue_render();

            }
            glib::Continue(true)
        }
    });*/

    w.show_all();
    gtk::main();
}


fn gl_realize(w: &gtk::GLArea, ctx: &Rc<RefCell<Option<MyContext>>>) {
    w.attach_buffers();
    let mut ctx = ctx.borrow_mut();
    let gl = w.context().unwrap();
    let backend = GdkGliumBackend { ctx: gl.clone() };
    let glctx = unsafe { glium::backend::Context::new(backend, false, glium::debug::DebugCallbackBehavior::Ignore).unwrap() };

    let vsh = r"
#version 150

uniform mat4 m;
uniform mat3 mnormal;

uniform vec3 lights[2];
in vec3 pos;
in vec3 normal;
in vec2 uv;

out vec2 v_uv;
out float v_light;

void main(void) {
    gl_Position = m * vec4(pos, 1.0);
    vec3 obj_normal = normalize(mnormal * normal);

    float light = 0.2;
    for (int i = 0; i < 2; ++i) {
        float diffuse = max(abs(dot(obj_normal, -lights[i])), 0.0);
        light += diffuse;
    }
    v_light = light;
    v_uv = uv;
}
";
    let fsh_solid = r"
#version 150

uniform sampler2D tex;

in vec2 v_uv;
in float v_light;
out vec4 out_frag_color;

void main(void) {
    vec4 base;
    if (gl_FrontFacing)
        base = texture2D(tex, v_uv);
    else
        base = vec4(0.8, 0.3, 0.3, 1.0);
    out_frag_color = vec4(v_light * base.rgb, base.a);
}
";
    let fsh_line = r"
    #version 150

    in float v_light;
    out vec4 out_frag_color;

    void main(void) {
        out_frag_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
    ";
    let prg_solid = glium::Program::from_source(&glctx, vsh, fsh_solid, None).unwrap();
    let prg_line = glium::Program::from_source(&glctx, vsh, fsh_line, None).unwrap();

    let f = std::fs::File::open("pikachu.obj").unwrap();
    let f = std::io::BufReader::new(f);
    let (matlibs, models) = waveobj::Model::from_reader(f).unwrap();

    // For now read only the first model from the file
    let obj = models.get(0).unwrap();
    let material = obj.material().map(String::from);
    let mut textures = HashMap::new();

    // Empty texture is just a single white texel
    let empty = glium::Texture2d::empty(&glctx, 1, 1).unwrap();
    empty.write(glium::Rect{ left: 0, bottom: 0, width: 1, height: 1 }, vec![vec![(255u8, 255u8, 255u8, 255u8)]]);
    textures.insert(String::new(), empty);

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
                let tex = glium::Texture2d::new(&glctx,  raw).unwrap();
                textures.insert(String::from(lib.name()), tex);
            }
        }
    }

    let mut model = paper::Model::from_waveobj(&obj);
    let (v_min, v_max) = util_3d::bounding_box(
        model
            .vertices()
            .map(|v| cgmath::Vector3::from(v.pos()))
    );
    let size = (v_max.x - v_min.x).max(v_max.y - v_min.y).max(v_max.z - v_min.z);
    let mscale = cgmath::Matrix4::<f32>::from_scale(1.0 / size);
    let center = (v_min + v_max) / 2.0;
    let mcenter = cgmath::Matrix4::<f32>::from_translation(-center);
    let m = mscale * mcenter;

    model.transform_vertices(|pos, _normal| {
        //only scale and translate, no need to touch normals
        *pos = m.transform_point(cgmath::Point3::from(*pos)).into();
    });

    let vertices: Vec<MVertex> = model.vertices()
        .map(|v| {
            let uv = v.uv();
            MVertex {
                pos: v.pos(),
                normal: v.normal(),
                uv: [uv[0], 1.0 - uv[1]],
            }
        }).collect();

    let mut indices_solid = Vec::new();
    let mut indices_edges = Vec::new();
    for (_, face) in model.faces() {
        indices_solid.extend(face.index_triangles().flatten());
    }
    for (_, edge) in model.edges() {
        indices_edges.push(edge.v0());
        indices_edges.push(edge.v1());
    }
    let gl = GlData {
        glctx,
        prg_solid,
        prg_line,
    };

    let vertex_buf = glium::VertexBuffer::immutable(&gl.glctx, &vertices).unwrap();
    let indices_solid_buf = glium::IndexBuffer::immutable(&gl.glctx, glium::index::PrimitiveType::TrianglesList, &indices_solid).unwrap();
    let indices_edges_buf = glium::IndexBuffer::persistent(&gl.glctx, glium::index::PrimitiveType::LinesList, &indices_edges).unwrap();

    let indices_face_sel = PersistentIndexBuffer::new(&gl.glctx, glium::index::PrimitiveType::TrianglesList);
    let indices_edge_sel = PersistentIndexBuffer::new(&gl.glctx, glium::index::PrimitiveType::LinesList);

    let persp = cgmath::perspective(cgmath::Deg(60.0), 1.0, 1.0, 100.0);
    let trans = Transformation::new(
        cgmath::Vector3::new(0.0, 0.0, -30.0),
        cgmath::Quaternion::one(),
         20.0,
         persp
    );
    *ctx = Some(MyContext {
        gl,
        model,

        textures,
        vertex_buf,
        indices_solid_buf,
        indices_edges_buf,
        indices_face_sel,
        indices_edge_sel,

        material,
        selected_face: None,
        selected_edge: None,

        last_cursor_pos: (0.0, 0.0),

        trans,
     });
}

fn gl_unrealize(_w: &gtk::GLArea, ctx: &Rc<RefCell<Option<MyContext>>>) {
    dbg!("GL unrealize!");
    let mut ctx = ctx.borrow_mut();
    *ctx = None;
}

struct MyUniforms<'a> {
    m: cgmath::Matrix4<f32>,
    mnormal: cgmath::Matrix3<f32>,
    lights: [cgmath::Vector3<f32>; 2],
    texture: glium::uniforms::Sampler<'a, glium::Texture2d>,
}

impl glium::uniforms::Uniforms for MyUniforms<'_> {
    fn visit_values<'a, F: FnMut(&str, glium::uniforms::UniformValue<'a>)>(&'a self, mut visit: F) {
        use glium::uniforms::UniformValue::*;

        visit("m", Mat4(array4x4(self.m)));
        visit("mnormal", Mat3(array3x3(self.mnormal)));
        visit("lights[0]", Vec3(array3(self.lights[0])));
        visit("lights[1]", Vec3(array3(self.lights[1])));
        visit("tex", self.texture.as_uniform_value());
    }
}

fn gl_render(w: &gtk::GLArea, _gl: &gdk::GLContext, ctx: &Rc<RefCell<Option<MyContext>>>) -> gtk::Inhibit {
    let rect = w.allocation();

    let mut ctx = ctx.borrow_mut();
    let ctx = ctx.as_mut().unwrap();
    let mut frm = glium::Frame::new(ctx.gl.glctx.clone(), (rect.width() as u32, rect.height() as u32));

    use glium::Surface;

    frm.clear_color_and_depth((0.2, 0.2, 0.4, 1.0), 1.0);

    let light0 = cgmath::Vector3::new(-0.5, -0.4, -0.8).normalize() * 0.55;
    let light1 = cgmath::Vector3::new(0.8, 0.2, 0.4).normalize() * 0.25;

    let mat_name = ctx.material.as_deref().unwrap_or("");
    let texture = ctx.textures.get(mat_name)
        .unwrap_or(ctx.textures.get("").unwrap());

    let mut u = MyUniforms {
        m: ctx.trans.persp * ctx.trans.obj,
        mnormal: ctx.trans.mnormal, // should be transpose of inverse
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
    frm.draw(&ctx.vertex_buf, &ctx.indices_solid_buf, &ctx.gl.prg_solid, &u, &dp).unwrap();

    if ctx.selected_face.is_some() {
        u.texture = ctx.textures.get("").unwrap().sampled();
        frm.draw(&ctx.vertex_buf, &ctx.indices_face_sel, &ctx.gl.prg_solid, &u, &dp).unwrap();
    }

    // Draw the lines:

    //dp.color_mask = (true, true, true, true);
    //dp.polygon_offset = PolygonOffset::default();
    dp.line_width = Some(1.0);
    dp.smooth = Some(glium::Smooth::Nicest);
    frm.draw(&ctx.vertex_buf, &ctx.indices_edges_buf, &ctx.gl.prg_line, &u, &dp).unwrap();

    dp.depth.test = glium::DepthTest::Overwrite;
    if ctx.selected_edge.is_some() {
        dp.line_width = Some(3.0);
        frm.draw(&ctx.vertex_buf, &ctx.indices_edge_sel, &ctx.gl.prg_line, &u, &dp).unwrap();
    }

    frm.finish().unwrap();

    gtk::Inhibit(false)
}

struct GdkGliumBackend {
    ctx: gdk::GLContext,
}

unsafe impl glium::backend::Backend for GdkGliumBackend {
    fn swap_buffers(&self) -> Result<(), glium::SwapBuffersError> {
        Ok(())
    }
    unsafe fn get_proc_address(&self, symbol: &str) -> *const core::ffi::c_void {
        gl_loader::get_proc_address(symbol) as _
    }
    fn get_framebuffer_dimensions(&self) -> (u32, u32) {
        let w = self.ctx.window().unwrap();
        (w.width() as u32, w.height() as u32)
    }
    fn is_current(&self) -> bool {
        gdk::GLContext::current().as_ref() == Some(&self.ctx)
    }
    unsafe fn make_current(&self) {
        self.ctx.make_current();
    }
}

// This contains GL objects that are overall constant
struct GlData {
    glctx: Rc<glium::backend::Context>,
    prg_solid: glium::Program,
    prg_line: glium::Program,
}

// This contains GL objects that are object specific
struct MyContext {
    gl: GlData,

    // The model
    model: paper::Model,

    // GL objects
    textures: HashMap<String, glium::Texture2d>,

    vertex_buf: glium::VertexBuffer<MVertex>,
    indices_solid_buf: glium::IndexBuffer<paper::VertexIndex>,
    indices_edges_buf: glium::IndexBuffer<paper::VertexIndex>,

    indices_face_sel: PersistentIndexBuffer,
    indices_edge_sel: PersistentIndexBuffer,

    // State
    material: Option<String>,
    selected_face: Option<paper::FaceIndex>,
    selected_edge: Option<paper::EdgeIndex>,

    last_cursor_pos: (f64, f64),

    trans: Transformation,
}

struct Transformation {
    location: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scale: f32,

    persp: cgmath::Matrix4<f32>,
    persp_inv: cgmath::Matrix4<f32>,
    obj: cgmath::Matrix4<f32>,
    obj_inv: cgmath::Matrix4<f32>,
    mnormal: cgmath::Matrix3<f32>,
}

impl Transformation {
    fn new(location: cgmath::Vector3<f32>, rotation: cgmath::Quaternion<f32>, scale: f32, persp: cgmath::Matrix4<f32>) -> Transformation {
        let mut tr = Transformation {
            location,
            rotation,
            scale,
            persp,
            persp_inv: persp.invert().unwrap(),
            obj: cgmath::Matrix4::one(),
            obj_inv: cgmath::Matrix4::one(),
            mnormal: cgmath::Matrix3::one(),
        };
        tr.recompute_obj();
        tr
    }
    fn recompute_obj(&mut self) {
        let r = cgmath::Matrix3::from(self.rotation);
        let t = cgmath::Matrix4::<f32>::from_translation(self.location);
        let s = cgmath::Matrix4::<f32>::from_scale(self.scale);

        self.obj = t * cgmath::Matrix4::from(r) * s;
        self.obj_inv = self.obj.invert().unwrap();
        self.mnormal = r; //should be inverse of transpose
    }

    fn set_ratio(&mut self, ratio: f32) {
        let f = self.persp[1][1];
        self.persp[0][0] = f / ratio;
        self.persp_inv = self.persp.invert().unwrap();
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

glium::implement_vertex!(MVertex, pos, normal, uv);

struct PersistentIndexBuffer {
    buffer: glium::IndexBuffer<paper::VertexIndex>,
    length: usize,
}

impl PersistentIndexBuffer {
    fn new(ctx: &impl glium::backend::Facade, prim: glium::index::PrimitiveType) -> PersistentIndexBuffer {
        let buffer = glium::IndexBuffer::empty_persistent(ctx, prim, 16).unwrap();
        PersistentIndexBuffer {
            buffer,
            length: 0,
        }
    }
    fn update(&mut self, data: &[paper::VertexIndex]) {
        if let Some(slice) = self.buffer.slice(0 .. data.len()) {
            self.length = data.len();
            slice.write(data);
        } else {
            // If the buffer is not big enough, remake it
            let ctx = self.buffer.get_context();
            self.buffer = glium::IndexBuffer::persistent(ctx, self.buffer.get_primitives_type(), data).unwrap();
            self.length = data.len();
        }
    }
}

impl<'a> Into<glium::index::IndicesSource<'a>> for &'a PersistentIndexBuffer {
    fn into(self) -> glium::index::IndicesSource<'a> {
        self.buffer.slice(0 .. self.length).unwrap().into()
    }
}


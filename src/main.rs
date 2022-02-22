use cgmath::{InnerSpace, Quaternion, One, Transform, SquareMatrix, EuclideanSpace};
use cgmath::conv::{array4x4, array3x3, array3};
use glium::draw_parameters::PolygonOffset;
use glium::uniforms::AsUniformValue;
use gtk::prelude::*;
//use cgmath::prelude::*;
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

    let theme = gtk::IconTheme::default().expect("icon_theme_default");
    theme.append_search_path("xxxx");

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

                    let ratio = (rect.width() as f32) / (rect.height() as f32);
                    let persp: cgmath::Matrix4<f32> = cgmath::perspective(cgmath::Deg(60.0), ratio, 1.0, 100.0);
                    let r = cgmath::Matrix3::from(ctx.rotation);
                    let t = cgmath::Matrix4::<f32>::from_translation(ctx.location);
                    let s = cgmath::Matrix4::<f32>::from_scale(ctx.scale);
                    let obj = t * cgmath::Matrix4::from(r) * s;
                    let persp_inv = persp.invert().unwrap();
                    let obj_inv = obj.invert().unwrap();

                    let click_camera = persp_inv.transform_point(click);
                    let click_obj = obj_inv.transform_point(click_camera);
                    let camera_obj = obj_inv.transform_point(cgmath::Point3::new(0.0, 0.0, 0.0));

                    let ray = (camera_obj.to_vec(), click_obj.to_vec());

                    let mut hit_face = None;
                    for (iface, face) in ctx.gldata.model.faces().iter().enumerate() {
                        for (a, b, c) in face.index_triangles() {
                            let a = ctx.gldata.model.vertex_by_index(a).pos();
                            let b = ctx.gldata.model.vertex_by_index(b).pos();
                            let c = ctx.gldata.model.vertex_by_index(c).pos();
                            let maybe_new_hit = util_3d::ray_crosses_face(ray, &[(*a).into(), (*b).into(), (*c).into()]);
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
                    ctx.selected_face = hit_face.map(|(selection, _distance)| {
                        let face = &ctx.gldata.model.faces()[selection];
                        let idxs: Vec<_> = face.index_triangles()
                            .flat_map(|(a, b, c)| {
                                [a, b, c]
                            })
                            .map(|x| x.0)
                            .collect();
                        udpate_persistent_index_buffer(&mut ctx.gldata.indices_face_sel, &idxs);

                        SelectedIndex {
                            selection,
                            index_len: idxs.len(),
                        }
                    });

                    /*
                    let mut hit_line = None;
                    for (iline, line) in ctx.indices_lines.chunks_exact(2).enumerate() {
                        let v1 = ctx.vertices[line[0] as usize].pos.into();
                        let v2 = ctx.vertices[line[1] as usize].pos.into();
                        let (ray_hit, _line_hit, new_dist_2) = util_3d::line_segment_distance(ray, (v1, v2));

                        if new_dist_2 > 0.01  || ray_hit < 0.0 {
                            continue;
                        }

                        match hit_line {
                            Some((_, _, p)) if p < new_dist_2 => { continue; }
                            _ => {}
                        }
                        match hit_face {
                            Some((_, p)) if p < 0.9 * ray_hit => { continue; }
                            _ => {}
                        }

                        hit_line = Some((iline, ray_hit, new_dist_2));
                    }
                    dbg!(hit_line);
                    ctx.selected_line = hit_line.map(|(iline, _, _)| iline);*/
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
                    _ => 0.0
                };
                ctx.scale *= dz;
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
                    let roty = Quaternion::new(cosy, 0.0, siny, 0.0);
                    let rotx = Quaternion::new(cosx, sinx, 0.0, 0.0);

                    ctx.rotation = (roty * rotx * ctx.rotation).normalize();
                    gl.queue_render();
                } else if ev.state().contains(gdk::ModifierType::BUTTON2_MASK) {
                    let dx = dx / 50.0;
                    let dy = -dy / 50.0;

                    ctx.location = ctx.location + cgmath::Vector3::new(dx, dy, 0.0);
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

    let f = std::fs::File::open("v2.obj").unwrap();
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
    let (v_min, v_max) = util_3d::bounding_box(model.vertices()
        .iter()
        .map(|v| {
            let pos = v.pos();
            cgmath::Vector3::from(*pos)
        }));
    let size = (v_max.x - v_min.x).max(v_max.y - v_min.y).max(v_max.z - v_min.z);
    let mscale = cgmath::Matrix4::<f32>::from_scale(1.0 / size);
    let center = (v_min + v_max) / 2.0;
    let mcenter = cgmath::Matrix4::<f32>::from_translation(-center);
    let m = mscale * mcenter;

    for v in model.vertices_mut() {
        *v.pos_mut() = m.transform_point(cgmath::Point3::from(*v.pos())).into();
    }

    let vertices: Vec<MVertex> = model.vertices()
        .iter()
        .map(|v| {
            let pos = v.pos();
            //let pos = m.transform_point(cgmath::Point3::from(*pos));
            let uv = v.uv();
            MVertex {
                pos: *pos,
                normal: *v.normal(),
                uv: [uv[0], 1.0 - uv[1]],
            }
        }).collect();

    let mut indices_solid = Vec::new();
    let mut indices_lines = Vec::new();
    for face in model.faces() {
        let tris = face.index_triangles();
        for (a, b, c) in tris {
            indices_solid.push(a.0);
            indices_solid.push(b.0);
            indices_solid.push(c.0);
        }
        let face_indices = face.index_vertices();
        for i in 0 .. face_indices.len() {
            indices_lines.push(face_indices[i].0);
            indices_lines.push(face_indices[(i + 1) % face_indices.len()].0);
        }
    }

    let gl = GlContext {
        glctx,
        prg_solid,
        prg_line,
    };

    let vertex_buf = glium::VertexBuffer::immutable(&gl.glctx, &vertices).unwrap();
    let indices_solid_buf = glium::IndexBuffer::immutable(&gl.glctx, glium::index::PrimitiveType::TrianglesList, &indices_solid).unwrap();
    let indices_lines_buf = glium::IndexBuffer::persistent(&gl.glctx, glium::index::PrimitiveType::LinesList, &indices_lines).unwrap();

    let indices_face_sel = glium::IndexBuffer::empty_persistent(&gl.glctx, glium::index::PrimitiveType::TrianglesList, 12).unwrap();
    let indices_edge_sel = glium::IndexBuffer::empty_persistent(&gl.glctx, glium::index::PrimitiveType::LinesList, 12).unwrap();

    let gldata = GlData {
        model,
        vertex_buf,
        indices_solid_buf,
        indices_lines_buf,
        indices_face_sel,
        indices_edge_sel,
    };

    *ctx = Some(MyContext {
        gl,
        gldata,
        vertices,
        indices_solid,
        indices_lines,
        textures,
        material,
        selected_face: None,
        selected_line: None,

        last_cursor_pos: (0.0, 0.0),
        rotation: Quaternion::one(),
        location: cgmath::Vector3::new(0.0, 0.0, -30.0),
        scale: 20.0,
     });
}

fn gl_unrealize(_w: &gtk::GLArea, ctx: &Rc<RefCell<Option<MyContext>>>) {
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

    let ratio = (rect.width() as f32) / (rect.height() as f32);
    let persp = cgmath::perspective(cgmath::Deg(60.0), ratio, 1.0, 100.0);
    let r = cgmath::Matrix3::from(ctx.rotation);
    let t = cgmath::Matrix4::<f32>::from_translation(ctx.location);
    let s = cgmath::Matrix4::<f32>::from_scale(ctx.scale);

    let light0 = cgmath::Vector3::new(-0.5, -0.4, -0.8).normalize() * 0.55;
    let light1 = cgmath::Vector3::new(0.8, 0.2, 0.4).normalize() * 0.25;

    let mat_name = ctx.material.as_deref().unwrap_or("");
    let texture = ctx.textures.get(mat_name)
        .unwrap_or(ctx.textures.get("").unwrap());

    let mut u = MyUniforms {
        m: persp * t * cgmath::Matrix4::from(r) * s,
        mnormal: r, //should be transpose of inverse
        lights: [light0, light1],
        texture: texture.sampled(),
    };

    // Draw de textured polys
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
    frm.draw(&ctx.gldata.vertex_buf, &ctx.gldata.indices_solid_buf, &ctx.gl.prg_solid, &u, &dp).unwrap();

    if let &Some(SelectedIndex { selection: _, index_len}) = &ctx.selected_face {
        u.texture = ctx.textures.get("").unwrap().sampled();
        frm.draw(&ctx.gldata.vertex_buf, ctx.gldata.indices_face_sel.slice(0 .. index_len).unwrap(), &ctx.gl.prg_solid, &u, &dp).unwrap();
    }

    // Draw the lines:

    //dp.color_mask = (true, true, true, true);
    dp.polygon_offset = PolygonOffset::default();
    dp.line_width = Some(1.0);
    dp.smooth = Some(glium::Smooth::Nicest);
    frm.draw(&ctx.gldata.vertex_buf, &ctx.gldata.indices_lines_buf, &ctx.gl.prg_line, &u, &dp).unwrap();

    /*
    dp.depth.test = glium::DepthTest::Overwrite;
    if let &Some(sel) = &ctx.selected_line {
        let idxs = [ctx.indices_lines[2 * sel], ctx.indices_lines[2*sel + 1]];
        udpate_persistent_index_buffer(&mut ctx.gldata.indices_edge_sel, &idxs);
        dp.line_width = Some(3.0);
        frm.draw(&ctx.gldata.vertex_buf, ctx.gldata.indices_edge_sel.slice(0 .. idxs.len()).unwrap(), &ctx.gl.prg_line, &u, &dp).unwrap();
    }*/

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
struct GlContext {
    glctx: Rc<glium::backend::Context>,
    prg_solid: glium::Program,
    prg_line: glium::Program,
}

// This contains GL objects that are object specific
struct GlData {
    model: paper::Model,
    vertex_buf: glium::VertexBuffer<MVertex>,
    indices_solid_buf: glium::IndexBuffer<u32>,
    indices_lines_buf: glium::IndexBuffer<u32>,

    indices_face_sel: glium::IndexBuffer<u32>,
    indices_edge_sel: glium::IndexBuffer<u32>,
}

struct SelectedIndex {
    selection: usize,
    index_len: usize,
}

struct MyContext {
    gl: GlContext,
    gldata: GlData,
    vertices: Vec<MVertex>,
    indices_solid: Vec<u32>,
    indices_lines: Vec<u32>,
    textures: HashMap<String, glium::Texture2d>,
    material: Option<String>,
    selected_face: Option<SelectedIndex>,
    selected_line: Option<SelectedIndex>,

    last_cursor_pos: (f64, f64),

    rotation: Quaternion<f32>,
    location: cgmath::Vector3<f32>,
    scale: f32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

glium::implement_vertex!(MVertex, pos, normal, uv);

fn udpate_persistent_index_buffer(buf: &mut glium::IndexBuffer<u32>, data: &[u32]) {
    if let Some(slice) = buf.slice(0 .. data.len()) {
        slice.write(data);
    } else {
        // If the buffer is not big enough, remake it
        let ctx = buf.get_context();
        *buf = glium::IndexBuffer::persistent(ctx, buf.get_primitives_type(), data).unwrap();
    }
}


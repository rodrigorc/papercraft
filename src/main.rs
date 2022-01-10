use cgmath::InnerSpace;
use cgmath::conv::{array4x4, array4};
use gtk::prelude::*;
//use cgmath::prelude::*;
use gtk::gdk;
use std::rc::Rc;
use std::cell::RefCell;

mod waveobj;

fn main() {
    std::env::set_var("GTK_CSD", "0");
    gtk::init().expect("gtk::init");

    let theme = gtk::IconTheme::default().expect("icon_theme_default");
    theme.append_search_path("xxxx");

    let w = gtk::Window::new(gtk::WindowType::Toplevel);
    w.connect_destroy(move |_| {
        gtk::main_quit();
    });
    gl_loader::init_gl();
    let gl = gtk::GLArea::new();
    gl.set_has_depth_buffer(true);
    let ctx = Rc::new(RefCell::new(None));

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
    glib::timeout_add_local(std::time::Duration::from_millis(50), {
        let ctx = ctx.clone();
        let gl = gl.clone();
        move || {
            if let Some(ctx) = ctx.borrow_mut().as_mut() {
                ctx.r += 1.0;
                gl.queue_render();

            }
            glib::Continue(true)
        }
    });

    w.show_all();
    gtk::main();
}


fn gl_realize(w: &gtk::GLArea, ctx: &Rc<RefCell<Option<MyContext>>>) {
    w.attach_buffers();
    let mut ctx = ctx.borrow_mut();
    let gl = w.context().unwrap();
    let backend = GdkGliumBackend { ctx: gl.clone() };
    let glctx = unsafe { glium::backend::Context::new(backend, false, glium::debug::DebugCallbackBehavior::Ignore).unwrap() };

    /*
    unsafe {
        let g = &glctx.gl;
        use glium::gl;
        dbg!(g.GetError());
        let mut i = 0;
        g.GetIntegerv(gl::DRAW_FRAMEBUFFER_BINDING, &mut i);
        dbg!(g.GetError(), i);
        g.GetFramebufferAttachmentParameteriv(
            gl::FRAMEBUFFER,
            gl::DEPTH_ATTACHMENT,
            gl::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,
            &mut i);
        dbg!(g.GetError(), i);

    }*/

    let vsh = r"
#version 150

uniform mat4 m;
uniform mat4 mnormal;

uniform vec4 lights[2];
in vec3 pos;
in vec3 normal;

out float v_color;

void main(void) {
    gl_Position = m * vec4(pos, 1.0);
    vec4 obj_normal = normalize(mnormal * vec4(normal, 0.0));

    float color = 0.2;
    for (int i = 0; i < 2; ++i) {
        float diffuse = max(abs(dot(obj_normal, -lights[i])), 0.0);
        color += diffuse;
    }
    v_color = color;
}
";
    let fsh = r"
#version 150

in float v_color;
out vec4 out_frag_color;

void main(void) {
    vec3 base;
    if (gl_FrontFacing)
        base = vec3(1.0, 1.0, 1.0);
    else
        base = vec3(0.8, 0.3, 0.3);
    out_frag_color = vec4(v_color * base, 1.0);
}
";
    let prg = glium::Program::from_source(&glctx, vsh, fsh, None).unwrap();

    let f = std::fs::File::open("v2.obj").unwrap();
    let f = std::io::BufReader::new(f);
    let m = waveobj::Model::from_reader(f).unwrap();
    let model = m.get(0).unwrap();

    let mut vs = Vec::new();
    for face in model.faces() {

        let face = face.indices()
            .iter()
            .map(|&idx| {
                let v = model.vertex_by_index(idx);
                MVertex {
                    pos: *v.pos(),
                    normal: *v.normal(),
                }
            })
            .collect::<Vec<_>>();
        match face.len() {
            0 | 1 | 2 => { /* ??? */ }
            3 => {
                vs.extend(face);
            }
            4 => {
                vs.push(face[0]);
                vs.push(face[1]);
                vs.push(face[2]);

                vs.push(face[0]);
                vs.push(face[2]);
                vs.push(face[3]);
            }
            _ => {

            }
        }
    }

    *ctx = Some(MyContext { glctx, prg, vs, r: 0.0 });
}

fn gl_unrealize(_w: &gtk::GLArea, ctx: &Rc<RefCell<Option<MyContext>>>) {
    let mut ctx = ctx.borrow_mut();
    *ctx = None;
}

struct MyUniforms {
    m: cgmath::Matrix4<f32>,
    mnormal: cgmath::Matrix4<f32>,
    lights: [cgmath::Vector4<f32>; 2],
}

impl glium::uniforms::Uniforms for MyUniforms {
    fn visit_values<'a, F: FnMut(&str, glium::uniforms::UniformValue<'a>)>(&'a self, mut visit: F) {
        use glium::uniforms::UniformValue::*;

        visit("m", Mat4(array4x4(self.m)));
        visit("mnormal", Mat4(array4x4(self.mnormal)));
        visit("lights[0]", Vec4(array4(self.lights[0])));
        visit("lights[1]", Vec4(array4(self.lights[1])));
    }
}

fn gl_render(w: &gtk::GLArea, _gl: &gdk::GLContext, ctx: &Rc<RefCell<Option<MyContext>>>) -> gtk::Inhibit {
    let rect = w.allocation();
    let ctx = ctx.borrow();
    let ctx = ctx.as_ref().unwrap();
    let mut frm = glium::Frame::new(ctx.glctx.clone(), (rect.width as u32, rect.height as u32));

    use glium::Surface;

    frm.clear_color_and_depth((0.2, 0.2, 0.4, 1.0), 1.0);

    let vs = glium::VertexBuffer::new(&ctx.glctx, &ctx.vs).unwrap();
    let idxs = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
    //let idxs = glium::index::IndexBuffer::new(&ctx.glctx, glium::index::PrimitiveType::TrianglesList, &ctx.faces).unwrap();

    let ratio = (rect.width as f32) / (rect.height as f32);
    let persp = cgmath::perspective(cgmath::Deg(60.0), ratio, 10.0, 100.0);
    let r = cgmath::Matrix4::from_angle_y(cgmath::Deg(ctx.r));
    let t = cgmath::Matrix4::<f32>::from_translation(cgmath::Vector3::new(4.0, -10.0, -30.0));
    let s = cgmath::Matrix4::<f32>::from_scale(0.1);

    let light0 = cgmath::Vector4::new(-0.5, -0.4, -0.8, 0.0f32).normalize() * 0.55;
    let light1 = cgmath::Vector4::new(0.8, 0.2, 0.4, 0.0f32).normalize() * 0.25;

    let u = MyUniforms {
        m: persp * t * r * s,
        mnormal: r, //should be transpose of inverse
        lights: [light0, light1],
    };

    let dp = glium::DrawParameters {
        viewport: Some(glium::Rect { left: 0, bottom: 0, width: rect.width as u32, height: rect.height as u32}),
        blend: glium::Blend::alpha_blending(),
        depth: glium::Depth {
            test: glium::DepthTest::IfLess,
            write: true,
            .. Default::default()
        },
        .. Default::default()
    };
    frm.draw(&vs, &idxs, &ctx.prg, &u, &dp).unwrap();

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

struct MyContext {
    glctx: Rc<glium::backend::Context>,
    prg: glium::Program,
    vs: Vec<MVertex>,
    r: f32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

glium::implement_vertex!(MVertex, pos, normal);


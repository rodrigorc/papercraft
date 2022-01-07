use cgmath::InnerSpace;
use cgmath::conv::{array4x4, array4};
use gtk::prelude::*;
//use cgmath::prelude::*;
use gtk::gdk;
use std::rc::Rc;
use std::cell::RefCell;

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
uniform mat4 m;
uniform mat4 mnormal;
uniform vec4 light;
attribute vec3 pos;
attribute vec3 normal;

varying vec3 v_color;

void main(void) {
    gl_Position = m * vec4(pos, 1.0);
    vec4 obj_normal = mnormal * vec4(normal, 0.0);
    float c = dot(obj_normal, light);
    if (c < 0.0)
        v_color = vec3(-c, 0.0, 0.0);
    else
        v_color = vec3(0.0, 0.0, c);
}
";
    let fsh = r"
varying vec3 v_color;

void main(void) {
    gl_FragColor = vec4(v_color.rgb, 1.0);
}
";
    let prg = glium::Program::from_source(&glctx, vsh, fsh, None).unwrap();

    use std::io::BufRead;

    let f = std::fs::File::open("v2.obj").unwrap();
    let f = std::io::BufReader::new(f);
    let mut pos = Vec::new();
    let mut normals = Vec::new();
    let mut vs = Vec::new();
    for line in f.lines() {
        let line = line.unwrap();
        let line = line.trim();

        if line.is_empty() || line.starts_with("#") {
            continue;
        }
        let mut words = line.split(' ');
        let first = words.next().unwrap();
        match first {
            "o" => {
                let name = words.next().unwrap();
                dbg!(name);
            }
            "v" => {
                let x: f32 = words.next().unwrap().parse().unwrap();
                let y: f32 = words.next().unwrap().parse().unwrap();
                let z: f32 = words.next().unwrap().parse().unwrap();
                pos.push([x, y, z]);
            }
            "vt" => {
            }
            "vn" => {
                let x: f32 = words.next().unwrap().parse().unwrap();
                let y: f32 = words.next().unwrap().parse().unwrap();
                let z: f32 = words.next().unwrap().parse().unwrap();
                normals.push([x, y, z]);
            }
            "f" => {
                let words = words.collect::<Vec<_>>();
                match words.len() {
                    3 => {
                        for word in &words {
                            vs.push(parse_vertex(word, &pos, &normals));
                        }
                    }
                    4 => {
                        let vf = words.iter().map(|s| parse_vertex(s, &pos, &normals)).collect::<Vec<_>>();
                        vs.push(vf[0]);
                        vs.push(vf[1]);
                        vs.push(vf[2]);

                        vs.push(vf[0]);
                        vs.push(vf[2]);
                        vs.push(vf[3]);

                    }
                    _ => {} //TODO triangulate
                }
            }
            p => {
                println!("{}??", p);
            }
        }
    }
    *ctx = Some(MyContext { glctx, prg, vs, r: 0.0 });
}

fn parse_vertex(s: &str, pos: &[[f32; 3]], normals: &[[f32; 3]]) -> MVertex {
    let mut vals = s.split('/');
    let v = vals.next().unwrap().parse::<usize>().unwrap() - 1;
    let _t = vals.next().unwrap().parse::<usize>().unwrap() - 1;
    let n = vals.next().unwrap().parse::<usize>().unwrap() - 1;

    let v = pos[v];
    let n = normals[n];

    //let c = (n[0] * 0.5 + n[1] * 0.4 + n[2] * 0.2).max(0.0) + 0.4;

    MVertex {
        pos: v,
        normal: n,
    }
}

fn gl_unrealize(_w: &gtk::GLArea, ctx: &Rc<RefCell<Option<MyContext>>>) {
    let mut ctx = ctx.borrow_mut();
    *ctx = None;
}

fn gl_render(w: &gtk::GLArea, _gl: &gdk::GLContext, ctx: &Rc<RefCell<Option<MyContext>>>) -> gtk::Inhibit {
    let rect = w.allocation();
    let ctx = ctx.borrow();
    let ctx = ctx.as_ref().unwrap();
    let mut frm = glium::Frame::new(ctx.glctx.clone(), (rect.width as u32, rect.height as u32));

    use glium::Surface;

    frm.clear_color_and_depth((0.3, 0.3, 0.3, 1.0), 1.0);

    let vs = glium::VertexBuffer::new(&ctx.glctx, &ctx.vs).unwrap();
    let idxs = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
    //let idxs = glium::index::IndexBuffer::new(&ctx.glctx, glium::index::PrimitiveType::TrianglesList, &ctx.faces).unwrap();

    let ratio = (rect.width as f32) / (rect.height as f32);
    let persp = cgmath::perspective(cgmath::Deg(90.0), ratio, 10.0, 100.0);
    let r = cgmath::Matrix4::from_angle_y(cgmath::Deg(ctx.r));
    let t = cgmath::Matrix4::<f32>::from_translation(cgmath::Vector3::new(4.0, -10.0, -30.0));
    let s = cgmath::Matrix4::<f32>::from_scale(0.1);

    let light = cgmath::Vector4::new(0.5, 0.4, 0.8, 0.0f32).normalize();

    let u = glium::uniform!{
        m: array4x4(persp * t * r * s),
        mnormal: array4x4(r), //should be transpose of inverse
        light: array4(light),
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


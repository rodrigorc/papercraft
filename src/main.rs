use cgmath::{
    prelude::*,
    Deg, Rad,
};
use glib::clone;
use gtk::{
    prelude::*,
    gdk::{self, EventMask},
};

use std::{collections::HashMap, ops::ControlFlow, time::Duration, path::Path};
use std::rc::Rc;
use std::cell::RefCell;

mod waveobj;
mod paper;
mod util_3d;
mod util_gl;
mod glr;

use paper::Papercraft;

use util_3d::{Matrix3, Matrix4, Quaternion, Vector2, Point2, Point3, Vector3, Matrix2};
use util_gl::{Uniforms2D, Uniforms3D, MVertex3D, MVertex2D, MVertexQuad, MStatus, MSTATUS_UNSEL, MSTATUS_SEL, MSTATUS_HI, MVertex3DLine};

fn on_app_startup(app: &gtk::Application, args: Rc<RefCell<Option<String>>>) {
    dbg!("startup");
    let builder = gtk::Builder::from_string(include_str!("menu.ui"));
    let menu: gio::MenuModel = builder.object("appmenu").unwrap();
    app.set_menubar(Some(&menu));

    let persp = cgmath::perspective(Deg(60.0), 1.0, 1.0, 100.0);
    let trans_scene = Transformation3D::new(
        Vector3::new(0.0, 0.0, -30.0),
        Quaternion::one(),
        20.0,
        persp
    );
    let trans_paper = {
        let mt = Matrix3::from_translation(Vector2::new(-210.0/2.0, -297.0/2.0));
        let ms = Matrix3::from_scale(1.0);
        TransformationPaper {
            ortho: util_3d::ortho2d(1.0, 1.0),
            //mx: mt * ms * mr,
            mx: ms * mt,
        }
    };

    let wscene = gtk::GLArea::new();
    let wpaper = gtk::GLArea::new();
    let w = gtk::ApplicationWindow::new(app);

    let gui = Gui {
        top_window: w.clone(),
        wscene: wscene.clone(),
        wpaper: wpaper.clone(),
        gl_fixs: None,
    };
    let ctx = MyContext {
        gui,

        gl_objs: None,

        papercraft: Papercraft::empty(),
        texture_images: { let mut texs = HashMap::default(); texs.insert(String::new(), None); texs },

        material: None,
        selected_face: None,
        selected_edge: None,
        selected_islands: Vec::new(),
        grabbed_island: false,
        scroll_timer: None,

        last_cursor_pos: Vector2::zero(),

        trans_scene,
        trans_paper,
    };

    let ctx: Rc<RefCell<MyContext>> = Rc::new(RefCell::new(ctx));

    let aquit = gio::SimpleAction::new("quit", None);
    aquit.connect_activate(clone!(@weak app => move |_, _| app.quit() ));
    app.add_action(&aquit);

    let aopen = gio::SimpleAction::new("open", None);
    aopen.connect_activate(clone!(
        @strong w as top_window, @strong app =>
        move |_, _| {
            let dlg = gtk::FileChooserDialog::with_buttons(
                Some("Open model"),
                Some(&top_window),
                gtk::FileChooserAction::Open,
                &[
                    ("Cancel", gtk::ResponseType::Cancel),
                    ("Open", gtk::ResponseType::Accept)
                ]
            );
            dlg.set_current_folder(".");
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("Papercraft models"));
            filter.add_pattern("*.json");
            dlg.add_filter(&filter);
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("All files"));
            filter.add_pattern("*");
            dlg.add_filter(&filter);

            let res = dlg.run();
            let name = if res == gtk::ResponseType::Accept {
                dlg.filename()
            } else {
                None
            };
            unsafe { dlg.destroy(); }

            if let Some(name) = name {
                let file = gio::File::for_path(name);
                app.open(&[file], "");
            }
        }
    ));
    app.add_action(&aopen);

    let aimport = gio::SimpleAction::new("import", None);
    aimport.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let top_window = ctx.borrow().gui.top_window.clone();
            let dlg = gtk::FileChooserDialog::with_buttons(
                Some("Import OBJ"),
                Some(&top_window),
                gtk::FileChooserAction::Open,
                &[
                    ("Cancel", gtk::ResponseType::Cancel),
                    ("Open", gtk::ResponseType::Accept)
                ]
            );
            dlg.set_current_folder(".");
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("WaveObj models"));
            filter.add_pattern("*.obj");
            dlg.add_filter(&filter);
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("All files"));
            filter.add_pattern("*");
            dlg.add_filter(&filter);

            let res = dlg.run();
            let name = if res == gtk::ResponseType::Accept {
                dlg.filename()
            } else {
                None
            };
            unsafe { dlg.destroy(); }

            if let Some(name) = name {
                ctx.borrow_mut().import_waveobj(name);
            }
        }
    ));
    app.add_action(&aimport);


    w.set_default_size(800, 600);
    w.connect_destroy(clone!(
        @strong app =>
        move |_| {
            app.quit();
        }
    ));
    gl_loader::init_gl();
    gl::load_with(|s| gl_loader::get_proc_address(s) as _);

    wscene.set_events(EventMask::BUTTON_PRESS_MASK | EventMask::BUTTON_MOTION_MASK | EventMask::POINTER_MOTION_MASK | EventMask::SCROLL_MASK);
    wscene.set_has_depth_buffer(true);

    wscene.connect_button_press_event(clone!(
        @strong ctx =>
        move |w, ev| {
            w.grab_focus();
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            let pos = ev.position();
            let pos = Vector2::new(pos.0 as f32, pos.1 as f32);
            ctx.last_cursor_pos = pos;

            if ev.button() == 1 && ev.event_type() == gdk::EventType::ButtonPress {
                let selection = ctx.analyze_click(pos);
                match selection {
                    ClickResult::None | ClickResult::Face(_) => {
                        ctx.set_selection(selection, true, ev.state().contains(gdk::ModifierType::CONTROL_MASK));
                    }
                    ClickResult::Edge(i_edge, priority_face) => {
                        if ev.state().contains(gdk::ModifierType::SHIFT_MASK) {
                            ctx.try_join_strip(i_edge);
                        } else {
                            ctx.edge_toggle_cut(i_edge, priority_face);
                        }
                        ctx.paper_build();
                        ctx.scene_edge_build();
                        ctx.update_scene_face_selection();
                        ctx.gui.wscene.queue_render();
                        ctx.gui.wpaper.queue_render();
                    }
                }
            }
            Inhibit(false)
        }
    ));
    wscene.connect_scroll_event(clone!(
        @strong ctx =>
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
            ctx.gui.wscene.queue_render();
            Inhibit(true)
        }
    ));
    wscene.connect_motion_notify_event(clone!(
        @strong ctx =>
        move |_w, ev|  {
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            let pos = ev.position();
            let pos = Vector2::new(pos.0 as f32, pos.1 as f32);

            ctx.scene_motion_notify_event(pos, ev);
            Inhibit(true)
        }
    ));
    wscene.connect_realize(clone!(
        @strong ctx =>
        move |w| {
            scene_realize(w, &mut *ctx.borrow_mut());
        }
    ));
    wscene.connect_render(clone!(
        @strong ctx =>
        move |_w, _gl| {
            ctx.borrow_mut().scene_render();
            gtk::Inhibit(false)
        }
    ));
    wscene.connect_resize(clone!(
        @strong ctx =>
        move |_w, width, height| {
            if height <= 0 || width <= 0 {
                return;
            }
            let ratio = width as f32 / height as f32;
            ctx.borrow_mut().trans_scene.set_ratio(ratio);
        }
    ));

    wpaper.set_events(EventMask::BUTTON_PRESS_MASK | EventMask::BUTTON_RELEASE_MASK | EventMask::BUTTON_MOTION_MASK | EventMask::POINTER_MOTION_MASK | EventMask::SCROLL_MASK);
    wpaper.set_has_stencil_buffer(true);

    wpaper.connect_realize(clone!(
        @strong ctx =>
        move |w| paper_realize(w, &mut *ctx.borrow_mut())
    ));
    wpaper.connect_render(clone!(
        @strong ctx =>
        move |_w, _gl| {
            ctx.borrow_mut().paper_render();
            Inhibit(true)
        }
    ));
    wpaper.connect_resize(clone!(
        @strong ctx =>
        move |_w, width, height| {
            if height <= 0 || width <= 0 {
                return;
            }
            ctx.borrow_mut().trans_paper.ortho = util_3d::ortho2d(width as f32, height as f32);
        }
    ));

    wpaper.connect_button_press_event(clone!(
        @strong ctx =>
        move |w, ev|  {
            w.grab_focus();
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            let pos = ev.position();
            let pos = Vector2::new(pos.0 as f32, pos.1 as f32);
            ctx.last_cursor_pos = pos;
            if ev.button() == 1 && ev.event_type() == gdk::EventType::ButtonPress {
                let selection = ctx.analyze_click_paper(pos);
                match selection {
                    ClickResult::Face(_) => {
                        ctx.set_selection(selection, true, ev.state().contains(gdk::ModifierType::CONTROL_MASK));
                        ctx.grabbed_island = true;
                    }
                    ClickResult::Edge(i_edge, i_face) => {
                        //ctx.papercraft.edge_toggle_cut(i_edge, priority_face);
                        ctx.grabbed_island = false;
                        if ev.state().contains(gdk::ModifierType::SHIFT_MASK) {
                            ctx.try_join_strip(i_edge);
                        } else {
                            ctx.edge_toggle_cut(i_edge, i_face);
                        }
                        ctx.paper_build();
                        ctx.scene_edge_build();
                        ctx.update_scene_face_selection();
                        ctx.gui.wscene.queue_render();
                        ctx.gui.wpaper.queue_render();
                    }
                    ClickResult::None => {
                        ctx.grabbed_island = false;
                        ctx.set_selection(selection, true, ev.state().contains(gdk::ModifierType::CONTROL_MASK));
                    }
                }
            }
            Inhibit(true)
        }
    ));
    wpaper.connect_button_release_event(clone!(
        @strong ctx =>
        move |_w, _ev|  {
            let mut ctx = ctx.borrow_mut();
            ctx.grabbed_island = false;
            ctx.set_scroll_timer(None);
            Inhibit(true)
        }
    ));
    wpaper.connect_motion_notify_event(clone!(
        @strong ctx =>
        move |w, ev| {
            let rect = w.allocation();
            let pos = ev.position();
            let pos = Vector2::new(pos.0 as f32, pos.1 as f32);
            let state = ev.state();

            let grabbed = {
                let mut ctx = ctx.borrow_mut();
                ctx.paper_motion_notify_event(pos, state);
                ctx.grabbed_island
            };

            if grabbed {
                let delta = if pos.x < 5.0 {
                    Some(Vector2::new((-pos.x).max(5.0).min(25.0), 0.0))
                } else if pos.x > rect.width() as f32 - 5.0 {
                    Some(Vector2::new(-(pos.x - rect.width() as f32).max(5.0).min(25.0), 0.0))
                } else if pos.y < 5.0 {
                    Some(Vector2::new(0.0, (-pos.y).max(5.0).min(25.0)))
                } else if pos.y > rect.height() as f32 - 5.0 {
                    Some(Vector2::new(0.0, -(pos.y - rect.height() as f32).max(5.0).min(25.0)))
                } else {
                    None
                };
                if let Some(delta) = delta {
                    let f = clone!(
                        @strong ctx =>
                        move || {
                            let mut ctx = ctx.borrow_mut();
                            ctx.last_cursor_pos += delta;
                            ctx.trans_paper.mx = Matrix3::from_translation(delta) * ctx.trans_paper.mx;
                            ctx.paper_motion_notify_event(pos, state);
                            ctx.gui.wpaper.queue_render();
                            glib::Continue(true)
                        }
                    );
                    // do not wait for the timer for the first call
                    f();
                    let timer = glib::timeout_add_local(Duration::from_millis(50), f);
                    ctx.borrow_mut().set_scroll_timer(Some(timer));
                } else {
                    ctx.borrow_mut().set_scroll_timer(None);
                }
            }

            Inhibit(true)
        }
    ));
    wpaper.connect_scroll_event(clone!(
        @strong ctx =>
        move |_w, ev|  {
            let mut ctx = ctx.borrow_mut();
            let dz = match ev.direction() {
                gdk::ScrollDirection::Up => 1.1,
                gdk::ScrollDirection::Down => 1.0 / 1.1,
                _ => 1.0,
            };
            ctx.trans_paper.mx = Matrix3::from_scale(dz) * ctx.trans_paper.mx;
            ctx.gui.wpaper.queue_render();
            Inhibit(true)
        }
    ));

    let hbin = gtk::Paned::new(gtk::Orientation::Horizontal);
    hbin.pack1(&wscene, true, true);
    hbin.pack2(&wpaper, true, true);

    let toolbar = gtk::Toolbar::new();
    let btn = gtk::ToolButton::new(gtk::Widget::NONE, None);
    btn.set_action_name(Some("app.quit"));
    btn.set_icon_name(Some("application-exit"));
    toolbar.add(&btn);

    //app.lookup_action("quit").unwrap().downcast::<gio::SimpleAction>().unwrap().set_enabled(false);

    let vbin = gtk::Box::new(gtk::Orientation::Vertical, 0);
    w.add(&vbin);

    vbin.pack_start(&toolbar, false, true, 0);
    vbin.pack_start(&hbin, true, true, 0);

	app.connect_activate(clone!(
        @strong ctx =>
        move |_app| {
            dbg!("activate");
            let w = ctx.borrow().gui.top_window.clone();
            w.show_all();
            w.present();
    	}
    ));
	app.connect_open(clone!(
        @strong ctx =>
        move |_app, files, _hint| {
            dbg!("open");
            let f = &files[0];
            let (data, _) = f.load_contents(gio::Cancellable::NONE).unwrap();
            let papercraft: Papercraft = serde_json::from_reader(&data[..]).unwrap();

            let w = {
                let mut ctx = ctx.borrow_mut();
                ctx.papercraft = papercraft;

                // State
                ctx.material = None;
                ctx.selected_face = None;
                ctx.selected_edge = None;
                ctx.selected_islands = Vec::new();
                ctx.grabbed_island = false;
                ctx.scroll_timer = None;
                ctx.last_cursor_pos = Vector2::zero();

                ctx.gl_objs = None;

                ctx.gui.top_window.clone()
            };
            w.show_all();
            w.present();
        }
	));

    let args = args.borrow();
    if let Some(args) = &*args {
        ctx.borrow_mut().import_waveobj(args);
    }
}

fn main() {
    std::env::set_var("GTK_CSD", "0");
    //gtk::init().expect("gtk::init");

    let app = gtk::Application::new(None,
        gio::ApplicationFlags::HANDLES_OPEN | gio::ApplicationFlags::NON_UNIQUE
    );
    app.add_main_option("import", glib::Char::from(b'I'), glib::OptionFlags::NONE, glib::OptionArg::String, "Import a WaveOBJ file", None);
    let args = Rc::new(RefCell::new(None));
    app.connect_handle_local_options(clone!(
        @strong args =>
        move |_app, dict| {
            dbg!("local_option");
            //It should be a OsString but that gets an \0 at the end that breaks everything
            let s: Option<String> = dict.lookup("import").unwrap();
            dbg!(&s);
            *args.borrow_mut() = s;
            -1
        }
    ));
	app.connect_startup(clone!(
        @strong args =>
        move |app: &gtk::Application| {
            on_app_startup(app, args.clone());
        }
    ));
    app.run();

    /*
    return;

    w.show_all();
    w.present();
    gtk::main();
    #[cfg(xxx)]
    {
        let ctx = ctx.borrow();
        {
            let f = std::fs::File::create("a.json").unwrap();
            let f = std::io::BufWriter::new(f);
            serde_json::to_writer(f, &ctx.papercraft).unwrap();
        }
    }*/
}

fn scene_realize(w: &gtk::GLArea, ctx: &mut MyContext) {
    w.attach_buffers();
    ctx.build_gl_objects();
    ctx.gui.gl_fixs.as_mut().unwrap().vao_scene = Some(glr::VertexArray::generate().unwrap());
}

fn paper_realize(w: &gtk::GLArea, ctx: &mut MyContext) {
    w.attach_buffers();
    ctx.build_gl_objects();
    ctx.gui.gl_fixs.as_mut().unwrap().vao_paper = Some(glr::VertexArray::generate().unwrap());
}

struct GLFixedObjects {
    //VAOs are not shareable between contexts, so we need two, one for each window
    vao_scene: Option<glr::VertexArray>,
    vao_paper: Option<glr::VertexArray>,

    prg_scene_solid: glr::Program,
    prg_scene_line: glr::Program,
    prg_paper_solid: glr::Program,
    prg_paper_line: glr::Program,
    #[allow(dead_code)]
    prg_quad: glr::Program,

    #[allow(dead_code)]
    quad_vertices: glr::DynamicVertexArray<MVertexQuad>,
}

struct GLObjects {
    textures: HashMap<String, glr::Texture>,

    //GL objects that are rebuild with the model
    vertices: glr::DynamicVertexArray<MVertex3D>,
    vertices_sel: glr::DynamicVertexArray<MStatus>,
    vertices_edges_joint: glr::DynamicVertexArray<MVertex3DLine>,
    vertices_edges_cut: glr::DynamicVertexArray<MVertex3DLine>,

    paper_vertices: glr::DynamicVertexArray<MVertex2D>,
    paper_vertices_edge: glr::DynamicVertexArray<MVertex2D>,
    paper_vertices_edge_sel: glr::DynamicVertexArray<MVertex2D>,
    paper_vertices_tab: glr::DynamicVertexArray<MVertex2D>,
    paper_vertices_tab_edge: glr::DynamicVertexArray<MVertex2D>,

    paper_vertices_page: glr::DynamicVertexArray<MVertex2D>,
}

//Objects that persist a file open action
struct Gui {
    top_window: gtk::ApplicationWindow,
    wscene: gtk::GLArea,
    wpaper: gtk::GLArea,

    gl_fixs: Option<GLFixedObjects>,
}

struct MyContext {
    gui: Gui,

    gl_objs: Option<GLObjects>,

    // The model
    papercraft: Papercraft,
    texture_images: HashMap<String, Option<gdk_pixbuf::Pixbuf>>,

    // State
    material: Option<String>,
    selected_face: Option<paper::FaceIndex>,
    selected_edge: Option<paper::EdgeIndex>,
    selected_islands: Vec<paper::IslandKey>,
    grabbed_island: bool,
    scroll_timer: Option<glib::SourceId>,

    last_cursor_pos: Vector2,

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
    fn import_waveobj(&mut self, file_name: impl AsRef<Path>) {
        let f = std::fs::File::open(file_name).unwrap();
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
        let mut trans_scene = Transformation3D::new(
            Vector3::new(0.0, 0.0, -30.0),
            Quaternion::one(),
             20.0,
             persp
        );
        let rscene = self.gui.wscene.allocation();
        let ratio = rscene.width() as f32 / rscene.height() as f32;
        trans_scene.set_ratio(ratio);

        let trans_paper = {
            let mt = Matrix3::from_translation(Vector2::new(-210.0/2.0, -297.0/2.0));
            let ms = Matrix3::from_scale(1.0);
            let rpaper = self.gui.wpaper.allocation();
            let ortho = util_3d::ortho2d(rpaper.width() as f32, rpaper.height() as f32);
            TransformationPaper {
                ortho,
                //mx: mt * ms * mr,
                mx: ms * mt,
            }
        };

        let papercraft = Papercraft::new(model, &facemap);

        self.papercraft = papercraft;
        self.texture_images = texture_images;

        // State
        self.material = material;
        self.selected_face = None;
        self.selected_edge = None;
        self.selected_islands = Vec::new();
        self.grabbed_island = false;
        self.scroll_timer = None;
        self.last_cursor_pos = Vector2::zero();
        self.trans_scene = trans_scene;
        self.trans_paper = trans_paper;

        self.gl_objs = None;

        self.paper_build();
        self.scene_edge_build();
        self.update_scene_face_selection();

        self.gui.wscene.queue_render();
        self.gui.wpaper.queue_render();
    }

    fn scene_motion_notify_event(&mut self, pos: Vector2, ev: &gdk::EventMotion) {
        let delta = pos - self.last_cursor_pos;
        self.last_cursor_pos = pos;
        if ev.state().contains(gdk::ModifierType::BUTTON3_MASK) {
            // half angles
            let ang = delta / 200.0 / 2.0;
            let cosy = ang.x.cos();
            let siny = ang.x.sin();
            let cosx = ang.y.cos();
            let sinx = ang.y.sin();
            let roty = Quaternion::new(cosy, 0.0, siny, 0.0);
            let rotx = Quaternion::new(cosx, sinx, 0.0, 0.0);

            self.trans_scene.rotation = (roty * rotx * self.trans_scene.rotation).normalize();
            self.trans_scene.recompute_obj();
            self.gui.wscene.queue_render();
        } else if ev.state().contains(gdk::ModifierType::BUTTON2_MASK) {
            let delta = delta / 50.0;
            self.trans_scene.location += Vector3::new(delta.x, -delta.y, 0.0);
            self.trans_scene.recompute_obj();
            self.gui.wscene.queue_render();
        } else {
            let selection = self.analyze_click(pos);
            self.set_selection(selection, false, false);
        }
    }

    fn paper_motion_notify_event(&mut self, pos: Vector2, ev_state: gdk::ModifierType) {
        let delta = pos - self.last_cursor_pos;
        self.last_cursor_pos = pos;
        if ev_state.contains(gdk::ModifierType::BUTTON2_MASK) {
            self.trans_paper.mx = Matrix3::from_translation(delta) * self.trans_paper.mx;
            self.gui.wpaper.queue_render();
        } else if ev_state.contains(gdk::ModifierType::BUTTON1_MASK) && self.grabbed_island {
            if !self.selected_islands.is_empty() {
                for &i_island in &self.selected_islands {
                    if let Some(island) = self.papercraft.island_by_key_mut(i_island) {
                        let delta_scaled = <Matrix3 as Transform<Point2>>::inverse_transform_vector(&self.trans_paper.mx, delta).unwrap();
                        if ev_state.contains(gdk::ModifierType::SHIFT_MASK) {
                            // Rotate island
                            island.rotate(Deg(delta.y));
                        } else {
                            // Move island
                            island.translate(delta_scaled);
                        }
                    }
                }
                self.paper_build();
                self.gui.wpaper.queue_render();
            }
        } else {
            let selection = self.analyze_click_paper(pos);
            self.set_selection(selection, false, false);
        }
    }

    fn set_scroll_timer(&mut self, tmr: Option<glib::SourceId>) {
        if let Some(t) = self.scroll_timer.take() {
            t.remove();
        }
        self.scroll_timer = tmr;
    }

    fn analyze_click(&self, pos: Vector2) -> ClickResult {
        let rect = self.gui.wscene.allocation();
        let x = (pos.x / rect.width() as f32) * 2.0 - 1.0;
        let y = -((pos.y / rect.height() as f32) * 2.0 - 1.0);
        let click = Point3::new(x, y, 1.0);
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

    fn analyze_click_paper(&self, pos: Vector2) -> ClickResult {
        let rect = self.gui.wpaper.allocation();
        let x = (pos.x / rect.width() as f32) * 2.0 - 1.0;
        let y = -((pos.y / rect.height() as f32) * 2.0 - 1.0);
        let click = Point2::new(x, y);

        let mx = self.trans_paper.ortho * self.trans_paper.mx;
        let mx_inv = mx.invert().unwrap();
        let click = mx_inv.transform_point(click).to_vec();

        let mut edge_sel = None;
        let mut face_sel = None;

        for (_i_island, island) in self.papercraft.islands().collect::<Vec<_>>().into_iter().rev() {
            self.papercraft.traverse_faces(island,
                |i_face, face, fmx| {
                    let normal = self.papercraft.face_plane(face);
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
    fn set_selection(&mut self, selection: ClickResult, clicked: bool, add_to_sel: bool) {
        //TODO check if nothing changed
        match selection {
            ClickResult::None => {
                self.selected_edge = None;
                self.selected_face = None;
                if clicked && !add_to_sel {
                    self.selected_islands.clear();
                }
            }
            ClickResult::Face(i_face) => {
                self.selected_edge = None;
                self.selected_face = Some(i_face);
                if clicked {
                    let island = self.papercraft.island_by_face(i_face);
                    if add_to_sel {
                        if let Some(_n) = self.selected_islands.iter().position(|i| *i == island) {
                            //unselect the island?
                        } else {
                            self.selected_islands.push(island);
                        }
                    } else {
                        self.selected_islands = vec![island];
                    }
                }
            }
            ClickResult::Edge(i_edge, _) => {
                self.selected_edge = Some(i_edge);
                self.selected_face = None;
            }
        }
        self.paper_build();
        self.scene_edge_build();
        self.update_scene_face_selection();
        self.gui.wscene.queue_render();
        self.gui.wpaper.queue_render();
    }

    fn update_scene_face_selection(&mut self) {
        if let Some(gl_objs) = &mut self.gl_objs {
            let n = gl_objs.vertices_sel.len();
            for i in 0..n {
                gl_objs.vertices_sel[i] = MSTATUS_UNSEL;
            }
            for &sel_island in &self.selected_islands {
                if let Some(island) = self.papercraft.island_by_key(sel_island) {
                    self.papercraft.traverse_faces_no_matrix(island, |i_face_2| {
                        let pos = 3 * usize::from(i_face_2);
                        for i in pos .. pos + 3 {
                            gl_objs.vertices_sel[i] = MSTATUS_SEL;
                        }
                        ControlFlow::Continue(())
                    });
                }
            }
            if let Some(i_sel_face) = self.selected_face {
                for i_face_2 in self.papercraft.get_flat_faces(i_sel_face) {
                    let pos = 3 * usize::from(i_face_2);
                    for i in pos .. pos + 3 {
                        gl_objs.vertices_sel[i] = MSTATUS_HI;
                    }
                }
            }
        }
    }
    fn build_gl_objects(&mut self) {
        if self.gui.gl_fixs.is_none() {
            let prg_scene_solid = util_gl::program_from_source(include_str!("shaders/scene_solid.glsl"));
            let prg_scene_line = util_gl::program_from_source(include_str!("shaders/scene_line.glsl"));
            let prg_paper_solid = util_gl::program_from_source(include_str!("shaders/paper_solid.glsl"));
            let prg_paper_line = util_gl::program_from_source(include_str!("shaders/paper_line.glsl"));
            let prg_quad = util_gl::program_from_source(include_str!("shaders/quad.glsl"));

            let quad_vertices = glr::DynamicVertexArray::from(vec![
                MVertexQuad { pos: [-1.0, -1.0] },
                MVertexQuad { pos: [ 3.0, -1.0] },
                MVertexQuad { pos: [-1.0,  3.0] },
            ]);

            self.gui.gl_fixs = Some(GLFixedObjects {
                vao_scene: None,
                vao_paper: None,
                prg_scene_solid,
                prg_scene_line,
                prg_paper_solid,
                prg_paper_line,
                prg_quad,

                quad_vertices,
            });
        }
        if self.gl_objs.is_none() {
            let textures = self.texture_images
                .iter()
                .map(|(name, pixbuf)| {
                    let texture = match pixbuf {
                        None => {
                            // Empty texture is just a single white texel
                            let empty = glr::Texture::generate().unwrap();
                            unsafe {
                                gl::BindTexture(gl::TEXTURE_2D, empty.id());
                                gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGB as i32, 1, 1, 0, gl::RGB, gl::UNSIGNED_BYTE, [0xffu8, 0xffu8, 0xffu8].as_ptr() as *const _);
                            }
                            empty
                        }
                        Some(img) => {
                            let bytes = img.read_pixel_bytes().unwrap();
                            let width = img.width();
                            let height = img.height();
                            let format = match img.n_channels() {
                                4 => gl::RGBA,
                                3 => gl::RGB,
                                2 => gl::RG,
                                _ => gl::RED,
                            };

                            let tex = glr::Texture::generate().unwrap();
                            unsafe {
                                gl::BindTexture(gl::TEXTURE_2D, tex.id());
                                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
                                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
                                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR as i32);
                                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR_MIPMAP_LINEAR as i32);
                                gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA as i32, width, height, 0, format, gl::UNSIGNED_BYTE, bytes.as_ptr() as *const _);
                                gl::GenerateMipmap(gl::TEXTURE_2D);
                            }
                            tex
                        }
                    };
                    (name.clone(), texture)
                })
                .collect();

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
            let vertices = glr::DynamicVertexArray::from(vertices);
            let vertices_sel = glr::DynamicVertexArray::from(vec![MSTATUS_UNSEL; vertices.len()]);
            let vertices_edges_joint = glr::DynamicVertexArray::new();
            let vertices_edges_cut = glr::DynamicVertexArray::new();

            let paper_vertices = glr::DynamicVertexArray::new();
            let paper_vertices_edge = glr::DynamicVertexArray::new();
            let paper_vertices_edge_sel = glr::DynamicVertexArray::new();
            let paper_vertices_tab = glr::DynamicVertexArray::new();
            let paper_vertices_tab_edge = glr::DynamicVertexArray::new();

            let page_0 = MVertex2D {
                pos: Vector2::new(0.0, 0.0),
                uv: Vector2::zero(),
                color: [1.0, 1.0, 1.0, 1.0],
            };
            let page_2 = MVertex2D {
                pos: Vector2::new(210.0, 297.0),
                uv: Vector2::zero(),
                color: [1.0, 1.0, 1.0, 1.0],
            };
            let page_1 = MVertex2D {
                pos: Vector2::new(page_2.pos.x, 0.0),
                uv: Vector2::zero(),
                color: [1.0, 1.0, 1.0, 1.0],
            };
            let page_3 = MVertex2D {
                pos: Vector2::new(0.0, page_2.pos.y),
                uv: Vector2::zero(),
                color: [1.0, 1.0, 1.0, 1.0],
            };
            let paper_vertices_page = glr::DynamicVertexArray::from(vec![page_0, page_2, page_1, page_0, page_3, page_2]);

            self.gl_objs = Some(GLObjects {
                textures,
                vertices,
                vertices_sel,
                vertices_edges_joint,
                vertices_edges_cut,

                paper_vertices,
                paper_vertices_edge,
                paper_vertices_edge_sel,
                paper_vertices_tab,
                paper_vertices_tab_edge,
                paper_vertices_page,
            });

            self.scene_edge_build();
            self.paper_build();
            self.update_scene_face_selection();
        }
    }

    fn paper_draw_face(&self, face: &paper::Face, i_face: paper::FaceIndex, m: &Matrix3, selected: bool, hi: bool, args: &mut PaperDrawFaceArgs) {
        for i_v in face.index_vertices() {
            let v = &self.papercraft.model()[i_v];
            let p = self.papercraft.face_plane(face).project(&v.pos());
            let pos = m.transform_point(Point2::from_vec(p)).to_vec();
            args.vertices.push(MVertex2D {
                pos,
                uv: v.uv(),
                color: if hi { [1.0, 0.0, 0.0, 0.75] } else if selected { [0.0, 0.0, 1.0, 0.5] } else { [0.0, 0.0, 0.0, 0.0] },
            });
        }

        for (i_v0, i_v1, i_edge) in face.vertices_with_edges() {
            let edge = &self.papercraft.model()[i_edge];
            let edge_status = self.papercraft.edge_status(i_edge);
            let draw = match edge_status {
                paper::EdgeStatus::Hidden => false,
                paper::EdgeStatus::Cut(_) => true,
                paper::EdgeStatus::Joined => edge.face_sign(i_face),
            };
            let plane = self.papercraft.face_plane(face);
            let selected_edge = self.selected_edge == Some(i_edge);
            let v0 = &self.papercraft.model()[i_v0];
            let p0 = plane.project(&v0.pos());
            let pos0 = m.transform_point(Point2::from_vec(p0)).to_vec();

            let v1 = &self.papercraft.model()[i_v1];
            let p1 = plane.project(&v1.pos());
            let pos1 = m.transform_point(Point2::from_vec(p1)).to_vec();

            if draw {
                //Dotted lines are drawn for negative 3d angles (valleys) if the edge is joined or
                //cut with a label
                let dotted = if edge_status == paper::EdgeStatus::Joined ||
                                edge_status == paper::EdgeStatus::Cut(edge.face_sign(i_face)) {
                    let angle_3d = self.papercraft.model().edge_angle(i_edge);
                    angle_3d < Rad(0.0)
                } else {
                    false
                };
                let v = pos1 - pos0;
                let mut v0 = MVertex2D {
                    pos: pos0,
                    uv: Vector2::zero(),
                    color: [0.0, 0.0, 0.0, 1.0],
                };
                let mut v1 = MVertex2D {
                    pos: pos1,
                    uv: Vector2::zero(),
                    color: [0.0, 0.0, 0.0, 1.0],
                };
                if false && edge_status == paper::EdgeStatus::Joined {
                    let vn = v.normalize_to(0.01);
                    v0.pos -= vn;
                    v1.pos += vn;
                    let x = (0.015 / 2.0) / v.magnitude();
                    v0.uv.x = 0.5 - x;
                    v1.uv.x = 1.0 + x;
                } else if dotted {
                    v1.uv.x = v.magnitude();
                }
                args.vertices_edge.push(v0);
                args.vertices_edge.push(v1);
            }

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

            if edge_status == paper::EdgeStatus::Cut(edge.face_sign(i_face)) {
                let i_face_b = match edge.faces() {
                    (fa, Some(fb)) if i_face == fb => Some(fa),
                    (fa, Some(fb)) if i_face == fa => Some(fb),
                    _ => None
                };
                if let Some(i_face_b) = i_face_b {
                    let face_b = &self.papercraft.model()[i_face_b];

                    //swap the angles because this is from the POV of the other face
                    let (angle_1, angle_0) = self.papercraft.flat_face_angles(i_face_b, i_edge);
                    let angle_0 = Rad(angle_0.0.min(Rad::from(Deg(45.0)).0));
                    let angle_1 = Rad(angle_1.0.min(Rad::from(Deg(45.0)).0));

                    const TAB: f32 = 3.0;
                    let v = pos1 - pos0;
                    let tan_0 = angle_0.cot();
                    let tan_1 = angle_1.cot();
                    let v_len = v.magnitude();

                    let mut tab_h_0 = tan_0 * TAB;
                    let mut tab_h_1 = tan_1 * TAB;
                    let just_one_tri = v_len - tab_h_0 - tab_h_1 <= 0.0;
                    if just_one_tri {
                        let sum = tab_h_0 + tab_h_1;
                        tab_h_0 = tab_h_0 * v_len / sum;
                        //this will not be used, eventually
                        tab_h_1 = tab_h_1 * v_len / sum;
                    }
                    let v_0 = v * (tab_h_0 / v_len);
                    let v_1 = v * (tab_h_1 / v_len);
                    let n = Vector2::new(-v_0.y, v_0.x) / tan_0;
                    let mut p = [
                        MVertex2D {
                            pos: pos0,
                            uv: Vector2::zero(),
                            color: [0.0, 0.0, 0.0, 1.0],
                        },
                        MVertex2D {
                            pos: pos0 + n + v_0,
                            uv: Vector2::zero(),
                            color: [0.0, 0.0, 0.0, 1.0],
                        },
                        MVertex2D {
                            pos: pos1 + n - v_1,
                            uv: Vector2::zero(),
                            color: [0.0, 0.0, 0.0, 1.0],
                        },
                        MVertex2D {
                            pos: pos1,
                            uv: Vector2::zero(),
                            color: [0.0, 0.0, 0.0, 1.0],
                        },
                    ];
                    let p = if just_one_tri {
                        //The unneeded vertex is actually [2], so remove that copying the [3] over
                        p[2] = p[3];
                        args.vertices_tab_edge_buf.extend([p[0], p[1], p[1], p[2]]);
                        &mut p[..3]
                    } else {
                        args.vertices_tab_edge_buf.extend([p[0], p[1], p[1], p[2], p[2], p[3]]);
                        &mut p[..]
                    };

                    //Now we have to compute the texture coordinates of `p` in the adjacent face
                    let plane_b = self.papercraft.face_plane(face_b);
                    let vs_b = face_b.index_vertices().map(|v| {
                        let v = &self.papercraft.model()[v];
                        let p = plane_b.project(&v.pos());
                        (v, p)
                    });
                    let mx_b = m * self.papercraft.model().face_to_face_edge_matrix(self.papercraft.scale(), edge, face, face_b);
                    let mx_b_inv = mx_b.invert().unwrap();
                    let mx_basis = Matrix2::from_cols(vs_b[1].1 - vs_b[0].1, vs_b[2].1 - vs_b[0].1).invert().unwrap();

                    // mx_b_inv converts from paper to local face_b coordinates
                    // mx_basis converts from local face_b to edge-relative coordinates, where position of the tri vertices are [(0,0), (1,0), (0,1)]
                    // mxx do both convertions at once
                    let mxx = Matrix3::from(mx_basis) * mx_b_inv;

                    for px in p.iter_mut() {
                        //vlocal is in edge-relative coordinates, that can be used to interpolate between UVs
                        let vlocal = mxx.transform_point(Point2::from_vec(px.pos)).to_vec();
                        let uv0 = vs_b[0].0.uv();
                        let uv1 = vs_b[1].0.uv();
                        let uv2 = vs_b[2].0.uv();
                        px.uv = uv0 + vlocal.x * (uv1 - uv0) + vlocal.y * (uv2 - uv0);
                    }

                    if just_one_tri {
                        p[0].color = [1.0, 1.0, 1.0, 0.0];
                        p[1].color = [1.0, 1.0, 1.0, 1.0];
                        p[2].color = [1.0, 1.0, 1.0, 0.0];
                        args.vertices_tab_buf.extend([p[0], p[2], p[1]]);
                    } else {
                        p[0].color = [1.0, 1.0, 1.0, 0.0];
                        p[1].color = [1.0, 1.0, 1.0, 1.0];
                        p[2].color = [1.0, 1.0, 1.0, 1.0];
                        p[3].color = [1.0, 1.0, 1.0, 0.0];
                        args.vertices_tab_buf.extend([p[0], p[2], p[1], p[0], p[3], p[2]]);
                    }
                }
            }
        }
    }

    fn paper_build(&mut self) {
        //Maps VertexIndex in the model to index in vertices
        let mut args = PaperDrawFaceArgs::default();

        let flat_sel = match self.selected_face {
            None => Default::default(),
            Some(i_face) => self.papercraft.get_flat_faces(i_face),
        };
        for (i_island, island) in self.papercraft.islands() {
            let selected = self.selected_islands.contains(&i_island);
            self.papercraft.traverse_faces(island,
                |i_face, face, mx| {
                    let hi = flat_sel.contains(&i_face);
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
            gl_objs.paper_vertices.set(args.vertices);
            gl_objs.paper_vertices_edge.set(args.vertices_edge);
            gl_objs.paper_vertices_edge_sel.set(args.vertices_edge_sel);
            gl_objs.paper_vertices_tab.set(args.vertices_tab_buf);
            gl_objs.paper_vertices_tab_edge.set(args.vertices_tab_edge_buf);
        }
    }

    fn scene_render(&mut self) {
        self.gl_objs = None;
        self.build_gl_objects();
        let gl_objs = self.gl_objs.as_ref().unwrap();
        let gl_fixs = self.gui.gl_fixs.as_ref().unwrap();

        let light0 = Vector3::new(-0.5, -0.4, -0.8).normalize() * 0.55;
        let light1 = Vector3::new(0.8, 0.2, 0.4).normalize() * 0.25;

        let mat_name = self.material.as_deref().unwrap_or("");
        let texture = gl_objs.textures.get(mat_name)
            .unwrap_or_else(|| gl_objs.textures.get("").unwrap());

        let u = Uniforms3D {
            m: self.trans_scene.persp * self.trans_scene.obj,
            mnormal: self.trans_scene.mnormal, // should be transpose of inverse
            lights: [light0, light1],
            texture: 0,
        };

        unsafe {
            gl::ClearColor(0.2, 0.2, 0.4, 1.0);
            gl::ClearDepth(1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, texture.id());

            gl::BindVertexArray(gl_fixs.vao_scene.as_ref().unwrap().id());

            gl::PolygonOffset(1.0, 1.0);
            gl::Enable(gl::POLYGON_OFFSET_FILL);
            gl_fixs.prg_scene_solid.draw(&u, (&gl_objs.vertices, &gl_objs.vertices_sel), gl::TRIANGLES);

            gl::LineWidth(1.0);
            gl::Disable(gl::LINE_SMOOTH);
            gl_fixs.prg_scene_line.draw(&u, &gl_objs.vertices_edges_joint, gl::LINES);

            gl::LineWidth(3.0);
            gl::Enable(gl::LINE_SMOOTH);
            gl_fixs.prg_scene_line.draw(&u, &gl_objs.vertices_edges_cut, gl::LINES);
        }
    }

    fn paper_render(&mut self) {
        self.build_gl_objects();
        let gl_objs = self.gl_objs.as_ref().unwrap();
        let gl_fixs = self.gui.gl_fixs.as_ref().unwrap();

        let mat_name = self.material.as_deref().unwrap_or("");
        let texture = gl_objs.textures.get(mat_name)
            .unwrap_or_else(|| gl_objs.textures.get("").unwrap());

        let u = Uniforms2D {
            m: self.trans_paper.ortho * self.trans_paper.mx,
            texture: 0,
            frac_dash: 0.5,
        };

        unsafe {
            gl::ClearColor(0.7, 0.7, 0.7, 1.0);
            gl::ClearDepth(1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, texture.id());

            gl::BindVertexArray(gl_fixs.vao_paper.as_ref().unwrap().id());

            gl_fixs.prg_paper_solid.draw(&u, &gl_objs.paper_vertices_page, gl::TRIANGLES);

            // Textured faces
            gl_fixs.prg_paper_solid.draw(&u, &gl_objs.paper_vertices, gl::TRIANGLES);

            // Solid Tabs
            gl_fixs.prg_paper_solid.draw(&u, &gl_objs.paper_vertices_tab, gl::TRIANGLES);

            // Line Tabs
            gl::Disable(gl::LINE_SMOOTH);
            gl::LineWidth(1.0);
            gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_tab_edge, gl::LINES);

            // Creases
            gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_edge, gl::LINES);

            // - Selected edge
            if self.selected_edge.is_some() {
                gl::Enable(gl::LINE_SMOOTH);
                gl::LineWidth(5.0);
                gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_edge_sel, gl::LINES);
            }
        }
    }

    fn scene_edge_build(&mut self) {
        if let Some(gl_objs) = &mut self.gl_objs {
            let mut edges_joint = Vec::new();
            let mut edges_cut = Vec::new();
            for (i_edge, edge) in self.papercraft.model().edges() {
                let selected = self.selected_edge == Some(i_edge);
                let status = self.papercraft.edge_status(i_edge);
                if status == paper::EdgeStatus::Hidden {
                    continue;
                }
                let cut = matches!(self.papercraft.edge_status(i_edge), paper::EdgeStatus::Cut(_));
                let color = match (selected, cut) {
                    (true, false) => [0.0, 0.0, 1.0, 1.0],
                    (true, true) => [0.5, 0.5, 1.0, 1.0],
                    (false, false) => [0.0, 0.0, 0.0, 1.0],
                    (false, true) => [1.0, 1.0, 1.0, 1.0],
                };
                let p0 = self.papercraft.model()[edge.v0()].pos();
                let p1 = self.papercraft.model()[edge.v1()].pos();

                let edges = if cut { &mut edges_cut } else { &mut edges_joint };
                edges.push(MVertex3DLine { pos: p0, color, top: selected as u8 });
                edges.push(MVertex3DLine { pos: p1, color, top: selected as u8 });
            }
            gl_objs.vertices_edges_joint.set(edges_joint);
            gl_objs.vertices_edges_cut.set(edges_cut);
        }
    }

    fn edge_toggle_cut(&mut self, i_edge: paper::EdgeIndex, priority_face: Option<paper::FaceIndex>) {
        let renames = self.papercraft.edge_toggle_cut(i_edge, priority_face);
        self.islands_renamed(&renames);
    }
    fn try_join_strip(&mut self, i_edge: paper::EdgeIndex) {
        let renames = self.papercraft.try_join_strip(i_edge);
        self.islands_renamed(&renames);
    }
    fn islands_renamed(&mut self, renames: &HashMap<paper::IslandKey, paper::IslandKey>) {
        for x in &mut self.selected_islands {
            while let Some(n) = renames.get(x) {
                *x = *n;
            }
        }
    }
}


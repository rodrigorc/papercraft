#![allow(unused_imports, dead_code)]

use std::{num::NonZeroU32, ffi::CString, time::Instant, rc::{Rc, Weak}, cell::RefCell, path::Path};

use glow::HasContext;
use glutin::{prelude::*, config::{ConfigTemplateBuilder, Config}, display::GetGlDisplay, context::{ContextAttributesBuilder, ContextApi}, surface::{SurfaceAttributesBuilder, WindowSurface, Surface}};
use glutin_winit::DisplayBuilder;
use imgui_winit_support::WinitPlatform;
use raw_window_handle::HasRawWindowHandle;
use winit::{event_loop::{EventLoopBuilder, EventLoop}, window::{WindowBuilder, Window}};


mod waveobj;
mod paper;
mod glr;
mod util_3d;
mod util_gl;
mod main_ui;
//mod options_dlg;

use main_ui::*;


use paper::{Papercraft, Model, PaperOptions, Face, EdgeStatus, JoinResult, IslandKey, FaceIndex, MaterialIndex, EdgeIndex, TabStyle};
use glr::Rgba;
use util_3d::{Matrix3, Matrix4, Quaternion, Vector2, Point2, Point3, Vector3, Matrix2};
use util_gl::{Uniforms2D, Uniforms3D, UniformQuad, MVertex3D, MVertex2D, MStatus3D, MSTATUS_UNSEL, MSTATUS_SEL, MSTATUS_HI, MVertex3DLine, MVertex2DColor, MVertex2DLine, MStatus2D};

use glr::{BinderRenderbuffer, BinderDrawFramebuffer, BinderReadFramebuffer};

fn main() {
    let event_loop = EventLoopBuilder::new().build();

    let window_builder = WindowBuilder::new();
    let template = ConfigTemplateBuilder::new()
        .with_depth_size(16)
        .with_stencil_size(8);

    let display_builder = DisplayBuilder::new()
        .with_window_builder(Some(window_builder));

    let (window, gl_config) = display_builder
        .build(&event_loop, template, |configs| {
            configs
                .reduce(|cfg1, cfg2| {
                    if cfg2.num_samples() > cfg1.num_samples() {
                        cfg2
                    } else {
                        cfg1
                    }
                })
                .unwrap()
        })
        .unwrap();

    let raw_window_handle = window.as_ref().map(|window| window.raw_window_handle());
    let gl_display = gl_config.display();
    let context_attributes = ContextAttributesBuilder::new()
        .build(raw_window_handle);
    let fallback_context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::Gles(None))
        .build(raw_window_handle);

    let mut not_current_gl_context = Some(unsafe {
        gl_display
            .create_context(&gl_config, &context_attributes)
            .unwrap_or_else(|_| {
                gl_display
                    .create_context(&gl_config, &fallback_context_attributes)
                    .expect("failed to create context")
            })
    });
    let window = window.unwrap();
    let gl_window = GlWindow::new(window, &gl_config);
    let gl_context = not_current_gl_context
        .take()
        .unwrap()
        .make_current(&gl_window.surface)
        .unwrap();

    let mut imgui_context = imgui::Context::create();
    imgui_context.set_ini_filename(None);
    let mut winit_platform = WinitPlatform::init(&mut imgui_context);
    winit_platform.attach_window(
        imgui_context.io_mut(),
        &gl_window.window,
        imgui_winit_support::HiDpiMode::Rounded,
    );
    //imgui_context.set_clipboard_backend(MyClip);
    imgui_context
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);
    imgui_context.io_mut().font_global_scale = (1.0 / winit_platform.hidpi_factor()) as f32;

    let gl = unsafe {
        let dsp = gl_context.display();
        gl::load_with(|s| dsp.get_proc_address(&CString::new(s).unwrap()));
        glow::Context::from_loader_function(|s| dsp.get_proc_address(&CString::new(s).unwrap()).cast())
    };
    let mut ig_renderer = imgui_glow_renderer::AutoRenderer::initialize(gl, &mut imgui_context)
        .expect("failed to create renderer");

    let mut last_frame = Instant::now();

    // Initialize papaercraft status
    let sz_dummy = Vector2::new(1.0, 1.0);
    let fs = std::fs::File::open("examples/pikachu.craft").unwrap();
    let fs = std::io::BufReader::new(fs);
    let papercraft = Papercraft::load(fs).unwrap();
    let data = PapercraftContext::from_papercraft(
        papercraft,
        Some(&std::path::PathBuf::from("test")),
        sz_dummy,
        sz_dummy
    );

    let gl_fixs = build_gl_fixs().unwrap();
    let ctx = Rc::new_cyclic(|this| {
        RefCell::new(GlobalContext {
            this: this.clone(),
            gl_fixs,
            data,
            dragging_left_mouse: false,
            scene_ui_status: Canvas3dStatus::default(),
            paper_ui_status: Canvas3dStatus::default(),
        })
    });

    event_loop.run(move |event, _, control_flow| {
        match event {
            winit::event::Event::NewEvents(_) => {
                let now = Instant::now();
                imgui_context
                    .io_mut()
                    .update_delta_time(now.duration_since(last_frame));
                last_frame = now;
            }
            winit::event::Event::MainEventsCleared => {
                winit_platform
                    .prepare_frame(imgui_context.io_mut(), &gl_window.window)
                    .unwrap();
                gl_window.window.request_redraw();
            }
            winit::event::Event::RedrawEventsCleared => {
                *control_flow = winit::event_loop::ControlFlow::Poll;
            }
            winit::event::Event::RedrawRequested(_) => {
                // The renderer assumes you'll be clearing the buffer yourself
                let gl = ig_renderer.gl_context();
                unsafe {
                    gl.clear_color(0.0, 0.0, 0.0, 1.0);
                    gl.clear(glow::COLOR_BUFFER_BIT);
                };

                let ui = imgui_context.frame();
                {
                    let _s1 = ui.push_style_var(imgui::StyleVar::WindowPadding([0.0, 0.0]));
                    let _s2 = ui.push_style_var(imgui::StyleVar::WindowRounding(0.0));

                    let _w = ui.window("xxx")
                        .position([0.0, 0.0], imgui::Condition::Always)
                        .size(ui.io().display_size, imgui::Condition::Always)
                        .flags(
                            imgui::WindowFlags::NO_DECORATION |
                            imgui::WindowFlags::NO_RESIZE |
                            imgui::WindowFlags::MENU_BAR |
                            imgui::WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS
                        )
                        .begin();

                    drop((_s2, _s1));
                    let mut ctx = ctx.borrow_mut();
                    ctx.build_ui(&ui);
                    gl_window.window.set_title(&ctx.title());
                }

                winit_platform.prepare_render(&ui, &gl_window.window);
                let draw_data = imgui_context.render();

                // This is the only extra render step to add
                ig_renderer
                    .render(draw_data)
                    .expect("error rendering imgui");
                gl_window.surface.swap_buffers(&gl_context).unwrap();
            }
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = winit::event_loop::ControlFlow::Exit;
            }
            event => {
                winit_platform.handle_event(imgui_context.io_mut(), &gl_window.window, &event);
            }
        }
    });
}


pub struct GlWindow {
    // The surface must be dropped before the window.
    pub surface: Surface<WindowSurface>,
    pub window: Window,
}

impl GlWindow {
    pub fn new(window: Window, config: &Config) -> Self {
        let (width, height): (u32, u32) = window.inner_size().into();
        let raw_window_handle = window.raw_window_handle();
        let attrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            raw_window_handle,
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
        );

        let surface = unsafe { config.display().create_window_surface(config, &attrs).unwrap() };

        Self { window, surface }
    }
}

fn build_gl_fixs() -> Result<GLFixedObjects> {
    let prg_scene_solid = util_gl::program_from_source(include_str!("shaders/scene_solid.glsl")).with_context(|| "scene_solid")?;
    let prg_scene_line = util_gl::program_from_source(include_str!("shaders/scene_line.glsl")).with_context(|| "scene_line")?;
    let prg_paper_solid = util_gl::program_from_source(include_str!("shaders/paper_solid.glsl")).with_context(|| "paper_solid")?;
    let prg_paper_line = util_gl::program_from_source(include_str!("shaders/paper_line.glsl")).with_context(|| "paper_line")?;
    let prg_quad = util_gl::program_from_source(include_str!("shaders/quad.glsl")).with_context(|| "quad")?;

    let vao_scene = glr::VertexArray::generate();
    let vao_paper = glr::VertexArray::generate();
    let fbo_paper = None; //Some(glr::Framebuffer::generate());
    let rbo_paper = None; //Some((glr::Renderbuffer::generate(), glr::Renderbuffer::generate()));

    Ok(GLFixedObjects {
        vao_scene,
        vao_paper,
        fbo_paper,
        rbo_paper,
        prg_scene_solid,
        prg_scene_line,
        prg_paper_solid,
        prg_paper_line,
        prg_quad,
    })
}

struct GLFixedObjects {
    //VAOs are not shareable between contexts, so we need two, one for each window
    vao_scene: glr::VertexArray,
    vao_paper: glr::VertexArray,
    fbo_paper: Option<glr::Framebuffer>,
    rbo_paper: Option<(glr::Renderbuffer, glr::Renderbuffer)>, //color, stencil

    prg_scene_solid: glr::Program,
    prg_scene_line: glr::Program,
    prg_paper_solid: glr::Program,
    prg_paper_line: glr::Program,
    prg_quad: glr::Program,
}

struct GlobalContext {
    this: Weak<RefCell<GlobalContext>>,
    gl_fixs: GLFixedObjects,
    data: PapercraftContext,
    dragging_left_mouse: bool,
    scene_ui_status: Canvas3dStatus,
    paper_ui_status: Canvas3dStatus,
}

impl GlobalContext {
    fn build_ui(&mut self, ui: &imgui::Ui) {
        ui.menu_bar(|| {
            ui.menu("File", || {});
        });
        //ui.show_demo_window(&mut true);

        let size = Vector2::from(ui.content_region_avail());

        self.build_scene(ui, size.x / 2.0);
        ui.same_line();
        self.build_paper(ui);
    }

    fn build_scene(&mut self, ui: &imgui::Ui, width: f32) {
        if let Some(_scene) = ui.child_window("scene")
            //.size([300.0, 300.0], imgui::Condition::Once)
            //.movable(true)
            .size([width, 0.0])
            .border(true)
            .begin()
        {
            let pos = Vector2::from(ui.cursor_screen_pos());
            let size = Vector2::from(ui.content_region_avail());
            let dsp_size = Vector2::from(ui.io().display_size);
            let ratio = size.x / size.y;
            let pos_y2 = dsp_size.y - pos.y - size.y;
            let wnd_x = -1.0 - 2.0 * pos.x / size.x;
            let wnd_y = -1.0 - 2.0 * pos_y2 / size.y;
            let wnd_width = 2.0 * dsp_size.x / size.x;
            let wnd_height = 2.0 * dsp_size.y / size.y;

            let mouse_pos = Vector2::from(ui.io().mouse_pos) - pos;
            let mut rebuild = RebuildFlags::empty();
            let mut ev_state = gdk::ModifierType::empty();
            if ui.io().key_shift {
                ev_state.insert(gdk::ModifierType::SHIFT_MASK);
            }
            if ui.io().key_ctrl {
                ev_state.insert(gdk::ModifierType::CONTROL_MASK);
            }

            canvas3d(ui, &mut self.scene_ui_status);

            match &self.scene_ui_status.action {
                Canvas3dAction::Hovering => {
                    let flags = self.data.scene_motion_notify_event(size, mouse_pos, ev_state);
                    rebuild.insert(flags);
                    'zoom: {
                        let dz = match ui.io().mouse_wheel {
                            x if x < 0.0 => 1.0 / 1.1,
                            x if x > 0.0 => 1.1,
                            _ => break 'zoom,
                        };
                        self.data.trans_scene.scale *= dz;
                        self.data.trans_scene.recompute_obj();
                        rebuild.insert(RebuildFlags::SCENE_REDRAW);
                    }
                }
                Canvas3dAction::Clicked(imgui::MouseButton::Left) => {
                    ev_state.insert(gdk::ModifierType::BUTTON1_MASK);
                    let selection = self.data.scene_analyze_click(self.data.mode, size, mouse_pos);
                    match (self.data.mode, selection) {
                        (MouseMode::Edge, ClickResult::Edge(i_edge, i_face)) => {
                            let undo = if ev_state.contains(gdk::ModifierType::SHIFT_MASK) {
                                self.data.try_join_strip(i_edge)
                            } else {
                                self.data.edge_toggle_cut(i_edge, i_face)
                            };
                            if let Some(undo) = undo {
                                self.push_undo_action(undo);
                            }
                            rebuild.insert(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                        }
                        (MouseMode::Tab, ClickResult::Edge(i_edge, _)) => {
                            self.push_undo_action(vec![UndoAction::TabToggle { i_edge } ]);
                            self.data.papercraft.edge_toggle_tab(i_edge);
                            rebuild.insert(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                        }
                        (_, ClickResult::Face(f)) => {
                            let flags = self.data.set_selection(ClickResult::Face(f), true, ev_state.contains(gdk::ModifierType::CONTROL_MASK));
                            rebuild.insert(flags);
                        }
                        (_, ClickResult::None) => {
                            let flags = self.data.set_selection(ClickResult::None, true, ev_state.contains(gdk::ModifierType::CONTROL_MASK));
                            rebuild.insert(flags);
                        }
                        _ => {}
                    };
                }
                Canvas3dAction::Dragging(bt) => {
                    match bt {
                        imgui::MouseButton::Left => ev_state.insert(gdk::ModifierType::BUTTON1_MASK),
                        imgui::MouseButton::Right => ev_state.insert(gdk::ModifierType::BUTTON2_MASK),
                        _ => ()
                    }
                    let flags = self.data.scene_motion_notify_event(size, mouse_pos, ev_state);
                    rebuild.insert(flags);
                }
                Canvas3dAction::Clicked(_) | Canvas3dAction::None => {}
            }
            self.add_rebuild(rebuild);

            let mx_ortho = cgmath::ortho(
                wnd_x, wnd_x + wnd_width,
                wnd_y, wnd_y + wnd_height,
                1.0, -1.0
            );
            self.data.trans_scene.persp = cgmath::perspective(Deg(60.0), 1.0, 1.0, 100.0);
            let f = self.data.trans_scene.persp[1][1];
            self.data.trans_scene.persp[0][0] = f / ratio;
            self.data.trans_scene.persp_inv = self.data.trans_scene.persp.invert().unwrap();

            self.data.pre_render();
            let draws = ui.get_window_draw_list();
            draws.add_callback({
                let this = self.this.clone();
                move || {
                    let this = this.upgrade().unwrap();
                    let mut this = this.borrow_mut();

                    if pos.y >= dsp_size.y || pos.x >= dsp_size.x {
                        return;
                    }

                    unsafe {
                        let _backup = BackupGlConfig::backup();

                        gl::Scissor(
                            pos.x as i32,
                            pos_y2 as i32,
                            size.x as i32,
                            size.y as i32,
                        );

                        this.scene_render(mx_ortho);
                    }
                }
            }).build();
        }
    }

    fn build_paper(&mut self, ui: &imgui::Ui) {
        if let Some(_paper) = ui.child_window("paper")
            //.size([300.0, 300.0], imgui::Condition::Once)
            //.movable(true)
            .size([0.0, 0.0])
            .border(true)
            .begin()
        {
            let pos = Vector2::from(ui.cursor_screen_pos());
            let size = Vector2::from(ui.content_region_avail());
            let dsp_size = Vector2::from(ui.io().display_size);
            //let ratio = size.x / size.y;
            let pos_y2 = dsp_size.y - pos.y - size.y;
            let wnd_x = -1.0 - 2.0 * pos.x / size.x;
            let wnd_y = -1.0 - 2.0 * pos_y2 / size.y;
            let wnd_width = 2.0 * dsp_size.x / size.x;
            let wnd_height = 2.0 * dsp_size.y / size.y;

            let mouse_pos = Vector2::from(ui.io().mouse_pos) - pos;
            let mut rebuild = RebuildFlags::empty();
            let mut ev_state = gdk::ModifierType::empty();
            if ui.io().key_shift {
                ev_state.insert(gdk::ModifierType::SHIFT_MASK);
            }
            if ui.io().key_ctrl {
                ev_state.insert(gdk::ModifierType::CONTROL_MASK);
            }

            canvas3d(ui, &mut self.paper_ui_status);

            match &self.paper_ui_status.action {
                Canvas3dAction::Hovering => {
                    let flags = self.data.paper_motion_notify_event(size, mouse_pos, ev_state);
                    rebuild.insert(flags);

                    'zoom: {
                        let dz = match ui.io().mouse_wheel {
                            x if x < 0.0 => 1.0 / 1.1,
                            x if x > 0.0 => 1.1,
                            _ => break 'zoom,
                        };
                        let pos = mouse_pos - size / 2.0;
                        self.data.trans_paper.mx = Matrix3::from_translation(pos) * Matrix3::from_scale(dz) * Matrix3::from_translation(-pos) * self.data.trans_paper.mx;
                        rebuild.insert(RebuildFlags::PAPER_REDRAW);
                    }
                }
                Canvas3dAction::Clicked(imgui::MouseButton::Left) => {
                    ev_state.insert(gdk::ModifierType::BUTTON1_MASK);
                    self.data.rotation_center = None;

                    let selection = self.data.paper_analyze_click(self.data.mode, size, mouse_pos);
                    match (self.data.mode, selection) {
                        (MouseMode::Edge, ClickResult::Edge(i_edge, i_face)) => {
                            self.data.grabbed_island = false;

                            let undo = if ev_state.contains(gdk::ModifierType::SHIFT_MASK) {
                                self.data.try_join_strip(i_edge)
                            } else {
                                self.data.edge_toggle_cut(i_edge, i_face)
                            };
                            if let Some(undo) = undo {
                                self.push_undo_action(undo);
                            }
                            rebuild.insert(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                        }
                        (MouseMode::Tab, ClickResult::Edge(i_edge, _)) => {
                            self.push_undo_action(vec![UndoAction::TabToggle { i_edge } ]);
                            self.data.papercraft.edge_toggle_tab(i_edge);
                            rebuild.insert(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                        }
                        (_, ClickResult::Face(f)) => {
                            rebuild.insert(self.data.set_selection(ClickResult::Face(f), true, ev_state.contains(gdk::ModifierType::CONTROL_MASK)));
                            let undo_action = self.data.selected_islands
                                .iter()
                                .map(|&i_island| {
                                    let island = self.data.papercraft.island_by_key(i_island).unwrap();
                                    UndoAction::IslandMove { i_root: island.root_face(), prev_rot: island.rotation(), prev_loc: island.location() }
                                })
                                .collect();
                            self.push_undo_action(undo_action);
                            self.data.grabbed_island = true;
                        }
                        (_, ClickResult::None) => {
                            rebuild.insert(self.data.set_selection(ClickResult::None, true, ev_state.contains(gdk::ModifierType::CONTROL_MASK)));
                            self.data.grabbed_island = false;
                        }
                        _ => {}
                    }
                }
                Canvas3dAction::Dragging(bt) => {
                    match bt {
                        imgui::MouseButton::Left => ev_state.insert(gdk::ModifierType::BUTTON1_MASK),
                        imgui::MouseButton::Right => ev_state.insert(gdk::ModifierType::BUTTON2_MASK),
                        _ => ()
                    }
                    let flags = self.data.paper_motion_notify_event(size, mouse_pos, ev_state);
                    rebuild.insert(flags);
                }
                Canvas3dAction::Clicked(_) | Canvas3dAction::None => {}

            }
            self.add_rebuild(rebuild);

            let mx_ortho = cgmath::ortho(
                wnd_x, wnd_x + wnd_width,
                wnd_y, wnd_y + wnd_height,
                1.0, -1.0
            );
            let mx_ortho = Matrix3::new(
                mx_ortho[0][0], mx_ortho[0][1], mx_ortho[0][3],
                mx_ortho[1][0], mx_ortho[1][1], mx_ortho[1][3],
                mx_ortho[3][0], mx_ortho[3][1], mx_ortho[3][3],
            );

            self.data.trans_paper.ortho = util_3d::ortho2d(size.x, size.y);

            self.data.pre_render();
            let draws = ui.get_window_draw_list();
            draws.add_callback({
                let this = self.this.clone();
                move || {
                    let this = this.upgrade().unwrap();
                    let mut this = this.borrow_mut();

                    unsafe {
                        let _backup = BackupGlConfig::backup();

                        gl::Scissor(
                            pos.x as i32,
                            pos_y2 as i32,
                            size.x as i32,
                            size.y as i32,
                        );
                        this.paper_render(size, mx_ortho);
                    }
            }
            }).build();
        }
    }

    fn scene_render(&mut self, mx_gui: Matrix4) {
        let gl_objs = self.data.gl_objs.as_ref().unwrap();
        let gl_fixs = &self.gl_fixs;

        let light0 = Vector3::new(-0.5, -0.4, -0.8).normalize() * 0.55;
        let light1 = Vector3::new(0.8, 0.2, 0.4).normalize() * 0.25;

        let mut u = Uniforms3D {
            m: mx_gui * self.data.trans_scene.persp * self.data.trans_scene.view,
            mnormal: self.data.trans_scene.mnormal, // should be transpose of inverse
            lights: [light0, light1],
            tex: 0,
            line_top: 0,
            texturize: 0,
        };
        unsafe {
            gl::ClearColor(0.2, 0.2, 0.4, 1.0);
            gl::ClearDepth(1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::BLEND);
            gl::Enable(gl::DEPTH_TEST);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            gl::BindVertexArray(gl_fixs.vao_scene.id());
            if let (Some(tex), true) = (&gl_objs.textures, self.data.show_textures) {
                gl::ActiveTexture(gl::TEXTURE0);
                gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex.id());
                u.texturize = 1;
            }

            gl::PolygonOffset(1.0, 1.0);
            gl::Enable(gl::POLYGON_OFFSET_FILL);

            gl_fixs.prg_scene_solid.draw(&u, (&gl_objs.vertices, &gl_objs.vertices_sel), gl::TRIANGLES);

            if self.data.show_3d_lines {
                //Joined edges
                gl::LineWidth(1.0);
                gl::Disable(gl::LINE_SMOOTH);
                gl_fixs.prg_scene_line.draw(&u, &gl_objs.vertices_edge_joint, gl::LINES);

                //Cut edges
                gl::LineWidth(3.0);
                gl::Enable(gl::LINE_SMOOTH);
                gl_fixs.prg_scene_line.draw(&u, &gl_objs.vertices_edge_cut, gl::LINES);
            }

            //Selected edge
            if self.data.selected_edge.is_some() {
                gl::LineWidth(5.0);
                gl::Enable(gl::LINE_SMOOTH);
                if self.data.xray_selection {
                    u.line_top = 1;
                }
                gl_fixs.prg_scene_line.draw(&u, &gl_objs.vertices_edge_sel, gl::LINES);
            }
        }
    }
    fn paper_render(&mut self, size: Vector2, mx_gui: Matrix3) {
        let gl_objs = self.data.gl_objs.as_ref().unwrap();
        let gl_fixs = &self.gl_fixs;

        let mut u = Uniforms2D {
            m: mx_gui * self.data.trans_paper.ortho * self.data.trans_paper.mx,
            tex: 0,
            frac_dash: 0.5,
            line_color: Rgba::new(0.0, 0.0, 0.0, 0.0),
            texturize: 0,
        };

        let width = size.x as i32;
        let height = size.y as i32;

        unsafe {
             let mut draw_fb_binder = if let Some(fbo) = &gl_fixs.fbo_paper {
                let binder = BinderDrawFramebuffer::new();
                binder.rebind(fbo);
                Some(binder)
            } else {
                None
            };

            gl::ClearColor(0.7, 0.7, 0.7, 1.0);
            gl::ClearStencil(1);
            gl::StencilMask(0xff);
            gl::StencilFunc(gl::ALWAYS, 0, 0);
            gl::Disable(gl::STENCIL_TEST);

            gl::Clear(gl::COLOR_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);

            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            gl::BindVertexArray(gl_fixs.vao_paper.id());
            if let (Some(tex), true) = (&gl_objs.textures, self.data.show_textures) {
                gl::ActiveTexture(gl::TEXTURE0);
                gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex.id());
                u.texturize = 1;
            }

            // The paper
            gl::Enable(gl::STENCIL_TEST);
            gl::StencilOp(gl::KEEP, gl::KEEP, gl::ZERO);

            gl_fixs.prg_paper_solid.draw(&u, &gl_objs.paper_vertices_page, gl::TRIANGLES);

            gl::Disable(gl::STENCIL_TEST);

            u.line_color = Rgba::new(0.5, 0.5, 0.5, 1.0);

            gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_margin, gl::LINES);

            u.line_color = Rgba::new(0.0, 0.0, 0.0, 1.0);

            // Line Tabs
            if self.data.show_tabs {
                gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_tab_edge, gl::LINES);
            }

            gl::Enable(gl::STENCIL_TEST);
            gl::StencilOp(gl::KEEP, gl::KEEP, gl::INCR);


            // Solid Tabs
            if self.data.show_tabs {
                gl_fixs.prg_paper_solid.draw(&u, &gl_objs.paper_vertices_tab, gl::TRIANGLES);
            }
            gl::Disable(gl::STENCIL_TEST);

            // Borders
            gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_edge_border, gl::LINES);

            gl::Enable(gl::STENCIL_TEST);

            // Textured faces
            gl_fixs.prg_paper_solid.draw(&u, (&gl_objs.paper_vertices, &gl_objs.paper_vertices_sel) , gl::TRIANGLES);

            gl::Disable(gl::STENCIL_TEST);

            // Creases
            gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_edge_crease, gl::LINES);

            // Selected edge
            if self.data.selected_edge.is_some() {
                u.line_color = if self.data.mode == MouseMode::Edge { Rgba::new(0.5, 0.5, 1.0, 1.0) } else { Rgba::new(0.0, 0.5, 0.0, 1.0) };
                gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_edge_sel, gl::LINES);
            }

            // Draw the highlight overlap if "1 < STENCIL"
            gl::Enable(gl::STENCIL_TEST);
            gl::StencilOp(gl::KEEP, gl::KEEP, gl::KEEP);


            if self.data.highlight_overlaps {
                // Draw the overlapped highlight if "1 < STENCIL"
                let uq = UniformQuad { color: Rgba::new(1.0, 0.0, 1.0, 0.9) };
                gl::StencilFunc(gl::LESS, 1, 0xff);
                gl_fixs.prg_quad.draw(&uq, glr::NilVertexAttrib(3), gl::TRIANGLES);

                // Draw the highlight dim if "1 >= STENCIL"
                let uq = UniformQuad { color: Rgba::new(1.0, 1.0, 1.0, 0.9) };
                gl::StencilFunc(gl::GEQUAL, 1, 0xff);
                gl_fixs.prg_quad.draw(&uq, glr::NilVertexAttrib(3), gl::TRIANGLES);
            } else {
                // If highlight is disabled wraw the overlaps anyway, but dimmer, or else it would be invisible
                let uq = UniformQuad { color: Rgba::new(1.0, 0.0, 1.0, 0.5) };
                gl::StencilFunc(gl::LESS, 1, 0xff);
                gl_fixs.prg_quad.draw(&uq, glr::NilVertexAttrib(3), gl::TRIANGLES);
            }

            gl::Disable(gl::STENCIL_TEST);

            // If there is an FBO, blit to the real FB
            if draw_fb_binder.take().is_some() { // check and drop
                let fbo = gl_fixs.fbo_paper.as_ref().unwrap();
                let _read_fb_binder = BinderReadFramebuffer::bind(fbo);
                gl::BlitFramebuffer(0, 0, width, height, 0, 0, width, height, gl::COLOR_BUFFER_BIT, gl::NEAREST);
            }
        }
    }

    fn add_rebuild(&mut self, flags: RebuildFlags) {
        self.data.rebuild |= flags;
        if flags.intersects(RebuildFlags::ANY_REDRAW_PAPER) {
            //self.wpaper.queue_render();
        }
        if flags.intersects(RebuildFlags::ANY_REDRAW_SCENE) {
            //self.wscene.queue_render();
        }
    }
    fn push_undo_action(&mut self, action: Vec<UndoAction>) {
        if !action.is_empty() {
            self.data.undo_stack.push(action);
        }
        if !self.data.modified {
            self.data.modified = true;
        }
    }
    fn title(&self) -> String {
        let unsaved = if self.data.modified { "*" } else { "" };
        let app_name = "Papercraft";
        match &self.data.file_name {
            Some(f) =>
                format!("{unsaved}{} - {app_name}", f.display()),
            None =>
                format!("{unsaved} - {app_name}"),
        }
    }
}

struct BackupGlConfig {
    p_vao: i32,
    p_prg: i32,
    p_buf: i32,
    p_atex: i32,
    p_tex: i32,
    p_tex_min: i32,
    p_tex_mag: i32,
}

impl BackupGlConfig {
    fn backup() -> BackupGlConfig {
        unsafe {
            let mut p_vao = 0;
            gl::GetIntegerv(gl::VERTEX_ARRAY_BINDING, &mut p_vao);
            let mut p_prg = 0;
            gl::GetIntegerv(gl::CURRENT_PROGRAM, &mut p_prg);
            let mut p_buf = 0;
            gl::GetIntegerv(gl::ARRAY_BUFFER_BINDING, &mut p_buf);
            let mut p_atex = 0;
            gl::GetIntegerv(gl::ACTIVE_TEXTURE, &mut p_atex);
            let mut p_tex = 0;
            gl::GetIntegerv(gl::TEXTURE_BINDING_2D, &mut p_tex);
            let mut p_tex_min = 0;
            gl::GetTexParameteriv(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, &mut p_tex_min);
            let mut p_tex_mag = 0;
            gl::GetTexParameteriv(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, &mut p_tex_mag);
                BackupGlConfig {
                p_vao, p_prg, p_buf,
                p_atex, p_tex,
                p_tex_min, p_tex_mag,
            }
        }
    }
}

impl Drop for BackupGlConfig {
    fn drop(&mut self) {
        unsafe {
            gl::BindVertexArray(self.p_vao as _);
            gl::UseProgram(self.p_prg as _);
            gl::BindBuffer(gl::ARRAY_BUFFER, self.p_buf as _);
            gl::ActiveTexture(self.p_atex as _);
            gl::BindTexture(gl::TEXTURE_2D, self.p_tex as _);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, self.p_tex_min as _);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, self.p_tex_mag as _);

            gl::Disable(gl::DEPTH_TEST);
            gl::Disable(gl::STENCIL_TEST);
            gl::Disable(gl::POLYGON_OFFSET_FILL);
        }
    }
}

#[derive(Debug, Default)]
struct Canvas3dStatus {
    action: Canvas3dAction,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum Canvas3dAction {
    #[default]
    None,
    Hovering,
    Clicked(imgui::MouseButton),
    Dragging(imgui::MouseButton),
}

fn canvas3d(ui: &imgui::Ui, st: &mut Canvas3dStatus) {
    ui.invisible_button(
        "canvas3d",
        ui.content_region_avail(),
    );
    let hovered = ui.is_item_hovered();

    let action = match &st.action {
        Canvas3dAction::Dragging(bt) => {
            if ui.is_mouse_dragging(*bt) {
                Canvas3dAction::Dragging(*bt)
            } else if hovered {
                Canvas3dAction::Hovering
            } else {
                Canvas3dAction::None
            }
        }
        Canvas3dAction::Hovering | Canvas3dAction::Clicked(_) => {
            if !hovered {
                Canvas3dAction::None
            } else if ui.is_mouse_dragging(imgui::MouseButton::Left) {
                Canvas3dAction::Dragging(imgui::MouseButton::Left)
            } else if ui.is_mouse_dragging(imgui::MouseButton::Right) {
                Canvas3dAction::Dragging(imgui::MouseButton::Right)
            } else if ui.is_mouse_clicked(imgui::MouseButton::Left) {
                Canvas3dAction::Clicked(imgui::MouseButton::Left)
            } else if ui.is_mouse_clicked(imgui::MouseButton::Right) {
                Canvas3dAction::Clicked(imgui::MouseButton::Right)
            } else {
                Canvas3dAction::Hovering
            }
        }
        Canvas3dAction::None => {
            // If the mouse is entered while dragging, it does not count, as if captured by other
            if hovered &&
                !ui.is_mouse_dragging(imgui::MouseButton::Left) &&
                !ui.is_mouse_dragging(imgui::MouseButton::Right)
            {
                Canvas3dAction::Hovering
            } else {
                Canvas3dAction::None
            }
        }
    };

    *st = Canvas3dStatus {
        action,
    };
}
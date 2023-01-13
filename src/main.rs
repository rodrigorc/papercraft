#![allow(clippy::collapsible_if)]

use std::{num::NonZeroU32, ffi::CString, time::{Instant, Duration}, rc::{Rc, Weak}, cell::RefCell, path::{Path, PathBuf}};
use std::io::Read;
use anyhow::{Result, anyhow, Context};
use cgmath::{
    prelude::*,
    Deg,
};
use glow::HasContext;
use glutin::{prelude::*, config::{ConfigTemplateBuilder, Config}, display::GetGlDisplay, context::{ContextAttributesBuilder, ContextApi}, surface::{SurfaceAttributesBuilder, WindowSurface, Surface}};
use glutin_winit::DisplayBuilder;
use image::DynamicImage;
use imgui_winit_support::WinitPlatform;
use raw_window_handle::{HasRawWindowHandle};
use winit::{event, event_loop::{EventLoopBuilder}, window::{WindowBuilder, Window}, event::VirtualKeyCode};
use imgui_glow_renderer::TextureMap;


mod imgui_filedialog;
mod waveobj;
mod paper;
mod glr;
mod util_3d;
mod util_gl;
mod ui;

use ui::*;

use paper::{Papercraft, TabStyle, FoldStyle, PaperOptions};
use glr::Rgba;
use util_3d::{Matrix3, Vector2, Vector3};
use util_gl::{Uniforms2D, Uniforms3D, UniformQuad};

use glr::{BinderRenderbuffer, BinderDrawFramebuffer, BinderReadFramebuffer};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// Long
struct Cli {
    #[arg(value_name = "CRAFT_FILE")]
    name: Option<PathBuf>,

    #[arg(short, long, value_name = "OBJ_FILE")]
    import: Option<PathBuf>,

    #[arg(long, help = "Uses Dear ImGui light theme instead of the default dark one")]
    light: bool,
}

fn main() {
    let cli = Cli::parse();

    let event_loop = EventLoopBuilder::new().build();

    // We render to FBOs so we do not need depth, stencil buffers or anything fancy.
    let window_builder = WindowBuilder::new();
    let template = ConfigTemplateBuilder::new()
        .prefer_hardware_accelerated(Some(true))
        .with_depth_size(0)
        .with_stencil_size(0)
    ;

    let display_builder = DisplayBuilder::new()
        .with_window_builder(Some(window_builder));

    let (window, gl_config) = display_builder
        .build(&event_loop, template, |configs| {
            configs
                .reduce(|cfg1, cfg2| {
                    let t = |c: &Config| (c.num_samples(), c.depth_size(), c.stencil_size());
                    if t(&cfg2) < t(&cfg1) {
                        cfg2
                    } else {
                        cfg1
                    }
                })
                .unwrap()
        })
        .unwrap();
    //dbg!(gl_config.num_samples(), gl_config.depth_size(), gl_config.stencil_size());
    let window = window.unwrap();
    window.set_title("Papercraft");
    window.set_ime_allowed(true);
    let raw_window_handle = Some(window.raw_window_handle());
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
    let gl_window = GlWindow::new(window, &gl_config);
    let gl_context = not_current_gl_context
        .take()
        .unwrap()
        .make_current(&gl_window.surface)
        .unwrap();
    // Enable v-sync to avoid consuming too much CPU
    let _ = gl_window.surface.set_swap_interval(&gl_context, glutin::surface::SwapInterval::Wait(NonZeroU32::new(1).unwrap()));

    let mut imgui_context = imgui::Context::create();
    imgui_context.set_ini_filename(None);

    let mut winit_platform = WinitPlatform::init(&mut imgui_context);
    winit_platform.attach_window(
        imgui_context.io_mut(),
        &gl_window.window,
        imgui_winit_support::HiDpiMode::Default,
    );
    //imgui_context.set_clipboard_backend(MyClip);

    let mut ttf = Vec::new();
    flate2::read::ZlibDecoder::new(include_bytes!("Karla-Regular.ttf.z").as_slice()).read_to_end(&mut ttf).unwrap();

    let hidpi_factor = winit_platform.hidpi_factor() as f32;
    let fonts = imgui_context.fonts();
    let _font_default = fonts.add_font(&[
            imgui::FontSource::TtfData {
                data: &ttf,
                size_pixels: (18.0 * hidpi_factor).floor(),
                config: None
            },
        ]);
    let font_big = fonts.add_font(&[
            imgui::FontSource::TtfData {
                data: &ttf,
                size_pixels: (28.0 * hidpi_factor).floor(),
                config: None
            },
        ]);
    let font_small = fonts.add_font(&[
            imgui::FontSource::TtfData {
                data: &ttf,
                size_pixels: (12.0 * hidpi_factor).floor(),
                config: None
            },
            imgui::FontSource::DefaultFontData { config: None }, // For the © in the about window :/
        ]);
    imgui_context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
    imgui_context.io_mut().font_allow_user_scaling = true;

    let style = imgui_context.style_mut();
    //style.scale_all_sizes(hidpi_factor);
    if cli.light {
        style.use_light_colors();
    }

    let gl = unsafe {
        let dsp = gl_context.display();
        gl::load_with(|s| dsp.get_proc_address(&CString::new(s).unwrap()));
        glow::Context::from_loader_function(|s| dsp.get_proc_address(&CString::new(s).unwrap()).cast())
    };
    let mut ig_renderer = imgui_glow_renderer::AutoRenderer::initialize(gl, &mut imgui_context)
        .expect("failed to create renderer");

    // Initialize papercraft status
    let sz_dummy = Vector2::new(1.0, 1.0);
    let papercraft = Papercraft::empty();
    let data = PapercraftContext::from_papercraft(
        papercraft,
        None,
        sz_dummy,
        sz_dummy
    );

    let mut cmd_file_action = match cli {
        Cli { name: Some(name), .. }  => {
            Some((FileAction::OpenCraft, name))
        }
        Cli { import: Some(import), .. }  => {
            Some((FileAction::ImportObj, import))
        }
        _ => { None }
    };

    let icons_tex = load_texture_from_memory(include_bytes!("icons.png"), true).unwrap();
    let icons_tex = ig_renderer.texture_map_mut().register(icons_tex).unwrap();

    let gl_fixs = build_gl_fixs().unwrap();
    let ctx = Rc::new_cyclic(|this| {
        RefCell::new(GlobalContext {
            this: this.clone(),
            gl_fixs,
            _font_default, font_big, font_small,
            icons_tex,
            data,
            splitter_pos: 1.0,
            sz_full: Vector2::new(2.0, 1.0),
            sz_scene: Vector2::new(1.0, 1.0),
            sz_paper: Vector2::new(1.0, 1.0),
            scene_ui_status: Canvas3dStatus::default(),
            paper_ui_status: Canvas3dStatus::default(),
            options_opened: None,
            about_visible: false,
            option_button_height: 0.0,
            file_dialog: None,
            file_action: None,
            error_message: None,
            confirmable_action: None,
            popup_time_start: Instant::now(),
            render_scene_pending: true,
            render_paper_pending: true,
        })
    });
    imgui_context.io_mut().config_flags |= imgui::ConfigFlags::NAV_ENABLE_KEYBOARD;

    //In Linux convert fatal signals to panics to save the crash backup
    #[cfg(target_os="linux")]
    {
        use signal_hook::consts::signal::*;
        use signal_hook::iterator::SignalsInfo;
        let event_loop = event_loop.create_proxy();
        std::thread::spawn(move || {
            let sigs = vec![SIGHUP, SIGINT, SIGTERM];
            let mut signals: SignalsInfo = SignalsInfo::new(&sigs).unwrap();
            for _ in &mut signals {
                let _ = event_loop.send_event(());
            }
        });
    }

    // Main loop, if it panics or somewhat crashes, try to save a backup
    let ctx0 = Rc::clone(&ctx);
    let maybe_fatal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let mut old_title = String::new();
        let mut last_frame = Instant::now();
        // The main loop will keep on rendering for 500 ms or 10 frames after the last input,
        // whatever is longer. The frames are needed in case a file operation takes a lot of time,
        // we want at least a render just after that.
        let mut last_input_time = Instant::now();
        let mut last_input_frame: u32 = 0;
        let mut quit_requested = BoolWithConfirm::None;

        event_loop.run(move |event, _, control_flow| {
            match event {
                event::Event::NewEvents(_) => {
                    let now = Instant::now();
                    imgui_context
                        .io_mut()
                        .update_delta_time(now.duration_since(last_frame));
                    last_frame = now;
                }
                event::Event::MainEventsCleared => {
                    winit_platform
                        .prepare_frame(imgui_context.io_mut(), &gl_window.window)
                        .unwrap();
                    gl_window.window.request_redraw();
                }
                event::Event::RedrawEventsCleared => {
                    let now = Instant::now();
                    // If the mouse is down, redraw all the time, maybe the user is dragging.
                    let mouse = unsafe { imgui_sys::igIsAnyMouseDown() };
                    last_input_frame += 1;
                    if mouse || now.duration_since(last_input_time) < Duration::from_millis(1000) || last_input_frame < 60 {
                        *control_flow = winit::event_loop::ControlFlow::Poll;
                    } else {
                        *control_flow = winit::event_loop::ControlFlow::Wait;
                    }
                }
                event::Event::RedrawRequested(_) => {
                    // The renderer assumes you'll be clearing the buffer yourself
                    let gl = ig_renderer.gl_context();
                    unsafe {
                        gl.clear_color(0.0, 0.0, 0.0, 1.0);
                        gl.clear(glow::COLOR_BUFFER_BIT);
                    };

                    let ui = imgui_context.frame();
                    //ui.show_demo_window(&mut true);

                    {
                        let _s1 = ui.push_style_var(imgui::StyleVar::WindowPadding([0.0, 0.0]));
                        let _s2 = ui.push_style_var(imgui::StyleVar::WindowRounding(0.0));

                        let _w = ui.window("Papercraft")
                            .position([0.0, 0.0], imgui::Condition::Always)
                            .size(ui.io().display_size, imgui::Condition::Always)
                            .flags(
                                imgui::WindowFlags::NO_DECORATION |
                                imgui::WindowFlags::NO_RESIZE |
                                imgui::WindowFlags::MENU_BAR |
                                imgui::WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS |
                                imgui::WindowFlags::NO_NAV
                                )
                            .begin();

                        drop((_s2, _s1));
                        let mut ctx = ctx.borrow_mut();

                        if let Some(cmd_file_action) = cmd_file_action.take() {
                            ctx.popup_time_start = Instant::now();
                            ctx.file_action = Some(cmd_file_action);
                            ui.open_popup("###Wait");
                        }

                        let menu_actions = ctx.build_ui(ui);
                        ctx.run_menu_actions(ui, &menu_actions);
                        ctx.run_mouse_actions(ui);

                        if ctx.data.rebuild.intersects(RebuildFlags::ANY_REDRAW_SCENE) {
                            ctx.render_scene_pending = true;
                        }
                        if ctx.data.rebuild.intersects(RebuildFlags::ANY_REDRAW_PAPER) {
                            ctx.render_paper_pending = true;
                        }
                        ctx.data.pre_render();
                        let new_title = ctx.title(true);
                        if new_title != old_title {
                            gl_window.window.set_title(&new_title);
                            old_title = new_title;
                        }

                        match (quit_requested, menu_actions.quit) {
                            (_, BoolWithConfirm::Confirmed) | (BoolWithConfirm::Confirmed, _) => {
                                *control_flow = winit::event_loop::ControlFlow::Exit;
                            }
                            (BoolWithConfirm::Requested, _) | (_, BoolWithConfirm::Requested) => {
                                quit_requested = BoolWithConfirm::None;
                                ctx.open_confirmation_dialog(ui,
                                    "Quit?",
                                    "The model has not been save, continue anyway?",
                                    |a| a.quit = BoolWithConfirm::Confirmed
                                );
                            }
                            (BoolWithConfirm::None, BoolWithConfirm::None) => {}
                        }
                    }

                    winit_platform.prepare_render(ui, &gl_window.window);
                    let draw_data = imgui_context.render();

                    // This is the only extra render step to add
                    ig_renderer
                        .render(draw_data)
                        .expect("error rendering imgui");
                    gl_window.surface.swap_buffers(&gl_context).unwrap();
                    {
                        let mut ctx = ctx.borrow_mut();
                        ctx.render_scene_pending = false;
                        ctx.render_paper_pending = false;
                    }
                }
                event::Event::UserEvent(()) | //Signal
                event::Event::WindowEvent {
                    event: event::WindowEvent::CloseRequested,
                    ..
                } => {
                    last_input_time = Instant::now();
                    last_input_frame = 0;
                    quit_requested = ctx.borrow().check_modified();
                }
                event::Event::DeviceEvent { .. } => {
                    // Ignore deviceevents, they are not used and they wake up the loop needlessly
                }
                event => {
                    last_input_time = Instant::now();
                    last_input_frame = 0;
                    winit_platform.handle_event(imgui_context.io_mut(), &gl_window.window, &event);
                }
            }
        });
    }));
    if let Err(e) = maybe_fatal {
        let ctx = ctx0.borrow();
        if ctx.data.modified {
            let mut dir = std::env::temp_dir();
            dir.push(format!("crashed-{}.craft", std::process::id()));
            eprintln!("Papercraft panicked! Saving backup at \"{}\"", dir.display());
            if let Err(e) = ctx.save_as_craft(dir) {
                eprintln!("backup failed with {e:?}");
            }
        }
        std::panic::resume_unwind(e);
    }
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

    let vao = glr::VertexArray::generate();

    let fbo_scene = glr::Framebuffer::generate();
    let rbo_scene_color = glr::Renderbuffer::generate();
    let rbo_scene_depth = glr::Renderbuffer::generate();

    unsafe {
        let fb_binder = BinderDrawFramebuffer::bind(&fbo_scene);

        let rb_binder = BinderRenderbuffer::bind(&rbo_scene_color);
        gl::RenderbufferStorage(rb_binder.target(), gl::RGBA8, 1, 1);
        gl::FramebufferRenderbuffer(fb_binder.target(), gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, rbo_scene_color.id());

        rb_binder.rebind(&rbo_scene_depth);
        gl::RenderbufferStorage(rb_binder.target(), gl::DEPTH_COMPONENT, 1, 1);
        gl::FramebufferRenderbuffer(fb_binder.target(), gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, rbo_scene_depth.id());
    }

    let fbo_paper = glr::Framebuffer::generate();
    let rbo_paper_color = glr::Renderbuffer::generate();
    let rbo_paper_stencil = glr::Renderbuffer::generate();

    unsafe {
        let fb_binder = BinderDrawFramebuffer::bind(&fbo_paper);

        let rb_binder = BinderRenderbuffer::bind(&rbo_paper_color);
        gl::RenderbufferStorage(rb_binder.target(), gl::RGBA8, 1,1);
        gl::FramebufferRenderbuffer(fb_binder.target(), gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, rbo_paper_color.id());

        rb_binder.rebind(&rbo_paper_stencil);
        gl::RenderbufferStorage(rb_binder.target(), gl::STENCIL_INDEX, 1, 1);
        gl::FramebufferRenderbuffer(fb_binder.target(), gl::STENCIL_ATTACHMENT, gl::RENDERBUFFER, rbo_paper_stencil.id());
    }


    Ok(GLFixedObjects {
        vao,

        fbo_scene,
        rbo_scene_color,
        rbo_scene_depth,

        fbo_paper,
        rbo_paper_color,
        rbo_paper_stencil,

        prg_scene_solid,
        prg_scene_line,
        prg_paper_solid,
        prg_paper_line,
        prg_quad,
    })
}

struct GLFixedObjects {
    vao: glr::VertexArray,

    fbo_scene: glr::Framebuffer,
    rbo_scene_color: glr::Renderbuffer,
    rbo_scene_depth: glr::Renderbuffer,

    fbo_paper: glr::Framebuffer,
    rbo_paper_color: glr::Renderbuffer,
    rbo_paper_stencil: glr::Renderbuffer,

    prg_scene_solid: glr::Program,
    prg_scene_line: glr::Program,
    prg_paper_solid: glr::Program,
    prg_paper_line: glr::Program,
    prg_quad: glr::Program,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FileAction {
    OpenCraft,
    SaveAsCraft,
    ImportObj,
    UpdateObj,
    ExportObj,
    GeneratePdf,
}

impl FileAction {
    fn title(&self) -> &'static str {
        match self {
            FileAction::OpenCraft => "Opening...",
            FileAction::SaveAsCraft => "Saving...",
            FileAction::ImportObj => "Importing...",
            FileAction::UpdateObj => "Updating...",
            FileAction::ExportObj => "Exporting...",
            FileAction::GeneratePdf => "Generating PDF...",
        }
    }
}

struct GlobalContext {
    this: Weak<RefCell<GlobalContext>>,
    gl_fixs: GLFixedObjects,
    _font_default: imgui::FontId,
    font_big: imgui::FontId,
    font_small: imgui::FontId,
    icons_tex: imgui::TextureId,
    data: PapercraftContext,
    splitter_pos: f32,
    sz_full: Vector2,
    sz_scene: Vector2,
    sz_paper: Vector2,
    scene_ui_status: Canvas3dStatus,
    paper_ui_status: Canvas3dStatus,
    options_opened: Option<PaperOptions>,
    option_button_height: f32,
    about_visible: bool,
    file_dialog: Option<(imgui_filedialog::FileDialog, &'static str, FileAction)>,
    file_action: Option<(FileAction, PathBuf)>,
    error_message: Option<String>,
    confirmable_action: Option<(String, String, Box<dyn Fn(&mut MenuActions)>)>,
    popup_time_start: Instant,
    render_scene_pending: bool,
    render_paper_pending: bool,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
enum BoolWithConfirm {
    #[default]
    None,
    Requested,
    Confirmed,
}

#[derive(Debug, Default)]
struct MenuActions {
    open: BoolWithConfirm,
    save: bool,
    save_as: bool,
    import_obj: BoolWithConfirm,
    update_obj: BoolWithConfirm,
    export_obj: bool,
    generate_pdf: bool,
    quit: BoolWithConfirm,
    reset_views: bool,
    undo: bool,
}

impl GlobalContext {
    fn build_modal_error_message(&mut self, ui: &imgui::Ui) {
        if let Some(_pop) = ui.modal_popup_config("Error")
            .resizable(false)
            .always_auto_resize(true)
            .opened(&mut true)
            .begin_popup()
        {
            ui.text(&self.error_message.as_deref().unwrap_or_default());

            ui.separator();

            if ui.button_with_size("OK", [ui.current_font_size() * 5.5, 0.0])
                || ui.is_key_pressed(imgui::Key::Enter)
                || ui.is_key_pressed(imgui::Key::KeyPadEnter)
            {
                if !ui.is_window_appearing() {
                    ui.close_current_popup();
                    self.error_message = None;
                }
            }
        }
    }
    fn build_confirm_message(&mut self, ui: &imgui::Ui, menu_actions: &mut MenuActions) {
        let mut closed = None;
        if let Some(action) = self.confirmable_action.take() {
            let (title, message, _) = &action;
            if let Some(_pop) = ui.modal_popup_config(&format!("{title}###Confirm"))
                .resizable(false)
                .always_auto_resize(true)
                .opened(&mut true)
                .begin_popup()
            {
                ui.text(&message);

                ui.separator();

                if ui.button_with_size("Cancel", [ui.current_font_size() * 5.5, 0.0]) {
                    if !ui.is_window_appearing() {
                        ui.close_current_popup();
                        closed = Some(false);
                    }
                }
                ui.same_line();
                if ui.button_with_size("Continue", [ui.current_font_size() * 5.5, 0.0]) {
                    if !ui.is_window_appearing() {
                        ui.close_current_popup();
                        closed = Some(true);
                    }
                }
            }
            if let Some(cont) = closed {
                if cont {
                    (action.2)(menu_actions);
                }
            } else {
                self.confirmable_action = Some(action);
            }
        }
    }
    fn build_about(&mut self, ui: &imgui::Ui) {
        if !self.about_visible {
            return;
        }
        if let Some(_options) = ui.window("About...###about")
            .movable(true)
            .resizable(false)
            .always_auto_resize(true)
            .opened(&mut self.about_visible)
            .begin()
        {
            let sz_full = Vector2::from(ui.content_region_avail());
            let _s = ui.push_font(self.font_big);
            center_text(ui, "Papercraft", sz_full.x);
            drop(_s);
            advance_cursor(ui, 0.0, 1.0);
            center_text(ui, &format!("Version {}", env!("CARGO_PKG_VERSION")), sz_full.x);
            advance_cursor(ui, 0.0, 1.0);
            center_text(ui, env!("CARGO_PKG_DESCRIPTION"), sz_full.x);
            advance_cursor(ui, 0.0, 0.5);
            center_url(ui, env!("CARGO_PKG_REPOSITORY"), "url", None, sz_full.x);
            advance_cursor(ui, 0.0, 0.5);
            let _s = ui.push_font(self.font_small);
            center_text(ui, "© Copyright 2022 - Rodrigo Rivas Costa", sz_full.x);
            center_text(ui, "This program comes with absolutely no warranty.", sz_full.x);
            center_url(
                ui, 
                "See the GNU General Public License, version 3 or later for details.", "gpl3",
                Some("https://www.gnu.org/licenses/gpl-3.0.html"),
                sz_full.x
            );
            drop(_s);

            //TODO: list third party SW
        }
    }
    // Returns true if the action has just been done successfully
    fn build_modal_wait_message_and_run_file_action(&mut self, ui: &imgui::Ui) -> bool {
        let mut ok = false;
        if let Some(file_action) = self.file_action.take() {
            let (action, file) = &file_action;
            let title = action.title();
            let mut res = None;
            // Build the modal itself
            unsafe {
                imgui_sys::igSetNextWindowSize([150.0, 0.0].into(), imgui::Condition::Once as _);
            }
            if let Some(_pop) = ui.modal_popup_config(&format!("{title}###Wait"))
                .resizable(false)
                .begin_popup()
            {
                ui.text("Please, wait...");

                // Give time to the fading modal, should be enough
                let run = self.popup_time_start.elapsed() > Duration::from_millis(250);
                if run {
                    res = Some(self.run_file_action(*action, file));
                    ui.close_current_popup();
                }
            }
            match res {
                None => {
                    // keep the action pending, for now.
                    self.file_action = Some(file_action);
                }
                Some(Ok(())) => {
                    ok = true;
                }
                Some(Err(e)) => {
                    self.error_message = Some(format!("{e:?}"));
                    ui.open_popup("Error");
                }
            }
        }
        ok
    }

    fn build_ui(&mut self, ui: &imgui::Ui) -> MenuActions {
        let mut menu_actions = self.build_menu_and_file_dialog(ui);

        let pad: f32 = ui.current_font_size() / 4.0;

        let _s = (
            ui.push_style_var(imgui::StyleVar::WindowPadding([pad, pad])),
            ui.push_style_var(imgui::StyleVar::ItemSpacing([0.0, 0.0])),
        );
        if let Some(_toolbar) = ui.child_window("toolbar")
            .size([0.0, ui.current_font_size() * 3.0 + 2.0 * pad])
            .always_use_window_padding(true)
            .border(false)
            .begin()
        {
            let _s3 = ui.push_style_var(imgui::StyleVar::ItemSpacing([ui.current_font_size() / 8.0, 0.0]));
            //The texture image is 128x128 pixels, but each image is 48x48
            let n = 48.0 / 128.0;
            let btn_sz = ui.current_font_size() * 3.0;
            let color_active = ui.style_color(imgui::StyleColor::ButtonActive);
            let color_white = [1.0, 1.0, 1.0, 1.0].into();
            let color_trans = [0.0, 0.0, 0.0, 0.0];

            if unsafe {
                let _t1 = ui.push_id("Face");
                imgui_sys::igImageButton(
                    self.icons_tex.id() as _,
                    [btn_sz, btn_sz].into(),
                    [0.0, 0.0].into(),
                    [n, n].into(),
                    0,
                    (if self.data.mode == MouseMode::Face { color_active } else { color_trans }).into(),
                    color_white,
                )
            } {
                self.set_mouse_mode(MouseMode::Face);
            }
            ui.same_line();
            if unsafe {
                let _t1 = ui.push_id("Edge");
                imgui_sys::igImageButton(
                    self.icons_tex.id() as _,
                    [btn_sz, btn_sz].into(),
                    [n, 0.0].into(),
                    [2.0*n, n].into(),
                    0,
                    (if self.data.mode == MouseMode::Edge { color_active } else { color_trans }).into(),
                    color_white,
                )
            } {
                self.set_mouse_mode(MouseMode::Edge);
            }
            ui.same_line();
            if unsafe {
                let _t1 = ui.push_id("Tab");
                imgui_sys::igImageButton(
                    self.icons_tex.id() as _,
                    [btn_sz, btn_sz].into(),
                    [0.0, n].into(),
                    [n, 2.0*n].into(),
                    0,
                    (if self.data.mode == MouseMode::Tab { color_active } else { color_trans }).into(),
                    color_white,
                )
            } {
                self.set_mouse_mode(MouseMode::Tab);
            }
        }
        drop(_s);

        let _s = (
            ui.push_style_var(imgui::StyleVar::ItemSpacing([2.0, 2.0])),
            ui.push_style_var(imgui::StyleVar::WindowPadding([0.0, 0.0])),
            ui.push_style_color(imgui::StyleColor::ButtonActive, ui.style_color(imgui::StyleColor::ButtonHovered)),
            ui.push_style_color(imgui::StyleColor::Button, ui.style_color(imgui::StyleColor::ButtonHovered)),
        );

        if let Some(_main_area) = ui.child_window("main_area")
            .size([0.0, -ui.frame_height()])
            .begin() {
            let sz_full = Vector2::from(ui.content_region_avail());

            if self.sz_full != sz_full {
                if self.sz_full.x > 1.0 {
                    self.splitter_pos = self.splitter_pos * sz_full.x / self.sz_full.x;
                }
                self.sz_full = sz_full;
            }

            let scale = Vector2::from(ui.io().display_framebuffer_scale);

            self.build_scene(ui, self.splitter_pos);
            let sz_scene = scale_size(scale, Vector2::from(ui.item_rect_size()));

            ui.same_line();

            ui.button_with_size("##vsplitter", [ui.current_font_size() / 2.0, -1.0]);
            if ui.is_item_active() {
                self.splitter_pos += ui.io().mouse_delta[0];
            }
            self.splitter_pos = self.splitter_pos.clamp(50.0, (sz_full.x - 50.0).max(50.0));
            if ui.is_item_hovered() || ui.is_item_active() {
                ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeEW));
            }

            ui.same_line();

            self.build_paper(ui);

            let sz_paper = scale_size(scale, Vector2::from(ui.item_rect_size()));

            // Resize FBOs
            if sz_scene != self.sz_scene && sz_scene.x > 1.0 && sz_scene.y > 1.0 {
                self.add_rebuild(RebuildFlags::SCENE_REDRAW);
                self.sz_scene = sz_scene;

                self.data.trans_scene.persp = cgmath::perspective(Deg(60.0), sz_scene.x / sz_scene.y, 1.0, 100.0);
                self.data.trans_scene.persp_inv = self.data.trans_scene.persp.invert().unwrap();

                let (x, y) = (sz_scene.x as i32, sz_scene.y as i32);

                unsafe {
                    let rb_binder = BinderRenderbuffer::bind(&self.gl_fixs.rbo_scene_color);

                    'no_aa: {
                        for samples in glr::available_multisamples(rb_binder.target(), gl::RGBA8) {
                            gl::GetError(); //clear error
                            rb_binder.rebind(&self.gl_fixs.rbo_scene_color);
                            gl::RenderbufferStorageMultisample(rb_binder.target(), samples, gl::RGBA8, x, y);
                            rb_binder.rebind(&self.gl_fixs.rbo_scene_depth);
                            gl::RenderbufferStorageMultisample(rb_binder.target(), samples, gl::DEPTH_COMPONENT, x, y);

                            if gl::GetError() != 0 || gl::CheckFramebufferStatus(gl::DRAW_FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
                                continue;
                            }
                            break 'no_aa;
                        }

                        rb_binder.rebind(&self.gl_fixs.rbo_scene_color);
                        gl::RenderbufferStorage(rb_binder.target(), gl::RGBA8, x, y);
                        rb_binder.rebind(&self.gl_fixs.rbo_scene_depth);
                        gl::RenderbufferStorage(rb_binder.target(), gl::DEPTH_COMPONENT, x, y);
                    }
                }
            }

            if sz_paper != self.sz_paper && sz_paper.x > 1.0 && sz_paper.y > 1.0 {
                self.add_rebuild(RebuildFlags::PAPER_REDRAW);
                self.sz_paper = sz_paper;

                let (x, y) = (sz_paper.x as i32, sz_paper.y as i32);
                self.data.trans_paper.ortho = util_3d::ortho2d(sz_paper.x, sz_paper.y);

                unsafe {
                    let rb_binder = BinderRenderbuffer::bind(&self.gl_fixs.rbo_paper_color);

                    'no_aa: {
                        for samples in glr::available_multisamples(rb_binder.target(), gl::RGBA8) {
                            gl::GetError(); //clear error
                            rb_binder.rebind(&self.gl_fixs.rbo_paper_color);
                            gl::RenderbufferStorageMultisample(rb_binder.target(), samples, gl::RGBA8, x, y);
                            rb_binder.rebind(&self.gl_fixs.rbo_paper_stencil);
                            gl::RenderbufferStorageMultisample(rb_binder.target(), samples, gl::STENCIL_INDEX, x, y);

                            if gl::GetError() != 0 || gl::CheckFramebufferStatus(gl::DRAW_FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
                                continue;
                            }
                            break 'no_aa;
                        }

                        rb_binder.rebind(&self.gl_fixs.rbo_paper_color);
                        gl::RenderbufferStorage(rb_binder.target(), gl::RGBA8, x, y);
                        rb_binder.rebind(&self.gl_fixs.rbo_paper_stencil);
                        gl::RenderbufferStorage(rb_binder.target(), gl::STENCIL_INDEX, x, y);
                    }
                }
            }

        }
        drop(_s);

        advance_cursor(ui, 0.25, 0.0);

        let status_text = match self.data.mode {
            MouseMode::Face => "Face mode. Click to select a piece. Drag on paper to move it. Shift-drag on paper to rotate it.",
            MouseMode::Edge => "Edge mode. Click on an edge to split/join pieces. Shift-click to join a full strip of quads.",
            MouseMode::Tab => "Tab mode. Click on an edge to swap the side of a tab.",
        };
        ui.text(status_text);

        self.build_options_dialog(ui);
        self.build_modal_error_message(ui);
        self.build_modal_wait_message_and_run_file_action(ui);
        self.build_confirm_message(ui, &mut menu_actions);
        self.build_about(ui);

        menu_actions
    }

    fn build_options_dialog(&mut self, ui: &imgui::Ui) {
        let options = match self.options_opened.as_mut() {
            Some(o) => o,
            None => return,
        };
        let mut options_opened = true;
        if let Some(_options) = ui.window("Document properties...###options")
            .size([600.0, 400.0], imgui::Condition::Once)
            .resizable(true)
            .scroll_bar(false)
            .movable(true)
            .opened(&mut options_opened)
            .begin()
        {
            let size = Vector2::from(ui.content_region_avail());
            if let Some(_ops) = ui.child_window("options")
                .size([size.x, -self.option_button_height])
                .horizontal_scrollbar(true)
                .begin()
            {
                if ui.collapsing_header("Model", imgui::TreeNodeFlags::empty()) {
                    ui.set_next_item_width(ui.current_font_size() * 5.5);
                    ui.input_float("Scale", &mut options.scale).display_format("%g").build();
                    ui.same_line_with_spacing(0.0, ui.current_font_size() * 3.0);
                    ui.checkbox("Textured", &mut options.texture);

                    if let Some(_t) = ui.tree_node_config("Tabs")
                        //.flags(imgui::TreeNodeFlags::DEFAULT_OPEN)
                        .push()
                    {
                        static TAB_STYLES: &[TabStyle] = &[
                            TabStyle::Textured,
                            TabStyle::HalfTextured,
                            TabStyle::White,
                            TabStyle::None,
                        ];
                        fn fmt_tab_style(s: TabStyle) -> &'static str {
                            match s {
                                TabStyle::Textured => "Textured",
                                TabStyle::HalfTextured => "Half textured",
                                TabStyle::White => "White",
                                TabStyle::None => "None",
                            }
                        }
                        let mut i_tab_style = TAB_STYLES.iter().position(|s| *s == options.tab_style).unwrap_or(0);
                        ui.set_next_item_width(ui.current_font_size() * 8.0);
                        if ui.combo("Style", &mut i_tab_style, TAB_STYLES, |s| fmt_tab_style(*s).into()) {
                            options.tab_style = TAB_STYLES[i_tab_style];
                        }
                        ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                        ui.set_next_item_width(ui.current_font_size() * 5.5);
                        ui.input_float("Width", &mut options.tab_width).display_format("%g").build();

                        ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                        ui.set_next_item_width(ui.current_font_size() * 5.5);
                        ui.input_float("Angle", &mut options.tab_angle).display_format("%g").build();
                    }
                    if let Some(_t) = ui.tree_node("Folds") {
                        static FOLD_STYLES: &[FoldStyle] = &[
                            FoldStyle::Full,
                            FoldStyle::FullAndOut,
                            FoldStyle::Out,
                            FoldStyle::In,
                            FoldStyle::InAndOut,
                            FoldStyle::None,
                        ];
                        fn fmt_fold_style(s: FoldStyle) -> &'static str {
                            match s {
                                FoldStyle::Full => "Full line",
                                FoldStyle::FullAndOut => "Full & out segment",
                                FoldStyle::Out => "Out segment",
                                FoldStyle::In => "In segment",
                                FoldStyle::InAndOut => "Out & in segment",
                                FoldStyle::None => "None",
                            }
                        }
                        let mut i_fold_style = FOLD_STYLES.iter().position(|s| *s == options.fold_style).unwrap_or(0);
                        ui.set_next_item_width(ui.current_font_size() * 8.0);
                        if ui.combo("Style", &mut i_fold_style, FOLD_STYLES, |s| fmt_fold_style(*s).into()) {
                            options.fold_style = FOLD_STYLES[i_fold_style];
                        }
                        ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                        ui.set_next_item_width(ui.current_font_size() * 5.5);
                        ui.input_float("Length", &mut options.fold_line_len).display_format("%g").build();
                        ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                        ui.set_next_item_width(ui.current_font_size() * 5.5);
                        ui.input_float("Line width", &mut options.fold_line_width).display_format("%g").build();
                    }
                    if let Some(_t) = ui.tree_node("Information") {
                        let n_pieces = self.data.papercraft.num_islands();
                        let n_tabs = self.data.papercraft.model().edges()
                            .filter(|(e, _)| matches!(self.data.papercraft.edge_status(*e), paper::EdgeStatus::Cut(_)))
                            .count();
                        let bbox = util_3d::bounding_box_3d(
                            self.data.papercraft.model()
                            .vertices()
                            .map(|(_, v)| v.pos())
                            );
                        let model_size = (bbox.1 - bbox.0) * options.scale;
                        let Vector3 { x, y, z } = model_size;
                        ui.text(&format!("Number of pieces: {n_pieces}\nNumber of tabs: {n_tabs}\nReal size (mm): {x:.0} x {y:.0} x {z:.0}"));
                    }
                }
                if ui.collapsing_header("Page layout", imgui::TreeNodeFlags::empty()) {
                    ui.set_next_item_width(ui.current_font_size() * 5.5);

                    let mut i = options.pages as _;
                    ui.input_int("Pages", &mut i).build();
                    options.pages = i.clamp(1, 1000) as _;

                    ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                    ui.set_next_item_width(ui.current_font_size() * 5.5);

                    let mut i = options.page_cols as _;
                    ui.input_int("Columns", &mut i).build();
                    options.page_cols = i.clamp(1, options.pages as _) as _;

                    ui.set_next_item_width(ui.current_font_size() * 11.0);
                    ui.checkbox("Print Papercraft signature", &mut options.show_self_promotion);

                    ui.same_line_with_spacing(0.0, ui.current_font_size() * 3.0);
                    ui.set_next_item_width(ui.current_font_size() * 11.0);
                    ui.checkbox("Print page number", &mut options.show_page_number);
                }
                if ui.collapsing_header("Paper size", imgui::TreeNodeFlags::empty()) {
                    ui.set_next_item_width(ui.current_font_size() * 5.5);
                    ui.input_float("Width", &mut options.page_size.0).display_format("%g").build();
                    ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                    ui.set_next_item_width(ui.current_font_size() * 5.5);
                    ui.input_float("Height", &mut options.page_size.1).display_format("%g").build();
                    ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                    ui.set_next_item_width(ui.current_font_size() * 5.5);
                    let mut resolution = options.resolution as f32;
                    ui.input_float("DPI", &mut resolution).display_format("%g").build();
                    options.resolution = resolution as u32;

                    struct PaperSize {
                        name: &'static str,
                        size: Vector2,
                    }
                    static PAPER_SIZES: &[PaperSize] = &[
                        PaperSize {
                            name: "A4",
                            size: Vector2::new(210.0, 297.0),

                        },
                        PaperSize {
                            name: "A3",
                            size: Vector2::new(297.0, 420.0),
                        },
                        PaperSize {
                            name: "Letter",
                            size: Vector2::new(215.9, 279.4),
                        },
                        PaperSize {
                            name: "Legal",
                            size: Vector2::new(215.9, 355.6),
                        },
                    ];

                    let paper_size = Vector2::from(options.page_size);
                    let mut i_paper_size = PAPER_SIZES.iter().position(|s| s.size == paper_size || s.size == Vector2::new(paper_size.y, paper_size.x)).unwrap_or(usize::MAX);
                    ui.set_next_item_width(ui.current_font_size() * 8.0);
                    if ui.combo("##", &mut i_paper_size, PAPER_SIZES, |t| t.name.into()) {
                        let portrait = options.page_size.1 >= options.page_size.0;
                        options.page_size = PAPER_SIZES[i_paper_size].size.into();
                        if !portrait {
                            std::mem::swap(&mut options.page_size.0, &mut options.page_size.1);
                        }
                    }
                    let mut portrait = options.page_size.1 >= options.page_size.0;
                    let old_portrait = portrait;
                    ui.radio_button("Portrait", &mut portrait, true);
                    ui.radio_button("Landscape", &mut portrait, false);
                    if portrait != old_portrait {
                        std::mem::swap(&mut options.page_size.0, &mut options.page_size.1);
                    }
                }
                if ui.collapsing_header("Margins", imgui::TreeNodeFlags::empty()) {
                    ui.set_next_item_width(ui.current_font_size() * 4.0);
                    ui.input_float("Top", &mut options.margin.0).display_format("%g").build();
                    ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                    ui.set_next_item_width(ui.current_font_size() * 4.0);
                    ui.input_float("Left", &mut options.margin.1).display_format("%g").build();
                    ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                    ui.set_next_item_width(ui.current_font_size() * 4.0);
                    ui.input_float("Right", &mut options.margin.2).display_format("%g").build();
                    ui.same_line_with_spacing(0.0, ui.current_font_size() * 1.5);
                    ui.set_next_item_width(ui.current_font_size() * 4.0);
                    ui.input_float("Bottom", &mut options.margin.3).display_format("%g").build();
                }
            }

            let mut apply_options = None;
            let pos1 = Vector2::from(ui.cursor_screen_pos());
            ui.separator();
            if ui.button_with_size("OK", [100.0, 0.0]) {
                apply_options = self.options_opened.take();
            }
            ui.same_line();
            if ui.button_with_size("Cancel", [100.0, 0.0]) {
                self.options_opened = None;
            }
            ui.same_line();
            if ui.button_with_size("Apply", [100.0, 0.0]) {
                apply_options = self.options_opened.clone();
            }
            // Compute the height of the buttons to avoid having an external scrollbar
            let pos2 = Vector2::from(ui.cursor_screen_pos());
            self.option_button_height = pos2.y - pos1.y;

            if let Some(options) = apply_options {
                let island_pos = self.data.papercraft.islands()
                    .map(|(_, island)| (island.root_face(), (island.rotation(), island.location())))
                    .collect();
                self.data.show_textures = options.texture;
                let old_options = self.data.papercraft.set_options(options);
                self.data.push_undo_action(vec![UndoAction::DocConfig { options: old_options, island_pos }]);
                self.add_rebuild(RebuildFlags::ALL);
            }
        }
        // If the windows was closed
        if !options_opened {
            self.options_opened = None;
        }
    }

    fn check_modified(&self) -> BoolWithConfirm {
        if self.data.modified {
            BoolWithConfirm::Requested
        } else {
            BoolWithConfirm::Confirmed
        }
    }
    fn build_menu_and_file_dialog(&mut self, ui: &imgui::Ui) -> MenuActions {
        let mut menu_actions = MenuActions::default();

        ui.menu_bar(|| {
            ui.menu("File", || {
                let title = "Open...";
                if ui.menu_item_config(title)
                    .shortcut("Ctrl+O")
                    .build()
                {
                    menu_actions.open = self.check_modified();
                }
                if ui.menu_item_config("Save")
                    .shortcut("Ctrl+S")
                    .build()
                {
                    menu_actions.save = true;
                }
                let title = "Save as...";
                if ui.menu_item(title) {
                    menu_actions.save_as = true;
                }
                let title = "Import OBJ...";
                if ui.menu_item(title) {
                    menu_actions.import_obj = self.check_modified();
                }
                let title = "Update with new OBJ...";
                if ui.menu_item(title) {
                    menu_actions.update_obj = self.check_modified();
                }
                let title = "Export OBJ...";
                if ui.menu_item(title) {
                    menu_actions.export_obj = true;
                }
                let title = "Generate PDF...";
                if ui.menu_item(title) {
                    menu_actions.generate_pdf = true;
                }
                ui.separator();
                if ui.menu_item_config("Quit")
                    .shortcut("Ctrl+Q")
                    .build()
                {
                    menu_actions.quit = self.check_modified();
                }
            });
            ui.menu("Edit", || {
                if ui.menu_item_config("Undo")
                    .shortcut("Ctrl+Z")
                    .enabled(self.data.can_undo())
                    .build()
                {
                    menu_actions.undo = true;
                }

                ui.separator();

                if ui.menu_item_config("Document properties")
                    .build_with_ref(&mut self.options_opened.is_some())
                {
                    self.options_opened = match self.options_opened {
                        Some(_) => None,
                        None => Some(self.data.papercraft.options().clone()),
                    }
                }

                ui.separator();

                if ui.menu_item_config("Face/Island")
                    .shortcut("F5")
                    .build_with_ref(&mut (self.data.mode == MouseMode::Face))
                {
                    self.set_mouse_mode(MouseMode::Face);
                }
                if ui.menu_item_config("Split/Join edge")
                    .shortcut("F6")
                    .build_with_ref(&mut (self.data.mode == MouseMode::Edge))
                {
                    self.set_mouse_mode(MouseMode::Edge);
                }
                if ui.menu_item_config("Tabs")
                    .shortcut("F7")
                    .build_with_ref(&mut (self.data.mode == MouseMode::Tab))
                {
                    self.set_mouse_mode(MouseMode::Tab);
                }

                ui.separator();

                if ui.menu_item("Reset views") {
                    menu_actions.reset_views = true;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
                }
                if ui.menu_item("Repack pieces") {
                    let undo = self.data.pack_islands();
                    self.data.push_undo_action(undo);
                    self.add_rebuild(RebuildFlags::PAPER | RebuildFlags::SELECTION);
                }
            });
            ui.menu("View", || {
                if ui.menu_item_config("Textures")
                    .enabled(self.data.papercraft.options().texture)
                    .build_with_ref(&mut self.data.show_textures)
                {
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
                }
                if ui.menu_item_config("3D lines")
                    .build_with_ref(&mut self.data.show_3d_lines)
                {
                    self.add_rebuild(RebuildFlags::SCENE_REDRAW);
                }
                if ui.menu_item_config("Tabs")
                    .build_with_ref(&mut self.data.show_tabs)
                {
                    self.add_rebuild(RebuildFlags::PAPER);
                }
                if ui.menu_item_config("X-ray selection")
                    .build_with_ref(&mut self.data.xray_selection)
                {
                    self.add_rebuild(RebuildFlags::SELECTION);
                }
                if ui.menu_item_config("Highlight overlaps")
                    .build_with_ref(&mut self.data.highlight_overlaps)
                {
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW);
                }
            });
            ui.menu("Help", || {
                ui.menu_item_config("About...")
                    .build_with_ref(&mut self.about_visible);
            });
        });

        let is_popup_open = unsafe {
            imgui_sys::igIsPopupOpen(std::ptr::null(), imgui_sys::ImGuiPopupFlags_AnyPopup as i32)
        };
        if !is_popup_open {
            if ui.is_key_index_pressed(VirtualKeyCode::F5 as _) {
                self.set_mouse_mode(MouseMode::Face);
            }
            if ui.is_key_index_pressed(VirtualKeyCode::F6 as _) {
                self.set_mouse_mode(MouseMode::Edge);
            }
            if ui.is_key_index_pressed(VirtualKeyCode::F7 as _) {
                self.set_mouse_mode(MouseMode::Tab);
            }
            if ui.io().key_ctrl && ui.is_key_index_pressed(VirtualKeyCode::Z as _) {
                menu_actions.undo = true;
            }
            if ui.io().key_ctrl && ui.is_key_index_pressed(VirtualKeyCode::Q as _) {
                menu_actions.quit = self.check_modified();
            }
            if ui.io().key_ctrl && ui.is_key_index_pressed(VirtualKeyCode::O as _) {
                menu_actions.open = self.check_modified();
            }
            if ui.io().key_ctrl && ui.is_key_index_pressed(VirtualKeyCode::S as _) {
                menu_actions.save = true;
            }
        }

        menu_actions
    }

    fn build_scene(&mut self, ui: &imgui::Ui, width: f32) {
        if let Some(_scene) = ui.child_window("scene")
            //.size([300.0, 300.0], imgui::Condition::Once)
            //.movable(true)
            .size([width, 0.0])
            .border(true)
            .begin()
        {
            let scale = Vector2::from(ui.io().display_framebuffer_scale);
            let pos = scale_size(scale, Vector2::from(ui.cursor_screen_pos()));
            let dsp_size = scale_size(scale, Vector2::from(ui.io().display_size));

            canvas3d(ui, &mut self.scene_ui_status);

            let draws = ui.get_window_draw_list();
            draws.add_callback({
                let this = self.this.clone();
                move || {
                    let this = this.upgrade().unwrap();
                    let mut this = this.borrow_mut();

                    unsafe {
                        gl::Disable(gl::SCISSOR_TEST);

                        if this.render_scene_pending {
                            let _backup = BackupGlConfig::backup();
                            let _draw_fb_binder = BinderDrawFramebuffer::bind(&this.gl_fixs.fbo_scene);
                            let _vp = glr::PushViewport::push(0, 0, this.sz_scene.x as i32, this.sz_scene.y as i32);
                            this.render_scene();
                        }

                        // blit the FBO to the real FB
                        let pos_y2 = dsp_size.y - pos.y - this.sz_scene.y;
                        let x = (pos.x * 1.0) as i32;
                        let y = (pos_y2 * 1.0) as i32;
                        let width = (this.sz_scene.x * 1.0) as i32;
                        let height = (this.sz_scene.y * 1.0) as i32;

                        let _read_fb_binder = BinderReadFramebuffer::bind(&this.gl_fixs.fbo_scene);
                        gl::BlitFramebuffer(
                            0, 0, width, height,
                            x, y, x + width, y + height,
                            gl::COLOR_BUFFER_BIT, gl::NEAREST
                        );
                        gl::Enable(gl::SCISSOR_TEST);
                    }
                }
            }).build();
        } else {
            self.scene_ui_status = Canvas3dStatus::default();
        }
    }

    fn build_paper(&mut self, ui: &imgui::Ui) {
        if let Some(_paper) = ui.child_window("paper")
            //.size([300.0, 300.0], imgui::Condition::Once)
            //.movable(true)
            .size([-1.0, -1.0])
            .border(true)
            .begin()
        {
            let scale = Vector2::from(ui.io().display_framebuffer_scale);
            let pos = scale_size(scale, Vector2::from(ui.cursor_screen_pos()));
            let dsp_size = scale_size(scale, Vector2::from(ui.io().display_size));

            canvas3d(ui, &mut self.paper_ui_status);

            let draws = ui.get_window_draw_list();
            draws.add_callback({
                let this = self.this.clone();
                move || {
                    let this = this.upgrade().unwrap();
                    let mut this = this.borrow_mut();

                    unsafe {
                        gl::Disable(gl::SCISSOR_TEST);

                        if this.render_paper_pending {
                            let _backup = BackupGlConfig::backup();
                            let _draw_fb_binder = BinderDrawFramebuffer::bind(&this.gl_fixs.fbo_paper);
                            let _vp = glr::PushViewport::push(0, 0, this.sz_paper.x as i32, this.sz_paper.y as i32);
                            this.render_paper();
                        }

                        // blit the FBO to the real FB
                        let pos_y2 = dsp_size.y - pos.y - this.sz_paper.y;
                        let x = pos.x as i32;
                        let y = pos_y2 as i32;
                        let width = this.sz_paper.x as i32;
                        let height = this.sz_paper.y as i32;

                        let _read_fb_binder = BinderReadFramebuffer::bind(&this.gl_fixs.fbo_paper);
                        gl::BlitFramebuffer(
                            0, 0, width, height,
                            x, y, x + width, y + height,
                            gl::COLOR_BUFFER_BIT, gl::NEAREST
                        );
                        gl::Enable(gl::SCISSOR_TEST);
                    }
                }
            }).build();
        } else {
            self.paper_ui_status = Canvas3dStatus::default();
        }
    }

    fn open_confirmation_dialog(&mut self, ui: &imgui::Ui, title: &str, message: &str, f: impl Fn(&mut MenuActions) + 'static) {
        self.confirmable_action = Some((
            String::from(title),
            String::from(message),
            Box::new(f),
        ));
        ui.open_popup("###Confirm");
    }

    fn run_menu_actions(&mut self, ui: &imgui::Ui, menu_actions: &MenuActions) {
        if menu_actions.reset_views {
            self.data.reset_views(self.sz_scene, self.sz_paper);
        }
        if menu_actions.undo {
            match self.data.undo_action() {
                UndoResult::Model => {
                    self.add_rebuild(RebuildFlags::ALL);
                }
                UndoResult::ModelAndOptions => {
                    if let Some(o) = self.options_opened.as_mut() {
                        *o = self.data.papercraft.options().clone();
                    }
                    self.add_rebuild(RebuildFlags::ALL);
                }
                UndoResult::False => {},
            }
        }

        let mut save_as = false;
        let mut open_file_dialog = false;
        let mut open_wait = false;

        match menu_actions.open {
            BoolWithConfirm::Requested => {
                self.open_confirmation_dialog(ui,
                    "Load model",
                    "The model has not been save, continue anyway?",
                    |a| a.open = BoolWithConfirm::Confirmed
                );
            }
            BoolWithConfirm::Confirmed => {
                let mut fd = imgui_filedialog::FileDialog::new();
                fd.open("fd", "", "Papercraft (*.craft) {.craft},All files {.*}", "", "", 1,
                    imgui_filedialog::Flags::DISABLE_CREATE_DIRECTORY_BUTTON | imgui_filedialog::Flags::NO_DIALOG);
                self.file_dialog = Some((fd, "Open...", FileAction::OpenCraft));
                open_file_dialog = true;
            }
            BoolWithConfirm::None => {}
        }
        if menu_actions.save {
            match &self.data.file_name {
                Some(f) => {
                    self.file_action = Some((FileAction::SaveAsCraft, f.clone()));
                    open_wait = true;
                }
                None => save_as = true,
            }
        }
        if menu_actions.save_as || save_as {
            let mut fd = imgui_filedialog::FileDialog::new();
            fd.open("fd", "", "Papercraft (*.craft) {.craft},All files {.*}", "", "", 1,
                imgui_filedialog::Flags::CONFIRM_OVERWRITE | imgui_filedialog::Flags::NO_DIALOG);
            self.file_dialog = Some((fd, "Save as...", FileAction::SaveAsCraft));
            open_file_dialog = true;
        }
        match menu_actions.import_obj {
            BoolWithConfirm::Requested => {
                self.open_confirmation_dialog(ui,
                    "Import model",
                    "The model has not been save, continue anyway?",
                    |a| a.import_obj = BoolWithConfirm::Confirmed
                );
            }
            BoolWithConfirm::Confirmed => {
                let mut fd = imgui_filedialog::FileDialog::new();
                fd.open("fd", "", "Wavefront (*.obj) {.obj},All files {.*}", "", "", 1,
                    imgui_filedialog::Flags::DISABLE_CREATE_DIRECTORY_BUTTON | imgui_filedialog::Flags::NO_DIALOG);
                self.file_dialog = Some((fd, "Import OBJ...", FileAction::ImportObj));
                open_file_dialog = true;
            }
            BoolWithConfirm::None => {}
        }
        match menu_actions.update_obj {
            BoolWithConfirm::Requested => {
                self.open_confirmation_dialog(ui,
                    "Update model",
                    "This model is not saved and this operation cannot be undone.\nContinue anyway?",
                    |a| a.update_obj = BoolWithConfirm::Confirmed
                );
            }
            BoolWithConfirm::Confirmed => {
                let mut fd = imgui_filedialog::FileDialog::new();
                fd.open("fd", "", "Wavefront (*.obj) {.obj},All files {.*}", "", "", 1,
                    imgui_filedialog::Flags::DISABLE_CREATE_DIRECTORY_BUTTON | imgui_filedialog::Flags::NO_DIALOG);
                self.file_dialog = Some((fd, "Update with new OBJ...", FileAction::UpdateObj));
                open_file_dialog = true;
            }
            BoolWithConfirm::None => {}
        }
        if menu_actions.export_obj {
            let mut fd = imgui_filedialog::FileDialog::new();
            fd.open("fd", "", "Wavefront (*.obj) {.obj},All files {.*}", "", "", 1,
                imgui_filedialog::Flags::CONFIRM_OVERWRITE | imgui_filedialog::Flags::NO_DIALOG);
            self.file_dialog = Some((fd, "Export OBJ...", FileAction::ExportObj));
            open_file_dialog = true;
        }
        if menu_actions.generate_pdf {
            let mut fd = imgui_filedialog::FileDialog::new();
            fd.open("fd", "", "PDF document (*.pdf) {.pdf},All files {.*}", "", "", 1,
                imgui_filedialog::Flags::CONFIRM_OVERWRITE | imgui_filedialog::Flags::NO_DIALOG);
            self.file_dialog = Some((fd, "Generate PDF...", FileAction::GeneratePdf));
            open_file_dialog = true;
        }

        // There are two Wait modals and two Error modals. One pair over the FileDialog, the other to be opened directly ("Save").

        if open_file_dialog {
            ui.open_popup("###file_dialog_modal");
        }
        if let Some((mut fd, title, action)) = self.file_dialog.take() {
            let dsp_size = Vector2::from(ui.io().display_size);
            let min_size: [f32; 2] = (dsp_size * 0.75).into();
            let max_size: [f32; 2] = dsp_size.into();
            unsafe {
                imgui_sys::igSetNextWindowSizeConstraints(
                    min_size.into(),
                    max_size.into(),
                    None,
                    std::ptr::null_mut(),
                );
            };
            if let Some(_pop) = ui.modal_popup_config(&format!("{title}###file_dialog_modal"))
                .opened(&mut true)
                .begin_popup()
            {
                let mut finish_file_dialog = false;
                let size = ui.content_region_avail();
                if let Some(fd2) = fd.display("fd", imgui::WindowFlags::empty(), size, size) {
                    if fd2.ok() {
                        if let Some(file) = fd2.file_path_name() {
                            self.file_action = Some((action, file.into()));
                            open_wait = true;
                        }
                    }
                    finish_file_dialog = true;
                    ui.close_current_popup();
                }
                if !finish_file_dialog {
                    self.file_dialog = Some((fd, title, action));
                }
            }
        }

        if open_wait {
            self.popup_time_start = Instant::now();
            ui.open_popup("###Wait");
        }
    }

    fn run_mouse_actions(&mut self, ui: &imgui::Ui) {
        let mut ev_state = ModifierType::empty();
        if ui.io().key_shift {
            ev_state.insert(ModifierType::SHIFT_MASK);
        }
        if ui.io().key_ctrl {
            ev_state.insert(ModifierType::CONTROL_MASK);
        }

        let mouse_pos = self.scene_ui_status.mouse_pos;
        match &self.scene_ui_status.action {
            Canvas3dAction::Hovering | Canvas3dAction::Pressed(_) => {
                let flags = self.data.scene_motion_notify_event(self.sz_scene, mouse_pos, ev_state);
                self.add_rebuild(flags);
                'zoom: {
                    let dz = match ui.io().mouse_wheel {
                        x if x < 0.0 => 1.0 / 1.1,
                        x if x > 0.0 => 1.1,
                        _ => break 'zoom,
                    };
                    self.data.trans_scene.scale *= dz;
                    self.data.trans_scene.recompute_obj();
                    self.add_rebuild(RebuildFlags::SCENE_REDRAW);
                }
            }
            Canvas3dAction::DoubleClicked(imgui::MouseButton::Left) => {
                let selection = self.data.scene_analyze_click(MouseMode::Face,self.sz_scene, mouse_pos);
                if let ClickResult::Face(i_face) = selection {
                    let gl_objs = self.data.gl_objs.as_ref().unwrap();
                    // Compute the average of all the faces flat with the selected one, and move it to the center of the paper.
                    // Some vertices are counted twice, but they tend to be in diagonally opposed so the compensate, and it is an approximation anyways.
                    let mut center = Vector2::zero();
                    let mut n = 0.0;
                    for i_face in self.data.papercraft.get_flat_faces(i_face) {
                        let idx = 3 * gl_objs.paper_face_index[usize::from(i_face)] as usize;
                        for i in idx .. idx + 3 {
                            center += gl_objs.paper_vertices[i].pos;
                            n += 1.0;
                        }
                    }
                    center /= n;
                    self.data.trans_paper.mx[2][0] = -center.x * self.data.trans_paper.mx[0][0];
                    self.data.trans_paper.mx[2][1] = -center.y * self.data.trans_paper.mx[1][1];
                    self.add_rebuild(RebuildFlags::SCENE_REDRAW);
                }
            }
            Canvas3dAction::Released(imgui::MouseButton::Left) => {
                ev_state.insert(ModifierType::BUTTON1_MASK);
                let selection = self.data.scene_analyze_click(self.data.mode, self.sz_scene, mouse_pos);
                match (self.data.mode, selection) {
                    (MouseMode::Edge, ClickResult::Edge(i_edge, i_face)) => {
                        let undo = if ev_state.contains(ModifierType::SHIFT_MASK) {
                            self.data.try_join_strip(i_edge)
                        } else {
                            self.data.edge_toggle_cut(i_edge, i_face)
                        };
                        if let Some(undo) = undo {
                            self.data.push_undo_action(undo);
                        }
                        self.add_rebuild(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                    }
                    (MouseMode::Tab, ClickResult::Edge(i_edge, _)) => {
                        self.data.papercraft.edge_toggle_tab(i_edge);
                        self.data.push_undo_action(vec![UndoAction::TabToggle { i_edge } ]);
                        self.add_rebuild(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                    }
                    (_, ClickResult::Face(f)) => {
                        let flags = self.data.set_selection(ClickResult::Face(f), true, ev_state.contains(ModifierType::CONTROL_MASK));
                        self.add_rebuild(flags);
                    }
                    (_, ClickResult::None) => {
                        let flags = self.data.set_selection(ClickResult::None, true, ev_state.contains(ModifierType::CONTROL_MASK));
                        self.add_rebuild(flags);
                    }
                    _ => {}
                };
            }
            Canvas3dAction::Dragging(bt) => {
                match bt {
                    imgui::MouseButton::Left => ev_state.insert(ModifierType::BUTTON1_MASK),
                    imgui::MouseButton::Right => ev_state.insert(ModifierType::BUTTON2_MASK),
                    _ => ()
                }
                let flags = self.data.scene_motion_notify_event(self.sz_scene, mouse_pos, ev_state);
                self.add_rebuild(flags);
            }
            _ => {}
        }

        let mouse_pos = self.paper_ui_status.mouse_pos;
        match &self.paper_ui_status.action {
            Canvas3dAction::Hovering | Canvas3dAction::Pressed(_) => {
                if self.paper_ui_status.action == Canvas3dAction::Hovering {
                    self.data.rotation_center = None;
                    self.data.grabbed_island = None;
                }

                let flags = self.data.paper_motion_notify_event(self.sz_paper, mouse_pos, ev_state);
                self.add_rebuild(flags);

                'zoom: {
                    let dz = match ui.io().mouse_wheel {
                        x if x < 0.0 => 1.0 / 1.1,
                        x if x > 0.0 => 1.1,
                        _ => break 'zoom,
                    };
                    let pos = mouse_pos - self.sz_paper / 2.0;
                    self.data.trans_paper.mx = Matrix3::from_translation(pos) * Matrix3::from_scale(dz) * Matrix3::from_translation(-pos) * self.data.trans_paper.mx;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW);
                }
            }
            Canvas3dAction::Clicked(imgui::MouseButton::Left) |
            Canvas3dAction::DoubleClicked(imgui::MouseButton::Left) => {
                ev_state.insert(ModifierType::BUTTON1_MASK);

                let selection = self.data.paper_analyze_click(self.data.mode, self.sz_paper, mouse_pos);
                match (self.data.mode, selection) {
                    (MouseMode::Edge, ClickResult::Edge(i_edge, i_face)) => {
                        self.data.grabbed_island = None;

                        let undo = if ev_state.contains(ModifierType::SHIFT_MASK) {
                            self.data.try_join_strip(i_edge)
                        } else {
                            self.data.edge_toggle_cut(i_edge, i_face)
                        };
                        if let Some(undo) = undo {
                            self.data.push_undo_action(undo);
                        }
                        self.add_rebuild(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                    }
                    (MouseMode::Tab, ClickResult::Edge(i_edge, _)) => {
                        self.data.papercraft.edge_toggle_tab(i_edge);
                        self.data.push_undo_action(vec![UndoAction::TabToggle { i_edge } ]);
                        self.add_rebuild(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                    }
                    (_, ClickResult::Face(f)) => {
                        let flags = self.data.set_selection(ClickResult::Face(f), true, ev_state.contains(ModifierType::CONTROL_MASK));
                        self.add_rebuild(flags);
                        let undo_action: Vec<_> = self.data.selected_islands
                            .iter()
                            .map(|&i_island| {
                                let island = self.data.papercraft.island_by_key(i_island).unwrap();
                                UndoAction::IslandMove { i_root: island.root_face(), prev_rot: island.rotation(), prev_loc: island.location() }
                            })
                            .collect();
                        self.data.grabbed_island.get_or_insert_with(Vec::new).extend(undo_action);
                    }
                    (_, ClickResult::None) => {
                        let flags = self.data.set_selection(ClickResult::None, true, ev_state.contains(ModifierType::CONTROL_MASK));
                        self.add_rebuild(flags);
                        self.data.grabbed_island = None;
                    }
                    _ => {}
                }
            }
            Canvas3dAction::Dragging(bt) => {
                match bt {
                    imgui::MouseButton::Left => ev_state.insert(ModifierType::BUTTON1_MASK),
                    imgui::MouseButton::Right => ev_state.insert(ModifierType::BUTTON2_MASK),
                    _ => ()
                }
                let flags = self.data.paper_motion_notify_event(self.sz_paper, mouse_pos, ev_state);
                self.add_rebuild(flags);

                //Scroll timer, if rotating the piece do not do the scroll, it will not go well.
                'scroll: {
                    if !ev_state.contains(ModifierType::SHIFT_MASK) && self.data.grabbed_island.is_some() {
                        let delta = if mouse_pos.x < 5.0 {
                            Vector2::new((-mouse_pos.x).max(5.0).min(25.0), 0.0)
                        } else if mouse_pos.x > self.sz_paper.x - 5.0 {
                            Vector2::new(-(mouse_pos.x - self.sz_paper.x).max(5.0).min(25.0), 0.0)
                        } else if mouse_pos.y < 5.0 {
                            Vector2::new(0.0, (-mouse_pos.y).max(5.0).min(25.0))
                        } else if mouse_pos.y > self.sz_paper.y - 5.0 {
                            Vector2::new(0.0, -(mouse_pos.y - self.sz_paper.y).max(5.0).min(25.0))
                        } else {
                            break 'scroll;
                        };
                        let delta = delta / 2.0;
                        self.data.last_cursor_pos += delta;
                        self.data.trans_paper.mx = Matrix3::from_translation(delta) * self.data.trans_paper.mx;
                        let flags = self.data.paper_motion_notify_event(self.sz_paper, mouse_pos, ev_state);
                        self.add_rebuild(flags | RebuildFlags::PAPER_REDRAW);
                    }
                }
            }
            _ => {}
        }

    }

    fn render_scene(&mut self) {
        let gl_objs = self.data.gl_objs.as_ref().unwrap();
        let gl_fixs = &self.gl_fixs;

        let light0 = Vector3::new(-0.5, -0.4, -0.8).normalize() * 0.55;
        let light1 = Vector3::new(0.8, 0.2, 0.4).normalize() * 0.25;

        let mut u = Uniforms3D {
            m: self.data.trans_scene.persp * self.data.trans_scene.view,
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
            gl::BlendFuncSeparate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA, gl::ONE, gl::ONE_MINUS_SRC_ALPHA);

            gl::BindVertexArray(gl_fixs.vao.id());
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
    fn render_paper(&mut self) {
        let gl_objs = self.data.gl_objs.as_ref().unwrap();
        let gl_fixs = &self.gl_fixs;

        let mut u = Uniforms2D {
            m: self.data.trans_paper.ortho * self.data.trans_paper.mx,
            tex: 0,
            frac_dash: 0.5,
            line_color: Rgba::new(0.0, 0.0, 0.0, 0.0),
            texturize: 0,
            notex_color: Rgba::new(0.75, 0.75, 0.75, 1.0),
        };

        unsafe {
            gl::ClearColor(0.7, 0.7, 0.7, 1.0);
            gl::ClearStencil(1);
            gl::StencilMask(0xff);
            gl::StencilFunc(gl::ALWAYS, 0, 0);
            gl::Disable(gl::STENCIL_TEST);

            gl::Clear(gl::COLOR_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);

            gl::Enable(gl::BLEND);
            gl::BlendFuncSeparate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA, gl::ONE, gl::ONE_MINUS_SRC_ALPHA);

            gl::BindVertexArray(gl_fixs.vao.id());
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
        }
    }

    fn add_rebuild(&mut self, flags: RebuildFlags) {
        self.data.rebuild.insert(flags);
    }
    fn set_mouse_mode(&mut self, mode: MouseMode) {
        self.data.mode = mode;
        self.add_rebuild(RebuildFlags::SELECTION | RebuildFlags::SCENE_REDRAW | RebuildFlags::PAPER_REDRAW);
    }

    fn title(&self, with_unsaved_check: bool) -> String {
        let unsaved = if with_unsaved_check && self.data.modified { "*" } else { "" };
        let app_name = "Papercraft";
        match &self.data.file_name {
            Some(f) => {
                let name = if let Some(name) = f.file_name() {
                    name.to_string_lossy()
                } else {
                    f.as_os_str().to_string_lossy()
                };
                format!("{unsaved}{name} - {app_name}")
            }
            None => {
                if unsaved.is_empty() {
                    app_name.to_owned()
                } else {
                    format!("{unsaved} - {app_name}")
                }
            }
        }
    }
    fn run_file_action(&mut self, action: FileAction, file_name: impl AsRef<Path>) -> anyhow::Result<()> {
        match action {
            FileAction::OpenCraft => self.open_craft(file_name)?,
            FileAction::SaveAsCraft => {
                self.save_as_craft(&file_name)?;
                self.data.modified = false;
                self.data.file_name = Some(file_name.as_ref().to_owned());
            }
            FileAction::ImportObj => {
                self.import_obj(file_name)?;
                self.data.modified = true;
            }
            FileAction::UpdateObj => {
                self.update_obj(file_name)?;
                self.data.modified = true;
            }
            FileAction::ExportObj => self.export_obj(file_name)?,
            FileAction::GeneratePdf => self.generate_pdf(file_name)?,
        }
        Ok(())
    }
    fn open_craft(&mut self, file_name: impl AsRef<Path>) -> anyhow::Result<()> {
        let fs = std::fs::File::open(&file_name)
            .with_context(|| format!("Error opening file {}", file_name.as_ref().display()))?;
        let fs = std::io::BufReader::new(fs);
        let papercraft = Papercraft::load(fs)
            .with_context(|| format!("Error loading file {}", file_name.as_ref().display()))?;
        self.data = PapercraftContext::from_papercraft(papercraft, Some(file_name.as_ref()), self.sz_scene, self.sz_paper);
        Ok(())
    }
    fn save_as_craft(&self, file_name: impl AsRef<Path>) -> anyhow::Result<()> {
        let f = std::fs::File::create(&file_name)
            .with_context(|| format!("Error creating file {}", file_name.as_ref().display()))?;
        let f = std::io::BufWriter::new(f);
        self.data.papercraft.save(f)
            .with_context(|| format!("Error saving file {}", file_name.as_ref().display()))?;
        Ok(())
    }
    fn import_obj(&mut self, file_name: impl AsRef<Path>) -> anyhow::Result<()> {
        let papercraft = Papercraft::import_waveobj(file_name.as_ref())
            .with_context(|| format!("Error reading Wavefront file {}", file_name.as_ref().display()))?;
        self.data = PapercraftContext::from_papercraft(papercraft, None, self.sz_scene, self.sz_paper);
        Ok(())
    }
    fn update_obj(&mut self, file_name: impl AsRef<Path>) -> anyhow::Result<()> {
        let mut new_papercraft = Papercraft::import_waveobj(file_name.as_ref())
            .with_context(|| format!("Error reading Wavefront file {}", file_name.as_ref().display()))?;
        new_papercraft.update_from_obj(&self.data.papercraft);
        let tp = self.data.trans_paper.clone();
        let ts = self.data.trans_scene.clone();
        let original_file_name = self.data.file_name.clone();
        self.data = PapercraftContext::from_papercraft(new_papercraft, original_file_name.as_deref(), self.sz_scene, self.sz_paper);
        self.data.trans_paper = tp;
        self.data.trans_scene = ts;
        Ok(())
    }
    fn export_obj(&self, file_name: impl AsRef<Path>) -> anyhow::Result<()> {
        self.data.papercraft.export_waveobj(file_name.as_ref())
            .with_context(|| format!("Error exporting to {}", file_name.as_ref().display()))?;
        Ok(())
    }
    fn generate_pdf(&self, file_name: impl AsRef<Path>) -> anyhow::Result<()> {
        unsafe { gl::Disable(gl::SCISSOR_TEST); }
        let _backup = BackupGlConfig::backup();
        self.generate_pdf_impl(file_name.as_ref())
            .with_context(|| format!("Error exporting to {}", file_name.as_ref().display()))?;
        unsafe { gl::Enable(gl::SCISSOR_TEST); }
        Ok(())
    }
    fn generate_pdf_impl(&self, file_name: &Path) -> anyhow::Result<()> {
        let options = self.data.papercraft.options();
        let resolution = options.resolution as f32;
        let (_margin_top, margin_left, margin_right, margin_bottom) = options.margin;
        let page_size_mm = Vector2::from(options.page_size);
        let page_size_inches = page_size_mm / 25.4;
        let page_size_dots = page_size_inches * 72.0;
        let page_size_pixels = page_size_inches * resolution;
        let page_size_pixels = cgmath::Vector2::new(page_size_pixels.x as i32, page_size_pixels.y as i32);

        let mut pixbuf = cairo::ImageSurface::create(cairo::Format::ARgb32, page_size_pixels.x, page_size_pixels.y)
            .with_context(|| anyhow!("Unable to create output pixbuf"))?;
        let stride = pixbuf.stride();
        let pdf = cairo::PdfSurface::new(page_size_dots.x as f64, page_size_dots.y as f64, file_name)?;
        let title  = self.title(false);
        let _ = pdf.set_metadata(cairo::PdfMetadata::Title, &title);
        let _ = pdf.set_metadata(cairo::PdfMetadata::Creator, signature());
        let cr = cairo::Context::new(&pdf)?;

        unsafe {
            gl::PixelStorei(gl::PACK_ROW_LENGTH, stride / 4);

            let fbo = glr::Framebuffer::generate();
            let rbo = glr::Renderbuffer::generate();

            let draw_fb_binder = BinderDrawFramebuffer::bind(&fbo);
            let read_fb_binder = BinderReadFramebuffer::bind(&fbo);
            let rb_binder = BinderRenderbuffer::bind(&rbo);
            gl::FramebufferRenderbuffer(draw_fb_binder.target(), gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, rbo.id());

            let rbo_fbo_no_aa = 'check_aa: {
                // multisample buffers cannot be read directly, it has to be copied to a regular one.
                for samples in glr::available_multisamples(rb_binder.target(), gl::RGBA8) {
                    // check if these many samples are usable
                    gl::RenderbufferStorageMultisample(rb_binder.target(), samples, gl::RGBA8, page_size_pixels.x, page_size_pixels.y);
                    if gl::CheckFramebufferStatus(gl::DRAW_FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
                        continue;
                    }

                    // If using AA create another FBO/RBO to blit the antialiased image before reading
                    let rbo2 = glr::Renderbuffer::generate();
                    rb_binder.rebind(&rbo2);
                    gl::RenderbufferStorage(rb_binder.target(), gl::RGBA8, page_size_pixels.x, page_size_pixels.y);

                    let fbo2 = glr::Framebuffer::generate();
                    read_fb_binder.rebind(&fbo2);
                    gl::FramebufferRenderbuffer(read_fb_binder.target(), gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, rbo2.id());

                    break 'check_aa Some((rbo2, fbo2));
                }
                println!("No multisample!");
                gl::RenderbufferStorage(rb_binder.target(), gl::RGBA8, page_size_pixels.x, page_size_pixels.y);
                None
            };
            let _vp = glr::PushViewport::push(0, 0, page_size_pixels.x, page_size_pixels.y);

            // Cairo surfaces are alpha-premultiplied:
            // * The framebuffer will be premultiplied, but the input fragments are not.
            // * The clear color is set to transparent (premultiplied).
            // * In the screen DST_ALPHA does not matter, because the framebuffer is not
            //   transparent, but here we have to set it to the proper value: use separate blend
            //   functions or we'll get the alpha squared.
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Enable(gl::BLEND);
            gl::BlendFuncSeparate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA, gl::ONE, gl::ONE_MINUS_SRC_ALPHA);

            let gl_objs = self.data.gl_objs.as_ref().unwrap();
            let gl_fixs = &self.gl_fixs;

            let mut texturize = 0;

            gl::BindVertexArray(gl_fixs.vao.id());
            if let (Some(tex), true) = (&gl_objs.textures, options.texture) {
                gl::ActiveTexture(gl::TEXTURE0);
                gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex.id());
                texturize = 1;
            }

            let ortho = util_3d::ortho2d_zero(page_size_mm.x, -page_size_mm.y);

            let page_count = options.pages;
            let tab_style = options.tab_style;

            for page in 0..page_count {
                const FONT_SIZE: f32 = 3.0;
                if options.show_self_promotion {
                    cr.set_source_rgba(0.0, 0.0, 0.0, 1.0);
                    cr.set_font_size(FONT_SIZE as f64 * 72.0 / 25.4);
                    let x = margin_left;
                    let y = (page_size_mm.y - margin_bottom + FONT_SIZE).min(page_size_mm.y - FONT_SIZE);
                    cr.move_to(x as f64 * 72.0 / 25.4, y as f64 * 72.0 / 25.4);
                    let _ = cr.show_text(signature());
                }
                if options.show_page_number {
                    cr.set_source_rgba(0.0, 0.0, 0.0, 1.0);
                    cr.set_font_size(FONT_SIZE as f64 * 72.0 / 25.4);
                    let text = format!("Page {}/{}", page + 1, page_count);
                    let ext = cr.text_extents(&text).unwrap();
                    let x = page_size_mm.x - margin_right;
                    let y = (page_size_mm.y - margin_bottom + FONT_SIZE).min(page_size_mm.y - FONT_SIZE);
                    cr.move_to(x as f64 * 72.0 / 25.4 - ext.width, y as f64 * 72.0 / 25.4);
                    let _ = cr.show_text(&text);
                }

                // Start render
                gl::Clear(gl::COLOR_BUFFER_BIT);
                let page_pos = options.page_position(page);
                let mt = Matrix3::from_translation(-page_pos);
                let u = Uniforms2D {
                    m: ortho * mt,
                    tex: 0,
                    frac_dash: 0.5,
                    line_color: Rgba::new(0.0, 0.0, 0.0, 1.0),
                    texturize,
                    notex_color: Rgba::new(1.0, 1.0, 1.0, 1.0),
                };
                // Line Tabs
                if tab_style != TabStyle::None {
                    gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_tab_edge, gl::LINES);
                }

                // Solid Tabs
                if tab_style != TabStyle::None && tab_style != TabStyle::White {
                    gl_fixs.prg_paper_solid.draw(&u, &gl_objs.paper_vertices_tab, gl::TRIANGLES);
                }

                // Borders
                gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_edge_border, gl::LINES);

                // Textured faces
                gl::VertexAttrib4f(gl_fixs.prg_paper_solid.attrib_by_name("color").unwrap().location() as u32, 0.0, 0.0, 0.0, 0.0);
                gl_fixs.prg_paper_solid.draw(&u, &gl_objs.paper_vertices, gl::TRIANGLES);

                // Creases
                gl_fixs.prg_paper_line.draw(&u, &gl_objs.paper_vertices_edge_crease, gl::LINES);
                // End render

                if let Some((_, fbo_no_aa)) = &rbo_fbo_no_aa {
                    read_fb_binder.rebind(&fbo);
                    draw_fb_binder.rebind(fbo_no_aa);
                    gl::BlitFramebuffer(
                        0, 0, page_size_pixels.x, page_size_pixels.y,
                        0, 0, page_size_pixels.x, page_size_pixels.y,
                        gl::COLOR_BUFFER_BIT, gl::NEAREST
                    );
                    read_fb_binder.rebind(fbo_no_aa);
                    draw_fb_binder.rebind(&fbo);
                }

                gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

                {
                    let mut data = pixbuf.data()?;
                    gl::ReadPixels(0, 0, page_size_pixels.x, page_size_pixels.y, gl::BGRA, gl::UNSIGNED_BYTE, data.as_mut_ptr() as *mut _);
                }

                cr.set_source_surface(&pixbuf, 0.0, 0.0)?;
                let pat = cr.source();
                let mut mc = cairo::Matrix::identity();
                let scale = resolution / 72.0;
                mc.scale(scale as f64, scale as f64);
                pat.set_matrix(mc);

                let _ = cr.paint();
                let _ = cr.show_page();
                let _ = pixbuf.write_to_png(&mut std::fs::File::create("test.png").unwrap());
            }
            gl::PixelStorei(gl::PACK_ROW_LENGTH, 0);
            drop(cr);
            drop(pdf);
        }
        Ok(())
    }
}

struct BackupGlConfig {
    p_vao: i32,
    p_prg: i32,
    p_buf: i32,
    p_atex: i32,
    p_tex: i32,
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
            BackupGlConfig {
                p_vao, p_prg, p_buf,
                p_atex, p_tex,
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

            gl::Disable(gl::DEPTH_TEST);
            gl::Disable(gl::STENCIL_TEST);
            gl::Disable(gl::POLYGON_OFFSET_FILL);
        }
    }
}

#[derive(Debug)]
struct Canvas3dStatus {
    mouse_pos: Vector2,
    action: Canvas3dAction,
}

impl Default for Canvas3dStatus {
    fn default() -> Self {
        Self {
            mouse_pos: Vector2::zero(),
            action: Canvas3dAction::None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Canvas3dAction {
    None,
    Hovering,
    Clicked(imgui::MouseButton),
    Pressed(imgui::MouseButton),
    Released(imgui::MouseButton),
    Dragging(imgui::MouseButton),
    DoubleClicked(imgui::MouseButton),
}

fn canvas3d(ui: &imgui::Ui, st: &mut Canvas3dStatus) {
    ui.invisible_button(
        "canvas3d",
        ui.content_region_avail(),
    );
    let hovered = ui.is_item_hovered();
    let pos = Vector2::from(ui.item_rect_min());
    let scale = Vector2::from(ui.io().display_framebuffer_scale);
    let mouse_pos = scale_size(scale, Vector2::from(ui.io().mouse_pos) - pos);

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
        Canvas3dAction::Hovering | Canvas3dAction::Pressed(_) | Canvas3dAction::Clicked(_) | Canvas3dAction::DoubleClicked(_) => {
            if !hovered {
                Canvas3dAction::None
            } else if ui.is_mouse_dragging(imgui::MouseButton::Left) {
                Canvas3dAction::Dragging(imgui::MouseButton::Left)
            } else if ui.is_mouse_dragging(imgui::MouseButton::Right) {
                Canvas3dAction::Dragging(imgui::MouseButton::Right)
            } else if ui.is_mouse_double_clicked(imgui::MouseButton::Left) {
                Canvas3dAction::DoubleClicked(imgui::MouseButton::Left)
            } else if ui.is_mouse_double_clicked(imgui::MouseButton::Right) {
                Canvas3dAction::DoubleClicked(imgui::MouseButton::Right)
            } else if ui.is_mouse_clicked(imgui::MouseButton::Left) {
                Canvas3dAction::Clicked(imgui::MouseButton::Left)
            } else if ui.is_mouse_clicked(imgui::MouseButton::Right) {
                Canvas3dAction::Clicked(imgui::MouseButton::Right)
            } else if ui.is_mouse_released(imgui::MouseButton::Left) {
                Canvas3dAction::Released(imgui::MouseButton::Left)
            } else if ui.is_mouse_released(imgui::MouseButton::Right) {
                Canvas3dAction::Released(imgui::MouseButton::Right)
            } else if ui.is_mouse_down(imgui::MouseButton::Left) {
                Canvas3dAction::Pressed(imgui::MouseButton::Left)
            } else if ui.is_mouse_down(imgui::MouseButton::Right) {
                Canvas3dAction::Pressed(imgui::MouseButton::Right)
            } else {
                Canvas3dAction::Hovering
            }
        }
        Canvas3dAction::None | Canvas3dAction::Released(_) => {
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
        mouse_pos,
        action,
    };
}

fn premultiply_image(img: DynamicImage) -> image::RgbaImage {
    let mut img = img.into_rgba8();
    for p in img.pixels_mut() {
        let a = p.0[3] as u32;
        for i in &mut p.0[0..3] {
            *i = (*i as u32 * a / 255) as u8;
        }
    }
    img
}

fn load_texture_from_memory(data: &[u8], premultply: bool) -> Result<glow::NativeTexture> {
    let data = std::io::Cursor::new(data);
    let image = image::io::Reader::with_format(data, image::ImageFormat::Png)
        .decode()?;
    let image = if premultply {
        premultiply_image(image)
    } else {
        image.into_rgba8()
    };
    unsafe {
        let tex = glr::Texture::generate();
        gl::BindTexture(gl::TEXTURE_2D, tex.id());
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(gl::TEXTURE_2D, 0, gl::SRGB8_ALPHA8 as i32,
            image.width() as i32, image.height() as i32, 0,
            gl::RGBA, gl::UNSIGNED_BYTE, image.as_ptr() as _);
        gl::GenerateMipmap(gl::TEXTURE_2D);
        let ntex = glow::NativeTexture(NonZeroU32::new(tex.into_id()).unwrap());
        gl::BindTexture(gl::TEXTURE_2D, 0);
        Ok(ntex)
    }
}

fn scale_size(s: Vector2, v: Vector2) -> Vector2 {
    Vector2::new(s.x * v.x, s.y * v.y)
}

fn advance_cursor(ui: &imgui::Ui, x: f32, y: f32) { 
    let f = ui.current_font_size();
    let mut pos: [f32; 2] = ui.cursor_screen_pos();
    pos[0] += f * x;
    pos[1] += f * y;
    ui.set_cursor_screen_pos(pos);
}
fn center_text(ui: &imgui::Ui, s: &str, w: f32) {
    let ss = ui.calc_text_size(s);
    let mut pos: [f32; 2] = ui.cursor_screen_pos();
    pos[0] += (w - ss[0]) / 2.0;
    ui.set_cursor_screen_pos(pos);
    ui.text(s);
}
fn center_url(ui: &imgui::Ui, s: &str, id: &str, cmd: Option<&str>, w: f32) {
    let ss = ui.calc_text_size(s);
    let mut pos: [f32; 2] = ui.cursor_screen_pos();
    let pos0 = pos;
    pos[0] += (w - ss[0]) / 2.0;
    ui.set_cursor_screen_pos(pos);
    let _s = ui.push_style_color(imgui::StyleColor::Text, [0.0, 0.0, 1.0, 1.0]);
    ui.text(s);
    ui.set_cursor_screen_pos(pos0);
    if ui.invisible_button(id, ss) {
        let _ = opener::open_browser(cmd.unwrap_or(s));
    }
    if ui.is_item_hovered() {
        ui.set_mouse_cursor(Some(imgui::MouseCursor::Hand));
    }
}

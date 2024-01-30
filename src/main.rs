//#![allow(clippy::collapsible_if)]

use std::{time::{Instant, Duration}, path::{Path, PathBuf}, rc::Rc, cell::RefCell};
use std::io::{Read, Write};
use anyhow::{Result, anyhow, Context};
use cgmath::{
    prelude::*,
    Deg, Rad,
};
use lazy_static::lazy_static;
use image::{DynamicImage, GenericImage, GenericImageView, EncodableLayout};
use easy_imgui_window::{MainWindow, MainWindowWithRenderer,
    easy_imgui::{self as imgui, Color, MouseButton, Vector2, vec2},
    easy_imgui_renderer::{
        glr::{self, GlContext, Rgba, BinderRenderbuffer, BinderDrawFramebuffer, BinderReadFramebuffer},
        glow::{self, HasContext},
    },
    winit::{self, event_loop::EventLoopBuilder},
};

type Ui = imgui::Ui<GlobalContext>;

static MULTISAMPLES: &[i32] = &[16, 8, 4, 2];

fn to_cgv2(v: imgui::Vector2) -> cgmath::Vector2<f32> {
    cgmath::Vector2::new(v.x, v.y)
}

mod imgui_filedialog;
mod paper;
mod util_3d;
mod util_gl;

mod ui;
use ui::*;

lazy_static! {
    static ref LOGO_IMG: image::RgbaImage = load_image_from_memory(include_bytes!("papercraft.png"), true).unwrap();
    static ref ICONS_IMG: image::RgbaImage = load_image_from_memory(include_bytes!("icons.png"), true).unwrap();
    static ref KARLA_TTF: Vec<u8> = {
        let mut ttf = Vec::new();
        flate2::read::ZlibDecoder::new(include_bytes!("Karla-Regular.ttf.z").as_slice()).read_to_end(&mut ttf).unwrap();
        ttf
    };
    static ref COPYRIGHT_TTF: Vec<u8> = {
        let mut ttf = Vec::new();
        flate2::read::ZlibDecoder::new(include_bytes!("copyright.ttf.z").as_slice()).read_to_end(&mut ttf).unwrap();
        ttf
    };
}
const FONT_SIZE: f32 = 3.0;

use paper::{Papercraft, FlapStyle, FoldStyle, EdgeIdPosition, PaperOptions, IslandKey, import::import_model_file};
use util_3d::{Matrix3, Vector3};
use util_gl::{Uniforms2D, Uniforms3D, UniformQuad, MVertex2DLine};


use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// Long
struct Cli {
    #[arg(value_name = "MODEL_FILE")]
    name: Option<PathBuf>,

    #[arg(long, help = "Uses Dear ImGui light theme instead of the default dark one")]
    light: bool,

    #[arg(short, long, help = "Prevents editing of the model, useful as reference to build a real model")]
    read_only: bool,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let event_loop = EventLoopBuilder::new().build().unwrap();
    let window = MainWindow::new(&event_loop, "Papercraft").unwrap();

    let icon = winit::window::Icon::from_rgba(LOGO_IMG.as_bytes().to_owned(), LOGO_IMG.width(), LOGO_IMG.height()).unwrap();
    window.window().set_window_icon(Some(icon));
    window.window().set_ime_allowed(true);
    let mut window = MainWindowWithRenderer::new(window);

    let renderer = window.renderer();
    renderer.set_background_color(Some(Color::BLACK));
    let gl = renderer.gl_context().clone();
    let imgui = renderer.imgui();
    let mut imgui = unsafe { imgui.set_current() };
    imgui.set_allow_user_scaling(true);
    imgui.nav_enable_keyboard();
    if cli.light {
        let mut style = imgui.style();
        style.set_colors_light();
    }

    // Initialize papercraft status
    let mut data = PapercraftContext::from_papercraft(Papercraft::empty(), &gl).unwrap();
    let cmd_file_action = match cli {
        Cli { name: Some(name), read_only: false, .. }  => {
            Some((FileAction::ImportModel, name))
        }
        Cli { name: Some(name), read_only: true, .. }  => {
            // This will be rewritten when/if the file is loaded, but setting it here avoids a UI flicker
            data.ui.mode = MouseMode::ReadOnly;
            Some((FileAction::OpenCraftReadOnly, name))
        }
        _ => { None }
    };

    let last_path = if let Some((_, path)) = &cmd_file_action {
        path.parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(String::new)
    } else {
        String::new()
    };

    let gl_fixs = build_gl_fixs(&gl).unwrap();
    let ctx = Rc::new(RefCell::new(GlobalContext {
        gl,
        gl_fixs,
        font_default: imgui::FontId::default(),
        font_big: imgui::FontId::default(),
        font_small: imgui::FontId::default(),
        icons_rect: [imgui::CustomRectIndex::default(); 3],
        logo_rect: imgui::CustomRectIndex::default(),
        data,
        file_name: None,
        rebuild: RebuildFlags::all(),
        splitter_pos: 1.0,
        sz_full: vec2(2.0, 1.0),
        sz_scene: vec2(1.0, 1.0),
        sz_paper: vec2(1.0, 1.0),
        scene_ui_status: Canvas3dStatus::default(),
        paper_ui_status: Canvas3dStatus::default(),
        options_opened: None,
        about_visible: false,
        option_button_height: 0.0,
        file_dialog: None,
        file_action: None,
        last_path,
        last_export: String::new(),
        error_message: None,
        confirmable_action: None,
        popup_time_start: Instant::now(),
        cmd_file_action,
        quit_requested: BoolWithConfirm::None,
        title: String::new(),
    }));
    unsafe { install_crash_backup(event_loop.create_proxy(), &ctx); }
    event_loop.run(move |event, w| {
        match &event {
            winit::event::Event::NewEvents(winit::event::StartCause::Init) => {
                // This fixes "keyboard non-responsive on startup because it doesn't detect FocusGained...":
                // https://github.com/rust-windowing/winit/issues/1558
                // https://github.com/rust-windowing/winit/issues/1558
                //
                use easy_imgui_window::winit::raw_window_handle::{
                    HasWindowHandle, RawWindowHandle::{Xcb, Xlib},
                };
                let w = window.main_window().window();
                if let Ok(h) = w.window_handle() {
                    if matches!(h.as_raw(), Xcb(_) | Xlib(_)) {
                        w.set_visible(false);
                        w.set_visible(true);
                    }
                }
            }
            _ => {}
        }
        // Main loop, if it panics or somewhat crashes, try to save a backup
        let mut ctx = ctx.borrow_mut();

        let maybe_fatal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Default message handling
            let loop_res = window.do_event(&mut *ctx, &event, w);

            if let Some(new_title) = ctx.updated_title() {
                window.main_window().window().set_title(&new_title);
            }

            // manually handle a few messages
            if let winit::event::Event::UserEvent(()) = &event {
                //Fatal signal: it is about to be aborted, just stop whatever it is doing and
                //let the crash handler do its job.
                loop { std::thread::park(); }
            }
            if loop_res.is_break() &&
                ctx.quit_requested == BoolWithConfirm::None
            {
                let quit = ctx.check_modified();
                ctx.quit_requested = quit;
            }
            if ctx.quit_requested == BoolWithConfirm::Confirmed {
                w.exit();
            }
        }));
        if let Err(e) = maybe_fatal {
            ctx.save_backup_on_panic();
            std::panic::resume_unwind(e);
        }
    }).unwrap();
}

fn build_gl_fixs(gl: &GlContext) -> Result<GLFixedObjects> {
    let prg_scene_solid = util_gl::program_from_source(gl, include_str!("shaders/scene_solid.glsl")).with_context(|| "scene_solid")?;
    let prg_scene_line = util_gl::program_from_source(gl, include_str!("shaders/scene_line.glsl")).with_context(|| "scene_line")?;
    let prg_paper_solid = util_gl::program_from_source(gl, include_str!("shaders/paper_solid.glsl")).with_context(|| "paper_solid")?;
    let prg_paper_line = util_gl::program_from_source(gl, include_str!("shaders/paper_line.glsl")).with_context(|| "paper_line")?;
    let prg_quad = util_gl::program_from_source(gl, include_str!("shaders/quad.glsl")).with_context(|| "quad")?;

    let vao = glr::VertexArray::generate(gl)?;

    let fbo_scene = glr::Framebuffer::generate(gl)?;
    let rbo_scene_color = glr::Renderbuffer::generate(gl)?;
    let rbo_scene_depth = glr::Renderbuffer::generate(gl)?;

    unsafe {
        let fb_binder = BinderDrawFramebuffer::bind(&fbo_scene);

        let rb_binder = BinderRenderbuffer::bind(&rbo_scene_color);
        gl.renderbuffer_storage(rb_binder.target(), glow::RGBA8, 1, 1);
        gl.framebuffer_renderbuffer(fb_binder.target(), glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(rbo_scene_color.id()));

        rb_binder.rebind(&rbo_scene_depth);
        gl.renderbuffer_storage(rb_binder.target(), glow::DEPTH_COMPONENT, 1, 1);
        gl.framebuffer_renderbuffer(fb_binder.target(), glow::DEPTH_ATTACHMENT, glow::RENDERBUFFER, Some(rbo_scene_depth.id()));
    }

    let fbo_paper = glr::Framebuffer::generate(gl)?;
    let rbo_paper_color = glr::Renderbuffer::generate(gl)?;
    let rbo_paper_stencil = glr::Renderbuffer::generate(gl)?;

    unsafe {
        let fb_binder = BinderDrawFramebuffer::bind(&fbo_paper);

        let rb_binder = BinderRenderbuffer::bind(&rbo_paper_color);
        gl.renderbuffer_storage(rb_binder.target(), glow::RGBA8, 1,1);
        gl.framebuffer_renderbuffer(fb_binder.target(), glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(rbo_paper_color.id()));

        rb_binder.rebind(&rbo_paper_stencil);
        gl.renderbuffer_storage(rb_binder.target(), glow::STENCIL_INDEX, 1, 1);
        gl.framebuffer_renderbuffer(fb_binder.target(), glow::STENCIL_ATTACHMENT, glow::RENDERBUFFER, Some(rbo_paper_stencil.id()));
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
    OpenCraftReadOnly,
    SaveAsCraft,
    ImportModel,
    UpdateObj,
    ExportObj,
    GeneratePrintable,
}

impl FileAction {
    fn title(&self) -> &'static str {
        match self {
            FileAction::OpenCraft |
            FileAction::OpenCraftReadOnly => "Opening...",
            FileAction::SaveAsCraft => "Saving...",
            FileAction::ImportModel => "Importing...",
            FileAction::UpdateObj => "Updating...",
            FileAction::ExportObj => "Exporting...",
            FileAction::GeneratePrintable => "Generating...",
        }
    }
}

struct ConfirmableAction {
    title: String,
    message: String,
    action: Box<dyn Fn(&mut MenuActions)>,
}

struct GlobalContext {
    gl: GlContext,
    gl_fixs: GLFixedObjects,
    font_default: imgui::FontId,
    font_big: imgui::FontId,
    font_small: imgui::FontId,
    icons_rect: [imgui::CustomRectIndex; 3],
    logo_rect: imgui::CustomRectIndex,
    data: PapercraftContext,
    file_name: Option<PathBuf>,
    rebuild: RebuildFlags,
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
    last_path: String,
    last_export: String,
    error_message: Option<String>,
    confirmable_action: Option<ConfirmableAction>,
    popup_time_start: Instant,
    cmd_file_action: Option<(FileAction, PathBuf)>,
    quit_requested: BoolWithConfirm,
    title: String,
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
    import_model: BoolWithConfirm,
    update_model: BoolWithConfirm,
    export_obj: bool,
    generate_printable: bool,
    quit: BoolWithConfirm,
    reset_views: bool,
    undo: bool,
}

impl GlobalContext {
    fn modifiable(&self) -> bool {
        self.data.ui.mode != MouseMode::ReadOnly
    }
    fn build_modal_error_message(&mut self, ui: &Ui) {
        ui.popup_modal_config("Error")
            .close_button(true)
            .flags(imgui::WindowFlags::NoResize | imgui::WindowFlags::AlwaysAutoResize)
            .with(|| {
                let font_sz = ui.get_font_size();

                ui.text(self.error_message.as_deref().unwrap_or_default());

                ui.separator();

                if ui.button_config("OK").size(vec2(font_sz * 5.5, 0.0)).build()
                    || ui.is_key_pressed(imgui::Key::Enter)
                        || ui.is_key_pressed(imgui::Key::KeypadEnter)
                        {
                            if !ui.is_window_appearing() {
                                ui.close_current_popup();
                                self.error_message = None;
                            }
                        }
            });
    }
    fn build_confirm_message(&mut self, ui: &Ui, menu_actions: &mut MenuActions) {
        let mut closed = None;
        if let Some(action) = self.confirmable_action.take() {
            let font_sz = ui.get_font_size();
            ui.popup_modal_config(format!("{}###Confirm", action.title))
                .close_button(true)
                .flags(imgui::WindowFlags::NoResize | imgui::WindowFlags::AlwaysAutoResize)
                .with(|| {
                    ui.text(&action.message);

                    ui.separator();

                    if ui.button_config("Cancel").size(vec2(font_sz * 5.5, 0.0)).build() {
                        if !ui.is_window_appearing() {
                            ui.close_current_popup();
                            closed = Some(false);
                        }
                    }
                    ui.same_line();
                    if ui.button_config("Continue").size(vec2(font_sz * 5.5, 0.0)).build() {
                        if !ui.is_window_appearing() {
                            ui.close_current_popup();
                            closed = Some(true);
                        }
                    }
                });
            if let Some(cont) = closed {
                if cont {
                    (action.action)(menu_actions);
                }
            } else {
                self.confirmable_action = Some(action);
            }
        }
    }

    fn build_about(&mut self, ui: &Ui) {
        if !self.about_visible {
            return;
        }
        ui.window_config("About...###about")
            .open(&mut self.about_visible)
            .flags(imgui::WindowFlags::NoResize | imgui::WindowFlags::AlwaysAutoResize)
            .with(|| {
                let sz_full = ui.get_content_region_avail();
                let f = ui.get_font_size();
                let logo_height = f * 8.0;
                let logo_rect = ui.font_atlas().get_custom_rect(self.logo_rect);
                let logo_scale = logo_height / logo_rect.Height as f32;
                let logo_width = logo_rect.Width as f32 * logo_scale;
                advance_cursor_pixels(ui, (sz_full.x - logo_width) / 2.0, 0.0);
                //ui.do_image(self.logo_tex, [logo_width, logo_height])
                ui.image_with_custom_rect_config(self.logo_rect, logo_scale)
                    .build();
                ui.with_push(self.font_big,
                    || {
                         center_text(ui, "Papercraft", sz_full.x);
                     });

                advance_cursor(ui, 0.0, 1.0);
                center_text(ui, &format!("Version {}", env!("CARGO_PKG_VERSION")), sz_full.x);
                advance_cursor(ui, 0.0, 1.0);
                center_text(ui, env!("CARGO_PKG_DESCRIPTION"), sz_full.x);
                advance_cursor(ui, 0.0, 0.5);
                center_url(ui, env!("CARGO_PKG_REPOSITORY"), "url", None, sz_full.x);
                advance_cursor(ui, 0.0, 0.5);
                ui.with_push(self.font_small,
                    || {
                         center_text(ui, "© Copyright 2022 - Rodrigo Rivas Costa", sz_full.x);
                         center_text(ui, "This program comes with absolutely no warranty.", sz_full.x);
                         center_url(
                             ui,
                             "See the GNU General Public License, version 3 or later for details.", "gpl3",
                             Some("https://www.gnu.org/licenses/gpl-3.0.html"),
                             sz_full.x
                             );
                     });
                //TODO: list third party SW
            });
    }
    // Returns true if the action has just been done successfully
    fn build_modal_wait_message_and_run_file_action(&mut self, ui: &Ui) -> bool {
        let mut ok = false;
        if let Some(file_action) = self.file_action.take() {
            let (action, file) = &file_action;
            let title = action.title();
            let mut res = None;
            // Build the modal itself
            ui.set_next_window_size(vec2(150.0, 0.0), imgui::Cond::Once);
            ui.popup_modal_config(format!("{title}###Wait"))
                .flags(imgui::WindowFlags::NoResize)
                .with(|| {
                    ui.text("Please, wait...");

                    // Give time to the fading modal, should be enough
                    let run = self.popup_time_start.elapsed() > Duration::from_millis(250);
                    if run {
                        res = Some(self.run_file_action(*action, file));
                        ui.close_current_popup();
                    }
                });

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

    fn build_ui(&mut self, ui: &Ui) -> MenuActions {
        let mut menu_actions = self.build_menu_and_file_dialog(ui);
        let font_sz = ui.get_font_size();

        // Toolbar is not needed in read-only mode
        if self.modifiable() {
            let pad: f32 = font_sz / 4.0;

            // Toolbar
            ui.with_push(
                (
                    (imgui::StyleVar::WindowPadding, imgui::StyleValue::Vec2(vec2(pad, pad))),
                    (imgui::StyleVar::ItemSpacing, imgui::StyleValue::Vec2(vec2(0.0, 0.0))),
                ),
                || {
                    ui.child_config("toolbar")
                        .child_flags(imgui::ChildFlags::AlwaysUseWindowPadding | imgui::ChildFlags::AutoResizeY)
                        .window_flags(imgui::WindowFlags::NoScrollbar)
                        .with(|| {
                            ui.with_push((imgui::StyleVar::ItemSpacing, imgui::StyleValue::Vec2(vec2(font_sz / 8.0, 0.0))),
                                || {
                                    let color_active = ui.style().color(imgui::ColorId::ButtonActive);
                                    if ui.image_button_with_custom_rect_config("Face", self.icons_rect[0], 1.0)
                                        .bg_col(if self.data.ui.mode == MouseMode::Face { color_active } else { imgui::Color::TRANSPARENT })
                                        .build()
                                    {
                                        self.set_mouse_mode(MouseMode::Face);
                                    }
                                    ui.same_line();
                                    if ui.image_button_with_custom_rect_config("Edge", self.icons_rect[1], 1.0)
                                        .bg_col(if self.data.ui.mode == MouseMode::Edge { color_active } else { imgui::Color::TRANSPARENT })
                                        .build()
                                    {
                                        self.set_mouse_mode(MouseMode::Edge);
                                    }
                                    ui.same_line();
                                    if ui.image_button_with_custom_rect_config("Tab", self.icons_rect[2], 1.0)
                                        .bg_col(if self.data.ui.mode == MouseMode::Flap { color_active } else { imgui::Color::TRANSPARENT })
                                        .build()
                                    {
                                        self.set_mouse_mode(MouseMode::Flap);
                                    }
                                });
                        }
                    );
                }
            );
        }

        let color_hovered = ui.style().color(imgui::ColorId::ButtonHovered);
        ui.with_push(
            (
                (imgui::StyleVar::ItemSpacing, imgui::StyleValue::Vec2(vec2(2.0, 2.0))),
                (imgui::StyleVar::WindowPadding, imgui::StyleValue::Vec2(vec2(0.0, 0.0))),
                (imgui::ColorId::ButtonActive, color_hovered),
                (imgui::ColorId::Button, color_hovered),
            ),
            || {
                let sz = vec2(0.0, -ui.get_frame_height());
                ui.child_config("main_area")
                    .size(sz)
                    .with(|| {
                        let sz_full = ui.get_content_region_avail();
                        if self.sz_full != sz_full {
                            if self.sz_full.x > 1.0 {
                                self.splitter_pos = self.splitter_pos * sz_full.x / self.sz_full.x;
                            }
                            self.sz_full = sz_full;
                        }

                        let scale = ui.display_scale();

                        self.build_scene(ui, self.splitter_pos);
                        let sz_scene = scale * ui.get_item_rect_size();

                        ui.same_line();

                        ui.button_config("##vsplitter")
                            .size(vec2(font_sz / 2.0, -1.0))
                            .build();
                        if ui.is_item_active() {
                            self.splitter_pos += ui.io().MouseDelta.x;
                        }
                        self.splitter_pos = self.splitter_pos.clamp(50.0, (sz_full.x - 50.0).max(50.0));
                        if ui.is_item_hovered() || ui.is_item_active() {
                            ui.set_mouse_cursor(imgui::MouseCursor::ResizeEW);
                        }

                        ui.same_line();

                        self.build_paper(ui);
                        let sz_paper = scale * Vector2::from(ui.get_item_rect_size());

                        // Resize FBOs
                        if sz_scene != self.sz_scene && sz_scene.x > 1.0 && sz_scene.y > 1.0 {
                            self.add_rebuild(RebuildFlags::SCENE_REDRAW);
                            self.sz_scene = sz_scene;

                            self.data.ui.trans_scene.persp = cgmath::perspective(Deg(60.0), sz_scene.x / sz_scene.y, 1.0, 100.0);
                            self.data.ui.trans_scene.persp_inv = self.data.ui.trans_scene.persp.invert().unwrap();

                            let (x, y) = (sz_scene.x as i32, sz_scene.y as i32);

                            unsafe {
                                let rb_binder = BinderRenderbuffer::bind(&self.gl_fixs.rbo_scene_color);

                                'no_aa: {
                                    for samples in MULTISAMPLES {
                                        self.gl.get_error(); //clear error
                                        rb_binder.rebind(&self.gl_fixs.rbo_scene_color);
                                        self.gl.renderbuffer_storage_multisample(rb_binder.target(), *samples, glow::RGBA8, x, y);
                                        rb_binder.rebind(&self.gl_fixs.rbo_scene_depth);
                                        self.gl.renderbuffer_storage_multisample(rb_binder.target(), *samples, glow::DEPTH_COMPONENT, x, y);

                                        if self.gl.get_error() != 0 || self.gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                                            continue;
                                        }
                                        break 'no_aa;
                                    }

                                    rb_binder.rebind(&self.gl_fixs.rbo_scene_color);
                                    self.gl.renderbuffer_storage(rb_binder.target(), glow::RGBA8, x, y);
                                    rb_binder.rebind(&self.gl_fixs.rbo_scene_depth);
                                    self.gl.renderbuffer_storage(rb_binder.target(), glow::DEPTH_COMPONENT, x, y);
                                }
                            }
                        }

                        if sz_paper != self.sz_paper && sz_paper.x > 1.0 && sz_paper.y > 1.0 {
                            self.add_rebuild(RebuildFlags::PAPER_REDRAW);
                            self.sz_paper = sz_paper;

                            let (x, y) = (sz_paper.x as i32, sz_paper.y as i32);
                            self.data.ui.trans_paper.ortho = util_3d::ortho2d(sz_paper.x, sz_paper.y);

                            unsafe {
                                let rb_binder = BinderRenderbuffer::bind(&self.gl_fixs.rbo_paper_color);

                                'no_aa: {
                                    for samples in MULTISAMPLES {
                                        self.gl.get_error(); //clear error
                                        rb_binder.rebind(&self.gl_fixs.rbo_paper_color);
                                        self.gl.renderbuffer_storage_multisample(rb_binder.target(), *samples, glow::RGBA8, x, y);
                                        rb_binder.rebind(&self.gl_fixs.rbo_paper_stencil);
                                        self.gl.renderbuffer_storage_multisample(rb_binder.target(), *samples, glow::STENCIL_INDEX, x, y);

                                        if self.gl.get_error() != 0 || self.gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                                            continue;
                                        }
                                        break 'no_aa;
                                    }

                                    rb_binder.rebind(&self.gl_fixs.rbo_paper_color);
                                    self.gl.renderbuffer_storage(rb_binder.target(), glow::RGBA8, x, y);
                                    rb_binder.rebind(&self.gl_fixs.rbo_paper_stencil);
                                    self.gl.renderbuffer_storage(rb_binder.target(), glow::STENCIL_INDEX, x, y);
                                }
                            }
                        }

                    });
            }
        );
        advance_cursor(ui, 0.25, 0.0);

        let status_text = match self.data.ui.mode {
            MouseMode::Face => "Face mode. Click to select a piece. Drag on paper to move it. Shift-drag on paper to rotate it.",
            MouseMode::Edge => "Edge mode. Click on an edge to split/join pieces. Shift-click to join a full strip of quads.",
            MouseMode::Flap => "Flap mode. Click on an edge to swap the side of a flap. Shift-click to hide a flap.",
            MouseMode::ReadOnly => "View mode. Click to highlight a piece. Move the mouse over an edge to highlight the matching pair.",
        };
        ui.text(status_text);

        self.build_options_dialog(ui);
        self.build_modal_error_message(ui);
        self.build_modal_wait_message_and_run_file_action(ui);
        self.build_confirm_message(ui, &mut menu_actions);
        self.build_about(ui);

        menu_actions
    }

    fn build_options_dialog(&mut self, ui: &Ui) {
        let options = match self.options_opened.take() {
            Some(o) => o,
            None => return,
        };
        let modifiable = self.modifiable();
        let mut options_opened = true;
        ui.set_next_window_size(if modifiable {vec2(650.0, 400.0)} else {vec2(300.0, 100.0)}, imgui::Cond::Once);
        ui.window_config("Document properties###options")
            .open(&mut options_opened)
            .flags(imgui::WindowFlags::NoScrollbar)
            .with_always(|opened| {
                if opened {
                    if modifiable {
                        let (keep_opened, apply) = self.build_full_options_inner_dialog(ui, options);
                        self.options_opened = keep_opened;
                        if let Some(apply_options) = apply {
                            self.data.set_papercraft_options(apply_options);
                            self.add_rebuild(RebuildFlags::all());
                        }
                    } else {
                        self.build_read_only_options_inner_dialog(ui, &options);
                        self.options_opened = Some(options);
                    }
                } else {
                    self.options_opened = Some(options);
                }
            });

        // If the window was closed with the X
        if !options_opened {
            self.options_opened = None;
        }
    }

    fn build_read_only_options_inner_dialog(&self, ui: &Ui, options: &PaperOptions) {
        let n_pieces = self.data.papercraft().num_islands();
        let n_flaps = self.data.papercraft().model().edges()
            .filter(|(e, _)| matches!(self.data.papercraft().edge_status(*e), paper::EdgeStatus::Cut(_)))
            .count();
        let bbox = util_3d::bounding_box_3d(
            self.data.papercraft().model()
            .vertices()
            .map(|(_, v)| v.pos())
            );
        let model_size = (bbox.1 - bbox.0) * options.scale;
        let Vector3 { x, y, z } = model_size;
        ui.text(&format!("Number of pieces: {n_pieces}\nNumber of flaps: {n_flaps}\nReal size (mm): {x:.0} x {y:.0} x {z:.0}"));
    }

    fn build_full_options_inner_dialog(&mut self, ui: &Ui, mut options: PaperOptions) -> (Option<PaperOptions>, Option<PaperOptions>) {
        let size = Vector2::from(ui.get_content_region_avail());
        let font_sz = ui.get_font_size();
        ui.child_config("options")
            .size(vec2(size.x, -self.option_button_height))
            .window_flags(imgui::WindowFlags::HorizontalScrollbar)
            .with(|| {
                ui.tree_node_config("Model").flags(imgui::TreeNodeFlags::Framed).with(|| {
                    ui.set_next_item_width(font_sz * 5.5);
                    ui.input_float_config("Scale", &mut options.scale)
                        .display_format(imgui::FloatFormat::G)
                        .build();
                    options.scale = options.scale.max(0.0);
                    ui.same_line_ex(0.0, font_sz * 3.0);
                    ui.with_disabled(!self.data.papercraft().model().has_textures(), || {
                        ui.checkbox("Textured", &mut options.texture);
                        ui.same_line_ex(0.0, font_sz * 3.0);
                        ui.checkbox("Texture filter", &mut options.tex_filter);
                    });

                    ui.tree_node_config("Flaps").with(|| {
                        static FLAP_STYLES: &[FlapStyle] = &[
                            FlapStyle::Textured,
                            FlapStyle::HalfTextured,
                            FlapStyle::White,
                            FlapStyle::None,
                        ];
                        fn fmt_flap_style(s: FlapStyle) -> &'static str {
                            match s {
                                FlapStyle::Textured => "Textured",
                                FlapStyle::HalfTextured => "Half textured",
                                FlapStyle::White => "White",
                                FlapStyle::None => "None",
                            }
                        }

                        ui.set_next_item_width(font_sz * 8.0);
                        ui.combo("Style", FLAP_STYLES.iter().copied(), fmt_flap_style, &mut options.flap_style);

                        ui.same_line_ex(font_sz * 12.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 8.0);
                        ui.slider_float_config("Shadow", &mut options.shadow_flap_alpha)
                            .range(0.0, 1.0)
                            .display_format(imgui::FloatFormat::F(2))
                            .build();

                        ui.set_next_item_width(font_sz * 8.0);
                        ui.input_float_config("Width", &mut options.flap_width)
                            .display_format(imgui::FloatFormat::G)
                            .build();
                        options.flap_width = options.flap_width.max(0.0);

                        ui.same_line_ex(font_sz * 12.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 8.0);
                        ui.input_float_config("Angle", &mut options.flap_angle)
                            .display_format(imgui::FloatFormat::G)
                            .build();
                        options.flap_angle = options.flap_angle.clamp(0.0, 180.0);
                    });
                    ui.tree_node_config("Folds").with(|| {
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

                        ui.set_next_item_width(font_sz * 8.0);
                        ui.combo("Style", FOLD_STYLES.iter().copied(), fmt_fold_style, &mut options.fold_style);

                        ui.same_line_ex(0.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 5.5);
                        ui.with_disabled(matches!(options.fold_style, FoldStyle::None | FoldStyle::Full), || {
                            ui.input_float_config("Length", &mut options.fold_line_len)
                                .display_format(imgui::FloatFormat::G)
                                .build();
                            options.fold_line_len = options.fold_line_len.max(0.0);
                        });
                        ui.same_line_ex(0.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 5.5);
                        ui.with_disabled(matches!(options.fold_style, FoldStyle::None), || {
                            ui.input_float_config("Line width", &mut options.fold_line_width)
                                .display_format(imgui::FloatFormat::G)
                                .build();
                            options.fold_line_width = options.fold_line_width.max(0.0);
                        });

                        ui.set_next_item_width(font_sz * 5.5);
                        ui.input_float_config("Hidden fold angle", &mut options.hidden_line_angle)
                            .display_format(imgui::FloatFormat::G)
                            .build();
                        options.hidden_line_angle = options.hidden_line_angle.clamp(0.0, 180.0);
                    });
                    ui.tree_node_config("Information").with(|| {
                        self.build_read_only_options_inner_dialog(ui, &options);
                    });
                });
                ui.tree_node_config("Page layout").flags(imgui::TreeNodeFlags::Framed).with(|| {
                    ui.set_next_item_width(font_sz * 5.5);

                    let mut i = options.pages as _;
                    ui.input_int_config("Pages", &mut i).build();
                    options.pages = i.clamp(1, 1000) as _;

                    ui.same_line_ex(0.0, font_sz * 1.5);
                    ui.set_next_item_width(font_sz * 5.5);

                    let mut i = options.page_cols as _;
                    ui.input_int_config("Columns", &mut i).build();
                    options.page_cols = i.clamp(1, options.pages as _) as _;

                    ui.set_next_item_width(font_sz * 11.0);
                    ui.checkbox("Print Papercraft signature", &mut options.show_self_promotion);

                    ui.same_line_ex(0.0, font_sz * 3.0);
                    ui.set_next_item_width(font_sz * 11.0);
                    ui.checkbox("Print page number", &mut options.show_page_number);

                    static EDGE_ID_POSITIONS: &[EdgeIdPosition] = &[
                        EdgeIdPosition::None,
                        EdgeIdPosition::Outside,
                        EdgeIdPosition::Inside,
                    ];
                    fn fmt_edge_id_position(s: EdgeIdPosition) -> &'static str {
                        match s {
                            EdgeIdPosition::None => "None",
                            EdgeIdPosition::Outside => "Outside",
                            EdgeIdPosition::Inside => "Inside",
                        }
                    }
                    ui.set_next_item_width(font_sz * 6.0);
                    ui.combo("Edge id position", EDGE_ID_POSITIONS.iter().copied(), fmt_edge_id_position, &mut options.edge_id_position);

                    ui.same_line_ex(0.0, font_sz * 1.5);

                    ui.set_next_item_width(font_sz * 3.0);
                    ui.with_disabled(options.edge_id_position == EdgeIdPosition::None, || {
                        ui.input_float_config("Edge id font size (pt)", &mut options.edge_id_font_size)
                            .display_format(imgui::FloatFormat::G)
                            .build();
                        options.edge_id_font_size = options.edge_id_font_size.clamp(1.0, 72.0);
                    });
                });
                ui.tree_node_config("Paper size").flags(imgui::TreeNodeFlags::Framed).with(|| {
                    ui.set_next_item_width(font_sz * 5.5);
                    ui.input_float_config("Width", &mut options.page_size.0)
                        .display_format(imgui::FloatFormat::G)
                        .build();
                    options.page_size.0 = options.page_size.0.max(1.0);
                    ui.same_line_ex(0.0, font_sz * 1.5);
                    ui.set_next_item_width(font_sz * 5.5);
                    ui.input_float_config("Height", &mut options.page_size.1)
                        .display_format(imgui::FloatFormat::G)
                        .build();
                    options.page_size.1 = options.page_size.1.max(1.0);
                    ui.same_line_ex(0.0, font_sz * 1.5);
                    ui.set_next_item_width(font_sz * 5.5);
                    let mut resolution = options.resolution as f32;
                    ui.input_float_config("DPI", &mut resolution)
                        .display_format(imgui::FloatFormat::G)
                        .build();
                    options.resolution = (resolution as u32).max(1);

                    struct PaperSize {
                        name: &'static str,
                        size: Vector2,
                    }

                    static PAPER_SIZES: &[PaperSize] = &[
                        PaperSize {
                            name: "A4",
                            size: vec2(210.0, 297.0),
                        },
                        PaperSize {
                            name: "A3",
                            size: vec2(297.0, 420.0),
                        },
                        PaperSize {
                            name: "Letter",
                            size: vec2(215.9, 279.4),
                        },
                        PaperSize {
                            name: "Legal",
                            size: vec2(215.9, 355.6),
                        },
                    ];

                    let portrait = options.page_size.1 >= options.page_size.0;
                    let paper_size = vec2(options.page_size.0, options.page_size.1);
                    let paper_size = PAPER_SIZES.iter().find(|s| s.size == paper_size || s.size == vec2(paper_size.y, paper_size.x));
                    ui.set_next_item_width(font_sz * 8.0);
                    ui.combo_config("##Paper")
                        .preview_value_opt(paper_size.map(|p| p.name))
                        .with(|| {
                            for op in PAPER_SIZES {
                                if ui.selectable_config(op.name)
                                    .selected(paper_size.map(|p| std::ptr::eq(p, op)).unwrap_or(false))
                                    .build()
                                {
                                    options.page_size = (op.size.x, op.size.y);
                                    if !portrait {
                                        std::mem::swap(&mut options.page_size.0, &mut options.page_size.1);
                                    }
                                }
                            }
                        });
                    let mut new_portrait = portrait;
                    if ui.radio_button_config("Portrait", portrait).build() {
                        new_portrait = true;
                    }
                    if ui.radio_button_config("Landscape", !portrait).build() { new_portrait = false;
                    }
                    if portrait != new_portrait {
                        std::mem::swap(&mut options.page_size.0, &mut options.page_size.1);
                    }
                });
                ui.tree_node_config("Margins").flags(imgui::TreeNodeFlags::Framed).with(|| {
                    ui.set_next_item_width(font_sz * 4.0);
                    ui.input_float_config("Top", &mut options.margin.0)
                        .display_format(imgui::FloatFormat::G)
                        .build();
                    ui.same_line_ex(0.0, font_sz * 1.5);
                    ui.set_next_item_width(font_sz * 4.0);
                    ui.input_float_config("Left", &mut options.margin.1)
                        .display_format(imgui::FloatFormat::G)
                        .build();
                    ui.same_line_ex(0.0, font_sz * 1.5);
                    ui.set_next_item_width(font_sz * 4.0);
                    ui.input_float_config("Right", &mut options.margin.2)
                        .display_format(imgui::FloatFormat::G)
                        .build();
                    ui.same_line_ex(0.0, font_sz * 1.5);
                    ui.set_next_item_width(font_sz * 4.0);
                    ui.input_float_config("Bottom", &mut options.margin.3)
                        .display_format(imgui::FloatFormat::G)
                        .build();
                });
            });

        let mut options_opened = Some(options);
        let mut apply_options = None;

        let pos1 = Vector2::from(ui.get_cursor_screen_pos());
        ui.separator();
        if ui.button_config("OK").size(vec2(100.0, 0.0)).build() {
            apply_options = options_opened.take();
        }
        ui.same_line();
        if ui.button_config("Cancel").size(vec2(100.0, 0.0)).build() {
            options_opened = None;
        }
        ui.same_line();
        if ui.button_config("Apply").size(vec2(100.0, 0.0)).build() {
            apply_options = options_opened.clone();
        }
        // Compute the height of the buttons to avoid having an external scrollbar
        let pos2 = Vector2::from(ui.get_cursor_screen_pos());
        self.option_button_height = pos2.y - pos1.y;

        (options_opened, apply_options)
    }

    fn check_modified(&self) -> BoolWithConfirm {
        if self.data.modified {
            BoolWithConfirm::Requested
        } else {
            BoolWithConfirm::Confirmed
        }
    }
    fn build_menu_and_file_dialog(&mut self, ui: &Ui) -> MenuActions {
        let mut menu_actions = MenuActions::default();

        ui.with_menu_bar(|| {
            ui.menu_config("File").with(|| {
                if ui.menu_item_config("Open...")
                    .shortcut("Ctrl+O")
                    .build()
                {
                    menu_actions.open = self.check_modified();
                }
                ui.with_disabled(self.data.papercraft().model().is_empty(), || {
                    if ui.menu_item_config("Save")
                        .shortcut("Ctrl+S")
                        .build()
                    {
                        menu_actions.save = true;
                    }
                    if ui.menu_item_config("Save as...").build() {
                        menu_actions.save_as = true;
                    }
                });
                if self.modifiable() {
                    if ui.menu_item_config("Import model...").build() {
                        menu_actions.import_model = self.check_modified();
                    }
                    if ui.menu_item_config("Update with new model...").build() {
                        menu_actions.update_model = self.check_modified();
                    }
                }
                if ui.menu_item_config("Export OBJ...").build() {
                    menu_actions.export_obj = true;
                }
                if ui.menu_item_config("Generate Printable...").build() {
                    menu_actions.generate_printable = true;
                }
                ui.separator();
                if ui.menu_item_config("Quit")
                    .shortcut("Ctrl+Q")
                    .build()
                {
                    menu_actions.quit = self.check_modified();
                }
            });
            ui.menu_config("Edit").with(|| {
                if self.modifiable() {
                    if ui.menu_item_config("Undo")
                        .shortcut("Ctrl+Z")
                        .enabled(self.data.can_undo())
                        .build()
                    {
                        menu_actions.undo = true;
                    }
                    ui.separator();
                }

                if ui.menu_item_config("Document properties")
                    .selected(self.options_opened.is_some())
                    .build()
                {
                    self.options_opened = match self.options_opened {
                        Some(_) => None,
                        None => Some(self.data.papercraft().options().clone()),
                    }
                }

                if self.modifiable() {
                    ui.separator();

                    if ui.menu_item_config("Face/Island")
                        .shortcut("F5")
                        .selected(self.data.ui.mode == MouseMode::Face)
                        .build()
                    {
                        self.set_mouse_mode(MouseMode::Face);
                    }
                    if ui.menu_item_config("Split/Join edge")
                        .shortcut("F6")
                        .selected(self.data.ui.mode == MouseMode::Edge)
                        .build()
                    {
                        self.set_mouse_mode(MouseMode::Edge);
                    }
                    if ui.menu_item_config("Flaps")
                        .shortcut("F7")
                        .selected(self.data.ui.mode == MouseMode::Flap)
                        .build()
                    {
                        self.set_mouse_mode(MouseMode::Flap);
                    }

                    ui.separator();

                    if ui.menu_item_config("Repack pieces").build() {
                        let undo = self.data.pack_islands();
                        self.data.push_undo_action(undo);
                        self.add_rebuild(RebuildFlags::PAPER | RebuildFlags::SELECTION);
                    }
                }
            });
            ui.menu_config("View").with(|| {
                if ui.menu_item_config("Textures")
                    .shortcut("T")
                    .enabled(self.data.papercraft().options().texture)
                    .selected(self.data.ui.show_textures)
                    .build()
                {
                    self.data.ui.show_textures ^= true;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
                }
                if ui.menu_item_config("3D lines")
                    .shortcut("D")
                    .selected(self.data.ui.show_3d_lines)
                    .build()
                {
                    self.data.ui.show_3d_lines ^= true;
                    self.add_rebuild(RebuildFlags::SCENE_REDRAW);
                }
                if ui.menu_item_config("Flaps")
                    .shortcut("B")
                    .selected(self.data.ui.show_flaps)
                    .build()
                {
                    self.data.ui.show_flaps ^= true;
                    self.add_rebuild(RebuildFlags::PAPER);
                }
                if ui.menu_item_config("X-ray selection")
                    .shortcut("X")
                    .selected(self.data.ui.xray_selection)
                    .build()
                {
                    self.data.ui.xray_selection ^= true;
                    self.add_rebuild(RebuildFlags::SELECTION);
                }
                if ui.menu_item_config("Highlight overlaps")
                    .shortcut("H")
                    .selected(self.data.ui.highlight_overlaps)
                    .build()
                {
                    self.data.ui.highlight_overlaps ^= true;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW);
                }
                if ui.menu_item_config("Reset views").build() {
                    menu_actions.reset_views = true;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
                }
            });
            ui.menu_config("Help").with(|| {
                if ui.menu_item_config("About...")
                    .selected(self.about_visible)
                    .build()
                {
                    self.about_visible ^= true;
                }
            });
        });
        if !ui.is_popup_open_ex(None, imgui::PopupFlags::AnyPopup) {
            if self.modifiable() {
                if ui.is_key_pressed(imgui::Key::F5) {
                    self.set_mouse_mode(MouseMode::Face);
                }
                if ui.is_key_pressed(imgui::Key::F6) {
                    self.set_mouse_mode(MouseMode::Edge);
                }
                if ui.is_key_pressed(imgui::Key::F7) {
                    self.set_mouse_mode(MouseMode::Flap);
                }
                if ui.is_key_down(imgui::Key::ModCtrl) && ui.is_key_pressed(imgui::Key::Z) {

                    menu_actions.undo = true;
                }
            }
            if ui.is_key_down(imgui::Key::ModCtrl) && ui.is_key_pressed(imgui::Key::Q) {
                menu_actions.quit = self.check_modified();
            }
            if ui.is_key_down(imgui::Key::ModCtrl) && ui.is_key_pressed(imgui::Key::O) {
                menu_actions.open = self.check_modified();
            }
            if ui.is_key_down(imgui::Key::ModCtrl) && ui.is_key_pressed(imgui::Key::S) {
                menu_actions.save = true;
            }
            if ui.is_key_pressed(imgui::Key::X) {
                self.data.ui.xray_selection = !self.data.ui.xray_selection;
                self.add_rebuild(RebuildFlags::SELECTION);
            }
            if ui.is_key_pressed(imgui::Key::H) {
                self.data.ui.highlight_overlaps = !self.data.ui.highlight_overlaps;
                self.add_rebuild(RebuildFlags::PAPER_REDRAW);
            }
            if ui.is_key_pressed(imgui::Key::T) && self.data.papercraft().options().texture {
                self.data.ui.show_textures = !self.data.ui.show_textures;
                self.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
            }
            if ui.is_key_pressed(imgui::Key::D) {
                self.data.ui.show_3d_lines = !self.data.ui.show_3d_lines;
                self.add_rebuild(RebuildFlags::SCENE_REDRAW);
            }
            if ui.is_key_pressed(imgui::Key::B) {
                self.data.ui.show_flaps = !self.data.ui.show_flaps;
                self.add_rebuild(RebuildFlags::PAPER);
            }
        }
        menu_actions
    }

    fn build_scene(&mut self, ui: &Ui, width: f32) {
        let w = ui.child_config("scene")
            .size(vec2(width, 0.0))
            .child_flags(imgui::ChildFlags::Border)
            .with(|| {
                let scale = ui.display_scale();
                let pos = scale * ui.get_cursor_screen_pos();
                let dsp_size = scale * ui.display_size();

                canvas3d(ui, &mut self.scene_ui_status);

                ui.window_draw_list().add_callback({
                    move |this| {
                        unsafe {
                            // blit the FBO to the real FB
                            let x = pos.x as i32;
                            let y = (dsp_size.y - pos.y) as i32;
                            let width = this.sz_scene.x as i32;
                            let height = this.sz_scene.y as i32;

                            let _read_fb_binder = BinderReadFramebuffer::bind(&this.gl_fixs.fbo_scene);
                            this.gl.blit_framebuffer(
                                0, 0, width, height,
                                x, y - height, x + width, y,
                                glow::COLOR_BUFFER_BIT, glow::NEAREST
                                );
                        }
                    }
                });
            });
        if w.is_none() {
            self.scene_ui_status = Canvas3dStatus::default();
        }
    }

    fn build_paper(&mut self, ui: &Ui) {
        let r = ui.child_config("paper")
            .child_flags(imgui::ChildFlags::Border)
            .with(|| {
                let scale = ui.display_scale();
                let pos = scale * ui.get_cursor_screen_pos();
                let dsp_size = scale * ui.display_size();

                canvas3d(ui, &mut self.paper_ui_status);

                ui.window_draw_list().add_callback({
                    move |this| {
                        unsafe {
                            // blit the FBO to the real FB
                            let x = pos.x as i32;
                            let y = (dsp_size.y - pos.y) as i32;
                            let width = this.sz_paper.x as i32;
                            let height = this.sz_paper.y as i32;

                            let _read_fb_binder = BinderReadFramebuffer::bind(&this.gl_fixs.fbo_paper);
                            this.gl.blit_framebuffer(
                                0, 0, width, height,
                                x, y - height, x + width, y,
                                glow::COLOR_BUFFER_BIT, glow::NEAREST
                                );
                        }
                    }
                });
            });
        if r.is_none() {
            self.paper_ui_status = Canvas3dStatus::default();
        }
    }

    fn open_confirmation_dialog(&mut self, ui: &Ui, title: &str, message: &str, f: impl Fn(&mut MenuActions) + 'static) {
        self.confirmable_action = Some(ConfirmableAction {
            title: title.to_owned(),
            message: message.to_owned(),
            action: Box::new(f),
        });
        ui.open_popup("###Confirm");
    }

    fn run_menu_actions(&mut self, ui: &Ui, menu_actions: &MenuActions) {
        if menu_actions.reset_views {
            self.data.reset_views(to_cgv2(self.sz_scene), to_cgv2(self.sz_paper));
        }
        if menu_actions.undo {
            match self.data.undo_action() {
                UndoResult::Model => {
                    self.add_rebuild(RebuildFlags::all());
                }
                UndoResult::ModelAndOptions => {
                    if let Some(o) = self.options_opened.as_mut() {
                        *o = self.data.papercraft().options().clone();
                    }
                    self.add_rebuild(RebuildFlags::all());
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
                let fd = imgui_filedialog::Builder::new("fd")
                    .filter("Papercraft (*.craft) {.craft},All files {.*}")
                    .path(&self.last_path)
                    .flags(imgui_filedialog::Flags::DISABLE_CREATE_DIRECTORY_BUTTON | imgui_filedialog::Flags::SHOW_READ_ONLY_CHECK | imgui_filedialog::Flags::NO_DIALOG)
                    .open();
                self.file_dialog = Some((fd, "Open...", FileAction::OpenCraft));
                open_file_dialog = true;
            }
            BoolWithConfirm::None => {}
        }
        if menu_actions.save {
            match &self.file_name {
                Some(f) => {
                    self.file_action = Some((FileAction::SaveAsCraft, f.clone()));
                    open_wait = true;
                }
                None => save_as = true,
            }
        }
        if menu_actions.save_as || save_as {
            let fd = imgui_filedialog::Builder::new("fd")
                .filter("Papercraft (*.craft) {.craft},All files {.*}")
                .path(&self.last_path)
                .flags(imgui_filedialog::Flags::CONFIRM_OVERWRITE | imgui_filedialog::Flags::NO_DIALOG)
                .open();
            self.file_dialog = Some((fd, "Save as...", FileAction::SaveAsCraft));
            open_file_dialog = true;
        }
        const LOAD_MODEL_FILTER: &str = "\
            All models (*.obj *.pdo *.stl) {.obj,.pdo,.stl},\
            Wavefront (*.obj) {.obj},\
            Pepakura (*.pdo) {.pdo},\
            Stl (*.stl) {.stl},\
            All files {.*}\
            ";
        match menu_actions.import_model {
            BoolWithConfirm::Requested => {
                self.open_confirmation_dialog(ui,
                    "Import model",
                    "The model has not been save, continue anyway?",
                    |a| a.import_model = BoolWithConfirm::Confirmed
                );
            }
            BoolWithConfirm::Confirmed => {
                let fd = imgui_filedialog::Builder::new("fd")
                    .filter(LOAD_MODEL_FILTER)
                    .path(&self.last_path)
                    .flags(imgui_filedialog::Flags::DISABLE_CREATE_DIRECTORY_BUTTON | imgui_filedialog::Flags::NO_DIALOG)
                    .open();
                self.file_dialog = Some((fd, "Import OBJ...", FileAction::ImportModel));
                open_file_dialog = true;
            }
            BoolWithConfirm::None => {}
        }
        match menu_actions.update_model {
            BoolWithConfirm::Requested => {
                self.open_confirmation_dialog(ui,
                    "Update model",
                    "This model is not saved and this operation cannot be undone.\nContinue anyway?",
                    |a| a.update_model = BoolWithConfirm::Confirmed
                );
            }
            BoolWithConfirm::Confirmed => {
                let fd = imgui_filedialog::Builder::new("fd")
                    .filter(LOAD_MODEL_FILTER)
                    .path(&self.last_path)
                    .flags(imgui_filedialog::Flags::DISABLE_CREATE_DIRECTORY_BUTTON | imgui_filedialog::Flags::NO_DIALOG)
                    .open();
                self.file_dialog = Some((fd, "Update with new OBJ...", FileAction::UpdateObj));
                open_file_dialog = true;
            }
            BoolWithConfirm::None => {}
        }
        if menu_actions.export_obj {
            let fd = imgui_filedialog::Builder::new("fd")
                .filter("Wavefront (*.obj) {.obj},All files {.*}")
                .path(&self.last_path)
                .flags(imgui_filedialog::Flags::CONFIRM_OVERWRITE | imgui_filedialog::Flags::NO_DIALOG)
                .open();
            self.file_dialog = Some((fd, "Export OBJ...", FileAction::ExportObj));
            open_file_dialog = true;
        }
        if menu_actions.generate_printable {
            use std::borrow::Cow::{Borrowed, Owned};

            let (last_path, last_file) = if self.last_export.is_empty() {
                (Borrowed(&self.last_path), Borrowed(""))
            } else {
                let path = PathBuf::from(&self.last_export);
                let last_path = path.parent()
                    .map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(String::new);
                let last_file = path.file_name()
                    .map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(String::new);
                (Owned(last_path), Owned(last_file))
            };
            let fd = imgui_filedialog::Builder::new("fd")
                .filter("PDF document (*.pdf) {.pdf},SVG documents (*.svg) {.svg},PNG documents (*.png) {.png},All files {.*}")
                .path(&last_path)
                .file(&last_file)
                .flags(imgui_filedialog::Flags::CONFIRM_OVERWRITE | imgui_filedialog::Flags::NO_DIALOG)
                .open();
            self.file_dialog = Some((fd, "Generate Printable...", FileAction::GeneratePrintable));
            open_file_dialog = true;
        }

        // There are two Wait modals and two Error modals. One pair over the FileDialog, the other to be opened directly ("Save").

        if open_file_dialog {
            ui.open_popup("###file_dialog_modal");
        }
        if let Some((mut fd, title, action)) = self.file_dialog.take() {
            let dsp_size = ui.display_size();
            let min_size = 0.75 * dsp_size;
            let max_size = dsp_size;
            ui.set_next_window_size_constraints(min_size, max_size);
            ui.popup_modal_config(format!("{title}###file_dialog_modal"))
                .close_button(true)
                .with(|| {
                    let mut finish_file_dialog = false;
                    let size = ui.get_content_region_avail();
                    if let Some(fd2) = fd.display("fd", imgui::WindowFlags::empty(), size, size) {
                        if fd2.ok() {
                            if let Some(file) = fd2.file_path_name(imgui_filedialog::ResultMode::AddIfNoFileExt) {
                                let action = if action == FileAction::OpenCraft && fd2.readonly() {
                                    FileAction::OpenCraftReadOnly
                                } else {
                                    action
                                };
                                self.file_action = Some((action, file.into()));
                                open_wait = true;
                                if let Some(path) = fd2.current_path() {
                                    self.last_path = path;
                                }
                            }
                        }
                        finish_file_dialog = true;
                    }
                    if ui.is_key_pressed(imgui::Key::Escape) {
                        finish_file_dialog = true;
                    }
                    if finish_file_dialog {
                        ui.close_current_popup();
                    } else {
                        self.file_dialog = Some((fd, title, action));
                    }
                });
        }
        if open_wait {
            self.popup_time_start = Instant::now();
            ui.open_popup("###Wait");
        }
    }

    fn run_mouse_actions(&mut self, ui: &Ui) {

        let shift_down = ui.is_key_down(imgui::Key::ModShift);
        let control_down = ui.is_key_down(imgui::Key::ModCtrl);
        let alt_down = ui.is_key_down(imgui::Key::ModAlt);

        let mouse_pos = self.scene_ui_status.mouse_pos;
        if self.scene_ui_status.action != Canvas3dAction::None {
            'zoom: {
                let dz = match ui.io().MouseWheel {
                    x if x < 0.0 => 1.0 / 1.1,
                    x if x > 0.0 => 1.1,
                    _ => break 'zoom,
                };
                let flags = self.data.scene_zoom(to_cgv2(self.sz_scene), to_cgv2(mouse_pos), dz);
                self.add_rebuild(flags);
            }
        }
        let flags = match &self.scene_ui_status.action {
            Canvas3dAction::Hovering => {
                self.data.scene_hover_event(to_cgv2(self.sz_scene), to_cgv2(mouse_pos), alt_down)
            }
            Canvas3dAction::Pressed(MouseButton::Left) |
            Canvas3dAction::Dragging(MouseButton::Left) => {
                self.data.scene_button1_click_event(self.sz_scene, mouse_pos)
            }
            Canvas3dAction::Pressed(MouseButton::Right) |
            Canvas3dAction::Dragging(MouseButton::Right) => {
                self.data.scene_button2_click_event(self.sz_scene, mouse_pos)
            }
            Canvas3dAction::DoubleClicked(MouseButton::Left) => {
                self.data.scene_button1_dblclick_event(self.sz_scene, mouse_pos)
            }
            Canvas3dAction::Released(MouseButton::Left) => {
                self.data.scene_button1_release_event(self.sz_scene, mouse_pos, shift_down, control_down)
            }
            _ => RebuildFlags::empty(),
        };
        self.add_rebuild(flags);

        let mouse_pos = self.paper_ui_status.mouse_pos;
        if self.paper_ui_status.action != Canvas3dAction::None {
            'zoom: {
                let dz = match ui.io().MouseWheel {
                    x if x < 0.0 => 1.0 / 1.1,
                    x if x > 0.0 => 1.1,
                    _ => break 'zoom,
                };
                let flags = self.data.paper_zoom(self.sz_paper, mouse_pos, dz);
                self.add_rebuild(flags);
            }
        }
        let flags = match &self.paper_ui_status.action {
            Canvas3dAction::Hovering => {
                self.data.paper_hover_event(self.sz_paper, mouse_pos, alt_down)
            }
            Canvas3dAction::Clicked(MouseButton::Left) |
            Canvas3dAction::DoubleClicked(MouseButton::Left) => {
                self.data.paper_button1_click_event(self.sz_paper, mouse_pos, shift_down, control_down, self.modifiable())
            }
            Canvas3dAction::Pressed(MouseButton::Right) |
            Canvas3dAction::Dragging(MouseButton::Right) => {
                self.data.paper_button2_event(self.sz_paper, mouse_pos)
            }
            Canvas3dAction::Pressed(MouseButton::Left) => {
                self.data.paper_button1_grab_event(self.sz_paper, mouse_pos, shift_down)
            }
            Canvas3dAction::Dragging(MouseButton::Left) => {
                self.data.paper_button1_grab_event(self.sz_paper, mouse_pos, shift_down)
            }
            _ => RebuildFlags::empty(),
        };
        self.add_rebuild(flags);
    }

    fn render_scene(&mut self) {
        let gl_fixs = &self.gl_fixs;

        let light0 = Vector3::new(-0.5, -0.4, -0.8).normalize() * 0.55;
        let light1 = Vector3::new(0.8, 0.2, 0.4).normalize() * 0.25;

        let mut u = Uniforms3D {
            m: self.data.ui.trans_scene.persp * self.data.ui.trans_scene.view,
            mnormal: self.data.ui.trans_scene.mnormal, // should be transpose of inverse
            lights: [light0, light1],
            tex: 0,
            line_top: 0,
            texturize: 0,
        };
        unsafe {
            self.gl.clear_color(0.2, 0.2, 0.4, 1.0);
            self.gl.clear_depth_f32(1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            self.gl.enable(glow::BLEND);
            self.gl.enable(glow::DEPTH_TEST);
            self.gl.depth_func(glow::LEQUAL);
            self.gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA);

            self.gl.bind_vertex_array(Some(gl_fixs.vao.id()));
            if let (Some(tex), true) = (&self.data.gl_objs().textures, self.data.ui.show_textures) {
                self.gl.active_texture(glow::TEXTURE0);
                self.gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(tex.id()));
                u.texturize = 1;
            }

            self.gl.polygon_offset(1.0, 1.0);
            self.gl.enable(glow::POLYGON_OFFSET_FILL);

            gl_fixs.prg_scene_solid.draw(&u, (&self.data.gl_objs().vertices, &self.data.gl_objs().vertices_sel), glow::TRIANGLES);

            if self.data.ui.show_3d_lines {
                //Joined edges
                self.gl.line_width(1.0);
                self.gl.disable(glow::LINE_SMOOTH);
                gl_fixs.prg_scene_line.draw(&u, &self.data.gl_objs().vertices_edge_joint, glow::LINES);

                //Cut edges
                self.gl.line_width(3.0);
                self.gl.enable(glow::LINE_SMOOTH);
                gl_fixs.prg_scene_line.draw(&u, &self.data.gl_objs().vertices_edge_cut, glow::LINES);
            }

            //Selected edge
            if self.data.has_selected_edge() {
                self.gl.line_width(5.0);
                self.gl.enable(glow::LINE_SMOOTH);
                if self.data.ui.xray_selection {
                    u.line_top = 1;
                }
                gl_fixs.prg_scene_line.draw(&u, &self.data.gl_objs().vertices_edge_sel, glow::LINES);
            }
        }
    }
    fn render_paper(&mut self) {
        let gl_fixs = &self.gl_fixs;

        let mut u = Uniforms2D {
            m: self.data.ui.trans_paper.ortho * self.data.ui.trans_paper.mx,
            tex: 0,
            frac_dash: 0.5,
            line_color: Rgba::new(0.0, 0.0, 0.0, 0.0),
            texturize: 0,
            notex_color: Rgba::new(0.75, 0.75, 0.75, 1.0),
        };

        unsafe {
            self.gl.clear_color(0.7, 0.7, 0.7, 1.0);
            self.gl.clear_stencil(1);
            self.gl.stencil_mask(0xff);
            self.gl.stencil_func(glow::ALWAYS, 0, 0);
            self.gl.disable(glow::STENCIL_TEST);

            self.gl.clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT);

            self.gl.enable(glow::BLEND);
            self.gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA);

            self.gl.bind_vertex_array(Some(gl_fixs.vao.id()));
            if let (Some(tex), true) = (&self.data.gl_objs().textures, self.data.ui.show_textures) {
                self.gl.active_texture(glow::TEXTURE0);
                self.gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(tex.id()));
                u.texturize = 1;
            }

            // The paper
            self.gl.enable(glow::STENCIL_TEST);
            self.gl.stencil_op(glow::KEEP, glow::KEEP, glow::ZERO);

            gl_fixs.prg_paper_solid.draw(&u, &self.data.gl_objs().paper_vertices_page, glow::TRIANGLES);

            self.gl.disable(glow::STENCIL_TEST);

            u.line_color = Rgba::new(0.5, 0.5, 0.5, 1.0);

            gl_fixs.prg_paper_line.draw(&u, &self.data.gl_objs().paper_vertices_margin, glow::LINES);

            u.line_color = Rgba::new(0.0, 0.0, 0.0, 1.0);

            // Line Flaps
            if self.data.ui.show_flaps {
                gl_fixs.prg_paper_line.draw(&u, &self.data.gl_objs().paper_vertices_flap_edge, glow::LINES);
            }

            self.gl.enable(glow::STENCIL_TEST);
            self.gl.stencil_op(glow::KEEP, glow::KEEP, glow::INCR);


            // Solid Flaps
            if self.data.ui.show_flaps {
                gl_fixs.prg_paper_solid.draw(&u, &self.data.gl_objs().paper_vertices_flap, glow::TRIANGLES);
            }
            self.gl.disable(glow::STENCIL_TEST);

            // Borders
            gl_fixs.prg_paper_line.draw(&u, &self.data.gl_objs().paper_vertices_edge_cut, glow::LINES);

            self.gl.enable(glow::STENCIL_TEST);

            // Textured faces
            gl_fixs.prg_paper_solid.draw(&u, (&self.data.gl_objs().paper_vertices, &self.data.gl_objs().paper_vertices_sel) , glow::TRIANGLES);

            self.gl.disable(glow::STENCIL_TEST);

            // Shadow Flaps
            u.texturize = 0;
            if self.data.ui.show_flaps {
                u.notex_color = Rgba::new(0.0, 0.0, 0.0, 0.0);
                gl_fixs.prg_paper_solid.draw(&u, &self.data.gl_objs().paper_vertices_shadow_flap, glow::TRIANGLES);
                u.notex_color = Rgba::new(0.75, 0.75, 0.75, 1.0);
            }

            // Creases
            gl_fixs.prg_paper_line.draw(&u, &self.data.gl_objs().paper_vertices_edge_crease, glow::LINES);

            // Selected edge
            if self.data.has_selected_edge() {
                u.line_color = color_edge(self.data.ui.mode);
                gl_fixs.prg_paper_line.draw(&u, &self.data.gl_objs().paper_vertices_edge_sel, glow::LINES);
            }

            // Draw the highlight overlap if "1 < STENCIL"
            self.gl.enable(glow::STENCIL_TEST);
            self.gl.stencil_op(glow::KEEP, glow::KEEP, glow::KEEP);

            if self.data.ui.highlight_overlaps {
                // Draw the overlapped highlight if "1 < STENCIL"
                let uq = UniformQuad { color: Rgba::new(1.0, 0.0, 1.0, 0.9) };
                self.gl.stencil_func(glow::LESS, 1, 0xff);
                gl_fixs.prg_quad.draw(&uq, glr::NilVertexAttrib(3), glow::TRIANGLES);

                // Draw the highlight dim if "1 >= STENCIL"
                let uq = UniformQuad { color: Rgba::new(1.0, 1.0, 1.0, 0.9) };
                self.gl.stencil_func(glow::GEQUAL, 1, 0xff);
                gl_fixs.prg_quad.draw(&uq, glr::NilVertexAttrib(3), glow::TRIANGLES);
            } else {
                // If highlight is disabled wraw the overlaps anyway, but dimmer, or else it would be invisible
                let uq = UniformQuad { color: Rgba::new(1.0, 0.0, 1.0, 0.5) };
                self.gl.stencil_func(glow::LESS, 1, 0xff);
                gl_fixs.prg_quad.draw(&uq, glr::NilVertexAttrib(3), glow::TRIANGLES);
            }

            self.gl.disable(glow::STENCIL_TEST);
        }
    }
    fn add_rebuild(&mut self, flags: RebuildFlags) {
        self.rebuild.insert(flags);
    }
    fn set_mouse_mode(&mut self, mode: MouseMode) {
        self.data.ui.mode = mode;
        self.add_rebuild(RebuildFlags::SELECTION | RebuildFlags::SCENE_REDRAW | RebuildFlags::PAPER_REDRAW);
    }

    fn title(&self, with_unsaved_check: bool) -> String {
        let unsaved = if with_unsaved_check && self.data.modified { "*" } else { "" };
        let app_name = "Papercraft";
        match &self.file_name {
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
    fn updated_title(&mut self) -> Option<&str> {
        let new_title = self.title(true);
        if new_title == self.title {
            None
        } else {
            self.title = new_title;
            Some(&self.title)
        }
    }
    fn run_file_action(&mut self, action: FileAction, file_name: impl AsRef<Path>) -> anyhow::Result<()> {
        let file_name = file_name.as_ref();
        match action {
            FileAction::OpenCraft => {
                self.open_craft(file_name)?;
                self.file_name = Some(file_name.to_owned());
            }
            FileAction::OpenCraftReadOnly => {
                self.open_craft(file_name)?;
                self.data.ui.mode = MouseMode::ReadOnly;
                self.file_name = Some(file_name.to_owned());
            }
            FileAction::SaveAsCraft => {
                self.save_as_craft(file_name)?;
                self.data.modified = false;
                self.file_name = Some(file_name.to_owned());
            }
            FileAction::ImportModel => {
                let is_native = self.import_model(file_name)?;
                if is_native {
                    // just like "open"
                    self.file_name = Some(file_name.to_owned());
                } else {
                    self.data.modified = true;
                    self.file_name = None;
                }
            }
            FileAction::UpdateObj => {
                self.update_obj(file_name)?;
                self.data.modified = true;
            }
            FileAction::ExportObj => self.export_obj(file_name)?,
            FileAction::GeneratePrintable => {
                self.generate_printable(file_name)?;
                self.last_export = file_name.to_string_lossy().into_owned();
            }
        }
        Ok(())
    }
    fn open_craft(&mut self, file_name: &Path) -> anyhow::Result<()> {
        let fs = std::fs::File::open(file_name)
            .with_context(|| format!("Error opening file {}", file_name.display()))?;
        let fs = std::io::BufReader::new(fs);
        let papercraft = Papercraft::load(fs)
            .with_context(|| format!("Error loading file {}", file_name.display()))?;
        self.data = PapercraftContext::from_papercraft(papercraft, &self.gl)?;
        self.data.reset_views(self.sz_scene, self.sz_paper);
        self.rebuild = RebuildFlags::all();
        Ok(())
    }
    fn save_as_craft(&self, file_name: &Path) -> anyhow::Result<()> {
        let f = std::fs::File::create(file_name)
            .with_context(|| format!("Error creating file {}", file_name.display()))?;
        let f = std::io::BufWriter::new(f);
        self.data.papercraft().save(f)
            .with_context(|| format!("Error saving file {}", file_name.display()))?;
        Ok(())
    }
    fn import_model(&mut self, file_name: &Path) -> anyhow::Result<bool> {
        let (papercraft, is_native) = import_model_file(file_name)?;
        self.data = PapercraftContext::from_papercraft(papercraft, &self.gl)?;
        self.data.reset_views(self.sz_scene, self.sz_paper);
        self.rebuild = RebuildFlags::all();
        Ok(is_native)
    }
    fn update_obj(&mut self, file_name: &Path) -> anyhow::Result<()> {
        let (mut new_papercraft, _) = import_model_file(file_name)?;
        new_papercraft.update_from_obj(self.data.papercraft());

        // Preserve the main user visible settings
        let prev_ui = self.data.ui.clone();
        self.data = PapercraftContext::from_papercraft(new_papercraft, &self.gl)?;
        self.rebuild = RebuildFlags::all();
        self.data.ui = prev_ui;
        self.data.modified = true;
        Ok(())
    }
    fn export_obj(&self, file_name: &Path) -> anyhow::Result<()> {
        self.data.papercraft().export_waveobj(file_name.as_ref())
            .with_context(|| format!("Error exporting to {}", file_name.display()))?;
        Ok(())
    }

    fn generate_printable(&self, file_name: &Path) -> anyhow::Result<()> {
        let res = match file_name.extension().map(|s| s.to_string_lossy().into_owned().to_ascii_lowercase()).as_deref() {
            Some("pdf") => self.generate_pdf(file_name),
            Some("svg") => self.generate_svg(file_name),
            Some("png") => self.generate_png(file_name),
            _ => anyhow::bail!("Don't know how to write the format of {}", file_name.display()),
        };
        res.with_context(|| format!("Error exporting to {}", file_name.display()))?;
        Ok(())
    }
    fn generate_pdf(&self, file_name: &Path) -> anyhow::Result<()> {
        let options = self.data.papercraft().options();
        let resolution = options.resolution as f32;
        let page_size_mm = Vector2::from(options.page_size);
        let page_size_inches = page_size_mm / 25.4;
        let page_size_dots = page_size_inches * 72.0;

        let pdf = cairo::PdfSurface::new(page_size_dots.x as f64, page_size_dots.y as f64, file_name)?;
        let title  = self.title(false);
        let _ = pdf.set_metadata(cairo::PdfMetadata::Title, &title);
        let _ = pdf.set_metadata(cairo::PdfMetadata::Creator, signature());
        let cr = cairo::Context::new(&pdf)?;

        self.generate_pages(|_page, pixbuf, texts, _| {
            // A PDF output is in "dots" for some reason
            // which is nice because 1pixel = 1dot
            let mut mc = cairo::Matrix::identity();
            cr.set_matrix(mc);

            // Scale pattern to in dots
            cr.set_source_surface(pixbuf, 0.0, 0.0)?;
            let pat = cr.source();
            let scale = resolution / 72.0; // to dots
            mc.scale(scale as f64, scale as f64);
            pat.set_matrix(mc);

            let _ = cr.paint();
            PrintableText::to_cairo_all(texts, options.edge_id_position, 72.0, &cr);
            let _ = cr.show_page();
            Ok(())
        })?;
        drop(cr);
        drop(pdf);

        Ok(())
    }

    fn generate_svg(&self, file_name: &Path) -> anyhow::Result<()> {
        let options = self.data.papercraft().options();
        let edge_id_position = options.edge_id_position;

        self.generate_pages(|page, pixbuf, texts, lines_by_island| {
            let name = Self::file_name_for_page(file_name, page);
            let out = std::fs::File::create(name)?;
            let mut out = std::io::BufWriter::new(out);

            let page_size = Vector2::from(options.page_size);
            let in_page = options.is_in_page_fn(page);

            let mut png = Vec::new();
            pixbuf.write_to_png(&mut png)?;

            let mut all_page_cuts = Vec::new();

            for (idx, (_, (lines, _))) in lines_by_island.iter().enumerate() {
                let cuts = lines.iter_cut();
                if let Some(page_cuts) = cuts_to_page_cuts(cuts, &in_page) {
                    all_page_cuts.push((idx, page_cuts));
                };
            }
            writeln!(&mut out, r#"<?xml version="1.0" encoding="UTF-8" standalone="no"?>"#)?;
            writeln!(
                &mut out,
                r#"<svg width="{0}mm" height="{1}mm" viewBox="0 0 {0} {1}" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" xmlns:xlink="http://www.w3.org/1999/xlink">"#,
                page_size.x, page_size.y
            )?;

            let write_layer_text = |out: &mut std::io::BufWriter<std::fs::File>| -> anyhow::Result<()> {
                if texts.is_empty() {
                    return Ok(());
                }
                // begin layer Text
                writeln!(out, r#"<g inkscape:label="Text" inkscape:groupmode="layer" id="Text">"#)?;
                for text in texts {
                    let basis2: cgmath::Basis2<f32> = Rotation2::from_angle(-text.angle);
                    let pos = basis2.rotate_vector(text.pos);
                    writeln!(out, r#"<text x="{}" y="{}" style="{}font-size:{};font-family:sans-serif;fill:#000000" transform="rotate({})">{}</text>"#,
                        pos.x,
                        pos.y,
                        match text.align {
                            TextAlign::Near => "",
                            TextAlign::Center => "text-anchor:middle;",
                            TextAlign::Far => "text-anchor:end;",
                        },
                        text.size,
                        Deg::from(text.angle).0,
                        text.text,
                    )?;
                }
                writeln!(out, r#"</g>"#)?;
                // end layer Text
                Ok(())
            };

            if edge_id_position != EdgeIdPosition::Inside {
                write_layer_text(&mut out)?;
            }

            // begin layer Background
            writeln!(&mut out, r#"<g inkscape:label="Background" inkscape:groupmode="layer" id="Background">"#)?;
            write!(
                &mut out,
                r#"<image width="{}" height="{}" preserveAspectRatio="none" xlink:href="data:image/png;base64,"#,
                page_size.x, page_size.y)?;
            {
                use base64::prelude::*;
                let mut b64png = base64::write::EncoderWriter::new(&mut out, &BASE64_STANDARD);
                b64png.write_all(&png)?;
                b64png.finish()?;
            }
            writeln!(&mut out, r#"" id="background" x="0" y="0" style="display:inline"/>"#)?;

            writeln!(&mut out, r#"</g>"#)?;
            // end layer Background

            if edge_id_position == EdgeIdPosition::Inside {
                write_layer_text(&mut out)?;
            }

            // begin layer Cut
            writeln!(&mut out, r#"<g inkscape:label="Cut" inkscape:groupmode="layer" id="Cut" style="display:none">"#)?;
            for (idx, page_cut) in all_page_cuts {
                writeln!(&mut out, r#"<path style="fill:none;stroke:#000000;stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter" id="cut_{}" d=""#, idx)?;
                write!(&mut out, r#"M "#)?;
                let page_contour = cut_to_contour(page_cut);
                for v in page_contour {
                    writeln!(&mut out, r#"{},{}"#, v.x, v.y)?;
                }
                writeln!(&mut out, r#"z"#)?;
                writeln!(&mut out, r#"" />"#)?;
            }
            writeln!(&mut out, r#"</g>"#)?;
            // end layer Cut

            // begin layer Fold
            writeln!(&mut out, r#"<g inkscape:label="Fold" inkscape:groupmode="layer" id="Fold" style="display:none">"#)?;
            for fold_kind in [EdgeDrawKind::Mountain, EdgeDrawKind::Valley] {
                writeln!(&mut out, r#"<g inkscape:label="{0}" inkscape:groupmode="layer" id="{0}">"#,
                    if fold_kind == EdgeDrawKind::Mountain { "Mountain"} else { "Valley" })?;
                for (idx, (_, (lines, extra))) in lines_by_island.iter().enumerate() {
                    let creases = lines.iter_crease(&extra, fold_kind);
                    // each crease can be checked for bounds individually
                    let page_creases = creases
                        .filter_map(|(a, b)| {
                            let (is_in_a, a) = in_page(a.pos);
                            let (is_in_b, b) = in_page(b.pos);
                            (is_in_a || is_in_b).then_some((a, b))
                        })
                        .collect::<Vec<_>>();
                    if !page_creases.is_empty() {
                        writeln!(&mut out, r#"<path style="fill:none;stroke:{1};stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter" id="{2}_{0}" d=""#,
                            idx,
                            if fold_kind == EdgeDrawKind::Mountain  { "#ff0000" } else { "#0000ff" },
                            if fold_kind == EdgeDrawKind::Mountain  { "foldm" } else { "foldv" }
                        )?;
                        for (a, b) in page_creases {
                            writeln!(&mut out, r#"M {},{} {},{}"#, a.x, a.y, b.x, b.y)?;
                        }
                        writeln!(&mut out, r#"" />"#)?;
                    }
                }
                writeln!(&mut out, r#"</g>"#)?;
            }
            writeln!(&mut out, r#"</g>"#)?;
            // end layer Fold

            writeln!(&mut out, r#"</svg>"#)?;
            Ok(())
        })?;
        Ok(())
    }
    fn generate_png(&self, file_name: &Path) -> anyhow::Result<()> {
        let options = self.data.papercraft().options();
        let resolution = options.resolution as f32;

        self.generate_pages(|page, pixbuf, texts, _| {
            let cr = cairo::Context::new(&pixbuf)?;
            PrintableText::to_cairo_all(texts, options.edge_id_position, resolution, &cr);

            let name = Self::file_name_for_page(file_name, page);
            let f = std::fs::File::create(name)?;
            let mut f = std::io::BufWriter::new(f);
            pixbuf.write_to_png(&mut f)?;
            Ok(())
        })?;
        Ok(())
    }

    fn file_name_for_page(file_name: &Path, page: u32) -> PathBuf {
        if page == 0 {
            return file_name.to_owned();
        }
        let ext = file_name.extension().unwrap_or_default();
        let stem = file_name.file_stem().unwrap_or_default();
        let stem = stem.to_string_lossy();
        let stem = stem.strip_suffix("_1").unwrap_or(&stem);
        let parent = file_name.parent().map(|p| p.to_owned()).unwrap_or_default();
        let mut name = PathBuf::from(format!("{}_{}", stem, page + 1));
        name.set_extension(ext);
        parent.join(name)
    }
    fn generate_pages<F>(&self, mut do_page_fn: F) -> anyhow::Result<()>
        where F: FnMut(u32, &cairo::ImageSurface, &[PrintableText], &[(IslandKey, (PaperDrawFaceArgs, PaperDrawFaceArgsExtra))]) -> anyhow::Result<()>
    {
        let options = self.data.papercraft().options();
        let (_margin_top, margin_left, margin_right, margin_bottom) = options.margin;
        let resolution = options.resolution as f32;
        let page_size_mm = Vector2::from(options.page_size);
        let page_size_inches = page_size_mm / 25.4;
        let page_size_pixels = page_size_inches * resolution;
        let page_size_pixels = cgmath::Vector2::new(page_size_pixels.x as i32, page_size_pixels.y as i32);

        let mut pixbuf = cairo::ImageSurface::create(cairo::Format::ARgb32, page_size_pixels.x, page_size_pixels.y)
            .with_context(|| anyhow!("Unable to create output pixbuf"))?;
        let stride = pixbuf.stride();

        unsafe {
            self.gl.pixel_store_i32(glow::PACK_ROW_LENGTH, stride / 4);

            let fbo = glr::Framebuffer::generate(&self.gl)?;
            let rbo = glr::Renderbuffer::generate(&self.gl)?;

            let draw_fb_binder = BinderDrawFramebuffer::bind(&fbo);
            let read_fb_binder = BinderReadFramebuffer::bind(&fbo);
            let rb_binder = BinderRenderbuffer::bind(&rbo);
            self.gl.framebuffer_renderbuffer(draw_fb_binder.target(), glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(rbo.id()));

            let rbo_fbo_no_aa = 'check_aa: {
                // multisample buffers cannot be read directly, it has to be copied to a regular one.
                for samples in MULTISAMPLES {
                    // check if these many samples are usable
                    self.gl.renderbuffer_storage_multisample(rb_binder.target(), *samples, glow::RGBA8, page_size_pixels.x, page_size_pixels.y);
                    if self.gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                        continue;
                    }

                    // If using AA create another FBO/RBO to blit the antialiased image before reading
                    let rbo2 = glr::Renderbuffer::generate(&self.gl)?;
                    rb_binder.rebind(&rbo2);
                    self.gl.renderbuffer_storage(rb_binder.target(), glow::RGBA8, page_size_pixels.x, page_size_pixels.y);

                    let fbo2 = glr::Framebuffer::generate(&self.gl)?;
                    read_fb_binder.rebind(&fbo2);
                    self.gl.framebuffer_renderbuffer(read_fb_binder.target(), glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(rbo2.id()));

                    break 'check_aa Some((rbo2, fbo2));
                }
                println!("No multisample!");
                self.gl.renderbuffer_storage(rb_binder.target(), glow::RGBA8, page_size_pixels.x, page_size_pixels.y);
                None
            };
            let _vp = glr::PushViewport::push(&self.gl, 0, 0, page_size_pixels.x, page_size_pixels.y);

            // Cairo surfaces are alpha-premultiplied:
            // * The framebuffer will be premultiplied, but the input fragments are not.
            // * The clear color is set to transparent (premultiplied).
            // * In the screen DST_ALPHA does not matter, because the framebuffer is not
            //   transparent, but here we have to set it to the proper value: use separate blend
            //   functions or we'll get the alpha squared.
            self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
            self.gl.enable(glow::BLEND);
            self.gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA);

            let gl_fixs = &self.gl_fixs;

            let mut texturize = 0;

            self.gl.bind_vertex_array(Some(gl_fixs.vao.id()));
            if let (Some(tex), true) = (&self.data.gl_objs().textures, options.texture) {
                self.gl.active_texture(glow::TEXTURE0);
                self.gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(tex.id()));
                texturize = 1;
            }

            let ortho = util_3d::ortho2d_zero(page_size_mm.x, -page_size_mm.y);

            let page_count = options.pages;
            let flap_style = options.flap_style;
            let edge_id_position = options.edge_id_position;

            let island_names = if edge_id_position != EdgeIdPosition::None {
                self.data.papercraft().build_island_names()
            } else {
                Default::default()
            };

            let mut texts = Vec::new();
            let lines_by_island = self.data.lines_by_island();

            for page in 0..page_count {
                // Start render
                self.gl.clear(glow::COLOR_BUFFER_BIT);
                let page_pos = options.page_position(page);
                let mt = Matrix3::from_translation(-page_pos);
                let mut u = Uniforms2D {
                    m: ortho * mt,
                    tex: 0,
                    frac_dash: 0.5,
                    line_color: Rgba::new(0.0, 0.0, 0.0, 1.0),
                    texturize,
                    notex_color: Rgba::new(1.0, 1.0, 1.0, 1.0),
                };
                // Line Flaps
                if flap_style != FlapStyle::None {
                    gl_fixs.prg_paper_line.draw(&u, &self.data.gl_objs().paper_vertices_flap_edge, glow::LINES);
                }

                // Solid Flaps
                if flap_style != FlapStyle::None && flap_style != FlapStyle::White {
                    gl_fixs.prg_paper_solid.draw(&u, &self.data.gl_objs().paper_vertices_flap, glow::TRIANGLES);
                }

                // Borders
                gl_fixs.prg_paper_line.draw(&u, &self.data.gl_objs().paper_vertices_edge_cut, glow::LINES);

                // Textured faces
                self.gl.vertex_attrib_4_f32(gl_fixs.prg_paper_solid.attrib_by_name("color").unwrap().location() as u32, 0.0, 0.0, 0.0, 0.0);
                gl_fixs.prg_paper_solid.draw(&u, &self.data.gl_objs().paper_vertices, glow::TRIANGLES);

                // Shadow Flaps
                u.texturize = 0;
                u.notex_color = Rgba::new(0.0, 0.0, 0.0, 0.0);
                gl_fixs.prg_paper_solid.draw(&u, &self.data.gl_objs().paper_vertices_shadow_flap, glow::TRIANGLES);
                u.notex_color = Rgba::new(1.0, 1.0, 1.0, 1.0);

                // Creases
                gl_fixs.prg_paper_line.draw(&u, &self.data.gl_objs().paper_vertices_edge_crease, glow::LINES);
                // End render

                if let Some((_, fbo_no_aa)) = &rbo_fbo_no_aa {
                    read_fb_binder.rebind(&fbo);
                    draw_fb_binder.rebind(fbo_no_aa);
                    self.gl.blit_framebuffer(
                        0, 0, page_size_pixels.x, page_size_pixels.y,
                        0, 0, page_size_pixels.x, page_size_pixels.y,
                        glow::COLOR_BUFFER_BIT, glow::NEAREST
                    );
                    read_fb_binder.rebind(fbo_no_aa);
                    draw_fb_binder.rebind(&fbo);
                }

                self.gl.read_buffer(glow::COLOR_ATTACHMENT0);

                {
                    let mut data = pixbuf.data()?;
                    self.gl.read_pixels(0, 0, page_size_pixels.x, page_size_pixels.y, glow::BGRA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(data.as_mut()));
                }

                let edge_id_font_size = options.edge_id_font_size * 25.4 / 72.0; // pt to mm
                let edge_id_position = options.edge_id_position;

                texts.clear();
                if options.show_self_promotion {
                    let x = margin_left;
                    let y = (page_size_mm.y - margin_bottom + FONT_SIZE).min(page_size_mm.y - FONT_SIZE);
                    let text = String::from(signature());
                    texts.push(PrintableText {
                        size: FONT_SIZE,
                        pos: Vector2::new(x, y),
                        angle: Rad(0.0),
                        align: TextAlign::Near,
                        text,
                    });
                }
                if options.show_page_number {
                    let x = page_size_mm.x - margin_right;
                    let y = (page_size_mm.y - margin_bottom + FONT_SIZE).min(page_size_mm.y - FONT_SIZE);
                    let text = format!("Page {}/{}", page + 1, page_count);
                    texts.push(PrintableText {
                        size: FONT_SIZE,
                        pos: Vector2::new(x, y),
                        angle: Rad(0.0),
                        align: TextAlign::Far,
                        text,
                    });
                }
                if edge_id_position != EdgeIdPosition::None {
                    let in_page = options.is_in_page_fn(page);
                    for (i_island, (lines, extra)) in &lines_by_island {
                        let cuts = lines.iter_cut();
                        let Some(page_cuts) = cuts_to_page_cuts(cuts, &in_page) else {
                            continue;
                        };
                        // Edge ids
                        for cut_idx in extra.cut_indices() {
                            let i_island_b = self.data.papercraft().island_by_face(cut_idx.i_face_b);
                            let ii = island_names.get(i_island_b).map(|s| s.as_str()).unwrap_or("?");
                            let text = format!("{}:{}", ii, cut_idx.id);
                            let pos = in_page(cut_idx.pos).1;
                            texts.push(PrintableText {
                                size: edge_id_font_size,
                                pos,
                                angle: cut_idx.angle,
                                align: TextAlign::Center,
                                text,
                            });
                        }
                        // Island ids
                        let pos = match edge_id_position {
                            // On top
                            EdgeIdPosition::None |
                            EdgeIdPosition::Outside => {
                                let top = page_cuts.iter().min_by(|a, b| a.0.y.total_cmp(&b.0.y)).unwrap().0;
                                top - Vector2::new(0.0, edge_id_font_size)
                            }
                            // In the middle
                            EdgeIdPosition::Inside => {
                                let island = self.data.papercraft().island_by_key(*i_island).unwrap();
                                let (flat_face, total_area) = self.data.papercraft().get_biggest_flat_face(island);
                                // Compute the center of mass of the flat-face, that will be the
                                // weighted mean of the centers of masses of each single face.
                                let center: Vector2 = flat_face
                                    .iter()
                                    .map(|(i_face, area)| {
                                        let vv: Vector2 = lines.vertices_for_face(*i_face).into_iter().sum();
                                        vv * *area
                                    })
                                    .sum();
                                // Don't forget to divide the center of each triangle by 3!
                                let center = center / total_area / 3.0;
                                let center = in_page(center).1;
                                center + Vector2::new(0.0, edge_id_font_size)
                            }
                        };
                        if let Some(ii) = island_names.get(*i_island) {
                            texts.push(PrintableText {
                                size: 2.0 * edge_id_font_size,
                                pos,
                                angle: Rad(0.0),
                                align: TextAlign::Center,
                                text: ii.clone(),
                            });
                        }
                    }
                }

                do_page_fn(page, &pixbuf, &texts, &lines_by_island)?;
            }
            self.gl.pixel_store_i32(glow::PACK_ROW_LENGTH, 0);
        }
        Ok(())
    }
    fn save_backup_on_panic(&self) {
        if !self.data.modified {
            return;
        }
        let mut dir = std::env::temp_dir();
        dir.push(format!("crashed-{}.craft", std::process::id()));
        eprintln!("Papercraft panicked! Saving backup at \"{}\"", dir.display());
        if let Err(e) = self.save_as_craft(&dir) {
            eprintln!("backup failed with {e:?}");
        } else {
            eprintln!("backup complete");
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
    Clicked(MouseButton),
    Pressed(MouseButton),
    Released(MouseButton),
    Dragging(MouseButton),
    DoubleClicked(MouseButton),
}

fn canvas3d(ui: &Ui, st: &mut Canvas3dStatus) {
    let sz = ui.get_content_region_avail();
    ui.invisible_button_config("canvas3d")
        .size(sz)
        .build();
    let hovered = ui.is_item_hovered();
    let pos = ui.get_item_rect_min();
    let scale = ui.display_scale();
    let mouse_pos = scale * (ui.get_mouse_pos() - pos);

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
            } else if ui.is_mouse_dragging(MouseButton::Left) {
                Canvas3dAction::Dragging(MouseButton::Left)
            } else if ui.is_mouse_dragging(MouseButton::Right) {
                Canvas3dAction::Dragging(MouseButton::Right)
            } else if ui.is_mouse_double_clicked(MouseButton::Left) {
                Canvas3dAction::DoubleClicked(MouseButton::Left)
            } else if ui.is_mouse_double_clicked(MouseButton::Right) {
                Canvas3dAction::DoubleClicked(MouseButton::Right)
            } else if ui.is_mouse_clicked(MouseButton::Left) {
                Canvas3dAction::Clicked(MouseButton::Left)
            } else if ui.is_mouse_clicked(MouseButton::Right) {
                Canvas3dAction::Clicked(MouseButton::Right)
            } else if ui.is_mouse_released(MouseButton::Left) {
                Canvas3dAction::Released(MouseButton::Left)
            } else if ui.is_mouse_released(MouseButton::Right) {
                Canvas3dAction::Released(MouseButton::Right)
            } else if ui.is_mouse_down(MouseButton::Left) {
                Canvas3dAction::Pressed(MouseButton::Left)
            } else if ui.is_mouse_down(MouseButton::Right) {
                Canvas3dAction::Pressed(MouseButton::Right)
            } else {
                Canvas3dAction::Hovering
            }
        }
        Canvas3dAction::None | Canvas3dAction::Released(_) => {
            // If the mouse is entered while dragging, it does not count, as if captured by other
            if hovered &&
                !ui.is_mouse_dragging(MouseButton::Left) &&
                !ui.is_mouse_dragging(MouseButton::Right)
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

fn load_image_from_memory(data: &[u8], premultiply: bool) -> Result<image::RgbaImage> {
    let image = image::load_from_memory_with_format(data, image::ImageFormat::Png)?;
    let image = if premultiply {
        premultiply_image(image)
    } else {
        image.into_rgba8()
    };
    Ok(image)
}

fn advance_cursor(ui: &Ui, x: f32, y: f32) {
    let f = ui.get_font_size();
    advance_cursor_pixels(ui, f * x, f * y);
}
fn advance_cursor_pixels(ui: &Ui, x: f32, y: f32) {
    let mut pos = ui.get_cursor_screen_pos();
    pos.x += x;
    pos.y += y;
    ui.set_cursor_screen_pos(pos);
}
fn center_text(ui: &Ui, s: &str, w: f32) {
    let ss = ui.calc_text_size(s);
    let mut pos = ui.get_cursor_screen_pos();
    pos.x += (w - ss.x) / 2.0;
    ui.set_cursor_screen_pos(pos);
    ui.text(s);
}

fn center_url(ui: &Ui, s: &str, id: &str, cmd: Option<&str>, w: f32) {
    let ss = ui.calc_text_size(s);
    let mut pos = ui.get_cursor_screen_pos();
    let pos0 = pos;
    pos.x += (w - ss.x) / 2.0;
    ui.set_cursor_screen_pos(pos);
    let color = ui.style().color(imgui::ColorId::ButtonActive);
    ui.with_push((imgui::ColorId::Text, color), || {
        ui.text(s);
        ui.set_cursor_screen_pos(pos0);
        if ui.invisible_button_config(id).size(ss).build() {
            let _ = opener::open_browser(cmd.unwrap_or(s));
        }
        if ui.is_item_hovered() {
            ui.set_mouse_cursor(imgui::MouseCursor::Hand);
        }
    });
}

pub fn cut_to_contour(mut cuts: Vec<(Vector2, Vector2)>) -> Vec<Vector2> {
    // Order the vertices in a closed loop
    let mut res = Vec::with_capacity(cuts.len());
    while let Some(mut p) = cuts.pop() {
        res.push(p.0);
        while let Some((next, _)) = cuts
            .iter()
            .enumerate()
            .map(|(idx, (v0, _))| (idx, v0.distance2(p.1)))
            .min_by(|(_, a), (_, b)| f32::total_cmp(a, b))
        {
            p = cuts.swap_remove(next);
            res.push(p.0);
        }
        // the last point should connect to the first
    }
    res
}

pub fn cuts_to_page_cuts<'c>(cuts: Vec<(&MVertex2DLine, &MVertex2DLine)>, in_page: impl Fn(Vector2) -> (bool, Vector2)) -> Option<Vec<(Vector2, Vector2)>>{
    let mut touching = false;
    let page_cut = cuts
        .iter()
        .map(|(v0, v1)| {
            let (is_in_0, v0) = in_page(v0.pos);
            let (is_in_1, v1) = in_page(v1.pos);
            touching |= is_in_0 | is_in_1;
            (v0, v1)
        })
        .collect::<Vec<_>>();
    touching.then_some(page_cut)
}

enum TextAlign {
    Near,
    Center,
    Far,
}

struct PrintableText {
    size: f32,
    pos: Vector2,
    angle: Rad<f32>,
    align: TextAlign,
    text: String,
}

impl PrintableText {
    fn to_cairo_all(texts: &[PrintableText], edge_id_position: EdgeIdPosition, resolution: f32, cr: &cairo::Context) {
        // Dest in millimeters
        let mut mc = cairo::Matrix::identity();
        let scale = resolution / 25.4; // to millimeters
        mc.scale(scale as f64, scale as f64);
        cr.set_matrix(mc);
        // Black
        cr.set_source_rgba(0.0, 0.0, 0.0, 1.0);
        match edge_id_position {
            // Text below the texture
            EdgeIdPosition::None |
            EdgeIdPosition::Outside => cr.set_operator(cairo::Operator::DestOver),
            // Text above the texture
            EdgeIdPosition::Inside => cr.set_operator(cairo::Operator::Over),
        }
        for text in texts {
            text.to_cairo(cr);
        }
    }
    fn to_cairo(&self, cr: &cairo::Context) {
        let m = cr.matrix();
        cr.set_font_size(self.size as f64);
        let align = match self.align {
            TextAlign::Near => { 0.0 }
            TextAlign::Center => {
                cr.text_extents(&self.text).unwrap().width() / 2.0
            }
            TextAlign::Far => {
                cr.text_extents(&self.text).unwrap().width()
            }
        };
        let basis2: cgmath::Basis2<f32> = Rotation2::from_angle(self.angle);
        let p = basis2.rotate_vector(Vector2::new(-align as f32, 0.0));

        let Vector2 { x, y } = self.pos + p;
        cr.move_to(x as f64, y as f64);
        cr.rotate(self.angle.0 as f64);
        let _ = cr.show_text(&self.text);
        cr.set_matrix(m);
    }
}

#[cfg(target_os="linux")]
#[inline(never)]
unsafe fn install_crash_backup(event_loop: winit::event_loop::EventLoopProxy<()>, ctx: &Rc<RefCell<GlobalContext>>) {
    // This is quite unsafe, maybe even UB, but we are crashing anyway, and we are trying to save
    // the user's data, what's the worst that could happen?
    // The `ctx` is not Send because of the Rc, the RefCell and the GlContext inside, but we won't touch any of these...
    struct SendWrapper<T>(T);
    unsafe impl<T> Send for SendWrapper<T> {}
    let ctx = SendWrapper(Rc::clone(ctx));
    use signal_hook::consts::signal::*;
    use signal_hook::iterator::Signals;
    std::thread::spawn(move || {
        let sigs = vec![SIGHUP, SIGINT, SIGTERM];
        let mut signals = Signals::new(sigs).unwrap();
        let _ = signals.into_iter().next();
        let _ = event_loop.send_event(());
        let _ = std::thread::Builder::new().spawn(move || {
            let _ = signals.into_iter().next();
            eprintln!("Double signal, aborting!");
            std::process::abort();
        });
        // the RefCell is probably borrowed, use it anyway :-/
        let ctx = ctx;
        let ctx = unsafe { &*ctx.0.as_ptr() };
        ctx.save_backup_on_panic();
        std::process::abort();
    });
}

#[cfg(not(target_os="linux"))]
unsafe fn install_crash_backup(_event_loop: winit::event_loop::EventLoopProxy<()>, _ctx: &Rc<RefCell<GlobalContext>>) { }

impl imgui::UiBuilder for GlobalContext {
    fn build_custom_atlas(&mut self, atlas: &mut imgui::FontAtlasMut<'_, Self>) {
        self.font_default = atlas.add_font(imgui::FontInfo::new(&*KARLA_TTF, 18.0));
        self.font_big = atlas.add_font(imgui::FontInfo::new(&*KARLA_TTF, 28.0));
        self.font_small = atlas.add_font_collection([
            imgui::FontInfo::new(&*KARLA_TTF, 12.0),
            imgui::FontInfo::new(&*COPYRIGHT_TTF, 12.0),
        ]);

        self.logo_rect = atlas.add_custom_rect_regular(
            [LOGO_IMG.width(), LOGO_IMG.height()],
            |_, img| { img.copy_from(&*LOGO_IMG, 0, 0).unwrap(); }
        );

        // Each image is 48x48
        const W: u32 = 48;
        const H: u32 = 48;
        self.icons_rect[0] = atlas.add_custom_rect_regular(
            [W, H],
            |_, img| { img.copy_from(&*ICONS_IMG.view(0, 0, W, H), 0, 0).unwrap(); }
        );
        self.icons_rect[1] = atlas.add_custom_rect_regular(
            [W, H],
            |_, img| { img.copy_from(&*ICONS_IMG.view(W, 0, W, H), 0, 0).unwrap(); }
        );
        self.icons_rect[2] = atlas.add_custom_rect_regular(
            [W, H],
            |_, img| { img.copy_from(&*ICONS_IMG.view(0, H, W, H), 0, 0).unwrap(); }
        );
    }
    fn do_ui(&mut self, ui: &Ui) {
        //ui.show_demo_window(None);
        ui.set_next_window_pos(vec2(0.0, 0.0), imgui::Cond::Always, vec2(0.0, 0.0));
        let vw = ui.get_main_viewport();
        let sz = vw.size(); //&ui.get_content_region_avail();
        ui.set_next_window_size(sz, imgui::Cond::Always);
        ui.window_config("Papercraft")
            .flags(
                imgui::WindowFlags::NoDecoration |
                imgui::WindowFlags::NoResize |
                imgui::WindowFlags::MenuBar |
                imgui::WindowFlags::NoBringToFrontOnFocus |
                imgui::WindowFlags::NoNav
            )
            .push_for_begin((
                (imgui::StyleVar::WindowPadding, imgui::StyleValue::Vec2([0.0, 0.0].into())),
                (imgui::StyleVar::WindowRounding, imgui::StyleValue::F32(0.0)),
            ))
            .with(|| {
                if let Some(cmd_file_action) = self.cmd_file_action.take() {
                    self.popup_time_start = Instant::now();
                    self.file_action = Some(cmd_file_action);
                    ui.open_popup("###Wait");
                }

                let menu_actions = self.build_ui(ui);
                self.run_menu_actions(ui, &menu_actions);
                self.run_mouse_actions(ui);

                if self.rebuild.intersects(RebuildFlags::ANY_REDRAW_SCENE | RebuildFlags::ANY_REDRAW_PAPER) {
                    self.data.pre_render(self.rebuild);
                    let vp = glr::PushViewport::new(&self.gl);
                    if self.rebuild.intersects(RebuildFlags::ANY_REDRAW_SCENE) {
                        let _draw_fb_binder = BinderDrawFramebuffer::bind(&self.gl_fixs.fbo_scene);
                        vp.viewport(0, 0, self.sz_scene.x as i32, self.sz_scene.y as i32);
                        self.render_scene();
                    }
                    if self.rebuild.intersects(RebuildFlags::ANY_REDRAW_PAPER) {
                        let _draw_fb_binder = BinderDrawFramebuffer::bind(&self.gl_fixs.fbo_paper);
                        vp.viewport(0, 0, self.sz_paper.x as i32, self.sz_paper.y as i32);
                        self.render_paper();
                    }
                    self.rebuild = RebuildFlags::empty();
                }

                match (menu_actions.quit, self.quit_requested) {
                    (BoolWithConfirm::Confirmed, _) | (_, BoolWithConfirm::Confirmed) => self.quit_requested = BoolWithConfirm::Confirmed,
                    (BoolWithConfirm::Requested, _) | (_, BoolWithConfirm::Requested)=> {
                        self.quit_requested = BoolWithConfirm::None;
                        self.open_confirmation_dialog(ui,
                            "Quit?",
                            "The model has not been save, continue anyway?",
                            |a| a.quit = BoolWithConfirm::Confirmed
                        );
                    }
                    _ => (),
                }
            });
    }
}


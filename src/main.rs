use anyhow::{Context, Result};
use cgmath::{prelude::*, Deg, Rad};
use easy_imgui_window::{
    easy_imgui::{self as imgui, id, lbl, lbl_id, vec2, Color, MouseButton, Vector2},
    easy_imgui_renderer::{
        glow::{self, HasContext},
        glr::{
            self, BinderDrawFramebuffer, BinderReadFramebuffer, BinderRenderbuffer, GlContext, Rgba,
        },
        Renderer,
    },
    winit,
};
use image::{DynamicImage, EncodableLayout, GenericImage, GenericImageView, Pixel};
use lazy_static::lazy_static;
use std::{
    f32,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use std::{io::Write, sync::atomic::AtomicPtr};
use tr::tr;
use winit::{
    event::WindowEvent,
    event_loop::{EventLoop, EventLoopProxy},
};

type Ui = imgui::Ui<Box<GlobalContext>>;

static MULTISAMPLES: &[i32] = &[16, 8, 4, 2];

use easy_imgui_filechooser as filechooser;

mod config;
mod paper;
mod pdf_metrics;
mod printable;
mod util_3d;
mod util_gl;

mod ui;

use ui::*;

lazy_static! {
    static ref LOGO_IMG: image::RgbaImage =
        load_image_from_memory(include_bytes!("papercraft.png"), true).unwrap();
    static ref ICONS_IMG: image::RgbaImage =
        load_image_from_memory(include_bytes!("icons.png"), true).unwrap();
}

const KARLA_TTF: &[u8] = include_bytes!("Karla-Regular.ttf");
const FONT_SIZE: f32 = 3.0;

use paper::{
    import::import_model_file, EdgeIdPosition, FlapStyle, FoldStyle, IslandKey, PaperOptions,
    Papercraft,
};
use util_3d::{Matrix3, Vector3};
use util_gl::{MVertex2DLine, UniformQuad, Uniforms2D, Uniforms3D};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// Long
struct Cli {
    #[arg(value_name = "MODEL_FILE")]
    name: Option<PathBuf>,

    #[arg(
        short,
        long,
        help = tr!("Prevents editing of the model, useful as reference to build a real model")
    )]
    read_only: bool,
}

include!(concat!(env!("OUT_DIR"), "/locale/translators.rs"));

fn set_locale(locale: &str) {
    log::info!("Change locale to {locale}");
    translators::set_locale(locale);
    easy_imgui_filechooser::set_locale(locale);
}

static LANGUAGES: &[(&str, &str)] = &[("en", "English"), ("es", "Español")];

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let config = config::Config::load_or_default();
    log::info!("Restored config {config:?}");
    // Set the preferred language now for early messages
    set_locale(&config.locale);

    let cli = Cli::parse();

    let event_loop = EventLoop::new().unwrap();
    let proxy = event_loop.create_proxy();

    let data = AppData { proxy, cli, config };
    let mut main = easy_imgui_window::AppHandler::<Box<GlobalContext>>::new(data);
    let icon = winit::window::Icon::from_rgba(
        LOGO_IMG.as_bytes().to_owned(),
        LOGO_IMG.width(),
        LOGO_IMG.height(),
    )
    .unwrap();
    let attrs = main.attributes();
    attrs.window_icon = Some(icon);
    attrs.title = tr!("Papercraft");

    let maybe_fatal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        event_loop.run_app(&mut main).unwrap();
    }));
    // If it crashes try to save a backup
    if let Err(e) = maybe_fatal {
        CTX.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
        if let Some(ctx) = &main.app() {
            ctx.save_backup_on_panic();
        }
        std::panic::resume_unwind(e);
    }
}

struct AppData {
    proxy: EventLoopProxy<()>,
    cli: Cli,
    config: config::Config,
}

static CTX: AtomicPtr<GlobalContext> = AtomicPtr::new(std::ptr::null_mut());

impl easy_imgui_window::Application for Box<GlobalContext> {
    type UserEvent = ();
    type Data = AppData;

    fn new(args: easy_imgui_window::Args<AppData>) -> Box<GlobalContext> {
        let easy_imgui_window::Args {
            window,
            data: AppData { proxy, cli, config },
            ..
        } = args;

        window.main_window().window().set_ime_allowed(true);
        let renderer = window.renderer();
        renderer.set_background_color(Some(Color::BLACK));
        let gl = renderer.gl_context().clone();
        let imgui = renderer.imgui();
        let mut imgui = unsafe { imgui.set_current() };
        imgui.set_allow_user_scaling(true);
        imgui.nav_enable_keyboard();

        // Initialize papercraft status
        let mut data = PapercraftContext::from_papercraft(Papercraft::empty(), &gl).unwrap();
        let cmd_file_action = match cli {
            Cli {
                name: Some(name),
                read_only: false,
                ..
            } => Some((FileAction::ImportModel, name.to_owned())),
            Cli {
                name: Some(name),
                read_only: true,
                ..
            } => {
                // This will be rewritten when/if the file is loaded, but setting it here avoids a UI flicker
                data.ui.mode = MouseMode::ReadOnly;
                Some((FileAction::OpenCraftReadOnly, name.to_owned()))
            }
            _ => None,
        };

        let last_path = if let Some((_, path)) = &cmd_file_action {
            path.parent().map(|p| p.to_owned()).unwrap_or_default()
        } else {
            PathBuf::new()
        };

        let gl_fixs = build_gl_fixs(&gl).unwrap();
        let ctx = GlobalContext {
            config: config.clone(),
            gl,
            gl_fixs,
            font_default: imgui::FontId::default(),
            font_big: imgui::FontId::default(),
            font_small: imgui::FontId::default(),
            font_text: imgui::FontId::default(),
            font_text_line_scale: 1.0,
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
            config_opened: None,
            // Apply the configuration as soon as possible
            config_applied: Some(config.clone()),
            options_opened: None,
            options_applied: None,
            about_visible: false,
            option_button_height: 0.0,
            file_dialog: None,
            filechooser_atlas: Default::default(),
            file_action: None,
            last_path,
            last_export: PathBuf::new(),
            error_message: None,
            confirmable_action: None,
            popup_time_start: Instant::now(),
            cmd_file_action,
            quit_requested: BoolWithConfirm::None,
            title: String::new(),
        };
        let mut ctx = Box::new(ctx);
        // SAFETY: This code will only be run once, and the ctx is in a box,
        // so it will not be moved around memory.
        // Using the pointer will still be technical UB, probably, but hopefully
        // we'll never need it.
        unsafe {
            CTX.store(&mut *ctx, std::sync::atomic::Ordering::Release);
            install_crash_backup(proxy.clone());
        }
        ctx
    }
    fn window_event(
        &mut self,
        args: easy_imgui_window::Args<AppData>,
        _event: WindowEvent,
        res: easy_imgui_window::EventResult,
    ) {
        let easy_imgui_window::Args {
            window, event_loop, ..
        } = args;

        if let Some(new_title) = self.updated_title() {
            window.main_window().window().set_title(new_title);
        }
        if let Some((options, push_undo)) = self.options_applied.take() {
            self.data.set_papercraft_options(options, push_undo);
            self.add_rebuild(RebuildFlags::all());
            window.renderer().imgui().invalidate_font_atlas();
        }
        if let Some(new_config) = self.config_applied.take() {
            self.config = new_config;
            set_locale(&self.config.locale);
            let mut imgui = unsafe { window.renderer().imgui().set_current() };
            if self.config.light_mode {
                imgui.style().set_colors_light();
            } else {
                imgui.style().set_colors_dark();
            }
            match self.config.save() {
                Ok(_) => log::info!("Saved config {:?}", self.config),
                Err(e) => log::error!("Error saving config {e}"),
            }
        }

        if res.window_closed && self.quit_requested == BoolWithConfirm::None {
            let quit = self.check_modified();
            self.quit_requested = quit;
        }
        if self.quit_requested == BoolWithConfirm::Confirmed {
            event_loop.exit();
        }
    }
    fn user_event(&mut self, _args: easy_imgui_window::Args<AppData>, _: ()) {
        //Fatal signal: it is about to be aborted, just stop whatever it is doing and
        //let the crash handler do its job.
        loop {
            std::thread::park();
        }
    }
}

fn build_gl_fixs(gl: &GlContext) -> Result<GLFixedObjects> {
    let prg_scene_solid =
        util_gl::program_from_source(gl, include_str!("shaders/scene_solid.glsl"))
            .with_context(|| "scene_solid")?;
    let prg_scene_line = util_gl::program_from_source(gl, include_str!("shaders/scene_line.glsl"))
        .with_context(|| "scene_line")?;
    let prg_paper_solid =
        util_gl::program_from_source(gl, include_str!("shaders/paper_solid.glsl"))
            .with_context(|| "paper_solid")?;
    let prg_paper_line = util_gl::program_from_source(gl, include_str!("shaders/paper_line.glsl"))
        .with_context(|| "paper_line")?;
    let prg_quad = util_gl::program_from_source(gl, include_str!("shaders/quad.glsl"))
        .with_context(|| "quad")?;
    let prg_text = util_gl::program_from_source(gl, include_str!("shaders/text.glsl"))
        .with_context(|| "text")?;

    let vao = glr::VertexArray::generate(gl)?;

    let fbo_scene = glr::Framebuffer::generate(gl)?;
    let rbo_scene_color = glr::Renderbuffer::generate(gl)?;
    let rbo_scene_depth = glr::Renderbuffer::generate(gl)?;

    unsafe {
        let fb_binder = BinderDrawFramebuffer::bind(&fbo_scene);

        let rb_binder = BinderRenderbuffer::bind(&rbo_scene_color);
        gl.renderbuffer_storage(rb_binder.target(), glow::RGBA8, 1, 1);
        gl.framebuffer_renderbuffer(
            fb_binder.target(),
            glow::COLOR_ATTACHMENT0,
            glow::RENDERBUFFER,
            Some(rbo_scene_color.id()),
        );

        rb_binder.rebind(&rbo_scene_depth);
        gl.renderbuffer_storage(rb_binder.target(), glow::DEPTH_COMPONENT, 1, 1);
        gl.framebuffer_renderbuffer(
            fb_binder.target(),
            glow::DEPTH_ATTACHMENT,
            glow::RENDERBUFFER,
            Some(rbo_scene_depth.id()),
        );
    }

    let fbo_paper = glr::Framebuffer::generate(gl)?;
    let rbo_paper_color = glr::Renderbuffer::generate(gl)?;
    let rbo_paper_stencil = glr::Renderbuffer::generate(gl)?;

    unsafe {
        let fb_binder = BinderDrawFramebuffer::bind(&fbo_paper);

        let rb_binder = BinderRenderbuffer::bind(&rbo_paper_color);
        gl.renderbuffer_storage(rb_binder.target(), glow::RGBA8, 1, 1);
        gl.framebuffer_renderbuffer(
            fb_binder.target(),
            glow::COLOR_ATTACHMENT0,
            glow::RENDERBUFFER,
            Some(rbo_paper_color.id()),
        );

        rb_binder.rebind(&rbo_paper_stencil);
        gl.renderbuffer_storage(rb_binder.target(), glow::STENCIL_INDEX, 1, 1);
        gl.framebuffer_renderbuffer(
            fb_binder.target(),
            glow::STENCIL_ATTACHMENT,
            glow::RENDERBUFFER,
            Some(rbo_paper_stencil.id()),
        );
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
        prg_text,
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
    prg_text: glr::Program,
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
    fn title(&self) -> String {
        match self {
            FileAction::OpenCraft | FileAction::OpenCraftReadOnly => tr!("Opening..."),
            FileAction::SaveAsCraft => tr!("Saving..."),
            FileAction::ImportModel => tr!("Importing..."),
            FileAction::UpdateObj => tr!("Updating..."),
            FileAction::ExportObj => tr!("Exporting..."),
            FileAction::GeneratePrintable => tr!("Generating..."),
        }
    }
    fn is_save(&self) -> bool {
        match self {
            FileAction::OpenCraft
            | FileAction::OpenCraftReadOnly
            | FileAction::ImportModel
            | FileAction::UpdateObj => false,
            FileAction::SaveAsCraft | FileAction::ExportObj | FileAction::GeneratePrintable => true,
        }
    }
}

struct ConfirmableAction {
    title: String,
    message: String,
    action: Box<dyn Fn(&mut MenuActions)>,
}

struct GlobalContext {
    config: config::Config,
    gl: GlContext,
    gl_fixs: GLFixedObjects,
    font_default: imgui::FontId,
    font_big: imgui::FontId,
    font_small: imgui::FontId,
    font_text: imgui::FontId,
    font_text_line_scale: f32,
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
    config_opened: Option<config::Config>,
    config_applied: Option<config::Config>,
    options_opened: Option<PaperOptions>,
    // the .1 is true if the Options was accepted,
    // false, when doing an "Undo".
    options_applied: Option<(PaperOptions, bool)>,
    option_button_height: f32,
    about_visible: bool,
    file_dialog: Option<FileDialog>,
    filechooser_atlas: filechooser::CustomAtlas,
    file_action: Option<(FileAction, PathBuf)>,
    last_path: PathBuf,
    last_export: PathBuf,
    error_message: Option<String>,
    confirmable_action: Option<ConfirmableAction>,
    popup_time_start: Instant,
    cmd_file_action: Option<(FileAction, PathBuf)>,
    quit_requested: BoolWithConfirm,
    title: String,
}

struct FileDialog {
    chooser: filechooser::FileChooser,
    title: String,
    action: FileAction,
    confirm: Option<PathBuf>,
}

impl FileDialog {
    fn new(chooser: filechooser::FileChooser, title: String, action: FileAction) -> FileDialog {
        FileDialog {
            chooser,
            title,
            action,
            confirm: None,
        }
    }
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

// Returns `Some(true)` if "OK", `Some(false)`, if "Cancel" or not opened, `None` if opened.
fn do_modal_dialog(
    ui: &Ui,
    title: &str,
    id: &str,
    message: &str,
    cancel: Option<&str>,
    ok: Option<&str>,
) -> Option<bool> {
    let mut output = None;
    let mut opened = true;
    ui.popup_modal_config(lbl_id(title, id))
        .opened(Some(&mut opened))
        .flags(imgui::WindowFlags::NoResize | imgui::WindowFlags::AlwaysAutoResize)
        .with(|| {
            let font_sz = ui.get_font_size();
            ui.text(message);

            ui.separator();

            if let Some(cancel) = cancel {
                if ui
                    .button_config(lbl_id(cancel, "cancel"))
                    .size(vec2(font_sz * 5.5, 0.0))
                    .build()
                    | ui.shortcut(imgui::Key::Escape)
                {
                    ui.close_current_popup();
                    output = Some(false);
                }
            }
            if let Some(ok) = ok {
                if cancel.is_some() {
                    ui.same_line();
                }
                if ui
                    .button_config(lbl_id(ok, "ok"))
                    .size(vec2(font_sz * 5.5, 0.0))
                    .build()
                    | ui.shortcut(imgui::Key::Enter)
                    | ui.shortcut(imgui::Key::KeypadEnter)
                {
                    ui.close_current_popup();
                    output = Some(true);
                }
            }
        });
    // Close button has been pressed
    if !opened && output.is_none() {
        output = Some(false);
    }
    output
}

#[allow(clippy::collapsible_if)]
impl GlobalContext {
    fn modifiable(&self) -> bool {
        self.data.ui.mode != MouseMode::ReadOnly
    }
    fn build_modal_error_message(&mut self, ui: &Ui) {
        if let Some(error_message) = self.error_message.take() {
            let reply = do_modal_dialog(
                ui,
                &tr!("Error"),
                "error",
                &error_message,
                None,
                Some(&tr!("OK")),
            );
            if reply.is_none() {
                self.error_message = Some(error_message);
            }
        }
    }
    fn build_confirm_message(&mut self, ui: &Ui, menu_actions: &mut MenuActions) {
        if let Some(action) = self.confirmable_action.take() {
            let reply = do_modal_dialog(
                ui,
                &action.title,
                "Confirm",
                &action.message,
                Some(&tr!("Cancel")),
                Some(&tr!("Continue")),
            );
            match reply {
                Some(true) => (action.action)(menu_actions),
                Some(false) => (),
                None => self.confirmable_action = Some(action),
            }
        }
    }

    fn build_about(&mut self, ui: &Ui) {
        if !self.about_visible {
            return;
        }
        ui.window_config(lbl_id(tr!("About..."), "about"))
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
                ui.with_push(self.font_big, || {
                    center_text(ui, &tr!("Papercraft"), sz_full.x);
                });

                advance_cursor(ui, 0.0, 1.0);
                center_text(ui, &tr!("Version {}", env!("CARGO_PKG_VERSION")), sz_full.x);
                advance_cursor(ui, 0.0, 1.0);
                center_text(ui, env!("CARGO_PKG_DESCRIPTION"), sz_full.x);
                advance_cursor(ui, 0.0, 0.5);
                center_url(ui, env!("CARGO_PKG_REPOSITORY"), "url", None, sz_full.x);
                advance_cursor(ui, 0.0, 0.5);
                // Keep the legal text untranslated
                ui.with_push(self.font_small, || {
                    center_text(ui, "© Copyright 2024 - Rodrigo Rivas Costa", sz_full.x);
                    center_text(
                        ui,
                        "This program comes with absolutely no warranty.",
                        sz_full.x,
                    );
                    center_url(
                        ui,
                        "See the GNU General Public License, version 3 or later for details.",
                        "gpl3",
                        Some("https://www.gnu.org/licenses/gpl-3.0.html"),
                        sz_full.x,
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
            ui.popup_modal_config(lbl_id(title, "Wait"))
                .flags(imgui::WindowFlags::NoResize)
                .with(|| {
                    ui.text(&tr!("Please, wait..."));

                    // Give time to the fading modal, because the action is blocking and will
                    // freeze the UI for a while. This should be enough.
                    let run = self.popup_time_start.elapsed() > Duration::from_millis(250);
                    if run {
                        res = Some(self.run_file_action(ui, *action, file));
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
                    ui.open_popup(id("error"));
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
                    (
                        imgui::StyleVar::WindowPadding,
                        imgui::StyleValue::Vec2(vec2(pad, pad)),
                    ),
                    (
                        imgui::StyleVar::ItemSpacing,
                        imgui::StyleValue::Vec2(vec2(0.0, 0.0)),
                    ),
                ),
                || {
                    ui.child_config(lbl("toolbar"))
                        .child_flags(
                            imgui::ChildFlags::AlwaysUseWindowPadding
                                | imgui::ChildFlags::AutoResizeY,
                        )
                        .window_flags(imgui::WindowFlags::NoScrollbar)
                        .with(|| {
                            ui.with_push(
                                (
                                    imgui::StyleVar::ItemSpacing,
                                    imgui::StyleValue::Vec2(vec2(font_sz / 8.0, 0.0)),
                                ),
                                || {
                                    let color_active =
                                        ui.style().color(imgui::ColorId::ButtonActive);
                                    if ui
                                        .image_button_with_custom_rect_config(
                                            id("Face"),
                                            self.icons_rect[0],
                                            1.0,
                                        )
                                        .bg_col(if self.data.ui.mode == MouseMode::Face {
                                            color_active
                                        } else {
                                            imgui::Color::TRANSPARENT
                                        })
                                        .build()
                                    {
                                        self.set_mouse_mode(MouseMode::Face);
                                    }
                                    ui.same_line();
                                    if ui
                                        .image_button_with_custom_rect_config(
                                            id("Edge"),
                                            self.icons_rect[1],
                                            1.0,
                                        )
                                        .bg_col(if self.data.ui.mode == MouseMode::Edge {
                                            color_active
                                        } else {
                                            imgui::Color::TRANSPARENT
                                        })
                                        .build()
                                    {
                                        self.set_mouse_mode(MouseMode::Edge);
                                    }
                                    ui.same_line();
                                    if ui
                                        .image_button_with_custom_rect_config(
                                            id("Tab"),
                                            self.icons_rect[2],
                                            1.0,
                                        )
                                        .bg_col(if self.data.ui.mode == MouseMode::Flap {
                                            color_active
                                        } else {
                                            imgui::Color::TRANSPARENT
                                        })
                                        .build()
                                    {
                                        self.set_mouse_mode(MouseMode::Flap);
                                    }
                                },
                            );
                        });
                },
            );
        }

        let color_hovered = ui.style().color(imgui::ColorId::ButtonHovered);
        ui.with_push(
            (
                (
                    imgui::StyleVar::ItemSpacing,
                    imgui::StyleValue::Vec2(vec2(2.0, 2.0)),
                ),
                (
                    imgui::StyleVar::WindowPadding,
                    imgui::StyleValue::Vec2(vec2(0.0, 0.0)),
                ),
                (imgui::ColorId::ButtonActive, color_hovered),
                (imgui::ColorId::Button, color_hovered),
            ),
            || {
                let sz = vec2(0.0, -ui.get_frame_height());
                ui.child_config(lbl("main_area")).size(sz).with(|| {
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

                    ui.button_config(lbl_id("", "vsplitter"))
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

                        self.data.ui.trans_scene.persp =
                            cgmath::perspective(Deg(60.0), sz_scene.x / sz_scene.y, 1.0, 100.0);
                        self.data.ui.trans_scene.persp_inv =
                            self.data.ui.trans_scene.persp.invert().unwrap();

                        let (x, y) = (sz_scene.x as i32, sz_scene.y as i32);

                        unsafe {
                            let rb_binder = BinderRenderbuffer::bind(&self.gl_fixs.rbo_scene_color);

                            'no_aa: {
                                for samples in MULTISAMPLES {
                                    self.gl.get_error(); //clear error
                                    rb_binder.rebind(&self.gl_fixs.rbo_scene_color);
                                    self.gl.renderbuffer_storage_multisample(
                                        rb_binder.target(),
                                        *samples,
                                        glow::RGBA8,
                                        x,
                                        y,
                                    );
                                    rb_binder.rebind(&self.gl_fixs.rbo_scene_depth);
                                    self.gl.renderbuffer_storage_multisample(
                                        rb_binder.target(),
                                        *samples,
                                        glow::DEPTH_COMPONENT,
                                        x,
                                        y,
                                    );

                                    if self.gl.get_error() != 0
                                        || self.gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER)
                                            != glow::FRAMEBUFFER_COMPLETE
                                    {
                                        continue;
                                    }
                                    break 'no_aa;
                                }

                                rb_binder.rebind(&self.gl_fixs.rbo_scene_color);
                                self.gl
                                    .renderbuffer_storage(rb_binder.target(), glow::RGBA8, x, y);
                                rb_binder.rebind(&self.gl_fixs.rbo_scene_depth);
                                self.gl.renderbuffer_storage(
                                    rb_binder.target(),
                                    glow::DEPTH_COMPONENT,
                                    x,
                                    y,
                                );
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
                                    self.gl.renderbuffer_storage_multisample(
                                        rb_binder.target(),
                                        *samples,
                                        glow::RGBA8,
                                        x,
                                        y,
                                    );
                                    rb_binder.rebind(&self.gl_fixs.rbo_paper_stencil);
                                    self.gl.renderbuffer_storage_multisample(
                                        rb_binder.target(),
                                        *samples,
                                        glow::STENCIL_INDEX,
                                        x,
                                        y,
                                    );

                                    if self.gl.get_error() != 0
                                        || self.gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER)
                                            != glow::FRAMEBUFFER_COMPLETE
                                    {
                                        continue;
                                    }
                                    break 'no_aa;
                                }

                                rb_binder.rebind(&self.gl_fixs.rbo_paper_color);
                                self.gl
                                    .renderbuffer_storage(rb_binder.target(), glow::RGBA8, x, y);
                                rb_binder.rebind(&self.gl_fixs.rbo_paper_stencil);
                                self.gl.renderbuffer_storage(
                                    rb_binder.target(),
                                    glow::STENCIL_INDEX,
                                    x,
                                    y,
                                );
                            }
                        }
                    }
                });
            },
        );
        advance_cursor(ui, 0.25, 0.0);

        let status_text = match self.data.ui.mode {
            MouseMode::Face => tr!("Face mode. Click to select a piece. Drag on paper to move it. Shift-drag on paper to rotate it."),
            MouseMode::Edge => tr!("Edge mode. Click on an edge to split/join pieces. Shift-click to join a full strip of quads."),
            MouseMode::Flap => tr!("Flap mode. Click on an edge to swap the side of a flap. Shift-click to hide a flap."),
            MouseMode::ReadOnly => tr!("View mode. Click to highlight a piece. Move the mouse over an edge to highlight the matching pair."),
        };
        ui.text(&status_text);

        self.build_config_dialog(ui);
        self.build_options_dialog(ui);
        self.build_modal_error_message(ui);
        self.build_modal_wait_message_and_run_file_action(ui);
        self.build_confirm_message(ui, &mut menu_actions);
        self.build_about(ui);

        menu_actions
    }

    fn build_config_dialog(&mut self, ui: &Ui) {
        let Some(config) = self.config_opened.as_mut() else {
            return;
        };
        let mut opened = true;
        let mut applied = false;
        ui.set_next_window_size(vec2(400.0, 200.0), imgui::Cond::Once);
        ui.window_config(lbl_id(tr!("Settings"), "settings"))
            .open(&mut opened)
            .flags(imgui::WindowFlags::NoScrollbar | imgui::WindowFlags::AlwaysAutoResize)
            .with(|| {
                let mut cur = LANGUAGES
                    .iter()
                    .find(|(s, _)| *s == config.locale)
                    .unwrap_or(&LANGUAGES[0]);
                if ui.combo(
                    lbl_id(tr!("Language"), "language"),
                    LANGUAGES,
                    |(_, n)| *n,
                    &mut cur,
                ) {
                    config.locale = String::from(cur.0);
                    applied = true;
                }
                if ui.combo(
                    lbl_id(tr!("Theme"), "theme"),
                    [false, true],
                    |b| {
                        if b {
                            tr!("Theme" => "Light")
                        } else {
                            tr!("Theme" => "Dark")
                        }
                    },
                    &mut config.light_mode,
                ) {
                    applied = true;
                }
            });
        if applied {
            self.config_applied = Some(config.clone());
        }
        if !opened {
            self.config_opened = None;
        }
    }
    fn build_options_dialog(&mut self, ui: &Ui) {
        let options = match self.options_opened.take() {
            Some(o) => o,
            None => return,
        };
        let modifiable = self.modifiable();
        let mut options_opened = true;
        ui.set_next_window_size(
            if modifiable {
                vec2(650.0, 400.0)
            } else {
                vec2(300.0, 100.0)
            },
            imgui::Cond::Once,
        );
        ui.window_config(lbl_id(tr!("Document properties"), "options"))
            .open(&mut options_opened)
            .flags(imgui::WindowFlags::NoScrollbar)
            .with_always(|opened| {
                if opened {
                    if modifiable {
                        let (keep_opened, apply) =
                            self.build_full_options_inner_dialog(ui, options);
                        self.options_opened = keep_opened;
                        if let Some(apply_options) = apply {
                            // Don't apply the options immediately because we are in the middle of a render,
                            // and that could cause inconsistencies
                            self.options_applied = Some((apply_options, true));
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
        let n_flaps = self
            .data
            .papercraft()
            .model()
            .edges()
            .filter(|(e, _)| {
                matches!(
                    self.data.papercraft().edge_status(*e),
                    paper::EdgeStatus::Cut(_)
                )
            })
            .count();
        let bbox = util_3d::bounding_box_3d(
            self.data
                .papercraft()
                .model()
                .vertices()
                .map(|(_, v)| v.pos()),
        );
        let model_size = (bbox.1 - bbox.0) * options.scale;
        let Vector3 { x, y, z } = model_size;
        let size = format!("{x:.0} x {y:.0} x {z:.0}");
        ui.text(&tr!(
            "Number of pieces: {0}\nNumber of flaps: {1}\nReal size (mm): {2}",
            n_pieces,
            n_flaps,
            size
        ));
    }

    fn build_full_options_inner_dialog(
        &mut self,
        ui: &Ui,
        mut options: PaperOptions,
    ) -> (Option<PaperOptions>, Option<PaperOptions>) {
        let size = Vector2::from(ui.get_content_region_avail());
        let font_sz = ui.get_font_size();
        ui.child_config(lbl("options"))
            .size(vec2(size.x, -self.option_button_height))
            .window_flags(imgui::WindowFlags::HorizontalScrollbar)
            .with(|| {
                ui.tree_node_config(lbl_id(tr!("Model"), "model"))
                    .flags(imgui::TreeNodeFlags::Framed)
                    .with(|| {
                        ui.set_next_item_width(font_sz * 5.5);
                        ui.input_float_config(lbl_id(tr!("Scale"), "scale"), &mut options.scale)
                            .display_format(imgui::FloatFormat::G)
                            .build();
                        options.scale = options.scale.max(0.0);
                        ui.same_line_ex(0.0, font_sz * 3.0);
                        ui.with_disabled(!self.data.papercraft().model().has_textures(), || {
                            ui.checkbox(lbl_id(tr!("Textured"), "textured"), &mut options.texture);
                            ui.same_line_ex(0.0, font_sz * 3.0);
                            ui.checkbox(
                                lbl_id(tr!("Texture filter"), "tex_filter"),
                                &mut options.tex_filter,
                            );
                        });

                        ui.tree_node_config(lbl_id(tr!("Flaps"), "flaps")).with(|| {
                            static FLAP_STYLES: &[FlapStyle] = &[
                                FlapStyle::Textured,
                                FlapStyle::HalfTextured,
                                FlapStyle::White,
                                FlapStyle::None,
                            ];
                            fn fmt_flap_style(s: FlapStyle) -> String {
                                match s {
                                    FlapStyle::Textured => tr!("FlapStyle" => "Textured"),
                                    FlapStyle::HalfTextured => tr!("FlapStyle" => "Half textured"),
                                    FlapStyle::White => tr!("FlapStyle" => "White"),
                                    FlapStyle::None => tr!("FlapStyle" => "None"),
                                }
                            }

                            ui.set_next_item_width(font_sz * 8.0);
                            ui.combo(
                                lbl_id(tr!("Style"), "style"),
                                FLAP_STYLES.iter().copied(),
                                fmt_flap_style,
                                &mut options.flap_style,
                            );

                            ui.same_line_ex(font_sz * 12.0, font_sz * 1.5);
                            ui.set_next_item_width(font_sz * 8.0);
                            ui.slider_float_config(
                                lbl_id(tr!("Shadow"), "shadow"),
                                &mut options.shadow_flap_alpha,
                            )
                            .range(0.0, 1.0)
                            .display_format(imgui::FloatFormat::F(2))
                            .build();

                            ui.set_next_item_width(font_sz * 8.0);
                            ui.input_float_config(
                                lbl_id(tr!("Width"), "width"),
                                &mut options.flap_width,
                            )
                            .display_format(imgui::FloatFormat::G)
                            .build();
                            options.flap_width = options.flap_width.max(0.0);

                            ui.same_line_ex(font_sz * 12.0, font_sz * 1.5);
                            ui.set_next_item_width(font_sz * 8.0);
                            ui.input_float_config(
                                lbl_id(tr!("Angle"), "angle"),
                                &mut options.flap_angle,
                            )
                            .display_format(imgui::FloatFormat::G)
                            .build();
                            options.flap_angle = options.flap_angle.clamp(0.0, 180.0);
                        });
                        ui.tree_node_config(lbl_id(tr!("Folds"), "folds")).with(|| {
                            static FOLD_STYLES: &[FoldStyle] = &[
                                FoldStyle::Full,
                                FoldStyle::FullAndOut,
                                FoldStyle::Out,
                                FoldStyle::In,
                                FoldStyle::InAndOut,
                                FoldStyle::None,
                            ];
                            fn fmt_fold_style(s: FoldStyle) -> String {
                                match s {
                                    FoldStyle::Full => tr!("FoldStyle" => "Full line"),
                                    FoldStyle::FullAndOut => {
                                        tr!("FoldStyle" => "Full & out segment")
                                    }
                                    FoldStyle::Out => tr!("FoldStyle" => "Out segment"),
                                    FoldStyle::In => tr!("FoldStyle" => "In segment"),
                                    FoldStyle::InAndOut => tr!("FoldStyle" => "Out & in segment"),
                                    FoldStyle::None => tr!("FoldStyle" => "None"),
                                }
                            }

                            ui.set_next_item_width(font_sz * 8.0);
                            ui.combo(
                                lbl_id(tr!("Style"), "style"),
                                FOLD_STYLES.iter().copied(),
                                fmt_fold_style,
                                &mut options.fold_style,
                            );

                            ui.same_line_ex(0.0, font_sz * 1.5);
                            ui.set_next_item_width(font_sz * 5.5);
                            ui.with_disabled(
                                matches!(options.fold_style, FoldStyle::None | FoldStyle::Full),
                                || {
                                    ui.input_float_config(
                                        lbl_id(tr!("Length"), "length"),
                                        &mut options.fold_line_len,
                                    )
                                    .display_format(imgui::FloatFormat::G)
                                    .build();
                                    options.fold_line_len = options.fold_line_len.max(0.0);
                                },
                            );
                            ui.same_line_ex(0.0, font_sz * 1.5);
                            ui.set_next_item_width(font_sz * 5.5);
                            ui.with_disabled(matches!(options.fold_style, FoldStyle::None), || {
                                ui.input_float_config(
                                    lbl_id(tr!("Line width"), "linewidth"),
                                    &mut options.fold_line_width,
                                )
                                .display_format(imgui::FloatFormat::G)
                                .build();
                                options.fold_line_width = options.fold_line_width.max(0.0);
                            });

                            ui.set_next_item_width(font_sz * 5.5);
                            ui.input_float_config(
                                lbl_id(tr!("Hidden fold angle"), "hiddenangle"),
                                &mut options.hidden_line_angle,
                            )
                            .display_format(imgui::FloatFormat::G)
                            .build();
                            options.hidden_line_angle = options.hidden_line_angle.clamp(0.0, 180.0);
                        });
                        ui.tree_node_config(lbl_id(tr!("Information"), "info"))
                            .with(|| {
                                self.build_read_only_options_inner_dialog(ui, &options);
                            });
                    });
                ui.tree_node_config(lbl_id(tr!("Layout"), "layout"))
                    .flags(imgui::TreeNodeFlags::Framed)
                    .with(|| {
                        ui.set_next_item_width(font_sz * 5.5);

                        let mut i = options.pages as _;
                        ui.input_int_config(lbl_id(tr!("Pages"), "pages"), &mut i)
                            .build();
                        options.pages = i.clamp(1, 1000) as _;

                        ui.same_line_ex(0.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 5.5);

                        let mut i = options.page_cols as _;
                        ui.input_int_config(lbl_id(tr!("Columns"), "cols"), &mut i)
                            .build();
                        options.page_cols = i.clamp(1, 1000) as _;

                        ui.set_next_item_width(font_sz * 11.0);
                        ui.checkbox(
                            lbl_id(tr!("Print Papercraft signature"), "signature"),
                            &mut options.show_self_promotion,
                        );

                        ui.same_line_ex(0.0, font_sz * 3.0);
                        ui.set_next_item_width(font_sz * 11.0);
                        ui.checkbox(
                            lbl_id(tr!("Print page number"), "page_num"),
                            &mut options.show_page_number,
                        );

                        static EDGE_ID_POSITIONS: &[EdgeIdPosition] = &[
                            EdgeIdPosition::None,
                            EdgeIdPosition::Outside,
                            EdgeIdPosition::Inside,
                        ];
                        fn fmt_edge_id_position(s: EdgeIdPosition) -> String {
                            match s {
                                EdgeIdPosition::None => tr!("EdgeIdPos" => "None"),
                                EdgeIdPosition::Outside => tr!("EdgeIdPos" => "Outside"),
                                EdgeIdPosition::Inside => tr!("EdgeIdPos" => "Inside"),
                            }
                        }
                        ui.set_next_item_width(font_sz * 6.0);
                        ui.combo(
                            lbl_id(tr!("Edge id position"), "edge_id_pos"),
                            EDGE_ID_POSITIONS.iter().copied(),
                            fmt_edge_id_position,
                            &mut options.edge_id_position,
                        );

                        ui.same_line_ex(0.0, font_sz * 1.5);

                        ui.set_next_item_width(font_sz * 3.0);
                        ui.with_disabled(options.edge_id_position == EdgeIdPosition::None, || {
                            ui.input_float_config(
                                lbl_id(tr!("Edge id font size (pt)"), "edgefont"),
                                &mut options.edge_id_font_size,
                            )
                            .display_format(imgui::FloatFormat::G)
                            .build();
                            options.edge_id_font_size = options.edge_id_font_size.clamp(1.0, 72.0);
                        });
                    });
                ui.tree_node_config(lbl_id(tr!("Paper size"), "papersize"))
                    .flags(imgui::TreeNodeFlags::Framed)
                    .with(|| {
                        ui.set_next_item_width(font_sz * 5.5);
                        ui.input_float_config(
                            lbl_id(tr!("Width"), "width"),
                            &mut options.page_size.0,
                        )
                        .display_format(imgui::FloatFormat::G)
                        .build();
                        options.page_size.0 = options.page_size.0.max(1.0);
                        ui.same_line_ex(0.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 5.5);
                        ui.input_float_config(
                            lbl_id(tr!("Height"), "height"),
                            &mut options.page_size.1,
                        )
                        .display_format(imgui::FloatFormat::G)
                        .build();
                        options.page_size.1 = options.page_size.1.max(1.0);
                        ui.same_line_ex(0.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 5.5);
                        let mut resolution = options.resolution as f32;
                        ui.input_float_config(lbl_id(tr!("DPI"), "dpi"), &mut resolution)
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
                        let paper_size = PAPER_SIZES.iter().find(|s| {
                            s.size == paper_size || s.size == vec2(paper_size.y, paper_size.x)
                        });
                        ui.set_next_item_width(font_sz * 8.0);
                        ui.combo_config(lbl_id("", "Paper"))
                            .preview_value_opt(paper_size.map(|p| p.name))
                            .with(|| {
                                for op in PAPER_SIZES {
                                    if ui
                                        .selectable_config(lbl(op.name))
                                        .selected(
                                            paper_size
                                                .map(|p| std::ptr::eq(p, op))
                                                .unwrap_or(false),
                                        )
                                        .build()
                                    {
                                        options.page_size = (op.size.x, op.size.y);
                                        if !portrait {
                                            std::mem::swap(
                                                &mut options.page_size.0,
                                                &mut options.page_size.1,
                                            );
                                        }
                                    }
                                }
                            });
                        let mut new_portrait = portrait;
                        if ui
                            .radio_button_config(lbl_id(tr!("Portrait"), "portrait"), portrait)
                            .build()
                        {
                            new_portrait = true;
                        }
                        if ui
                            .radio_button_config(lbl_id(tr!("Landscape"), "landscape"), !portrait)
                            .build()
                        {
                            new_portrait = false;
                        }
                        if portrait != new_portrait {
                            std::mem::swap(&mut options.page_size.0, &mut options.page_size.1);
                        }
                    });
                ui.tree_node_config(lbl_id(tr!("Margins"), "margins"))
                    .flags(imgui::TreeNodeFlags::Framed)
                    .with(|| {
                        ui.set_next_item_width(font_sz * 4.0);
                        ui.input_float_config(lbl_id(tr!("Top"), "top"), &mut options.margin.0)
                            .display_format(imgui::FloatFormat::G)
                            .build();
                        ui.same_line_ex(0.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 4.0);
                        ui.input_float_config(lbl_id(tr!("Left"), "left"), &mut options.margin.1)
                            .display_format(imgui::FloatFormat::G)
                            .build();
                        ui.same_line_ex(0.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 4.0);
                        ui.input_float_config(lbl_id(tr!("Right"), "right"), &mut options.margin.2)
                            .display_format(imgui::FloatFormat::G)
                            .build();
                        ui.same_line_ex(0.0, font_sz * 1.5);
                        ui.set_next_item_width(font_sz * 4.0);
                        ui.input_float_config(
                            lbl_id(tr!("Bottom"), "bottom"),
                            &mut options.margin.3,
                        )
                        .display_format(imgui::FloatFormat::G)
                        .build();
                    });
            });

        let mut options_opened = Some(options);
        let mut apply_options = None;

        let pos1 = Vector2::from(ui.get_cursor_screen_pos());
        ui.separator();
        if ui
            .button_config(lbl_id(tr!("OK"), "ok"))
            .size(vec2(100.0, 0.0))
            .build()
        {
            apply_options = options_opened.take();
        }
        ui.same_line();
        if ui
            .button_config(lbl_id(tr!("Cancel"), "cancel"))
            .size(vec2(100.0, 0.0))
            .build()
        {
            options_opened = None;
        }
        ui.same_line();
        if ui
            .button_config(lbl_id(tr!("Apply"), "apply"))
            .size(vec2(100.0, 0.0))
            .build()
        {
            apply_options.clone_from(&options_opened);
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
            ui.menu_config(lbl(tr!("File"))).with(|| {
                if ui
                    .menu_item_config(lbl(tr!("Open...")))
                    .shortcut("Ctrl+O")
                    .build()
                {
                    menu_actions.open = self.check_modified();
                }
                ui.with_disabled(self.data.papercraft().model().is_empty(), || {
                    if ui
                        .menu_item_config(lbl(tr!("Save")))
                        .shortcut("Ctrl+S")
                        .build()
                    {
                        menu_actions.save = true;
                    }
                    if ui.menu_item_config(lbl(tr!("Save as..."))).build() {
                        menu_actions.save_as = true;
                    }
                });
                if self.modifiable() {
                    if ui.menu_item_config(lbl(tr!("Import model..."))).build() {
                        menu_actions.import_model = self.check_modified();
                    }
                    if ui
                        .menu_item_config(lbl(tr!("Update with new model...")))
                        .build()
                    {
                        menu_actions.update_model = self.check_modified();
                    }
                }
                if ui.menu_item_config(lbl(tr!("Export model..."))).build() {
                    menu_actions.export_obj = true;
                }
                if ui
                    .menu_item_config(lbl(tr!("Generate Printable...")))
                    .build()
                {
                    menu_actions.generate_printable = true;
                }
                ui.separator();
                if ui
                    .menu_item_config(lbl(tr!("Settings...")))
                    .selected(self.config_opened.is_some())
                    .build()
                {
                    self.config_opened = match self.config_opened {
                        Some(_) => None,
                        None => Some(self.config.clone()),
                    };
                }
                ui.separator();
                if ui
                    .menu_item_config(lbl(tr!("Quit")))
                    .shortcut("Ctrl+Q")
                    .build()
                {
                    menu_actions.quit = self.check_modified();
                }
            });
            ui.menu_config(lbl(tr!("Edit"))).with(|| {
                if self.modifiable() {
                    if ui
                        .menu_item_config(lbl(tr!("Undo")))
                        .shortcut("Ctrl+Z")
                        .enabled(self.data.can_undo())
                        .build()
                    {
                        menu_actions.undo = true;
                    }
                    ui.separator();
                }

                if ui
                    .menu_item_config(lbl(tr!("Document properties")))
                    .shortcut("Enter")
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

                    if ui
                        .menu_item_config(lbl(tr!("Face/Island")))
                        .shortcut("F5")
                        .selected(self.data.ui.mode == MouseMode::Face)
                        .build()
                    {
                        self.set_mouse_mode(MouseMode::Face);
                    }
                    if ui
                        .menu_item_config(lbl(tr!("Split/Join edge")))
                        .shortcut("F6")
                        .selected(self.data.ui.mode == MouseMode::Edge)
                        .build()
                    {
                        self.set_mouse_mode(MouseMode::Edge);
                    }
                    if ui
                        .menu_item_config(lbl(tr!("Flaps")))
                        .shortcut("F7")
                        .selected(self.data.ui.mode == MouseMode::Flap)
                        .build()
                    {
                        self.set_mouse_mode(MouseMode::Flap);
                    }

                    ui.separator();

                    if ui.menu_item_config(lbl(tr!("Repack pieces"))).build() {
                        let undo = self.data.pack_islands();
                        self.data.push_undo_action(undo);
                        self.add_rebuild(RebuildFlags::PAPER | RebuildFlags::SELECTION);
                    }
                }
            });
            ui.menu_config(lbl(tr!("View"))).with(|| {
                if ui
                    .menu_item_config(lbl(tr!("Textures")))
                    .shortcut("T")
                    .enabled(self.data.papercraft().options().texture)
                    .selected(self.data.ui.show_textures)
                    .build()
                {
                    self.data.ui.show_textures ^= true;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
                }
                if ui
                    .menu_item_config(lbl(tr!("3D lines")))
                    .shortcut("D")
                    .selected(self.data.ui.show_3d_lines)
                    .build()
                {
                    self.data.ui.show_3d_lines ^= true;
                    self.add_rebuild(RebuildFlags::SCENE_REDRAW);
                }
                if ui
                    .menu_item_config(lbl(tr!("Flaps")))
                    .shortcut("B")
                    .selected(self.data.ui.show_flaps)
                    .build()
                {
                    self.data.ui.show_flaps ^= true;
                    self.add_rebuild(RebuildFlags::PAPER);
                }
                if ui
                    .menu_item_config(lbl(tr!("X-ray selection")))
                    .shortcut("X")
                    .selected(self.data.ui.xray_selection)
                    .build()
                {
                    self.data.ui.xray_selection ^= true;
                    self.add_rebuild(RebuildFlags::SELECTION);
                }
                if ui
                    .menu_item_config(lbl(tr!("Texts")))
                    .shortcut("E")
                    .selected(self.data.ui.show_texts)
                    .build()
                {
                    self.data.ui.show_texts ^= true;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW);
                    if self.data.ui.show_texts {
                        self.add_rebuild(RebuildFlags::ISLANDS | RebuildFlags::PAPER);
                    }
                }
                if ui
                    .menu_item_config(lbl(tr!("Paper")))
                    .shortcut("P")
                    .selected(self.data.ui.draw_paper)
                    .build()
                {
                    self.data.ui.draw_paper ^= true;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW);
                }
                if ui
                    .menu_item_config(lbl(tr!("Highlight overlaps")))
                    .shortcut("H")
                    .selected(self.data.ui.highlight_overlaps)
                    .build()
                {
                    self.data.ui.highlight_overlaps ^= true;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW);
                }
                if ui.menu_item_config(lbl(tr!("Reset views"))).build() {
                    menu_actions.reset_views = true;
                    self.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
                }
            });
            ui.menu_config(lbl(tr!("Help"))).with(|| {
                if ui
                    .menu_item_config(lbl(tr!("About...")))
                    .selected(self.about_visible)
                    .build()
                {
                    self.about_visible ^= true;
                }
            });
        });

        // Modal pop-ups should disable the shortcuts
        if !ui.is_blocking_modal() {
            if self.modifiable() {
                if ui.shortcut_ex(imgui::Key::F5, imgui::InputFlags::RouteGlobal) {
                    self.set_mouse_mode(MouseMode::Face);
                }
                if ui.shortcut_ex(imgui::Key::F6, imgui::InputFlags::RouteGlobal) {
                    self.set_mouse_mode(MouseMode::Edge);
                }
                if ui.shortcut_ex(imgui::Key::F7, imgui::InputFlags::RouteGlobal) {
                    self.set_mouse_mode(MouseMode::Flap);
                }
                if ui.shortcut_ex(
                    (imgui::KeyMod::Ctrl, imgui::Key::Z),
                    imgui::InputFlags::RouteGlobal,
                ) {
                    menu_actions.undo = true;
                }
            }
            if ui.shortcut_ex(
                (imgui::KeyMod::Ctrl, imgui::Key::Q),
                imgui::InputFlags::RouteGlobal,
            ) {
                menu_actions.quit = self.check_modified();
            }
            if ui.shortcut_ex(
                (imgui::KeyMod::Ctrl, imgui::Key::O),
                imgui::InputFlags::RouteGlobal,
            ) {
                menu_actions.open = self.check_modified();
            }
            if ui.shortcut_ex(
                (imgui::KeyMod::Ctrl, imgui::Key::S),
                imgui::InputFlags::RouteGlobal,
            ) {
                menu_actions.save = true;
            }
            if ui.shortcut_ex(imgui::Key::X, imgui::InputFlags::RouteGlobal) {
                self.data.ui.xray_selection ^= true;
                self.add_rebuild(RebuildFlags::SELECTION);
            }
            if ui.shortcut_ex(imgui::Key::H, imgui::InputFlags::RouteGlobal) {
                self.data.ui.highlight_overlaps ^= true;
                self.add_rebuild(RebuildFlags::PAPER_REDRAW);
            }
            if ui.shortcut_ex(imgui::Key::P, imgui::InputFlags::RouteGlobal) {
                self.data.ui.draw_paper ^= true;
                self.add_rebuild(RebuildFlags::PAPER_REDRAW);
            }
            if self.data.papercraft().options().texture
                && ui.shortcut_ex(imgui::Key::T, imgui::InputFlags::RouteGlobal)
            {
                self.data.ui.show_textures ^= true;
                self.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
            }
            if ui.shortcut_ex(imgui::Key::D, imgui::InputFlags::RouteGlobal) {
                self.data.ui.show_3d_lines ^= true;
                self.add_rebuild(RebuildFlags::SCENE_REDRAW);
            }
            if ui.shortcut_ex(imgui::Key::B, imgui::InputFlags::RouteGlobal) {
                self.data.ui.show_flaps ^= true;
                self.add_rebuild(RebuildFlags::PAPER);
            }
            if ui.shortcut_ex(imgui::Key::E, imgui::InputFlags::RouteGlobal) {
                self.data.ui.show_texts ^= true;
                self.add_rebuild(RebuildFlags::PAPER_REDRAW);
                if self.data.ui.show_texts {
                    self.add_rebuild(RebuildFlags::ISLANDS | RebuildFlags::PAPER);
                }
            }
            if ui.shortcut_ex(imgui::Key::Enter, imgui::InputFlags::RouteGlobal) {
                match &self.options_opened {
                    None => {
                        self.options_opened = Some(self.data.papercraft().options().clone());
                    }
                    // Pressing enter closes the options only if nothing is changed, else you should press Ok or Cancel
                    Some(op) if op == self.data.papercraft().options() => {
                        self.options_opened = None;
                    }
                    _ => {}
                }
            }
        }
        menu_actions
    }

    fn build_scene(&mut self, ui: &Ui, width: f32) {
        let w = ui
            .child_config(lbl("scene"))
            .size(vec2(width, 0.0))
            .child_flags(imgui::ChildFlags::Borders)
            .with(|| {
                let scale = ui.display_scale();
                let pos = scale * ui.get_cursor_screen_pos();
                let dsp_size = scale * ui.display_size();

                canvas3d(ui, &mut self.scene_ui_status);

                let x = pos.x as i32;
                let y = (dsp_size.y - pos.y) as i32;
                let width = self.sz_scene.x as i32;
                let height = self.sz_scene.y as i32;

                ui.window_draw_list().add_callback({
                    move |this| {
                        unsafe {
                            // blit the FBO to the real FB

                            let _read_fb_binder =
                                BinderReadFramebuffer::bind(&this.gl_fixs.fbo_scene);
                            this.gl.blit_framebuffer(
                                0,
                                0,
                                width,
                                height,
                                x,
                                y - height,
                                x + width,
                                y,
                                glow::COLOR_BUFFER_BIT,
                                glow::NEAREST,
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
        let r = ui
            .child_config(lbl("paper"))
            .child_flags(imgui::ChildFlags::Borders)
            .with(|| {
                let scale = ui.display_scale();
                let pos = scale * ui.get_cursor_screen_pos();
                let dsp_size = scale * ui.display_size();

                canvas3d(ui, &mut self.paper_ui_status);

                let x = pos.x as i32;
                let y = (dsp_size.y - pos.y) as i32;
                let width = self.sz_paper.x as i32;
                let height = self.sz_paper.y as i32;

                let draw_list = ui.window_draw_list();
                draw_list.add_callback({
                    move |this| {
                        unsafe {
                            // blit the FBO to the real FB
                            let _read_fb_binder =
                                BinderReadFramebuffer::bind(&this.gl_fixs.fbo_paper);
                            this.gl.blit_framebuffer(
                                0,
                                0,
                                width,
                                height,
                                x,
                                y - height,
                                x + width,
                                y,
                                glow::COLOR_BUFFER_BIT,
                                glow::NEAREST,
                            );
                        }
                    }
                });
                if let Some(rect) = self.data.pre_selection_rectangle() {
                    if !rect.is_null() {
                        let sel_a = (pos
                            + self
                                .data
                                .ui
                                .trans_paper
                                .paper_unclick(self.sz_paper, rect.a))
                            / scale;
                        let sel_b = (pos
                            + self
                                .data
                                .ui
                                .trans_paper
                                .paper_unclick(self.sz_paper, rect.b))
                            / scale;
                        draw_list.add_rect_filled(
                            sel_a,
                            sel_b,
                            Color::new(0.25, 0.25, 0.25, 0.5),
                            4.0,
                            imgui::DrawFlags::empty(),
                        );
                        draw_list.add_rect(
                            sel_a,
                            sel_b,
                            Color::new(0.2, 0.2, 1.0, 1.0),
                            4.0,
                            imgui::DrawFlags::empty(),
                            2.0,
                        );
                    }
                }
            });
        if r.is_none() {
            self.paper_ui_status = Canvas3dStatus::default();
        }
    }

    fn open_confirmation_dialog(
        &mut self,
        ui: &Ui,
        title: &str,
        message: &str,
        f: impl Fn(&mut MenuActions) + 'static,
    ) {
        self.confirmable_action = Some(ConfirmableAction {
            title: title.to_owned(),
            message: message.to_owned(),
            action: Box::new(f),
        });
        ui.open_popup(id("Confirm"));
    }

    fn run_menu_actions(&mut self, ui: &Ui, menu_actions: &MenuActions) {
        if menu_actions.reset_views {
            self.data.reset_views(self.sz_scene, self.sz_paper);
        }
        if menu_actions.undo {
            match self.data.undo_action() {
                UndoResult::Model => {
                    self.add_rebuild(RebuildFlags::all());
                }
                UndoResult::ModelAndOptions(options) => {
                    // If the "Options" window is opened, just overwrite the values
                    if let Some(o) = self.options_opened.as_mut() {
                        *o = self.data.papercraft().options().clone();
                    }
                    self.options_applied = Some((options, false));
                }
                UndoResult::False => {}
            }
        }

        let mut save_as = false;
        let mut open_file_dialog = false;
        let mut open_wait = false;

        match menu_actions.open {
            BoolWithConfirm::Requested => {
                self.open_confirmation_dialog(
                    ui,
                    &tr!("Load model"),
                    &tr!("The model has not been save, continue anyway?"),
                    |a| a.open = BoolWithConfirm::Confirmed,
                );
            }
            BoolWithConfirm::Confirmed => {
                let mut chooser = filechooser::FileChooser::new();
                let _ = chooser.set_path(&self.last_path);
                chooser.add_flags(filechooser::Flags::SHOW_READ_ONLY);
                chooser.add_filter(filters::craft());
                chooser.add_filter(filters::all_files());
                self.file_dialog = Some(FileDialog::new(
                    chooser,
                    tr!("Open..."),
                    FileAction::OpenCraft,
                ));
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
            let mut chooser = filechooser::FileChooser::new();
            let _ = chooser.set_path(&self.last_path);
            chooser.add_filter(filters::craft());
            chooser.add_filter(filters::all_files());
            self.file_dialog = Some(FileDialog::new(
                chooser,
                tr!("Save as..."),
                FileAction::SaveAsCraft,
            ));
            open_file_dialog = true;
        }
        match menu_actions.import_model {
            BoolWithConfirm::Requested => {
                self.open_confirmation_dialog(
                    ui,
                    &tr!("Import model"),
                    &tr!("The model has not been save, continue anyway?"),
                    |a| a.import_model = BoolWithConfirm::Confirmed,
                );
            }
            BoolWithConfirm::Confirmed => {
                let mut chooser = filechooser::FileChooser::new();
                let _ = chooser.set_path(&self.last_path);
                chooser.add_filter(filters::all_models());
                chooser.add_filter(filters::wavefront());
                chooser.add_filter(filters::pepakura());
                chooser.add_filter(filters::stl());
                chooser.add_filter(filters::all_files());
                self.file_dialog = Some(FileDialog::new(
                    chooser,
                    tr!("Import model..."),
                    FileAction::ImportModel,
                ));
                open_file_dialog = true;
            }
            BoolWithConfirm::None => {}
        }
        match menu_actions.update_model {
            BoolWithConfirm::Requested => {
                self.open_confirmation_dialog(ui,
                    &tr!("Update model"),
                    &tr!("This model is not saved and this operation cannot be undone.\nContinue anyway?"),
                    |a| a.update_model = BoolWithConfirm::Confirmed
                );
            }
            BoolWithConfirm::Confirmed => {
                let mut chooser = filechooser::FileChooser::new();
                let _ = chooser.set_path(&self.last_path);
                chooser.add_filter(filters::all_models());
                chooser.add_filter(filters::wavefront());
                chooser.add_filter(filters::pepakura());
                chooser.add_filter(filters::stl());
                chooser.add_filter(filters::all_files());
                self.file_dialog = Some(FileDialog::new(
                    chooser,
                    tr!("Update with new model..."),
                    FileAction::UpdateObj,
                ));
                open_file_dialog = true;
            }
            BoolWithConfirm::None => {}
        }
        if menu_actions.export_obj {
            let mut chooser = filechooser::FileChooser::new();
            let _ = chooser.set_path(&self.last_path);
            chooser.add_filter(filters::wavefront());
            chooser.add_filter(filters::all_files());
            self.file_dialog = Some(FileDialog::new(
                chooser,
                tr!("Export model..."),
                FileAction::ExportObj,
            ));
            open_file_dialog = true;
        }
        if menu_actions.generate_printable {
            use std::ffi::OsStr;

            let (last_path, last_file) = if self.last_export.as_os_str().is_empty() {
                (self.last_path.as_path(), OsStr::new(""))
            } else {
                let last_path = self.last_export.parent().unwrap_or_else(|| Path::new(""));
                let last_file = self.last_export.file_name().unwrap_or_default();
                (last_path, last_file)
            };
            let mut chooser = filechooser::FileChooser::new();
            let _ = chooser.set_path(last_path);
            chooser.set_file_name(last_file);
            chooser.add_filter(filters::pdf());
            chooser.add_filter(filters::svg());
            chooser.add_filter(filters::png());
            chooser.add_filter(filters::all_files());
            self.file_dialog = Some(FileDialog::new(
                chooser,
                tr!("Generate Printable..."),
                FileAction::GeneratePrintable,
            ));
            open_file_dialog = true;
        }

        // There are two Wait modals and two Error modals. One pair over the FileDialog, the other to be opened directly ("Save").

        if open_file_dialog {
            ui.open_popup(id("file_dialog_modal"));
        }
        if let Some(mut fd) = self.file_dialog.take() {
            let dsp_size = ui.display_size();
            let min_size = 0.25 * dsp_size;
            let max_size = 0.90 * dsp_size;
            ui.set_next_window_size_constraints(min_size, max_size);
            ui.set_next_window_size(0.75 * dsp_size, imgui::Cond::Once);
            ui.popup_modal_config(lbl_id(&fd.title, "file_dialog_modal"))
                .close_button(true)
                .with(|| {
                    let mut finish_file_dialog = false;
                    let res = fd.chooser.do_ui(ui, &self.filechooser_atlas);
                    match res {
                        filechooser::Output::Continue => {}
                        filechooser::Output::Cancel => {
                            finish_file_dialog = true;
                        }
                        filechooser::Output::Ok => {
                            let file = fd.chooser.full_path(filters::ext(fd.chooser.active_filter()));
                            if let Some(path) = file.parent() {
                                self.last_path = path.to_owned();
                            }
                            let action = if fd.action == FileAction::OpenCraft && fd.chooser.read_only() {
                                FileAction::OpenCraftReadOnly
                            } else {
                                fd.action
                            };
                            // A confirmation dialog is shown if the user:
                            // * Tries to open a file that doesn't exist.
                            // * Tries to save a file that does exist.
                            match (action.is_save(), file.exists()) {
                                (true, true) | (false, false) => {
                                    ui.open_popup(id("FileDialogConfirm"));
                                    fd.confirm = Some(file);
                                }
                                _ => {
                                    finish_file_dialog = true;
                                    open_wait = true;
                                    self.file_action = Some((action, file));
                                }
                            }
                        }
                    }

                    if let Some(confirm_name) = fd.confirm.take() {
                        let name = confirm_name.file_name().unwrap_or_default().to_string_lossy();
                        let reply = if fd.action.is_save() {
                            do_modal_dialog(ui,
                                &tr!("Overwrite?"), "FileDialogConfirm",
                                &tr!("The file '{}' already exists!\nWould you like to overwrite it?", name),
                                Some(&tr!("Cancel")),
                                Some(&tr!("Continue")),
                                )
                        } else {
                            do_modal_dialog(ui,
                                &tr!("Error!"), "FileDialogConfirm",
                                &tr!("The file '{}' doesn't exist!", name),
                                Some(&tr!("OK")), None)
                        };
                        match reply {
                            Some(true) => {
                                finish_file_dialog = true;
                                open_wait = true;
                                self.file_action = Some((fd.action, confirm_name));
                            }
                            Some(false) => {}
                            None => { fd.confirm = Some(confirm_name); }
                        }
                    }
                    if finish_file_dialog {
                        ui.close_current_popup();
                    } else {
                        self.file_dialog = Some(fd);
                    }
                });
        }
        if open_wait {
            self.popup_time_start = Instant::now();
            ui.open_popup(id("Wait"));
        }
    }

    fn run_mouse_actions(&mut self, ui: &Ui) {
        let shift_down = ui.is_key_down(imgui::Key::ModShift);
        let control_down = ui.is_key_down(imgui::Key::ModCtrl);
        let plus = ui.is_key_down(imgui::Key::KeypadAdd);
        let minus = ui.is_key_down(imgui::Key::KeypadSubtract);
        let alt_down = ui.is_key_down(imgui::Key::ModAlt);

        let mouse_pos = self.scene_ui_status.mouse_pos;
        if self.scene_ui_status.action != Canvas3dAction::None {
            'zoom: {
                let dz = match ui.io().MouseWheel {
                    x if x < 0.0 || minus => 1.0 / 1.1,
                    x if x > 0.0 || plus => 1.1,
                    _ => break 'zoom,
                };
                let flags = self.data.scene_zoom(self.sz_scene, mouse_pos, dz);
                self.add_rebuild(flags);
            }
        }
        let flags = match &self.scene_ui_status.action {
            Canvas3dAction::Hovering => {
                self.data
                    .scene_hover_event(self.sz_scene, mouse_pos, alt_down)
            }
            Canvas3dAction::Pressed(MouseButton::Left)
            | Canvas3dAction::Dragging(MouseButton::Left) => self
                .data
                .scene_button1_click_event(self.sz_scene, mouse_pos),
            Canvas3dAction::Pressed(MouseButton::Right)
            | Canvas3dAction::Dragging(MouseButton::Right) => self
                .data
                .scene_button2_click_event(self.sz_scene, mouse_pos),
            Canvas3dAction::DoubleClicked(MouseButton::Left) => self
                .data
                .scene_button1_dblclick_event(self.sz_scene, mouse_pos),
            Canvas3dAction::Released(MouseButton::Left) => self.data.scene_button1_release_event(
                self.sz_scene,
                mouse_pos,
                shift_down,
                control_down,
            ),
            _ => RebuildFlags::empty(),
        };
        self.add_rebuild(flags);

        let mouse_pos = self.paper_ui_status.mouse_pos;
        if self.paper_ui_status.action != Canvas3dAction::None {
            'zoom: {
                let dz = match ui.io().MouseWheel {
                    x if x < 0.0 || minus => 1.0 / 1.1,
                    x if x > 0.0 || plus => 1.1,
                    _ => break 'zoom,
                };
                let flags = self.data.paper_zoom(self.sz_paper, mouse_pos, dz);
                self.add_rebuild(flags);
            }
        }
        let flags = match &self.paper_ui_status.action {
            Canvas3dAction::Hovering => {
                self.data
                    .paper_hover_event(self.sz_paper, mouse_pos, alt_down)
            }
            Canvas3dAction::Clicked(MouseButton::Left)
            | Canvas3dAction::DoubleClicked(MouseButton::Left) => {
                self.data.paper_button1_click_event(
                    self.sz_paper,
                    mouse_pos,
                    shift_down,
                    control_down,
                    self.modifiable(),
                )
            }
            Canvas3dAction::Released(MouseButton::Left) => {
                self.data
                    .paper_button1_release_event(self.sz_paper, mouse_pos, control_down)
            }
            Canvas3dAction::Pressed(MouseButton::Right)
            | Canvas3dAction::Dragging(MouseButton::Right) => {
                self.data.paper_button2_event(self.sz_paper, mouse_pos)
            }
            Canvas3dAction::Pressed(MouseButton::Left) => {
                self.data.paper_button1_grab_event(
                    self.sz_paper,
                    mouse_pos,
                    shift_down,
                    /*dragging*/ false,
                )
            }
            Canvas3dAction::Dragging(MouseButton::Left) => {
                self.data.paper_button1_grab_event(
                    self.sz_paper,
                    mouse_pos,
                    shift_down,
                    /*dragging*/ true,
                )
            }
            Canvas3dAction::DragEnd(MouseButton::Left) => {
                self.data.paper_button1_drag_complete_event()
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
            self.gl
                .clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            self.gl.enable(glow::BLEND);
            self.gl.enable(glow::DEPTH_TEST);
            self.gl.depth_func(glow::LEQUAL);
            self.gl.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
            );

            self.gl.bind_vertex_array(Some(gl_fixs.vao.id()));
            if let (Some(tex), true) = (&self.data.gl_objs().textures, self.data.ui.show_textures) {
                self.gl.active_texture(glow::TEXTURE0);
                self.gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(tex.id()));
                u.texturize = 1;
            }

            self.gl.polygon_offset(1.0, 1.0);
            self.gl.enable(glow::POLYGON_OFFSET_FILL);

            gl_fixs.prg_scene_solid.draw(
                &u,
                (
                    &self.data.gl_objs().vertices,
                    &self.data.gl_objs().vertices_sel,
                ),
                glow::TRIANGLES,
            );

            if self.data.ui.show_3d_lines {
                //Joined edges
                self.gl.line_width(1.0);
                self.gl.disable(glow::LINE_SMOOTH);
                gl_fixs.prg_scene_line.draw(
                    &u,
                    &self.data.gl_objs().vertices_edge_joint,
                    glow::LINES,
                );

                //Cut edges
                self.gl.line_width(3.0);
                self.gl.enable(glow::LINE_SMOOTH);
                gl_fixs.prg_scene_line.draw(
                    &u,
                    &self.data.gl_objs().vertices_edge_cut,
                    glow::LINES,
                );
            }

            //Selected edge
            if self.data.has_selected_edge() {
                self.gl.line_width(5.0);
                self.gl.enable(glow::LINE_SMOOTH);
                if self.data.ui.xray_selection {
                    u.line_top = 1;
                }
                gl_fixs.prg_scene_line.draw(
                    &u,
                    &self.data.gl_objs().vertices_edge_sel,
                    glow::LINES,
                );
            }
        }
    }
    fn render_paper(&mut self, ui: &Ui) {
        let gl_fixs = &self.gl_fixs;

        let mut u = Uniforms2D {
            m: self.data.ui.trans_paper.ortho * self.data.ui.trans_paper.mx,
            tex: 0,
            frac_dash: 0.5,
            line_color: Rgba::new(0.0, 0.0, 0.0, 1.0),
            texturize: 0,
            notex_color: Rgba::new(0.75, 0.75, 0.75, 1.0),
        };

        unsafe {
            if self.data.ui.draw_paper {
                self.gl.clear_color(0.7, 0.7, 0.7, 1.0);
                // Out-of-paper area counts as a big imaginary piece, for overlapping purposes.
                // The stencil starts with 1 and is reduced to 0 when drawing the pages.
                self.gl.clear_stencil(1);
            } else {
                self.gl.clear_color(1.0, 1.0, 1.0, 1.0);
                // If pages are not drawn, then out-of-paper doesn't count as an overlap.
                // The stencil starts with 0 directly
                self.gl.clear_stencil(0);
            }
            self.gl.stencil_mask(0xff);
            self.gl.stencil_func(glow::ALWAYS, 0, 0);
            self.gl.disable(glow::STENCIL_TEST);

            self.gl
                .clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT);

            self.gl.enable(glow::BLEND);
            self.gl.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
            );

            self.gl.bind_vertex_array(Some(gl_fixs.vao.id()));
            if let (Some(tex), true) = (&self.data.gl_objs().textures, self.data.ui.show_textures) {
                self.gl.active_texture(glow::TEXTURE0);
                self.gl.bind_texture(glow::TEXTURE_2D_ARRAY, Some(tex.id()));
                u.texturize = 1;
            }

            // The paper
            if self.data.ui.draw_paper {
                self.gl.enable(glow::STENCIL_TEST);
                self.gl.stencil_op(glow::KEEP, glow::KEEP, glow::ZERO);

                gl_fixs.prg_paper_solid.draw(
                    &u,
                    &self.data.gl_objs().paper_vertices_page,
                    glow::TRIANGLES,
                );

                self.gl.disable(glow::STENCIL_TEST);

                u.line_color = Rgba::new(0.5, 0.5, 0.5, 1.0);

                gl_fixs.prg_paper_line.draw(
                    &u,
                    &self.data.gl_objs().paper_vertices_margin,
                    glow::LINES,
                );

                u.line_color = Rgba::new(0.0, 0.0, 0.0, 1.0);
            }

            // Line Flaps
            if self.data.ui.show_flaps {
                gl_fixs.prg_paper_line.draw(
                    &u,
                    &self.data.gl_objs().paper_vertices_flap_edge,
                    glow::LINES,
                );
            }

            self.gl.enable(glow::STENCIL_TEST);
            self.gl.stencil_op(glow::KEEP, glow::KEEP, glow::INCR);

            // Solid Flaps
            if self.data.ui.show_flaps {
                gl_fixs.prg_paper_solid.draw(
                    &u,
                    &self.data.gl_objs().paper_vertices_flap,
                    glow::TRIANGLES,
                );
            }
            self.gl.disable(glow::STENCIL_TEST);

            // Borders
            gl_fixs.prg_paper_line.draw(
                &u,
                &self.data.gl_objs().paper_vertices_edge_cut,
                glow::LINES,
            );

            self.gl.enable(glow::STENCIL_TEST);

            // Textured faces
            gl_fixs.prg_paper_solid.draw(
                &u,
                (
                    &self.data.gl_objs().paper_vertices,
                    &self.data.gl_objs().paper_vertices_sel,
                ),
                glow::TRIANGLES,
            );

            self.gl.disable(glow::STENCIL_TEST);

            // Shadow Flaps
            u.texturize = 0;
            if self.data.ui.show_flaps {
                u.notex_color = Rgba::new(0.0, 0.0, 0.0, 0.0);
                gl_fixs.prg_paper_solid.draw(
                    &u,
                    &self.data.gl_objs().paper_vertices_shadow_flap,
                    glow::TRIANGLES,
                );
                u.notex_color = Rgba::new(0.75, 0.75, 0.75, 1.0);
            }

            // Creases
            gl_fixs.prg_paper_line.draw(
                &u,
                &self.data.gl_objs().paper_vertices_edge_crease,
                glow::LINES,
            );

            // Selected edge
            if self.data.has_selected_edge() {
                u.line_color = color_edge(self.data.ui.mode);
                gl_fixs.prg_paper_line.draw(
                    &u,
                    &self.data.gl_objs().paper_vertices_edge_sel,
                    glow::LINES,
                );
            }

            // Draw the highlight overlap if "1 < STENCIL"
            self.gl.enable(glow::STENCIL_TEST);
            self.gl.stencil_op(glow::KEEP, glow::KEEP, glow::KEEP);

            if self.data.ui.highlight_overlaps {
                // Draw the overlapped highlight if "1 < STENCIL"
                let uq = UniformQuad {
                    color: Rgba::new(1.0, 0.0, 1.0, 0.9),
                };
                self.gl.stencil_func(glow::LESS, 1, 0xff);
                gl_fixs
                    .prg_quad
                    .draw(&uq, glr::NilVertexAttrib(3), glow::TRIANGLES);

                // Draw the highlight dim if "1 >= STENCIL"
                let uq = UniformQuad {
                    color: Rgba::new(1.0, 1.0, 1.0, 0.9),
                };
                self.gl.stencil_func(glow::GEQUAL, 1, 0xff);
                gl_fixs
                    .prg_quad
                    .draw(&uq, glr::NilVertexAttrib(3), glow::TRIANGLES);
            } else {
                // If highlight is disabled wraw the overlaps anyway, but dimmer, or else it would be invisible
                let uq = UniformQuad {
                    color: Rgba::new(1.0, 0.0, 1.0, 0.5),
                };
                self.gl.stencil_func(glow::LESS, 1, 0xff);
                gl_fixs
                    .prg_quad
                    .draw(&uq, glr::NilVertexAttrib(3), glow::TRIANGLES);
            }

            self.gl.disable(glow::STENCIL_TEST);

            // Draw the texts
            if self.data.ui.show_texts {
                self.gl.active_texture(glow::TEXTURE0);
                self.gl.bind_texture(
                    glow::TEXTURE_2D,
                    Renderer::unmap_tex(ui.font_atlas().texture_id()),
                );
                gl_fixs
                    .prg_text
                    .draw(&u, &self.data.gl_objs().paper_text, glow::TRIANGLES);
            }
        }
    }
    fn add_rebuild(&mut self, flags: RebuildFlags) {
        self.rebuild.insert(flags);
    }
    fn set_mouse_mode(&mut self, mode: MouseMode) {
        self.data.ui.mode = mode;
        self.add_rebuild(
            RebuildFlags::SELECTION | RebuildFlags::SCENE_REDRAW | RebuildFlags::PAPER_REDRAW,
        );
    }

    fn title(&self, with_unsaved_check: bool) -> String {
        let unsaved = if with_unsaved_check && self.data.modified {
            "*"
        } else {
            ""
        };
        let app_name = tr!("Papercraft");
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
    fn run_file_action(
        &mut self,
        ui: &Ui,
        action: FileAction,
        file_name: impl AsRef<Path>,
    ) -> anyhow::Result<()> {
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
                self.generate_printable(ui, file_name)?;
            }
        }
        Ok(())
    }
    fn open_craft(&mut self, file_name: &Path) -> anyhow::Result<()> {
        let fs = std::fs::File::open(file_name)
            .with_context(|| tr!("Error opening file {}", file_name.display()))?;
        let fs = std::io::BufReader::new(fs);
        let papercraft = Papercraft::load(fs)
            .with_context(|| tr!("Error loading file {}", file_name.display()))?;
        self.data = PapercraftContext::from_papercraft(papercraft, &self.gl)?;
        self.data.reset_views(self.sz_scene, self.sz_paper);
        if let Some(o) = self.options_opened.as_mut() {
            *o = self.data.papercraft().options().clone();
        }
        self.rebuild = RebuildFlags::all();
        Ok(())
    }
    fn save_as_craft(&self, file_name: &Path) -> anyhow::Result<()> {
        let f = std::fs::File::create(file_name)
            .with_context(|| tr!("Error creating file {}", file_name.display()))?;
        let f = std::io::BufWriter::new(f);
        self.data
            .papercraft()
            .save(f)
            .with_context(|| tr!("Error saving file {}", file_name.display()))?;
        Ok(())
    }
    fn import_model(&mut self, file_name: &Path) -> anyhow::Result<bool> {
        let (papercraft, is_native) = import_model_file(file_name)?;
        self.data = PapercraftContext::from_papercraft(papercraft, &self.gl)?;
        self.data.reset_views(self.sz_scene, self.sz_paper);
        if let Some(o) = self.options_opened.as_mut() {
            *o = self.data.papercraft().options().clone();
        }
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
        self.data
            .papercraft()
            .export_waveobj(file_name.as_ref())
            .with_context(|| tr!("Error exporting to {}", file_name.display()))?;
        Ok(())
    }

    fn save_backup_on_panic(&self) {
        log::info!("Crashing! trying to backup the current document");
        if !self.data.modified {
            log::info!("backup not needed");
            return;
        }
        let mut dir = std::env::temp_dir();
        dir.push(format!("crashed-{}.craft", std::process::id()));
        log::error!(
            "Papercraft panicked! Saving backup at \"{}\"",
            dir.display()
        );
        if let Err(e) = self.save_as_craft(&dir) {
            log::error!("backup failed with {e:?}");
        } else {
            log::error!("backup complete");
        }
    }
    fn pre_render_flags(&mut self, ui: &Ui, rebuild: RebuildFlags) {
        let text_helper = TextHelper {
            ui,
            font_text_line_scale: self.font_text_line_scale,
            font_id: self.font_text,
        };
        self.data.pre_render(rebuild, &text_helper);
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
    // The mouse is nowhere to be seen.
    None,
    // The mouse is moved over the canvas without pressing buttons
    Hovering,
    // A button has just been clicked
    Clicked(MouseButton),
    // A button is held pressed, but without moving
    Pressed(MouseButton),
    // A pressed button has been released
    Released(MouseButton),
    // A button has just been double-clicked, this replaces the second Clicked.
    DoubleClicked(MouseButton),
    // The mouse is being dragged while a button is pressed, it is converted from a Pressed.
    Dragging(MouseButton),
    // The Drag has ended (Released is not generated)
    DragEnd(MouseButton),
}

fn canvas3d(ui: &Ui, st: &mut Canvas3dStatus) {
    let sz = ui.get_content_region_avail();
    ui.invisible_button_config("canvas3d").size(sz).build();
    let hovered = ui.is_item_hovered();
    let pos = ui.get_item_rect_min();
    let scale = ui.display_scale();
    let mouse_pos = scale * (ui.get_mouse_pos() - pos);

    let action = match &st.action {
        Canvas3dAction::Dragging(bt) => {
            if ui.is_mouse_dragging(*bt) {
                Canvas3dAction::Dragging(*bt)
            } else {
                Canvas3dAction::DragEnd(*bt)
            }
        }
        Canvas3dAction::Hovering
        | Canvas3dAction::Pressed(_)
        | Canvas3dAction::Clicked(_)
        | Canvas3dAction::DoubleClicked(_) => {
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
        Canvas3dAction::None | Canvas3dAction::Released(_) | Canvas3dAction::DragEnd(_) => {
            // If the mouse is entered while dragging, it does not count, as if captured by other
            if hovered
                && !ui.is_mouse_dragging(MouseButton::Left)
                && !ui.is_mouse_dragging(MouseButton::Right)
            {
                Canvas3dAction::Hovering
            } else {
                Canvas3dAction::None
            }
        }
    };

    *st = Canvas3dStatus { mouse_pos, action };
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

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn install_crash_backup(event_loop: winit::event_loop::EventLoopProxy<()>) {
    // This is quite unsafe, maybe even UB, but we are crashing anyway, and we are trying to save
    // the user's data, what's the worst that could happen?
    use signal_hook::consts::signal::*;
    use signal_hook::iterator::Signals;
    std::thread::spawn(move || {
        let sigs = vec![SIGHUP, SIGINT, SIGTERM];
        let Ok(mut signals) = Signals::new(sigs) else {
            return;
        };
        let signal = signals.into_iter().next().unwrap();
        log::error!(
            "Got signal {}",
            signal_hook::low_level::signal_name(signal).unwrap_or(&signal.to_string())
        );

        // Lock the main loop
        if event_loop.send_event(()).is_ok() {
            let ctx = unsafe { &*CTX.load(std::sync::atomic::Ordering::Acquire) };
            ctx.save_backup_on_panic();
            std::process::abort();
        }
    });
}

#[cfg(not(target_os = "linux"))]
unsafe fn install_crash_backup(_event_loop: winit::event_loop::EventLoopProxy<()>) {}

impl imgui::UiBuilder for Box<GlobalContext> {
    fn build_custom_atlas(&mut self, atlas: &mut imgui::FontAtlasMut<'_, Self>) {
        self.add_rebuild(RebuildFlags::all());

        self.font_default = atlas.add_font(imgui::FontInfo::new(KARLA_TTF, 18.0));
        self.font_big = atlas.add_font(imgui::FontInfo::new(KARLA_TTF, 28.0));
        self.font_small = atlas.add_font(imgui::FontInfo::new(KARLA_TTF, 12.0));
        let options = self.data.papercraft().options();

        // Do not go too big or Dear ImGui will assert!
        let edge_id_font_size =
            (options.edge_id_font_size * options.resolution as f32 / 72.0).min(350.0);
        self.font_text = atlas.add_font(imgui::FontInfo::new(KARLA_TTF, edge_id_font_size));
        // This is eye-balled, depending on the particular font
        self.font_text_line_scale = 0.80;

        self.logo_rect =
            atlas.add_custom_rect_regular([LOGO_IMG.width(), LOGO_IMG.height()], |_, img| {
                img.copy_from(&*LOGO_IMG, 0, 0).unwrap();
            });

        // Each image is 48x48
        const W: u32 = 48;
        const H: u32 = 48;
        self.icons_rect[0] = atlas.add_custom_rect_regular([W, H], |_, img| {
            img.copy_from(&*ICONS_IMG.view(0, 0, W, H), 0, 0).unwrap();
        });
        self.icons_rect[1] = atlas.add_custom_rect_regular([W, H], |_, img| {
            img.copy_from(&*ICONS_IMG.view(W, 0, W, H), 0, 0).unwrap();
        });
        self.icons_rect[2] = atlas.add_custom_rect_regular([W, H], |_, img| {
            img.copy_from(&*ICONS_IMG.view(0, H, W, H), 0, 0).unwrap();
        });

        self.filechooser_atlas = filechooser::build_custom_atlas(atlas);
    }
    fn do_ui(&mut self, ui: &Ui) {
        //ui.show_demo_window(None);
        ui.set_next_window_pos(vec2(0.0, 0.0), imgui::Cond::Always, vec2(0.0, 0.0));
        let vw = ui.get_main_viewport();
        let sz = vw.size(); //&ui.get_content_region_avail();
        ui.set_next_window_size(sz, imgui::Cond::Always);
        ui.window_config(lbl("papercraft"))
            .flags(
                imgui::WindowFlags::NoDecoration
                    | imgui::WindowFlags::NoResize
                    | imgui::WindowFlags::MenuBar
                    | imgui::WindowFlags::NoBringToFrontOnFocus
                    | imgui::WindowFlags::NoNav,
            )
            .push_for_begin((
                (
                    imgui::StyleVar::WindowPadding,
                    imgui::StyleValue::Vec2([0.0, 0.0].into()),
                ),
                (imgui::StyleVar::WindowRounding, imgui::StyleValue::F32(0.0)),
            ))
            .with(|| {
                if let Some(cmd_file_action) = self.cmd_file_action.take() {
                    self.popup_time_start = Instant::now();
                    self.file_action = Some(cmd_file_action);
                    ui.open_popup(id("Wait"));
                }

                let menu_actions = self.build_ui(ui);
                self.run_menu_actions(ui, &menu_actions);
                self.run_mouse_actions(ui);

                if self
                    .rebuild
                    .intersects(RebuildFlags::ANY_REDRAW_SCENE | RebuildFlags::ANY_REDRAW_PAPER)
                {
                    self.pre_render_flags(ui, self.rebuild);
                    let vp = glr::PushViewport::new(&self.gl);
                    if self.rebuild.intersects(RebuildFlags::ANY_REDRAW_SCENE) {
                        let _draw_fb_binder = BinderDrawFramebuffer::bind(&self.gl_fixs.fbo_scene);
                        vp.viewport(0, 0, self.sz_scene.x as i32, self.sz_scene.y as i32);
                        self.render_scene();
                    }
                    if self.rebuild.intersects(RebuildFlags::ANY_REDRAW_PAPER) {
                        let _draw_fb_binder = BinderDrawFramebuffer::bind(&self.gl_fixs.fbo_paper);
                        vp.viewport(0, 0, self.sz_paper.x as i32, self.sz_paper.y as i32);
                        self.render_paper(ui);
                    }
                    self.rebuild = RebuildFlags::empty();
                }

                match (menu_actions.quit, self.quit_requested) {
                    (BoolWithConfirm::Confirmed, _) | (_, BoolWithConfirm::Confirmed) => {
                        self.quit_requested = BoolWithConfirm::Confirmed
                    }
                    (BoolWithConfirm::Requested, _) | (_, BoolWithConfirm::Requested) => {
                        self.quit_requested = BoolWithConfirm::None;
                        self.open_confirmation_dialog(
                            ui,
                            &tr!("Quit?"),
                            &tr!("The model has not been save, continue anyway?"),
                            |a| a.quit = BoolWithConfirm::Confirmed,
                        );
                    }
                    _ => (),
                }
            });
    }
}

struct TextHelper<'a> {
    ui: &'a Ui,
    font_text_line_scale: f32,
    font_id: imgui::FontId,
}

trait TextBuilder {
    fn font_text_line_scale(&self) -> f32;
    fn make_text(&self, p: &PrintableText, v: &mut Vec<util_gl::MVertexText>);
}

impl TextBuilder for TextHelper<'_> {
    fn font_text_line_scale(&self) -> f32 {
        self.font_text_line_scale
    }
    // To use the imgui fonts we need a Ui, so this is the only class that can do it.
    fn make_text(&self, pt: &PrintableText, vs: &mut Vec<util_gl::MVertexText>) {
        let f = self.ui.get_font(self.font_id);
        let mut x = 0.0;
        let width = pt
            .text
            .chars()
            .map(|c| {
                let r = self.ui.find_glyph(self.font_id, c);
                r.advance_x()
            })
            .sum::<f32>();
        let x_offset = match pt.align {
            TextAlign::Near => 0.0,
            TextAlign::Center => -width / 2.0,
            TextAlign::Far => -width,
        };
        let m = Matrix3::from_translation(pt.pos)
            * Matrix3::from(cgmath::Matrix2::from_angle(pt.angle))
            * Matrix3::from_scale(pt.size / f.FontSize)
            * Matrix3::from_translation(Vector2::new(x_offset, -f.Ascent));
        for c in pt.text.chars() {
            let r = self.ui.find_glyph(self.font_id, c);
            let mut p0 = r.p0();
            let mut p1 = r.p1();
            p0.x += x;
            p1.x += x;
            let mut q = [
                util_gl::MVertexText {
                    pos: p0,
                    uv: r.uv0(),
                },
                util_gl::MVertexText {
                    pos: Vector2::new(p1.x, p0.y),
                    uv: Vector2::new(r.uv1().x, r.uv0().y),
                },
                util_gl::MVertexText {
                    pos: Vector2::new(p0.x, p1.y),
                    uv: Vector2::new(r.uv0().x, r.uv1().y),
                },
                util_gl::MVertexText {
                    pos: Vector2::new(p0.x, p1.y),
                    uv: Vector2::new(r.uv0().x, r.uv1().y),
                },
                util_gl::MVertexText {
                    pos: Vector2::new(p1.x, p0.y),
                    uv: Vector2::new(r.uv1().x, r.uv0().y),
                },
                util_gl::MVertexText {
                    pos: p1,
                    uv: r.uv1(),
                },
            ];
            for v in &mut q {
                let p = m * Vector3::new(v.pos.x, v.pos.y, 1.0);
                v.pos.x = p.x;
                v.pos.y = p.y;
            }
            vs.extend(q);
            x += r.advance_x();
        }
    }
}

mod filters {
    use easy_imgui_filechooser::{Filter, FilterId, Pattern};
    use tr::tr;

    const ALL_FILES: FilterId = FilterId(0);
    const ALL_MODELS: FilterId = FilterId(1);
    const CRAFT: FilterId = FilterId(2);
    const WAVEFRONT: FilterId = FilterId(3);
    const PEPAKURA: FilterId = FilterId(4);
    const STL: FilterId = FilterId(5);
    const PDF: FilterId = FilterId(6);
    const SVG: FilterId = FilterId(7);
    const PNG: FilterId = FilterId(8);

    pub fn ext(filter: Option<FilterId>) -> Option<&'static str> {
        let ext = match filter? {
            CRAFT => "craft",
            WAVEFRONT => "obj",
            PEPAKURA => "pdo",
            STL => "stl",
            PDF => "pdf",
            SVG => "svg",
            PNG => "png",
            _ => return None,
        };
        Some(ext)
    }

    pub fn craft() -> Filter {
        Filter {
            id: CRAFT,
            text: tr!("Papercraft") + "(*.craft)",
            globs: vec![Pattern::new("*.craft").unwrap()],
        }
    }

    pub fn all_models() -> Filter {
        Filter {
            id: ALL_MODELS,
            text: tr!("All models") + "(*.obj *.pdo *.stl)",
            globs: vec![
                Pattern::new("*.obj").unwrap(),
                Pattern::new("*.pdo").unwrap(),
                Pattern::new("*.stl").unwrap(),
            ],
        }
    }

    pub fn wavefront() -> Filter {
        Filter {
            id: WAVEFRONT,
            text: tr!("Wavefront") + "(*.obj)",
            globs: vec![Pattern::new("*.obj").unwrap()],
        }
    }

    pub fn pepakura() -> Filter {
        Filter {
            id: PEPAKURA,
            text: tr!("Pepakura") + "(*.pdo)",
            globs: vec![Pattern::new("*.pdo").unwrap()],
        }
    }

    pub fn stl() -> Filter {
        Filter {
            id: STL,
            text: tr!("Stl") + "(*.stl)",
            globs: vec![Pattern::new("*.stl").unwrap()],
        }
    }

    pub fn pdf() -> Filter {
        Filter {
            id: PDF,
            text: tr!("PDF documents") + "(*.pdf)",
            globs: vec![Pattern::new("*.pdf").unwrap()],
        }
    }

    pub fn svg() -> Filter {
        Filter {
            id: SVG,
            text: tr!("SVG images") + "(*.svg)",
            globs: vec![Pattern::new("*.svg").unwrap()],
        }
    }

    pub fn png() -> Filter {
        Filter {
            id: PNG,
            text: tr!("PNG images") + "(*.png)",
            globs: vec![Pattern::new("*.png").unwrap()],
        }
    }

    pub fn all_files() -> Filter {
        Filter {
            id: ALL_FILES,
            text: tr!("All files"),
            globs: vec![],
        }
    }
}

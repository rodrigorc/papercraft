#![allow(unused_imports, dead_code)]

use std::{num::NonZeroU32, ffi::CString, time::Instant, rc::{Rc, Weak}, cell::RefCell, path::{Path, PathBuf}};
use anyhow::{Result, anyhow, Context};
use cgmath::{
    prelude::*,
    Deg, Rad,
};
use glow::HasContext;
use glutin::{prelude::*, config::{ConfigTemplateBuilder, Config}, display::GetGlDisplay, context::{ContextAttributesBuilder, ContextApi}, surface::{SurfaceAttributesBuilder, WindowSurface, Surface}};
use glutin_winit::DisplayBuilder;
use image::DynamicImage;
use imgui_winit_support::WinitPlatform;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use winit::{event_loop::{EventLoopBuilder, EventLoop}, window::{WindowBuilder, Window}, event::VirtualKeyCode};

mod imgui_filedialog;
mod waveobj;
mod paper;
mod glr;
mod util_3d;
mod util_gl;
mod ui;
//mod options_dlg;

use ui::*;

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
    window.set_title("Papercraft");
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

    use imgui_glow_renderer::TextureMap;

    let mut last_frame = Instant::now();

    // Initialize papaercraft status
    let sz_dummy = Vector2::new(1.0, 1.0);
    let fs = std::fs::File::open("examples/pikachu.craft").unwrap();
    let fs = std::io::BufReader::new(fs);
    let papercraft = Papercraft::load(fs).unwrap();
    let data = PapercraftContext::from_papercraft(
        papercraft,
        Some(&PathBuf::from("test")),
        sz_dummy,
        sz_dummy
    );

    let icons_tex = load_texture_from_memory(include_bytes!("icons.png"), true).unwrap();
    let icons_tex = ig_renderer.texture_map_mut().register(icons_tex).unwrap();

    let gl_fixs = build_gl_fixs().unwrap();
    let ctx = Rc::new_cyclic(|this| {
        RefCell::new(GlobalContext {
            this: this.clone(),
            gl_fixs,
            icons_tex,
            data,
            splitter_pos: 0.0,
            scene_ui_status: Canvas3dStatus::default(),
            paper_ui_status: Canvas3dStatus::default(),
            options_opened: false,
            mouse_mode: MouseMode::Face,
            quit: false,
            file_dialog: None,
            file_action: None,
            error_message: String::new(),
            popup_frame_start: 0,
        })
    });

    imgui_context.io_mut().config_flags |= imgui::ConfigFlags::NAV_ENABLE_KEYBOARD;
    let mut old_title = String::new();

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
                    ctx.build_ui(&ui);
                    //ui.show_demo_window(&mut true);
                    let new_title = ctx.title();
                    if new_title != old_title {
                        gl_window.window.set_title(&new_title);
                        old_title = new_title;
                    }
                    if ctx.quit {
                        *control_flow = winit::event_loop::ControlFlow::Exit;
                    }
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

    let vao = glr::VertexArray::generate();
    let fbo_paper = None; //Some(glr::Framebuffer::generate());
    let rbo_paper = None; //Some((glr::Renderbuffer::generate(), glr::Renderbuffer::generate()));

    Ok(GLFixedObjects {
        vao,
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
    vao: glr::VertexArray,
    fbo_paper: Option<glr::Framebuffer>,
    rbo_paper: Option<(glr::Renderbuffer, glr::Renderbuffer)>, //color, stencil

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
    icons_tex: imgui::TextureId,
    data: PapercraftContext,
    splitter_pos: f32,
    scene_ui_status: Canvas3dStatus,
    paper_ui_status: Canvas3dStatus,
    options_opened: bool,
    mouse_mode: MouseMode,
    quit: bool,
    file_dialog: Option<(imgui_filedialog::FileDialog, &'static str, FileAction)>,
    file_action: Option<(FileAction, PathBuf)>,
    error_message: String,
    popup_frame_start: i32,
}

impl GlobalContext {
    fn build_modal_error_message(&mut self, ui: &imgui::Ui) {
        if let Some(_pop) = ui.modal_popup_config("Error")
            .resizable(false)
            .always_auto_resize(true)
            .opened(&mut true)
            .begin_popup()
        {
            ui.text(&self.error_message);
            if ui.button_with_size("OK", [100.0, 0.0])
                || ui.is_key_pressed(imgui::Key::Enter)
                || ui.is_key_pressed(imgui::Key::KeyPadEnter)
            {
                if !ui.is_window_appearing() {
                    ui.close_current_popup();
                }
            }
        }
    }
    // Returns true if the action has just been done successfully
    fn build_modal_wait_message_and_run_file_action(&mut self, ui: &imgui::Ui) -> bool {
        let mut ok = false;
        if let Some(file_action) = self.file_action.take() {
            let (action, file) = &file_action;
            // These many frames to give time to the fading modal, should be enough
            let run = (ui.frame_count().wrapping_sub(self.popup_frame_start)) > 10;
            let title = action.title();
            if run {
                match self.run_file_action(*action, file) {
                    Ok(()) => {
                        ok = true;
                    }
                    Err(e) => {
                        self.error_message = format!("{e:?}");
                        ui.open_popup("Error");
                    }
                }
            } else {
                // keep the action pending, for now.
                self.file_action = Some(file_action);
            }

            // Build the modal itself
            if let Some(_pop) = ui.modal_popup_config(&format!("{title}###Wait"))
                .resizable(false)
                .begin_popup()
            {
                ui.text("Please, wait...");
                if run {
                    ui.close_current_popup();
                }
            }
        }
        ok
    }

    fn build_ui(&mut self, ui: &imgui::Ui) {
        let reset_views = self.build_menu_and_file_dialog(ui);

        const PAD: f32 = 4.0;
        let _s1 = ui.push_style_var(imgui::StyleVar::WindowPadding([PAD, PAD]));
        let _s2 = ui.push_style_var(imgui::StyleVar::ItemSpacing([0.0, 0.0]));
        if let Some(_toolbar) = ui.child_window("toolbar")
            .size([0.0, 48.0 + 2.0 * PAD])
            .always_use_window_padding(true)
            .border(false)
            .begin()
        {
            let _s3 = ui.push_style_var(imgui::StyleVar::ItemSpacing([2.0, 0.0]));
            let n = 48.0 / 128.0;
            let color_active = ui.style_color(imgui::StyleColor::ButtonActive);
            let color_white = [1.0, 1.0, 1.0, 1.0].into();
            let color_trans = [0.0, 0.0, 0.0, 0.0];

            if unsafe {
                let _t1 = ui.push_id("Face");
                imgui_sys::igImageButton(
                    self.icons_tex.id() as _,
                    [48.0, 48.0].into(),
                    [0.0, 0.0].into(),
                    [n, n].into(),
                    0,
                    (if self.data.mode == MouseMode::Face { color_active } else { color_trans }).into(),
                    color_white,
                )
            } {
                self.data.mode = MouseMode::Face;
            }
            ui.same_line();
            if unsafe {
                let _t1 = ui.push_id("Edge");
                imgui_sys::igImageButton(
                    self.icons_tex.id() as _,
                    [48.0, 48.0].into(),
                    [n, 0.0].into(),
                    [2.0*n, n].into(),
                    0,
                    (if self.data.mode == MouseMode::Edge { color_active } else { color_trans }).into(),
                    color_white,
                )
            } {
                self.data.mode = MouseMode::Edge;
            }
            ui.same_line();
            if unsafe {
                let _t1 = ui.push_id("Tab");
                imgui_sys::igImageButton(
                    self.icons_tex.id() as _,
                    [48.0, 48.0].into(),
                    [0.0, n].into(),
                    [n, 2.0*n].into(),
                    0,
                    (if self.data.mode == MouseMode::Tab { color_active } else { color_trans }).into(),
                    color_white,
                )
            } {
                self.data.mode = MouseMode::Tab;
            }
        }
        drop(_s1);
        drop(_s2);

        let _s1 = ui.push_style_var(imgui::StyleVar::ItemSpacing([2.0, 2.0]));
        let _s2 = ui.push_style_var(imgui::StyleVar::WindowPadding([0.0, 0.0]));
        let _s3 = ui.push_style_color(imgui::StyleColor::ButtonActive, ui.style_color(imgui::StyleColor::ButtonHovered));
        let _s4 = ui.push_style_color(imgui::StyleColor::Button, ui.style_color(imgui::StyleColor::ButtonHovered));

        let size = Vector2::from(ui.content_region_avail());

        if self.splitter_pos == 0.0 {
            self.splitter_pos = size.x / 2.0;
        }

        self.build_scene(ui, self.splitter_pos);
        let sz_scene = ui.item_rect_size();

        ui.same_line();

        ui.button_with_size("##vsplitter", [8.0, -1.0]);
        if ui.is_item_active() {
            self.splitter_pos += ui.io().mouse_delta[0];
        }
        self.splitter_pos = self.splitter_pos.clamp(50.0, size.x - 50.0);
        if ui.is_item_hovered() || ui.is_item_active() {
            ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeEW));
        }

        ui.same_line();

        self.build_paper(ui);
        let sz_paper = ui.item_rect_size();

        if reset_views {
            self.data.reset_views(sz_scene.into(), sz_paper.into());
        }

        if self.options_opened {
            if let Some(_options) = ui.window("Options##options")
                //.size([300.0, 300.0], imgui::Condition::Once)
                .resizable(false)
                .movable(true)
                .opened(&mut self.options_opened)
                .begin()
            {
                //TODO
                ui.label_text("", "hola");
            }
        }
    }

    fn build_menu_and_file_dialog(&mut self, ui: &imgui::Ui) -> bool {
        let mut reset_views = false;
        let mut save_as = false;
        let mut open_file_dialog = false;
        let mut open_wait = false;

        ui.menu_bar(|| {
            ui.menu("File", || {
                let title = "Open...";
                if ui.menu_item(title) {
                    let mut fd = imgui_filedialog::FileDialog::new();
                    fd.open("fd", "", "Papercraft (*.craft) {.craft},All files {.*}", "", "", 1,
                        imgui_filedialog::Flags::DISABLE_CREATE_DIRECTORY_BUTTON | imgui_filedialog::Flags::NO_DIALOG);
                    self.file_dialog = Some((fd, title, FileAction::OpenCraft));
                    open_file_dialog = true;
                }
                if ui.menu_item("Save") {
                    match &self.data.file_name {
                        Some(f) => {
                            self.file_action = Some((FileAction::SaveAsCraft, f.clone()));
                            open_wait = true;
                        }
                        None => save_as = true,
                    }
                }
                let title = "Save as...";
                if ui.menu_item(title) || save_as {
                    let mut fd = imgui_filedialog::FileDialog::new();
                    fd.open("fd", "", "Papercraft (*.craft) {.craft},All files {.*}", "", "", 1,
                        imgui_filedialog::Flags::CONFIRM_OVERWRITE | imgui_filedialog::Flags::NO_DIALOG);
                    self.file_dialog = Some((fd, title, FileAction::SaveAsCraft));
                    open_file_dialog = true;
                }
                let title = "Import OBJ...";
                if ui.menu_item(title) {
                    let mut fd = imgui_filedialog::FileDialog::new();
                    fd.open("fd", "", "Wavefront (*.obj) {.obj},All files {.*}", "", "", 1,
                        imgui_filedialog::Flags::DISABLE_CREATE_DIRECTORY_BUTTON | imgui_filedialog::Flags::NO_DIALOG);
                    self.file_dialog = Some((fd, title, FileAction::ImportObj));
                    open_file_dialog = true;
                }
                let title = "Update with new OBJ...";
                if ui.menu_item(title) {
                    let mut fd = imgui_filedialog::FileDialog::new();
                    fd.open("fd", "", "Wavefront (*.obj) {.obj},All files {.*}", "", "", 1,
                        imgui_filedialog::Flags::DISABLE_CREATE_DIRECTORY_BUTTON | imgui_filedialog::Flags::NO_DIALOG);
                    self.file_dialog = Some((fd, title, FileAction::UpdateObj));
                    open_file_dialog = true;
                }
                let title = "Export OBJ...";
                if ui.menu_item(title) {
                    let mut fd = imgui_filedialog::FileDialog::new();
                    fd.open("fd", "", "Wavefront (*.obj) {.obj},All files {.*}", "", "", 1,
                        imgui_filedialog::Flags::CONFIRM_OVERWRITE | imgui_filedialog::Flags::NO_DIALOG);
                    self.file_dialog = Some((fd, title, FileAction::ExportObj));
                    open_file_dialog = true;
                }
                let title = "Generate PDF...";
                if ui.menu_item(title) {
                    let mut fd = imgui_filedialog::FileDialog::new();
                    fd.open("fd", "", "PDF document (*.pdf) {.pdf},All files {.*}", "", "", 1,
                        imgui_filedialog::Flags::CONFIRM_OVERWRITE | imgui_filedialog::Flags::NO_DIALOG);
                    self.file_dialog = Some((fd, title, FileAction::GeneratePdf));
                    open_file_dialog = true;
                }
                ui.separator();
                if ui.menu_item("Quit") {
                    self.quit = true;
                }
            });
            ui.menu("Edit", || {
                if ui.menu_item_config("Undo")
                    .enabled(self.data.can_undo())
                    .build()
                {
                    if self.data.undo_action() {
                        self.add_rebuild(RebuildFlags::ALL);
                    }
                }

                ui.menu_item_config("Document properties")
                    .build_with_ref(&mut self.options_opened);

                ui.separator();

                if ui.menu_item_config("Face/Island")
                    .shortcut("F5")
                    .build_with_ref(&mut (self.data.mode == MouseMode::Face))
                {
                    self.data.mode = MouseMode::Face;
                }
                if ui.menu_item_config("Split/Join edge")
                    .shortcut("F6")
                    .build_with_ref(&mut (self.data.mode == MouseMode::Edge))
                {
                    self.data.mode = MouseMode::Edge;
                }
                if ui.menu_item_config("Tabs")
                    .shortcut("F7")
                    .build_with_ref(&mut (self.data.mode == MouseMode::Tab))
                {
                    self.data.mode = MouseMode::Tab;
                }

                ui.separator();

                if ui.menu_item("Reset views") {
                    reset_views = true;
                }
                if ui.menu_item("Repack pieces") {
                    let undo = self.data.pack_islands();
                    self.data.push_undo_action(undo);
                    self.add_rebuild(RebuildFlags::PAPER | RebuildFlags::SELECTION);
                }
            });
            ui.menu("View", || {
                if ui.menu_item_config("Textures")
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
                if ui.menu_item("About...") {
                    //TODO
                }
            });
        });

        if ui.is_key_index_pressed(VirtualKeyCode::F5 as _) {
            self.data.mode = MouseMode::Face;
        }
        if ui.is_key_index_pressed(VirtualKeyCode::F6 as _) {
            self.data.mode = MouseMode::Edge;
        }
        if ui.is_key_index_pressed(VirtualKeyCode::F7 as _) {
            self.data.mode = MouseMode::Tab;
        }

        if open_file_dialog {
            ui.open_popup("###file_dialog_modal");
        }
        if open_wait {
            self.popup_frame_start = ui.frame_count();
            ui.open_popup("###Wait");
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
                        // OK FD
                        if let Some(file) = fd2.file_path_name() {
                            self.file_action = Some((action, file.into()));
                            self.popup_frame_start = ui.frame_count();
                            ui.open_popup("###Wait");
                        }
                    } else {
                        // Cancel FD
                        finish_file_dialog = true;
                        ui.close_current_popup();
                    }
                }

                self.build_modal_error_message(ui);
                if self.build_modal_wait_message_and_run_file_action(ui) {
                    finish_file_dialog = true;
                    ui.close_current_popup();
                }

                if !finish_file_dialog {
                    // When pressing OK the FD tends to try and close itself, but if the file operation
                    // fails we want the dialog to keep on
                    self.file_dialog = Some((fd, title, action));
                }
            }
        }

        self.build_modal_error_message(ui);
        self.build_modal_wait_message_and_run_file_action(ui);

        reset_views
    }

    fn build_scene(&mut self, ui: &imgui::Ui, width: f32) {
        if let Some(_scene) = ui.child_window("scene")
            //.size([300.0, 300.0], imgui::Condition::Once)
            //.movable(true)
            .size([width, 0.0])
            .border(true)
            .begin()
        {
            let size = Vector2::from(ui.content_region_avail());
            if size.x <= 0.0 || size.y <= 0.0 {
                return;
            }
            let pos = Vector2::from(ui.cursor_screen_pos());
            let dsp_size = Vector2::from(ui.io().display_size);
            let ratio = size.x / size.y;
            let pos_y2 = dsp_size.y - pos.y - size.y;
            let wnd_x = -1.0 - 2.0 * pos.x / size.x;
            let wnd_y = -1.0 - 2.0 * pos_y2 / size.y;
            let wnd_width = 2.0 * dsp_size.x / size.x;
            let wnd_height = 2.0 * dsp_size.y / size.y;

            let mouse_pos = Vector2::from(ui.io().mouse_pos) - pos;
            let mut rebuild = RebuildFlags::empty();
            let mut ev_state = ModifierType::empty();
            if ui.io().key_shift {
                ev_state.insert(ModifierType::SHIFT_MASK);
            }
            if ui.io().key_ctrl {
                ev_state.insert(ModifierType::CONTROL_MASK);
            }

            canvas3d(ui, &mut self.scene_ui_status);

            match &self.scene_ui_status.action {
                Canvas3dAction::Hovering | Canvas3dAction::Pressed(_) => {
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
                Canvas3dAction::Clicked(imgui::MouseButton::Left) => {}
                Canvas3dAction::Released(imgui::MouseButton::Left) => {
                    ev_state.insert(ModifierType::BUTTON1_MASK);
                    let selection = self.data.scene_analyze_click(self.data.mode, size, mouse_pos);
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
                            rebuild.insert(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                        }
                        (MouseMode::Tab, ClickResult::Edge(i_edge, _)) => {
                            self.data.papercraft.edge_toggle_tab(i_edge);
                            self.data.push_undo_action(vec![UndoAction::TabToggle { i_edge } ]);
                            rebuild.insert(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                        }
                        (_, ClickResult::Face(f)) => {
                            let flags = self.data.set_selection(ClickResult::Face(f), true, ev_state.contains(ModifierType::CONTROL_MASK));
                            rebuild.insert(flags);
                        }
                        (_, ClickResult::None) => {
                            let flags = self.data.set_selection(ClickResult::None, true, ev_state.contains(ModifierType::CONTROL_MASK));
                            rebuild.insert(flags);
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
                    let flags = self.data.scene_motion_notify_event(size, mouse_pos, ev_state);
                    rebuild.insert(flags);
                }
                Canvas3dAction::Clicked(_) | Canvas3dAction::Released(_) | Canvas3dAction::None => {}
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
            .size([-1.0, -1.0])
            .border(true)
            .begin()
        {
            let size = Vector2::from(ui.content_region_avail());
            if size.x <= 0.0 || size.y <= 0.0 {
                return;
            }
            let pos = Vector2::from(ui.cursor_screen_pos());
            let dsp_size = Vector2::from(ui.io().display_size);
            //let ratio = size.x / size.y;
            let pos_y2 = dsp_size.y - pos.y - size.y;
            let wnd_x = -1.0 - 2.0 * pos.x / size.x;
            let wnd_y = -1.0 - 2.0 * pos_y2 / size.y;
            let wnd_width = 2.0 * dsp_size.x / size.x;
            let wnd_height = 2.0 * dsp_size.y / size.y;

            let mouse_pos = Vector2::from(ui.io().mouse_pos) - pos;
            let mut rebuild = RebuildFlags::empty();
            let mut ev_state = ModifierType::empty();
            if ui.io().key_shift {
                ev_state.insert(ModifierType::SHIFT_MASK);
            }
            if ui.io().key_ctrl {
                ev_state.insert(ModifierType::CONTROL_MASK);
            }

            canvas3d(ui, &mut self.paper_ui_status);

            match &self.paper_ui_status.action {
                Canvas3dAction::Hovering | Canvas3dAction::Pressed(_) => {
                    if self.paper_ui_status.action == Canvas3dAction::Hovering {
                        self.data.rotation_center = None;
                        self.data.grabbed_island = false;
                    }

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
                    ev_state.insert(ModifierType::BUTTON1_MASK);

                    let selection = self.data.paper_analyze_click(self.data.mode, size, mouse_pos);
                    match (self.data.mode, selection) {
                        (MouseMode::Edge, ClickResult::Edge(i_edge, i_face)) => {
                            self.data.grabbed_island = false;

                            let undo = if ev_state.contains(ModifierType::SHIFT_MASK) {
                                self.data.try_join_strip(i_edge)
                            } else {
                                self.data.edge_toggle_cut(i_edge, i_face)
                            };
                            if let Some(undo) = undo {
                                self.data.push_undo_action(undo);
                            }
                            rebuild.insert(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                        }
                        (MouseMode::Tab, ClickResult::Edge(i_edge, _)) => {
                            self.data.papercraft.edge_toggle_tab(i_edge);
                            self.data.push_undo_action(vec![UndoAction::TabToggle { i_edge } ]);
                            rebuild.insert(RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION);
                        }
                        (_, ClickResult::Face(f)) => {
                            rebuild.insert(self.data.set_selection(ClickResult::Face(f), true, ev_state.contains(ModifierType::CONTROL_MASK)));
                            let undo_action = self.data.selected_islands
                                .iter()
                                .map(|&i_island| {
                                    let island = self.data.papercraft.island_by_key(i_island).unwrap();
                                    UndoAction::IslandMove { i_root: island.root_face(), prev_rot: island.rotation(), prev_loc: island.location() }
                                })
                                .collect();
                            self.data.push_undo_action(undo_action);
                            self.data.grabbed_island = true;
                        }
                        (_, ClickResult::None) => {
                            rebuild.insert(self.data.set_selection(ClickResult::None, true, ev_state.contains(ModifierType::CONTROL_MASK)));
                            self.data.grabbed_island = false;
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
                    let flags = self.data.paper_motion_notify_event(size, mouse_pos, ev_state);
                    rebuild.insert(flags);
                }
                Canvas3dAction::Clicked(_) | Canvas3dAction::Released(_) | Canvas3dAction::None => {}

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

            // If there is an FBO, blit to the real FB
            if draw_fb_binder.take().is_some() { // check and drop
                let fbo = gl_fixs.fbo_paper.as_ref().unwrap();
                let _read_fb_binder = BinderReadFramebuffer::bind(fbo);
                gl::BlitFramebuffer(0, 0, width, height, 0, 0, width, height, gl::COLOR_BUFFER_BIT, gl::NEAREST);
            }
        }
    }

    fn add_rebuild(&mut self, flags: RebuildFlags) {
        self.data.rebuild.insert(flags);
        if flags.intersects(RebuildFlags::ANY_REDRAW_PAPER) {
            //self.wpaper.queue_render();
        }
        if flags.intersects(RebuildFlags::ANY_REDRAW_SCENE) {
            //self.wscene.queue_render();
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
    fn run_file_action(&mut self, action: FileAction, file_name: impl AsRef<Path>) -> anyhow::Result<()> {
        let sz_dummy = Vector2::new(1.0, 1.0);
        match action {
            FileAction::OpenCraft => {
                let fs = std::fs::File::open(&file_name)
                    .with_context(|| format!("Error opening file {}", file_name.as_ref().display()))?;
                let fs = std::io::BufReader::new(fs);
                let papercraft = Papercraft::load(fs)
                    .with_context(|| format!("Error loading file {}", file_name.as_ref().display()))?;
                self.data = PapercraftContext::from_papercraft(papercraft, Some(file_name.as_ref()), sz_dummy, sz_dummy);
            }
            FileAction::SaveAsCraft => {
                let f = std::fs::File::create(&file_name)
                    .with_context(|| format!("Error creating file {}", file_name.as_ref().display()))?;
                let f = std::io::BufWriter::new(f);
                self.data.papercraft.save(f)
                    .with_context(|| format!("Error saving file {}", file_name.as_ref().display()))?;
            }
            FileAction::ImportObj => {
                let papercraft = Papercraft::import_waveobj(file_name.as_ref())
                    .with_context(|| format!("Error reading Wavefront file {}", file_name.as_ref().display()))?;
                self.data = PapercraftContext::from_papercraft(papercraft, None, sz_dummy, sz_dummy);
                // set the modified flag
                self.data.push_undo_action(Vec::new());
            }
            FileAction::UpdateObj => {
                let mut new_papercraft = Papercraft::import_waveobj(file_name.as_ref())
                    .with_context(|| format!("Error reading Wavefront file {}", file_name.as_ref().display()))?;
                new_papercraft.update_from_obj(&self.data.papercraft);
                let tp = self.data.trans_paper.clone();
                let ts = self.data.trans_scene.clone();
                let original_file_name = self.data.file_name.clone();
                self.data = PapercraftContext::from_papercraft(new_papercraft, original_file_name.as_deref(), sz_dummy, sz_dummy);
                self.data.trans_paper = tp;
                self.data.trans_scene = ts;
            }
            FileAction::ExportObj => {
                self.data.papercraft.export_waveobj(file_name.as_ref())
                    .with_context(|| format!("Error exporting to {}", file_name.as_ref().display()))?;
            }
            FileAction::GeneratePdf => {
                let _backup = BackupGlConfig::backup();
                self.generate_pdf(file_name.as_ref())
                    .with_context(|| format!("Error exporting to {}", file_name.as_ref().display()))?;
            }
        }
        Ok(())
    }

    fn generate_pdf(&self, file_name: &Path) -> anyhow::Result<()> {
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
        let title = match &self.data.file_name {
            Some(f) => f.file_stem().map(|s| s.to_string_lossy()).unwrap_or_else(|| "".into()),
            None => "untitled".into()
        };
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
                    gl::BlitFramebuffer(0, 0, page_size_pixels.x, page_size_pixels.y, 0, 0, page_size_pixels.x, page_size_pixels.y, gl::COLOR_BUFFER_BIT, gl::NEAREST);
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
    Pressed(imgui::MouseButton),
    Released(imgui::MouseButton),
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
        Canvas3dAction::Hovering | Canvas3dAction::Pressed(_) | Canvas3dAction::Clicked(_) => {
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
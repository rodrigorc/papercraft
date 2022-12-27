use std::{num::NonZeroU32, ffi::CString, time::Instant};

use glow::HasContext;
use glutin::{prelude::*, config::{ConfigTemplateBuilder, Config}, display::GetGlDisplay, context::{ContextAttributesBuilder, ContextApi}, surface::{SurfaceAttributesBuilder, WindowSurface, Surface}};
use glutin_winit::DisplayBuilder;
use imgui_winit_support::WinitPlatform;
use raw_window_handle::HasRawWindowHandle;
use winit::{event_loop::EventLoopBuilder, window::{WindowBuilder, Window}};

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
                    //data.borrow_mut().build_ui(&ui);
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

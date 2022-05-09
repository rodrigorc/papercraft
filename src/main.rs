use anyhow::{Result, anyhow, Context};
use cgmath::{
    prelude::*,
    Deg, Rad,
};
use glib::clone;
use gtk::{
    prelude::*,
    gdk::{self, EventMask},
};

use std::{collections::HashMap, ops::ControlFlow, time::Duration, path::{Path, PathBuf}, cell::Cell};
use std::rc::Rc;
use std::cell::RefCell;

mod waveobj;
mod paper;
mod glr;
mod util_3d;
mod util_gl;
mod options_dlg;

use paper::{Papercraft, Model, PaperOptions, Face, EdgeStatus, JoinResult, IslandKey, FaceIndex, MaterialIndex, EdgeIndex, TabStyle};
use glr::Rgba;
use util_3d::{Matrix3, Matrix4, Quaternion, Vector2, Point2, Point3, Vector3, Matrix2};
use util_gl::{Uniforms2D, Uniforms3D, UniformQuad, MVertex3D, MVertex2D, MStatus3D, MSTATUS_UNSEL, MSTATUS_SEL, MSTATUS_HI, MVertex3DLine, MVertex2DColor, MVertex2DLine, MStatus2D};

use crate::glr::{BinderRenderbuffer, BinderDrawFramebuffer, BinderReadFramebuffer};

// In millimeters
const TAB_LINE_WIDTH: f32 = 0.2;
const BORDER_LINE_WIDTH: f32 = 0.1;
const CREASE_LINE_WIDTH: f32 = 0.05;

// In pixels
const LINE_SEL_WIDTH: f32 = 5.0;

pub trait SizeAsVector {
    fn size_as_vector(&self) -> Vector2;
}

impl<T: glib::IsA<gtk::Widget>> SizeAsVector for T {
    fn size_as_vector(&self) -> Vector2 {
        let r = self.allocation();
        Vector2::new(r.width() as f32, r.height() as f32)
    }
}

pub trait PositionAsVector {
    fn position_as_vector(&self) -> Vector2;
}

impl PositionAsVector for gdk::EventButton {
    fn position_as_vector(&self) -> Vector2 {
        let pos = self.position();
        Vector2::new(pos.0 as f32, pos.1 as f32)
    }
}
impl PositionAsVector for gdk::EventMotion {
    fn position_as_vector(&self) -> Vector2 {
        let pos = self.position();
        Vector2::new(pos.0 as f32, pos.1 as f32)
    }
}

impl PositionAsVector for gdk::EventScroll {
    fn position_as_vector(&self) -> Vector2 {
        let pos = self.position();
        Vector2::new(pos.0 as f32, pos.1 as f32)
    }
}

fn app_set_default_options(app: &gtk::Application) {
    app.lookup_action("mode").unwrap().change_state(&"face".to_variant());
    app.lookup_action("view_textures").unwrap().change_state(&true.to_variant());
    app.lookup_action("view_tabs").unwrap().change_state(&true.to_variant());
    app.lookup_action("3d_lines").unwrap().change_state(&true.to_variant());
    app.lookup_action("xray_selection").unwrap().change_state(&true.to_variant());
    app.lookup_action("overlap").unwrap().change_state(&false.to_variant());
}

fn on_app_startup(app: &gtk::Application, imports: Rc<RefCell<Option<String>>>) {
    dbg!("startup");
    let builder = gtk::Builder::from_string(include_str!("menu.ui"));
    let menu: gio::MenuModel = builder.object("appmenu").unwrap();
    app.set_menubar(Some(&menu));

    let wscene = gtk::GLArea::new();
    let wpaper = gtk::GLArea::new();
    let top_window = gtk::ApplicationWindow::new(app);
    let status = gtk::Label::new(None);

    let sz_dummy = Vector2::new(1.0, 1.0);
    let data = PapercraftContext::from_papercraft(Papercraft::empty(), None, sz_dummy, sz_dummy);
    let ctx = GlobalContext {
        top_window: top_window.clone(),
        status: status.clone(),
        wscene: wscene.clone(),
        wpaper: wpaper.clone(),
        gl_fixs: None,
        data,
    };

    let ctx: Rc<RefCell<GlobalContext>> = Rc::new(RefCell::new(ctx));

    let aquit = gio::SimpleAction::new("quit", None);
    app.add_action(&aquit);
    aquit.connect_activate(clone!(
        @strong app, @strong ctx =>
        move |_, _| {
            let w = ctx.borrow().top_window.clone();
            w.close();
    }));

    top_window.connect_delete_event(clone!(
        @strong ctx =>
        move |_, _| {
            let ok = GlobalContext::confirm_if_modified(&ctx, "Quit?");
            gtk::Inhibit(!ok)
        }
    ));
    let acrash = gio::SimpleAction::new("crash", None);
    app.add_action(&acrash);
    acrash.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let ctx = ctx.borrow();
            if !ctx.data.modified {
                return;
            }
            let mut dir = std::env::temp_dir();
            dir.push(format!("crashed-{}.craft", std::process::id()));

            eprintln!("Papercraft panicked! Saving backup at \"{}\"", dir.display());

            if let Err(e) = ctx.save(&dir) {
                eprintln!("backup failed with {e:?}");
            }

        }
    ));

    let aopen = gio::SimpleAction::new("open", None);
    app.add_action(&aopen);
    aopen.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            if !GlobalContext::confirm_if_modified(&ctx, "Load model") {
                return;
            }
            let top_window = ctx.borrow().top_window.clone();
            let app = top_window.application().unwrap();
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
            filter.add_pattern("*.craft");
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

    let aimport = gio::SimpleAction::new("import", None);
    app.add_action(&aimport);
    aimport.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            if !GlobalContext::confirm_if_modified(&ctx, "Import model") {
                return;
            }
            let top_window = ctx.borrow().top_window.clone();
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
                let e = ctx.borrow_mut().import_waveobj(name);
                if show_error_result(e, &top_window) {
                    app_set_default_options(&top_window.application().unwrap());
                }
            }
        }
    ));
    let aupdate = gio::SimpleAction::new("update", None);
    app.add_action(&aupdate);
    aupdate.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let top_window = ctx.borrow().top_window.clone();
            let dlg = gtk::FileChooserDialog::with_buttons(
                Some("Update from OBJ"),
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
                let res = Papercraft::import_waveobj(&name);
                match res {
                    Err(e) => {
                        show_error_result(Err(e), &top_window);
                    }
                    Ok(pc) => {
                        ctx.borrow_mut().update_from_obj(pc);
                        // We could make update_from_obj() to keep the current options, but now it resets to defaults, as if a new object was loaded
                        app_set_default_options(&top_window.application().unwrap());
                    }
                }
            }
        }
    ));

    let aexport = gio::SimpleAction::new("export", None);
    app.add_action(&aexport);
    aexport.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let top_window = ctx.borrow().top_window.clone();
            let dlg = gtk::FileChooserDialog::with_buttons(
                Some("Export OBJ"),
                Some(&top_window),
                gtk::FileChooserAction::Save,
                &[
                    ("Cancel", gtk::ResponseType::Cancel),
                    ("Export", gtk::ResponseType::Accept)
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
            dlg.set_do_overwrite_confirmation(true);

            let res = dlg.run();
            let name = if res == gtk::ResponseType::Accept {
                dlg.filename()
            } else {
                None
            };
            unsafe { dlg.destroy(); }

            if let Some(mut name) = name {
                if name.extension().is_none() {
                    name.set_extension("obj");
                }
                let e = ctx.borrow().data.papercraft.export_waveobj(name.as_ref())
                    .with_context(|| format!("Error exporting to {}", name.display()));
                show_error_result(e, &top_window);
            }
        }
    ));
    let agenerate_pdf = gio::SimpleAction::new("generate_pdf", None);
    app.add_action(&agenerate_pdf);
    agenerate_pdf.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let top_window = ctx.borrow().top_window.clone();
            let dlg = gtk::FileChooserDialog::with_buttons(
                Some("Generate PDF"),
                Some(&top_window),
                gtk::FileChooserAction::Save,
                &[
                    ("Cancel", gtk::ResponseType::Cancel),
                    ("Generate PDF", gtk::ResponseType::Accept)
                ]
            );
            dlg.set_current_folder(".");
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("PDF files"));
            filter.add_pattern("*.pdf");
            dlg.add_filter(&filter);
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("All files"));
            filter.add_pattern("*");
            dlg.add_filter(&filter);
            dlg.set_do_overwrite_confirmation(true);

            let res = dlg.run();
            let name = if res == gtk::ResponseType::Accept {
                dlg.filename()
            } else {
                None
            };
            unsafe { dlg.destroy(); }

            if let Some(mut name) = name {
                if name.extension().is_none() {
                    name.set_extension("pdf");
                }
                let e = ctx.borrow().generate_pdf(&name)
                    .with_context(|| format!("Error exporting to {}", name.display()));
                show_error_result(e, &top_window);
            }
        }
    ));

    let asave = gio::SimpleAction::new("save", None);
    app.add_action(&asave);
    asave.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let ctx_ = ctx.borrow();
            let top_window = ctx_.top_window.clone();
            if let Some(name) = ctx_.data.file_name.clone() {
                let e = ctx_.save(&name);
                drop(ctx_);
                if show_error_result(e, &top_window) {
                    let title = name.display().to_string();
                    let mut ctx = ctx.borrow_mut();
                    ctx.data.modified = false;
                    ctx.set_title(Some(&title));
                }
                return;
            }
            let app = top_window.application().unwrap();
            drop(ctx_);
            app.lookup_action("save_as").unwrap().activate(None);
        }
    ));

    let asave_as = gio::SimpleAction::new("save_as", None);
    app.add_action(&asave_as);
    asave_as.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let top_window = ctx.borrow().top_window.clone();
            let dlg = gtk::FileChooserDialog::with_buttons(
                Some("Save as"),
                Some(&top_window),
                gtk::FileChooserAction::Save,
                &[
                    ("Cancel", gtk::ResponseType::Cancel),
                    ("Save", gtk::ResponseType::Accept)
                ]
            );
            dlg.set_current_folder(".");
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("Craft models"));
            filter.add_pattern("*.craft");
            dlg.add_filter(&filter);
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("All files"));
            filter.add_pattern("*");
            dlg.add_filter(&filter);
            dlg.set_do_overwrite_confirmation(true);

            let res = dlg.run();
            let name = if res == gtk::ResponseType::Accept {
                dlg.filename()
            } else {
                None
            };
            unsafe { dlg.destroy(); }

            if let Some(mut name) = name {
                if name.extension().is_none() {
                    name.set_extension("craft");
                }
                let e = ctx.borrow().save(&name);
                if show_error_result(e, &top_window) {
                    let mut ctx = ctx.borrow_mut();
                    ctx.data.modified = false;
                    ctx.set_title(Some(&name));
                    ctx.data.file_name = Some(name);
                }
            }
        }
    ));

    let aundo = gio::SimpleAction::new("undo", None);
    app.add_action(&aundo);
    aundo.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let mut ctx = ctx.borrow_mut();
            if ctx.data.undo_action() {
                ctx.add_rebuild(RebuildFlags::ALL);
            }
        }
    ));

    let amode = gio::SimpleAction::new_stateful("mode", Some(glib::VariantTy::STRING), &"face".to_variant());
    app.add_action(&amode);
    amode.connect_change_state(clone!(
        @strong ctx =>
        move |a, v| {
            if let Some(v) = v {
                // Without this hack ToggleButtons can be unseledted.
                a.set_state(&"".to_variant());

                a.set_state(v);
                let mut ctx = ctx.borrow_mut();
                match v.str().unwrap() {
                    "face" => {
                        ctx.data.mode = MouseMode::Face;
                        ctx.status.set_text("Face mode. Click to select a piece. Drag on paper to move it. Shift-drag on paper to rotate it.");
                    }
                    "edge" => {
                        ctx.data.mode = MouseMode::Edge;
                        ctx.status.set_text("Edge mode. Click on an edge to split/join pieces. Shift-click to join a full strip of quads.");
                    }
                    "tab" => {
                        ctx.data.mode = MouseMode::Tab;
                        ctx.status.set_text("Tab mode. Click on an edge to swap the side of a tab.");
                    }
                    _ => {}
                }
            }
        }
    ));

    let areset_views = gio::SimpleAction::new("reset_views", None);
    app.add_action(&areset_views);
    areset_views.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let mut ctx = ctx.borrow_mut();
            let sz_scene = ctx.wscene.size_as_vector();
            let sz_paper = ctx.wpaper.size_as_vector();
            ctx.data.reset_views(sz_scene, sz_paper);
            ctx.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
        }
    ));

    let aoptions = gio::SimpleAction::new("options", None);
    app.add_action(&aoptions);
    aoptions.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            options_dlg::do_options_dialog(&ctx);
        }
    ));

    let arepack = gio::SimpleAction::new("repack", None);
    app.add_action(&arepack);
    arepack.connect_activate(clone!(
        @strong ctx =>
        move |_, _| {
            let mut ctx = ctx.borrow_mut();
            let undo = ctx.data.pack_islands();
            ctx.push_undo_action(undo);
            ctx.add_rebuild(RebuildFlags::PAPER | RebuildFlags::SELECTION);
        }
    ));

    let atexture = gio::SimpleAction::new_stateful("view_textures", None, &true.to_variant());
    app.add_action(&atexture);
    atexture.connect_change_state(clone!(
        @strong ctx =>
        move |a, v| {
            if let Some(v)  = v {
                a.set_state(v);
                let mut ctx = ctx.borrow_mut();
                ctx.data.show_textures = v.get().unwrap();
                ctx.add_rebuild(RebuildFlags::PAPER_REDRAW | RebuildFlags::SCENE_REDRAW);
            }
        }
    ));

    let view_3d_lines = gio::SimpleAction::new_stateful("3d_lines", None, &true.to_variant());
    app.add_action(&view_3d_lines);
    view_3d_lines.connect_change_state(clone!(
        @strong ctx =>
        move |a, v| {
            if let Some(v)  = v {
                a.set_state(v);
                let mut ctx = ctx.borrow_mut();
                ctx.data.show_3d_lines = v.get().unwrap();
                ctx.add_rebuild(RebuildFlags::SCENE_REDRAW);
            }
        }
    ));

    let atabs = gio::SimpleAction::new_stateful("view_tabs", None, &true.to_variant());
    app.add_action(&atabs);
    atabs.connect_change_state(clone!(
        @strong ctx =>
        move |a, v| {
            if let Some(v)  = v {
                a.set_state(v);
                let mut ctx = ctx.borrow_mut();
                ctx.data.show_tabs = v.get().unwrap();
                ctx.add_rebuild(RebuildFlags::PAPER);
            }
        }
    ));

    let axraysel = gio::SimpleAction::new_stateful("xray_selection", None, &true.to_variant());
    app.add_action(&axraysel);
    axraysel.connect_change_state(clone!(
        @strong ctx =>
        move |a, v| {
            if let Some(v)  = v {
                a.set_state(v);
                let mut ctx = ctx.borrow_mut();
                ctx.data.xray_selection = v.get().unwrap();
                ctx.add_rebuild(RebuildFlags::SELECTION);
            }
        }
    ));

    let aoverlap = gio::SimpleAction::new_stateful("overlap", None, &false.to_variant());
    app.add_action(&aoverlap);
    aoverlap.connect_change_state(clone!(
        @strong ctx =>
        move |a, v| {
            if let Some(v)  = v {
                a.set_state(v);
                let mut ctx = ctx.borrow_mut();
                ctx.data.highlight_overlaps = v.get().unwrap();
                ctx.add_rebuild(RebuildFlags::PAPER_REDRAW);
            }
        }
    ));

    top_window.set_default_size(800, 600);
    top_window.connect_destroy(clone!(
        @strong app =>
        move |_| {
            app.quit();
        }
    ));

    wscene.set_events(EventMask::BUTTON_PRESS_MASK | EventMask::BUTTON_MOTION_MASK | EventMask::POINTER_MOTION_MASK | EventMask::SCROLL_MASK);
    wscene.set_has_depth_buffer(true);

    wscene.connect_button_press_event(clone!(
        @strong ctx =>
        move |w, ev| {
            w.grab_focus();
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            let pos = ev.position_as_vector();
            ctx.data.last_cursor_pos = pos;

            let rebuild = match (ev.button(), ev.event_type()) {
                (1, gdk::EventType::ButtonPress) => {
                    let selection = ctx.data.scene_analyze_click(ctx.data.mode, ctx.wscene.size_as_vector(), pos);
                    match (ctx.data.mode, selection) {
                        (MouseMode::Edge, ClickResult::Edge(i_edge, i_face)) => {

                            let undo = if ev.state().contains(gdk::ModifierType::SHIFT_MASK) {
                                ctx.data.try_join_strip(i_edge)
                            } else {
                                ctx.data.edge_toggle_cut(i_edge, i_face)
                            };
                            if let Some(undo) = undo {
                                ctx.push_undo_action(undo);
                            }
                            RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION
                        }
                        (MouseMode::Tab, ClickResult::Edge(i_edge, _)) => {
                            ctx.push_undo_action(vec![UndoAction::TabToggle { i_edge } ]);
                            ctx.data.papercraft.edge_toggle_tab(i_edge);
                            RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION
                        }
                        (_, ClickResult::Face(f)) => {
                            ctx.data.set_selection(ClickResult::Face(f), true, ev.state().contains(gdk::ModifierType::CONTROL_MASK))
                        }
                        (_, ClickResult::None) => {
                            ctx.data.set_selection(ClickResult::None, true, ev.state().contains(gdk::ModifierType::CONTROL_MASK))
                        }
                        _ => {
                            RebuildFlags::empty()
                        }
                    }
                }
                (1, gdk::EventType::DoubleButtonPress) => {
                    let selection = ctx.data.scene_analyze_click(MouseMode::Face, ctx.wscene.size_as_vector(), pos);
                    if let ClickResult::Face(i_face) = selection {
                        if let Some(gl_objs) = &ctx.data.gl_objs {
                            // Compute the average of all the faces flat with the selected one, and move it to the center of the paper.
                            // Some vertices are counted twice, but they tend to be in diagonally opposed so the compensate, and it is an approximation anyways.
                            let mut center = Vector2::zero();
                            let mut n = 0.0;
                            for i_face in ctx.data.papercraft.get_flat_faces(i_face) {
                                let idx = 3 * gl_objs.paper_face_index[usize::from(i_face)] as usize;
                                for i in idx .. idx + 3 {
                                    center += gl_objs.paper_vertices[i].pos;
                                    n += 1.0;
                                }
                            }
                            center /= n;
                            ctx.data.trans_paper.mx[2][0] = -center.x * ctx.data.trans_paper.mx[0][0];
                            ctx.data.trans_paper.mx[2][1] = -center.y * ctx.data.trans_paper.mx[1][1];
                        }
                    }
                    RebuildFlags::PAPER_REDRAW
                }
                _ => {
                    RebuildFlags::empty()
                }
            };
            ctx.add_rebuild(rebuild);
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
            ctx.data.trans_scene.scale *= dz;
            ctx.data.trans_scene.recompute_obj();
            ctx.add_rebuild(RebuildFlags::SCENE_REDRAW);
            Inhibit(true)
        }
    ));
    wscene.connect_motion_notify_event(clone!(
        @strong ctx =>
        move |_w, ev|  {
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            let pos = ev.position_as_vector();

            let size = ctx.wscene.size_as_vector();
            let rebuild = ctx.data.scene_motion_notify_event(size, pos, ev);
            ctx.add_rebuild(rebuild);
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
            ctx.borrow_mut().data.trans_scene.set_ratio(ratio);
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
            let mut ctx = ctx.borrow_mut();
            ctx.data.trans_paper.ortho = util_3d::ortho2d(width as f32, height as f32);
            ctx.wpaper.make_current();
            let gl_fixs = ctx.gl_fixs.as_mut().unwrap();

            if let Some((rbo_color, rbo_stencil)) = &gl_fixs.rbo_paper {
                let rb_binder = BinderRenderbuffer::bind(rbo_color);
                let multisamples_color = glr::try_renderbuffer_storage_multisample(rb_binder.target(), gl::RGBA8, width, height);
                rb_binder.rebind(rbo_stencil);
                let multisamples_stencil = glr::try_renderbuffer_storage_multisample(rb_binder.target(), gl::STENCIL_INDEX8, width, height);
                if multisamples_color.is_some() && multisamples_color == multisamples_stencil {
                    let fbo = gl_fixs.fbo_paper.as_ref().unwrap();
                    let fb_binder = BinderDrawFramebuffer::new();
                    fb_binder.rebind(fbo);
                    unsafe {
                        gl::FramebufferRenderbuffer(fb_binder.target(), gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, rbo_color.id());
                        gl::FramebufferRenderbuffer(fb_binder.target(), gl::STENCIL_ATTACHMENT, gl::RENDERBUFFER, rbo_stencil.id());
                    }

                } else {
                    println!("Disable multisample!");
                    gl_fixs.fbo_paper = None;
                    gl_fixs.rbo_paper = None;
                }
            }
        }
    ));

    wpaper.connect_button_press_event(clone!(
        @strong ctx =>
        move |w, ev|  {
            w.grab_focus();
            let mut ctx = ctx.borrow_mut();
            let ctx = &mut *ctx;
            let pos = ev.position_as_vector();
            ctx.data.last_cursor_pos = pos;

            let rebuild = match (ev.button(), ev.event_type()) {
                (1, gdk::EventType::ButtonPress) => {
                    let selection = ctx.data.paper_analyze_click(ctx.data.mode, ctx.wpaper.size_as_vector(), pos);
                    match (ctx.data.mode, selection) {
                        (MouseMode::Edge, ClickResult::Edge(i_edge, i_face)) => {
                            ctx.data.grabbed_island = false;
                            let undo = if ev.state().contains(gdk::ModifierType::SHIFT_MASK) {
                                ctx.data.try_join_strip(i_edge)
                            } else {
                                ctx.data.edge_toggle_cut(i_edge, i_face)
                            };
                            if let Some(undo) = undo {
                                ctx.push_undo_action(undo);
                            }
                            RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION
                        }
                        (MouseMode::Tab, ClickResult::Edge(i_edge, _)) => {
                            ctx.push_undo_action(vec![UndoAction::TabToggle { i_edge } ]);
                            ctx.data.papercraft.edge_toggle_tab(i_edge);
                            RebuildFlags::PAPER | RebuildFlags::SCENE_EDGE | RebuildFlags::SELECTION
                        }
                        (_, ClickResult::Face(f)) => {
                            let rebuild = ctx.data.set_selection(ClickResult::Face(f), true, ev.state().contains(gdk::ModifierType::CONTROL_MASK));
                            let undo_action = ctx.data.selected_islands
                                .iter()
                                .map(|&i_island| {
                                    let island = ctx.data.papercraft.island_by_key(i_island).unwrap();
                                    UndoAction::IslandMove { i_root: island.root_face(), prev_rot: island.rotation(), prev_loc: island.location() }
                                })
                                .collect();
                            ctx.push_undo_action(undo_action);
                            ctx.data.grabbed_island = true;
                            rebuild
                        }
                        (_, ClickResult::None) => {
                            let rebuild = ctx.data.set_selection(ClickResult::None, true, ev.state().contains(gdk::ModifierType::CONTROL_MASK));
                            ctx.data.grabbed_island = false;
                            rebuild
                        }
                        _ => {
                            RebuildFlags::empty()
                        }
                    }
                }
                _ => {
                    RebuildFlags::empty()
                }
            };
            ctx.add_rebuild(rebuild);
            Inhibit(true)
        }
    ));
    wpaper.connect_button_release_event(clone!(
        @strong ctx =>
        move |_w, _ev|  {
            let mut ctx = ctx.borrow_mut();
            ctx.data.grabbed_island = false;
            ctx.data.rotation_center = None;
            ctx.set_scroll_timer(None);
            Inhibit(true)
        }
    ));
    wpaper.connect_motion_notify_event(clone!(
        @strong ctx =>
        move |w, ev| {
            let size = w.size_as_vector();
            let pos = ev.position_as_vector();
            let state = ev.state();

            let grabbed = {
                let mut ctx = ctx.borrow_mut();
                let rebuild = ctx.data.paper_motion_notify_event(size, pos, state);
                ctx.add_rebuild(rebuild);
                ctx.data.grabbed_island
            };

            if grabbed {
                let delta = if pos.x < 5.0 {
                    Some(Vector2::new((-pos.x).max(5.0).min(25.0), 0.0))
                } else if pos.x > size.x - 5.0 {
                    Some(Vector2::new(-(pos.x - size.x).max(5.0).min(25.0), 0.0))
                } else if pos.y < 5.0 {
                    Some(Vector2::new(0.0, (-pos.y).max(5.0).min(25.0)))
                } else if pos.y > size.y - 5.0 {
                    Some(Vector2::new(0.0, -(pos.y - size.y).max(5.0).min(25.0)))
                } else {
                    None
                };
                if let Some(delta) = delta {
                    let f = clone!(
                        @strong ctx =>
                        move || {
                            let mut ctx = ctx.borrow_mut();
                            ctx.data.last_cursor_pos += delta;
                            ctx.data.trans_paper.mx = Matrix3::from_translation(delta) * ctx.data.trans_paper.mx;
                            let rebuild = ctx.data.paper_motion_notify_event(size, pos, state);
                            ctx.add_rebuild(rebuild);
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
        move |w, ev|  {
            let mut ctx = ctx.borrow_mut();
            let dz = match ev.direction() {
                gdk::ScrollDirection::Up => 1.1,
                gdk::ScrollDirection::Down => 1.0 / 1.1,
                _ => 1.0,
            };
            let pos = ev.position_as_vector() - w.size_as_vector() / 2.0;
            ctx.data.trans_paper.mx = Matrix3::from_translation(pos) * Matrix3::from_scale(dz) * Matrix3::from_translation(-pos) * ctx.data.trans_paper.mx;
            ctx.add_rebuild(RebuildFlags::PAPER_REDRAW);
            Inhibit(true)
        }
    ));

    let hbin = gtk::Paned::new(gtk::Orientation::Horizontal);
    hbin.pack1(&wscene, true, true);
    hbin.pack2(&wpaper, true, true);

    let toolbar = gtk::Toolbar::new();

    let btn = gtk::ToggleToolButton::new();
    btn.set_action_name(Some("app.mode"));
    btn.set_action_target_value(Some(&"face".to_variant()));
    btn.set_icon_name(Some("media-playback-stop"));
    toolbar.add(&btn);

    let btn = gtk::ToggleToolButton::new();
    btn.set_action_name(Some("app.mode"));
    btn.set_action_target_value(Some(&"edge".to_variant()));
    btn.set_icon_name(Some("list-remove"));
    toolbar.add(&btn);

    let btn = gtk::ToggleToolButton::new();
    btn.set_action_name(Some("app.mode"));
    btn.set_action_target_value(Some(&"tab".to_variant()));
    btn.set_icon_name(Some("object-flip-horizontal"));
    toolbar.add(&btn);

    status.set_ellipsize(gtk::pango::EllipsizeMode::End);
    status.set_halign(gtk::Align::Start);
    status.set_margin(1);

    let status_frame = gtk::Frame::new(None);
    status_frame.set_shadow_type(gtk::ShadowType::EtchedIn);
    status_frame.set_margin(1);
    status_frame.add(&status);

    let vbin = gtk::Box::new(gtk::Orientation::Vertical, 0);
    top_window.add(&vbin);

    vbin.pack_start(&toolbar, false, true, 0);
    vbin.pack_start(&hbin, true, true, 0);
    vbin.pack_start(&status_frame, false, true, 0);

	app.connect_activate(clone!(
        @strong ctx =>
        move |_app| {
            dbg!("activate");
            let w = ctx.borrow().top_window.clone();
            w.show_all();
            w.present();
            app_set_default_options(&w.application().unwrap());
    	}
    ));

    fn app_open(ctx: &RefCell<GlobalContext>, file: &gio::File) -> Result<()> {
        let (data, _) = file.load_contents(gio::Cancellable::NONE)?;
        let papercraft = Papercraft::load(std::io::Cursor::new(&data[..]))?;
        let mut ctx = ctx.borrow_mut();
        let file_name = file.path();
        ctx.data = PapercraftContext::from_papercraft(papercraft, file_name.as_deref(), ctx.wscene.size_as_vector(), ctx.wpaper.size_as_vector());
        ctx.set_title(file_name.as_ref());
        Ok(())
    }

	app.connect_open(clone!(
        @strong ctx =>
        move |_app, files, _hint| {
            let w = ctx.borrow().top_window.clone();
            w.show_all();
            w.present();
            app_set_default_options(&w.application().unwrap());

            let e = app_open(&ctx, &files[0]);
            show_error_result(e, &w);
        }
	));

    let imports = imports.borrow();
    if let Some(args) = &*imports {
        let mut ctx = ctx.borrow_mut();
        let w = ctx.top_window.clone();
        let e = ctx.import_waveobj(args);
        drop(ctx);
        show_error_result(e, &w);
    }
}

fn show_error_result(e: Result<()>, parent: &impl IsA<gtk::Window>) -> bool {
    let e = match e {
        Ok(()) => return true,
        Err(e) => e,
    };
    show_error_message(&format!("{:?}", e), parent);
    false
}

fn show_error_message(msg: &str, parent: &impl IsA<gtk::Window>) {
    let dlg = gtk::MessageDialog::builder()
        .title("Error")
        .text(msg)
        .transient_for(parent)
        .message_type(gtk::MessageType::Error)
        .buttons(gtk::ButtonsType::Ok)
        .build();
    dlg.run();
    unsafe { dlg.destroy(); }
}

fn main() {
    if cfg!(windows) {
        // If you have this variable in Windows (Wine?) it will break the GSchemas
        std::env::set_var("XDG_DATA_DIRS", "");
        // The CSD is Windows is a bad idea
        std::env::set_var("GTK_CSD", "0");
    }

    let app = gtk::Application::new(None,
        gio::ApplicationFlags::HANDLES_OPEN | gio::ApplicationFlags::NON_UNIQUE
    );
    app.add_main_option("import", glib::Char::from(b'I'), glib::OptionFlags::NONE, glib::OptionArg::String, "Import a WaveOBJ file", None);
    let imports = Rc::new(RefCell::new(None));
    app.connect_handle_local_options(clone!(
        @strong imports =>
        move |_app, dict| {
            dbg!("local_option");
            //It should be a OptionArg::Filename and a PathBuf but that gets an \0 at the end that breaks everything
            let s: Option<String> = dict.lookup("import").unwrap();
            *imports.borrow_mut() = dbg!(s);
            -1
        }
    ));
	app.connect_startup(clone!(
        @strong imports =>
        move |app| {
            on_app_startup(app, imports.clone());
        }
    ));

    // When the application started a "crash" action will be installed, and when it is quit it will be removed.
    // If the application panics during the normal operation we will try and save a copy of the file in /tmp
    app.connect_shutdown(
        |app| {
            app.remove_action("crash");

        }
    );

    //In Linux convert fatal signals to panics to save the crash backup
    #[cfg(target_os="linux")]
    {
        const SIGHUP: i32 = 1;
        const SIGINT: i32 = 2;
        const SIGTERM: i32 = 15;
        const SIGUSR1: i32 = 10;
        const SIGUSR2: i32 = 12;
        glib::source::unix_signal_add_local(SIGHUP, || {
            panic!("SIGHUP");
        });
        glib::source::unix_signal_add_local(SIGINT, || {
            panic!("SIGINT");
        });
        glib::source::unix_signal_add_local(SIGTERM, || {
            panic!("SIGTERM");
        });
        glib::source::unix_signal_add_local(SIGUSR1, || {
            glib::Continue(true)
        });
        glib::source::unix_signal_add_local(SIGUSR2, || {
            glib::Continue(true)
        });
    }

    if let Err(e) = std::panic::catch_unwind(|| app.run()) {
        if let Some(a) = app.lookup_action("crash") {
            a.activate(None);
        }
        std::panic::resume_unwind(e);
    }
}

fn scene_realize(w: &gtk::GLArea, ctx: &mut GlobalContext) {
    w.attach_buffers();
    ctx.build_gl_fixs().unwrap();
    ctx.gl_fixs.as_mut().unwrap().vao_scene = Some(glr::VertexArray::generate());
}

fn paper_realize(w: &gtk::GLArea, ctx: &mut GlobalContext) {
    w.attach_buffers();
    ctx.build_gl_fixs().unwrap();
    let gl_fixs = ctx.gl_fixs.as_mut().unwrap();
    gl_fixs.vao_paper = Some(glr::VertexArray::generate());
    gl_fixs.fbo_paper = Some(glr::Framebuffer::generate());
    gl_fixs.rbo_paper = Some((glr::Renderbuffer::generate(), glr::Renderbuffer::generate()));
}

struct GLFixedObjects {
    //VAOs are not shareable between contexts, so we need two, one for each window
    vao_scene: Option<glr::VertexArray>,
    vao_paper: Option<glr::VertexArray>,
    fbo_paper: Option<glr::Framebuffer>,
    rbo_paper: Option<(glr::Renderbuffer, glr::Renderbuffer)>, //color, stencil

    prg_scene_solid: glr::Program,
    prg_scene_line: glr::Program,
    prg_paper_solid: glr::Program,
    prg_paper_line: glr::Program,
    prg_quad: glr::Program,
}

struct GLObjects {
    textures: Option<glr::Texture>,

    //GL objects that are rebuild with the model
    vertices: glr::DynamicVertexArray<MVertex3D>,
    vertices_sel: glr::DynamicVertexArray<MStatus3D>,
    vertices_edge_joint: glr::DynamicVertexArray<MVertex3DLine>,
    vertices_edge_cut: glr::DynamicVertexArray<MVertex3DLine>,
    vertices_edge_sel: glr::DynamicVertexArray<MVertex3DLine>,

    paper_vertices: glr::DynamicVertexArray<MVertex2D>,
    paper_vertices_sel: glr::DynamicVertexArray<MStatus2D>,
    paper_vertices_edge_border: glr::DynamicVertexArray<MVertex2DLine>,
    paper_vertices_edge_crease: glr::DynamicVertexArray<MVertex2DLine>,
    paper_vertices_tab: glr::DynamicVertexArray<MVertex2DColor>,
    paper_vertices_tab_edge: glr::DynamicVertexArray<MVertex2DLine>,
    paper_vertices_edge_sel: glr::DynamicVertexArray<MVertex2DLine>,

    // Maps a FaceIndex to the index into paper_vertices
    paper_face_index: Vec<u32>,

    paper_vertices_page: glr::DynamicVertexArray<MVertex2DColor>,
    paper_vertices_margin: glr::DynamicVertexArray<MVertex2DLine>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum MouseMode {
    Face,
    Edge,
    Tab,
}


//UndoItem cannot store IslandKey, because they are dynamic, use the root of the island instead
#[derive(Debug)]
enum UndoAction {
    IslandMove { i_root: FaceIndex, prev_rot: Rad<f32>, prev_loc: Vector2 },
    TabToggle { i_edge: EdgeIndex },
    EdgeCut { i_edge: EdgeIndex },
    EdgeJoin { join_result: JoinResult },
    DocConfig { options: PaperOptions },
}

bitflags::bitflags! {
    struct RebuildFlags: u32 {
        const PAGES = 0x0001;
        const PAPER = 0x0002;
        const SCENE_EDGE = 0x0004;
        const SELECTION = 0x0008;
        const PAPER_REDRAW = 0x0010;
        const SCENE_REDRAW = 0x0020;

        const ALL = Self::PAGES.bits | Self::PAPER.bits | Self::SCENE_EDGE.bits | Self::SELECTION.bits;

        const ANY_REDRAW_PAPER = Self::PAGES.bits | Self::PAPER.bits | Self::SELECTION.bits | Self::PAPER_REDRAW.bits;
        const ANY_REDRAW_SCENE = Self::SCENE_EDGE.bits | Self::SELECTION.bits | Self::SCENE_REDRAW.bits;
    }
}

//Objects that are recreated when a new model is loaded
struct PapercraftContext {
    // The model
    file_name: Option<PathBuf>,
    papercraft: Papercraft,
    undo_stack: Vec<Vec<UndoAction>>,
    modified: bool,

    rebuild: RebuildFlags,
    gl_objs: Option<GLObjects>,

    // State
    selected_face: Option<FaceIndex>,
    selected_edge: Option<EdgeIndex>,
    selected_islands: Vec<IslandKey>,
    grabbed_island: bool,
    scroll_timer: Option<glib::SourceId>,

    last_cursor_pos: Vector2,
    rotation_center: Option<Vector2>,

    mode: MouseMode,
    show_textures: bool,
    show_tabs: bool,
    show_3d_lines: bool,
    xray_selection: bool,
    highlight_overlaps: bool,
    trans_scene: Transformation3D,
    trans_paper: TransformationPaper,
}

struct GlobalContext {
    top_window: gtk::ApplicationWindow,
    status: gtk::Label,
    wscene: gtk::GLArea,
    wpaper: gtk::GLArea,

    gl_fixs: Option<GLFixedObjects>,

    data: PapercraftContext,
}

#[derive(Clone)]
struct Transformation3D {
    location: Vector3,
    rotation: Quaternion,
    scale: f32,

    obj: Matrix4,
    persp: Matrix4,
    persp_inv: Matrix4,
    view: Matrix4,
    view_inv: Matrix4,
    mnormal: Matrix3,
}

impl Transformation3D {
    fn new(obj: Matrix4, location: Vector3, rotation: Quaternion, scale: f32, persp: Matrix4) -> Transformation3D {
        let mut tr = Transformation3D {
            location,
            rotation,
            scale,
            obj,
            persp,
            persp_inv: persp.invert().unwrap(),
            view: Matrix4::one(),
            view_inv: Matrix4::one(),
            mnormal: Matrix3::one(),
        };
        tr.recompute_obj();
        tr
    }
    fn recompute_obj(&mut self) {
        let r = Matrix3::from(self.rotation);
        let t = Matrix4::from_translation(self.location);
        let s = Matrix4::from_scale(self.scale);

        self.view = t * Matrix4::from(r) * s * self.obj;
        self.view_inv = self.view.invert().unwrap();
        self.mnormal = r; //should be inverse of transpose
    }

    fn set_ratio(&mut self, ratio: f32) {
        let f = self.persp[1][1];
        self.persp[0][0] = f / ratio;
        self.persp_inv = self.persp.invert().unwrap();
    }
}

#[derive(Clone)]
struct TransformationPaper {
    ortho: Matrix3,
    mx: Matrix3,
}

impl TransformationPaper {
    fn paper_click(&self, size: Vector2, pos: Vector2) -> Vector2 {
        let x = (pos.x / size.x) * 2.0 - 1.0;
        let y = -((pos.y / size.y) * 2.0 - 1.0);
        let click = Point2::new(x, y);

        let mx = self.ortho * self.mx;
        let mx_inv = mx.invert().unwrap();
        mx_inv.transform_point(click).to_vec()
    }
}

#[derive(Debug)]
enum ClickResult {
    None,
    Face(FaceIndex),
    Edge(EdgeIndex, Option<FaceIndex>),
}

struct PaperDrawFaceArgs {
    vertices: Vec<MVertex2D>,
    vertices_edge_border: Vec<MVertex2DLine>,
    vertices_edge_crease: Vec<MVertex2DLine>,
    vertices_tab: Vec<MVertex2DColor>,
    vertices_tab_edge: Vec<MVertex2DLine>,
    face_index: Vec<u32>,
}

impl PaperDrawFaceArgs {
    fn new(model: &Model) -> PaperDrawFaceArgs {
        PaperDrawFaceArgs {
            vertices: Vec::new(),
            vertices_edge_border: Vec::new(),
            vertices_edge_crease: Vec::new(),
            vertices_tab: Vec::new(),
            vertices_tab_edge: Vec::new(),
            face_index: vec![0; model.num_faces()],
        }
    }
}

impl PapercraftContext {
    fn default_transformations(obj: Matrix4, sz_scene: Vector2, sz_paper: Vector2, ops: &PaperOptions) -> (Transformation3D, TransformationPaper) {
        let page = Vector2::from(ops.page_size);
        let persp = cgmath::perspective(Deg(60.0), 1.0, 1.0, 100.0);
        let mut trans_scene = Transformation3D::new(
            obj,
            Vector3::new(0.0, 0.0, -30.0),
            Quaternion::one(),
            20.0,
            persp
        );
        let ratio = sz_scene.x / sz_scene.y;
        trans_scene.set_ratio(ratio);

        let trans_paper = {
            let mt = Matrix3::from_translation(Vector2::new(-page.x / 2.0, -page.y / 2.0));
            let ms = Matrix3::from_scale(1.0);
            let ortho = util_3d::ortho2d(sz_paper.x, sz_paper.y);
            TransformationPaper {
                ortho,
                mx: ms * mt,
            }
        };
        (trans_scene, trans_paper)
    }

    fn from_papercraft(papercraft: Papercraft, file_name: Option<&Path>, sz_scene: Vector2, sz_paper: Vector2) -> PapercraftContext {
        // Compute the bounding box, then move to the center and scale to a standard size
        let (v_min, v_max) = util_3d::bounding_box_3d(
            papercraft.model()
                .vertices()
                .map(|(_, v)| v.pos())
        );
        let size = (v_max.x - v_min.x).max(v_max.y - v_min.y).max(v_max.z - v_min.z);
        let mscale = Matrix4::from_scale(1.0 / size);
        let center = (v_min + v_max) / 2.0;
        let mcenter = Matrix4::from_translation(-center);
        let obj = mscale * mcenter;

        let (trans_scene, trans_paper) = Self::default_transformations(obj, sz_scene, sz_paper, papercraft.options());

        PapercraftContext {
            file_name: file_name.map(|f| f.to_owned()),
            papercraft,
            undo_stack: Vec::new(),
            modified: false,
            rebuild: RebuildFlags::ALL,
            gl_objs: None,
            selected_face: None,
            selected_edge: None,
            selected_islands: Vec::new(),
            grabbed_island: false,
            scroll_timer: None,
            last_cursor_pos: Vector2::zero(),
            rotation_center: None,
            mode: MouseMode::Face,
            show_textures: true,
            show_tabs: true,
            show_3d_lines: true,
            xray_selection: true,
            highlight_overlaps: false,
            trans_scene,
            trans_paper,
        }
    }

    fn build_gl_objs(&mut self) {
        if self.gl_objs.is_none() {
            let images = self.papercraft.model()
                .textures()
                .map(|tex| tex.pixbuf())
                .collect::<Vec<_>>();

            let sizes = images
                .iter()
                .filter_map(|i| i.as_ref())
                .map(|i| {
                    (i.width(), i.height())
                });
            let max_width = sizes.clone().map(|(w, _)| w).max();
            let max_height = sizes.map(|(_, h)| h).max();

            let textures = match max_width.zip(max_height) {
                None => None,
                Some((width, height)) => {
                    let mut blank = None;
                    unsafe {
                        let textures = glr::Texture::generate();
                        gl::BindTexture(gl::TEXTURE_2D_ARRAY, textures.id());
                        gl::TexImage3D(gl::TEXTURE_2D_ARRAY, 0, gl::RGBA8 as i32, width, height, images.len() as i32, 0, gl::RGB, gl::UNSIGNED_BYTE, std::ptr::null());
                        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
                        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
                        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR as i32);
                        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

                        for (layer, image) in images.iter().enumerate() {
                            if let Some(image) = image {
                                let scaled_image;
                                let image = if width == image.width() && height == image.height() {
                                    image
                                } else {
                                    scaled_image = image.scale_simple(width, height, gdk_pixbuf::InterpType::Bilinear).unwrap();
                                    &scaled_image
                                };
                                let bytes = image.read_pixel_bytes().unwrap();
                                let format = match image.n_channels() {
                                    4 => gl::RGBA,
                                    3 => gl::RGB,
                                    2 => gl::RG,
                                    _ => gl::RED,
                                };
                                gl::TexSubImage3D(gl::TEXTURE_2D_ARRAY, 0, 0, 0, layer as i32, width, height, 1, format, gl::UNSIGNED_BYTE, bytes.as_ptr() as *const _);
                            } else {
                                let blank = blank.get_or_insert_with(|| {
                                    let c = (0x80u8, 0x80u8, 0x80u8);
                                    vec![c; width as usize * height as usize]
                                });
                                gl::TexSubImage3D(gl::TEXTURE_2D_ARRAY, 0, 0, 0, layer as i32, width, height, 1, gl::RGB, gl::UNSIGNED_BYTE, blank.as_ptr() as *const _);
                            }
                        }
                        gl::GenerateMipmap(gl::TEXTURE_2D_ARRAY);
                        Some(textures)
                    }
                }
            };
            let mut vertices = Vec::new();
            let mut face_map = vec![Vec::new(); self.papercraft.model().num_textures()];
            for (i_face, face) in self.papercraft.model().faces() {
                for i_v in face.index_vertices() {
                    let v = &self.papercraft.model()[i_v];
                    vertices.push(MVertex3D {
                        pos: v.pos(),
                        normal: v.normal(),
                        uv: v.uv(),
                        mat: face.material(),
                    });
                }
                face_map[usize::from(face.material())].push(i_face);
            }

            let mut face_index = vec![0; self.papercraft.model().num_faces()];
            let mut f_idx = 0;
            for fm in face_map {
                for f in fm {
                    face_index[usize::from(f)] = f_idx;
                    f_idx += 1;
                }
            }

            let vertices = glr::DynamicVertexArray::from(vertices);
            let vertices_sel = glr::DynamicVertexArray::from(vec![MSTATUS_UNSEL; 3 * self.papercraft.model().num_faces()]);
            let vertices_edge_joint = glr::DynamicVertexArray::new();
            let vertices_edge_cut = glr::DynamicVertexArray::new();
            let vertices_edge_sel = glr::DynamicVertexArray::new();

            let paper_vertices = glr::DynamicVertexArray::new();
            let paper_vertices_sel = glr::DynamicVertexArray::from(vec![MStatus2D { color: MSTATUS_UNSEL.color }; 3 * self.papercraft.model().num_faces()]);
            let paper_vertices_edge_border = glr::DynamicVertexArray::new();
            let paper_vertices_edge_crease = glr::DynamicVertexArray::new();
            let paper_vertices_tab = glr::DynamicVertexArray::new();
            let paper_vertices_tab_edge = glr::DynamicVertexArray::new();
            let paper_vertices_edge_sel = glr::DynamicVertexArray::new();

            let paper_vertices_page = glr::DynamicVertexArray::new();
            let paper_vertices_margin = glr::DynamicVertexArray::new();

            self.gl_objs = Some(GLObjects {
                textures,
                vertices,
                vertices_sel,
                vertices_edge_joint,
                vertices_edge_cut,
                vertices_edge_sel,

                paper_vertices,
                paper_vertices_sel,
                paper_vertices_edge_border,
                paper_vertices_edge_crease,
                paper_vertices_tab,
                paper_vertices_tab_edge,
                paper_vertices_edge_sel,

                paper_face_index: Vec::new(),

                paper_vertices_page,
                paper_vertices_margin,
            });
        }
        self.rebuild_pending();
    }

    fn pre_render(&mut self) {
        self.build_gl_objs();
        self.rebuild_pending();
    }

    fn rebuild_pending(&mut self) {
        if self.rebuild.contains(RebuildFlags::PAGES) {
            self.pages_rebuild();
        }
        if self.rebuild.contains(RebuildFlags::PAPER) {
            self.paper_rebuild();
        }
        if self.rebuild.contains(RebuildFlags::SCENE_EDGE) {
            self.scene_edge_rebuild();
        }
        if self.rebuild.contains(RebuildFlags::SELECTION) {
            self.selection_rebuild();
        }
        self.rebuild = RebuildFlags::empty();
    }

    fn reset_views(&mut self, sz_scene: Vector2, sz_paper: Vector2) {
        (self.trans_scene, self.trans_paper) = Self::default_transformations(self.trans_scene.obj, sz_scene, sz_paper, self.papercraft.options());
    }

    fn paper_draw_face(&self, face: &Face, i_face: FaceIndex, m: &Matrix3, args: &mut PaperDrawFaceArgs) {
        args.face_index[usize::from(i_face)] = args.vertices.len() as u32 / 3;

        for i_v in face.index_vertices() {
            let v = &self.papercraft.model()[i_v];
            let p = self.papercraft.face_plane(face).project(&v.pos());
            let pos = m.transform_point(Point2::from_vec(p)).to_vec();

            args.vertices.push(MVertex2D {
                pos,
                uv: v.uv(),
                mat: face.material(),
            });
        }

        let tab_style = self.papercraft.options().tab_style;
        for (i_v0, i_v1, i_edge) in face.vertices_with_edges() {
            let edge = &self.papercraft.model()[i_edge];
            let edge_status = self.papercraft.edge_status(i_edge);
            let draw_tab = match edge_status {
                EdgeStatus::Hidden => {
                    // hidden edges are never drawn
                    continue;
                }
                EdgeStatus::Cut(c) => {
                    // cut edges are always drawn, the tab on c == face_sign
                    tab_style != TabStyle::None && c == edge.face_sign(i_face)
                }
                EdgeStatus::Joined => {
                    // joined edges are drawn from one side only, no matter which one
                    if !edge.face_sign(i_face) {
                        continue;
                    }
                    // but never with a tab
                    false
                }
            };

            let plane = self.papercraft.face_plane(face);
            //let selected_edge = self.selected_edge == Some(i_edge);
            let v0 = &self.papercraft.model()[i_v0];
            let p0 = plane.project(&v0.pos());
            let pos0 = m.transform_point(Point2::from_vec(p0)).to_vec();

            let v1 = &self.papercraft.model()[i_v1];
            let p1 = plane.project(&v1.pos());
            let pos1 = m.transform_point(Point2::from_vec(p1)).to_vec();

            //Dotted lines are drawn for negative 3d angles (valleys) if the edge is joined or
            //cut with a tab
            let dotted = if edge_status == EdgeStatus::Joined || draw_tab {
                let angle_3d = self.papercraft.model().edge_angle(i_edge);
                angle_3d < Rad(0.0)
            } else {
                false
            };
            let v = pos1 - pos0;
            let fold_faces = edge_status == EdgeStatus::Joined;
            let v2d = MVertex2DLine {
                pos: pos0,
                line_dash: 0.0,
                width_left: if fold_faces { CREASE_LINE_WIDTH / 2.0 } else if draw_tab { CREASE_LINE_WIDTH } else { BORDER_LINE_WIDTH },
                width_right: if fold_faces { CREASE_LINE_WIDTH / 2.0 } else { 0.0 },
            };

            let v_len = v.magnitude();
            let (new_lines_, new_lines_2_);

            let fold_factor = self.papercraft.options().fold_line_len / v_len;
            let visible_line =
                if edge_status == EdgeStatus::Joined || draw_tab {
                    match self.papercraft.options().fold_style {
                        paper::FoldStyle::Full => (Some(0.0), None),
                        paper::FoldStyle::FullAndOut => (Some(fold_factor), None),
                        paper::FoldStyle::Out => (Some(fold_factor), Some(0.0)),
                        paper::FoldStyle::In => (Some(0.0), Some(fold_factor)),
                        paper::FoldStyle::InAndOut => (Some(fold_factor), Some(fold_factor)),
                        paper::FoldStyle::None => (None, None),
                    }
                } else {
                    (Some(0.0), None)
                };

            let new_lines: &[_] = match visible_line  {
                (None, None) | (None, Some(_)) => { &[] }
                (Some(f), None) => {
                    let vn = v * f;
                    let v0 = MVertex2DLine {
                        pos: pos0 - vn,
                        line_dash: 0.0,
                        .. v2d
                    };
                    let v1 = MVertex2DLine {
                        pos: pos1 + vn,
                        line_dash: if dotted { v_len * (1.0 + 2.0 * f) } else { 0.0 },
                        .. v0
                    };
                    new_lines_ = [v0, v1];
                    &new_lines_
                }
                (Some(f_a), Some(f_b)) => {
                    let vn_a = v * f_a;
                    let vn_b = v * f_b;
                    let va0 = MVertex2DLine {
                        pos: pos0 - vn_a,
                        line_dash: 0.0,
                        .. v2d
                    };
                    let va1 = MVertex2DLine {
                        pos: pos0 + vn_b,
                        line_dash: if dotted { v_len * (f_a + f_b) } else { 0.0 },
                        .. v2d
                    };
                    let vb0 = MVertex2DLine {
                        pos: pos1 - vn_b,
                        line_dash: 0.0,
                        .. v2d
                    };
                    let vb1 = MVertex2DLine {
                        pos: pos1 + vn_a,
                        line_dash: va1.line_dash,
                        .. v2d
                    };
                    new_lines_2_ = [va0, va1, vb0, vb1];
                    &new_lines_2_
                }
            };

            let edge_container = if fold_faces {
                &mut args.vertices_edge_crease
            } else {
                &mut args.vertices_edge_border
            };
            edge_container.extend_from_slice(new_lines);

            // Draw the tab?
            if draw_tab {
                let i_face_b = match edge.faces() {
                    (fa, Some(fb)) if i_face == fb => Some(fa),
                    (fa, Some(fb)) if i_face == fa => Some(fb),
                    _ => None
                };
                if let Some(i_face_b) = i_face_b {
                    let tab = self.papercraft.options().tab_width;
                    let tab_angle = Rad::from(Deg(self.papercraft.options().tab_angle));
                    let face_b = &self.papercraft.model()[i_face_b];

                    //swap the angles because this is from the POV of the other face
                    let (angle_1, angle_0) = self.papercraft.flat_face_angles(i_face_b, i_edge);
                    let angle_0 = Rad(angle_0.0.min(tab_angle.0));
                    let angle_1 = Rad(angle_1.0.min(tab_angle.0));

                    let v = pos1 - pos0;
                    let tan_0 = angle_0.cot();
                    let tan_1 = angle_1.cot();
                    let v_len = v.magnitude();

                    let mut tab_h_0 = tan_0 * tab;
                    let mut tab_h_1 = tan_1 * tab;
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
                        MVertex2DLine {
                            pos: pos0,
                            line_dash: 0.0,
                            width_left: TAB_LINE_WIDTH,
                            width_right: 0.0,
                        },
                        MVertex2DLine {
                            pos: pos0 + n + v_0,
                            line_dash: 0.0,
                            width_left: TAB_LINE_WIDTH,
                            width_right: 0.0,
                        },
                        MVertex2DLine {
                            pos: pos1 + n - v_1,
                            line_dash: 0.0,
                            width_left: TAB_LINE_WIDTH,
                            width_right: 0.0,
                        },
                        MVertex2DLine {
                            pos: pos1,
                            line_dash: 0.0,
                            width_left: TAB_LINE_WIDTH,
                            width_right: 0.0,
                        },
                    ];
                    let p = if just_one_tri {
                        //The unneeded vertex is actually [2], so remove that copying the [3] over
                        p[2] = p[3];
                        args.vertices_tab_edge.extend_from_slice(&[p[0], p[1], p[1], p[2]]);
                        &mut p[..3]
                    } else {
                        args.vertices_tab_edge.extend_from_slice(&[p[0], p[1], p[1], p[2], p[2], p[3]]);
                        &mut p[..]
                    };

                    if tab_style == TabStyle::Textured || tab_style == TabStyle::HalfTextured {
                        //Now we have to compute the texture coordinates of `p` in the adjacent face
                        let plane_b = self.papercraft.face_plane(face_b);
                        let vs_b = face_b.index_vertices().map(|v| {
                            let v = &self.papercraft.model()[v];
                            let p = plane_b.project(&v.pos());
                            (v, p)
                        });
                        let mx_b = m * self.papercraft.model().face_to_face_edge_matrix(self.papercraft.options().scale, edge, face, face_b);
                        let mx_b_inv = mx_b.invert().unwrap();
                        let mx_basis = Matrix2::from_cols(vs_b[1].1 - vs_b[0].1, vs_b[2].1 - vs_b[0].1).invert().unwrap();

                        // mx_b_inv converts from paper to local face_b coordinates
                        // mx_basis converts from local face_b to edge-relative coordinates, where position of the tri vertices are [(0,0), (1,0), (0,1)]
                        // mxx do both convertions at once
                        let mxx = Matrix3::from(mx_basis) * mx_b_inv;

                        let uvs: Vec<Vector2> = p.iter().map(|px| {
                            //vlocal is in edge-relative coordinates, that can be used to interpolate between UVs
                            let vlocal = mxx.transform_point(Point2::from_vec(px.pos)).to_vec();
                            let uv0 = vs_b[0].0.uv();
                            let uv1 = vs_b[1].0.uv();
                            let uv2 = vs_b[2].0.uv();
                            uv0 + vlocal.x * (uv1 - uv0) + vlocal.y * (uv2 - uv0)
                        }).collect();

                        let mat = face_b.material();
                        let root_color = Rgba::new(1.0, 1.0, 1.0, 0.0);
                        let tip_color = Rgba::new(1.0, 1.0, 1.0, if tab_style == TabStyle::HalfTextured { 1.0 } else { 0.0 });
                        if just_one_tri {
                            args.vertices_tab.push(MVertex2DColor { pos: p[0].pos, uv: uvs[0], mat, color: root_color });
                            args.vertices_tab.push(MVertex2DColor { pos: p[1].pos, uv: uvs[1], mat, color: tip_color });
                            args.vertices_tab.push(MVertex2DColor { pos: p[2].pos, uv: uvs[2], mat, color: root_color });
                        } else {
                            let pp = [
                                MVertex2DColor { pos: p[0].pos, uv: uvs[0], mat, color: root_color },
                                MVertex2DColor { pos: p[1].pos, uv: uvs[1], mat, color: tip_color },
                                MVertex2DColor { pos: p[2].pos, uv: uvs[2], mat, color: tip_color },
                                MVertex2DColor { pos: p[3].pos, uv: uvs[3], mat, color: root_color },
                            ];
                            args.vertices_tab.extend_from_slice(&[pp[0], pp[2], pp[1], pp[0], pp[3], pp[2]]);
                        }
                    }
                }
            }
        }
    }

    fn paper_rebuild(&mut self) {
        //Maps VertexIndex in the model to index in vertices
        let mut args = PaperDrawFaceArgs::new(self.papercraft.model());

        for (_, island) in self.papercraft.islands() {
            self.papercraft.traverse_faces(island,
                |i_face, face, mx| {
                    self.paper_draw_face(face, i_face, mx, &mut args);
                    ControlFlow::Continue(())
                }
            );
        }

        if let Some(gl_objs) = &mut self.gl_objs {
            gl_objs.paper_vertices.set(args.vertices);
            gl_objs.paper_vertices_edge_border.set(args.vertices_edge_border);
            gl_objs.paper_vertices_edge_crease.set(args.vertices_edge_crease);
            gl_objs.paper_vertices_tab.set(args.vertices_tab);
            gl_objs.paper_vertices_tab_edge.set(args.vertices_tab_edge);

            gl_objs.paper_face_index = args.face_index;
        }
    }

    fn pages_rebuild(&mut self) {
        if let Some(gl_objs) = &mut self.gl_objs {
            let color = Rgba::new(1.0, 1.0, 1.0, 1.0);
            let mat = MaterialIndex::from(0);
            let mut page_vertices = Vec::new();
            let mut margin_vertices = Vec::new();
            let margin_line_width = 0.5;

            let page_size = Vector2::from(self.papercraft.options().page_size);
            let margin = self.papercraft.options().margin;
            let page_count = self.papercraft.options().pages;

            for page in 0 .. page_count {
                let page_pos = self.papercraft.options().page_position(page);

                let page_0 = MVertex2DColor {
                    pos: page_pos,
                    uv: Vector2::zero(),
                    mat,
                    color,
                };
                let page_2 = MVertex2DColor {
                    pos: page_pos + page_size,
                    uv: Vector2::zero(),
                    mat,
                    color,
                };
                let page_1 = MVertex2DColor {
                    pos: Vector2::new(page_2.pos.x, page_0.pos.y),
                    uv: Vector2::zero(),
                    mat,
                    color,
                };
                let page_3 = MVertex2DColor {
                    pos: Vector2::new(page_0.pos.x, page_2.pos.y),
                    uv: Vector2::zero(),
                    mat,
                    color,
                };
                page_vertices.extend_from_slice(&[page_0, page_2, page_1, page_0, page_3, page_2]);

                let mut margin_0 = MVertex2DLine {
                    pos: page_0.pos + Vector2::new(margin.1, margin.0),
                    line_dash: 0.0,
                    width_left: margin_line_width,
                    width_right: 0.0,
                };
                let mut margin_1 = MVertex2DLine {
                    pos: page_3.pos + Vector2::new(margin.1, -margin.3),
                    line_dash: 0.0,
                    width_left: margin_line_width,
                    width_right: 0.0,
                };
                let mut margin_2 = MVertex2DLine {
                    pos: page_2.pos + Vector2::new(-margin.2, -margin.3),
                    line_dash: 0.0,
                    width_left: margin_line_width,
                    width_right: 0.0,
                };
                let mut margin_3 = MVertex2DLine {
                    pos: page_1.pos + Vector2::new(-margin.2, margin.0),
                    line_dash: 0.0,
                    width_left: margin_line_width,
                    width_right: 0.0,
                };
                margin_0.line_dash = 0.0;
                margin_1.line_dash = page_size.y / 10.0;
                margin_vertices.extend_from_slice(&[margin_0, margin_1]);
                margin_1.line_dash = 0.0;
                margin_2.line_dash = page_size.x / 10.0;
                margin_vertices.extend_from_slice(&[margin_1, margin_2]);
                margin_2.line_dash = 0.0;
                margin_3.line_dash = page_size.y / 10.0;
                margin_vertices.extend_from_slice(&[margin_2, margin_3]);
                margin_3.line_dash = 0.0;
                margin_0.line_dash = page_size.x / 10.0;
                margin_vertices.extend_from_slice(&[margin_3, margin_0]);
            }
            gl_objs.paper_vertices_page.set(page_vertices);
            gl_objs.paper_vertices_margin.set(margin_vertices);
        }
    }

    fn scene_edge_rebuild(&mut self) {
        if let Some(gl_objs) = &mut self.gl_objs {
            let mut edges_joint = Vec::new();
            let mut edges_cut = Vec::new();
            for (i_edge, edge) in self.papercraft.model().edges() {
                let status = self.papercraft.edge_status(i_edge);
                if status == EdgeStatus::Hidden {
                    continue;
                }
                let cut = matches!(self.papercraft.edge_status(i_edge), EdgeStatus::Cut(_));
                let p0 = self.papercraft.model()[edge.v0()].pos();
                let p1 = self.papercraft.model()[edge.v1()].pos();

                let (edges, color) = if cut {
                    (&mut edges_cut, Rgba::new(1.0, 1.0, 1.0, 1.0))
                } else {
                    (&mut edges_joint, Rgba::new(0.0, 0.0, 0.0, 1.0))
                };
                edges.push(MVertex3DLine { pos: p0, color });
                edges.push(MVertex3DLine { pos: p1, color });
            }
            gl_objs.vertices_edge_joint.set(edges_joint);
            gl_objs.vertices_edge_cut.set(edges_cut);
        }
    }

    fn selection_rebuild(&mut self) {
        let gl_objs = match &mut self.gl_objs {
            Some(x) => x,
            None => return,
        };
        let n = gl_objs.vertices_sel.len();
        for i in 0..n {
            gl_objs.vertices_sel[i] = MSTATUS_UNSEL;
            gl_objs.paper_vertices_sel[i] = MStatus2D { color: MSTATUS_UNSEL.color };
        }
        let top = self.xray_selection as u8;
        for &sel_island in &self.selected_islands {
            if let Some(island) = self.papercraft.island_by_key(sel_island) {
                self.papercraft.traverse_faces_no_matrix(island, |i_face| {
                    let pos = 3 * usize::from(i_face);
                    for i in pos .. pos + 3 {
                        gl_objs.vertices_sel[i] = MStatus3D { color: MSTATUS_SEL.color, top };
                    }
                    let pos = 3 * gl_objs.paper_face_index[usize::from(i_face)] as usize;
                    for i in pos .. pos + 3 {
                        gl_objs.paper_vertices_sel[i] = MStatus2D { color: MSTATUS_SEL.color };
                    }
                    ControlFlow::Continue(())
                });
            }
        }
        if let Some(i_sel_face) = self.selected_face {
            for i_face in self.papercraft.get_flat_faces(i_sel_face) {
                let pos = 3 * usize::from(i_face);
                for i in pos .. pos + 3 {
                    gl_objs.vertices_sel[i] = MStatus3D { color: MSTATUS_HI.color, top };
                }
                let pos = 3 * gl_objs.paper_face_index[usize::from(i_face)] as usize;
                for i in pos .. pos + 3 {
                    gl_objs.paper_vertices_sel[i] = MStatus2D { color: MSTATUS_HI.color };
                }
        }
        }
        if let Some(i_sel_edge) = self.selected_edge {
            let mut edges_sel = Vec::new();
            let color = if self.mode == MouseMode::Edge { Rgba::new(0.5, 0.5, 1.0, 1.0) } else { Rgba::new(0.0, 0.5, 0.0, 1.0) };
            let edge = &self.papercraft.model()[i_sel_edge];
            let p0 = self.papercraft.model()[edge.v0()].pos();
            let p1 = self.papercraft.model()[edge.v1()].pos();
            edges_sel.push(MVertex3DLine { pos: p0, color });
            edges_sel.push(MVertex3DLine { pos: p1, color });
            gl_objs.vertices_edge_sel.set(edges_sel);

            let (i_face_a, i_face_b) = edge.faces();

            // Returns the 2D vertices of i_sel_edge that belong to face i_face
            let get_vx = |i_face: FaceIndex| {
                let face_a = &self.papercraft.model()[i_face];
                let idx_face = 3 * gl_objs.paper_face_index[usize::from(i_face)] as usize;
                let idx_edge = face_a.index_edges().iter().position(|&e| e == i_sel_edge).unwrap();
                let v0 = &gl_objs.paper_vertices[idx_face + idx_edge];
                let v1 = &gl_objs.paper_vertices[idx_face + (idx_edge + 1) % 3];
                (v0, v1)
            };

            let mut edge_sel = Vec::with_capacity(6);
            let line_width = LINE_SEL_WIDTH / 2.0 / self.trans_paper.mx[0][0];

            let (v0, v1) = get_vx(i_face_a);
            edge_sel.extend_from_slice(&[
                MVertex2DLine {
                    pos: v0.pos,
                    line_dash: 0.0,
                    width_left: line_width,
                    width_right: line_width,
                },
                MVertex2DLine {
                    pos: v1.pos,
                    line_dash: 0.0,
                    width_left: line_width,
                    width_right: line_width,
                },
            ]);
            if let Some(i_face_b) = i_face_b {
                let (vb0, vb1) = get_vx(i_face_b);
                edge_sel.extend_from_slice(&[
                    MVertex2DLine {
                        pos: vb0.pos,
                        line_dash: 0.0,
                        width_left: line_width,
                        width_right: line_width,
                    },
                    MVertex2DLine {
                        pos: vb1.pos,
                        line_dash: 0.0,
                        width_left: line_width,
                        width_right: line_width,
                    },
                ]);
                let mut link_line = [
                    MVertex2DLine {
                        pos: (edge_sel[0].pos + edge_sel[1].pos) / 2.0,
                        line_dash: 0.0,
                        width_left: line_width,
                        width_right: line_width,
                    },
                    MVertex2DLine {
                        pos: (edge_sel[2].pos + edge_sel[3].pos) / 2.0,
                        line_dash: 0.0,
                        width_left: line_width,
                        width_right: line_width,
                    },
                ];
                link_line[1].line_dash = (link_line[1].pos - link_line[0].pos).magnitude();
                edge_sel.extend_from_slice(&link_line);
            }
            gl_objs.paper_vertices_edge_sel.set(edge_sel);
        }
    }

    #[must_use]
    fn set_selection(&mut self, selection: ClickResult, clicked: bool, add_to_sel: bool) -> RebuildFlags {
        let mut island_changed = false;
        let (new_edge, new_face) = match selection {
            ClickResult::None => {
                if clicked && !add_to_sel  && !self.selected_islands.is_empty() {
                    self.selected_islands.clear();
                    island_changed = true;
                }
                (None, None)
            }
            ClickResult::Face(i_face) => {
                if clicked {
                    let island = self.papercraft.island_by_face(i_face);
                    if add_to_sel {
                        if let Some(_n) = self.selected_islands.iter().position(|i| *i == island) {
                            //unselect the island?
                        } else {
                            self.selected_islands.push(island);
                            island_changed = true;
                        }
                    } else {
                        self.selected_islands = vec![island];
                        island_changed = true;
                    }
                }
                (None, Some(i_face))
            }
            ClickResult::Edge(i_edge, _) => {
                (Some(i_edge), None)
            }
        };
        let rebuild = if island_changed || self.selected_edge != new_edge || self.selected_face != new_face {
            RebuildFlags::SELECTION
        } else {
            RebuildFlags::empty()
        };
        self.selected_edge = new_edge;
        self.selected_face = new_face;
        rebuild
    }

    #[must_use]
    fn edge_toggle_cut(&mut self, i_edge: EdgeIndex, priority_face: Option<FaceIndex>) -> Option<Vec<UndoAction>> {
        match self.papercraft.edge_status(i_edge) {
            EdgeStatus::Hidden => { None }
            EdgeStatus::Joined => {
                let offset = self.papercraft.options().tab_width * 2.0;
                self.papercraft.edge_cut(i_edge, Some(offset));
                Some(vec![UndoAction::EdgeCut { i_edge }])
            }
            EdgeStatus::Cut(_) => {
                let renames = self.papercraft.edge_join(i_edge, priority_face);
                if renames.is_empty() {
                    return None;
                }
                let undo_actions = renames
                    .iter()
                    .map(|(_, join_result)| {
                        UndoAction::EdgeJoin { join_result: *join_result }
                    })
                    .collect();
                self.islands_renamed(&renames);
                Some(undo_actions)
            }
        }
    }

    #[must_use]
    fn try_join_strip(&mut self, i_edge: EdgeIndex) -> Option<Vec<UndoAction>> {
        let renames = self.papercraft.try_join_strip(i_edge);
        if renames.is_empty() {
            return None;
        }

        let undo_actions = renames
            .iter()
            .map(|(_, join_result)| {
                UndoAction::EdgeJoin { join_result: *join_result }
            })
            .collect();
        self.islands_renamed(&renames);
        Some(undo_actions)
    }

    fn islands_renamed(&mut self, renames: &HashMap<IslandKey, JoinResult>) {
        for x in &mut self.selected_islands {
            while let Some(jr) = renames.get(x) {
                *x = jr.i_island;
            }
        }
    }

    fn scene_analyze_click(&self, mode: MouseMode, size: Vector2, pos: Vector2) -> ClickResult {
        let x = (pos.x / size.x) * 2.0 - 1.0;
        let y = -((pos.y / size.y) * 2.0 - 1.0);
        let click = Point3::new(x, y, 1.0);
        let height = size.y * self.trans_scene.obj[1][1];

        let click_camera = self.trans_scene.persp_inv.transform_point(click);
        let click_obj = self.trans_scene.view_inv.transform_point(click_camera);
        let camera_obj = self.trans_scene.view_inv.transform_point(Point3::new(0.0, 0.0, 0.0));

        let ray = (camera_obj.to_vec(), click_obj.to_vec());

        //Faces has to be checked both in Edge and Face mode, because Edges can be hidden by a face.
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

        if mode == MouseMode::Face {
            return match hit_face {
                None => ClickResult::None,
                Some((f, _)) => ClickResult::Face(f),
            };
        }

        let mut hit_edge = None;
        for (i_edge, edge) in self.papercraft.model().edges() {
            match (self.papercraft.edge_status(i_edge), mode) {
                (EdgeStatus::Hidden, _) => continue,
                (EdgeStatus::Joined, MouseMode::Tab) => continue,
                _ => (),
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
            if new_dist > 5.0 {
                continue;
            }

            // If there is a face 99% nearer this edge, it is hidden, probably, so it does not count
            match hit_face {
                Some((_, p)) if p < 0.99 * ray_hit => { continue; }
                _ => {}
            }

            hit_edge = Some((i_edge, ray_hit, new_dist));
        }

        // Edge has priority
        match (hit_edge, hit_face) {
            (Some((e, _, _)), _) => ClickResult::Edge(e, None),
            (None, Some((f, _))) => ClickResult::Face(f),
            (None, None) => ClickResult::None,
        }
    }

    fn paper_analyze_click(&self, mode: MouseMode, size: Vector2, pos: Vector2) -> ClickResult {
        let click = self.trans_paper.paper_click(size, pos);
        let mx = self.trans_paper.ortho * self.trans_paper.mx;

        let mut hit_edge = None;
        let mut hit_face = None;

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
                    if hit_face.is_none() && util_3d::point_in_triangle(click, tri[0], tri[1], tri[2]) {
                        hit_face = Some(i_face);
                    }
                    match mode {
                        MouseMode::Face => { }
                        MouseMode::Edge | MouseMode::Tab => {
                            for i_edge in face.index_edges() {
                                match (self.papercraft.edge_status(i_edge), mode) {
                                    (EdgeStatus::Hidden, _) => continue,
                                    (EdgeStatus::Joined, MouseMode::Tab) => continue,
                                    _ => (),
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
                                match &hit_edge {
                                    None => {
                                        hit_edge = Some((d, i_edge, i_face));
                                    }
                                    &Some((d_prev, _, _)) if d < d_prev => {
                                        hit_edge = Some((d, i_edge, i_face));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    ControlFlow::Continue(())
                }
            );
        }

        // Edge has priority
        match (hit_edge, hit_face) {
            (Some((_d, i_edge, i_face)), _) => ClickResult::Edge(i_edge, Some(i_face)),
            (None, Some(i_face)) => ClickResult::Face(i_face),
            (None, None) => ClickResult::None,
        }
    }

    fn scene_motion_notify_event(&mut self, size: Vector2, pos: Vector2, ev: &gdk::EventMotion) -> RebuildFlags {
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
            RebuildFlags::SCENE_REDRAW
        } else if ev.state().contains(gdk::ModifierType::BUTTON2_MASK) {
            let delta = delta / 50.0;
            self.trans_scene.location += Vector3::new(delta.x, -delta.y, 0.0);
            self.trans_scene.recompute_obj();
            RebuildFlags::SCENE_REDRAW
        } else {
            let selection = self.scene_analyze_click(self.mode, size, pos);
            self.set_selection(selection, false, false)
        }
    }

    #[must_use]
    fn paper_motion_notify_event(&mut self, size: Vector2, pos: Vector2, ev_state: gdk::ModifierType) -> RebuildFlags {
        let delta = pos - self.last_cursor_pos;
        self.last_cursor_pos = pos;
        if ev_state.contains(gdk::ModifierType::BUTTON2_MASK) {
            self.trans_paper.mx = Matrix3::from_translation(delta) * self.trans_paper.mx;
            RebuildFlags::PAPER_REDRAW
        } else if ev_state.contains(gdk::ModifierType::BUTTON1_MASK) && self.grabbed_island {
            if !self.selected_islands.is_empty() {
                let rotating = ev_state.contains(gdk::ModifierType::SHIFT_MASK);

                if !rotating {
                    if let Some(c) = &mut self.rotation_center {
                        *c += delta;
                    }
                }

                for &i_island in &self.selected_islands {
                    if let Some(island) = self.papercraft.island_by_key_mut(i_island) {
                        let delta_scaled = <Matrix3 as Transform<Point2>>::inverse_transform_vector(&self.trans_paper.mx, delta).unwrap();
                        if rotating {
                            // Rotate island
                            let center = *self.rotation_center.get_or_insert(pos);
                            //Rotating when the pointer is very near to the center or rotation the angle could go crazy, so disable it
                            if (pos - center).magnitude() > 10.0 {
                                let pcenter = self.trans_paper.paper_click(size, center);
                                let ppos_prev = self.trans_paper.paper_click(size, pos - delta);
                                let ppos = self.trans_paper.paper_click(size, pos);
                                let angle = (ppos_prev - pcenter).angle(ppos - pcenter);
                                island.rotate(angle, pcenter);
                            }
                        } else {
                            // Move island
                            island.translate(delta_scaled);
                        }
                    }
                }
                RebuildFlags::PAPER
            } else {
                RebuildFlags::empty()
            }
        } else {
            let selection = self.paper_analyze_click(self.mode, size, pos);
            self.set_selection(selection, false, false)
        }
    }

    #[must_use]
    fn pack_islands(&mut self) -> Vec<UndoAction> {
        let undo_actions = self.papercraft.islands()
            .map(|(_, island)| {
                UndoAction::IslandMove{ i_root: island.root_face(), prev_rot: island.rotation(), prev_loc: island.location() }
            })
            .collect();
        self.papercraft.pack_islands();
        undo_actions
    }

    fn undo_action(&mut self) -> bool {
        //Do not undo while grabbing or the stack will be messed up
        if self.grabbed_island {
            return false;
        }

        let action_pack = match self.undo_stack.pop() {
            None => return false,
            Some(a) => a,
        };

        for action in action_pack.into_iter().rev() {
            match action {
                UndoAction::IslandMove { i_root, prev_rot, prev_loc } => {
                    if let Some(i_island) = self.papercraft.island_by_root(i_root) {
                        let island = self.papercraft.island_by_key_mut(i_island).unwrap();
                        island.reset_transformation(i_root, prev_rot, prev_loc);
                    }
                }
                UndoAction::TabToggle { i_edge } => {
                    self.papercraft.edge_toggle_tab(i_edge);
                }
                UndoAction::EdgeCut { i_edge } => {
                    self.papercraft.edge_join(i_edge, None);
                }
                UndoAction::EdgeJoin { join_result } => {
                    self.papercraft.edge_cut(join_result.i_edge, None);
                    let i_prev_island = self.papercraft.island_by_face(join_result.prev_root);
                    let island = self.papercraft.island_by_key_mut(i_prev_island).unwrap();

                    island.reset_transformation(join_result.prev_root, join_result.prev_rot, join_result.prev_loc);
                }
                UndoAction::DocConfig { options } => {
                    self.papercraft.set_options(options);
                }
            }
        }
        true
    }
}

impl GlobalContext {
    fn import_waveobj(&mut self, file_name: impl AsRef<Path>) -> Result<()> {
        let papercraft = Papercraft::import_waveobj(file_name.as_ref())
            .with_context(|| format!("Error reading Wavefront file {}", file_name.as_ref().display()))?;
        self.from_papercraft(papercraft, Some(file_name.as_ref()));
        Ok(())
    }
    fn from_papercraft(&mut self, papercraft: Papercraft, file_name: Option<&Path>) {
        let sz_scene = self.wscene.size_as_vector();
        let sz_paper = self.wpaper.size_as_vector();
        self.data = PapercraftContext::from_papercraft(papercraft, file_name, sz_scene, sz_paper);
        self.push_undo_action(Vec::new());
        self.add_rebuild(RebuildFlags::ALL);
    }
    fn update_from_obj(&mut self, mut new_papercraft: Papercraft) {
        let file_name = self.data.file_name.clone();
        new_papercraft.update_from_obj(&self.data.papercraft);
        let tp = self.data.trans_paper.clone();
        let ts = self.data.trans_scene.clone();
        self.from_papercraft(new_papercraft, file_name.as_deref());
        self.data.trans_paper = tp;
        self.data.trans_scene = ts;
    }
    fn add_rebuild(&mut self, flags: RebuildFlags) {
        self.data.rebuild |= flags;
        if flags.intersects(RebuildFlags::ANY_REDRAW_PAPER) {
            self.wpaper.queue_render();
        }
        if flags.intersects(RebuildFlags::ANY_REDRAW_SCENE) {
            self.wscene.queue_render();
        }
    }
    fn save(&self, file_name: impl AsRef<Path>) -> Result<()> {
        let f = std::fs::File::create(&file_name)
            .with_context(|| format!("Error creating file {}", file_name.as_ref().display()))?;

        let f = std::io::BufWriter::new(f);
        self.data.papercraft.save(f)
            .with_context(|| format!("Error saving file {}", file_name.as_ref().display()))?;
        Ok(())
    }

    fn set_title(&self, file_name: Option<impl AsRef<Path>>) {
        let unsaved = if self.data.modified { "*" } else { "" };
        let app_name = "Papercraft";
        let title = match &file_name {
            Some(f) =>
                format!("{unsaved}{} - {app_name}", f.as_ref().display()),
            None =>
                format!("{unsaved} - {app_name}"),
        };
        self.top_window.set_title(&title);
    }

    fn confirm_if_modified(ctx: &RefCell<GlobalContext>, title: &str) -> bool {
        if !ctx.borrow().data.modified {
            return true;
        }
        let dlg = gtk::MessageDialog::builder()
            .title(title)
            .transient_for(&ctx.borrow().top_window)
            .text("The model has not been save, continue anyway?")
            .message_type(gtk::MessageType::Question)
            .buttons(gtk::ButtonsType::OkCancel)
            .build();
        let res = dlg.run();
        unsafe { dlg.destroy(); }
        res == gtk::ResponseType::Ok
    }

    fn set_scroll_timer(&mut self, tmr: Option<glib::SourceId>) {
        if let Some(t) = self.data.scroll_timer.take() {
            t.remove();
        }
        self.data.scroll_timer = tmr;
    }

    fn push_undo_action(&mut self, action: Vec<UndoAction>) {
        if !action.is_empty() {
            self.data.undo_stack.push(action);
        }
        if !self.data.modified {
            self.data.modified = true;
        }
        self.set_title(self.data.file_name.as_ref());
    }

    fn build_gl_fixs(&mut self) -> Result<()> {
        if self.gl_fixs.is_none() {
            gl_loader::init_gl();
            gl::load_with(|s| gl_loader::get_proc_address(s) as _);

            let prg_scene_solid = util_gl::program_from_source(include_str!("shaders/scene_solid.glsl")).with_context(|| "scene_solid")?;
            let prg_scene_line = util_gl::program_from_source(include_str!("shaders/scene_line.glsl")).with_context(|| "scene_line")?;
            let prg_paper_solid = util_gl::program_from_source(include_str!("shaders/paper_solid.glsl")).with_context(|| "paper_solid")?;
            let prg_paper_line = util_gl::program_from_source(include_str!("shaders/paper_line.glsl")).with_context(|| "paper_line")?;
            let prg_quad = util_gl::program_from_source(include_str!("shaders/quad.glsl")).with_context(|| "quad")?;

            self.gl_fixs = Some(GLFixedObjects {
                vao_scene: None,
                vao_paper: None,
                fbo_paper: None,
                rbo_paper: None,
                prg_scene_solid,
                prg_scene_line,
                prg_paper_solid,
                prg_paper_line,
                prg_quad,
            });
        }
        Ok(())
    }

    fn scene_render(&mut self) {
        self.data.pre_render();
        let gl_objs = self.data.gl_objs.as_ref().unwrap();
        let gl_fixs = self.gl_fixs.as_ref().unwrap();

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
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            gl::BindVertexArray(gl_fixs.vao_scene.as_ref().unwrap().id());
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

    fn paper_render(&mut self) {
        self.data.pre_render();
        let gl_objs = self.data.gl_objs.as_ref().unwrap();
        let gl_fixs = self.gl_fixs.as_ref().unwrap();

        let mut u = Uniforms2D {
            m: self.data.trans_paper.ortho * self.data.trans_paper.mx,
            tex: 0,
            frac_dash: 0.5,
            line_color: Rgba::new(0.0, 0.0, 0.0, 0.0),
            texturize: 0,
        };

        let alloc = self.wpaper.allocation();
        let width = alloc.width();
        let height = alloc.height();

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

            gl::BindVertexArray(gl_fixs.vao_paper.as_ref().unwrap().id());
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

    fn generate_pdf(&self, file_name: impl AsRef<Path>) -> Result<()> {
        let options = self.data.papercraft.options();
        let resolution = options.resolution as f32;
        let (_margin_top, margin_left, margin_right, margin_bottom) = options.margin;
        let page_size_mm = Vector2::from(options.page_size);
        let page_size_inches = page_size_mm / 25.4;
        let page_size_dots = page_size_inches * 72.0;
        let page_size_pixels = page_size_inches * resolution;
        let page_size_pixels = cgmath::Vector2::new(page_size_pixels.x as i32, page_size_pixels.y as i32);

        let pixbuf = gdk_pixbuf::Pixbuf::new(gdk_pixbuf::Colorspace::Rgb, true, 8, page_size_pixels.x, page_size_pixels.y)
            .ok_or_else(|| anyhow!("Unable to create output pixbuf"))?;
        let pdf = cairo::PdfSurface::new(page_size_dots.x as f64, page_size_dots.y as f64, file_name)?;
        let title = match &self.data.file_name {
            Some(f) => f.file_stem().map(|s| s.to_string_lossy()).unwrap_or_else(|| "".into()),
            None => "untitled".into()
        };
        let _ = pdf.set_metadata(cairo::PdfMetadata::Title, &title);
        let _ = pdf.set_metadata(cairo::PdfMetadata::Creator, signature());
        let cr = cairo::Context::new(&pdf)?;

        unsafe {
            self.wpaper.make_current();

            gl::PixelStorei(gl::PACK_ROW_LENGTH, pixbuf.rowstride() / 4);

            let fbo = glr::Framebuffer::generate();
            let rbo = glr::Renderbuffer::generate();

            let draw_fb_binder = BinderDrawFramebuffer::bind(&fbo);
            let read_fb_binder = BinderReadFramebuffer::bind(&fbo);
            let rb_binder = BinderRenderbuffer::bind(&rbo);

            let samples = glr::try_renderbuffer_storage_multisample(rb_binder.target(), gl::RGBA8, page_size_pixels.x, page_size_pixels.y);
            let rbo_fbo_no_aa = if samples.is_none() {
                println!("No multisample!");
                gl::RenderbufferStorage(rb_binder.target(), gl::RGBA8, page_size_pixels.x, page_size_pixels.y);
                None
            } else {
                // multisample buffers cannot be read directly, it has to be copied to a regular one.
                let rbo2 = glr::Renderbuffer::generate();
                rb_binder.rebind(&rbo2);
                gl::RenderbufferStorage(rb_binder.target(), gl::RGBA8, page_size_pixels.x, page_size_pixels.y);

                let fbo2 = glr::Framebuffer::generate();
                read_fb_binder.rebind(&fbo2);
                gl::FramebufferRenderbuffer(read_fb_binder.target(), gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, rbo2.id());
                Some((rbo2, fbo2))
            };
            gl::FramebufferRenderbuffer(draw_fb_binder.target(), gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, rbo.id());
            let _vp = glr::PushViewport::push(0, 0, page_size_pixels.x, page_size_pixels.y);

            gl::ClearColor(1.0, 1.0, 1.0, 0.0);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            let gl_objs = self.data.gl_objs.as_ref().unwrap();
            let gl_fixs = self.gl_fixs.as_ref().unwrap();

            let mut texturize = 0;

            gl::BindVertexArray(gl_fixs.vao_paper.as_ref().unwrap().id());
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
                let data = pixbuf.pixels();
                gl::ReadPixels(0, 0, page_size_pixels.x, page_size_pixels.y, gl::RGBA, gl::UNSIGNED_BYTE, data.as_mut_ptr() as *mut _);

                cr.set_source_pixbuf(&pixbuf, 0.0, 0.0);
                let pat = cr.source();
                let mut mc = cairo::Matrix::identity();
                let scale = resolution / 72.0;
                mc.scale(scale as f64, scale as f64);
                pat.set_matrix(mc);

                let _ = cr.paint();

                let _ = cr.show_page();
                //let _ = pixbuf.savev("test.png", "png", &[]);
            }
            gl::PixelStorei(gl::PACK_ROW_LENGTH, 0);
            drop(cr);
            drop(pdf);
        }
        Ok(())
    }
}

fn signature() -> &'static str {
    "Created with Papercraft. https://github.com/rodrigorc/papercraft"
}
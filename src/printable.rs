use super::*;
use anyhow::Result;
use rayon::prelude::*;

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

impl GlobalContext {
    pub fn generate_printable(&mut self, ui: &Ui, file_name: &Path) -> Result<()> {
        // Rebuild everything, just in case
        //TODO: should pass show_texts as argument?
        let old_show_texts = self.data.ui.show_texts;
        self.data.ui.show_texts = true;
        self.pre_render_flags(ui, RebuildFlags::all());
        self.data.ui.show_texts = old_show_texts;

        let res = match file_name
            .extension()
            .map(|s| s.to_string_lossy().into_owned().to_ascii_lowercase())
            .as_deref()
        {
            Some("pdf") => self.generate_pdf(file_name),
            Some("svg") => self.generate_svg(file_name),
            Some("png") => {
                let text_tex_id = Renderer::unmap_tex(ui.font_atlas().texture_id());
                self.generate_png(text_tex_id, file_name)
            }
            _ => anyhow::bail!(
                "{}",
                tr!(
                    "Don't know how to write the format of {}",
                    file_name.display()
                )
            ),
        };
        res.with_context(|| tr!("Error exporting to {}", file_name.display()))?;
        self.last_export = file_name.to_path_buf();
        Ok(())
    }

    fn generate_pdf(&self, file_name: &Path) -> Result<()> {
        use lopdf::{
            content::{Content, Operation},
            dictionary,
            xref::XrefType,
            Document, Object, ObjectId, Stream, StringFormat,
        };

        let options = self.data.papercraft().options();
        let page_size_mm = Vector2::from(options.page_size);
        let edge_id_position = options.edge_id_position;

        let mut doc = Document::with_version("1.4");
        doc.reference_table.cross_reference_type = XrefType::CrossReferenceTable;

        let id_pages = doc.new_object_id();

        let id_font = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
            "Encoding" => "WinAnsiEncoding",
        });

        let mut pages = vec![];

        let (tx_compress, rx_compress) =
            std::sync::mpsc::channel::<(ObjectId, ObjectId, image::RgbaImage)>();
        let (tx_done, rx_done) = std::sync::mpsc::channel::<(ObjectId, Stream)>();

        rayon::spawn(move || {
            rx_compress
                .into_iter()
                .par_bridge()
                .flat_map(|(id_mask, id_image, pixbuf)| {
                    log::debug!("Splitting...");
                    let (width, height) = pixbuf.dimensions();
                    let mut rgb = vec![0; (3 * width * height) as usize];
                    let mut alpha = vec![0; (width * height) as usize];
                    for (n, px) in pixbuf.pixels().enumerate() {
                        let c = px.channels();
                        rgb[3 * n..][..3].copy_from_slice(&c[..3]);
                        alpha[n] = c[3];
                    }
                    let stream_mask = Stream::new(
                        dictionary! {
                            "Type" => "XObject",
                            "Subtype" => "Image",
                            "Width" => width,
                            "Height" => height,
                            "BitsPerComponent" => 8,
                            "ColorSpace" => "DeviceGray",
                        },
                        alpha,
                    );
                    let stream_image = Stream::new(
                        dictionary! {
                            "Type" => "XObject",
                            "Subtype" => "Image",
                            "Width" => width,
                            "Height" => height,
                            "BitsPerComponent" => 8,
                            "ColorSpace" => "DeviceRGB",
                            "SMask" => id_mask,
                        },
                        rgb,
                    );
                    log::debug!("Split");
                    [(id_mask, stream_mask), (id_image, stream_image)]
                })
                .for_each(|(id, mut stream)| {
                    log::debug!("Compressing...");
                    let _ = stream.compress();
                    log::debug!("Compressed");
                    tx_done.send((id, stream)).unwrap();
                });
            drop(tx_done);
        });

        self.generate_pages(None, |page, pixbuf, _, texts, _| {
            let write_texts = |ops: &mut Vec<Operation>| {
                if texts.is_empty() {
                    return;
                }
                // Begin Text
                ops.push(Operation::new("BT", Vec::new()));
                let mut last_font = None;
                for text in texts {
                    let size = text.size * 72.0 / 25.4;
                    // PDF fonts are just slightly larger than expected
                    let size = size / 1.1;
                    let x = text.pos.x;
                    // (0,0) is in lower-left
                    let y = page_size_mm.y - text.pos.y;
                    let (width, cps) = pdf_metrics::measure_helvetica(&text.text);
                    let width = width as f32 * text.size / 1000.0;
                    let dx = match text.align {
                        TextAlign::Near => 0.0,
                        TextAlign::Center => -width / 2.0,
                        TextAlign::Far => -width,
                    };
                    let cos = text.angle.cos();
                    let sin = text.angle.sin();
                    let x = x + dx * cos;
                    let y = y - dx * sin;

                    // Set font
                    if last_font != Some(size) {
                        last_font = Some(size);
                        ops.push(Operation::new("Tf", vec!["F1".into(), size.into()]));
                    }

                    let mx: Vec<Object> = vec![
                        cos.into(),
                        (-sin).into(),
                        sin.into(),
                        cos.into(),
                        (x * 72.0 / 25.4).into(),
                        (y * 72.0 / 25.4).into(),
                    ];
                    ops.push(Operation::new("Tm", mx));

                    let mut tj = Vec::new();
                    let mut codepoints = Vec::new();
                    for (kern, cp) in cps {
                        if kern != 0 {
                            if !codepoints.is_empty() {
                                tj.push(Object::String(
                                    std::mem::take(&mut codepoints),
                                    StringFormat::Literal,
                                ));
                            }
                            tj.push(kern.into());
                        }
                        if let Ok(c) = u8::try_from(cp) {
                            codepoints.push(c);
                        }
                    }
                    if !codepoints.is_empty() {
                        tj.push(Object::String(
                            std::mem::take(&mut codepoints),
                            StringFormat::Literal,
                        ));
                    }
                    match tj.len() {
                        0 => (),
                        1 => ops.push(Operation::new("Tj", tj)),
                        _ => ops.push(Operation::new("TJ", vec![Object::Array(tj)])),
                    }
                }
                // End Text
                ops.push(Operation::new("ET", Vec::new()));
            };

            let mut ops: Vec<Operation> = Vec::new();

            if edge_id_position != EdgeIdPosition::Inside {
                write_texts(&mut ops);
            }

            let img_name = format!("IMG{page}");

            ops.push(Operation::new("q", vec![]));
            let mx: Vec<Object> = vec![
                (page_size_mm.x * 72.0 / 25.4).into(),
                0.into(),
                0.into(),
                (page_size_mm.y * 72.0 / 25.4).into(),
                0.into(),
                0.into(),
            ];
            ops.push(Operation::new("cm", mx));
            ops.push(Operation::new("Do", vec![img_name.clone().into()]));
            ops.push(Operation::new("Q", vec![]));

            if edge_id_position == EdgeIdPosition::Inside {
                write_texts(&mut ops);
            }

            let content = Content { operations: ops };
            let id_content = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

            let id_mask = doc.new_object_id();
            let id_image = doc.new_object_id();
            tx_compress.send((id_mask, id_image, pixbuf)).unwrap();

            let id_resources = doc.add_object(dictionary! {
                "Font" => dictionary! {
                    "F1" => id_font,
                },
                "XObject" => dictionary! {
                    img_name => id_image,
                },
            });
            let id_page = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => id_pages,
                "Contents" => id_content,
                "Resources" => id_resources,
            });
            pages.push(id_page.into());
            Ok(())
        })?;

        drop(tx_compress);
        for (id, stream) in rx_done.into_iter() {
            doc.set_object(id, stream);
        }

        let pdf_pages = dictionary! {
            "Type" => "Pages",
            "Count" => pages.len() as i32,
            "Kids" => pages,
            "MediaBox" => vec![
                0.into(), 0.into(),
                (page_size_mm.x * 72.0 / 25.4).into(), (page_size_mm.y * 72.0 / 25.4).into()
            ],
        };
        doc.set_object(id_pages, pdf_pages);

        let id_catalog = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => id_pages,
        });
        doc.trailer.set("Root", id_catalog);

        let date = time::OffsetDateTime::now_utc();
        let s_date = format!(
            "D:{:04}{:02}{:02}{:02}{:02}{:02}Z",
            date.year(),
            u8::from(date.month()),
            date.day(),
            date.hour(),
            date.minute(),
            date.second(),
        );

        let id_info = doc.add_object(dictionary! {
            "Title" => Object::string_literal(self.title(false)),
            "Creator" => Object::string_literal(signature()),
            "CreationDate" => Object::string_literal(s_date.clone()),
            "ModDate" => Object::string_literal(s_date),
        });
        doc.trailer.set("Info", id_info);
        doc.compress();
        doc.save(file_name)?;
        Ok(())
    }

    fn generate_svg(&self, file_name: &Path) -> Result<()> {
        let options = self.data.papercraft().options();
        let edge_id_position = options.edge_id_position;

        let (tx_compress, rx_compress) =
            std::sync::mpsc::channel::<(u32, image::RgbaImage, Vec<u8>, Vec<u8>)>();
        let (tx_done, rx_done) = std::sync::mpsc::channel::<Result<()>>();

        let file_name = PathBuf::from(file_name);
        rayon::spawn(move || {
            let res: Result<()> = rx_compress.into_iter().par_bridge().try_for_each(
                |(page, pixbuf, prefix, suffix)| {
                    let name = file_name_for_page(&file_name, page);
                    // A try block would be nice here
                    ((|| -> Result<()> {
                        log::debug!("Saving page {}", name.display());

                        let out = std::fs::File::create(&name)?;
                        let mut out = std::io::BufWriter::new(out);

                        out.write_all(&prefix)?;

                        // Can't write directly the image to the file as a base64, because Image::write_to requires `Seek`, but `base64::EncoderWriter` doesn't implement it.
                        {
                            use base64::prelude::*;
                            let mut png = Vec::new();
                            let mut cpng = std::io::Cursor::new(&mut png);
                            pixbuf.write_to(&mut cpng, image::ImageFormat::Png)?;

                            let mut b64png =
                                base64::write::EncoderWriter::new(&mut out, &BASE64_STANDARD);
                            b64png.write_all(&png)?;
                            b64png.finish()?;
                        }

                        out.write_all(&suffix)?;

                        log::debug!("Saved page {}", name.display());
                        Ok(())
                    })())
                    .with_context(|| tr!("Error saving file {}", name.display()))
                },
            );
            // Send the possible error back to the main thread
            tx_done.send(res).unwrap();
            drop(tx_done);
        });

        self.generate_pages(None, |page, pixbuf, extra, texts, lines_by_island| {
            let page_size = Vector2::from(options.page_size);
            let in_page = options.is_in_page_fn(page);

            let mut prefix = Vec::new();

            writeln!(&mut prefix, r#"<?xml version="1.0" encoding="UTF-8" standalone="no"?>"#)?;
            writeln!(
                &mut prefix,
                r#"<svg width="{0}mm" height="{1}mm" viewBox="0 0 {0} {1}" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" xmlns:xlink="http://www.w3.org/1999/xlink">"#,
                page_size.x, page_size.y
            )?;

            let write_layer_text = |out: &mut Vec<u8>| -> Result<()> {
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
                write_layer_text(&mut prefix)?;
            }

            // begin layer Background
            writeln!(&mut prefix, r#"<g inkscape:label="Background" inkscape:groupmode="layer" id="Background">"#)?;
            write!(
                &mut prefix,
                r#"<image width="{}" height="{}" preserveAspectRatio="none" xlink:href="data:image/png;base64,"#,
                page_size.x, page_size.y)?;

            let mut suffix = Vec::new();
            writeln!(&mut suffix, r#"" id="background" x="0" y="0" style="display:inline"/>"#)?;

            writeln!(&mut suffix, r#"</g>"#)?;
            // end layer Background

            if edge_id_position == EdgeIdPosition::Inside {
                write_layer_text(&mut suffix)?;
            }

            // begin layer Cut
            writeln!(&mut suffix, r#"<g inkscape:label="Cut" inkscape:groupmode="layer" id="Cut" style="display:none">"#)?;
            for (idx, (i_island, lines)) in lines_by_island.iter().enumerate() {
                let perimeter = self.data.papercraft().island_perimeter(*i_island);
                let mut contour_points = Vec::with_capacity(perimeter.len());
                let mut touching = false;
                for peri in perimeter.iter() {
                    lines.lines_by_cut_info(extra.cut_info().unwrap(), peri.i_edge(), peri.face_sign(), |p0, _| {
                        let (is_in, p0) = in_page(p0);
                        touching |= is_in;
                        contour_points.push(p0);
                    });
                }
                if touching {
                    writeln!(&mut suffix, r#"<path style="fill:none;stroke:#000000;stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter" id="cut_{}" d=""#, idx)?;
                    write!(&mut suffix, r#"M "#)?;
                    for p in contour_points {
                        writeln!(&mut suffix, r#"{},{}"#, p.x, p.y)?;
                    }
                    writeln!(&mut suffix, r#"z"#)?;
                    writeln!(&mut suffix, r#"" />"#)?;
                }
            }
            writeln!(&mut suffix, r#"</g>"#)?;
            // end layer Cut

            // begin layer Fold
            writeln!(&mut suffix, r#"<g inkscape:label="Fold" inkscape:groupmode="layer" id="Fold" style="display:none">"#)?;
            for fold_kind in [EdgeDrawKind::Mountain, EdgeDrawKind::Valley] {
                writeln!(&mut suffix, r#"<g inkscape:label="{0}" inkscape:groupmode="layer" id="{0}">"#,
                    if fold_kind == EdgeDrawKind::Mountain { "Mountain"} else { "Valley" })?;
                for (idx, (_, lines)) in lines_by_island.iter().enumerate() {
                    let creases = lines.iter_crease(fold_kind);
                    // each crease can be checked for bounds individually
                    let page_creases = creases
                        .filter_map(|(a, b)| {
                            let (is_in_a, a) = in_page(a);
                            let (is_in_b, b) = in_page(b);
                            (is_in_a || is_in_b).then_some((a, b))
                        })
                        .collect::<Vec<_>>();
                    if !page_creases.is_empty() {
                        writeln!(&mut suffix, r#"<path style="fill:none;stroke:{1};stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter" id="{2}_{0}" d=""#,
                            idx,
                            if fold_kind == EdgeDrawKind::Mountain  { "#ff0000" } else { "#0000ff" },
                            if fold_kind == EdgeDrawKind::Mountain  { "foldm" } else { "foldv" }
                        )?;
                        for (a, b) in page_creases {
                            writeln!(&mut suffix, r#"M {},{} {},{}"#, a.x, a.y, b.x, b.y)?;
                        }
                        writeln!(&mut suffix, r#"" />"#)?;
                    }
                }
                writeln!(&mut suffix, r#"</g>"#)?;
            }
            writeln!(&mut suffix, r#"</g>"#)?;
            // end layer Fold

            writeln!(&mut suffix, r#"</svg>"#)?;

            tx_compress.send((page, pixbuf, prefix, suffix)).unwrap();
            Ok(())
        })?;

        drop(tx_compress);
        // Forward the error, if any
        rx_done.into_iter().collect::<Result<()>>()?;

        Ok(())
    }

    fn generate_png(&self, text_tex_id: Option<glow::Texture>, file_name: &Path) -> Result<()> {
        let (tx_compress, rx_compress) = std::sync::mpsc::channel::<(u32, image::RgbaImage)>();
        let (tx_done, rx_done) = std::sync::mpsc::channel::<Result<()>>();

        let file_name = PathBuf::from(file_name);
        rayon::spawn(move || {
            let res: Result<()> =
                rx_compress
                    .into_iter()
                    .par_bridge()
                    .try_for_each(|(page, pixbuf)| {
                        let name = file_name_for_page(&file_name, page);
                        // A try block would be nice here
                        ((|| -> Result<()> {
                            log::debug!("Saving page {}", name.display());
                            let f = std::fs::File::create(&name)?;
                            let mut f = std::io::BufWriter::new(f);
                            pixbuf.write_to(&mut f, image::ImageFormat::Png)?;
                            log::debug!("Saved page {}", name.display());
                            Ok(())
                        })())
                        .with_context(|| tr!("Error saving file {}", name.display()))
                    });
            // Send the possible error back to the main thread
            tx_done.send(res).unwrap();
            drop(tx_done);
        });

        self.generate_pages(text_tex_id, |page, pixbuf, _, _texts, _| {
            tx_compress.send((page, pixbuf)).unwrap();
            Ok(())
        })?;

        drop(tx_compress);
        // Forward the error, if any
        rx_done.into_iter().collect::<Result<()>>()?;

        Ok(())
    }

    fn generate_pages<F>(&self, text_tex_id: Option<glow::Texture>, mut do_page_fn: F) -> Result<()>
    where
        F: FnMut(
            u32,
            image::RgbaImage,
            &PaperDrawFaceArgsExtra,
            &[PrintableText],
            &[(IslandKey, PaperDrawFaceArgs)],
        ) -> Result<()>,
    {
        let options = self.data.papercraft().options();
        let (_margin_top, margin_left, margin_right, margin_bottom) = options.margin;
        let resolution = options.resolution as f32;
        let page_size_mm = Vector2::from(options.page_size);
        let page_size_inches = page_size_mm / 25.4;
        let page_size_pixels = page_size_inches * resolution;
        let page_size_pixels =
            cgmath::Vector2::new(page_size_pixels.x as i32, page_size_pixels.y as i32);

        unsafe {
            let fbo = glr::Framebuffer::generate(&self.gl)?;
            let rbo = glr::Renderbuffer::generate(&self.gl)?;

            let draw_fb_binder = BinderDrawFramebuffer::bind(&fbo);
            let read_fb_binder = BinderReadFramebuffer::bind(&fbo);
            let rb_binder = BinderRenderbuffer::bind(&rbo);
            self.gl.framebuffer_renderbuffer(
                draw_fb_binder.target(),
                glow::COLOR_ATTACHMENT0,
                glow::RENDERBUFFER,
                Some(rbo.id()),
            );

            let rbo_fbo_no_aa = 'check_aa: {
                // multisample buffers cannot be read directly, it has to be copied to a regular one.
                for samples in MULTISAMPLES {
                    // check if these many samples are usable
                    self.gl.renderbuffer_storage_multisample(
                        rb_binder.target(),
                        *samples,
                        glow::RGBA8,
                        page_size_pixels.x,
                        page_size_pixels.y,
                    );
                    if self.gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER)
                        != glow::FRAMEBUFFER_COMPLETE
                    {
                        continue;
                    }

                    // If using AA create another FBO/RBO to blit the antialiased image before reading
                    let rbo2 = glr::Renderbuffer::generate(&self.gl)?;
                    rb_binder.rebind(&rbo2);
                    self.gl.renderbuffer_storage(
                        rb_binder.target(),
                        glow::RGBA8,
                        page_size_pixels.x,
                        page_size_pixels.y,
                    );

                    let fbo2 = glr::Framebuffer::generate(&self.gl)?;
                    read_fb_binder.rebind(&fbo2);
                    self.gl.framebuffer_renderbuffer(
                        read_fb_binder.target(),
                        glow::COLOR_ATTACHMENT0,
                        glow::RENDERBUFFER,
                        Some(rbo2.id()),
                    );

                    break 'check_aa Some((rbo2, fbo2));
                }
                log::warn!("No multisample!");
                self.gl.renderbuffer_storage(
                    rb_binder.target(),
                    glow::RGBA8,
                    page_size_pixels.x,
                    page_size_pixels.y,
                );
                None
            };
            // Consume the possible error from the multisample above
            let _ = self.gl.get_error();

            let _vp =
                glr::PushViewport::push(&self.gl, 0, 0, page_size_pixels.x, page_size_pixels.y);

            // Cairo surfaces are alpha-premultiplied:
            // * The framebuffer will be premultiplied, but the input fragments are not.
            // * The clear color is set to transparent (premultiplied).
            // * In the screen DST_ALPHA does not matter, because the framebuffer is not
            //   transparent, but here we have to set it to the proper value: use separate blend
            //   functions or we'll get the alpha squared.
            self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
            self.gl.enable(glow::BLEND);
            self.gl.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
            );

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

            let mut texts = Vec::new();
            let (lines_by_island, extra) = self.data.lines_by_island();

            for page in 0..page_count {
                log::debug!("Rendering page {}", page + 1);

                // Start render
                self.gl.clear(glow::COLOR_BUFFER_BIT);
                let page_pos = options.page_position(page);
                let mt = Matrix3::from_translation(-page_pos);
                let mut u = Uniforms2D {
                    m: ortho * mt,
                    tex: 0,
                    texturize,
                    notex_color: Rgba::new(1.0, 1.0, 1.0, 1.0),
                };

                // Draw the texts
                if text_tex_id.is_some() && options.edge_id_position == EdgeIdPosition::Outside {
                    self.gl.active_texture(glow::TEXTURE0);
                    self.gl.bind_texture(glow::TEXTURE_2D, text_tex_id);
                    gl_fixs
                        .prg_text
                        .draw(&u, &self.data.gl_objs().paper_text, glow::TRIANGLES);
                }

                // Line Flaps
                if flap_style != FlapStyle::None {
                    gl_fixs.prg_paper_line.draw(
                        &u,
                        &self.data.gl_objs().paper_vertices_flap_edge,
                        glow::TRIANGLES,
                    );
                }

                // Solid Flaps
                if flap_style != FlapStyle::None && flap_style != FlapStyle::White {
                    gl_fixs.prg_paper_solid.draw(
                        &u,
                        &self.data.gl_objs().paper_vertices_flap,
                        glow::TRIANGLES,
                    );
                }

                // Borders
                gl_fixs.prg_paper_line.draw(
                    &u,
                    &self.data.gl_objs().paper_vertices_edge_cut,
                    glow::TRIANGLES,
                );

                // Textured faces
                self.gl.vertex_attrib_4_f32(
                    gl_fixs
                        .prg_paper_solid
                        .attrib_by_name("color")
                        .unwrap()
                        .location(),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                );
                gl_fixs.prg_paper_solid.draw(
                    &u,
                    (
                        &self.data.gl_objs().vertices,
                        &self.data.gl_objs().paper_vertices,
                    ),
                    glow::TRIANGLES,
                );

                // Shadow Flaps
                u.texturize = 0;
                u.notex_color = Rgba::new(0.0, 0.0, 0.0, 0.0);
                gl_fixs.prg_paper_solid.draw(
                    &u,
                    &self.data.gl_objs().paper_vertices_shadow_flap,
                    glow::TRIANGLES,
                );
                u.notex_color = Rgba::new(1.0, 1.0, 1.0, 1.0);

                // Creases
                gl_fixs.prg_paper_line.draw(
                    &u,
                    &self.data.gl_objs().paper_vertices_edge_crease,
                    glow::TRIANGLES,
                );

                // Draw the texts
                if text_tex_id.is_some() && options.edge_id_position == EdgeIdPosition::Inside {
                    self.gl.active_texture(glow::TEXTURE0);
                    self.gl.bind_texture(glow::TEXTURE_2D, text_tex_id);
                    gl_fixs
                        .prg_text
                        .draw(&u, &self.data.gl_objs().paper_text, glow::TRIANGLES);
                }
                // End render

                if let Some((_, fbo_no_aa)) = &rbo_fbo_no_aa {
                    read_fb_binder.rebind(&fbo);
                    draw_fb_binder.rebind(fbo_no_aa);
                    self.gl.blit_framebuffer(
                        0,
                        0,
                        page_size_pixels.x,
                        page_size_pixels.y,
                        0,
                        0,
                        page_size_pixels.x,
                        page_size_pixels.y,
                        glow::COLOR_BUFFER_BIT,
                        glow::NEAREST,
                    );
                    read_fb_binder.rebind(fbo_no_aa);
                    draw_fb_binder.rebind(&fbo);
                }

                self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                self.gl.read_buffer(glow::COLOR_ATTACHMENT0);

                let mut pixbuf =
                    image::RgbaImage::new(page_size_pixels.x as u32, page_size_pixels.y as u32);

                self.gl.read_pixels(
                    0,
                    0,
                    page_size_pixels.x,
                    page_size_pixels.y,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelPackData::Slice(Some(&mut pixbuf)),
                );

                let edge_id_font_size = options.edge_id_font_size * 25.4 / 72.0; // pt to mm
                let edge_id_position = options.edge_id_position;

                texts.clear();
                if options.show_self_promotion {
                    let x = margin_left;
                    let y = (page_size_mm.y - margin_bottom + FONT_SIZE)
                        .min(page_size_mm.y - FONT_SIZE);
                    let text = signature();
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
                    let y = (page_size_mm.y - margin_bottom + FONT_SIZE)
                        .min(page_size_mm.y - FONT_SIZE);
                    let text = tr!("Page {}/{}", page + 1, page_count);
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
                    for (i_island, lines) in &lines_by_island {
                        // Island ids
                        let mut text =
                            printable_island_name(self.data.papercraft(), *i_island, lines, &extra);
                        let (is_in_page, pos) = in_page(text.pos);
                        if is_in_page {
                            text.pos = pos;
                            texts.push(text);
                        }
                    }
                    // Edge ids
                    if !options.island_name_only {
                        for cut_idx in extra
                            .cut_info()
                            .unwrap()
                            .iter()
                            .flat_map(|ci| ci.descriptions())
                        {
                            let i_island_b =
                                self.data.papercraft().island_by_face(cut_idx.i_face_b);
                            let ii = self
                                .data
                                .papercraft()
                                .island_by_key(i_island_b)
                                .map(|island_b| island_b.name())
                                .unwrap_or("?");
                            let text = format!("{}:{}", ii, cut_idx.id);
                            let (is_in_page, pos) =
                                in_page(cut_idx.pos(self.font_text_line_scale * edge_id_font_size));
                            if is_in_page {
                                texts.push(PrintableText {
                                    size: edge_id_font_size,
                                    pos,
                                    angle: cut_idx.angle,
                                    align: TextAlign::Center,
                                    text,
                                });
                            }
                        }
                    }
                }
                log::debug!("Render {} complete", page + 1);
                do_page_fn(page, pixbuf, &extra, &texts, &lines_by_island)?;
            }
        }
        Ok(())
    }
}

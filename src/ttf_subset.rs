#![allow(dead_code)]

// https://learn.microsoft.com/en-us/typography/opentype/spec/

use std::fmt::Write as _;
use std::{
    char::TryFromCharError,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    io::{Cursor, Read, Seek, Write},
    range::{RangeInclusive, RangeInclusiveIter},
};

use anyhow::{Result, bail};

#[derive(Debug, Copy, Clone)]
pub struct Scaler(f32);

impl Scaler {
    pub fn new(sz: f32, font: &FontFile) -> Scaler {
        let scale = sz / font.units_per_em().0 as f32;
        Scaler(scale)
    }
    pub fn scale(&self, x: FixWord<impl Into<f64>>) -> f32 {
        x.0.into() as f32 * self.0
    }
}

#[derive(Debug, Clone)]
pub enum CMapTrans {
    Single(Char),
    Seq(Vec<Char>),
}

pub fn encode_glyphs(
    text: &str,
    font: &FontFile<'_>,
    mut old_chars: Option<&mut Vec<CMapTrans>>,
) -> Vec<GlyphId> {
    let mut old_glyphs = Vec::new();
    for c in text.chars() {
        let Ok(c) = Char::try_from(c) else { continue };
        let Some(old_glyph) = font.map_char(c) else {
            continue;
        };
        old_glyphs.push(old_glyph);
        if let Some(old_chars) = old_chars.as_mut() {
            old_chars.push(CMapTrans::Single(c));
        }
    }

    // apply ligatures
    if !old_glyphs.is_empty() {
        let mut i = 0;
        while i < old_glyphs.len() - 1 {
            if let Some((liga_sz, liga_id)) = font.ligature(&old_glyphs[i..]) {
                old_glyphs[i] = liga_id;
                old_glyphs.drain(i + 1..i + liga_sz);

                if let Some(old_chars) = old_chars.as_mut() {
                    // concatenate all the old_chars from the ligature to the new value
                    let mut liga_str = Vec::new();
                    for n in i..i + liga_sz {
                        match &old_chars[n] {
                            CMapTrans::Single(c) => liga_str.push(*c),
                            CMapTrans::Seq(s) => liga_str.extend_from_slice(&s),
                        }
                    }
                    old_chars[i] = CMapTrans::Seq(liga_str);
                    old_chars.drain(i + 1..i + liga_sz);
                }
            }
            i += 1;
        }
    }

    old_glyphs
}

pub fn measure_glyphs(glyphs: &[GlyphId], font: &FontFile<'_>) -> DFWord {
    // Don't use FWord for width, it will overflow easily
    let mut width = 0_i32;
    for (i, &glyph) in glyphs.iter().enumerate() {
        if i > 0
            && let Some(kern) = font.kerning(glyphs[i - 1], glyph)
        {
            width -= i32::from(kern.0);
        }
        if let Some(w) = font.glyph_advance_width(glyph) {
            width += i32::from(w.0);
        }
    }

    FixWord(width)
}

pub fn measure_text(text: &str, font: &FontFile<'_>) -> DFWord {
    let glyphs = encode_glyphs(text, font, None);
    measure_glyphs(&glyphs, font)
}

pub fn encode_text_ex(
    text: &str,
    font: &FontFile<'_>,
    glyph_map: &mut GlyphMap,
    char_map: &mut BTreeMap<GlyphId, CMapTrans>,
) -> (Vec<GlyphId>, Option<lopdf::content::Operation>) {
    let mut old_chars = Vec::new();
    let old_glyphs = encode_glyphs(text, font, Some(&mut old_chars));

    let scaler = Scaler::new(1000.0, font);

    // translate to subset CID and apply kernings
    let mut tj = Vec::new();
    let mut glyph_run = Vec::new();
    for (i, (&old_glyph, old_char)) in old_glyphs.iter().zip(old_chars).enumerate() {
        if i > 0
            && let Some(kern) = font.kerning(old_glyphs[i - 1], old_glyph)
        {
            if !glyph_run.is_empty() {
                tj.push(lopdf::Object::String(
                    std::mem::take(&mut glyph_run),
                    lopdf::StringFormat::Hexadecimal,
                ));
            }
            let adj = -scaler.scale(kern);
            tj.push(adj.into())
        };

        let new_glyph = glyph_map.map(old_glyph);
        char_map.entry(new_glyph).or_insert(old_char);
        glyph_run.extend_from_slice(&new_glyph.0.to_be_bytes());
    }
    if !glyph_run.is_empty() {
        tj.push(lopdf::Object::String(
            std::mem::take(&mut glyph_run),
            lopdf::StringFormat::Hexadecimal,
        ));
    }

    let op = match tj.len() {
        0 => None,
        1 => Some(lopdf::content::Operation::new("Tj", tj)),
        _ => Some(lopdf::content::Operation::new("TJ", vec![tj.into()])),
    };
    (old_glyphs, op)
}

pub fn encode_text(
    text: &str,
    font: &FontFile<'_>,
    glyph_map: &mut GlyphMap,
    char_map: &mut BTreeMap<GlyphId, CMapTrans>,
) -> Option<lopdf::content::Operation> {
    let (_, op) = encode_text_ex(text, font, glyph_map, char_map);
    op
}

pub fn encode_measure_text(
    text: &str,
    font: &FontFile<'_>,
    glyph_map: &mut GlyphMap,
    char_map: &mut BTreeMap<GlyphId, CMapTrans>,
) -> (DFWord, Option<lopdf::content::Operation>) {
    let (glyphs, op) = encode_text_ex(text, font, glyph_map, char_map);
    let width = measure_glyphs(&glyphs, font);
    (width, op)
}

/////////////////////

pub struct LopdfSubsetBuilder<'a> {
    ttf: &'a FontFile<'a>,
    base_font_name: &'a str,
    id_font_file: lopdf::ObjectId,
    id_unicode_cmap: lopdf::ObjectId,
    id_font_descriptor: lopdf::ObjectId,
    id_font: lopdf::ObjectId,
    glyph_map: GlyphMap,
    char_map: BTreeMap<GlyphId, CMapTrans>,
}

impl<'a> LopdfSubsetBuilder<'a> {
    pub fn new(ttf: &'a FontFile<'a>, doc: &mut lopdf::Document, base_font_name: &'a str) -> Self {
        Self {
            ttf,
            base_font_name,
            id_font_file: doc.new_object_id(),
            id_unicode_cmap: doc.new_object_id(),
            id_font_descriptor: doc.new_object_id(),
            id_font: doc.new_object_id(),
            glyph_map: GlyphMap::default(),
            char_map: BTreeMap::default(),
        }
    }

    pub fn encode_measure_text(
        &mut self,
        text: &str,
    ) -> (DFWord, Option<lopdf::content::Operation>) {
        encode_measure_text(&text, &self.ttf, &mut self.glyph_map, &mut self.char_map)
    }

    pub fn id_font(&self) -> lopdf::ObjectId {
        self.id_font
    }

    pub fn complete_document(self, doc: &mut lopdf::Document) {
        use lopdf::{Object, dictionary};

        let scaler = Scaler::new(1000.0, &self.ttf);

        let subset_widths: Vec<Object> = self
            .glyph_map
            .iter_old_by_new()
            .map(|old| {
                let w = self.ttf.glyph_advance_width(old).unwrap_or_default();
                let w = scaler.scale(w);
                w.into()
            })
            .collect();

        let sub_font = self
            .ttf
            .subset(Coverage::GlyphsMap(self.glyph_map))
            .unwrap();

        let mut ff = lopdf::Stream::new(dictionary! {}, sub_font);
        ff.compress().unwrap();
        doc.set_object(self.id_font_file, ff);

        let ascent = scaler.scale(self.ttf.ascender());
        let descent = scaler.scale(self.ttf.descender());
        let cap_height = scaler.scale(self.ttf.cap_height());
        let italic_angle = self.ttf.italic_angle();
        // latin | italic?
        let flags = (1 << 5) | if italic_angle != 0 { 1 << 6 } else { 0 };
        let stem_v = self.ttf.stem_v();
        doc.set_object(
            self.id_font_descriptor,
            dictionary! {
                "Type" => "FontDescriptor",
                "FontName" => self.base_font_name,
                "Ascent" => ascent,
                "Descent" => descent,
                "CapHeight" => cap_height,
                "ItalicAngle" => italic_angle,
                "Flags" => flags,
                "StemV" => stem_v,
                "FontBBox" => self.ttf.bounding_box().map(|x| scaler.scale(x).into()).to_vec(),
                "FontFile2" => self.id_font_file,
            },
        );

        doc.set_object(
            self.id_font,
            dictionary! {
                "Type" => "Font",
                "Subtype" => "Type0",
                "BaseFont" => self.base_font_name,
                "Encoding" => "Identity-H",
                "ToUnicode" => self.id_unicode_cmap,
                "DescendantFonts" => vec![Object::Dictionary(dictionary! {
                    "Type" => "Font",
                    "Subtype" => "CIDFontType2",
                    "BaseFont" => self.base_font_name,
                    "CIDSystemInfo" => dictionary! {
                        "Registry" => Object::string_literal("Adobe"),
                        "Ordering" => Object::string_literal("Identity"),
                        "Supplement" => 0,
                    },
                    "W" => vec![0.into(), Object::Array(subset_widths)],
                    "DW" => 1000,
                    "FontDescriptor" => self.id_font_descriptor,
                    "CIDToGIDMap" => "Identity",
                })],
            },
        );

        let mut cmap = String::new();
        cmap.push_str(
            "\
            /CIDInit /ProcSet findresource begin\n\
            12 dict begin\n\
            begincmap\n\
            /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
            /CMapName /Adobe-Identity-UCS def\n\
            /CMapType 2 def\n\
            1 begincodespacerange\n\
            <0000> <FFFF>\n\
            endcodespacerange\n\
            ",
        );

        // The bfchar table is limited to 100 entries
        let mut remaining = self.char_map.len();
        let mut iter = self.char_map.iter();
        while remaining > 0 {
            let block = remaining.min(100);
            remaining -= block;
            writeln!(&mut cmap, "{} beginbfchar", block).unwrap();
            for _ in 0..block {
                let (i, c) = iter.next().unwrap();
                write!(&mut cmap, "<{:04X}> <", i.0).unwrap();
                match c {
                    CMapTrans::Single(c) => {
                        write!(&mut cmap, "{:04X}", c.0).unwrap();
                    }
                    CMapTrans::Seq(cs) => {
                        for c in cs {
                            write!(&mut cmap, "{:04X}", c.0).unwrap();
                        }
                    }
                }
                writeln!(&mut cmap, ">").unwrap();
            }
            cmap.push_str("endbfchar\n");
        }

        cmap.push_str(
            "\
            endcmap\n\
            CMapName currentdict /CMap defineresource pop\n\
            end\n\
            end\n\
            ",
        );

        let cmap = lopdf::Stream::new(dictionary! {}, cmap.into_bytes());
        doc.set_object(self.id_unicode_cmap, cmap);
    }
}

/////////////////////

/// A table inside a font file.
struct TableHeader<'a> {
    data: &'a [u8],
}

impl<'a> std::fmt::Debug for TableHeader<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} bytes]", self.data.len())
    }
}

/// A Tag is a u32 number that is read as a 4-letter word, big-endian
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Tag(pub u32);

impl std::fmt::Debug for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bs = self.0.to_be_bytes();
        let s = String::from_utf8_lossy(&bs);
        write!(f, "{s}")
    }
}

impl Tag {
    pub const fn from_bytes(bs: [u8; 4]) -> Tag {
        Tag(u32::from_be_bytes(bs))
    }
}

const HEAD: Tag = Tag::from_bytes(*b"head");
const CMAP: Tag = Tag::from_bytes(*b"cmap");
const HHEA: Tag = Tag::from_bytes(*b"hhea");
const MAXP: Tag = Tag::from_bytes(*b"maxp");
const HMTX: Tag = Tag::from_bytes(*b"hmtx");
const LOCA: Tag = Tag::from_bytes(*b"loca");
const GLYF: Tag = Tag::from_bytes(*b"glyf");
const CVT: Tag = Tag::from_bytes(*b"cvt ");
const PREP: Tag = Tag::from_bytes(*b"prep");
const FPGM: Tag = Tag::from_bytes(*b"fpgm");
const OS_2: Tag = Tag::from_bytes(*b"OS/2");

const GPOS: Tag = Tag::from_bytes(*b"GPOS");
const GSUB: Tag = Tag::from_bytes(*b"GSUB");
const KERN: Tag = Tag::from_bytes(*b"kern");
const LIGA: Tag = Tag::from_bytes(*b"liga");

const TTF_SIGNATURE: u32 = 0x0001_0000;

const GLOBAL_CHECKSUM: u32 = 0xB1B0_AFBA;

bitflags::bitflags! {
    #[derive(Debug, Copy, Clone)]
    struct CompositeGlyphFlags: u16 {
        const ARG_1_AND_2_ARE_WORDS = 0x0001;
        const ARGS_ARE_XY_VALUES = 0x0002;
        const ROUND_XY_TO_GRID = 0x0004;
        const WE_HAVE_A_SCALE = 0x0008;
        const MORE_COMPONENTS = 0x0020;
        const WE_HAVE_AN_X_AND_Y_SCALE = 0x0040;
        const WE_HAVE_A_TWO_BY_TWO = 0x0080;
        const WE_HAVE_INSTRUCTIONS = 0x0100;
        const USE_MY_METRICS = 0x0200;
        const OVERLAP_COMPOUND = 0x0400;
        const SCALED_COMPONENT_OFFSET = 0x0800;
        const UNSCALED_COMPONENT_OFFSET = 0x1000;
    }
}

/// A Rust `char` is a u32, but we want a u16.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Char(pub u16);

impl std::fmt::Debug for Char {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "U+{:04x}", self.0)
    }
}

impl std::fmt::Display for Char {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", char::try_from(u32::from(self.0)).unwrap_or('?'))
    }
}

impl TryFrom<char> for Char {
    type Error = TryFromCharError;

    fn try_from(value: char) -> std::result::Result<Self, Self::Error> {
        let x = u16::try_from(value)?;
        Ok(Char(x))
    }
}

impl From<u16> for Char {
    fn from(value: u16) -> Char {
        Char(value)
    }
}

impl From<u8> for Char {
    fn from(value: u8) -> Char {
        let x = u16::from(value);
        Char(x)
    }
}

/// A glyph id in a TTF font is a u16
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct GlyphId(pub u16);

impl std::fmt::Debug for GlyphId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GlyphId({})", self.0)
    }
}

/// A length, in font design units (see units_per_em).
#[derive(Debug, Default, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct FixWord<T>(pub T);

pub type FWord = FixWord<i16>;
pub type UFWord = FixWord<u16>;
pub type DFWord = FixWord<i32>;

/*
#[derive(Debug, Default, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct FWord(pub i16);

/// An unsinged length, in font design units (see units_per_em).
#[derive(Debug, Default, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct UFWord(pub u16);

/// An 32-bit length, in font design units (see units_per_em).
#[derive(Debug, Default, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct DFWord(pub i32);
 */

// A few interesting tables

#[derive(Debug)]
struct Head {
    units_per_em: UFWord,
    bounding_box: [FWord; 4],
    index_to_loc_format: IndexToLocFormat,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum IndexToLocFormat {
    Short16,
    Long32,
}

impl Head {
    // Font header table
    fn parse(data: &[u8]) -> Result<Head> {
        let mut r = Cursor::new(data);
        let r = &mut r;

        let _major = read_u16be(r)?;
        let _minor = read_u16be(r)?;
        let _rev = read_u32be(r)?;

        let _checksum_adjust = read_u32be(r)?;
        let _magic = read_u32be(r)?;

        let _flags = read_u16be(r)?;
        let units_per_em = read_ufword(r)?;

        let _c1 = read_u32be(r)?;
        let _c2 = read_u32be(r)?;
        let _m1 = read_u32be(r)?;
        let _m2 = read_u32be(r)?;

        let xmin = read_fword(r)?;
        let ymin = read_fword(r)?;
        let xmax = read_fword(r)?;
        let ymax = read_fword(r)?;

        let _mac_style = read_u16be(r)?;
        let _smallest = read_u16be(r)?;
        let _font_direction = read_u16be(r)?;
        let index_to_loc_format = match read_u16be(r)? {
            0 => IndexToLocFormat::Short16,
            1 => IndexToLocFormat::Long32,
            _ => bail!("indexToLocFormat"),
        };
        let _glyph_data_format = read_u16be(r)?;

        Ok(Head {
            units_per_em,
            bounding_box: [xmin, ymin, xmax, ymax],
            index_to_loc_format,
        })
    }
}

#[derive(Debug)]
struct MaximumProfile {
    num_glyphs: u16,
}

impl MaximumProfile {
    fn parse(data: &[u8]) -> Result<MaximumProfile> {
        let mut r = Cursor::new(data);
        let r = &mut r;

        let _major = read_u16be(r)?;
        let _minor = read_u16be(r)?;
        let num_glyphs = read_u16be(r)?;

        Ok(MaximumProfile { num_glyphs })
    }
}

#[derive(Debug)]
pub struct HorizontalHeaderTable {
    pub ascender: FWord,
    pub descender: FWord,
    pub line_gap: FWord,
    pub caret_slope_rise: i16,
    pub caret_slope_run: i16,
    pub caret_slope_offset: i16,

    num_hmetrics: u16,
}

impl HorizontalHeaderTable {
    fn parse(data: &[u8]) -> Result<HorizontalHeaderTable> {
        let mut r = Cursor::new(data);
        let r = &mut r;

        let _major = read_u16be(r)?;
        let _minor = read_u16be(r)?;

        let ascender = read_fword(r)?;
        let descender = read_fword(r)?;
        let line_gap = read_fword(r)?;

        let _advance_width_max = read_ufword(r)?;
        let _min_lsb = read_fword(r)?;
        let _min_rsb = read_fword(r)?;
        let _max_extent = read_fword(r)?;
        let caret_slope_rise = read_i16be(r)?;
        let caret_slope_run = read_i16be(r)?;
        let caret_slope_offset = read_i16be(r)?;

        // reserved
        r.seek_relative(2 * 4)?;

        let _format = read_u16be(r)?;
        let num_hmetrics = read_u16be(r)?;

        Ok(HorizontalHeaderTable {
            ascender,
            descender,
            line_gap,
            caret_slope_rise,
            caret_slope_run,
            caret_slope_offset,
            num_hmetrics,
        })
    }
}

#[derive(Debug)]
struct HorizontalMetricsTable {
    metrics: Vec<Metric>,
}

impl HorizontalMetricsTable {
    fn parse(num_hmetrics: u16, num_glyphs: u16, data: &[u8]) -> Result<HorizontalMetricsTable> {
        let mut r = Cursor::new(data);
        let r = &mut r;

        let mut metrics = Vec::new();
        for _ in 0..num_hmetrics {
            let advance_width = read_ufword(r)?;
            let lsb = read_fword(r)?;
            metrics.push(Metric { advance_width, lsb });
        }
        let advance_width = metrics.last().map(|m| m.advance_width).unwrap_or_default();
        for _ in num_hmetrics..num_glyphs {
            let lsb = read_fword(r)?;
            metrics.push(Metric { advance_width, lsb });
        }

        Ok(HorizontalMetricsTable { metrics })
    }

    fn get(&self, id: GlyphId) -> Option<&Metric> {
        self.metrics.get(usize::from(id.0))
    }
}

#[derive(Debug)]
struct Metric {
    advance_width: UFWord,
    lsb: FWord,
}

/// Character to Glyph Index Mapping Table
#[derive(Debug)]
struct CharMap {
    segments: Vec<CharMapSegment>,
    ids: Vec<u16>,
}

#[derive(Debug, Copy, Clone)]
struct CharMapSegment {
    range: RangeInclusive<u16>, // should be <Char>
    delta: i16,
    offset: Option<u16>,
}

impl Default for CharMapSegment {
    fn default() -> Self {
        Self {
            range: RangeInclusive { start: 1, last: 0 },
            delta: 0,
            offset: None,
        }
    }
}

impl CharMap {
    fn parse(data: &[u8]) -> Result<CharMap> {
        let mut r = Cursor::new(data);
        let r = &mut r;

        let _version = read_u16be(r)?;
        let ntables = read_u16be(r)?;

        let mut unicode_offset = None;
        let mut microsoft_offset = None;
        for _ in 0..ntables {
            let platform = read_u16be(r)?;
            let encoding = read_u16be(r)?;
            let offset = read_u32be(r)?;

            match (platform, encoding) {
                (0, 3) => unicode_offset = Some(offset),
                (3, 1) => microsoft_offset = Some(offset),
                _ => (),
            }
        }

        let offset = unicode_offset
            .or(microsoft_offset)
            .expect("unicode cmap not found");
        let data = &data[offset as usize..];
        let mut r = Cursor::new(data);
        let r = &mut r;
        let format = read_u16be(r)?;
        if format != 4 {
            panic!("only cmap format 4!");
        }
        let len = read_u16be(r)?;
        *r.get_mut() = &r.get_ref()[..len as usize];

        let _lang = read_u16be(r)?;
        let seg_count2 = read_u16be(r)?;
        let _range = read_u16be(r)?;
        let _selector = read_u16be(r)?;
        let _shift = read_u16be(r)?;

        let seg_count = seg_count2 / 2;

        let mut segments = vec![CharMapSegment::default(); seg_count as usize];
        for s in &mut segments {
            s.range.last = read_u16be(r)?;
        }
        let _pad = read_u16be(r)?;
        for s in &mut segments {
            s.range.start = read_u16be(r)?;
        }
        for s in &mut segments {
            s.delta = read_i16be(r)?;
        }
        for (i, s) in segments.iter_mut().enumerate() {
            let offset = read_u16be(r)?;
            s.offset = match offset {
                0 => None,
                x => Some(x / 2 - (seg_count - i as u16)),
            };
        }

        let mut ids = Vec::new();
        loop {
            let Ok(id) = read_u16be(r) else { break };
            ids.push(id);
        }

        Ok(CharMap { segments, ids })
    }

    fn iter(&self) -> CharMapIter<'_> {
        let range = self
            .segments
            .first()
            .map(|seg| seg.range)
            .unwrap_or_else(|| RangeInclusive { start: 1, last: 0 })
            .into_iter();
        CharMapIter {
            cmap: self,
            segment: 0,
            range,
        }
    }
    fn map_char_in_seg(&self, c: Char, seg: &CharMapSegment) -> Option<GlyphId> {
        let id = match seg.offset {
            None => c.0.wrapping_add_signed(seg.delta),
            Some(offs) => {
                let idx = c.0.wrapping_sub(seg.range.start).wrapping_add(offs);
                let id = self.ids.get(usize::from(idx)).copied().unwrap_or(0);
                if id == 0 {
                    return None;
                }
                id.wrapping_add_signed(seg.delta)
            }
        };
        Some(GlyphId(id))
    }
}

struct CharMapIter<'a> {
    cmap: &'a CharMap,
    segment: usize,
    range: RangeInclusiveIter<u16>, // <Char>
}

impl Iterator for CharMapIter<'_> {
    type Item = (Char, GlyphId);

    fn next(&mut self) -> Option<(Char, GlyphId)> {
        loop {
            let mut seg = self.cmap.segments.get(self.segment)?;
            while let Some(c) = self.range.next() {
                let c = Char(c);
                if let Some(id) = self.cmap.map_char_in_seg(c, seg) {
                    return Some((c, id));
                }
            }
            self.segment += 1;
            seg = self.cmap.segments.get(self.segment)?;
            self.range = seg.range.into_iter();
        }
    }
}

#[derive(Debug)]
struct IndexToLocation {
    offsets: Vec<u32>,
}

impl IndexToLocation {
    fn parse(format: IndexToLocFormat, num_glyphs: u16, data: &[u8]) -> Result<IndexToLocation> {
        let mut r = Cursor::new(data);
        let r = &mut r;

        let read_loca = match format {
            IndexToLocFormat::Short16 => {
                |r: &mut Cursor<&[u8]>| read_u16be(r).map(|s| 2 * u32::from(s))
            }
            IndexToLocFormat::Long32 => |r: &mut Cursor<&[u8]>| read_u32be(r),
        };

        let mut offsets = Vec::new();
        for _ in 0..=num_glyphs {
            let x = read_loca(r)?;
            offsets.push(x);
        }

        Ok(IndexToLocation { offsets })
    }

    fn find<'a>(&self, id: GlyphId, glyf_data: &'a [u8]) -> &'a [u8] {
        let id = usize::from(id.0);
        let Some(id1) = id.checked_add(1) else {
            return &[];
        };
        if id1 >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[id];
        let end = self.offsets[id1];
        &glyf_data[start as usize..end as usize]
    }
}

// Glyph positioning (just the kerning table)
#[derive(Debug, Default)]
struct KerningTable {
    kernings: HashMap<(GlyphId, GlyphId), FWord>,
}

impl KerningTable {
    fn parse_kern(&mut self, data: &[u8]) -> Result<()> {
        let mut r = Cursor::new(data);
        let r = &mut r;

        let _version = read_u16be(r)?;
        let num_subtables = read_u16be(r)?;

        for _ in 0..num_subtables {
            let _subtable_version = read_u16be(r)?;
            let subtable_len = read_u16be(r)?;
            let coverage_flags = read_u16be(r)?;

            // 0x01: horizontal,
            // 0x02: minimum instead of kerning
            // 0x04: perpendicular
            // 0x08: override
            if (coverage_flags & 0x0f) != 1 {
                continue;
            }
            let format = coverage_flags >> 4;
            // format 2 not supported, I've never seen it here.
            if format != 0 {
                continue;
            }

            // subtable_len includes the header (6 bytes) but we already consumed those
            let Some(subtable_len) = subtable_len.checked_sub(6) else {
                continue;
            };

            let subtable_data = &data[r.position() as usize..][..usize::from(subtable_len)];
            let mut subr = Cursor::new(subtable_data);
            let subr = &mut subr;

            let npairs = read_u16be(subr)?;
            let _ra = read_u16be(subr)?;
            let _se = read_u16be(subr)?;
            let _sh = read_u16be(subr)?;
            for _ in 0..npairs {
                let left = GlyphId(read_u16be(subr)?);
                let right = GlyphId(read_u16be(subr)?);
                let value = read_fword(subr)?;
                self.kernings.insert((left, right), value);
            }

            r.seek_relative(i64::from(subtable_len)).unwrap();
        }

        Ok(())
    }

    fn parse_gpos(&mut self, data: &[u8]) -> Result<()> {
        let mut r = Cursor::new(data);
        let r = &mut r;

        let _major = read_u16be(r)?;
        let _minor = read_u16be(r)?;
        let _script_offs = read_u16be(r)?;
        let feature_offs = read_u16be(r)?;
        let lookup_offs = read_u16be(r)?;

        let mut kern_indices = BTreeSet::new();

        // features
        let feature_list_data = &r.get_ref()[feature_offs as usize..];
        let mut rfeat = Cursor::new(feature_list_data);
        let rfeat = &mut rfeat;

        let fl_count = read_u16be(rfeat)?;
        for _ in 0..fl_count {
            let fl_tag = Tag(read_u32be(rfeat)?);
            let fl_offs = read_u16be(rfeat)?;

            if fl_tag == KERN {
                let kern_data = &rfeat.get_ref()[fl_offs as usize..];
                let mut rkd = Cursor::new(kern_data);
                let rkd = &mut rkd;

                let _kparm_offs = read_u16be(rkd)?;
                let k_index_count = read_u16be(rkd)?;

                for _ in 0..k_index_count {
                    let index = read_u16be(rkd)?;
                    kern_indices.insert(index);
                }
            }
        }

        if !kern_indices.is_empty() {
            // lookups
            let lookup_list_data = &r.get_ref()[lookup_offs as usize..];
            let mut rlut = Cursor::new(lookup_list_data);
            let rlut = &mut rlut;

            let lu_count = read_u16be(rlut)?;
            for i in 0..lu_count {
                let lu_offs = read_u16be(rlut)?;
                if !kern_indices.contains(&i) {
                    continue;
                }

                let lu_data = &rlut.get_ref()[lu_offs as usize..];
                let mut rlu = Cursor::new(lu_data);
                let rlu = &mut rlu;

                let lu_type = read_u16be(rlu)?;
                // Lookup type 2 subtable: pair adjustment positioning
                if lu_type != 2 {
                    continue;
                }
                let _lu_flags = read_u16be(rlu)?;
                let lu_sub_count = read_u16be(rlu)?;
                for _ in 0..lu_sub_count {
                    let lu_sub_offs = read_u16be(rlu)?;

                    let lu_sub_data = &rlu.get_ref()[lu_sub_offs as usize..];
                    let mut rsub = Cursor::new(lu_sub_data);
                    let rsub = &mut rsub;
                    let lu_sub_fmt = read_u16be(rsub)?;

                    match lu_sub_fmt {
                        // format 1, this is the easy one
                        1 => {
                            let coverage_offs = read_u16be(rsub)?;
                            let vf_1 = read_u16be(rsub)?;
                            let vf_2 = read_u16be(rsub)?;
                            // 4: X_ADVANCE
                            // 0: not included
                            if vf_1 != 4 || vf_2 != 0 {
                                continue;
                            }

                            // pair_set_count is the count of first_ids
                            // each first_id has a coverage_index that maps to the list of second_ids and the X_ADVANCE
                            let pair_set_count = read_u16be(rsub)?;

                            // Read the pair sets
                            let mut pair_sets = Vec::new();
                            for _ in 0..pair_set_count {
                                let pair_set_offset = read_u16be(rsub)?;

                                let pair_set_data = &rsub.get_ref()[pair_set_offset as usize..];
                                let mut rpair = Cursor::new(pair_set_data);
                                let rpair = &mut rpair;

                                let pair_value_count = read_u16be(rpair)?;

                                let mut pair_set = Vec::new();
                                for _ in 0..pair_value_count {
                                    let second_id = GlyphId(read_u16be(rpair)?);
                                    let data1 = read_fword(rpair)?;
                                    pair_set.push((second_id, data1));
                                    // data2 is not included
                                }
                                pair_sets.push(pair_set);
                            }

                            // index is the first_id, value is the pair_set
                            let mut coverages = Vec::new();

                            // Read the coverages
                            let coverage_data = &rsub.get_ref()[coverage_offs as usize..];
                            let mut rcov = Cursor::new(coverage_data);
                            let rcov = &mut rcov;

                            let cov_fmt = read_u16be(rcov)?;

                            match cov_fmt {
                                1 => {
                                    let count = read_u16be(rcov)?;
                                    for c in 0..count {
                                        let id = GlyphId(read_u16be(rcov)?);
                                        coverages.push((id, c));
                                    }
                                }
                                2 => {
                                    let count = read_u16be(rcov)?;
                                    for _ in 0..count {
                                        let start_id = read_u16be(rcov)?;
                                        let end_id = read_u16be(rcov)?;
                                        let index = read_u16be(rcov)?;
                                        for c in start_id..=end_id {
                                            coverages.push((GlyphId(c), index + (c - start_id)));
                                        }
                                    }
                                }
                                _ => continue,
                            }

                            for (first_id, set) in coverages {
                                if let Some(set) = pair_sets.get(usize::from(set)) {
                                    for &(second_id, value) in set {
                                        self.kernings.insert((first_id, second_id), value);
                                    }
                                }
                            }
                        }
                        // format 2, kernings between glyph classes
                        2 => {
                            let coverage_offs = read_u16be(rsub)?;
                            let vf_1 = read_u16be(rsub)?;
                            let vf_2 = read_u16be(rsub)?;
                            // 4: X_ADVANCE
                            // 0: not included
                            if vf_1 != 4 || vf_2 != 0 {
                                continue;
                            }

                            let class_offset_1 = read_u16be(rsub)?;
                            let class_offset_2 = read_u16be(rsub)?;
                            let class_count_1 = read_u16be(rsub)?;
                            let class_count_2 = read_u16be(rsub)?;

                            let mut classes_1 = vec![vec![]; class_count_1 as usize];
                            let mut classes_2 = vec![vec![]; class_count_2 as usize];
                            for (class_offset, classes) in [
                                (class_offset_1, &mut classes_1),
                                (class_offset_2, &mut classes_2),
                            ] {
                                let class_data = &rsub.get_ref()[class_offset as usize..];
                                let mut rc = Cursor::new(class_data);
                                let rc = &mut rc;

                                let format = read_u16be(rc)?;

                                // Format of the glyph class
                                match format {
                                    1 => {
                                        let start = read_u16be(rc)?;
                                        let count = read_u16be(rc)?;
                                        for i in 0..count {
                                            let class_id = read_u16be(rc)?;
                                            //class_id <- start + i
                                            if let Some(class) = classes.get_mut(class_id as usize)
                                            {
                                                class.push(GlyphId(start + i));
                                            }
                                        }
                                    }
                                    2 => {
                                        let count = read_u16be(rc)?;
                                        for _ in 0..count {
                                            let start = read_u16be(rc)?;
                                            let end = read_u16be(rc)?;
                                            let class_id = read_u16be(rc)?;
                                            if let Some(class) = classes.get_mut(class_id as usize)
                                            {
                                                for g in start..=end {
                                                    class.push(GlyphId(g));
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            let mut coverages = HashSet::new();

                            // Read the coverages
                            let coverage_data = &rsub.get_ref()[coverage_offs as usize..];
                            let mut rcov = Cursor::new(coverage_data);
                            let rcov = &mut rcov;

                            let cov_fmt = read_u16be(rcov)?;

                            match cov_fmt {
                                1 => {
                                    let count = read_u16be(rcov)?;
                                    for _ in 0..count {
                                        let id = GlyphId(read_u16be(rcov)?);
                                        coverages.insert(id);
                                    }
                                }
                                2 => {
                                    let count = read_u16be(rcov)?;
                                    for _ in 0..count {
                                        let start_id = read_u16be(rcov)?;
                                        let end_id = read_u16be(rcov)?;
                                        let _index = read_u16be(rcov)?;
                                        for c in start_id..=end_id {
                                            coverages.insert(GlyphId(c));
                                        }
                                    }
                                }
                                _ => continue,
                            }

                            // Any glyph not in coverage should be ignored from classes_1.
                            // Any glyph in coverage but not in any classes_1 is actually class_0.
                            // A consequence is that the data read in classes_1[0] is meaningless.

                            for class in &mut classes_1[1..] {
                                class.retain(|g| coverages.remove(g));
                            }
                            classes_1[0] = coverages.into_iter().collect();

                            for c1 in &classes_1 {
                                for c2 in &classes_2 {
                                    let data1 = read_fword(rsub)?;
                                    // data2 is not included

                                    if data1.0 == 0 {
                                        continue;
                                    }

                                    //data1 is the X_ADJUST between all the glyphs in c1 and c2
                                    for &a in c1 {
                                        for &b in c2 {
                                            self.kernings.insert((a, b), data1);
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    fn get(&self, a: GlyphId, b: GlyphId) -> Option<FWord> {
        self.kernings.get(&(a, b)).copied()
    }
}

#[derive(Debug)]
struct LigatureTable {
    max_ligature_len: usize,
    ligatures: Vec<(Vec<GlyphId>, GlyphId)>,
}

impl LigatureTable {
    fn parse(data: &[u8]) -> Result<LigatureTable> {
        // TODO: remove duplicated ligatures
        let mut r = Cursor::new(data);
        let r = &mut r;

        let _major = read_u16be(r)?;
        let _minor = read_u16be(r)?;
        let _script_offs = read_u16be(r)?;
        let feature_offs = read_u16be(r)?;
        let lookup_offs = read_u16be(r)?;

        let mut liga_indices = BTreeSet::new();

        // features
        let feature_list_data = &r.get_ref()[feature_offs as usize..];
        let mut rfeat = Cursor::new(feature_list_data);
        let rfeat = &mut rfeat;

        let fl_count = read_u16be(rfeat)?;
        for _ in 0..fl_count {
            let fl_tag = Tag(read_u32be(rfeat)?);
            let fl_offs = read_u16be(rfeat)?;

            if fl_tag == LIGA {
                let liga_data = &rfeat.get_ref()[fl_offs as usize..];
                let mut rld = Cursor::new(liga_data);
                let rld = &mut rld;

                let _lparm_offs = read_u16be(rld)?;
                let l_index_count = read_u16be(rld)?;

                for _ in 0..l_index_count {
                    let index = read_u16be(rld)?;
                    liga_indices.insert(index);
                }
            }
        }

        let mut ligatures = Vec::new();
        if !liga_indices.is_empty() {
            // lookups
            let lookup_list_data = &r.get_ref()[lookup_offs as usize..];
            let mut rlut = Cursor::new(lookup_list_data);
            let rlut = &mut rlut;

            let lu_count = read_u16be(rlut)?;
            for i in 0..lu_count {
                let lu_offs = read_u16be(rlut)?;
                if !liga_indices.contains(&i) {
                    continue;
                }

                let lu_data = &rlut.get_ref()[lu_offs as usize..];
                let mut rlu = Cursor::new(lu_data);
                let rlu = &mut rlu;

                let lu_type = read_u16be(rlu)?;
                // Lookup type 4 subtable: ligature subst
                if lu_type != 4 {
                    continue;
                }
                let _lu_flags = read_u16be(rlu)?;
                let lu_sub_count = read_u16be(rlu)?;
                for _ in 0..lu_sub_count {
                    let lu_sub_offs = read_u16be(rlu)?;

                    let lu_sub_data = &rlu.get_ref()[lu_sub_offs as usize..];
                    let mut rsub = Cursor::new(lu_sub_data);
                    let rsub = &mut rsub;
                    let lu_sub_fmt = read_u16be(rsub)?;

                    // format 1
                    if lu_sub_fmt != 1 {
                        continue;
                    }

                    let coverage_offs = read_u16be(rsub)?;

                    // liga_set_count is the count of first_ids
                    // each first_id has a coverage_index that maps to the liga_set
                    let liga_set_count = read_u16be(rsub)?;

                    // Read the liga sets
                    let mut liga_sets = Vec::new();
                    for _ in 0..liga_set_count {
                        let liga_set_offset = read_u16be(rsub)?;

                        let liga_set_data = &rsub.get_ref()[liga_set_offset as usize..];
                        let mut rls = Cursor::new(liga_set_data);
                        let rls = &mut rls;

                        let liga_count = read_u16be(rls)?;
                        //liga: (ligatureGlyph, components[1..])
                        let mut liga = Vec::new();
                        for _ in 0..liga_count {
                            let liga_offset = read_u16be(rls)?;

                            let liga_data = &rls.get_ref()[liga_offset as usize..];
                            let mut rl = Cursor::new(liga_data);
                            let rl = &mut rl;

                            let liga_glyph = GlyphId(read_u16be(rl)?);
                            let component_count = read_u16be(rl)?;
                            // should not happen
                            if component_count < 2 {
                                continue;
                            }
                            let mut comps = Vec::new();
                            for _ in 0..component_count - 1 {
                                let comp = GlyphId(read_u16be(rl)?);
                                comps.push(comp);
                            }
                            liga.push((liga_glyph, comps));
                        }
                        liga_sets.push(liga);
                    }

                    // index is the first_id, value is the liga_set
                    let mut coverages = Vec::new();

                    // Read the coverages
                    let coverage_data = &rsub.get_ref()[coverage_offs as usize..];
                    let mut rcov = Cursor::new(coverage_data);
                    let rcov = &mut rcov;

                    let cov_fmt = read_u16be(rcov)?;

                    match cov_fmt {
                        1 => {
                            let count = read_u16be(rcov)?;
                            for c in 0..count {
                                let id = GlyphId(read_u16be(rcov)?);
                                coverages.push((id, c));
                            }
                        }
                        2 => {
                            let count = read_u16be(rcov)?;
                            for _ in 0..count {
                                let start_id = read_u16be(rcov)?;
                                let end_id = read_u16be(rcov)?;
                                let index = read_u16be(rcov)?;
                                for c in start_id..=end_id {
                                    coverages.push((GlyphId(c), index + (c - start_id)));
                                }
                            }
                        }
                        _ => continue,
                    }

                    for (first_id, set) in coverages {
                        if let Some(liga_set) = liga_sets.get(usize::from(set)) {
                            for &(target_id, ref tail_ids) in liga_set {
                                let mut ids = Vec::with_capacity(1 + tail_ids.len());
                                ids.push(first_id);
                                ids.extend_from_slice(tail_ids);
                                ligatures.push((ids, target_id));
                            }
                        }
                    }
                }
            }
        }

        let max_ligature_len = ligatures.iter().map(|l| l.0.len()).max().unwrap_or(0);
        Ok(LigatureTable {
            max_ligature_len,
            ligatures,
        })
    }

    fn get(&self, ids: &[GlyphId]) -> Option<(usize, GlyphId)> {
        let first = ids.first()?;
        for (seq, id) in &self.ligatures {
            // fast check with the fist char
            if seq.first().unwrap() != first {
                continue;
            }

            if &ids.get(..seq.len()) == &Some(&seq[..]) {
                return Some((seq.len(), *id));
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct Os2Metrics {
    pub version: u16,
    pub avg_char_width: FWord,
    pub weight_class: u16,
    pub width_class: u16,
    pub flags: u16,
    pub subscript_x_size: FWord,
    pub subscript_y_size: FWord,
    pub subscript_x_offset: FWord,
    pub subscript_y_offset: FWord,
    pub superscript_x_size: FWord,
    pub superscript_y_size: FWord,
    pub superscript_x_offset: FWord,
    pub superscript_y_offset: FWord,
    pub strikeout_size: FWord,
    pub strikeout_position: FWord,
    pub family_class: i16,
    pub panose: [u8; 10],
    pub unicode_range: [u8; 16],
    pub ach_vend_id: Tag,
    pub selection: u16,
    pub first_char_index: Char,
    pub last_char_index: Char,
    pub typo_ascender: FWord,
    pub typo_descender: FWord,
    pub typo_line_gap: FWord,
    pub win_ascent: UFWord,
    pub win_descent: UFWord,

    pub v1: Option<Os2MetricsV1>,
    pub v2: Option<Os2MetricsV2>,
    // v3 and v4 don't have extra fields
    pub v5: Option<Os2MetricsV5>,
}

#[derive(Debug)]
pub struct Os2MetricsV1 {
    pub code_page_range: [u8; 8],
}

#[derive(Debug)]
pub struct Os2MetricsV2 {
    pub height: FWord,
    pub cap_height: FWord,
    pub default_char: Char,
    pub break_char: Char,
    pub max_context: u16,
}

#[derive(Debug)]
pub struct Os2MetricsV5 {
    pub lower_optical_point_size: u16,
    pub upper_optical_point_size: u16,
}

impl Os2Metrics {
    fn parse(data: &[u8]) -> Result<Os2Metrics> {
        let mut r = Cursor::new(data);
        let r = &mut r;

        let version = read_u16be(r)?;
        let avg_char_width = read_fword(r)?;
        let weight_class = read_u16be(r)?;
        let width_class = read_u16be(r)?;
        let flags = read_u16be(r)?;
        let subscript_x_size = read_fword(r)?;
        let subscript_y_size = read_fword(r)?;
        let subscript_x_offset = read_fword(r)?;
        let subscript_y_offset = read_fword(r)?;
        let superscript_x_size = read_fword(r)?;
        let superscript_y_size = read_fword(r)?;
        let superscript_x_offset = read_fword(r)?;
        let superscript_y_offset = read_fword(r)?;
        let strikeout_size = read_fword(r)?;
        let strikeout_position = read_fword(r)?;
        let family_class = read_i16be(r)?;
        let mut panose = [0; 10];
        r.read_exact(&mut panose)?;
        let mut unicode_range = [0; 16];
        r.read_exact(&mut unicode_range)?;
        let ach_vend_id = Tag(read_u32be(r)?);
        let selection = read_u16be(r)?;
        let first_char_index = read_char(r)?;
        let last_char_index = read_char(r)?;
        let typo_ascender = read_fword(r)?;
        let typo_descender = read_fword(r)?;
        let typo_line_gap = read_fword(r)?;
        let win_ascent = read_ufword(r)?;
        let win_descent = read_ufword(r)?;

        let v1 = if version >= 1 {
            let mut code_page_range = [0; 8];
            r.read_exact(&mut code_page_range)?;
            Some(Os2MetricsV1 { code_page_range })
        } else {
            None
        };

        let v2 = if version >= 2 {
            let height = read_fword(r)?;
            let cap_height = read_fword(r)?;
            let default_char = read_char(r)?;
            let break_char = read_char(r)?;
            let max_context = read_u16be(r)?;
            Some(Os2MetricsV2 {
                height,
                cap_height,
                default_char,
                break_char,
                max_context,
            })
        } else {
            None
        };

        let v5 = if version >= 5 {
            let lower_optical_point_size = read_u16be(r)?;
            let upper_optical_point_size = read_u16be(r)?;
            Some(Os2MetricsV5 {
                lower_optical_point_size,
                upper_optical_point_size,
            })
        } else {
            None
        };

        Ok(Os2Metrics {
            version,
            avg_char_width,
            weight_class,
            width_class,
            flags,
            subscript_x_size,
            subscript_y_size,
            subscript_x_offset,
            subscript_y_offset,
            superscript_x_size,
            superscript_y_size,
            superscript_x_offset,
            superscript_y_offset,
            strikeout_size,
            strikeout_position,
            family_class,
            panose,
            unicode_range,
            ach_vend_id,
            selection,
            first_char_index,
            last_char_index,
            typo_ascender,
            typo_descender,
            typo_line_gap,
            win_ascent,
            win_descent,
            v1,
            v2,
            v5,
        })
    }
}

/// Compute the range/selector/shift of a table.
///
/// Ordered tables in a TTF are intended for binary search.
/// These values are usually precomputed, then ignored by the reader.
fn range_selector_shift(n: u16, sz: u16) -> (u16, u16, u16) {
    let pow2 = n.isolate_highest_one();
    let log2 = pow2.highest_one().unwrap_or(0) as u16;
    (pow2 * sz, log2, (n - pow2) * sz)
}

/// Computes the checksum of a data block
fn compute_checksum(data: &[u8]) -> u32 {
    let (chks, chk_r) = data.as_chunks::<4>();
    let checksum = chks
        .iter()
        .map(|c| u32::from_be_bytes(*c))
        .fold(0, u32::wrapping_add);
    let mut rem = [0; 4];
    rem[..chk_r.len()].copy_from_slice(chk_r);
    let r = u32::from_be_bytes(rem);
    checksum.wrapping_add(r)
}

// Helper I/O functions
fn read_u32be<R: Read>(data: &mut R) -> Result<u32> {
    let mut bs = [0; 4];
    data.read_exact(&mut bs)?;
    Ok(u32::from_be_bytes(bs))
}

fn read_u16be<R: Read>(data: &mut R) -> Result<u16> {
    let mut bs = [0; 2];
    data.read_exact(&mut bs)?;
    Ok(u16::from_be_bytes(bs))
}

fn write_u32be<W: Write>(data: &mut W, x: u32) -> Result<()> {
    data.write_all(&x.to_be_bytes())?;
    Ok(())
}

fn write_u16be<W: Write>(data: &mut W, x: u16) -> Result<()> {
    data.write_all(&x.to_be_bytes())?;
    Ok(())
}

fn read_i16be<R: Read>(data: &mut R) -> Result<i16> {
    let mut bs = [0; 2];
    data.read_exact(&mut bs)?;
    Ok(i16::from_be_bytes(bs))
}

fn write_i16be<W: Write>(data: &mut W, x: i16) -> Result<()> {
    data.write_all(&x.to_be_bytes())?;
    Ok(())
}

fn read_char<R: Read>(data: &mut R) -> Result<Char> {
    read_u16be(data).map(Char)
}

fn read_fword<R: Read>(data: &mut R) -> Result<FWord> {
    read_i16be(data).map(FixWord)
}

fn read_ufword<R: Read>(data: &mut R) -> Result<UFWord> {
    read_u16be(data).map(FixWord)
}

/// A TTF font file, partially parsed
#[derive(Debug)]
pub struct FontFile<'a> {
    tables: BTreeMap<Tag, TableHeader<'a>>,
    head: Head,
    hhea: HorizontalHeaderTable,
    hmtx: HorizontalMetricsTable,
    loca: IndexToLocation,
    cmap: HashMap<Char, GlyphId>,
    cmap_inv: Vec<Char>, // indexed by glyph_id
    kernings: KerningTable,
    ligatures: Option<LigatureTable>,
    os_2: Option<Os2Metrics>,
}

pub enum Coverage<'a> {
    Chars(&'a [Char]),
    Glyphs(&'a [GlyphId]),
    GlyphsMap(GlyphMap),
}

pub struct GlyphMap {
    // maps the old id to the new id
    old_to_new: BTreeMap<GlyphId, GlyphId>,
    // the new glyph ids are sequential, so the inverse map is just a vec:
    // the index is the new_id, the value the old_id.
    by_new: Vec<GlyphId>,
}

impl Default for GlyphMap {
    fn default() -> GlyphMap {
        let mut res = GlyphMap {
            old_to_new: BTreeMap::new(),
            by_new: Vec::new(),
        };
        // the .notdef is fixed at id=0
        res.map(GlyphId(0));
        res
    }
}

impl GlyphMap {
    pub fn map(&mut self, old_id: GlyphId) -> GlyphId {
        let entry = self.old_to_new.entry(old_id);
        use std::collections::btree_map::Entry;
        match entry {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let Ok(id) = u16::try_from(self.by_new.len()) else {
                    return GlyphId(0);
                };
                let id = GlyphId(id);
                self.by_new.push(old_id);
                *v.insert(id)
            }
        }
    }

    // Returns the mapped old glyphs in the order the were registered, so that the position is the new glyph.
    pub fn iter_old_by_new(&self) -> impl Iterator<Item = GlyphId> {
        self.by_new.iter().copied()
    }
}

impl<'a> FontFile<'a> {
    pub fn parse_ttf(data: &'a [u8]) -> Result<FontFile<'a>> {
        let mut r = Cursor::new(data);
        let r = &mut r;
        let sig = read_u32be(r)?;
        if sig != TTF_SIGNATURE {
            bail!("invalid signature");
        }

        let num_tables = read_u16be(r)?;
        let _search_range = read_u16be(r)?;
        let _entry_selector = read_u16be(r)?;
        let _range_shift = read_u16be(r)?;

        let mut tables = BTreeMap::new();
        for _ in 0..num_tables {
            let table_tag = Tag(read_u32be(r)?);
            let _checksum = read_u32be(r)?;
            let offset = read_u32be(r)?;
            let length = read_u32be(r)?;

            let table_data = &data[offset as usize..][..length as usize];
            let th = TableHeader { data: table_data };
            tables.insert(table_tag, th);
        }

        let head: Head = Head::parse(tables.get(&HEAD).unwrap().data)?;
        let maxp: MaximumProfile = MaximumProfile::parse(tables.get(&MAXP).unwrap().data)?;
        let hhea: HorizontalHeaderTable =
            HorizontalHeaderTable::parse(tables.get(&HHEA).unwrap().data)?;
        let hmtx: HorizontalMetricsTable = HorizontalMetricsTable::parse(
            hhea.num_hmetrics,
            maxp.num_glyphs,
            tables.get(&HMTX).unwrap().data,
        )?;
        let loca: IndexToLocation = IndexToLocation::parse(
            head.index_to_loc_format,
            maxp.num_glyphs,
            tables.get(&LOCA).unwrap().data,
        )?;
        let cmap: CharMap = CharMap::parse(tables.get(&CMAP).unwrap().data)?;

        // Convert the parsed charmap into a HashMap and inverse map, that is easier to use.
        // The glyph set is sequential, so instead of a map just use a vec indexed by glyph_id.
        let mut cmap_inv = vec![Char(0); maxp.num_glyphs as usize];
        let cmap: HashMap<Char, GlyphId> = cmap
            .iter()
            .inspect(|(c, g)| {
                if let Some(ci) = cmap_inv.get_mut(g.0 as usize) {
                    *ci = *c
                }
            })
            .collect();

        // We handle two sources of kernings: GPOS and KERN.
        let mut kernings = KerningTable::default();
        if let Some(gpos) = tables.get(&GPOS) {
            kernings.parse_gpos(gpos.data)?;
        }
        if let Some(kern) = tables.get(&KERN) {
            kernings.parse_kern(kern.data)?;
        }

        let ligatures = tables
            .get(&GSUB)
            .and_then(|gsub| LigatureTable::parse(gsub.data).ok());
        let os_2 = tables
            .get(&OS_2)
            .and_then(|os_2| Os2Metrics::parse(os_2.data).ok());

        Ok(FontFile {
            tables,
            head,
            hhea,
            hmtx,
            loca,
            cmap,
            cmap_inv,
            kernings,
            ligatures,
            os_2,
        })
    }

    pub fn units_per_em(&self) -> UFWord {
        self.head.units_per_em
    }

    pub fn bounding_box(&self) -> [FWord; 4] {
        self.head.bounding_box
    }

    pub fn os_2_metrics(&self) -> Option<&Os2Metrics> {
        self.os_2.as_ref()
    }

    pub fn h_header(&self) -> &HorizontalHeaderTable {
        &self.hhea
    }

    /// The best-guess ascender.
    pub fn ascender(&self) -> FWord {
        self.os_2
            .as_ref()
            .map(|m| m.typo_ascender)
            .unwrap_or(self.hhea.ascender)
    }

    /// The best-guess descender.
    pub fn descender(&self) -> FWord {
        self.os_2
            .as_ref()
            .map(|m| m.typo_descender)
            .unwrap_or(self.hhea.descender)
    }

    /// The best-guess line gap.
    pub fn line_gap(&self) -> FWord {
        self.os_2
            .as_ref()
            .map(|m| m.typo_line_gap)
            .unwrap_or(self.hhea.line_gap)
    }

    /// The best-guess line advance.
    pub fn line_advance(&self) -> FWord {
        let ascender = self.ascender();
        let descender = self.descender();
        let line_gap = self.line_gap();
        FixWord(ascender.0 - descender.0 + line_gap.0)
    }

    /// The best-guess cap-height (uppercase 'H').
    pub fn cap_height(&self) -> FWord {
        // The proper CapHeight is in OS/2 v2
        // If not, use the ymax of the 'H' glyph, that is usually the right one.
        // If there is no H glyph, we could try some other such as X or M, but now we are just guessing,
        // so just use 70% of the ascender.
        self.os_2
            .as_ref()
            .and_then(|m| m.v2.as_ref())
            .map(|m| m.cap_height)
            .or_else(|| {
                let h_id = self.map_char(Char(u16::from(b'H')))?;
                let bb = self.glyph_bounding_box(h_id)?;
                Some(bb[3])
            })
            .unwrap_or_else(|| {
                let ascender = self.ascender();
                FixWord(7 * ascender.0 / 10)
            })
    }

    /// The best-guess height (lowercase 'x').
    pub fn height(&self) -> FWord {
        // The proper Height is in OS/2 v2
        // If not, use the ymax of the 'x' glyph, that is usually the right one.
        // If there is no x glyph just use 50% of the ascender.
        self.os_2
            .as_ref()
            .and_then(|m| m.v2.as_ref())
            .map(|m| m.height)
            .or_else(|| {
                let h_id = self.map_char(Char(u16::from(b'x')))?;
                let bb = self.glyph_bounding_box(h_id)?;
                Some(bb[3])
            })
            .unwrap_or_else(|| {
                let ascender = self.ascender();
                FixWord(ascender.0 / 2)
            })
    }

    /// Return in degrees
    pub fn italic_angle(&self) -> i32 {
        // This is copied from printpdf...
        if self.hhea.caret_slope_run == 0 {
            // short circuit the common case
            0
        } else {
            -(self.hhea.caret_slope_run as f32)
                .atan2(self.hhea.caret_slope_rise as f32)
                .to_degrees()
                .round() as i32
        }
    }

    pub fn stem_v(&self) -> i32 {
        // Copied from printpdf...
        match self.os_2.as_ref() {
            None => 80,
            Some(os2) => 50 + ((os2.weight_class as f32 / 65.0).powi(2)).round() as i32,
        }
    }

    pub fn map_char(&self, c: Char) -> Option<GlyphId> {
        self.cmap.get(&c).copied()
    }

    pub fn unmap_glyph(&self, g: GlyphId) -> Option<Char> {
        let c = self.cmap_inv.get(g.0 as usize);
        match c {
            Some(&Char(x)) if x != 0 => Some(Char(x)),
            _ => None,
        }
    }

    pub fn glyph_advance_width(&self, g: GlyphId) -> Option<UFWord> {
        self.hmtx.get(g).map(|m| m.advance_width)
    }

    pub fn glyph_bounding_box(&self, g: GlyphId) -> Option<[FWord; 4]> {
        let glyf_data = self.tables.get(&GLYF)?.data;
        let glyf = self.loca.find(g, glyf_data);

        let mut r = Cursor::new(&glyf);
        let r = &mut r;
        let _num_contours = read_i16be(r).ok()?;
        let xmin = read_fword(r).ok()?;
        let ymin = read_fword(r).ok()?;
        let xmax = read_fword(r).ok()?;
        let ymax = read_fword(r).ok()?;
        Some([xmin, ymin, xmax, ymax])
    }

    pub fn kerning(&self, a: GlyphId, b: GlyphId) -> Option<FWord> {
        self.kernings.get(a, b)
    }

    pub fn max_ligature_len(&self) -> Option<usize> {
        self.ligatures.as_ref().map(|lt| lt.max_ligature_len)
    }

    pub fn ligature(&self, ids: &[GlyphId]) -> Option<(usize, GlyphId)> {
        let table = self.ligatures.as_ref()?;
        table.get(ids)
    }

    pub fn subset(&self, coverage: Coverage<'_>) -> Result<Vec<u8>> {
        let mut cmap_ranges = Vec::<CharMapSegment>::new();

        let mut cmap_add = |c: Char, id: GlyphId| {
            let c = c.0;
            let id = id.0;
            if let Some(cur) = cmap_ranges.last_mut() {
                if c <= cur.range.last {
                    // should not happen
                    return;
                }
                if cur.range.last == c.wrapping_sub(1)
                    && cur.range.last.wrapping_add_signed(cur.delta) == id.wrapping_sub(1)
                {
                    // extend the last range
                    cur.range.last = c;
                    return;
                }
                // gap
            }

            // new range
            cmap_ranges.push(CharMapSegment {
                range: RangeInclusive { start: c, last: c },
                delta: id.wrapping_sub(c).cast_signed(),
                offset: None,
            });
        };

        let mut glyph_map = match coverage {
            Coverage::Chars(cov_chars) => {
                let mut glyph_map = GlyphMap::default();
                let mut cov = cov_chars
                    .iter()
                    .map(|c| TryInto::try_into(*c))
                    .collect::<std::result::Result<Vec<Char>, _>>()?;
                // coverage must be sorted
                cov.sort();

                for &c in &cov {
                    let Some(old_id) = self.map_char(c) else {
                        continue;
                    };
                    let new_id = glyph_map.map(old_id);
                    cmap_add(c, new_id);
                }
                if cov.last() != Some(&Char(0xFFFF)) {
                    // U+FFFF is usually mapped to .notdef
                    cmap_add(Char(0xFFFF), GlyphId(0));
                }
                glyph_map
            }
            Coverage::Glyphs(cov_glyphs) => {
                if cov_glyphs.get(0) != Some(&GlyphId(0)) {
                    bail!(".notdef 0 should be map to 0");
                }
                let mut glyph_map = GlyphMap::default();
                for (i, g) in cov_glyphs.iter().enumerate() {
                    glyph_map.old_to_new.insert(*g, GlyphId(i as u16));
                }
                glyph_map.by_new = cov_glyphs.to_vec();
                glyph_map
            }
            Coverage::GlyphsMap(glyph_map) => glyph_map,
        };

        let glyf_data = self.tables.get(&GLYF).unwrap().data;

        let mut new_glyf = Vec::<u8>::new();
        let mut new_loca = Vec::<u8>::new();
        let mut new_hmtx = Vec::<u8>::new();
        //let mut new_cmap = Vec::<u8>::new();
        let mut new_hhea = self.tables.get(&HHEA).unwrap().data.to_vec();
        let mut new_maxp = self.tables.get(&MAXP).unwrap().data.to_vec();
        let mut new_head = self.tables.get(&HEAD).unwrap().data.to_vec();

        // This loop writes the _end_ of the current glyph, not the start as you would expect.
        let write_loca = match self.head.index_to_loc_format {
            IndexToLocFormat::Short16 => |w: &mut Vec<u8>, x: usize| write_u16be(w, (x / 2) as u16),
            IndexToLocFormat::Long32 => |w: &mut Vec<u8>, x: usize| write_u32be(w, x as u32),
        };

        write_loca(&mut new_loca, 0).unwrap();

        // use a custom loop because compose glyph can insert extra glyph ids, that adds to the list to be iterated
        let mut i = 0;
        while i < glyph_map.by_new.len() {
            let old_id = glyph_map.by_new[i];
            i += 1;

            let hm = self.hmtx.get(old_id).unwrap();
            let glyf = self.loca.find(old_id, glyf_data);

            if !glyf.is_empty() {
                let mut r = Cursor::new(&glyf);
                let r = &mut r;
                let num_contours = read_i16be(r)?;
                let _xmin = read_fword(r)?;
                let _ymin = read_fword(r)?;
                let _xmax = read_fword(r)?;
                let _ymax = read_fword(r)?;

                if num_contours < 0 {
                    // It is a compose glyph, we have to replace the old glyph ids for the new ones
                    let mut patched_glyph = glyf.to_vec();

                    loop {
                        let flags = read_u16be(r).unwrap();
                        let flags = CompositeGlyphFlags::from_bits_retain(flags);

                        let sub_id_offs = r.position() as usize;
                        let sub_glyph_id = GlyphId(read_u16be(r).unwrap());
                        let sub_new_id = glyph_map.map(sub_glyph_id);
                        patched_glyph[sub_id_offs..][..2]
                            .copy_from_slice(&sub_new_id.0.to_be_bytes());

                        if !flags.contains(CompositeGlyphFlags::MORE_COMPONENTS) {
                            break;
                        }
                        // the sub-glyph has variable length depending on the flags
                        let mut skip = 0;
                        if flags.contains(CompositeGlyphFlags::ARG_1_AND_2_ARE_WORDS) {
                            skip += 2 * 2; // arg1 + arg2
                        } else {
                            skip += 2 * 1; // arg12
                        }
                        if flags.contains(CompositeGlyphFlags::WE_HAVE_A_SCALE) {
                            skip += 2; // scale
                        } else if flags.contains(CompositeGlyphFlags::WE_HAVE_AN_X_AND_Y_SCALE) {
                            skip += 2 * 2; // scalex, scaley
                        } else if flags.contains(CompositeGlyphFlags::WE_HAVE_A_TWO_BY_TWO) {
                            skip += 4 * 2; // matrix 2x2
                        }
                        r.seek_relative(skip).unwrap();
                    }
                    new_glyf.write_all(&patched_glyph).unwrap();
                } else {
                    new_glyf.write_all(&glyf).unwrap();
                }
            }

            write_loca(&mut new_loca, new_glyf.len()).unwrap();

            write_u16be(&mut new_hmtx, hm.advance_width.0).unwrap();
            write_i16be(&mut new_hmtx, hm.lsb.0).unwrap();
        }

        // fix numberOfHMetrics
        new_hhea[2 * 17..][..2].copy_from_slice(&(glyph_map.old_to_new.len() as u16).to_be_bytes());
        // fix numGlyphs
        new_maxp[2 * 2..][..2].copy_from_slice(&(glyph_map.old_to_new.len() as u16).to_be_bytes());
        // fix checksum_adjust
        new_head[8..][..4].copy_from_slice(&0u32.to_be_bytes());

        // Beware of the alphabetical order!
        let mut new_tables = Vec::new();
        if let Some(cvt) = self.tables.get(&CVT) {
            new_tables.push((CVT, cvt.data));
        }
        if let Some(fpgm) = self.tables.get(&FPGM) {
            new_tables.push((FPGM, fpgm.data));
        }
        new_tables.push((GLYF, &new_glyf));
        new_tables.push((HEAD, &new_head));
        new_tables.push((HHEA, &new_hhea));
        new_tables.push((HMTX, &new_hmtx));
        new_tables.push((LOCA, &new_loca));
        new_tables.push((MAXP, &new_maxp));
        if let Some(prep) = self.tables.get(&PREP) {
            new_tables.push((PREP, prep.data));
        }

        let (r, se, sh) = range_selector_shift(new_tables.len() as u16, 16);

        let mut file = Vec::new();
        write_u32be(&mut file, TTF_SIGNATURE).unwrap(); // signature
        write_u16be(&mut file, new_tables.len() as u16).unwrap(); // num tables
        write_u16be(&mut file, r).unwrap(); // range
        write_u16be(&mut file, se).unwrap(); // selector
        write_u16be(&mut file, sh).unwrap(); // shift

        let mut offset_head = None;
        let mut offset = 12 + 16 * new_tables.len() as u32;

        for t in &mut new_tables {
            if t.0 == HEAD {
                offset_head = Some(offset);
            }

            let checksum = compute_checksum(t.1);
            write_u32be(&mut file, t.0.0).unwrap(); // tag
            write_u32be(&mut file, checksum).unwrap(); // checksum
            write_u32be(&mut file, offset).unwrap(); // offset
            write_u32be(&mut file, t.1.len() as u32).unwrap(); // data
            offset += t.1.len().next_multiple_of(4) as u32;
        }

        for t in &new_tables {
            file.write_all(t.1).unwrap();
            let pad = t.1.len().next_multiple_of(4) - t.1.len();
            file.write_all(&[0; 3][..pad]).unwrap();
        }

        if let Some(offset_head) = offset_head {
            let checksum = compute_checksum(&file);
            let checksum_adjust = GLOBAL_CHECKSUM.wrapping_sub(checksum);
            file[offset_head as usize + 8..][..4].copy_from_slice(&checksum_adjust.to_be_bytes());
        }
        Ok(file)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rss_16() {
        // tables: 19 range: 256 selector: 4 shift: 48
        let res = range_selector_shift(19, 16);
        assert_eq!(res, (256, 4, 48));

        // tables: 12 range: 128 selector: 3 shift: 64
        let res = range_selector_shift(12, 16);
        assert_eq!(res, (128, 3, 64));
    }

    #[test]
    fn test_rss_2() {
        let res = range_selector_shift(3754, 2);
        assert_eq!(res, (4096, 11, 3412));

        let res = range_selector_shift(11, 2);
        assert_eq!(res, (16, 3, 6));

        let res = range_selector_shift(173, 2);
        assert_eq!(res, (256, 7, 90));
    }
}

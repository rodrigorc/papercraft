// The lopdf crate does not have font metrics, that is a viewer thing...
// That said there is a bunch of well known files *.afm, that contain the metrics of each standard PDF font.
// I have written a quick'n'dirty `build.rs` parser that reads the *.afm file and converts it into
// a few static Rust structures.

mod helvetica {
    include!(concat!(env!("OUT_DIR"), "/helvetica_afm.rs"));
}

fn find_in_vec_tuple<V>(key: char, data: &[(char, V)]) -> Option<&V> {
    let i = data.binary_search_by_key(&key, |(a, _)| *a).ok()?;
    Some(&data[i].1)
}

/// Given a text returns the total width and a list of (kerning, glyph-id).
pub fn measure_helvetica(text: &str) -> (i32, Vec<(i64, u16)>) {
    let mut width = 0;
    let mut prev = '\u{0}';
    let mut cps = Vec::with_capacity(text.len());
    for c in text.chars() {
        let Some(info) = find_in_vec_tuple(c, &helvetica::CHARS) else {
            continue;
        };
        let kern = find_in_vec_tuple(prev, info.kerns).copied().unwrap_or(0);

        width += info.width as i32 + kern;
        cps.push((-kern as i64, c as u16));
        prev = c;
    }
    (width, cps)
}

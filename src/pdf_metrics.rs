// The printpdf crate does not have font metrics, that is a viewer thing...
// That said there is a bunch of well known files *.afm, that contain the metrics of each standard PDF font.
// I have written a quick'n'dirty `build.rs` parser that reads the *.afm file and converts it into
// a few static Rust structures.

mod helvetica {
    include!(concat!(env!("OUT_DIR"), "/helvetica_afm.rs"));
}

pub fn measure_helvetica(text: &str) -> (i32, Vec<(i64, u16)>) {
    let mut width = 0;
    let mut prev = '\u{0}';
    let mut cps = Vec::with_capacity(text.len());
    for c in text.chars() {
        let Some(&w) = helvetica::WIDTHS.get(&c) else {
            continue;
        };
        let kern = helvetica::KERNS
            .get(&prev)
            .and_then(|ks| ks.iter().find(|(b, _)| *b == c))
            .map(|(_, k)| *k)
            .unwrap_or(0);

        width += w as i32 + kern;
        cps.push((-kern as i64, c as u16));
        prev = c;
    }
    (width, cps)
}

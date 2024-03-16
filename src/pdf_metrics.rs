// The printpdf crate does not have font metrics, that is a viewer thing...
// That said there is a bunch of well known files *.afm, that contain the metrics of each standard PDF font.
// I have written a quick'n'dirty `build.rs` parser that reads the *.afm file and converts it into
// a few static Rust structures.

mod helvetica {
    include!(concat!(env!("OUT_DIR"), "/helvetica_afm.rs"));
}

pub fn measure_helvetica(text: &str) -> (i32, Vec<(i64, u16)>) {
    let mut width = 0;
    let mut prev = 0;
    let mut cps = Vec::with_capacity(text.len());
    for c in text.chars() {
        let c = u32::from(c);
        // ASCII only, for now
        let c = if c < 128 { c as u8 } else { b'?' };

        let w = helvetica::WIDTHS[c as usize];
        let kern = if let Some((_, k)) = helvetica::KERNS[prev as usize]
            .iter()
            .find(|(b, _)| *b == c)
        {
            *k
        } else {
            0
        };
        width += w + kern;
        cps.push((-kern as i64, c as u16));
        prev = c;
    }
    (width, cps)
}

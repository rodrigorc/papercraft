use std::env;
use std::io::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    build_resource()?;
    build_helvetica()?;
    build_locales()?;
    Ok(())
}

fn build_resource() -> Result<()> {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS");
    if target_os.as_deref() == Ok("windows") {
        let name = env!("CARGO_PKG_NAME");
        let version = env!("CARGO_PKG_VERSION");
        let repo = env!("CARGO_PKG_REPOSITORY");
        let output_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| ".".to_string());
        let header = std::path::PathBuf::from(&output_dir).join("papercraft.h");
        std::fs::write(
            header,
            format!(
                r#"
#define PC_PROJECT "{name}"
#define PC_VERSION "{version}"
#define PC_REPO "{repo}"
"#
            ),
        )?;
        let output = std::path::PathBuf::from(&output_dir).join("resource.o");
        #[allow(clippy::option_env_unwrap)]
        let status = if let Some(windres) = option_env!("WINDRES") {
            std::process::Command::new(windres)
                .arg("-I")
                .arg(&output_dir)
                .arg("res/resource.rc")
                .arg(&output)
                .status()?
        } else if let Some(rc) = option_env!("RC") {
            std::process::Command::new(rc)
                .arg("/i")
                .arg(&output_dir)
                .arg("/fo")
                .arg(&output)
                .arg("res/resource.rc")
                .status()?
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "RC or WINDRES should be defined",
            ));
        };
        if !status.success() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "windres error",
            ));
        }
        println!("cargo:rustc-link-arg={}", output.display());
        for entry in std::fs::read_dir("res")? {
            let entry = entry?;
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }
    Ok(())
}

// Metrics for well-known PDF fonts are in AFM files
fn build_helvetica() -> Result<()> {
    use std::{
        collections::HashMap,
        fs::File,
        io::{BufRead, BufReader, BufWriter, Write},
    };

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out = File::create(out_path.join("helvetica_afm.rs"))?;
    let mut out = BufWriter::new(out);

    let mut widths = HashMap::<u16, u32>::new(); // Unicode to width
    let mut names = HashMap::<String, u16>::new(); // name to Unicode
    let mut kerns = HashMap::<u16, Vec<(u16, i32)>>::new(); // Unicode to list of (Unicode, kerning)

    println!("cargo:rerun-if-changed=thirdparty/afm/names.txt");
    let char_names = File::open("thirdparty/afm/names.txt").unwrap();
    let char_names = BufReader::new(char_names);
    for line in char_names.lines() {
        let line = line.unwrap();
        let pieces: Vec<&str> = line.split('\t').collect();
        let name = pieces[0];
        let code = u16::from_str_radix(&pieces[1], 16).unwrap();
        names.insert(name.to_owned(), code);
    }

    println!("cargo:rerun-if-changed=thirdparty/afm/Helvetica.afm");
    let afm = File::open("thirdparty/afm/Helvetica.afm").unwrap();
    let afm = BufReader::new(afm);

    for line in afm.lines() {
        let line = line.unwrap();
        let pieces: Vec<&str> = line.split(';').collect();
        let words0: Vec<&str> = pieces[0].split_ascii_whitespace().collect();
        if words0.is_empty() {
            continue;
        }
        match words0[0] {
            "C" => {
                let mut width: Option<u32> = None;
                let mut name: Option<&str> = None;
                for piece in &pieces[1..] {
                    let words: Vec<&str> = piece.split_ascii_whitespace().collect();
                    if words.is_empty() {
                        continue;
                    }
                    match words[0] {
                        "WX" => {
                            width = Some(words[1].parse().unwrap());
                        }
                        "N" => {
                            name = Some(words[1]);
                        }
                        _ => {}
                    }
                }
                if let (Some(width), Some(name)) = (width, name) {
                    let Some(&char) = names.get(name) else {
                        continue;
                    };
                    widths.insert(char, width);
                }
            }
            "KPX" => {
                let Some(&c1) = names.get(words0[1]) else {
                    continue;
                };
                let Some(&c2) = names.get(words0[2]) else {
                    continue;
                };
                let kern: i32 = words0[3].parse().unwrap();
                kerns.entry(c1).or_default().push((c2, kern));
            }
            _ => {}
        }
    }

    writeln!(out, "use std::collections::HashMap;")?;
    writeln!(out, "use std::sync::LazyLock;")?;
    writeln!(
        out,
        "pub static WIDTHS: LazyLock<HashMap<char, u32>> = LazyLock::new(|| ["
    )?;
    for (c, w) in widths {
        writeln!(out, "('\\u{{{c:x}}}', {w}),")?;
    }
    writeln!(out, "].into());")?;
    writeln!(out)?;

    write!(
        out,
        "pub static KERNS: LazyLock<HashMap<char, Vec<(char, i32)>>> = LazyLock::new(|| ["
    )?;
    for (c1, kerns) in kerns {
        write!(out, "('\\u{{{c1:x}}}', vec![")?;
        for (c2, kern) in kerns {
            write!(out, "('\\u{{{c2:x}}}', {kern}), ")?;
        }
        writeln!(out, "]),")?;
    }
    writeln!(out, "].into());")?;
    Ok(())
}

fn build_locales() -> Result<()> {
    let output_dir = std::env::var("OUT_DIR").unwrap();
    let out = PathBuf::from(&output_dir).join("locale/translators.rs");
    include_po::generate_locales_from_dir("locales", out).unwrap();
    Ok(())
}

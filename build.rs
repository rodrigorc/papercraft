use std::env;
use std::io::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    build_resource()?;
    build_helvetica()?;
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
    let afm = File::open("thirdparty/afm/Helvetica.afm").unwrap();
    let afm = BufReader::new(afm);

    let mut widths = [0; 128];
    let mut names = HashMap::new();
    let mut kerns = vec![Vec::new(); 128];

    for line in afm.lines() {
        let line = line.unwrap();
        let pieces: Vec<&str> = line.split(';').collect();
        let words0: Vec<&str> = pieces[0].split_ascii_whitespace().collect();
        if words0.is_empty() {
            continue;
        }
        match words0[0] {
            "C" => {
                let code: i32 = words0[1].parse().unwrap();
                if !(0..128).contains(&code) {
                    continue;
                }
                for piece in &pieces[1..] {
                    let words: Vec<&str> = piece.split_ascii_whitespace().collect();
                    if words.is_empty() {
                        continue;
                    }
                    match words[0] {
                        "WX" => {
                            let width: u32 = words[1].parse().unwrap();
                            widths[code as usize] = width;
                        }
                        "N" => {
                            let name = words[1];
                            names.insert(String::from(name), code);
                        }
                        _ => {}
                    }
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
                kerns[c1 as usize].push((c2, kern));
            }
            _ => {}
        }
    }

    write!(out, "pub static WIDTHS: [i32; 128] = [")?;
    for w in widths {
        write!(out, "{w},")?;
    }
    writeln!(out, "];")?;
    write!(out, "pub static KERNS: [&[(u8, i32)]; 128] = [")?;
    for bks in &kerns {
        write!(out, "&[")?;
        for &(b, k) in bks {
            write!(out, "({b},{k}),")?;
        }
        write!(out, "],")?;
    }
    writeln!(out, "];")?;
    Ok(())
}

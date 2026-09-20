use anyhow::{Result, bail};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    build_resource()?;
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
            bail!("RC or WINDRES should be defined");
        };
        if !status.success() {
            bail!("windres error");
        }
        println!("cargo:rustc-link-arg={}", output.display());
        for entry in std::fs::read_dir("res")? {
            let entry = entry?;
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }
    Ok(())
}

fn build_locales() -> Result<()> {
    let output_dir = std::env::var("OUT_DIR")?;
    let out = PathBuf::from(&output_dir).join("locale/translators.rs");
    include_po::generate_locales_from_dir("locales", out)?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS");
    if target_os.as_deref() == Ok("windows") {
        let name = env!("CARGO_PKG_NAME");
        let version = env!("CARGO_PKG_VERSION");
        let repo = env!("CARGO_PKG_REPOSITORY");
        let output_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| ".".to_string());
        let header = std::path::PathBuf::from(&output_dir).join("papercraft.h");
        std::fs::write(&header,
            format!(r#"
#define PC_PROJECT "{name}"
#define PC_VERSION "{version}"
#define PC_REPO "{repo}"
"#)
        )?;
        let output = std::path::PathBuf::from(&output_dir).join("resource.o");
        let status = std::process::Command::new("i686-w64-mingw32-windres")
            .arg("-I").arg(&output_dir)
            .arg("res/resource.rc")
            .arg(&output)
            .status()?;
        if !status.success() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "windres error",
            ));
        }
        println!("cargo:rustc-link-arg={}", output.display());
        println!("cargo:rerun-if-changed=build.rs");
        for entry in std::fs::read_dir("res")? {
            let entry = entry?;
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }
    Ok(())
}

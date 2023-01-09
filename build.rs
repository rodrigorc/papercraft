use std::io::Result;
use std::path::{PathBuf, Path};
use std::env;

fn main() -> Result<()> {
    build_resource()?;
	build_imgui_filedialog()?;
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
        std::fs::write(&header,
            format!(r#"
#define PC_PROJECT "{name}"
#define PC_VERSION "{version}"
#define PC_REPO "{repo}"
"#)
        )?;
        let output = std::path::PathBuf::from(&output_dir).join("resource.o");
        let status = std::process::Command::new(option_env!("WINDRES").expect("WINDRES envvar is undefined"))
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

fn build_imgui_filedialog() -> Result<()> {
    let dep_imgui_path =
        env::var("DEP_IMGUI_THIRD_PARTY")
            .expect("DEP_IMGUI_THIRD_PARTY not defined");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let igfd_path = "thirdparty/ImGuiFileDialog";

    bindgen::Builder::default()
        .clang_arg(format!("-I{dep_imgui_path}"))
        .clang_arg(format!("-I{igfd_path}"))
        .allowlist_function("IGFD_.*")
        .allowlist_function("free") // standard libc free to release strings
        .allowlist_type("IGFD_.*")
        .allowlist_type("ImGuiFileDialog.*")
        .header("ImGuiFileDialogWrapper.h")
        .generate()
        .expect("Error ImGuiFileDialog building bindings")
        .write_to_file(out_path.join("imgui_filedialog_bindings.rs"))
        .expect("Error ImGuiFileDialog writing bindings");

    cc::Build::new()
        .include(Path::new(&dep_imgui_path).join("imgui"))
        .include(igfd_path)
        .warnings(false)
        .file(Path::new(igfd_path).join("ImGuiFileDialog.cpp"))
        .compile("libimguifd.a");

    Ok(())
}


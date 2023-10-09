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
    for x in env::vars() {
        dbg!(x);
    }
    let dep_imgui_path =
        env::var("DEP_IMGUI_THIRD_PARTY")
            .expect("DEP_IMGUI_THIRD_PARTY not defined");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let igfd_path = "thirdparty/ImGuiFileDialog";

    bindgen::Builder::default()
        .clang_args(["-x", "c++"])
        .clang_arg(format!("-I{dep_imgui_path}"))
        .clang_arg(format!("-I{igfd_path}"))
        .header("ImGuiFileDialogWrapper.h")
        .allowlist_recursively(false)
        .allowlist_function("free") // standard libc free to release strings
        //.allowlist_type("IGFD_.*")
        //.allowlist_function("ImGuiFileDialog.*")
        //.allowlist_file("thirdparty/ImGuiFileDialog/ImGuiFileDialog.h")
        //.blocklist_function(".*")
        //.blocklist_type(".*")
        .allowlist_function("IGFD_.*")
        .blocklist_function(".*Pane.*")
        .blocklist_type("ImGuiFileDialog")
        .allowlist_type("IGFD_Selection_Pair")
        .allowlist_type("IGFD_Selection")
        .allowlist_type("ImGuiFileDialogFlags.*")
        .allowlist_type("IGFD_FileStyleFlags")
        .allowlist_type("IGFD_ResultMode.*")
        .generate()
        .expect("Error ImGuiFileDialog building bindings")
        .write_to_file(out_path.join("imgui_filedialog_bindings.rs"))
        .expect("Error ImGuiFileDialog writing bindings");

    cc::Build::new()
        .cpp(true)
        .include(dep_imgui_path)
        .include(igfd_path)
        .define("USE_EXPLORATION_BY_KEYS", None)
        .define("IGFD_KEY_UP", "ImGuiKey_UpArrow")
        .define("IGFD_KEY_DOWN", "ImGuiKey_DownArrow")
        .define("IGFD_KEY_ENTER", "ImGuiKey_Enter")
        .define("IGFD_KEY_BACKSPACE", "ImGuiKey_Backspace")
        .define("USE_DIALOG_EXIT_WITH_KEY", None)
        .define("IGFD_EXIT_KEY", "ImGuiKey_Escape")
        .define("FILTER_COMBO_WIDTH", "200.0f")
        .define("dirNameString", r#""Directory Path:""#)
        .define("OverWriteDialogTitleString", r#""The file already exists!""#)
        .define("OverWriteDialogMessageString", r#""Would you like to overwrite it?""#)
        .define("okButtonWidth", "100.0f")
        .define("cancelButtonWidth", "100.0f")
        .define("fileSizeBytes", r#""B""#)
        .define("fileSizeKiloBytes", r#""KiB""#)
        .define("fileSizeMegaBytes", r#""MiB""#)
        .define("fileSizeGigaBytes", r#""GiB""#)
        .warnings(false)
        .file(Path::new(igfd_path).join("ImGuiFileDialog.cpp"))
        .compile("imguifd");

    println!("cargo:rerun-if-changed=ImGuiFileDialogWrapper.h");
    for entry in std::fs::read_dir(igfd_path)? {
        let entry = entry?;
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
    Ok(())
}


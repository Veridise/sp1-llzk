use cmake::Config;
use std::path::PathBuf;
use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs::read_dir,
    path::Path,
    process::{exit, Command},
    str,
};

const BRIDGE_DIR: &str = "bridge";
const BRIDGE_LIB_NAME: &str = "llzkbridge";
const BRIDGE_HEADER: &str = "include/bridge.h";

// Taken from mlir-sys
const LLVM_MAJOR_VERSION: usize = 18;

fn link_mlir() -> Result<(), Box<dyn Error>> {
    let version = llvm_config("--version")?;

    if !version.starts_with(&format!("{LLVM_MAJOR_VERSION}.",)) {
        return Err(format!(
            "failed to find correct version ({LLVM_MAJOR_VERSION}.x.x) of llvm-config (found {version})"
        )
        .into());
    }

    let llvm_libdir = llvm_config("--libdir")?;
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rustc-link-search={}", llvm_libdir);

    for entry in read_dir(llvm_libdir)? {
        if let Some(name) = entry?.path().file_name().and_then(OsStr::to_str) {
            if name.starts_with("libMLIR") {
                if let Some(name) = parse_archive_name(name) {
                    println!("cargo:rustc-link-lib=static={name}");
                }
            }
        }
    }

    let mlir_dir = env::var("MLIR_DIR")?;
    let mlir_path = PathBuf::from(mlir_dir).join("lib");
    println!("cargo:rustc-link-search={}", mlir_path.display());
    for entry in read_dir(mlir_path)? {
        if let Some(name) = entry?.path().file_name().and_then(OsStr::to_str) {
            if name.starts_with("libMLIR") {
                if let Some(name) = parse_archive_name(name) {
                    println!("cargo:rustc-link-lib=static={name}");
                }
            }
        }
    }

    if let Ok(path) = env::var("LLZK_LIB_PATH") {
        println!("cargo:rustc-link-search={}", path);
    }
    println!("cargo:rustc-link-lib=LLZKDialect");
    println!("cargo:rustc-link-lib=LLZKDialectRegistration");

    for name in llvm_config("--libnames")?.trim().split(' ') {
        if let Some(name) = parse_archive_name(name) {
            println!("cargo:rustc-link-lib={name}");
        }
    }

    for flag in llvm_config("--system-libs")?.trim().split(' ') {
        let flag = flag.trim_start_matches("-l");

        if flag.starts_with('/') {
            // llvm-config returns absolute paths for dynamically linked libraries.
            let path = Path::new(flag);

            println!("cargo:rustc-link-search={}", path.parent().unwrap().display());
            println!(
                "cargo:rustc-link-lib={}",
                path.file_stem().unwrap().to_str().unwrap().trim_start_matches("lib")
            );
        } else {
            println!("cargo:rustc-link-lib={flag}");
        }
    }

    if let Some(name) = get_system_libcpp() {
        println!("cargo:rustc-link-lib={name}");
    }

    Ok(())
}

fn get_system_libcpp() -> Option<&'static str> {
    if cfg!(target_env = "msvc") {
        None
    } else if cfg!(target_os = "macos") {
        Some("c++")
    } else {
        Some("stdc++")
    }
}

fn llvm_config_path() -> PathBuf {
    let prefix = env::var(format!("MLIR_SYS_{LLVM_MAJOR_VERSION}0_PREFIX"))
        .map(|path| Path::new(&path).join("bin"))
        .unwrap_or_default();
    let llvm_config_exe =
        if cfg!(target_os = "windows") { "llvm-config.exe" } else { "llvm-config" };
    prefix.join(llvm_config_exe)
}

fn llvm_config(argument: &str) -> Result<String, Box<dyn Error>> {
    let call = format!("{} --link-static {argument}", llvm_config_path().display(),);

    Ok(str::from_utf8(
        &if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", &call]).output()?
        } else {
            Command::new("sh").arg("-c").arg(&call).output()?
        }
        .stdout,
    )?
    .trim()
    .to_string())
}

fn parse_archive_name(name: &str) -> Option<&str> {
    if let Some(name) = name.strip_prefix("lib") {
        name.strip_suffix(".a")
    } else {
        None
    }
}

fn generate_rust_bindings<H, O>(header: H, output: O) -> Result<(), Box<dyn Error>>
where
    H: AsRef<Path>,
    O: AsRef<Path>,
{
    let bindings = bindgen::Builder::default()
        .header(header.as_ref().to_str().unwrap())
        .clang_arg(format!("-I{}", llvm_config("--includedir")?))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()?;

    Ok(bindings.write_to_file(output)?)
}

fn build_bridge(src: &str) {
    let dst = Config::new(src)
        .define("LLVM_DIR", llvm_config("--prefix").unwrap())
        .define("MLIR_DIR", env::var("MLIR_DIR").unwrap())
        .build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static={}", BRIDGE_LIB_NAME);
}

fn main() {
    if let Err(error) = link_mlir() {
        eprintln!("{}", error);
        exit(1);
    }
    let src_dir = Path::new(BRIDGE_DIR);
    println!("cargo:rerun-if-changed={}", BRIDGE_DIR);
    build_bridge(BRIDGE_DIR);

    let header = src_dir.join(BRIDGE_HEADER);
    let bindings_output_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    if let Err(error) = generate_rust_bindings(header, bindings_output_path) {
        eprintln!("{}", error);
        exit(1);
    }
}

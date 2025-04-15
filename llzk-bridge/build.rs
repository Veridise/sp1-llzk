use cmake::Config;
use glob::glob;
use std::{env, path::Path, path::PathBuf};

const BRIDGE_DIR: &str = "bridge";
const BRIDGE_LIB_NAME: &str = "llzkbridge";
const BRIDGE_HEADER: &str = "include/bridge.h";

fn generate_rust_bindings<H, O>(header: H, output: O)
where
    H: AsRef<Path>,
    O: AsRef<Path>,
{
    let bindings = bindgen::Builder::default()
        .header(header.as_ref().to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    bindings.write_to_file(output).expect("Couldn't write bindings to file");
}

fn build_bridge(src: &str) {
    let dst = Config::new(src).build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static={}", BRIDGE_LIB_NAME);
}
// glob(src.as_ref().join("**/CMakeLists.txt"))
fn mark_sources<I, P>(it: I)
where
    I: Iterator<Item = P>,
    P: AsRef<Path>,
{
    it.for_each(|file| println!("cargo:rerun-if-changed={}", file.as_ref().to_str().unwrap()));
}

fn glob_files_helper<P: AsRef<Path>>(p: P) -> Box<dyn Iterator<Item = PathBuf>> {
    Box::new(glob(p.as_ref().to_str().unwrap()).unwrap().map(|path| {
        let path = path.unwrap();
        if !path.is_file() {
            panic!("was expecting a file!")
        }
        path
    }))
}

fn main() {
    let src_dir = Path::new(BRIDGE_DIR);
    let cmakelists = glob_files_helper(src_dir.join("**/CMakeLists.txt"));
    let headers = glob_files_helper(src_dir.join("**/*.h"));
    let sources = glob_files_helper(src_dir.join("**/*.cpp"));
    mark_sources(cmakelists.chain(headers).chain(sources));
    build_bridge(BRIDGE_DIR);

    let header = src_dir.join(BRIDGE_HEADER);
    let bindings_output_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    generate_rust_bindings(header, bindings_output_path);
}

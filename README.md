# SP1-LLZK

This repository contains Rust modules for lowering SP1 circuits into LLZK and Picus. It includes two extractors:
- The `air -> llzk` extractor, located in the `air-llzk/` directory
- The direct `air -> picus` extractor, located in `air-picus/`

The `air -> picus` path is included as a temporary solution while LLZK undergoes updates to better support lookup constraints. We expect LLZK to have full support by early May, at which point the direct `air -> picus` extractor will no longer be necessary.

## Building

The prototype has been designed to build with nix and, while theoretically is possible to build and link the required libraries without it, only nix has been tested. Instructions for installing nix can be found [here](https://nixos.org/download/).

In the root of this repository run `nix develop` to enter the development environment. Inside, the rest of the workflow is the same as any other rust project (`cargo build`, `cargo test`, etc).

### Inspecting generated test output

The `air-llzk` crate has a series of tests that can dump the generated output to a file for inspection. The procedure for obtaining these outputs is as follows.

```sh 
mkdir circuit_outputs
# To obtain the paths of each generated file.
export RUST_LOG=info 
# Instructs the tests suite where to dump the files.
# It is necessary that is an absolute path.
export TEST_CIRCUIT_OUTPUTS_DIR=$(pwd)/circuit_outputs 
cargo test -- --nocapture
```

## Usage 

This module exposes a set of functions for lowering to LLZK (and supported LLZK backends).

The following example showcases the general workflow, assuming a type `ChipImpl` that defines a particular `Air`.   

```rust 
use air_llzk::{ChipInputs, codegen_llzk_eval, OutputFormats};
use sp1_stark::Chip;

// Optionally the implementation of the chip can implement this trait
// to automatically declare what number of columns are inputs.
impl ChipInputs for ChipImpl {
    fn inputs() -> usize {
        5 // For example's sake
    }
}

fn main() {
    // Assuming that ChipImpl implements Default
    let chip = ChipImpl::default();
    let chip = Chip::new(chip);

    let format = OutputFormats::Assembly; // To generate MLIR assembly
    let output = codegen_llzk_eval(&chip, format).unwrap();

    fs::write("output.mlir", output).expect("Failed to write output to file");
}
```

See the `air-llzk/examples` directory for more examples of how to use the crate. To run a particular example do `cargo run --example <example_name>`.

### Supported output formats 

| Format  | Description  |
|---|---|
| `Assembly`  | Outputs textual MLIR assembly. More information about this format can be found [here](https://mlir.llvm.org/docs/LangRef/)  |
|  `Bytecode` | Outputs MLIR IR in bytecode form. This is intended for feeding IR to other LLZK tools.  |
| `Picus` | Outputs a Picus program that can be used for verifying the circuit. |


## Workspace crates 

Brief list of the crates in this workspace.

| Name | Description |
|---|---|
| `air-llzk` | Public API for generating LLZK code from AIR chips. It can also generate Picus programs via LLZK. |
| `air-picus` | Public API for generating Picus programs from AIR chips. |
| `llzk-bridge` | Support module that implements an FFI layer for working with LLZK's C++ classes. |

use air_llzk::{air::CodegenBuilder, codegen_llzk_eval, field::Felt, ChipInputs, OutputFormats};
use log::{info, warn};
use p3_air::Air;
use p3_uni_stark::SymbolicAirBuilder;
use sp1_core_machine::utils::setup_logger;
use sp1_stark::air::MachineAir;
use sp1_stark::{Chip, InteractionBuilder};
use std::{env, fs, path::PathBuf};

#[inline]
fn ext(format: OutputFormats) -> &'static str {
    match format {
        OutputFormats::Assembly => "mlir",
        OutputFormats::Bytecode => "bc",
        OutputFormats::Picus => "picus",
    }
}

pub fn test_chip<C>(format: OutputFormats)
where
    C: Default
        + ChipInputs
        + MachineAir<Felt>
        + Air<InteractionBuilder<Felt>>
        + Air<SymbolicAirBuilder<Felt>>
        + for<'a> Air<CodegenBuilder<'a>>,
{
    let _ = env_logger::builder().is_test(true).try_init();
    let chip = C::default();
    let chip = Chip::new(chip);
    let output = codegen_llzk_eval(&chip, format.clone()).unwrap();
    if let Some(dump_dir) = env::var_os("TEST_CIRCUIT_OUTPUTS_DIR") {
        let chip_name = chip.name().to_lowercase();
        let path = PathBuf::from(dump_dir).join(format!("p3_{}.{}", chip_name, ext(format)));
        match fs::write(&path, output) {
            Ok(()) => info!("Wrote output for chip {chip_name} in {}", path.display()),
            Err(err) => {
                warn!("Failed to write output for chip {chip_name} in {}: {err}", path.display())
            }
        }
    }
}

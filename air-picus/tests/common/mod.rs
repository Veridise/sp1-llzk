use air_picus::{air::CodegenBuilder, codegen_picus_eval, field::Felt, ChipInputs};
use log::{info, warn};
use p3_air::Air;
use p3_uni_stark::SymbolicAirBuilder;
use sp1_stark::air::MachineAir;
use sp1_stark::{Chip, InteractionBuilder};
use std::{env, fs, path::PathBuf};


pub fn test_chip<C>()
where
    C: Default
        + ChipInputs
        + MachineAir<Felt>
        + Air<InteractionBuilder<Felt>>
        + Air<SymbolicAirBuilder<Felt>>
        + for<'a> Air<CodegenBuilder<'a>>,
{
    // let _ = env_logger::builder().is_test(true).try_init();
    // assert!(false);
    let chip = C::default();
    let chip = Chip::new(chip);
    let output = codegen_picus_eval(&chip);
    if let Some(dump_dir) = env::var_os("TEST_CIRCUIT_OUTPUTS_DIR") {
        let chip_name = chip.name().to_lowercase();
        let path = PathBuf::from(dump_dir).join(format!("p3_{}.picus", chip_name));
        match fs::write(&path, output) {
            Ok(()) => info!("Wrote output for chip {chip_name} in {}", path.display()),
            Err(err) => {
                warn!("Failed to write output for chip {chip_name} in {}: {err}", path.display())
            }
        }
    }
}

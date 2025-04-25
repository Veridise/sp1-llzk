pub mod air;
mod picus_codegen;
pub mod field;
mod helpers;
mod value;
mod vars;

use picus_codegen::PicusExtractor;
use field::Felt;
use lazy_static::lazy_static;
use p3_air::Air;
use sp1_stark::air::MachineAir;
use sp1_stark::Chip;
use std::sync::Mutex;

/// Convenience trait for chip implementors that declares how many columns of the chip's width are
/// considered inputs. Assumes that the columns to the left are inputs and the columns to the right
/// are outputs.
pub trait ChipInputs {
    /// Returns how many columns of the chip's width are considered inputs.
    fn inputs() -> usize;
}

lazy_static! {
    pub static ref PICUS_EXTRACTOR: Mutex<PicusExtractor> = Mutex::new(PicusExtractor::new());
}

/// Generate Picus code from a chip.
pub fn codegen_picus_eval_with_inputs<A>(
    chip: &Chip<Felt, A>,
    n_inputs: usize,
) -> String
where
    A: for<'a> Air<air::CodegenBuilder<'a>> + MachineAir<Felt>,
{

    let vars = air::CodegenChipVars::from_chip::<A>(chip, n_inputs);
    let mut builder = air::CodegenBuilder::new(&vars);
    chip.eval(&mut builder);
    let mut pe = PICUS_EXTRACTOR.lock().unwrap();
    let res = pe.serialize();
    pe.reset();
    drop(pe);
    res
}

/// Generate Picus code from a chip.
pub fn codegen_picus_eval<A>(
    chip: &Chip<Felt, A>,
) -> String
where
    A: for<'a> Air<air::CodegenBuilder<'a>> + MachineAir<Felt> + ChipInputs,
{
    codegen_picus_eval_with_inputs::<A>(chip, A::inputs())
}

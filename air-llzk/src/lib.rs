pub mod air;
mod codegen;
pub mod field;
mod helpers;
mod output;
mod value;
mod vars;

use codegen::Codegen;
pub use codegen::{CodegenError, CodegenResult};
use field::Felt;
use lazy_static::lazy_static;
pub use output::{CodegenOutput, OutputFormats};
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
    pub static ref LLZK_CODEGEN_LOCK: Mutex<()> = Mutex::new(());
}

/// Generate code from a chip using LLZK.
pub fn codegen_llzk_eval_with_inputs<A>(
    chip: &Chip<Felt, A>,
    n_inputs: usize,
    format: OutputFormats,
) -> CodegenResult<CodegenOutput>
where
    A: for<'a> Air<air::CodegenBuilder<'a>> + MachineAir<Felt>,
{
    let lock = LLZK_CODEGEN_LOCK.lock().unwrap();
    let binding = Codegen::instance()?;
    let codegen = binding.initialize(&chip, n_inputs)?;

    let vars = air::CodegenChipVars::from_chip::<A>(chip, n_inputs, &codegen);
    let mut builder = air::CodegenBuilder::new(&vars);
    chip.eval(&mut builder);
    let output = codegen.extract_output(format.clone());
    codegen.reset();
    drop(lock);
    output
}

/// Generate code from a chip using LLZK.
pub fn codegen_llzk_eval<A>(
    chip: &Chip<Felt, A>,
    format: OutputFormats,
) -> CodegenResult<CodegenOutput>
where
    A: for<'a> Air<air::CodegenBuilder<'a>> + MachineAir<Felt> + ChipInputs,
{
    codegen_llzk_eval_with_inputs::<A>(chip, A::inputs(), format)
}

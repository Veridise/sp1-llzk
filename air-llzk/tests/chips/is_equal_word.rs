use air_llzk::ChipInputs;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use sp1_core_executor::ExecutionRecord;
use sp1_core_executor::Program;
use sp1_core_machine::operations::IsEqualWordOperation as Operation;
use sp1_derive::AlignedBorrow;
use sp1_stark::air::MachineAir;
use sp1_stark::{air::SP1AirBuilder, Word};
use std::borrow::Borrow;

#[derive(Default, Clone, Copy)]
#[repr(C)]
struct Ins<T> {
    a: Word<T>,
    b: Word<T>,
}

#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
struct Cols<T> {
    ins: Ins<T>,
    op: Operation<T>,
}

#[derive(Default)]
pub struct Chip;

impl<AB> Air<AB> for Chip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Cols<AB::Var> = (*local).borrow();
        Operation::<AB::F>::eval(
            builder,
            local.ins.a.map(|part| part.into()),
            local.ins.b.map(|part| part.into()),
            local.op,
            AB::Expr::one(),
        );
    }
}

impl<F: PrimeField32> MachineAir<F> for Chip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        std::path::Path::new(file!()).file_stem().unwrap().to_str().unwrap().to_string()
    }

    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        todo!()
    }

    fn generate_trace(
        &self,
        _input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        todo!()
    }

    fn included(&self, _shard: &Self::Record) -> bool {
        todo!()
    }

    fn local_only(&self) -> bool {
        todo!()
    }
}

impl ChipInputs for Chip {
    fn inputs() -> usize {
        size_of::<Ins<u8>>()
    }
}

impl<F> BaseAir<F> for Chip {
    fn width(&self) -> usize {
        size_of::<Cols<u8>>()
    }
}

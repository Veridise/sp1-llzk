use air_llzk::ChipInputs;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use sp1_core_executor::ExecutionRecord;
use sp1_core_executor::Program;
use sp1_core_machine::operations::AddOperation;
use sp1_derive::AlignedBorrow;
use sp1_stark::air::MachineAir;
use sp1_stark::{air::SP1AirBuilder, Word};
use std::borrow::Borrow;

#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
struct AddCols<T> {
    a: Word<T>,
    b: Word<T>,
    op: AddOperation<T>,
}

#[derive(Default)]
pub struct AddChip;

impl<F: PrimeField32> MachineAir<F> for AddChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Add".to_string()
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

impl ChipInputs for AddChip {
    fn inputs() -> usize {
        size_of::<AddCols<u8>>() - size_of::<AddOperation<u8>>()
    }
}

impl<F> BaseAir<F> for AddChip {
    fn width(&self) -> usize {
        size_of::<AddCols<u8>>()
    }
}

impl<AB> Air<AB> for AddChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AddCols<AB::Var> = (*local).borrow();
        AddOperation::<AB::F>::eval(builder, local.a, local.b, local.op, AB::Expr::one());
    }
}

use air_llzk::ChipInputs;
use p3_air::{Air, BaseAir};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use sp1_core_executor::ExecutionRecord;
use sp1_core_executor::Program;
use sp1_core_machine::operations::Add5Operation;
use sp1_derive::AlignedBorrow;
use sp1_stark::air::MachineAir;
use sp1_stark::{air::SP1AirBuilder, Word};
use std::borrow::Borrow;

#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
struct Add5Cols<T> {
    a: Word<T>,
    b: Word<T>,
    c: Word<T>,
    d: Word<T>,
    e: Word<T>,
    is_real: T,
    op: Add5Operation<T>,
}

#[derive(Default)]
pub struct Add5Chip;

impl<F: PrimeField32> MachineAir<F> for Add5Chip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Add5".to_string()
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

impl ChipInputs for Add5Chip {
    fn inputs() -> usize {
        size_of::<Add5Cols<u8>>() - size_of::<Add5Operation<u8>>()
    }
}

impl<F> BaseAir<F> for Add5Chip {
    fn width(&self) -> usize {
        size_of::<Add5Cols<u8>>()
    }
}

impl<AB> Air<AB> for Add5Chip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Add5Cols<AB::Var> = (*local).borrow();
        let words: [Word<AB::Var>; 5] = [local.a, local.b, local.c, local.d, local.e];
        // constraining `is_real`` to one because it is defined as a variable and generating a constant
        // var is not super easy
        builder.assert_one(local.is_real);
        Add5Operation::<AB::F>::eval(
            builder,
            &words,
            local.is_real,
            local.op,
        );
    }
}
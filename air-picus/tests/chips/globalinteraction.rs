use air_picus::ChipInputs;
use p3_air::PairBuilder;
use p3_air::{Air, BaseAir};
use p3_field::PrimeField32;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use sp1_core_executor::ExecutionRecord;
use sp1_core_executor::Program;
use sp1_core_machine::operations::GlobalInteractionOperation as Operation;
use sp1_derive::AlignedBorrow;
use sp1_stark::air::MachineAir;
use sp1_stark::air::SP1AirBuilder;
use std::borrow::Borrow;

#[derive(Default, Clone, Copy)]
#[repr(C)]
struct Ins<T> {
    values: [T; 7],
    is_real: T,
    kind: T,
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
struct Cols<T: std::marker::Copy> {
    ins: Ins<T>,
    op: Operation<T>,
}

#[derive(Default)]
pub struct Chip;

impl<AB> Air<AB> for Chip
where
    AB: SP1AirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Cols<AB::Var> = (*local).borrow();
        // constraining `is_real`` to one because it is defined as a variable and generating a constant
        // var is not super easy
        let values_expr: [AB::Expr; 7] = [local.ins.values[0].into(), local.ins.values[1].into(), local.ins.values[2].into(), local.ins.values[3].into(), local.ins.values[4].into(), local.ins.values[5].into(), local.ins.values[6].into()];
        builder.assert_one(local.ins.is_real);
        Operation::<AB::F>::eval_single_digest(builder, values_expr, local.op, AB::Expr::one(), AB::Expr::one(), local.ins.is_real, local.ins.kind);
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

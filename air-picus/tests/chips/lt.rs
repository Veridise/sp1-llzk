use air_picus::ChipInputs;
use p3_air::{Air, BaseAir};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use sp1_core_executor::ExecutionRecord;
use sp1_core_executor::Program;
use sp1_core_machine::alu::LtChip;
use sp1_derive::AlignedBorrow;
use sp1_stark::air::MachineAir;
use sp1_stark::{air::SP1AirBuilder, Word};

#[derive(Default, Clone, Copy)]
#[repr(C)]
struct Ins<T> {
    /// The program counter.
    pub pc: T,

    /// If the opcode is SLT.
    pub is_slt: T,

    /// If the opcode is SLTU.
    pub is_sltu: T,

    /// The output operand.
    pub a: T,

    /// The first input operand.
    pub b: Word<T>,

    /// The second input operand.
    pub c: Word<T>,

    /// Whether the first operand is not register 0.
    pub op_a_not_0: T,

    /// Boolean flag to indicate which byte pair differs if the operands are not equal.
    pub byte_flags: [T; 4],

    /// The masking b\[3\] & 0x7F.
    pub b_masked: T,
    /// The masking c\[3\] & 0x7F.
    pub c_masked: T,
    /// An inverse of differing byte if c_comp != b_comp.
    pub not_eq_inv: T,

    /// The most significant bit of operand b.
    pub msb_b: T,
    /// The most significant bit of operand c.
    pub msb_c: T,
    /// The multiplication msb_b * is_slt.
    pub bit_b: T,
    /// The multiplication msb_c * is_slt.
    pub bit_c: T,

}

#[derive(Default, Clone, Copy)]
#[repr(C)]
struct Outs<T> {
    /// The result of the intermediate SLTU operation `b_comp < c_comp`.
    pub sltu: T,
    /// A bollean flag for an intermediate comparison.
    pub is_comp_eq: T,
    /// A boolean flag for comparing the sign bits.
    pub is_sign_eq: T,
    /// The comparison bytes to be looked up.
    pub comparison_bytes: [T; 2],
    /// Boolean fags to indicate which byte differs between the perands `b_comp`, `c_comp`.
    pub byte_equality_check: [T; 4],
}

#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
struct Cols<T> {
    ins: Ins<T>,
    outs: Outs<T>,
}

#[derive(Default)]
pub struct Chip;

impl<AB> Air<AB> for Chip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let ltchip = LtChip;
        ltchip.eval(builder);
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

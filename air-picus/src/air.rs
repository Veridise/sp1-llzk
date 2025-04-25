use crate::picus_codegen::{Args, PicusConstraint, PicusVar};
use crate::field::{ExtFelt, Felt};
use crate::helpers::{air_felt_values_from_args, air_values, felt_arg, init_vars, main_vars};
use crate::picus_codegen::PicusExpr;
use crate::value::ext_felt::ExtFeltValue;
use crate::vars::ExtFeltVar;
use crate::PICUS_EXTRACTOR;
use crate::picus_codegen::SignalType;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};
use rug::Integer;
use sp1_core_executor::ByteOpcode;
use sp1_stark::PROOF_MAX_NUM_PVS;
use sp1_stark::{
    air::{AirInteraction, MachineAir, MessageBuilder, MultiTableAirBuilder},
    septic_curve::SepticCurve,
    septic_digest::SepticDigest,
    septic_extension::SepticExtension,
    AirOpenedValues, Chip,
};

/// Implements the interfaces used to communicate with the Air.
pub struct CodegenChipVars {
    preprocessed: AirOpenedValues<PicusVar>,
    main: AirOpenedValues<PicusVar>,
    perm: AirOpenedValues<ExtFeltVar>,
    pub perm_challenges: Vec<ExtFeltVar>,
    pub local_cumulative_sum: ExtFeltVar,
    pub global_cumulative_sum: SepticDigest<PicusVar>,
    pub is_first_row: PicusVar,
    pub is_last_row: PicusVar,
    pub is_transition: PicusVar,
    pub public_values: Vec<PicusVar>,
}

const PERM_CHALLENGES_COUNT: usize = 2;

impl<'a> CodegenChipVars {
    /// Initializes the variables used for code generation given a chip and the number of values in
    /// the chip's width that are considered inputs.
    pub fn from_chip<A>(
        chip: &Chip<Felt, A>,
        n_inputs: usize,
    ) -> Self
    where
        A: Air<CodegenBuilder<'a>> + MachineAir<Felt>,
    {
        let name = chip.name();
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.init_cur_module(&name);
        let first_row_var = pe.fresh_input_picus_var();
        let last_row_var = pe.fresh_input_picus_var();
        let transition_row_var = pe.fresh_input_picus_var();
        drop(pe);
        let res = Self {
            preprocessed: air_felt_values_from_args(
                chip.preprocessed_width(),
                Args::Preprocessed,
                Args::PreprocessedNext
            ),
            main: air_values(
                chip.width(),
                main_vars(n_inputs),
                |_| PicusVar((0, SignalType::Ignore)),
            ),
            perm: AirOpenedValues {
                local: (0..chip.permutation_width()).map(|_| ExtFeltVar {}).collect::<Vec<_>>(),
                next: (0..chip.permutation_width()).map(|_| ExtFeltVar {}).collect::<Vec<_>>(),
            },
            perm_challenges: (0..PERM_CHALLENGES_COUNT).map(|_| ExtFeltVar {}).collect::<Vec<_>>(),
            local_cumulative_sum: ExtFeltVar {},
            global_cumulative_sum: SepticDigest(SepticCurve {
                x: SepticExtension(core::array::from_fn(|_| PicusVar::default())),
                y: SepticExtension(core::array::from_fn(|_| PicusVar::default())),
            }),
            public_values: init_vars(PROOF_MAX_NUM_PVS, felt_arg(Args::PublicValues)),
            is_first_row: first_row_var,
            is_last_row: last_row_var,
            is_transition: transition_row_var,
        };
        res
    }
}

pub struct CodegenBuilder<'a> {
    vars: &'a CodegenChipVars,
}

impl<'a> CodegenBuilder<'a> {
    pub fn new(vars: &'a CodegenChipVars) -> Self {
        Self { vars }
    }
}

type M<'a, Var> = VerticalPair<RowMajorMatrixView<'a, Var>, RowMajorMatrixView<'a, Var>>;

impl<'a> AirBuilder for CodegenBuilder<'a> {
    type F = Felt;
    type Var = PicusVar;
    type Expr = PicusExpr;
    type M = M<'a, Self::Var>;

    fn main(&self) -> Self::M {
        self.vars.main.view()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.vars.is_first_row.into()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.vars.is_last_row.into()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.vars.is_transition.into()
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let mut picusextractor = PICUS_EXTRACTOR.lock().unwrap();
        let picus_constraint = PicusConstraint::Eq(Box::new(x.into()));
        picusextractor.add_constraint(picus_constraint);
        drop(picusextractor);
    }
}

impl ExtensionBuilder for CodegenBuilder<'_> {
    type EF = ExtFelt;
    type ExprEF = ExtFeltValue;
    type VarEF = ExtFeltVar;

    fn assert_zero_ext<I>(&mut self, _x: I)
    where
        I: Into<Self::ExprEF>,
    {
    }
}

impl<'a> PermutationAirBuilder for CodegenBuilder<'a> {
    type MP = M<'a, Self::RandomVar>;
    type RandomVar = ExtFeltVar;

    fn permutation(&self) -> Self::MP {
        self.vars.perm.view()
    }
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.vars.perm_challenges
    }
}
impl<'a> MultiTableAirBuilder<'a> for CodegenBuilder<'a> {
    type LocalSum = ExtFeltVar;
    type GlobalSum = PicusVar;

    fn local_cumulative_sum(&self) -> &'a Self::LocalSum {
        &self.vars.local_cumulative_sum
    }

    fn global_cumulative_sum(&self) -> &'a SepticDigest<Self::GlobalSum> {
        &self.vars.global_cumulative_sum
    }
}

impl PairBuilder for CodegenBuilder<'_> {
    fn preprocessed(&self) -> Self::M {
        self.vars.preprocessed.view()
    }
}

impl<'a> AirBuilderWithPublicValues for CodegenBuilder<'a> {
    type PublicVar = PicusVar;

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.vars.public_values
    }
}

fn get_opcode(values: &[PicusExpr]) -> PicusExpr {
    values[0].clone()
}

// We encode all range checks as `less-than` constraints in Picus and all bitwise operations
// as module calls. A quirk here is that the module is empty which would typically mean
// it is underconstrained. As such, it is expected that the caller of Picus assumes the modules
// are deterministic.
fn handle_byte_interaction(values: &[PicusExpr]) {
    if values.is_empty() {
        return;
    }
    let mut pe = PICUS_EXTRACTOR.lock().unwrap();
    if let PicusExpr::Const(op) = get_opcode(values) {
        let op_field = Felt::from_canonical_u32(op.to_u32_wrapping());
        if op_field == ByteOpcode::U8Range.as_field() {
            for value in values.iter().skip(1) {
                if let PicusExpr::Const(_) = value {
                    continue
                }
                let range_constraint = PicusConstraint::Lt(Box::new(value.clone()), Box::new(PicusExpr::Const(Integer::from(256))));
                pe.add_constraint(range_constraint);
            }
        } else if op_field == ByteOpcode::U16Range.as_field() {
            for value in values.iter().skip(1) {
                if let PicusExpr::Const(_) = value {
                    continue
                }
                let range_constraint = PicusConstraint::Lt(Box::new(value.clone()), Box::new(PicusExpr::Const(Integer::from(256))));
                pe.add_constraint(range_constraint);
            }
        } else if op_field == ByteOpcode::AND.as_field() {
            pe.build_bitwiseop_module(ByteOpcode::AND);
            pe.build_bitwiseop_call(ByteOpcode::AND, values[3].clone(), values[4].clone(),values[1].clone());
        } else if op_field == ByteOpcode::ShrCarry.as_field() {
            pe.build_shift_carry_module();
            pe.build_shift_carry_call(values[3].clone(), values[2].clone(), values[1].clone());
        } else if op_field == ByteOpcode::XOR.as_field() {
            pe.build_bitwiseop_module(ByteOpcode::XOR);
            pe.build_bitwiseop_call(ByteOpcode::XOR, values[3].clone(), values[4].clone(),values[1].clone());
        } else if op_field == ByteOpcode::OR.as_field() {
            pe.build_bitwiseop_module(ByteOpcode::OR);
            pe.build_bitwiseop_call(ByteOpcode::OR, values[3].clone(), values[4].clone(),values[1].clone());
        }
    }
    drop(pe);
}

impl MessageBuilder<AirInteraction<PicusExpr>> for CodegenBuilder<'_> {
    fn send(
        &mut self,
        message: AirInteraction<PicusExpr>,
        _scope: sp1_stark::air::InteractionScope,
    ) {
        match message.kind {
            sp1_stark::InteractionKind::Byte => handle_byte_interaction(&message.values),
            _ => {}
        }
    }

    fn receive(
        &mut self,
        _message: AirInteraction<PicusExpr>,
        _scope: sp1_stark::air::InteractionScope,
    ) {
        unreachable!()
    }
}

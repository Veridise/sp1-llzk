use crate::codegen::{Args, Codegen, StructCodegen};
use crate::field::{ExtFelt, Felt};
use crate::helpers::{air_felt_values_from_args, air_values, felt_arg, init_vars, main_vars};
use crate::value::Value;
use crate::value::{ExtFeltValue, FeltValue};
use crate::vars::{ExtFeltVar, FeltVar};
use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};
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
    preprocessed: AirOpenedValues<FeltVar>,
    main: AirOpenedValues<FeltVar>,
    perm: AirOpenedValues<ExtFeltVar>,
    pub perm_challenges: Vec<ExtFeltVar>,
    pub local_cumulative_sum: ExtFeltVar,
    pub global_cumulative_sum: SepticDigest<FeltVar>,
    pub is_first_row: FeltVar,
    pub is_last_row: FeltVar,
    pub is_transition: FeltVar,
    pub public_values: Vec<FeltVar>,
}

const PERM_CHALLENGES_COUNT: usize = 2;

impl<'a> CodegenChipVars {
    /// Initializes the variables used for code generation given a chip and the number of values in
    /// the chip's width that are considered inputs.
    pub fn from_chip<A>(
        chip: &Chip<Felt, A>,
        n_inputs: usize,
        codegen: &'a StructCodegen<'a>,
    ) -> Self
    where
        A: Air<CodegenBuilder<'a>> + MachineAir<Felt>,
    {
        Self {
            preprocessed: air_felt_values_from_args(
                chip.preprocessed_width(),
                Args::Preprocessed,
                Args::PreprocessedNext,
                codegen,
            ),
            main: air_values(
                chip.width(),
                main_vars(codegen, Args::Inputs, n_inputs, "output"),
                |_| FeltVar::Ignore,
            ),
            perm: AirOpenedValues {
                local: (0..chip.permutation_width()).map(|_| ExtFeltVar {}).collect::<Vec<_>>(),
                next: (0..chip.permutation_width()).map(|_| ExtFeltVar {}).collect::<Vec<_>>(),
            },
            //perm: air_extfelt_values_from_args(
            //    chip.permutation_width(),
            //    Args::Permutations,
            //    Args::PermutationsNext,
            //    codegen,
            //),
            perm_challenges: (0..PERM_CHALLENGES_COUNT).map(|_| ExtFeltVar {}).collect::<Vec<_>>(),
            //perm_challenges: init_vars(
            //    PERM_CHALLENGES_COUNT,
            //    extfelt_arg(codegen, Args::PermutationChallenges),
            //),
            local_cumulative_sum: ExtFeltVar {},
            //local_cumulative_sum: ExtFeltVar::Arg {
            //    arg: codegen.get_func_arg(Args::LocalCumulativeSum).into(),
            //},
            global_cumulative_sum: SepticDigest(SepticCurve {
                x: SepticExtension(core::array::from_fn(|_| FeltVar::Arg {
                    arg: FeltValue::default(),
                })),
                y: SepticExtension(core::array::from_fn(|_| FeltVar::Arg {
                    arg: FeltValue::default(),
                })),
            }),
            //global_cumulative_sum: SepticDigest(SepticCurve {
            //    x: SepticExtension(core::array::from_fn(felt_arg(
            //        codegen,
            //        Args::GlobalCumulativeSum,
            //    ))),
            //    y: SepticExtension(core::array::from_fn(felt_arg_offset::<7>(
            //        codegen,
            //        Args::GlobalCumulativeSum,
            //    ))),
            //}),
            public_values: init_vars(PROOF_MAX_NUM_PVS, felt_arg(codegen, Args::PublicValues)),
            is_first_row: FeltVar::Arg {
                arg: codegen.get_func_arg(Args::IsFirstRow).unwrap().into(),
            },
            is_last_row: FeltVar::Arg {
                arg: codegen.get_func_arg(Args::IsLastRow).unwrap().into(),
            },
            is_transition: FeltVar::Arg {
                arg: codegen.get_func_arg(Args::IsTransition).unwrap().into(),
            },
        }
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
    type Var = FeltVar;
    type Expr = FeltValue;
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
        let x: Self::Expr = x.into();
        let zero: Self::Expr = Felt::zero().into();
        let codegen = Codegen::instance().unwrap();
        codegen.emit_eq(x.into(), zero.into()).unwrap();
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
    type GlobalSum = FeltVar;

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
    type PublicVar = FeltVar;

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.vars.public_values
    }
}

fn select_range(opcode: Felt) -> Option<Value> {
    let codegen = Codegen::instance().unwrap();
    let u8opc: Felt = Felt::from_canonical_u8(ByteOpcode::U8Range as u8);
    let u16opc: Felt = Felt::from_canonical_u8(ByteOpcode::U16Range as u8);
    if opcode == u8opc {
        Some(codegen.get_8bit_range().unwrap())
    } else if opcode == u16opc {
        Some(codegen.get_16bit_range().unwrap())
    } else {
        None
    }
}

fn is_not_zero(value: &&FeltValue) -> bool {
    let codegen = Codegen::instance().unwrap();
    codegen.get_const_felt_from_value((**value).into()) != Some(Felt::zero())
}

fn get_opcode(values: &[FeltValue]) -> Option<Felt> {
    let codegen = Codegen::instance().unwrap();
    codegen.get_const_felt_from_value(values[0].into())
}

fn handle_byte_interaction(values: &[FeltValue]) {
    if values.is_empty() {
        return;
    }

    let codegen = Codegen::instance().unwrap();
    if let Some(opcode) = get_opcode(values) {
        if let Some(range) = select_range(opcode) {
            for value in values.iter().skip(1).filter(is_not_zero) {
                codegen.emit_in((*value).into(), range).unwrap();
            }
        }
    }
}

impl MessageBuilder<AirInteraction<FeltValue>> for CodegenBuilder<'_> {
    fn send(
        &mut self,
        message: AirInteraction<FeltValue>,
        _scope: sp1_stark::air::InteractionScope,
    ) {
        match message.kind {
            sp1_stark::InteractionKind::Byte => handle_byte_interaction(&message.values),
            _ => {}
        }
    }

    fn receive(
        &mut self,
        _message: AirInteraction<FeltValue>,
        _scope: sp1_stark::air::InteractionScope,
    ) {
        unreachable!()
    }
}

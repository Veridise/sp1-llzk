mod babybear;
mod codegen;
mod helpers;
mod value;
mod vars;

use core::fmt;

use babybear as field;
pub use codegen::Codegen;
use codegen::CodegenWithStruct;
use helpers::air_extfelt_values_from_args;
use helpers::air_felt_values_from_args;
use helpers::air_values;
use helpers::extfelt_arg;
use helpers::felt_arg;
use helpers::felt_arg_offset;
use helpers::init_vars;
use helpers::main_vars;
use p3_air::Air;
use p3_air::AirBuilder;
use p3_air::AirBuilderWithPublicValues;
use p3_air::BaseAir;
use p3_air::ExtensionBuilder;
use p3_air::PairBuilder;
use p3_air::PermutationAirBuilder;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use sp1_core_executor::ByteOpcode;
use sp1_core_machine::air::WordAirBuilder;
use sp1_stark::air::AirInteraction;
use sp1_stark::air::BaseAirBuilder;
use sp1_stark::air::ByteAirBuilder;
use sp1_stark::air::MachineAir;
use sp1_stark::air::MessageBuilder;
use sp1_stark::air::MultiTableAirBuilder;
use sp1_stark::septic_curve::SepticCurve;
use sp1_stark::septic_digest::SepticDigest;
use sp1_stark::septic_extension::SepticExtension;
use sp1_stark::AirOpenedValues;
use sp1_stark::Chip;
use sp1_stark::PROOF_MAX_NUM_PVS;
use value::ExtFeltValue;
use value::FeltValue;
use vars::ExtFeltVar;
use vars::FeltVar;

pub type Felt = field::Felt;
pub type ExtFelt = field::ExtFelt;
const EXT_FELT_DEGREE: usize = field::EXT_FELT_DEGREE;
const FIELD_BETA: usize = field::FIELD_BETA;

/// Opaque type that represents a IR type in llzk.
pub type Type = llzk_bridge::ValueType;

pub enum OutputFormats {
    Assembly,
    Bytecode,
}

/// Final output type of the llzk code generator.
pub struct CodegenOutput {
    bytes: *mut u8,
    size: usize,
    format: OutputFormats,
}

impl Drop for CodegenOutput {
    fn drop(&mut self) {
        let codegen = Codegen::instance();
        codegen.release_output(self);
    }
}

impl fmt::Display for CodegenOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let str = std::str::from_utf8(unsafe { std::slice::from_raw_parts(self.bytes, self.size) })
            .unwrap();
        write!(f, "{}", str)
    }
}

/// A reference to a name in MLIR.
pub type Symbol = llzk_bridge::Symbol;

/// The order of the arguments in the constraint function
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Args {
    SelfArg = 0,
    Inputs,
    InputsNext,
    Preprocessed,
    PreprocessedNext,
    Permutations,
    PermutationsNext,
    PublicValues,
    PermutationChallenges,
    GlobalCumulativeSum,
    LocalCumulativeSum,
    IsFirstRow,
    IsLastRow,
    IsTransition,
}

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

pub(crate) const PERM_CHALLENGES_COUNT: usize = 2;

impl<'a> CodegenChipVars {
    /// Initializes the variables used for code generation given a chip and the number of values in
    /// the chip's width that are considered inputs.
    pub fn from_chip<A>(
        chip: &Chip<Felt, A>,
        n_inputs: usize,
        codegen: &'a CodegenWithStruct<'a>,
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
                main_vars(codegen, Args::InputsNext, n_inputs, "output_next"),
            ),
            perm: air_extfelt_values_from_args(
                chip.permutation_width(),
                Args::Permutations,
                Args::PermutationsNext,
                codegen,
            ),
            perm_challenges: init_vars(
                PERM_CHALLENGES_COUNT,
                extfelt_arg(codegen, Args::PermutationChallenges),
            ),
            local_cumulative_sum: ExtFeltVar::Arg {
                arg: codegen.get_func_arg(Args::LocalCumulativeSum).into(),
            },
            global_cumulative_sum: SepticDigest(SepticCurve {
                x: SepticExtension(core::array::from_fn(felt_arg(
                    codegen,
                    Args::GlobalCumulativeSum,
                ))),
                y: SepticExtension(core::array::from_fn(felt_arg_offset::<7>(
                    codegen,
                    Args::GlobalCumulativeSum,
                ))),
            }),
            public_values: init_vars(PROOF_MAX_NUM_PVS, felt_arg(codegen, Args::PublicValues)),
            is_first_row: FeltVar::Arg { arg: codegen.get_func_arg(Args::IsFirstRow).into() },
            is_last_row: FeltVar::Arg { arg: codegen.get_func_arg(Args::IsLastRow).into() },
            is_transition: FeltVar::Arg { arg: codegen.get_func_arg(Args::IsTransition).into() },
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
        let codegen = Codegen::instance();
        codegen.emit_eq(x.into(), zero.into());
    }
}

impl ExtensionBuilder for CodegenBuilder<'_> {
    type EF = ExtFelt;
    type ExprEF = ExtFeltValue;
    type VarEF = ExtFeltVar;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: ExtFeltValue = x.into();
        let zero: ExtFeltValue = ExtFelt::zero().into();
        let codegen = Codegen::instance();
        codegen.emit_eq(x.into(), zero.into());
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

impl MessageBuilder<AirInteraction<FeltValue>> for CodegenBuilder<'_> {
    fn send(
        &mut self,
        message: AirInteraction<FeltValue>,
        scope: sp1_stark::air::InteractionScope,
    ) {
        match message.kind {
            sp1_stark::InteractionKind::Byte => {
                if (message.values.len() < 4) {
                    panic!("Expected to have at least 4 inputs");
                }

                let codegen = Codegen::instance();
                if let Some(opcode) = codegen.get_const_felt_from_value(message.values[0].into()) {
                    let u8opc: Felt = Felt::from_canonical_u8(ByteOpcode::U8Range as u8);
                    let u16opc: Felt = Felt::from_canonical_u8(ByteOpcode::U16Range as u8);
                    if let Some(range) = match opcode {
                        u8opc => Some(codegen.get_8bit_range()),
                        u16opc => Some(codegen.get_16bit_range()),
                        _ => None,
                    } {
                        let value1 = message.values[2];
                        codegen.emit_in(value1.into(), range);
                        let value2 = message.values[3];
                        codegen.emit_in(value2.into(), range);
                    }
                }
            }
            _ => {}
        }
    }

    fn receive(
        &mut self,
        message: AirInteraction<FeltValue>,
        scope: sp1_stark::air::InteractionScope,
    ) {
        unreachable!()
    }
}

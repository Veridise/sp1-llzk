//! The specific selection of field is kept isolated from the rest of the codebase on its own
//! module to allow for easily adding feature guards in the future if other finite fields are going
//! to be used.

use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

pub type Felt = BabyBear;
pub const EXT_FELT_DEGREE: usize = 4;
pub const FIELD_BETA: usize = 11;
pub type ExtFelt = BinomialExtensionField<Felt, EXT_FELT_DEGREE>;

use crate::{
    codegen::Codegen,
    field::Felt,
    value::{FeltValue, Value},
};
use llzk_bridge::Symbol;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy)]
pub enum FeltVar {
    /// Field vars are used to encode the circuit outputs
    Field { name: Symbol },
    /// Array argument are used to encode the circuit inputs packed in arrays
    ArrayArg { arg: Value, idx: usize },
    /// Scalar argument used to encode a circuit input
    Arg { arg: FeltValue },
    /// Marks a circuit variable as ignored to avoid generating LLZK IR for it.
    Ignore,
}

impl From<FeltVar> for FeltValue {
    fn from(val: FeltVar) -> Self {
        let codegen = Codegen::instance().unwrap();
        match val {
            FeltVar::Field { name } => {
                codegen.read_self_field(name, codegen.get_felt_type()).try_into().unwrap()
            }
            FeltVar::ArrayArg { arg, idx } => {
                let index = codegen.const_index(idx).unwrap();
                codegen.read_array(arg, index).try_into().unwrap()
            }
            FeltVar::Arg { arg } => arg,
            FeltVar::Ignore => unreachable!(),
        }
    }
}

impl Add<Felt> for FeltVar {
    type Output = FeltValue;

    fn add(self, rhs: Felt) -> Self::Output {
        let lhs: FeltValue = self.into();
        lhs + rhs
    }
}

impl Add<FeltVar> for FeltVar {
    type Output = FeltValue;

    fn add(self, rhs: FeltVar) -> Self::Output {
        let lhs: FeltValue = self.into();
        lhs + rhs
    }
}

impl Add<FeltValue> for FeltVar {
    type Output = FeltValue;

    fn add(self, rhs: FeltValue) -> Self::Output {
        let lhs: FeltValue = self.into();
        lhs + rhs
    }
}

impl Sub<Felt> for FeltVar {
    type Output = FeltValue;

    fn sub(self, rhs: Felt) -> Self::Output {
        let lhs: FeltValue = self.into();
        lhs - rhs
    }
}

impl Sub<FeltVar> for FeltVar {
    type Output = FeltValue;

    fn sub(self, rhs: FeltVar) -> Self::Output {
        let lhs: FeltValue = self.into();
        lhs - rhs
    }
}

impl Sub<FeltValue> for FeltVar {
    type Output = FeltValue;

    fn sub(self, rhs: FeltValue) -> Self::Output {
        let lhs: FeltValue = self.into();
        lhs - rhs
    }
}

impl Mul<Felt> for FeltVar {
    type Output = FeltValue;

    fn mul(self, rhs: Felt) -> Self::Output {
        let lhs: FeltValue = self.into();
        lhs * rhs
    }
}

impl Mul<FeltVar> for FeltVar {
    type Output = FeltValue;

    fn mul(self, rhs: FeltVar) -> Self::Output {
        let lhs: FeltValue = self.into();
        lhs * rhs
    }
}

impl Mul<FeltValue> for FeltVar {
    type Output = FeltValue;

    fn mul(self, rhs: FeltValue) -> Self::Output {
        let lhs: FeltValue = self.into();
        lhs * rhs
    }
}

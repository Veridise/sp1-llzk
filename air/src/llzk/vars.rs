use std::ops::{Add, Mul, Sub};

use super::{
    value::{ExtFeltValue, FeltValue, Value},
    Codegen, ExtFelt, Felt, Symbol,
};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
pub enum FeltVar {
    /// Field vars are used to encode the circuit outputs
    Field { name: Symbol },
    /// Array argument are used to encode the circuit inputs packed in arrays
    ArrayArg { arg: Value, idx: usize },
    /// Scalar argument used to encode a circuit input
    Arg { arg: FeltValue },
    /// Mark a circuit variable as ignored to avoid generating LLZK IR for it.
    Ignore,
}

impl From<FeltVar> for FeltValue {
    fn from(val: FeltVar) -> Self {
        let codegen = Codegen::instance();
        match val {
            FeltVar::Field { name } => {
                codegen.read_self_field(name, codegen.get_felt_type()).into()
            }
            FeltVar::ArrayArg { arg, idx } => {
                let index = codegen.const_index(idx);
                codegen.read_array(arg, index).into()
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

#[derive(Clone, Copy)]
pub struct ExtFeltVar {}
/// Array input of extended field elements.
//ArrayArg { arg: Value, idx: usize },
/// Scalar input of extended field element type.
//Arg { arg: ExtFeltValue },
//}

impl From<ExtFeltVar> for ExtFeltValue {
    fn from(value: ExtFeltVar) -> Self {
        Self {}
        //let codegen = Codegen::instance();
        //match value {
        //    ExtFeltVar::ArrayArg { arg, idx } => {
        //        let index = codegen.const_index(idx);
        //        codegen.read_array(arg, index).into()
        //    }
        //    ExtFeltVar::Arg { arg } => arg,
        //}
    }
}

impl Add<ExtFelt> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn add(self, rhs: ExtFelt) -> Self::Output {
        let lhs: ExtFeltValue = self.into();
        lhs + rhs
    }
}

impl Add<ExtFeltVar> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn add(self, rhs: ExtFeltVar) -> Self::Output {
        let lhs: ExtFeltValue = self.into();
        lhs + rhs
    }
}

impl Add<ExtFeltValue> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn add(self, rhs: ExtFeltValue) -> Self::Output {
        let lhs: ExtFeltValue = self.into();
        lhs + rhs
    }
}

impl Sub<ExtFelt> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn sub(self, rhs: ExtFelt) -> Self::Output {
        let lhs: ExtFeltValue = self.into();
        lhs - rhs
    }
}

impl Sub<ExtFeltVar> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn sub(self, rhs: ExtFeltVar) -> Self::Output {
        let lhs: ExtFeltValue = self.into();
        lhs - rhs
    }
}

impl Sub<ExtFeltValue> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn sub(self, rhs: ExtFeltValue) -> Self::Output {
        let lhs: ExtFeltValue = self.into();
        lhs - rhs
    }
}

impl Mul<ExtFelt> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn mul(self, rhs: ExtFelt) -> Self::Output {
        let lhs: ExtFeltValue = self.into();
        lhs * rhs
    }
}

impl Mul<ExtFeltVar> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn mul(self, rhs: ExtFeltVar) -> Self::Output {
        let lhs: ExtFeltValue = self.into();
        lhs * rhs
    }
}

impl Mul<ExtFeltValue> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn mul(self, rhs: ExtFeltValue) -> Self::Output {
        let lhs: ExtFeltValue = self.into();
        lhs * rhs
    }
}

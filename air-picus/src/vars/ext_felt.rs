use crate::{field::ExtFelt, value::ext_felt::ExtFeltValue};
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Default)]
pub struct ExtFeltVar {}

impl From<ExtFeltVar> for ExtFeltValue {
    fn from(_value: ExtFeltVar) -> Self {
        Default::default()
    }
}

impl Add<ExtFelt> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn add(self, _rhs: ExtFelt) -> Self::Output {
        Default::default()
    }
}

impl Add<ExtFeltVar> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn add(self, _rhs: ExtFeltVar) -> Self::Output {
        Default::default()
    }
}

impl Add<ExtFeltValue> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn add(self, _rhs: ExtFeltValue) -> Self::Output {
        Default::default()
    }
}

impl Sub<ExtFelt> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn sub(self, _rhs: ExtFelt) -> Self::Output {
        Default::default()
    }
}

impl Sub<ExtFeltVar> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn sub(self, _rhs: ExtFeltVar) -> Self::Output {
        Default::default()
    }
}

impl Sub<ExtFeltValue> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn sub(self, _rhs: ExtFeltValue) -> Self::Output {
        Default::default()
    }
}

impl Mul<ExtFelt> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn mul(self, _rhs: ExtFelt) -> Self::Output {
        Default::default()
    }
}

impl Mul<ExtFeltVar> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn mul(self, _rhs: ExtFeltVar) -> Self::Output {
        Default::default()
    }
}

impl Mul<ExtFeltValue> for ExtFeltVar {
    type Output = ExtFeltValue;

    fn mul(self, _rhs: ExtFeltValue) -> Self::Output {
        Default::default()
    }
}

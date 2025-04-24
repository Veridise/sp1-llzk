use crate::{
    codegen::{Codegen, CodegenError, CodegenResult},
    field::Felt,
    vars::FeltVar,
};
use llzk_bridge::Value;
use p3_field::AbstractField;
use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Clone, Copy, Debug)]
pub struct FeltValue {
    pub(crate) inner: Value,
}

impl Default for FeltValue {
    fn default() -> Self {
        <Self as AbstractField>::zero()
    }
}

impl AbstractField for FeltValue {
    type F = Felt;

    fn zero() -> Self {
        Self::F::zero().into()
    }

    fn one() -> Self {
        Self::F::one().into()
    }

    fn two() -> Self {
        Self::F::two().into()
    }

    fn neg_one() -> Self {
        Self::F::neg_one().into()
    }

    fn from_f(f: Self::F) -> Self {
        f.into()
    }

    fn from_bool(b: bool) -> Self {
        Self::F::from_bool(b).into()
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::F::from_canonical_u8(n).into()
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::F::from_canonical_u16(n).into()
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::F::from_canonical_u32(n).into()
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::F::from_canonical_u64(n).into()
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::F::from_canonical_usize(n).into()
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::F::from_wrapped_u32(n).into()
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::F::from_wrapped_u64(n).into()
    }

    fn generator() -> Self {
        Self::F::generator().into()
    }
}

impl From<Felt> for FeltValue {
    fn from(value: Felt) -> Self {
        let codegen = Codegen::instance().unwrap();
        codegen.const_felt(value).try_into().unwrap()
    }
}

impl From<Value> for FeltValue {
    fn from(value: Value) -> Self {
        Self { inner: value }
    }
}

impl TryFrom<CodegenResult<Value>> for FeltValue {
    type Error = CodegenError;

    fn try_from(value: CodegenResult<Value>) -> Result<Self, Self::Error> {
        match value {
            Ok(value) => Ok(value.into()),
            Err(err) => Err(err),
        }
    }
}

impl Add<FeltValue> for FeltValue {
    type Output = FeltValue;

    fn add(self, rhs: FeltValue) -> Self::Output {
        let codegen = Codegen::instance().unwrap();
        codegen.felt_add(self, rhs).unwrap()
    }
}

impl Add<Felt> for FeltValue {
    type Output = FeltValue;

    fn add(self, rhs: Felt) -> Self::Output {
        self + Into::<FeltValue>::into(rhs)
    }
}

impl<'a> Add<FeltVar> for FeltValue {
    type Output = FeltValue;

    fn add(self, rhs: FeltVar) -> Self::Output {
        self + Into::<FeltValue>::into(rhs)
    }
}

impl Sub<FeltValue> for FeltValue {
    type Output = FeltValue;

    fn sub(self, rhs: FeltValue) -> Self::Output {
        let codegen = Codegen::instance().unwrap();
        codegen.felt_sub(self, rhs).unwrap()
    }
}

impl Sub<Felt> for FeltValue {
    type Output = FeltValue;

    fn sub(self, rhs: Felt) -> Self::Output {
        self - Into::<FeltValue>::into(rhs)
    }
}

impl<'a> Sub<FeltVar> for FeltValue {
    type Output = FeltValue;

    fn sub(self, rhs: FeltVar) -> Self::Output {
        self - Into::<FeltValue>::into(rhs)
    }
}

impl Mul<FeltValue> for FeltValue {
    type Output = FeltValue;

    fn mul(self, rhs: FeltValue) -> Self::Output {
        let codegen = Codegen::instance().unwrap();
        codegen.felt_mul(self, rhs).unwrap()
    }
}

impl Mul<Felt> for FeltValue {
    type Output = FeltValue;

    fn mul(self, rhs: Felt) -> Self::Output {
        self * Into::<FeltValue>::into(rhs)
    }
}

impl<'a> Mul<FeltVar> for FeltValue {
    type Output = FeltValue;

    fn mul(self, rhs: FeltVar) -> Self::Output {
        self * Into::<FeltValue>::into(rhs)
    }
}

impl AddAssign for FeltValue {
    fn add_assign(&mut self, rhs: Self) {
        let new_value = *self + rhs;
        self.inner = new_value.into();
    }
}

impl SubAssign for FeltValue {
    fn sub_assign(&mut self, rhs: Self) {
        let new_value = *self - rhs;
        self.inner = new_value.into();
    }
}

impl MulAssign for FeltValue {
    fn mul_assign(&mut self, rhs: Self) {
        let new_value = *self * rhs;
        self.inner = new_value.into();
    }
}

impl Neg for FeltValue {
    type Output = FeltValue;

    fn neg(self) -> Self::Output {
        let codegen = Codegen::instance().unwrap();
        codegen.felt_neg(self).unwrap()
    }
}

impl Sum for FeltValue {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = Self::zero();
        for item in iter {
            output = output + item;
        }
        output
    }
}

impl Product for FeltValue {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = Self::one();
        for item in iter {
            output = output * item;
        }
        output
    }
}

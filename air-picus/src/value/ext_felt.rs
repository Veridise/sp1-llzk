use crate::{field::{ExtFelt, EXT_FELT_DEGREE}, picus_codegen::PicusExpr, vars::ExtFeltVar};
use p3_field::{AbstractExtensionField, AbstractField};
use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Clone, Debug, Copy, Default)]
pub struct ExtFeltValue {}

impl From<PicusExpr> for ExtFeltValue {
    fn from(_value: PicusExpr) -> Self {
        Default::default()
    }
}

impl AbstractField for ExtFeltValue {
    type F = ExtFelt;

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

impl From<ExtFelt> for ExtFeltValue {
    fn from(_value: ExtFelt) -> Self {
        Default::default()
    }
}

impl Add<ExtFeltValue> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn add(self, _rhs: ExtFeltValue) -> Self::Output {
        Default::default()
    }
}

impl Add<ExtFelt> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn add(self, _rhs: ExtFelt) -> Self::Output {
        Default::default()
    }
}

impl Add<ExtFeltVar> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn add(self, _rhs: ExtFeltVar) -> Self::Output {
        Default::default()
    }
}

impl Sub<ExtFeltValue> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn sub(self, _rhs: ExtFeltValue) -> Self::Output {
        Default::default()
    }
}

impl Sub<ExtFelt> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn sub(self, _rhs: ExtFelt) -> Self::Output {
        Default::default()
    }
}

impl Sub<ExtFeltVar> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn sub(self, _rhs: ExtFeltVar) -> Self::Output {
        Default::default()
    }
}

impl Mul<ExtFeltValue> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn mul(self, _rhs: ExtFeltValue) -> Self::Output {
        Default::default()
    }
}

impl Mul<ExtFelt> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn mul(self, _rhs: ExtFelt) -> Self::Output {
        Default::default()
    }
}

impl Mul<ExtFeltVar> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn mul(self, _rhs: ExtFeltVar) -> Self::Output {
        Default::default()
    }
}

impl AddAssign for ExtFeltValue {
    fn add_assign(&mut self, _rhs: Self) {}
}

impl SubAssign for ExtFeltValue {
    fn sub_assign(&mut self, _rhs: Self) {}
}

impl MulAssign for ExtFeltValue {
    fn mul_assign(&mut self, _rhs: Self) {}
}

impl Neg for ExtFeltValue {
    type Output = ExtFeltValue;

    fn neg(self) -> Self::Output {
        Default::default()
    }
}

impl Sum for ExtFeltValue {
    fn sum<I: Iterator<Item = Self>>(_iter: I) -> Self {
        Default::default()
    }
}

impl Product for ExtFeltValue {
    fn product<I: Iterator<Item = Self>>(_iter: I) -> Self {
        Default::default()
    }
}

impl Add<PicusExpr> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn add(self, _rhs: PicusExpr) -> Self::Output {
        Default::default()
    }
}

impl Sub<PicusExpr> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn sub(self, _rhs: PicusExpr) -> Self::Output {
        Default::default()
    }
}

impl Mul<PicusExpr> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn mul(self, _rhs: PicusExpr) -> Self::Output {
        Default::default()
    }
}

impl AddAssign<PicusExpr> for ExtFeltValue {
    fn add_assign(&mut self, _rhs: PicusExpr) {}
}

impl SubAssign<PicusExpr> for ExtFeltValue {
    fn sub_assign(&mut self, _rhs: PicusExpr) {}
}

impl MulAssign<PicusExpr> for ExtFeltValue {
    fn mul_assign(&mut self, _rhs: PicusExpr) {}
}

impl AbstractExtensionField<PicusExpr> for ExtFeltValue {
    const D: usize = EXT_FELT_DEGREE;

    fn from_base(_value: PicusExpr) -> Self {
        Default::default()
    }

    fn from_base_slice(_values: &[PicusExpr]) -> Self {
        Default::default()
    }

    fn from_base_fn<F: FnMut(usize) -> PicusExpr>(_f: F) -> Self {
        Default::default()
    }

    fn as_base_slice(&self) -> &[PicusExpr] {
        &[]
    }
}

use std::{
    iter::{zip, Product, Sum},
    ops::{Add, AddAssign, Deref, Mul, MulAssign, Neg, Sub, SubAssign},
};

use p3_field::{AbstractExtensionField, AbstractField};

use super::{
    vars::{ExtFeltVar, FeltVar},
    Codegen, ExtFelt, Felt, EXT_FELT_DEGREE, FIELD_BETA,
};

/// Opaque struct that represents a IR value in llzk.
pub type Value = llzk_bridge::Value;

impl From<FeltValue> for Value {
    fn from(value: FeltValue) -> Self {
        value.inner
    }
}

impl From<ExtFeltValue> for Value {
    fn from(value: ExtFeltValue) -> Self {
        value.inner
    }
}

#[derive(Clone, Default, Copy, Debug)]
pub struct FeltValue {
    pub(crate) inner: Value,
}

impl Deref for FeltValue {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        &self.inner
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
        let codegen = Codegen::instance();
        codegen.const_felt(value).into()
    }
}

impl From<Value> for FeltValue {
    fn from(value: Value) -> Self {
        Self { inner: value }
    }
}

impl Add<FeltValue> for FeltValue {
    type Output = FeltValue;

    fn add(self, rhs: FeltValue) -> Self::Output {
        let codegen = Codegen::instance();
        codegen.felt_add(self, rhs)
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
        let codegen = Codegen::instance();
        codegen.felt_sub(self, rhs)
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
        let codegen = Codegen::instance();
        codegen.felt_mul(self, rhs)
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
        self.inner = *new_value;
    }
}

impl SubAssign for FeltValue {
    fn sub_assign(&mut self, rhs: Self) {
        let new_value = *self - rhs;
        self.inner = *new_value;
    }
}

impl MulAssign for FeltValue {
    fn mul_assign(&mut self, rhs: Self) {
        let new_value = *self * rhs;
        self.inner = *new_value;
    }
}

impl Neg for FeltValue {
    type Output = FeltValue;

    fn neg(self) -> Self::Output {
        let codegen = Codegen::instance();
        codegen.felt_neg(self)
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

#[derive(Clone, Default, Debug, Copy)]
pub struct ExtFeltValue {
    inner: Value,
}

impl Deref for ExtFeltValue {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl From<Value> for ExtFeltValue {
    fn from(value: Value) -> Self {
        Self { inner: value }
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
    fn from(value: ExtFelt) -> Self {
        let values = value
            .as_base_slice()
            .iter()
            .map(|s: &Felt| Into::<FeltValue>::into(*s))
            .collect::<Vec<_>>();
        ExtFeltValue::from_base_slice(&values)
    }
}

fn linear_ext_felt_op<F: FnMut(FeltValue, FeltValue) -> FeltValue>(
    lhs: ExtFeltValue,
    rhs: ExtFeltValue,
    mut f: F,
) -> ExtFeltValue {
    let result = zip(lhs.as_base_slice(), rhs.as_base_slice())
        .map(|t| {
            let (lhs, rhs) = t;
            f(*lhs, *rhs)
        })
        .collect::<Vec<_>>();
    ExtFeltValue::from_base_slice(&result)
}

impl Add<ExtFeltValue> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn add(self, rhs: ExtFeltValue) -> Self::Output {
        linear_ext_felt_op(self, rhs, |lhs, rhs| lhs + rhs)
    }
}

impl Add<ExtFelt> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn add(self, rhs: ExtFelt) -> Self::Output {
        self + Into::<ExtFeltValue>::into(rhs)
    }
}

impl Add<ExtFeltVar> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn add(self, rhs: ExtFeltVar) -> Self::Output {
        self + Into::<ExtFeltValue>::into(rhs)
    }
}

impl Sub<ExtFeltValue> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn sub(self, rhs: ExtFeltValue) -> Self::Output {
        linear_ext_felt_op(self, rhs, |lhs, rhs| lhs - rhs)
    }
}

impl Sub<ExtFelt> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn sub(self, rhs: ExtFelt) -> Self::Output {
        self - Into::<ExtFeltValue>::into(rhs)
    }
}

impl Sub<ExtFeltVar> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn sub(self, rhs: ExtFeltVar) -> Self::Output {
        self - Into::<ExtFeltValue>::into(rhs)
    }
}

impl Mul<ExtFeltValue> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn mul(self, rhs: ExtFeltValue) -> Self::Output {
        let lhs: &[FeltValue] = self.as_base_slice();
        let rhs: &[FeltValue] = rhs.as_base_slice();
        let nbeta: FeltValue = (-Felt::from_canonical_usize(FIELD_BETA)).into();

        let out0 = lhs[0] * rhs[0] + nbeta * (lhs[1] * rhs[3] + lhs[2] * rhs[2] + lhs[3] * rhs[1]);
        let out1 = lhs[0] * rhs[1] + lhs[1] * rhs[0] + nbeta * (lhs[2] * rhs[3] + lhs[3] * rhs[2]);
        let out2 = lhs[0] * rhs[2] + lhs[1] * rhs[1] + lhs[2] * rhs[0] + nbeta * (lhs[3] * rhs[3]);
        let out3 = lhs[0] * rhs[3] + lhs[1] * rhs[2] + lhs[2] * rhs[1] + lhs[3] * rhs[0];

        Self::from_base_slice(&[out0, out1, out2, out3])
    }
}

impl Mul<ExtFelt> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn mul(self, rhs: ExtFelt) -> Self::Output {
        self * Into::<ExtFeltValue>::into(rhs)
    }
}

impl Mul<ExtFeltVar> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn mul(self, rhs: ExtFeltVar) -> Self::Output {
        self * Into::<ExtFeltValue>::into(rhs)
    }
}

impl AddAssign for ExtFeltValue {
    fn add_assign(&mut self, rhs: Self) {
        self.inner = *(*self + rhs);
    }
}

impl SubAssign for ExtFeltValue {
    fn sub_assign(&mut self, rhs: Self) {
        self.inner = *(*self - rhs);
    }
}

impl MulAssign for ExtFeltValue {
    fn mul_assign(&mut self, rhs: Self) {
        self.inner = *(*self * rhs);
    }
}

impl Neg for ExtFeltValue {
    type Output = ExtFeltValue;

    fn neg(self) -> Self::Output {
        Self::zero() - self
    }
}

impl Sum for ExtFeltValue {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = Self::zero();
        for item in iter {
            output = output + item;
        }
        output
    }
}

impl Product for ExtFeltValue {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = Self::one();
        for item in iter {
            output = output * item;
        }
        output
    }
}

impl From<FeltValue> for ExtFeltValue {
    fn from(value: FeltValue) -> Self {
        let zero = Felt::zero().into();
        let bases: [FeltValue; <Self as AbstractExtensionField<FeltValue>>::D] =
            core::array::from_fn(|i| if i == 0 { value } else { zero });
        Self::from_base_slice(&bases)
    }
}

impl Add<FeltValue> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn add(self, rhs: FeltValue) -> Self::Output {
        self + Into::<ExtFeltValue>::into(rhs)
    }
}

impl Sub<FeltValue> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn sub(self, rhs: FeltValue) -> Self::Output {
        self - Into::<ExtFeltValue>::into(rhs)
    }
}

impl Mul<FeltValue> for ExtFeltValue {
    type Output = ExtFeltValue;

    fn mul(self, rhs: FeltValue) -> Self::Output {
        self * Into::<ExtFeltValue>::into(rhs)
    }
}

impl AddAssign<FeltValue> for ExtFeltValue {
    fn add_assign(&mut self, rhs: FeltValue) {
        self.inner = *(*self + rhs);
    }
}

impl SubAssign<FeltValue> for ExtFeltValue {
    fn sub_assign(&mut self, rhs: FeltValue) {
        self.inner = *(*self - rhs);
    }
}

impl MulAssign<FeltValue> for ExtFeltValue {
    fn mul_assign(&mut self, rhs: FeltValue) {
        self.inner = *(*self * rhs);
    }
}

impl AbstractExtensionField<FeltValue> for ExtFeltValue {
    const D: usize = EXT_FELT_DEGREE;

    fn from_base(value: FeltValue) -> Self {
        value.into()
    }

    fn from_base_slice(values: &[FeltValue]) -> Self {
        let codegen = Codegen::instance();
        codegen.literal_array(values, &[<Self as AbstractExtensionField<FeltValue>>::D]).into()
    }

    fn from_base_fn<F: FnMut(usize) -> FeltValue>(f: F) -> Self {
        let values = (0..<Self as AbstractExtensionField<FeltValue>>::D).map(f).collect::<Vec<_>>();
        Self::from_base_slice(&values)
    }

    fn as_base_slice(&self) -> &[FeltValue] {
        let codegen = Codegen::instance();
        let arr: [FeltValue; <Self as AbstractExtensionField<FeltValue>>::D] =
            core::array::from_fn(|i: usize| {
                let index = codegen.const_index(i);
                codegen.read_array(**self, index).into()
            });
        codegen.manage(&arr)
    }
}

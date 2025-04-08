use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    instruction::Instruction32,
    llzk,
    picusextractor::{self, PicusExtractor},
    symbolic_var_f::SymbolicVarF,
    CUDA_P3_EVAL_CODE, CUDA_P3_EVAL_EXPR_F_CTR, F, LLZK_CODEGEN, PICUS_EXTRACTOR,
};

use p3_field::{AbstractField, PrimeField32};

#[derive(Debug, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SymbolicExprF(pub u32);

impl SymbolicExprF {
    // #[instrument(skip_all, level = "trace", name = "Empty for SymbolicExprF")]
    pub fn empty() -> Self {
        Self(u32::MAX)
    }

    // #[instrument(skip_all, level = "trace", name = "Alloc for SymbolicExprF")]
    pub fn alloc() -> Self {
        let mut tmp = CUDA_P3_EVAL_EXPR_F_CTR.lock().unwrap();
        let id = *tmp;
        *tmp += 1;
        drop(tmp);
        Self(id)
    }

    pub fn variant(&self) -> u8 {
        0
    }

    pub fn data(&self) -> u32 {
        self.0
    }
}

impl Default for SymbolicExprF {
    // #[instrument(skip_all, level = "trace", name = "Default for SymbolicExprF")]
    fn default() -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::zero()));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.expr_map.insert(output, picusextractor::PicusExpr::Const(0));
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::zero()));
        output
    }
}

impl From<F> for SymbolicExprF {
    // #[instrument(skip_all, level = "trace", name = "From<F> for SymbolicExprF")]
    fn from(f: F) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, f));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.expr_map.insert(
            output,
            picusextractor::PicusExpr::Const(PrimeField32::as_canonical_u32(&f).into()),
        );
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(f));
        output
    }
}

impl Add<F> for SymbolicExprF {
    type Output = Self;

    // #[instrument(skip_all, level = "trace", name = "Add<F> for SymbolicExprF")]
    fn add(self, rhs: F) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_add_ec(output, self, rhs));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        let maybe_self_expr = pe.expr_map.get(&self).cloned();
        if let Some(picus_expr) = maybe_self_expr {
            let const_expr =
                picusextractor::PicusExpr::Const(PrimeField32::as_canonical_u32(&rhs).into());
            pe.expr_map.insert(
                output,
                picusextractor::PicusExpr::Add(Box::new(picus_expr), Box::new(const_expr)),
            );
        }
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let rhs = llzk.const_f(rhs);
        let lhs = llzk.get_f(self);
        llzk.assign_f(output, llzk.binop(llzk::BinOps::Add, lhs, rhs));
        output
    }
}

impl Add<SymbolicVarF> for SymbolicExprF {
    type Output = Self;

    // #[instrument(skip_all, level = "trace", name = "Add<SymbolicVarF> for SymbolicExprF")]
    fn add(self, rhs: SymbolicVarF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_add_ev(output, self, rhs));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        let self_expr = pe.expr_map.get(&self).cloned().unwrap();
        let maybe_var_expr = pe.var_map.get(&rhs).cloned();
        if let Some(var_expr) = maybe_var_expr {
            pe.expr_map.insert(
                output,
                picusextractor::PicusExpr::Add(Box::new(self_expr), Box::new(var_expr)),
            );
        } else {
            let picus_var = pe.fresh_var();
            pe.var_map.insert(rhs, picus_var.clone());
            pe.expr_map.insert(
                output,
                picusextractor::PicusExpr::Add(Box::new(self_expr), Box::new(picus_var)),
            );
        }
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let rhs = llzk.load_var(rhs);
        let lhs = llzk.get_f(self);
        llzk.assign_f(output, llzk.binop(llzk::BinOps::Add, lhs, rhs));
        output
    }
}

impl Add<SymbolicExprF> for SymbolicExprF {
    type Output = SymbolicExprF;

    // #[instrument(skip_all, level = "trace", name = "Add<SymbolicExprF> for SymbolicExprF")]
    fn add(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_add_ee(output, self, rhs));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_binop(&output, &self, &rhs, picusextractor::PicusBinop::Add);
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let rhs = llzk.get_f(rhs);
        let lhs = llzk.get_f(self);
        llzk.assign_f(output, llzk.binop(llzk::BinOps::Add, lhs, rhs));
        output
    }
}

impl AddAssign<SymbolicExprF> for SymbolicExprF {
    // #[instrument(skip_all, level = "trace", name = "AddAssign<SymbolicExprF> for SymbolicExprF")]
    fn add_assign(&mut self, _: SymbolicExprF) {
        unreachable!()
    }
}

impl Sub<F> for SymbolicExprF {
    type Output = Self;

    // #[instrument(skip_all, level = "trace", name = "Sub<F> for SymbolicExprF")]
    fn sub(self, rhs: F) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_sub_ec(output, self, rhs));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_binop_const(&output, &self, &rhs, picusextractor::PicusBinop::Sub);
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let rhs = llzk.const_f(rhs);
        let lhs = llzk.get_f(self);
        llzk.assign_f(output, llzk.binop(llzk::BinOps::Sub, lhs, rhs));
        output
    }
}

impl Sub<SymbolicVarF> for SymbolicExprF {
    type Output = Self;

    // #[instrument(skip_all, level = "trace", name = "Sub<SymbolicVarF> for SymbolicExprF")]
    fn sub(self, rhs: SymbolicVarF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_sub_ev(output, self, rhs));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_binop_var(&output, &self, &rhs, picusextractor::PicusBinop::Sub);
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let rhs = llzk.load_var(rhs);
        let lhs = llzk.get_f(self);
        llzk.assign_f(output, llzk.binop(llzk::BinOps::Sub, lhs, rhs));
        output
    }
}

impl Sub<SymbolicExprF> for SymbolicExprF {
    type Output = Self;

    // #[instrument(skip_all, level = "trace", name = "Sub<SymbolicExprF> for SymbolicExprF")]
    fn sub(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_sub_ee(output, self, rhs));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_binop(&output, &self, &rhs, picusextractor::PicusBinop::Sub);
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let rhs = llzk.get_f(rhs);
        let lhs = llzk.get_f(self);
        llzk.assign_f(output, llzk.binop(llzk::BinOps::Sub, lhs, rhs));
        output
    }
}

impl SubAssign<SymbolicExprF> for SymbolicExprF {
    // #[instrument(skip_all, level = "trace", name = "SubAssign<SymbolicExprF> for SymbolicExprF")]
    fn sub_assign(&mut self, _: SymbolicExprF) {
        unreachable!()
    }
}

impl Mul<F> for SymbolicExprF {
    type Output = Self;

    // #[instrument(skip_all, level = "trace", name = "Mul<F> for SymbolicExprF")]
    fn mul(self, rhs: F) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_mul_ec(output, self, rhs));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_binop_const(&output, &self, &rhs, picusextractor::PicusBinop::Mul);
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let rhs = llzk.const_f(rhs);
        let lhs = llzk.get_f(self);
        llzk.assign_f(output, llzk.binop(llzk::BinOps::Mul, lhs, rhs));
        output
    }
}

impl Mul<SymbolicVarF> for SymbolicExprF {
    type Output = Self;

    // #[instrument(skip_all, level = "trace", name = "Mul<SymbolicVarF> for SymbolicExprF")]
    fn mul(self, rhs: SymbolicVarF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_mul_ev(output, self, rhs));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_binop_var(&output, &self, &rhs, picusextractor::PicusBinop::Mul);
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let rhs = llzk.load_var(rhs);
        let lhs = llzk.get_f(self);
        llzk.assign_f(output, llzk.binop(llzk::BinOps::Mul, lhs, rhs));
        output
    }
}

impl Mul<SymbolicExprF> for SymbolicExprF {
    type Output = Self;

    // #[instrument(skip_all, level = "trace", name = "Mul<SymbolicExprF> for SymbolicExprF")]
    fn mul(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_mul_ee(output, self, rhs));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_binop(&output, &self, &rhs, picusextractor::PicusBinop::Mul);
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let rhs = llzk.get_f(rhs);
        let lhs = llzk.get_f(self);
        llzk.assign_f(output, llzk.binop(llzk::BinOps::Mul, lhs, rhs));
        output
    }
}

impl MulAssign<SymbolicExprF> for SymbolicExprF {
    // #[instrument(skip_all, level = "trace", name = "MulAssign<SymbolicExprF> for SymbolicExprF")]
    fn mul_assign(&mut self, _: SymbolicExprF) {
        unreachable!()
    }
}

impl Neg for SymbolicExprF {
    type Output = Self;

    // #[instrument(skip_all, level = "trace", name = "Neg for SymbolicExprF")]
    fn neg(self) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_neg_e(output, self));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        let self_expr = pe.expr_map.get(&self).cloned().unwrap();
        pe.expr_map.insert(output, picusextractor::PicusExpr::Neg(Box::new(self_expr.clone())));
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        let value = llzk.get_f(self);
        llzk.assign_f(output, llzk.unop(llzk::UnOps::Neg, value));
        output
    }
}

impl Sum for SymbolicExprF {
    // #[instrument(skip_all, level = "trace", name = "Sum for SymbolicExprF")]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = SymbolicExprF::zero();
        for item in iter {
            output = output + item;
        }
        output
    }
}

impl Product for SymbolicExprF {
    // #[instrument(skip_all, level = "trace", name = "Product for SymbolicExprF")]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = SymbolicExprF::one();
        for item in iter {
            output = output * item;
        }
        output
    }
}

impl Clone for SymbolicExprF {
    #[allow(clippy::non_canonical_clone_impl)]
    // #[instrument(skip_all, level = "trace", name = "Clone for SymbolicExprF")]
    fn clone(&self) -> Self {
        // let output = SymbolicExprF::alloc();
        // let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        // code.push(Instruction32::f_assign_e(output, *self));
        // drop(code);
        // output
        *self
    }
}

impl AbstractField for SymbolicExprF {
    type F = F;

    // #[instrument(skip_all, level = "trace", name = "Zero for SymbolicExprF")]
    fn zero() -> Self {
        let output = SymbolicExprF::alloc();
        let mut code: std::sync::MutexGuard<'_, Vec<Instruction32>> =
            CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::zero()));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::zero());
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::zero()));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "One for SymbolicExprF")]
    fn one() -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::one()));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::one());
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::one()));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "Two for SymbolicExprF")]
    fn two() -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::two()));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::two());
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::two()));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "NegOne for SymbolicExprF")]
    fn neg_one() -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::neg_one()));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::neg_one());
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::neg_one()));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "From<F> for SymbolicExprF")]
    fn from_f(f: Self::F) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, f));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &f);
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(f));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "From<bool> for SymbolicExprF")]
    fn from_bool(b: bool) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::from_bool(b)));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::from_bool(b));
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::from_bool(b)));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "From<u8> for SymbolicExprF")]
    fn from_canonical_u8(n: u8) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::from_canonical_u8(n)));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.add_range_constraint(&output, 256);
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::from_canonical_u8(n)));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "From<u16> for SymbolicExprF")]
    fn from_canonical_u16(n: u16) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::from_canonical_u16(n)));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.add_range_constraint(&output, 2u64.pow(16));
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::from_canonical_u16(n)));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "From<u32> for SymbolicExprF")]
    fn from_canonical_u32(n: u32) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::from_canonical_u32(n)));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.add_range_constraint(&output, 2u64.pow(32));
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::from_canonical_u32(n)));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "From<u64> for SymbolicExprF")]
    fn from_canonical_u64(n: u64) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::from_canonical_u64(n)));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::from_canonical_u64(n));
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::from_canonical_u64(n)));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "From<usize> for SymbolicExprF")]
    fn from_canonical_usize(n: usize) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::from_canonical_usize(n)));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::from_canonical_usize(n));
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::from_canonical_usize(n)));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "From<u32> for SymbolicExprF")]
    fn from_wrapped_u32(n: u32) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::from_wrapped_u32(n)));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::from_wrapped_u32(n));
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::from_wrapped_u32(n)));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "From<u64> for SymbolicExprF")]
    fn from_wrapped_u64(n: u64) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::from_wrapped_u64(n)));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::from_wrapped_u64(n));
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::from_wrapped_u64(n)));
        output
    }

    // #[instrument(skip_all, level = "trace", name = "Generator for SymbolicExprF")]
    fn generator() -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assign_c(output, F::generator()));
        drop(code);
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        pe.process_field(&output, &F::generator());
        drop(pe);
        // LLZK_CODEGEN
        let llzk = LLZK_CODEGEN.lock().unwrap();
        llzk.assign_f(output, llzk.const_f(F::generator()));
        output
    }
}

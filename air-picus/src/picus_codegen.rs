use crate::field::Felt;
use crate::PICUS_EXTRACTOR;
use rug::Integer;
use p3_field::{AbstractField, PrimeField32};
use sp1_core_executor::ByteOpcode;
use std::collections::HashMap;
use std::iter::{Product, Sum};
use std::ops::{AddAssign, MulAssign, Neg, SubAssign};
use std::ops::{Add, Mul, Sub};


const SHIFT_CARRY_MOD: &str  = "shift_carry";
const BITWISE_AND_MOD: &str  = "bitand";
const BITWISE_XOR_MOD: &str  = "bitxor";
const BITWISE_OR_MOD: &str  = "bitor";


const MAX_EXPR_SIZE: usize = 10;

/// The order of the arguments in the constraint function
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Args {
    _SelfArg = 0,
    _Inputs,
    Preprocessed,
    PreprocessedNext,
    PublicValues,
    _IsFirstRow,
    _IsLastRow,
    _InputIsTransition,
}

#[repr(u8)]
#[derive(Clone, Debug, Copy)]
pub enum SignalType {
    Input = 0,
    Temporary,
    Output,
    Ignore
}

#[derive(Debug, Clone, Copy)]
pub struct PicusVar(pub (u64, SignalType));

impl Default for PicusVar {
    fn default() -> Self {
        PicusVar((0, SignalType::Ignore))
    }
}

#[derive(Debug, Clone)]
pub enum PicusExpr {
    Const(Integer),
    Var(PicusVar),
    Add(Box<PicusExpr>, Box<PicusExpr>),
    Sub(Box<PicusExpr>, Box<PicusExpr>),
    Mul(Box<PicusExpr>, Box<PicusExpr>),
    Div(Box<PicusExpr>, Box<PicusExpr>),
    Neg(Box<PicusExpr>),
}

impl PicusExpr {
    pub fn size(&self) -> usize {
        match self {
            Self::Const(_) | Self::Var(_) => 1,
            Self::Add(v1, v2) |
            Self::Sub(v1, v2) |
            Self::Mul(v1, v2) |
            Self::Div(v1, v2)=> v1.size() + v2.size(),
            Self::Neg(v) => v.size() + 1,
        }
    }
}

impl Default for PicusExpr {
    fn default() -> Self {
        PicusExpr::Const(Integer::from(0))
    }
}

#[derive(Debug, Clone)]
pub enum PicusConstraint {
    Lt(Box<PicusExpr>, Box<PicusExpr>),
    Leq(Box<PicusExpr>, Box<PicusExpr>),
    Gt(Box<PicusExpr>, Box<PicusExpr>),
    Geq(Box<PicusExpr>, Box<PicusExpr>),
    Eq(Box<PicusExpr>), // expr = 0
}

#[derive(Debug, Clone, Default)]
pub struct PicusCall {
    pub mod_name: String,
    pub outputs: Vec<PicusExpr>,
    pub inputs: Vec<PicusExpr>,
}

#[derive(Debug, Clone, Default)]
pub struct PicusModule {
    pub name: String,
    pub inputs: Vec<PicusExpr>,
    pub outputs: Vec<PicusExpr>, 
    pub constraints: Vec<PicusConstraint>,
    pub calls: Vec<PicusCall>,
}

impl PicusModule {
    pub fn new(name: String) -> Self {
        Self {
            name,
            inputs: Vec::new(),
            outputs: Vec::new(),
            constraints: Vec::new(),
            calls: Vec::new(),
        }
    }

    pub fn dump(&self) -> Vec<String> {
        let mut res = Vec::new();
        res.push(format!("(begin-module {})", self.name));
        for inp in &self.inputs {
            res.push(format!("(input {})", dump_expr(inp)))
        }
        for out in &self.outputs {
            res.push(format!("(output {})", dump_expr(out)))
        }
        
        for constraint in &self.constraints {
            res.push(dump_constraint(constraint));
        }
        for call in &self.calls {
            res.push(dump_call(call));
        }
        res.push(format!("(end-module)"));
        res
    }
}

pub fn dump_constraint(constraint: &PicusConstraint) -> String {
    match constraint {
        PicusConstraint::Lt(e1, e2) => format!("(assert (< {} {}))", dump_expr(e1), dump_expr(e2)),
        PicusConstraint::Leq(e1, e2) => format!("(assert (<= {} {}))", dump_expr(e1), dump_expr(e2)),
        PicusConstraint::Gt(e1, e2) => format!("(assert (> {} {}))", dump_expr(e1), dump_expr(e2)),
        PicusConstraint::Geq(e1, e2) => format!("(assert (>= {} {}))", dump_expr(e1), dump_expr(e2)),
        PicusConstraint::Eq(e) => format!("(assert (= {} 0))", dump_expr(e))
    }
}

fn dump_expr_array(exprs: &[PicusExpr]) -> String {
    let body = exprs.iter()
        .map(dump_expr) 
        .collect::<Vec<_>>()
        .join(" ");
    format!("[{}]", body)
}

pub fn dump_expr(expr: &PicusExpr) -> String {
    match expr {
        PicusExpr::Const(v) => v.to_string(),
        PicusExpr::Var(v) => format!("x{:?}", v.0.0),
        PicusExpr::Add(e1, e2) => format!("(+ {} {})", dump_expr(e1), dump_expr(e2)),
        PicusExpr::Mul(e1, e2) => format!("(* {} {})", dump_expr(e1), dump_expr(e2)),
        PicusExpr::Div(e1, e2) => format!("(/ {} {})", dump_expr(e1), dump_expr(e2)),
        PicusExpr::Sub(e1, e2) => format!("(- {} {})", dump_expr(e1), dump_expr(e2)),
        PicusExpr::Neg(e) => format!("(- {})", dump_expr(e)),
    }
}

pub fn dump_call(call: &PicusCall) -> String {
    let outputs = dump_expr_array(&call.outputs);
    let inputs = dump_expr_array(&call.inputs);
    format!("(call {} {} {})", outputs, call.mod_name, inputs)
}

pub struct PicusExtractor {
    pub module_map: HashMap<String, PicusModule>,
    pub cur_module: String,
    var_cntr: u64,
}

impl Add<Felt> for PicusVar {
    type Output = PicusExpr;

    fn add(self, rhs: Felt) -> Self::Output {
        let lhs: PicusExpr = self.into();
        lhs + rhs
    }
}

impl Add<PicusVar> for PicusVar {
    type Output = PicusExpr;

    fn add(self, rhs: PicusVar) -> Self::Output {
        let lhs: PicusExpr = self.into();
        lhs + rhs
    }
}

impl Add<PicusExpr> for PicusVar {
    type Output = PicusExpr;

    fn add(self, rhs: PicusExpr) -> Self::Output {
        let lhs: PicusExpr = self.into();
        lhs + rhs
    }
}

impl Sub<Felt> for PicusVar {
    type Output = PicusExpr;

    fn sub(self, rhs: Felt) -> Self::Output {
        let lhs: PicusExpr = self.into();
        lhs - rhs
    }
}

impl Sub<PicusVar> for PicusVar {
    type Output = PicusExpr;

    fn sub(self, rhs: PicusVar) -> Self::Output {
        let lhs: PicusExpr = self.into();
        lhs - rhs
    }
}

impl Sub<PicusExpr> for PicusVar {
    type Output = PicusExpr;

    fn sub(self, rhs: PicusExpr) -> Self::Output {
        let lhs: PicusExpr = self.into();
        lhs - rhs
    }
}

impl Mul<Felt> for PicusVar {
    type Output = PicusExpr;

    fn mul(self, rhs: Felt) -> Self::Output {
        let lhs: PicusExpr = self.into();
        lhs * rhs
    }
}

impl Mul<PicusVar> for PicusVar {
    type Output = PicusExpr;

    fn mul(self, rhs: PicusVar) -> Self::Output {
        let lhs: PicusExpr = self.into();
        lhs * rhs
    }
}

impl Mul<PicusExpr> for PicusVar {
    type Output = PicusExpr;

    fn mul(self, rhs: PicusExpr) -> Self::Output {
        let lhs: PicusExpr = self.into();
        lhs * rhs
    }
}

impl From<Felt> for PicusExpr {
    fn from(f: Felt) -> Self {
        PicusExpr::Const(Integer::from(PrimeField32::as_canonical_u32(&f)))
    }
}

impl From<PicusVar> for PicusExpr {
    fn from(f: PicusVar) -> Self {
        PicusExpr::Var(f)
    }
}

impl Add<PicusExpr> for PicusExpr {
    type Output = PicusExpr;

    fn add(self, rhs: PicusExpr) -> Self::Output {
        let res = PicusExpr::Add(Box::new(self.clone()), Box::new(rhs.clone()));
        if self.size() + rhs.size() > MAX_EXPR_SIZE {
            let mut pe = PICUS_EXTRACTOR.lock().unwrap();
            let res = pe.build_temporary_var(res);
            drop(pe);
            return res;
        }
        res
    }
}

impl Add<PicusVar> for PicusExpr {
    type Output = PicusExpr;

    fn add(self, rhs: PicusVar) -> Self::Output {
        PicusExpr::Add(Box::new(self), Box::new(rhs.into()))
    }
}

impl Add<Felt> for PicusExpr {
    type Output = PicusExpr;

    fn add(self, rhs: Felt) -> Self::Output {
        self + Into::<PicusExpr>::into(rhs)
    }
}

impl Sub<PicusVar> for PicusExpr {
    type Output = PicusExpr;

    fn sub(self, rhs: PicusVar) -> Self::Output {
        PicusExpr::Sub(Box::new(self), Box::new(rhs.into()))
    }
}

impl Sub<PicusExpr> for PicusExpr {
    type Output = PicusExpr;

    fn sub(self, rhs: PicusExpr) -> Self::Output {
        let res = PicusExpr::Sub(Box::new(self.clone()), Box::new(rhs.clone()));
        // technically the expression can overflow the max size here but it hasn't caused any
        // issues for now
        if self.size() + rhs.size() > MAX_EXPR_SIZE {
            let mut pe = PICUS_EXTRACTOR.lock().unwrap();
            let res = pe.build_temporary_var(res);
            drop(pe);
            return res;
        }
        res
    }
}

impl Sub<Felt> for PicusExpr {
    type Output = PicusExpr;

    fn sub(self, rhs: Felt) -> Self::Output {
        self - Into::<PicusExpr>::into(rhs)
    }
}

impl Mul<PicusExpr> for PicusExpr {
    type Output = PicusExpr;

    fn mul(self, rhs: PicusExpr) -> Self::Output {
        let res = PicusExpr::Mul(Box::new(self.clone()), Box::new(rhs.clone()));
        if self.size() + rhs.size() > MAX_EXPR_SIZE {
            let mut pe = PICUS_EXTRACTOR.lock().unwrap();
            let res = pe.build_temporary_var(res);
            drop(pe);
            return res;
        }
        res
    }
}

impl Mul<PicusVar> for PicusExpr {
    type Output = PicusExpr;

    fn mul(self, rhs: PicusVar) -> Self::Output {
        if self.size() + 1 > MAX_EXPR_SIZE {
        }
        PicusExpr::Mul(Box::new(self), Box::new(rhs.into()))
    }
}

impl Mul<Felt> for PicusExpr {
    type Output = PicusExpr;

    fn mul(self, rhs: Felt) -> Self::Output {
        self * Into::<PicusExpr>::into(rhs)
    }
}

impl AddAssign for PicusExpr {
    fn add_assign(&mut self, rhs: Self) {
        let new_value = self.clone() + rhs;
        (*self) = new_value.into();
    }
}

impl SubAssign for PicusExpr {
    fn sub_assign(&mut self, rhs: Self) {
        let new_value = self.clone() - rhs;
        (*self) = new_value.into();
    }
}

impl MulAssign for PicusExpr {
    fn mul_assign(&mut self, rhs: Self) {
        let new_value = self.clone() * rhs;
        (*self) = new_value.into();
    }
}

impl Neg for PicusExpr {
    type Output = PicusExpr;

    fn neg(self) -> Self::Output {
        PicusExpr::Neg(Box::new(self))
    }
}

impl Sum for PicusExpr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = PicusExpr::Const(Integer::ZERO);
        for item in iter {
            output = output + item;
        }
        output
    }
}

impl Product for PicusExpr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = PicusExpr::Const(Integer::ONE.clone());
        for item in iter {
            output = output * item;
        }
        output
    }
}

impl PicusExtractor {

    pub fn new() -> Self {
        PicusExtractor {  
            module_map: HashMap::new(),
            cur_module: "".to_string(),
            var_cntr: 0 }
    }

    pub fn reset(&mut self) {
        (*self) = PicusExtractor::new();
    }

    pub fn build_temporary_var(&mut self, large_expr: PicusExpr) -> PicusExpr {
        let picus_var = self.fresh_var(SignalType::Temporary);
        let diff_expr = PicusExpr::Sub(Box::new(picus_var.into()), Box::new(large_expr));
        let eq_constraint = PicusConstraint::Eq(Box::new(diff_expr));
        self.add_constraint(eq_constraint);
        picus_var.into()
    }

    pub fn init_cur_module(&mut self, name: &String) {
        self.cur_module = name.clone();
        assert!(!self.module_map.contains_key(name));
        let module = PicusModule::new(name.clone());
        self.module_map.insert(name.clone(), module);
    }
    pub fn add_input(&mut self, input: PicusExpr) {
        let cur_mod = self.module_map.get_mut(&self.cur_module).unwrap();
        cur_mod.inputs.push(input);
    }

    pub fn add_output(&mut self, output: PicusExpr) {
        let cur_mod = self.module_map.get_mut(&self.cur_module).unwrap();
        cur_mod.outputs.push(output);
    }

    pub fn add_call(&mut self, call: PicusCall) {
        let cur_mod = self.module_map.get_mut(&self.cur_module).unwrap();
        cur_mod.calls.push(call);
    }

    pub fn add_constraint(&mut self, constraint: PicusConstraint) {
        let cur_mod = self.module_map.get_mut(&self.cur_module).unwrap();
        cur_mod.constraints.push(constraint);
    }

    pub fn fresh_input_picus_var(&mut self) -> PicusVar {
        let var = self.fresh_var(SignalType::Input);
        self.add_input(var.into());
        var
    }

    pub fn fresh_var(&mut self, st: SignalType) -> PicusVar {
        let pv = PicusVar((self.var_cntr, st));
        self.var_cntr += 1;
        pv
    }

    pub fn build_shift_carry_module(&mut self) {
        let module_name = "shift_carry".to_string();
        if self.module_map.contains_key(&module_name) {
            return;
        }
        let mut module = PicusModule::new(module_name);
        let inp = self.fresh_var(SignalType::Input);
        let o1 = self.fresh_var(SignalType::Output);
        let o2 = self.fresh_var(SignalType::Output);
        module.inputs.push(inp.into());
        module.outputs.push(o1.into());
        module.outputs.push(o2.into());
        self.module_map.insert(module.name.clone(), module);
    }

    pub fn build_shift_carry_call(&mut self, inp: PicusExpr, out0: PicusExpr, out1: PicusExpr) {
        let call = PicusCall {
            inputs: vec![inp],
            outputs: vec![out0, out1],
            mod_name: SHIFT_CARRY_MOD.to_string(),
        };
        self.add_call(call);
    }

    pub fn get_bitwise_op(op: ByteOpcode) -> String {
        if op == ByteOpcode::AND {
            BITWISE_AND_MOD.to_string()
        } else if op == ByteOpcode::OR {
            BITWISE_OR_MOD.to_string()
        } else if op == ByteOpcode::XOR {
            BITWISE_XOR_MOD.to_string()
        } else {
            panic!("Operation can only be and, or or xor. Found: {:?}", op);
        }
    }

    pub fn build_bitwiseop_module(&mut self, op: ByteOpcode) {
        let module_name = Self::get_bitwise_op(op);
        if self.module_map.contains_key(&module_name) {
            return;
        }
        let mut module = PicusModule::new(module_name);
        let inp0 = self.fresh_var(SignalType::Input);
        let inp1 = self.fresh_var(SignalType::Input);
        let out = self.fresh_var(SignalType::Output);
        module.inputs.push(inp0.into());
        module.inputs.push(inp1.into());
        module.outputs.push(out.into());
        self.module_map.insert(module.name.clone(), module);
    }

    pub fn build_bitwiseop_call(&mut self, op: ByteOpcode, inp0: PicusExpr, inp1: PicusExpr, out: PicusExpr) {
        let module_name = Self::get_bitwise_op(op);
        let call = PicusCall {
            inputs: vec![inp0, inp1],
            outputs: vec![out],
            mod_name: module_name,
        };
        self.add_call(call);
    }

    pub fn serialize(&self) -> String {
        let mut res = Vec::new();
        for module in self.module_map.values() {
            res.push(module.dump());
        }
        let mod_str = res
                    .into_iter()
                    .map(|module_lines| module_lines.join("\n")) // join lines within each module
                    .collect::<Vec<_>>()
                    .join("\n\n"); // separate modules with double newline
        format!("(prime-number {})\n{}", crate::field::P, mod_str)
    }

}

impl AbstractField for PicusExpr {
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
        PicusExpr::Const(Integer::from(n))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        PicusExpr::Const(Integer::from(n))
    }

    fn generator() -> Self {
        Self::F::generator().into()
    }
}

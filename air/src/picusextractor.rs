use std::{collections::HashMap, fmt::Debug};
use std::mem::size_of;

use crate::symbolic_expr_f::{SymbolicExprF};
use crate::symbolic_var_f::SymbolicVarF;
use crate::{F};
use p3_field::{AbstractField, PrimeField32};


#[derive(Debug, Clone)]
pub enum PicusExpr {
    Const(u64),
    Var(String),
    Add(Box<PicusExpr>, Box<PicusExpr>),
    Sub(Box<PicusExpr>, Box<PicusExpr>),
    Mul(Box<PicusExpr>, Box<PicusExpr>),
    Div(Box<PicusExpr>, Box<PicusExpr>),
    Neg(Box<PicusExpr>),
}

#[derive(Debug, Clone)]
pub enum PicusConstraint {
    Lt(Box<PicusExpr>, Box<PicusExpr>),
    Leq(Box<PicusExpr>, Box<PicusExpr>),
    Gt(Box<PicusExpr>, Box<PicusExpr>),
    Geq(Box<PicusExpr>, Box<PicusExpr>),
}

pub enum PicusBinop {
    Add,
    Sub,
    Mul,
    Div,
}
#[derive(Debug, Clone, Default)]
pub struct PicusExtractor {
    pub inputs: Vec<PicusExpr>,
    pub outputs: Vec<PicusExpr>,
    pub expr_map: HashMap<SymbolicExprF, PicusExpr>,
    pub var_map: HashMap<SymbolicVarF, PicusExpr>,
    pub eqs: Vec<PicusExpr>,
    pub constraints: Vec<PicusConstraint>,
    var_cntr: u64,
}

impl PicusExtractor {
    pub fn new() -> Self {
        PicusExtractor {
            inputs: Vec::new(),
            outputs: Vec::new(),
            expr_map: HashMap::new(),
            var_map: HashMap::new(),
            eqs: Vec::new(),
            constraints: Vec::new(),
            var_cntr: 0,
        }
    }

    pub fn add_input(&mut self, var: &SymbolicVarF) {
        let picus_var = self.get_picus_var(var);
        self.inputs.push(picus_var);
    }

    pub fn add_output(&mut self, var: &SymbolicVarF) {
        let picus_var = self.get_picus_var(var);
        self.outputs.push(picus_var);
    }

    pub fn dump(&self) -> Vec<String> {
        let mut res = Vec::new();
        for inp in &self.inputs {
            res.push(format!("(input {})", Self::dump_expr(inp)))
        }
        for out in &self.outputs {
            res.push(format!("(output {})", Self::dump_expr(out)))
        }
        for eq in &self.eqs {
            res.push(format!("(assert (= 0 {}))", Self::dump_expr(eq)));
        }
        for constraint in &self.constraints {
            res.push(Self::dump_constraint(constraint));
        }
        res
    }

    pub fn dump_constraint(constraint: &PicusConstraint) -> String {
        match constraint {
            PicusConstraint::Lt(e1, e2) => format!("(assert (< {} {}))", Self::dump_expr(e1), Self::dump_expr(e2)),
            PicusConstraint::Leq(e1, e2) => format!("(assert (<= {} {}))", Self::dump_expr(e1), Self::dump_expr(e2)),
            PicusConstraint::Gt(e1, e2) => format!("(assert (> {} {}))", Self::dump_expr(e1), Self::dump_expr(e2)),
            PicusConstraint::Geq(e1, e2) => format!("(assert (>= {} {}))", Self::dump_expr(e1), Self::dump_expr(e2)),
        }
    }

    pub fn dump_expr(expr: &PicusExpr) -> String {
        match expr {
            PicusExpr::Const(v) => v.to_string(),
            PicusExpr::Var(v) => format!("x{}", v.clone()),
            PicusExpr::Add(e1, e2) => format!("(+ {} {})", Self::dump_expr(e1), Self::dump_expr(e2)),
            PicusExpr::Mul(e1, e2) => format!("(* {} {})", Self::dump_expr(e1), Self::dump_expr(e2)),
            PicusExpr::Div(e1, e2) => format!("(/ {} {})", Self::dump_expr(e1), Self::dump_expr(e2)),
            PicusExpr::Sub(e1, e2) => format!("(- {} {})", Self::dump_expr(e1), Self::dump_expr(e2)),
            PicusExpr::Neg(e) => format!("(- {})", Self::dump_expr(e)),
        }
    }

    pub fn fresh_var(&mut self) -> PicusExpr {
        let ret = PicusExpr::Var(self.var_cntr.to_string());
        self.var_cntr += 1;
        ret
    }

    fn insert_picus_binop(&mut self, res: &SymbolicExprF, lhs_expr: PicusExpr, rhs_expr: PicusExpr, op: PicusBinop) {
        match op {
            PicusBinop::Add => {self.expr_map.insert(*res, PicusExpr::Add(Box::new(lhs_expr), Box::new(rhs_expr)));}
            PicusBinop::Sub => {self.expr_map.insert(*res, PicusExpr::Sub(Box::new(lhs_expr), Box::new(rhs_expr)));}
            PicusBinop::Mul => {self.expr_map.insert(*res, PicusExpr::Mul(Box::new(lhs_expr), Box::new(rhs_expr)));}
            PicusBinop::Div => {self.expr_map.insert(*res, PicusExpr::Div(Box::new(lhs_expr), Box::new(rhs_expr)));}
        }
    }
    pub fn process_binop(&mut self, res: &SymbolicExprF, lhs: &SymbolicExprF, rhs: &SymbolicExprF, op: PicusBinop) {
        let lhs_expr = self.get_picus_expr(lhs);
        let rhs_expr = self.get_picus_expr(rhs);
        self.insert_picus_binop(res, lhs_expr, rhs_expr, op);
    }

    fn get_picus_expr(&mut self, expr: &SymbolicExprF) -> PicusExpr {
        if let Some(picus_expr) = self.expr_map.get(expr) {
            picus_expr.clone()
        } else {
            let picus_var = self.fresh_var();
            self.expr_map.insert(*expr, picus_var.clone());
            picus_var
        }
    }
    pub fn get_picus_var(&mut self, var: &SymbolicVarF) -> PicusExpr {
        if let Some(var_expr) = self.var_map.get(var) {
            var_expr.clone()
        } else {
            let picus_var = self.fresh_var();
            self.var_map.insert(*var, picus_var.clone());
            picus_var
        }
    }

    pub fn process_binop_var(&mut self, res: &SymbolicExprF, lhs: &SymbolicExprF, var: &SymbolicVarF, op: PicusBinop) {
        let lhs_expr = self.get_picus_expr(lhs);
        let picus_var = self.get_picus_var(var);
        self.insert_picus_binop(res, lhs_expr, picus_var, op);
    }

    pub fn process_binop_two_var(&mut self, res: &SymbolicExprF, lhs: &SymbolicVarF, rhs: &SymbolicVarF, op: PicusBinop) {
        let picus_var_lhs = self.get_picus_var(lhs);
        let picus_var_rhs = self.get_picus_var(rhs);
        self.insert_picus_binop(res, picus_var_lhs, picus_var_rhs, op);
    }

    pub fn process_binop_const(&mut self, res: &SymbolicExprF, lhs: &SymbolicExprF, field: &F, op: PicusBinop) {
        let lhs_expr = self.get_picus_expr(lhs);
        let const_expr = PicusExpr::Const(PrimeField32::as_canonical_u32(field).into());
        self.insert_picus_binop(res, lhs_expr, const_expr, op);
    }

    pub fn process_binop_var_const(&mut self, res: &SymbolicExprF, lhs: &SymbolicVarF, field: &F, op: PicusBinop) {
        let lhs_expr = self.get_picus_var(lhs);
        let const_expr = PicusExpr::Const(PrimeField32::as_canonical_u32(field).into());
        self.insert_picus_binop(res, lhs_expr, const_expr, op);
    }

    pub fn process_field(&mut self, res: &SymbolicExprF, field: &F) {
        let const_expr = PicusExpr::Const(PrimeField32::as_canonical_u32(field).into());
        self.expr_map.insert(*res, const_expr.clone());
    }

    pub fn add_range_constraint_var(&mut self, var: &SymbolicVarF, upper: u64) {
        let var_expr = self.get_picus_var(var);
        self.constraints.push(PicusConstraint::Lt(Box::new(var_expr), Box::new(PicusExpr::Const(upper))));
    }

    pub fn add_range_constraint(&mut self, expr: &SymbolicExprF, upper: u64) {
        let picus_expr = self.get_picus_expr(expr);
        self.constraints.push(PicusConstraint::Lt(Box::new(picus_expr.clone()), Box::new(PicusExpr::Const(upper))));
    }
}
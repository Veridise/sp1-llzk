use crate::CUDA_P3_EVAL_F_CONSTANTS;
use crate::{
    symbolic_expr_ef::SymbolicExprEF, symbolic_expr_f::SymbolicExprF,
    symbolic_var_ef::SymbolicVarEF, symbolic_var_f::SymbolicVarF, EF, F,
};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Default)]
struct CodegenInner {
    current_struct_name: String,
    value_counter: u32,
    exprf_to_value: HashMap<SymbolicExprF, Value>,
    expref_to_value: HashMap<SymbolicExprEF, Value>,
}

/// Implements the logic for generating llzk IR from an Air.
#[derive(Default)]
pub struct Codegen {
    inner: RefCell<CodegenInner>,
}

/// Final output type of the llzk code generator.
pub type CodegenOutput = ();

pub type VarF = SymbolicVarF;
pub type VarEF = SymbolicVarEF;
pub type ExprF = SymbolicExprF;
pub type ExprEF = SymbolicExprEF;
pub type Felt = F;
pub type ExtF = EF;

pub enum BinOps {
    Add,
    Sub,
    Mul,
}

impl std::fmt::Display for BinOps {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BinOps::Add => write!(f, "add"),
            BinOps::Sub => write!(f, "sub"),
            BinOps::Mul => write!(f, "mul"),
        }
    }
}

pub enum UnOps {
    Neg,
}

impl std::fmt::Display for UnOps {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            UnOps::Neg => write!(f, "neg"),
        }
    }
}

#[derive(PartialEq, Clone)]
pub enum ValueType {
    Felt,
    ExtFelt,
}

impl std::fmt::Display for ValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ValueType::Felt => write!(f, "!llzk.felt"),
            ValueType::ExtFelt => write!(f, "!llzk.array<4 x !llzk.felt>"),
        }
    }
}

/// Opaque struct that represents a IR value in llzk.
#[derive(Clone)]
pub struct Value {
    value_type: ValueType,
    /// A string that represents faux llzk IR, used for debugging at this stage
    psuedo_ir: String,
    psuedo_ssa_var: u32,
}

impl Value {
    fn print(self) -> Self {
        println!("{}", self);
        self
    }

    pub fn felt(n: u32, ir: String) -> Self {
        Self { psuedo_ssa_var: n, psuedo_ir: ir, value_type: ValueType::Felt }.print()
    }

    pub fn extfelt(n: u32, ir: String) -> Self {
        Self { psuedo_ssa_var: n, psuedo_ir: ir, value_type: ValueType::ExtFelt }.print()
    }

    pub fn is_felt(&self) -> bool {
        self.value_type == ValueType::Felt
    }

    pub fn is_extfelt(&self) -> bool {
        self.value_type == ValueType::ExtFelt
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "    %{} = {} {}", self.psuedo_ssa_var, self.psuedo_ir, self.value_type)
    }
}

impl Codegen {
    pub fn new() -> Self {
        Default::default()
    }

    fn next_pseudo_ssa(&self) -> u32 {
        let n = self.inner.borrow().value_counter;
        self.inner.borrow_mut().value_counter += 1;
        n
    }

    /// Initializes a LLZK struct with the given name and sets it as the struct where IR is going
    /// to be generated
    pub fn initialize_struct(&self, name: &str) {
        self.inner.borrow_mut().current_struct_name = name.to_string();

        println!("llzk.struct @{name} {{\n  func @constrain(%self: !llzk.struct<@{name}>) {{");
    }

    /// Returns the final output of generating LLZK. This output does not depend on the internal
    /// state of the code generator and can be safely used after its lifetime ends.
    pub fn extract_output(&self) -> CodegenOutput {
        Default::default()
    }

    /// Generates a `llzk.constfelt` op with the given value
    pub fn const_f(&self, value: Felt) -> Value {
        Value::felt(self.next_pseudo_ssa(), format!("constfelt {value} :"))
    }

    /// Generates an operation representing an extended field element with the value zero.
    pub fn const_ef(&self, value: ExtF) -> Value {
        Value::extfelt(self.next_pseudo_ssa(), format!("constextfelt {value} :"))
    }

    /// Generates a `llzk.emit_eq` operation over two field elements.
    pub fn emit_eq(&self, lhs: Value, rhs: Value) {
        println!(
            "    emit_eq  %{}, %{} : {}, {}",
            lhs.psuedo_ssa_var, rhs.psuedo_ssa_var, lhs.value_type, rhs.value_type
        );
    }

    /// Returns the Value associated with the given expression. Panics if the value was not be
    /// found.
    pub fn get_f(&self, f: ExprF) -> Value {
        self.inner.borrow().exprf_to_value.get(&f).unwrap().clone()
    }

    /// Returns the Value associated with the given expression. Panics if the value was not be
    /// found.
    pub fn get_ef(&self, ef: ExprEF) -> Value {
        self.inner.borrow().expref_to_value.get(&ef).unwrap().clone()
    }

    /// Generates IR for loading the given variable
    pub fn load_var(&self, v: VarF) -> Value {
        match v {
            SymbolicVarF::Empty => Value::felt(self.next_pseudo_ssa(), "empty_var :".to_string()),
            SymbolicVarF::Constant(idx) => {
                self.const_f(CUDA_P3_EVAL_F_CONSTANTS.lock().unwrap()[idx as usize])
            }
            SymbolicVarF::PreprocessedLocal(idx) => Value::felt(
                self.next_pseudo_ssa(),
                format!("readf %self[@PreprocessedLocal{idx}] :"),
            ),
            SymbolicVarF::PreprocessedNext(idx) => Value::felt(
                self.next_pseudo_ssa(),
                format!("readf %self[@PreprocessedNext{idx}] :"),
            ),
            SymbolicVarF::MainLocal(idx) => {
                Value::felt(self.next_pseudo_ssa(), format!("readf %self[@MainLocal{idx}] :"))
            }
            SymbolicVarF::MainNext(idx) => {
                Value::felt(self.next_pseudo_ssa(), format!("readf %self[@MainNext{idx}] :"))
            }
            SymbolicVarF::IsFirstRow => {
                Value::felt(self.next_pseudo_ssa(), "first_row :".to_string())
            }
            SymbolicVarF::IsLastRow => {
                Value::felt(self.next_pseudo_ssa(), "last_row :".to_string())
            }
            SymbolicVarF::IsTransition => {
                Value::felt(self.next_pseudo_ssa(), "transition :".to_string())
            }
            SymbolicVarF::PublicValue(idx) => {
                Value::felt(self.next_pseudo_ssa(), format!("readf %self[@PublicValue{idx}] :"))
            }
            SymbolicVarF::GlobalCumulativeSum(idx) => Value::felt(
                self.next_pseudo_ssa(),
                format!("readf %self[@GlobalCumulativeSum{idx}] :"),
            ),
        }
    }

    /// Generates IR for loading the given variable
    pub fn load_var_ef(&self, v: VarEF) -> Value {
        match v {
            SymbolicVarEF::Empty => {
                Value::extfelt(self.next_pseudo_ssa(), "empty_var :".to_string())
            }
            SymbolicVarEF::PermutationLocal(idx) => Value::extfelt(
                self.next_pseudo_ssa(),
                format!("readf %self[@PermutationLocal{idx}] :"),
            ),
            SymbolicVarEF::PermutationNext(idx) => Value::extfelt(
                self.next_pseudo_ssa(),
                format!("readf %self[@PermutationNext{idx}] :"),
            ),
            SymbolicVarEF::PermutationChallenge(idx) => Value::extfelt(
                self.next_pseudo_ssa(),
                format!("readf %self[@PermutationChallenge{idx}] :"),
            ),
            SymbolicVarEF::CumulativeSum(idx) => Value::extfelt(
                self.next_pseudo_ssa(),
                format!("readf %self[@CumulativeSum{idx}] :"),
            ),
        }
    }

    /// Associates the IR defined by the given value to the output expression.
    pub fn assign_ef(&self, output: ExprEF, value: Value) {
        assert!(value.is_extfelt());
        self.inner.borrow_mut().expref_to_value.insert(output, value);
    }

    /// Associates the IR defined by the given value to the output expression.
    pub fn assign_f(&self, output: ExprF, value: Value) {
        assert!(value.is_felt());
        self.inner.borrow_mut().exprf_to_value.insert(output, value);
    }

    /// Generates IR representing the given operation and returns a Value with the result.
    pub fn binop(&self, op: BinOps, lhs: Value, rhs: Value) -> Value {
        let res = Value {
            psuedo_ir: format!(
                "{} %{}, %{} : ({}, {}) ->",
                op, lhs.psuedo_ssa_var, rhs.psuedo_ssa_var, lhs.value_type, rhs.value_type
            ),
            psuedo_ssa_var: self.next_pseudo_ssa(),
            value_type: lhs.value_type.clone(),
        }
        .print();
        if rhs.value_type != lhs.value_type {
            println!(
                "      Malformed IR in value %{}: incompatible types {} != {}",
                res.psuedo_ssa_var, rhs.value_type, lhs.value_type
            );
        }
        res
    }

    /// Generates IR representing the given operation and returns a Value with the result.
    pub fn unop(&self, op: UnOps, value: Value) -> Value {
        Value {
            psuedo_ir: format!("{} %{} : ({}) ->", op, value.psuedo_ssa_var, value.value_type),
            psuedo_ssa_var: self.next_pseudo_ssa(),
            value_type: value.value_type,
        }
        .print()
    }

    /// Generates IR for converting a Value of type Felt into a Value of type ExtFelt.
    pub fn f_to_ef(&self, value: Value) -> Value {
        let v = Value::extfelt(
            self.next_pseudo_ssa(),
            format!("felt_to_extfelt %{} : {},", value.psuedo_ssa_var, value.value_type),
        );

        assert!(value.is_felt());
        v
    }
}

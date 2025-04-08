use crate::{
    symbolic_expr_ef::SymbolicExprEF, symbolic_expr_f::SymbolicExprF,
    symbolic_var_ef::SymbolicVarEF, symbolic_var_f::SymbolicVarF, EF, F,
};

/// Implements the logic for generating llzk IR from an Air.
pub struct Codegen {}

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

pub enum UnOps {
    Neg,
}

#[derive(PartialEq)]
pub enum ValueType {
    Felt,
    ExtFelt,
}

/// Opaque struct that represents a IR value in llzk.
pub struct Value {
    value_type: ValueType,
}

impl Value {
    pub fn is_felt(&self) -> bool {
        self.value_type == ValueType::Felt
    }

    pub fn is_extfelt(&self) -> bool {
        self.value_type == ValueType::ExtFelt
    }
}

impl Codegen {
    pub fn new() -> Self {
        Self {}
    }

    /// Initializes a LLZK struct with the given name and sets it as the struct where IR is going
    /// to be generated
    pub fn initialize_struct(&self, name: &str) {
        todo!()
    }

    /// Returns the final output of generating LLZK. This output does not depend on the internal
    /// state of the code generator and can be safely used after its lifetime ends.
    pub fn extract_output(&self) -> CodegenOutput {
        Default::default()
    }

    /// Generates a `llzk.constfelt` op with the given value
    pub fn const_f(&self, value: Felt) -> Value {
        todo!()
    }

    /// Generates an operation representing an extended field element with the value zero.
    pub fn const_ef(&self, value: ExtF) -> Value {
        todo!()
    }

    /// Generates a `llzk.emit_eq` operation over two field elements.
    pub fn emit_eq(&self, lhs: Value, rhs: Value) {
        todo!()
    }

    /// Returns a expression associated with the given value.
    pub fn unwrap_f(&self, value: Value) -> ExprF {
        assert!(value.is_felt());
        todo!()
    }

    /// Returns a expression associated with the given value.
    pub fn unwrap_ef(&self, value: Value) -> ExprEF {
        assert!(value.is_extfelt());
        todo!()
    }

    /// Returns the Value associated with the given expression. Panics if the value was not be
    /// found.
    pub fn get_f(&self, f: ExprF) -> Value {
        todo!()
    }

    /// Returns the Value associated with the given expression. Panics if the value was not be
    /// found.
    pub fn get_ef(&self, ef: ExprEF) -> Value {
        todo!()
    }

    /// Generates IR for loading the given variable
    pub fn load_var(&self, v: VarF) -> Value {
        todo!()
    }

    /// Generates IR for loading the given variable
    pub fn load_var_ef(&self, v: VarEF) -> Value {
        todo!()
    }

    /// Associates the IR defined by the given value to the output expression.
    pub fn assign_ef(&self, output: ExprEF, value: Value) {
        assert!(value.is_extfelt());
        todo!()
    }

    /// Associates the IR defined by the given value to the output expression.
    pub fn assign_f(&self, output: ExprF, value: Value) {
        assert!(value.is_felt());
        todo!()
    }

    /// Generates IR representing the given operation and returns a Value with the result.
    pub fn binop(&self, op: BinOps, lhs: Value, rhs: Value) -> Value {
        todo!()
    }

    /// Generates IR representing the given operation and returns a Value with the result.
    pub fn unop(&self, op: UnOps, value: Value) -> Value {
        todo!()
    }

    /// Generates IR for converting a Value of type Felt into a Value of type ExtFelt.
    pub fn f_to_ef(&self, value: Value) -> Value {
        assert!(value.is_felt());
        todo!()
    }
}

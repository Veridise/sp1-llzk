use p3_air::Air;
use sp1_stark::{air::MachineAir, Chip};
use std::ops::Deref;

use super::value::FeltValue;
use super::{value::Value, CodegenOutput, Felt};
use super::{Args, CodegenBuilder, Symbol};

/// A wrapper struct used for type sefety around the initialization of the llzk struct
pub struct CodegenWithStruct<'a> {
    codegen: &'a Codegen,
}

impl Deref for CodegenWithStruct<'_> {
    type Target = Codegen;

    fn deref(&self) -> &Self::Target {
        self.codegen
    }
}

/// Implements the logic for generating llzk IR from an Air.
pub struct Codegen {}

impl Codegen {
    pub fn str_to_symbol(&self, _str: String) -> Symbol {
        todo!()
    }

    pub fn get_func_arg(&self, _arg: Args) -> Value {
        todo!()
    }

    /// Returns a singleton instance of the code generator
    pub fn instance() -> Self {
        todo!()
    }

    /// Initializes a LLZK struct with the given chip and sets it as the struct where IR is going
    /// to be generated
    pub fn initialize<'a, A>(&self, chip: &Chip<Felt, A>) -> CodegenWithStruct
    where
        A: Air<CodegenBuilder<'a>> + MachineAir<Felt>,
    {
        todo!();
        CodegenWithStruct { codegen: self }
    }

    /// Returns the final output of generating LLZK. This output does not depend on the internal
    /// state of the code generator and can be safely used after its lifetime ends.
    pub fn extract_output(&self) -> CodegenOutput {
        Default::default()
    }

    /// Returns a constant value of index type
    pub fn const_index(&self, _idx: usize) -> Value {
        todo!()
    }

    /// Returns the value inside an array at the given index
    pub fn read_array(&self, _arr: Value, _idx: Value) -> Value {
        todo!()
    }

    /// Creates a literal array of values
    pub fn literal_array<I: Into<Value>>(&self, _values: &[I], _sizes: &[usize]) -> Value {
        todo!()
    }

    /// Returns the value contained in a field of the current struct
    pub fn read_self_field(&self, _name: Symbol) -> Value {
        todo!()
    }

    /// Generates a `llzk.constfelt` op with the given value
    pub fn const_felt(&self, _value: Felt) -> Value {
        todo!()
    }

    /// Generates a `llzk.emit_eq` operation over two field elements.
    pub fn emit_eq(&self, _lhs: Value, _rhs: Value) {
        todo!()
    }

    /// Generates an add operation between two felts.
    pub fn felt_add(&self, _lhs: FeltValue, _rhs: FeltValue) -> FeltValue {
        todo!()
    }

    /// Generates an sub operation between two felts.
    pub fn felt_sub(&self, _lhs: FeltValue, _rhs: FeltValue) -> FeltValue {
        todo!()
    }

    /// Generates an mul operation between two felts.
    pub fn felt_mul(&self, _lhs: FeltValue, _rhs: FeltValue) -> FeltValue {
        todo!()
    }

    /// Generates an neg operation over a felt.
    pub fn felt_neg(&self, _value: FeltValue) -> FeltValue {
        todo!()
    }

    /// Given a slice moves its memory to a location linked to the lifetime of Codegen
    pub fn manage<'a, T>(&self, _slice: &[T]) -> &'a [T] {
        todo!()
    }
}

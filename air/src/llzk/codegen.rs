use p3_air::Air;
use sp1_stark::{air::MachineAir, Chip};
use std::ops::Deref;

use super::value::FeltValue;
use super::{value::Value, CodegenOutput, Felt};
use super::{Args, CodegenBuilder, Symbol};
use llzk_bridge::CodegenState;
use std::ffi::c_void;

/// A wrapper struct used for type safety around the initialization of the llzk struct
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
pub struct Codegen {
    inner: *mut CodegenState,
}

impl Codegen {
    fn assert_has_struct(&self, msg: &str) {
        if unsafe { llzk_bridge::has_struct(self.inner) } == 0 {
            panic!("{}", msg);
        }
    }

    pub fn str_to_symbol(&self, str: &str) -> Symbol {
        let mut copy = String::from(str);
        unsafe {
            llzk_bridge::create_symbol(
                self.inner,
                copy.as_mut_ptr(),
                copy.len().try_into().unwrap(),
            )
        }
    }

    pub fn get_func_arg(&self, arg: Args) -> Value {
        self.assert_has_struct("Cannot get function argument if there is no struct");
        unsafe { llzk_bridge::get_func_argument(self.inner, arg as u8) }
    }

    /// Returns a singleton instance of the code generator
    pub fn instance() -> Self {
        let inner = unsafe { llzk_bridge::get_state() };
        if inner == std::ptr::null_mut() {
            panic!("Missing internal state");
        }
        Self { inner }
    }

    /// Initializes a LLZK struct with the given chip and sets it as the struct where IR is going
    /// to be generated
    pub fn initialize<'a, A>(&self, chip: &Chip<Felt, A>) -> CodegenWithStruct
    where
        A: Air<CodegenBuilder<'a>> + MachineAir<Felt>,
    {
        if unsafe { llzk_bridge::has_struct(self.inner) } != 0 {
            panic!("Cannot initialize a struct without commiting the previous one!");
        }
        let mut name = chip.name().clone();
        let spec = llzk_bridge::StructSpec {
            name: name.as_mut_ptr(),
            namelen: name.len().try_into().unwrap(),
        };
        unsafe {
            llzk_bridge::initialize_struct(self.inner, spec);
        }
        CodegenWithStruct { codegen: self }
    }

    /// Returns the final output of generating LLZK. This output does not depend on the internal
    /// state of the code generator and can be safely used after its lifetime ends.
    pub fn extract_output(&self) -> CodegenOutput {
        let mut output = CodegenOutput {
            bytes: std::ptr::null_mut(),
            size: 0,
            format: super::OutputFormats::Assembly, // Only assembly IR for now.
        };
        if unsafe { llzk_bridge::has_struct(self.inner) } != 0 {
            let res = unsafe {
                llzk_bridge::commit_struct(
                    self.inner,
                    &mut output.bytes,
                    &mut output.size,
                    llzk_bridge::OutputFormat_OF_Assembly,
                )
            };
            if res > 0 {
                panic!("Failed to generate the final output");
            }
        }
        output
    }

    /// Releases the memory of a codegen output.
    pub fn release_output(&self, output: &mut CodegenOutput) {
        unsafe {
            llzk_bridge::release_output_buffer(self.inner, &mut output.bytes);
        }
    }

    /// Returns a constant value of index type
    pub fn const_index(&self, idx: usize) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe { llzk_bridge::create_const_index(self.inner, idx.try_into().unwrap()) }
    }

    /// Returns the value inside an array at the given index
    pub fn read_array(&self, arr: Value, idx: Value) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe { llzk_bridge::create_read_array(self.inner, arr, idx) }
    }

    /// Creates a literal array of values
    pub fn literal_array<I: Into<Value> + Clone>(&self, values: &[I], sizes: &[usize]) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        let mut values = values.iter().map(|v| v.clone().into()).collect::<Vec<Value>>();
        let mut sizes = sizes.into_iter().map(|s| *s as u64).collect::<Vec<u64>>();
        unsafe {
            llzk_bridge::create_array(
                self.inner,
                values.as_mut_ptr(),
                values.len().try_into().unwrap(),
                sizes.as_mut_ptr(),
                sizes.len().try_into().unwrap(),
            )
        }
    }

    /// Returns the value contained in a field of the current struct
    pub fn read_self_field(&self, name: Symbol) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe {
            let struct_self = llzk_bridge::get_self_value(self.inner);
            llzk_bridge::create_field_read(self.inner, struct_self, name)
        }
    }

    /// Generates a `llzk.constfelt` op with the given value
    pub fn const_felt(&self, value: Felt) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe { llzk_bridge::create_const_felt(self.inner, todo!()) }
    }

    /// Generates a `llzk.emit_eq` operation over two field elements.
    pub fn emit_eq(&self, lhs: Value, rhs: Value) {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe { llzk_bridge::create_emit_eq(self.inner, lhs, rhs) }
    }

    /// Generates an add operation between two felts.
    pub fn felt_add(&self, lhs: FeltValue, rhs: FeltValue) -> FeltValue {
        self.assert_has_struct("Cannot create op without a target struct");
        FeltValue { inner: unsafe { llzk_bridge::create_felt_add(self.inner, *lhs, *rhs) } }
    }

    /// Generates an sub operation between two felts.
    pub fn felt_sub(&self, lhs: FeltValue, rhs: FeltValue) -> FeltValue {
        self.assert_has_struct("Cannot create op without a target struct");
        FeltValue { inner: unsafe { llzk_bridge::create_felt_sub(self.inner, *lhs, *rhs) } }
    }

    /// Generates an mul operation between two felts.
    pub fn felt_mul(&self, lhs: FeltValue, rhs: FeltValue) -> FeltValue {
        self.assert_has_struct("Cannot create op without a target struct");
        FeltValue { inner: unsafe { llzk_bridge::create_felt_mul(self.inner, *lhs, *rhs) } }
    }

    /// Generates an neg operation over a felt.
    pub fn felt_neg(&self, value: FeltValue) -> FeltValue {
        self.assert_has_struct("Cannot create op without a target struct");
        FeltValue { inner: unsafe { llzk_bridge::create_felt_neg(self.inner, *value) } }
    }

    /// Given a slice moves its memory to a location linked to the lifetime of Codegen
    pub fn manage<'a, T>(&self, slice: &[T]) -> &'a [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                llzk_bridge::manage_data_lifetime(
                    self.inner,
                    slice.as_ptr() as *const c_void,
                    slice.len().try_into().unwrap(),
                ) as *mut T,
                slice.len(),
            )
        }
    }
}

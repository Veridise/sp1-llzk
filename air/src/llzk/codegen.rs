use p3_air::Air;
use sp1_stark::{air::MachineAir, Chip};
use std::ops::Deref;

use super::value::FeltValue;
use super::{value::Value, CodegenOutput, Felt};
use super::{Args, CodegenBuilder, Symbol};
use crate::llzk::ExtFeltValue;
use crate::llzk::Type;
use crate::llzk::EXT_FELT_DEGREE;
use crate::llzk::PERM_CHALLENGES_COUNT;
use llzk_bridge::CodegenState;
use p3_air::BaseAir;
use p3_field::AbstractField;
use sp1_stark::PROOF_MAX_NUM_PVS;
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

    /// Creates a symbol with the contents of the given str managed by Codegen.
    /// Internally points to a MLIR StringAttr's content.
    pub fn str_to_symbol(&self, str: &str) -> Symbol {
        unsafe { llzk_bridge::create_symbol(self.inner, str.as_ptr() as *const i8, str.len()) }
    }

    /// Returns a Value for the given arguments of the constrain function.
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
    pub fn initialize<'a, A>(&self, chip: &Chip<Felt, A>, n_inputs: usize) -> CodegenWithStruct
    where
        A: Air<CodegenBuilder<'a>> + MachineAir<Felt>,
    {
        if unsafe { llzk_bridge::has_struct(self.inner) } != 0 {
            panic!("Cannot initialize a struct without commiting the previous one!");
        }
        let name = chip.name();
        let spec = llzk_bridge::StructSpec {
            name: name.as_str().into(),
            n_inputs,
            n_outputs: chip.width() - n_inputs,
            n_preprocessed: chip.preprocessed_width(),
            n_permutations: chip.permutation_width(),
            n_permutation_challenges: PERM_CHALLENGES_COUNT,
            n_public_values: PROOF_MAX_NUM_PVS,
            global_cumulative_sum_total: 14,
            extfelt_degree: EXT_FELT_DEGREE as i64,
        };
        unsafe {
            llzk_bridge::initialize_struct(self.inner, spec);
        }
        CodegenWithStruct { codegen: self }
    }

    /// Returns the final output of generating LLZK. This output does not depend on the internal
    /// state of the code generator and can be safely used after its lifetime ends.
    pub fn extract_output(&self, format: super::OutputFormats) -> CodegenOutput {
        let mut output =
            CodegenOutput { bytes: std::ptr::null_mut(), size: 0, format: format.clone() };
        if unsafe { llzk_bridge::has_struct(self.inner) } != 0 {
            let res = unsafe {
                llzk_bridge::commit_struct(
                    self.inner,
                    &mut output.bytes,
                    &mut output.size,
                    match format {
                        super::OutputFormats::Assembly => llzk_bridge::OutputFormat_OF_Assembly,
                        super::OutputFormats::Bytecode => todo!(),
                        super::OutputFormats::Picus => llzk_bridge::OutputFormat_OF_Picus,
                    },
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
            llzk_bridge::release_output_buffer(self.inner, output.bytes);
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
    pub fn literal_array<I: Into<Value> + Clone>(&self, values: &[I], sizes: &[i64]) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        let values = values.iter().map(|v| v.clone().into()).collect::<Vec<Value>>();
        unsafe {
            llzk_bridge::create_array(
                self.inner,
                values.as_ptr(),
                values.len(),
                sizes.as_ptr(),
                sizes.len(),
            )
        }
    }

    /// Returns the type representing felts.
    pub fn get_felt_type(&self) -> Type {
        unsafe { llzk_bridge::get_felt_type(self.inner) }
    }

    /// Returns the value contained in a field of the current struct
    pub fn read_self_field(&self, name: Symbol, field_type: Type) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe {
            let struct_self = llzk_bridge::get_self_value(self.inner);
            llzk_bridge::create_field_read(self.inner, struct_self, name, field_type)
        }
    }

    /// Generates a `llzk.constfelt` op with the given value
    pub fn const_felt(&self, value: Felt) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe { llzk_bridge::create_const_felt(self.inner, value.to_string().as_str().into()) }
    }

    /// Generates a `llzk.emit_eq` operation over two field elements.
    pub fn emit_eq(&self, lhs: Value, rhs: Value) {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe { llzk_bridge::create_emit_eq(self.inner, lhs, rhs) }
    }

    /// Generates a `llzk.emit_in` operation over a field element and an array of field elements.
    pub fn emit_in(&self, lhs: Value, rhs: Value) {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe { llzk_bridge::create_emit_in(self.inner, lhs, rhs) }
    }

    /// Returns an array of field element values that contain all the numbers from 0 to 255.
    pub fn get_8bit_range(&self) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe { llzk_bridge::get_8bit_range(self.inner) }
    }

    /// Returns an array of field element values that contain all the numbers from 0 to 65535.
    pub fn get_16bit_range(&self) -> Value {
        self.assert_has_struct("Cannot create op without a target struct");
        unsafe { llzk_bridge::get_16bit_range(self.inner) }
    }

    /// If the given Value is a constant field element returns it, None otherwise.
    pub fn get_const_felt_from_value(&self, value: Value) -> Option<Felt> {
        if unsafe { llzk_bridge::value_is_constfelt(self.inner, value) } == 0 {
            return None;
        }
        Some(Felt::from_canonical_u64(
            unsafe { llzk_bridge::extract_constfelt(self.inner, value) } as u64
        ))
    }

    /// Generates an add operation between two felts.
    pub fn felt_add(&self, lhs: FeltValue, rhs: FeltValue) -> FeltValue {
        self.assert_has_struct("Cannot create op without a target struct");
        FeltValue::from(unsafe { llzk_bridge::create_felt_add(self.inner, lhs.into(), rhs.into()) })
    }

    /// Generates an sub operation between two felts.
    pub fn felt_sub(&self, lhs: FeltValue, rhs: FeltValue) -> FeltValue {
        self.assert_has_struct("Cannot create op without a target struct");
        FeltValue::from(unsafe { llzk_bridge::create_felt_sub(self.inner, lhs.into(), rhs.into()) })
    }

    /// Generates an mul operation between two felts.
    pub fn felt_mul(&self, lhs: FeltValue, rhs: FeltValue) -> FeltValue {
        self.assert_has_struct("Cannot create op without a target struct");
        FeltValue::from(unsafe { llzk_bridge::create_felt_mul(self.inner, lhs.into(), rhs.into()) })
    }

    /// Generates an neg operation over a felt.
    pub fn felt_neg(&self, value: FeltValue) -> FeltValue {
        self.assert_has_struct("Cannot create op without a target struct");
        FeltValue::from(unsafe { llzk_bridge::create_felt_neg(self.inner, value.into()) })
    }

    /// Allocates a piece of memory that is managed by an internall allocator tied to the lifetime
    /// of the current codegen.
    pub fn allocate<'a, T>(&self, len: usize) -> &'a mut [T] {
        unsafe {
            let addr: *mut c_void =
                llzk_bridge::allocate_chunk(self.inner, len * std::mem::size_of::<T>());
            std::slice::from_raw_parts_mut(addr as *mut T, len)
        }
    }
}

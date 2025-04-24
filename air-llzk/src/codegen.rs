use crate::air::CodegenBuilder;
//use crate::air::PERM_CHALLENGES_COUNT;
use crate::field::Felt;
use crate::output::{CodegenOutput, OutputFormats};
use crate::value::{FeltValue, Value};
use llzk_bridge::{CodegenState, Symbol, ValueType as Type};
use p3_air::Air;
use p3_air::BaseAir;
use p3_field::AbstractField;
use sp1_stark::PROOF_MAX_NUM_PVS;
use sp1_stark::{air::MachineAir, Chip};
use std::error::Error;
use std::ffi::c_void;
use std::fmt;
use std::ops::Deref;

/// Opaque type that represents a IR type in llzk.
//type Type = llzk_bridge::ValueType;

/// A reference to a name in MLIR.
//type Symbol = llzk_bridge::Symbol;

/// The order of the arguments in the constraint function
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Args {
    SelfArg = 0,
    Inputs,
    Preprocessed,
    PreprocessedNext,
    PublicValues,
    IsFirstRow,
    IsLastRow,
    IsTransition,
}

#[derive(Debug)]
pub enum CodegenError {
    UncommitedStruct,
    NoTargetStruct,
    OutputGenFailure(i32),
    Error(&'static str),
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CodegenError::NoTargetStruct => {
                write!(f, "Target struct not initialized")
            }
            CodegenError::Error(s) => {
                write!(f, "Codegen failed: {s}")
            }
            CodegenError::OutputGenFailure(code) => {
                write!(f, "Generating the final output failed with code {code}")
            }
            CodegenError::UncommitedStruct => {
                write!(f, "Cannot initialize a struct without commiting the previous one!")
            }
        }
    }
}

impl Error for CodegenError {}

pub type CodegenResult<T> = Result<T, CodegenError>;

/// A wrapper struct used for type safety around the initialization of the llzk struct
pub struct StructCodegen<'a> {
    codegen: &'a Codegen,
}

impl Deref for StructCodegen<'_> {
    type Target = Codegen;

    fn deref(&self) -> &Self::Target {
        self.codegen
    }
}

impl<'a> TryFrom<&'a Codegen> for StructCodegen<'a> {
    type Error = CodegenError;

    fn try_from(codegen: &'a Codegen) -> Result<Self, Self::Error> {
        if codegen.has_struct() {
            Ok(StructCodegen { codegen })
        } else {
            Err(CodegenError::NoTargetStruct)
        }
    }
}

/// Implements the logic for generating llzk IR from an Air.
pub struct Codegen {
    inner: *mut CodegenState,
}

impl Codegen {
    /// Returns wether the current codegen inner state has a target struct.
    pub fn has_struct(&self) -> bool {
        return unsafe { llzk_bridge::has_struct(self.inner) } != 0;
    }

    /// Creates a symbol with the contents of the given str managed by Codegen.
    /// Internally points to a MLIR StringAttr's content.
    pub fn str_to_symbol(&self, str: &str) -> Symbol {
        unsafe { llzk_bridge::create_symbol(self.inner, str.as_ptr() as *const i8, str.len()) }
    }

    /// Returns a singleton instance of the code generator
    pub fn instance() -> CodegenResult<Self> {
        let inner = unsafe { llzk_bridge::get_state() };
        if inner.is_null() {
            Err(CodegenError::Error("Missing internal state"))
        } else {
            Ok(Self { inner })
        }
    }

    /// Initializes a LLZK struct with the given chip and sets it as the struct where IR is going
    /// to be generated
    pub fn initialize<'a, A>(
        &self,
        chip: &Chip<Felt, A>,
        n_inputs: usize,
    ) -> CodegenResult<StructCodegen>
    where
        A: Air<CodegenBuilder<'a>> + MachineAir<Felt>,
    {
        if self.has_struct() {
            return Err(CodegenError::UncommitedStruct);
        }
        let name = chip.name();
        let spec = llzk_bridge::StructSpec {
            name: name.as_str().into(),
            n_inputs,
            n_outputs: chip.width() - n_inputs,
            n_preprocessed: chip.preprocessed_width(),
            n_public_values: PROOF_MAX_NUM_PVS,
        };
        unsafe {
            llzk_bridge::initialize_struct(self.inner, spec);
        }
        self.try_into()
    }

    /// Returns the final output of generating LLZK. This output does not depend on the internal
    /// state of the code generator and can be safely used after its lifetime ends.
    pub fn extract_output(&self, format: OutputFormats) -> CodegenResult<CodegenOutput> {
        let mut output = CodegenOutput::new(format.clone());
        if unsafe { llzk_bridge::has_struct(self.inner) } != 0 {
            let res = unsafe {
                llzk_bridge::commit_struct(
                    self.inner,
                    output.bytes_ref_mut(),
                    output.size_ref_mut(),
                    format.clone() as u32,
                    format.into(),
                )
            };
            if res > 0 {
                return Err(CodegenError::OutputGenFailure(res));
            }
        }
        Ok(output)
    }

    /// Erases the current target struct and cleans up the state.
    /// If the state does not have a target this method is a no-op
    pub fn reset(&self) {
        if self.has_struct() {
            unsafe {
                llzk_bridge::reset_target(self.inner);
            }
        }
    }

    /// Releases the memory of a codegen output.
    pub fn release_output(&self, output: &mut CodegenOutput) {
        unsafe {
            llzk_bridge::release_output_buffer(self.inner, output.bytes());
        }
    }

    /// Returns the type representing felts.
    pub fn get_felt_type(&self) -> Type {
        unsafe { llzk_bridge::get_felt_type(self.inner) }
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

    /// Allocates a piece of memory that is managed by an internal allocator tied to the lifetime
    /// of the current codegen.
    pub fn allocate<'a, T>(&self, len: usize) -> &'a mut [T] {
        unsafe {
            let addr: *mut c_void =
                llzk_bridge::allocate_chunk(self.inner, len * std::mem::size_of::<T>());
            std::slice::from_raw_parts_mut(addr as *mut T, len)
        }
    }

    fn check_has_struct(&self) -> CodegenResult<()> {
        if self.has_struct() {
            return Ok(());
        }
        Err(CodegenError::NoTargetStruct)
    }

    /// Returns a Value for the given arguments of the constrain function.
    pub fn get_func_arg(&self, arg: Args) -> CodegenResult<Value> {
        self.check_has_struct()?;
        Ok(unsafe { llzk_bridge::get_func_argument(self.inner, arg as u8) })
    }

    /// Returns a constant value of index type
    pub fn const_index(&self, idx: usize) -> CodegenResult<Value> {
        self.check_has_struct()?;
        Ok(unsafe { llzk_bridge::create_const_index(self.inner, idx.try_into().unwrap()) })
    }

    /// Returns the value inside an array at the given index
    pub fn read_array(&self, arr: Value, idx: Value) -> CodegenResult<Value> {
        self.check_has_struct()?;
        Ok(unsafe { llzk_bridge::create_read_array(self.inner, arr, idx) })
    }

    /// Creates a literal array of values
    pub fn literal_array<I: Into<Value> + Clone>(
        &self,
        values: &[I],
        sizes: &[i64],
    ) -> CodegenResult<Value> {
        self.check_has_struct()?;
        let values = values.iter().map(|v| v.clone().into()).collect::<Vec<Value>>();
        Ok(unsafe {
            llzk_bridge::create_array(
                self.inner,
                values.as_ptr(),
                values.len(),
                sizes.as_ptr(),
                sizes.len(),
            )
        })
    }

    /// Returns the value contained in a field of the current struct
    pub fn read_self_field(&self, name: Symbol, field_type: Type) -> CodegenResult<Value> {
        self.check_has_struct()?;
        Ok(unsafe {
            let struct_self = llzk_bridge::get_self_value(self.inner);
            llzk_bridge::create_field_read(self.inner, struct_self, name, field_type)
        })
    }

    /// Generates a `llzk.constfelt` op with the given value
    pub fn const_felt(&self, value: Felt) -> CodegenResult<Value> {
        self.check_has_struct()?;
        Ok(unsafe { llzk_bridge::create_const_felt(self.inner, value.to_string().as_str().into()) })
    }

    /// Generates a `llzk.emit_eq` operation over two field elements.
    pub fn emit_eq(&self, lhs: Value, rhs: Value) -> CodegenResult<()> {
        self.check_has_struct()?;
        unsafe { llzk_bridge::create_emit_eq(self.inner, lhs, rhs) };
        Ok(())
    }

    /// Generates a `llzk.emit_in` operation over a field element and an array of field elements.
    pub fn emit_in(&self, lhs: Value, rhs: Value) -> CodegenResult<()> {
        self.check_has_struct()?;
        unsafe { llzk_bridge::create_emit_in(self.inner, lhs, rhs) };
        Ok(())
    }

    /// Returns an array of field element values that contain all the numbers from 0 to 255.
    pub fn get_8bit_range(&self) -> CodegenResult<Value> {
        self.check_has_struct()?;
        Ok(unsafe { llzk_bridge::get_8bit_range(self.inner) })
    }

    /// Returns an array of field element values that contain all the numbers from 0 to 65535.
    pub fn get_16bit_range(&self) -> CodegenResult<Value> {
        self.check_has_struct()?;
        Ok(unsafe { llzk_bridge::get_16bit_range(self.inner) })
    }

    /// Generates an add operation between two felts.
    pub fn felt_add(&self, lhs: FeltValue, rhs: FeltValue) -> CodegenResult<FeltValue> {
        self.check_has_struct()?;
        Ok(FeltValue::from(unsafe {
            llzk_bridge::create_felt_add(self.inner, lhs.into(), rhs.into())
        }))
    }

    /// Generates an sub operation between two felts.
    pub fn felt_sub(&self, lhs: FeltValue, rhs: FeltValue) -> CodegenResult<FeltValue> {
        self.check_has_struct()?;
        Ok(FeltValue::from(unsafe {
            llzk_bridge::create_felt_sub(self.inner, lhs.into(), rhs.into())
        }))
    }

    /// Generates an mul operation between two felts.
    pub fn felt_mul(&self, lhs: FeltValue, rhs: FeltValue) -> CodegenResult<FeltValue> {
        self.check_has_struct()?;
        Ok(FeltValue::from(unsafe {
            llzk_bridge::create_felt_mul(self.inner, lhs.into(), rhs.into())
        }))
    }

    /// Generates an neg operation over a felt.
    pub fn felt_neg(&self, value: FeltValue) -> CodegenResult<FeltValue> {
        self.check_has_struct()?;
        Ok(FeltValue::from(unsafe { llzk_bridge::create_felt_neg(self.inner, value.into()) }))
    }
}

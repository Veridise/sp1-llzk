#ifndef _OPS_GEN_H
#define _OPS_GEN_H

#include "CodegenState.h"
#include "Symbol.h"
#include <mlir-c/IR.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef MlirValue Value;
typedef MlirType ValueType;

/// Returns llzk::FeltType
ValueType get_felt_type(CodegenState *);

/// Creates an arith.constant op of Index type.
Value create_const_index(CodegenState *, unsigned long);

/// Creates an llzk.constfelt op of Felt type.
Value create_const_felt(CodegenState *, MlirStringRef);

/// Creates a llzk.new_array op with the given values.
Value create_array(CodegenState *, const Value *, size_t, const int64_t *,
                   size_t);

/// Returns a value representing the n-th element of an array.
Value create_read_array(CodegenState *, Value, Value);

/// Returns a Value representing the current struct.
Value get_self_value(CodegenState *);

/// Returns a Value representing the contents of a field.
Value create_field_read(CodegenState *, Value, Symbol, ValueType);

/// Returns the n-th argument of the target struct's constrain function.
Value get_func_argument(CodegenState *, unsigned char);

/// Creates a llzk.emit_eq operation.
void create_emit_eq(CodegenState *, Value, Value);

/// Creates a llzk.emit_in operation.
void create_emit_in(CodegenState *, Value, Value);

/// Creates a llzk.feltadd operation.
Value create_felt_add(CodegenState *, Value, Value);

/// Creates a llzk.feltsub operation.
Value create_felt_sub(CodegenState *, Value, Value);

/// Creates a llzk.feltmul operation.
Value create_felt_mul(CodegenState *, Value, Value);

/// Creates a llzk.feltneg operation.
Value create_felt_neg(CodegenState *, Value);

/// Returns a Value pointing to an array of felt 0..255
Value get_8bit_range(CodegenState *);

/// Returns a Value pointing to an array of felt 0..65535
Value get_16bit_range(CodegenState *);

/// Returns true if the Value comes from a llzk.constfelt op.
int value_is_constfelt(CodegenState *, Value);

/// Returns the field element held by the given value as a 64 bit integer.
/// If the value is not a constant field element this function aborts.
int64_t extract_constfelt(CodegenState *, Value);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif

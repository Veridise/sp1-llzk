#ifndef _OPS_GEN_H
#define _OPS_GEN_H

#include "CodegenState.h"
#include "Symbol.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef int Value; // TODO Use the actual MLIR-C type

/// Creates an arith.constant op of Index type.
Value create_const_index(CodegenState *, unsigned long);

/// Creates an llzk.constfelt op of Felt type.
Value create_const_felt(CodegenState *, unsigned long);

/// Creates a llzk.new_array op with the given values.
Value create_array(CodegenState *, Value *, unsigned long, unsigned long *,
                   unsigned long);

/// Returns a value representing the n-th element of an array.
Value create_read_array(CodegenState *, Value, Value);

/// Returns a Value representing the current struct.
Value get_self_value(CodegenState *);

/// Returns a Value representing the contents of a field.
Value create_field_read(CodegenState *, Value, Symbol);

/// Returns the n-th argument of the target struct's constrain function.
Value get_func_argument(CodegenState *, unsigned char);

/// Creates a llzk.emit_eq operation.
void create_emit_eq(CodegenState *, Value, Value);

/// Creates a llzk.feltadd operation.
Value create_felt_add(CodegenState *, Value, Value);

/// Creates a llzk.feltsub operation.
Value create_felt_sub(CodegenState *, Value, Value);

/// Creates a llzk.feltmul operation.
Value create_felt_mul(CodegenState *, Value, Value);

/// Creates a llzk.feltneg operation.
Value create_felt_neg(CodegenState *, Value);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif

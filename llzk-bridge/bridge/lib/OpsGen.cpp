#include <OpsGen.h>

Value create_const_index(CodegenState *, unsigned long) { return {}; }

Value create_const_felt(CodegenState *, unsigned long) { return {}; }

Value create_array(CodegenState *, Value *, unsigned long, unsigned long *,
                   unsigned long) {
  return {};
}

Value create_read_array(CodegenState *, Value, Value) { return {}; }

Value get_self_value(CodegenState *) { return {}; }

Value create_field_read(CodegenState *, Value, Symbol) { return {}; }

Value get_func_argument(CodegenState *, unsigned char) { return {}; }

void create_emit_eq(CodegenState *, Value, Value) { return; }

Value create_felt_add(CodegenState *, Value, Value) { return {}; }

Value create_felt_sub(CodegenState *, Value, Value) { return {}; }

Value create_felt_mul(CodegenState *, Value, Value) { return {}; }

Value create_felt_neg(CodegenState *, Value) { return {}; }

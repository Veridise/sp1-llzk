#include "CodegenStateImpl.h"
#include <Symbol.h>
#include <cstddef>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/BuiltinAttributes.h>

Symbol create_symbol(CodegenState *state, const char *str, size_t len) {
  mlir::StringRef inputRef(str, len);
  return wrap(mlir::StringAttr::get(&unwrap(state).ctx, inputRef).getValue());
}

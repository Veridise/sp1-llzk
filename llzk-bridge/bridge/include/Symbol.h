#ifndef _SYMBOL_H
#define _SYMBOL_H

#include "CodegenState.h"
#include <mlir-c/Support.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef MlirStringRef Symbol;

/// Creates a StringAttr with the contents of the given string. Returns a
/// StringRef pointing to the content of the StringAttr.
Symbol create_symbol(CodegenState *, const char *, size_t);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif

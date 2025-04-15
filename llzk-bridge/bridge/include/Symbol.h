#ifndef _SYMBOL_H
#define _SYMBOL_H

#include "CodegenState.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// TODO: Use MLIR-C's FlatSymbolRef to represent this
typedef struct Symbol {
  unsigned dummy;
} Symbol;

Symbol create_symbol(CodegenState *, unsigned char *, unsigned long);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif

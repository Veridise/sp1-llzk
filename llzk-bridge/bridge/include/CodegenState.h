#ifndef _CODEGEN_STATE
#define _CODEGEN_STATE

#include <mlir-c/Support.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque type that holds the internal state to the codegen state.
typedef struct CodegenState {
  void *impl;
} CodegenState;

/// Information used for initializing a struct target.
typedef struct StructSpec {
  MlirStringRef name;
  size_t n_inputs, n_outputs, n_preprocessed, n_public_values;
} StructSpec;

/// Final output format of the IR.
enum OutputFormat { OF_Assembly, OF_Bytecode, OF_Picus };

typedef struct AssemblyFormatData {
  char dummy;
} AssemblyFormatData;

typedef struct BytecodeFormatData {
  char dummy;
} BytecodeFormatData;

typedef struct PicusFormatData {
  size_t prime;
} PicusFormatData;

/// Additional data required by the selected output format.
typedef union {
  AssemblyFormatData assembly;
  BytecodeFormatData bytecode;
  PicusFormatData picus;
} FormatData;

/// Returns the current codegen inner state. The first time this function
/// is called or the first time after calling `release_state` it will initialize
/// it automatically.
CodegenState *get_state();

/// Cleans up the codegen state.
void release_state(CodegenState *);

/// Initializes a struct that is going to be the target of the IR generation.
void initialize_struct(CodegenState *, StructSpec);

/// Drops the current target.
void reset_target(CodegenState *);

/// Returns 1 if the given codegen state has an initialized struct. 0 otherwise.
int has_struct(CodegenState *);

/// Writes the IR generated for the current struct into the output buffer.
/// The caller needs to free the pointer with `release_output_buffer()`.
int commit_struct(CodegenState *, unsigned char **, size_t *, enum OutputFormat,
                  FormatData);

/// Releases the memory used to store the IR output.
void release_output_buffer(CodegenState *, unsigned char *);

/// Allocates a chunk of bytes and ties it to the lifetime of the
/// state.
void *allocate_chunk(CodegenState *, size_t);

#ifdef __cplusplus
}
#endif

#endif

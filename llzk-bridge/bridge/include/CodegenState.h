#ifndef _CODEGEN_STATE
#define _CODEGEN_STATE

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque type that holds the internal state to the codegen state.
typedef struct CodegenState {
  void *impl;
} CodegenState;

/// Information used for initializing a struct target.
typedef struct StructSpec {
  unsigned char *name;
  int namelen;
} StructSpec;

/// Final output format of the IR.
enum OutputFormat { OF_Assembly, OF_Bytecode };

/// Returns the current codegen inner state. The first time this function
/// is called or the first time after calling `release_state` it will initialize
/// it automatically.
CodegenState *get_state();

/// Cleans up the codegen state.
void release_state(CodegenState *);

/// Initializes a struct that is going to be the target of the IR generation.
void initialize_struct(CodegenState *, StructSpec);

/// Returns 1 if the given codegen state has an initialized struct. 0 otherwise.
int has_struct(CodegenState *);

/// Writes the IR generated for the current struct into the output buffer.
/// The caller needs to free the pointer with `release_output_buffer()`.
int commit_struct(CodegenState *, unsigned char **, int *, enum OutputFormat);

/// Releases the memory used to store the IR output.
void release_output_buffer(CodegenState *, unsigned char **);

/// Creates a copy of a chunk of bytes and ties the copy to the lifetime of the
/// state.
void *manage_data_lifetime(CodegenState *, const void *, unsigned long);

#ifdef __cplusplus
}
#endif

#endif

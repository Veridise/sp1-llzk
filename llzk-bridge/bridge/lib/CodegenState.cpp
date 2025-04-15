#include <CodegenState.h>
#include <CodegenStateImpl.h>
#include <cassert>

namespace llzk {

static CodegenState sharedState = {.impl = nullptr};

CodegenStateImpl &CodegenStateImpl::fromWrapper(CodegenState *wrapper) {
  assert(wrapper);
  assert(wrapper->impl);
  return *reinterpret_cast<CodegenStateImpl *>(wrapper->impl);
}

} // namespace llzk

CodegenState *get_state() {
  if (!llzk::sharedState.impl) {
    auto impl = new llzk::CodegenStateImpl();
    llzk::sharedState.impl = reinterpret_cast<void *>(impl);
  }
  return &llzk::sharedState;
}

/// Cleans up the codegen state.
void release_state(CodegenState *state) {
  delete reinterpret_cast<llzk::CodegenStateImpl *>(state->impl);
  state->impl = nullptr;
}

/// Initializes a struct that is going to be the target of the IR generation.
void initialize_struct(CodegenState *, StructSpec) {}

/// Returns 1 if the given codegen state has an initialized struct. 0 otherwise.
int has_struct(CodegenState *) { return false; }

/// Writes the IR generated for the current struct into the output buffer.
/// The caller needs to free the pointer with `release_output_buffer()`.
int commit_struct(CodegenState *, unsigned char **, int *, OutputFormat) {
  return 0;
}

/// Releases the memory used to store the IR output.
void release_output_buffer(CodegenState *, unsigned char **) {}

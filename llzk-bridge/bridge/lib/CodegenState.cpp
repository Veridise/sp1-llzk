#include "CodegenStateImpl.h"
#include <CodegenState.h>
#include <cassert>
#include <llvm/Support/Debug.h>
#include <llzk/Dialect/LLZK/IR/Dialect.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/Types.h>
#include <mlir/CAPI/Support.h>
#include <string>

using namespace mlir;

namespace llzk {

static CodegenState sharedState = {.impl = nullptr};

CodegenStateImpl::CodegenStateImpl()
    : registry{dialects()}, ctx{registry}, builder{&ctx}, allocator{} {
  ctx.loadAllAvailableDialects();
  llvm::dbgs() << "Created a new codegen state\n";
  dump();
}

CodegenStateImpl &CodegenStateImpl::fromWrapper(CodegenState *wrapper) {
  assert(wrapper);
  assert(wrapper->impl);
  return *reinterpret_cast<CodegenStateImpl *>(wrapper->impl);
}

void CodegenStateImpl::dump() {
  llvm::interleave(registry.getDialectNames(),
                   llvm::dbgs() << "Loaded dialects:\n- ", "\n- ");
  llvm::dbgs() << "\n";

  llvm::dbgs() << "Current op: ";
  if (currentTarget) {
    currentTarget.dump();
  } else {
    llvm::dbgs() << "<<NULL>>";
  }
  llvm::dbgs() << "\n";
}

mlir::DialectRegistry dialects() {
  mlir::DialectRegistry registry;
  registry.insert<llzk::LLZKDialect>();
  return registry;
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

static void reset_target(ModuleOp op, OpBuilder &builder) {
  llvm::dbgs() << "reset_target()\n";
  mlir::MLIRContext *ctx = builder.getContext();
  if (op) {
    llvm::dbgs() << "erasing: " << op << "\n";
    op.erase();
  } else {
    llvm::dbgs() << "no op to remove\n";
  }
  llvm::dbgs() << "creating a new builder\n";
  builder = OpBuilder(ctx);
  llvm::dbgs() << "done resetting\n";
}

/// Initializes a struct that is going to be the target of the IR generation.
void initialize_struct(CodegenState *state, StructSpec spec) {
  llvm::dbgs() << "initialize_struct()\n";
  auto &builder = unwrap(state).builder;
  auto unk = builder.getUnknownLoc();
  // 1. Destroy the current module if any
  llvm::dbgs() << "1. Destroy the current module if any\n";
  reset_target(unwrap(state).currentTarget, builder);
  // 2. Create the module
  llvm::dbgs() << "2. Create the module\n";
  unwrap(state).currentTarget = builder.create<ModuleOp>(unk);
  unwrap(state).currentTarget->setAttr(
      llzk::LANG_ATTR_NAME,
      builder.getStringAttr(llzk::LLZKDialect::getDialectNamespace()));
  builder.setInsertionPointToStart(
      &unwrap(state).currentTarget.getBodyRegion().front());
  // 3. Create the support globals
  // TODO
  // 4. Create the struct
  llvm::dbgs() << "4. Create the struct\n";
  llvm::dbgs() << "Before:\n";
  unwrap(state).dump();
  auto newStruct = builder.create<llzk::StructDefOp>(
      unk, builder.getStringAttr(unwrap(spec.name)), builder.getArrayAttr({}));
  llvm::dbgs() << "After:\n";
  unwrap(state).dump();
  builder.setInsertionPointToStart(&newStruct.getBodyRegion().emplaceBlock());
  // 5. Create the output fields
  llvm::dbgs() << "5. Create the output fields\n";
  Twine o("output");
  Twine on("output_next");
  auto felt = llzk::FeltType::get(&unwrap(state).ctx);
  for (size_t i = 0; i < spec.n_outputs; i++) {
    builder.create<llzk::FieldDefOp>(unk, builder.getStringAttr(o + Twine(i)),
                                     felt);
    builder.create<llzk::FieldDefOp>(unk, builder.getStringAttr(on + Twine(i)),
                                     felt);
  }
  // 6. Create the constrain function with the required arguments
  llvm::dbgs()
      << "6. Create the constrain function with the required arguments\n";
  auto selfType = newStruct.getType();
  auto extfelt = llzk::ArrayType::get(felt, {spec.extfelt_degree});
#define FELT_ARR(n) llzk::ArrayType::get(felt, {static_cast<long long>(n)})
#define EXTFELT_ARR(n)                                                         \
  llzk::ArrayType::get(felt, {static_cast<long long>(n), spec.extfelt_degree})
  TypeRange funcInputs({
      /*SelfArg*/
      selfType,
      /*Inputs*/
      FELT_ARR(spec.n_inputs),
      /*InputsNext*/
      FELT_ARR(spec.n_inputs),
      /*Preprocessed*/
      FELT_ARR(spec.n_preprocessed),
      /*PreprocessedNext*/
      FELT_ARR(spec.n_preprocessed),
      /*Permutations*/
      EXTFELT_ARR(spec.n_permutations),
      /*PermutationsNext*/
      EXTFELT_ARR(spec.n_permutations),
      /*PublicValues*/
      FELT_ARR(spec.n_public_values),
      /*PermutationChallenges*/
      EXTFELT_ARR(spec.n_permutation_challenges),
      /*GlobalCumulativeSum*/
      FELT_ARR(spec.global_cumulative_sum_total),
      /*LocalCumulativeSum*/
      extfelt,
      /*IsFirstRow*/
      felt,
      /*IsLastRow*/
      felt,
      /*IsTransition*/
      felt,
  });
#undef FELT_ARR
#undef EXTFELT_ARR
  TypeRange funcOutputs;

  auto funcType =
      mlir::FunctionType::get(builder.getContext(), funcInputs, funcOutputs);

  auto func = builder.create<llzk::FuncOp>(
      unk, builder.getStringAttr("constrain"), funcType);
  auto &block = func.getRegion().emplaceBlock();
  mlir::SmallVector<Location> locs(funcType.getInputs().size(), unk);
  block.addArguments(funcType.getInputs(), locs);
  // 7. Create an OpBuilder that points to the beginning of the constrain
  // function.
  llvm::dbgs() << "7. Create an OpBuilder that points to the beginning of the "
                  "constrain function\n";
  builder.setInsertionPointToStart(&func.getRegion().front());
  unwrap(state).dump();
}

/// Returns 1 if the given codegen state has an initialized struct. 0 otherwise.
int has_struct(CodegenState *state) {
  return unwrap(state).currentTarget != nullptr;
}

static int dump_assembly(ModuleOp op, unsigned char **out, int *size) {
  if (!op)
    return 2;
  std::string s;
  llvm::raw_string_ostream ss(s);
  op.print(ss);
  *size = s.size();
  *out = reinterpret_cast<unsigned char *>(malloc(s.size()));
  if (!*out)
    return 2;
  memcpy(s.data(), *out, s.size());
  return 0;
}

/// Writes the IR generated for the current struct into the output buffer.
/// The caller needs to free the pointer with `release_output_buffer()`.
int commit_struct(CodegenState *state, unsigned char **out, int *size,
                  OutputFormat format) {
  if (!size)
    return 3;
  *size = -1;
  if (!out)
    return 3;
  int res = 0;
  switch (format) {
  case OF_Assembly:
    res = dump_assembly(unwrap(state).currentTarget, out, size);
    if (res != 0)
      return res;
    reset_target(unwrap(state).currentTarget, unwrap(state).builder);
    break;
  case OF_Bytecode:
    llvm::errs() << "Bytecode output is not supported yet\n";
    res = 1;
    break;
  }
  return res;
}

/// Releases the memory used to store the IR output.
void release_output_buffer(CodegenState *, unsigned char **buf) { free(buf); }

void *manage_data_lifetime(CodegenState *state, const void *buf, size_t len) {
  void *ptr = unwrap(state).allocator.Allocate(len, llvm::Align());
  if (ptr)
    memcpy(ptr, buf, len);
  return ptr;
}

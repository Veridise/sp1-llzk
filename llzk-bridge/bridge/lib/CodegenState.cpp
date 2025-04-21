#include "CodegenStateImpl.h"
#include "GlobalsNames.h"
#include "Utils.h"
#include <CodegenState.h>
#include <cassert>
#include <llvm/Support/Debug.h>
#include <llzk/Dialect/LLZK/IR/Attrs.h>
#include <llzk/Dialect/LLZK/IR/Dialect.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/Types.h>
#include <llzk/Dialect/LLZK/Util/AttributeHelper.h>
#include <mlir/CAPI/Support.h>
#include <string>

using namespace mlir;

namespace llzk {

static CodegenState sharedState = {.impl = nullptr};

static mlir::DialectRegistry dialects() {
  mlir::DialectRegistry registry;
  registry.insert<llzk::LLZKDialect>();
  return registry;
}

CodegenStateImpl::CodegenStateImpl()
    : registry{dialects()}, ctx{registry}, builder{&ctx}, allocator{} {
  ctx.loadAllAvailableDialects();
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
  mlir::MLIRContext *ctx = builder.getContext();
  if (op)
    op.erase();
  builder = OpBuilder(ctx);
}

static void create_range_global(CodegenState *state, unsigned bitsize,
                                mlir::StringRef name) {
  auto &builder = unwrap(state).builder;
  auto unk = builder.getUnknownLoc();
  mlir::SmallVector<mlir::Attribute> felts;
  felts.reserve(1 << bitsize);

  for (unsigned val = 0; val < (1 << bitsize); val++) {
    felts.push_back(
        llzk::FeltConstAttr::get(builder.getContext(), llzk::toAPInt(val)));
  }
  builder.create<llzk::GlobalDefOp>(unk, name, true,
                                    po2ArrayType(builder, bitsize),
                                    builder.getArrayAttr(felts));
}

/// Initializes a struct that is going to be the target of the IR generation.
void initialize_struct(CodegenState *state, StructSpec spec) {
  auto &builder = unwrap(state).builder;
  auto unk = builder.getUnknownLoc();
  // 1. Destroy the current module if any
  reset_target(unwrap(state).currentTarget, builder);
  // 2. Create the module
  unwrap(state).currentTarget = builder.create<ModuleOp>(unk);
  unwrap(state).currentTarget->setAttr(
      llzk::LANG_ATTR_NAME,
      builder.getStringAttr(llzk::LLZKDialect::getDialectNamespace()));
  builder.setInsertionPointToStart(
      &unwrap(state).currentTarget.getBodyRegion().front());
  // 3. Create the support globals
  create_range_global(state, 8, llzk::NAME_8BITRANGE);
  create_range_global(state, 16, llzk::NAME_16BITRANGE);
  // 4. Create the struct
  auto newStruct = builder.create<llzk::StructDefOp>(
      unk, builder.getStringAttr(unwrap(spec.name)), builder.getArrayAttr({}));
  builder.setInsertionPointToStart(&newStruct.getBodyRegion().emplaceBlock());
  // 5. Create the output fields
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
  builder.setInsertionPointToStart(&func.getRegion().front());
}

/// Returns 1 if the given codegen state has an initialized struct. 0 otherwise.
int has_struct(CodegenState *state) {
  return unwrap(state).currentTarget != nullptr;
}

static int dump_assembly(ModuleOp op, unsigned char **out, size_t *size) {
  if (!op)
    return 2;
  std::string s;
  llvm::raw_string_ostream ss(s);
  op.print(ss);
  *size = s.size();
  *out = reinterpret_cast<unsigned char *>(malloc(s.size()));
  if (!*out)
    return 2;
  memcpy(*out, s.data(), s.size());
  return 0;
}

/// Writes the IR generated for the current struct into the output buffer.
/// The caller needs to free the pointer with `release_output_buffer()`.
int commit_struct(CodegenState *state, unsigned char **out, size_t *size,
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
void release_output_buffer(CodegenState *, unsigned char *buf) { free(buf); }

/// Allocates a chunk of bytes and ties it to the lifetime of the
/// state.
void *allocate_chunk(CodegenState *state, size_t len) {
  return unwrap(state).allocator.Allocate(len, llvm::Align());
}

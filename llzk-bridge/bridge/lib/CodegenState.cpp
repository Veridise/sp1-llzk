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
#include <llzk/Target/Picus/Export.h>
#include <llzk/Target/Picus/Language/Circuit.h>
#include <mlir/Bytecode/BytecodeWriter.h>
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
    : registry{dialects()}, ctx{registry}, builder{&ctx} {
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
  if (_currentTarget) {
    _currentTarget->dump();
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

void reset_target(CodegenState *state) {
  ModuleOp op = unwrap(state).currentTarget();
  OpBuilder &builder = unwrap(state).builder;
  MLIRContext *ctx = builder.getContext();
  if (op) {
    unwrap(state).noTarget();
  }
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
  reset_target(state);

  // 2. Create the module
  unwrap(state).setTarget(builder.create<ModuleOp>(unk));
  unwrap(state).currentTarget()->setAttr(
      llzk::LANG_ATTR_NAME,
      builder.getStringAttr(llzk::LLZKDialect::getDialectNamespace()));
  builder.setInsertionPointToStart(
      &unwrap(state).currentTarget().getBodyRegion().front());

  // 3. Create the support globals
  create_range_global(state, 8, llzk::NAME_8BITRANGE);
  create_range_global(state, 16, llzk::NAME_16BITRANGE);

  // 4. Create the struct
  auto newStruct = builder.create<llzk::StructDefOp>(
      unk, builder.getStringAttr(unwrap(spec.name)), builder.getArrayAttr({}));
  builder.setInsertionPointToStart(&newStruct.getBodyRegion().emplaceBlock());

  // 5. Create the output fields
  auto felt = llzk::FeltType::get(&unwrap(state).ctx);
  for (size_t i = 0; i < spec.n_outputs; i++) {
    builder.create<llzk::FieldDefOp>(
        unk, builder.getStringAttr("output" + Twine(i)), felt);
  }

  // 6. Create the constrain function with the required arguments
  auto selfType = newStruct.getType();
#define FELT_ARR(n) llzk::ArrayType::get(felt, {static_cast<long long>(n)})
  TypeRange funcInputs({
      /*SelfArg*/
      selfType,
      /*Inputs*/
      FELT_ARR(spec.n_inputs),
      /*Preprocessed*/
      FELT_ARR(spec.n_preprocessed),
      /*PreprocessedNext*/
      FELT_ARR(spec.n_preprocessed),
      /*PublicValues*/
      FELT_ARR(spec.n_public_values),
      /*IsFirstRow*/
      felt,
      /*IsLastRow*/
      felt,
      /*IsTransition*/
      felt,
  });
#undef FELT_ARR
  TypeRange funcOutputs;

  auto funcType =
      mlir::FunctionType::get(builder.getContext(), funcInputs, funcOutputs);

  auto func = builder.create<llzk::FuncOp>(
      unk, builder.getStringAttr("constrain"), funcType);
  auto &block = func.getRegion().emplaceBlock();
  mlir::SmallVector<Location> locs(funcType.getInputs().size(), unk);
  block.addArguments(funcType.getInputs(), locs);
  builder.setInsertionPointToStart(&func.getRegion().front());
}

/// Returns 1 if the given codegen state has an initialized struct. 0 otherwise.
int has_struct(CodegenState *state) {
  return unwrap(state).currentTarget() != nullptr;
}

namespace {

struct OutputDataCtx {
  using ptr = unsigned char *;

  ptr *out;
  size_t *size;
  OutputFormat format;
  FormatData data;

  template <typename T, typename F> LogicalResult dump(const T &op, F f) {
    if (!op)
      return failure();

    auto s = dump_to_string(op, f);
    if (failed(s))
      return failure();
    auto ptr = allocate(s->size());
    if (failed(ptr))
      return failure();
    *out = *ptr;
    *size = s->size();
    memcpy(*out, s->data(), s->size());
    return success();
  }

  LogicalResult validate() {
    return failure(out == nullptr || size == nullptr);
  }

private:
  FailureOr<ptr> allocate(size_t size) {
    auto p = new unsigned char[size];
    if (!p)
      return failure();
    return p;
  }

  template <typename T, typename F>
  FailureOr<std::string> dump_to_string(const T &op, F f) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    if (failed(f(op, ss))) {
      return failure();
    }
    return s;
  }
};

} // namespace

static LogicalResult commit_struct_impl(ModuleOp op, OutputDataCtx ctx) {
  if (failed(ctx.validate()))
    return failure();
  switch (ctx.format) {
  case OF_Assembly:
    return ctx.dump(op, [](auto op, auto &os) {
      op->print(os);
      return success();
    });
  case OF_Picus: {
    auto prime =
        llvm::APInt(sizeof(ctx.data.picus.prime) << 3, ctx.data.picus.prime);
    auto picusCircuit = llzk::translateModuleToPicus(op, prime);
    return ctx.dump(picusCircuit, [](auto &op, auto &os) {
      op->print(os);
      return success();
    });
  }
  case OF_Bytecode:
    return ctx.dump(op, [](auto op, auto &os) {
      mlir::BytecodeWriterConfig config;
      return mlir::writeBytecodeToFile(op, os, config);
    });
  }
}

/// Writes the IR generated for the current struct into the output buffer.
/// The caller needs to free the pointer with `release_output_buffer()`.
int commit_struct(CodegenState *state, unsigned char **out, size_t *size,
                  OutputFormat format, FormatData data) {
  if (failed(commit_struct_impl(
          unwrap(state).currentTarget(),
          {.out = out, .size = size, .format = format, .data = data})))
    return 1;
  return 0;
}

/// Releases the memory used to store the IR output.
void release_output_buffer(CodegenState *, unsigned char *buf) { delete[] buf; }

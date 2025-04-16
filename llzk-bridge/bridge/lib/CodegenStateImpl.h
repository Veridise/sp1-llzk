#pragma once

#include <CodegenState.h>
#include <llvm/Support/Allocator.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace llzk {

struct CodegenStateImpl {
  static CodegenStateImpl &fromWrapper(CodegenState *);

  CodegenStateImpl() : ctx{}, builder{&ctx}, allocator{} {}

  mlir::MLIRContext ctx;
  mlir::ModuleOp currentTarget = nullptr;
  mlir::OpBuilder builder;
  llvm::BumpPtrAllocator allocator;
};

} // namespace llzk

static llzk::CodegenStateImpl &unwrap(CodegenState *ptr) {
  return llzk::CodegenStateImpl::fromWrapper(ptr);
}

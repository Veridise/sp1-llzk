#pragma once

#include <CodegenState.h>
#include <llvm/Support/Allocator.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

namespace llzk {

struct CodegenStateImpl {
  static CodegenStateImpl &fromWrapper(CodegenState *);

  CodegenStateImpl();

  mlir::DialectRegistry registry;
  mlir::MLIRContext ctx;
  mlir::OpBuilder builder;
  llvm::BumpPtrAllocator allocator;

  void dump();

  mlir::ModuleOp currentTarget() { return *_currentTarget; }
  void noTarget() { _currentTarget = nullptr; }
  void setTarget(mlir::ModuleOp op) { _currentTarget = op; }

private:
  mlir::OwningOpRef<mlir::ModuleOp> _currentTarget = nullptr;
};

} // namespace llzk

static llzk::CodegenStateImpl &unwrap(CodegenState *ptr) {
  return llzk::CodegenStateImpl::fromWrapper(ptr);
}

#pragma once

#include <llzk/Dialect/LLZK/IR/Types.h>
#include <mlir/IR/Builders.h>

inline mlir::Type po2ArrayType(mlir::OpBuilder &builder, unsigned bitsize) {
  return llzk::ArrayType::get(llzk::FeltType::get(builder.getContext()),
                              {1 << bitsize});
}

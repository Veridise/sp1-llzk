#include "CodegenStateImpl.h"
#include "GlobalsNames.h"
#include "Utils.h"
#include <OpsGen.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/Types.h>
#include <llzk/Dialect/LLZK/Util/AttributeHelper.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

ValueType get_felt_type(CodegenState *state) {
  return wrap(llzk::FeltType::get(&unwrap(state).ctx));
}

Value create_const_index(CodegenState *state, unsigned long value) {
  return wrap(unwrap(state)
                  .builder
                  .create<mlir::arith::ConstantIndexOp>(
                      unwrap(state).builder.getUnknownLoc(), value)
                  .getResult());
}

Value create_const_felt(CodegenState *state, MlirStringRef valueAsString) {
  return wrap(unwrap(state)
                  .builder
                  .create<llzk::FeltConstantOp>(
                      unwrap(state).builder.getUnknownLoc(),
                      llzk::FeltConstAttr::get(
                          &unwrap(state).ctx,
                          mlir::APInt(64, unwrap(valueAsString), 10)))
                  .getResult());
}

Value create_array(CodegenState *state, const Value *values, size_t valueCount,
                   const int64_t *dims, size_t dimCount) {
  llvm::ArrayRef<Value> ref(values, valueCount);
  llvm::SmallVector<mlir::Value> arrayValues =
      llvm::map_to_vector(ref, [](auto v) { return unwrap(v); });
  llvm::ArrayRef<int64_t> dimsRef(dims, dimCount);
  assert(!arrayValues.empty());
  auto innerType = arrayValues.front().getType();
  auto type = llzk::ArrayType::get(innerType, dimsRef);
  return wrap(unwrap(state)
                  .builder
                  .create<llzk::CreateArrayOp>(
                      unwrap(state).builder.getUnknownLoc(), type, arrayValues)
                  .getResult());
}

Value create_read_array(CodegenState *state, Value arr, Value index) {
  mlir::Value value = unwrap(arr);
  auto type = mlir::cast<llzk::ArrayType>(value.getType());
  if (type.getDimensionSizes().size() == 1) {
    return wrap(
        unwrap(state)
            .builder
            .create<llzk::ReadArrayOp>(unwrap(state).builder.getUnknownLoc(),
                                       value, mlir::ValueRange({unwrap(index)}))
            .getResult());
  }
  return wrap(unwrap(state)
                  .builder
                  .create<llzk::ExtractArrayOp>(
                      unwrap(state).builder.getUnknownLoc(), value,
                      mlir::ValueRange({unwrap(index)}))
                  .getResult());
}

Value get_self_value(CodegenState *state) {
  return get_func_argument(state, 0);
}

Value create_field_read(CodegenState *state, Value strct, Symbol field,
                        ValueType type) {
  auto op = unwrap(state).builder.create<llzk::FieldReadOp>(
      unwrap(state).builder.getUnknownLoc(), unwrap(type), unwrap(strct),
      unwrap(state).builder.getStringAttr(unwrap(field)));
  return wrap(op.getResult());
}

Value get_func_argument(CodegenState *state, unsigned char idx) {
  mlir::Block *currentBlock = unwrap(state).builder.getInsertionBlock();
  mlir::Operation *op = currentBlock->getParentOp();
  llzk::FuncOp func = nullptr;
  if (auto f = mlir::dyn_cast_if_present<llzk::FuncOp>(op)) {
    func = f;
  } else if (auto f = op->getParentOfType<llzk::FuncOp>()) {
    func = f;
  }
  assert(func);

  return wrap(func.getArgument(idx));
}

void create_emit_eq(CodegenState *state, Value lhs, Value rhs) {
  unwrap(state).builder.create<llzk::EmitEqualityOp>(
      unwrap(state).builder.getUnknownLoc(), unwrap(lhs), unwrap(rhs));
}

void create_emit_in(CodegenState *state, Value lhs, Value rhs) {
  unwrap(state).builder.create<llzk::EmitContainmentOp>(
      unwrap(state).builder.getUnknownLoc(), unwrap(lhs), unwrap(rhs));
}

static Value load_global(CodegenState *state, mlir::Type type,
                         mlir::StringRef name) {
  return wrap(unwrap(state)
                  .builder
                  .create<llzk::GlobalReadOp>(
                      unwrap(state).builder.getUnknownLoc(), type,
                      mlir::FlatSymbolRefAttr::get(
                          unwrap(state).builder.getStringAttr(name)))
                  .getResult());
}

Value get_8bit_range(CodegenState *state) {
  return load_global(state, po2ArrayType(unwrap(state).builder, 8),
                     llzk::NAME_8BITRANGE);
}

Value get_16bit_range(CodegenState *state) {
  return load_global(state, po2ArrayType(unwrap(state).builder, 16),
                     llzk::NAME_16BITRANGE);
}

template <typename Op>
Value create_bin_op(CodegenState *state, Value lhs, Value rhs) {
  return wrap(unwrap(state)
                  .builder
                  .create<Op>(unwrap(state).builder.getUnknownLoc(),
                              unwrap(lhs), unwrap(rhs))
                  .getResult());
}

Value create_felt_add(CodegenState *state, Value lhs, Value rhs) {
  return create_bin_op<llzk::AddFeltOp>(state, lhs, rhs);
}

Value create_felt_sub(CodegenState *state, Value lhs, Value rhs) {
  return create_bin_op<llzk::SubFeltOp>(state, lhs, rhs);
}

Value create_felt_mul(CodegenState *state, Value lhs, Value rhs) {
  return create_bin_op<llzk::MulFeltOp>(state, lhs, rhs);
}

Value create_felt_neg(CodegenState *state, Value val) {
  return wrap(unwrap(state)
                  .builder
                  .create<llzk::NegFeltOp>(
                      unwrap(state).builder.getUnknownLoc(), unwrap(val))
                  .getResult());
}

int value_is_constfelt(CodegenState *, Value value) {
  if (!unwrap(value).getDefiningOp())
    return false;
  return mlir::isa<llzk::FeltConstantOp>(unwrap(value).getDefiningOp());
}

int64_t extract_constfelt(CodegenState *state, Value value) {
  assert(unwrap(value).getDefiningOp());
  auto constFeltOp =
      mlir::cast<llzk::FeltConstantOp>(unwrap(value).getDefiningOp());
  return llzk::fromAPInt(constFeltOp.getValue().getValue());
}

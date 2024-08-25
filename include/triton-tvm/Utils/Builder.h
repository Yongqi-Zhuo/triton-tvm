#ifndef TRITON_TVM_UTILS_BUILDER_H
#define TRITON_TVM_UTILS_BUILDER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"

namespace mlir::tvm::utils {

inline arith::ConstantOp getConstantOpI32(OpBuilder &builder, Location loc,
                                          int32_t value) {
  return arith::ConstantOp::materialize(
      builder, builder.getI32IntegerAttr(value), builder.getI32Type(), loc);
}

inline arith::ConstantOp getConstantOpIndex(OpBuilder &builder, Location loc,
                                            int64_t value) {
  return arith::ConstantOp::materialize(builder, builder.getIndexAttr(value),
                                        builder.getIndexType(), loc);
}

inline SmallVector<Value> delinearizeIndex(OpBuilder &b, Location loc,
                                           Value linearIndex,
                                           ArrayRef<int64_t> strides) {
  unsigned numDims = strides.size();
  assert(numDims > 0 && "expected at least one dimension");
  assert(std::is_sorted(strides.rbegin(), strides.rend()) &&
         "strides must be sorted in descending order, i.e., row-major order");
  for (unsigned i = 1; i < numDims; ++i) {
    assert(strides[i - 1] % strides[i] == 0 &&
           "strides must be multiples of the next stride");
  }

  auto type = linearIndex.getType();

  SmallVector<Value> results;
  results.reserve(numDims);
  Value residual = linearIndex;
  for (auto stride : strides) {
    auto strideValue = arith::ConstantOp::materialize(
        b, b.getIntegerAttr(type, stride), type, loc);
    auto quotient = b.create<arith::DivUIOp>(loc, residual, strideValue);
    auto remainder = b.create<arith::RemUIOp>(loc, residual, strideValue);
    results.push_back(quotient);
    residual = remainder;
  }
  return results;
}

} // namespace mlir::tvm::utils

#endif

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

} // namespace mlir::tvm::utils

#endif

#include "triton-tvm/Dialect/TritonMemRef/TritonMemRef.h"

namespace mlir::ttm {

void TritonMemRefDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-tvm/Dialect/TritonMemRef/TritonMemRefOps.cpp.inc"
      >();
}

} // namespace mlir::ttm

#include "triton-tvm/Dialect/TritonMemRef/TritonMemRefDialect.cpp.inc"

#define GET_OP_CLASSES
#include "triton-tvm/Dialect/TritonMemRef/TritonMemRefOps.cpp.inc"

namespace mlir::ttm {

LogicalResult MemRefToPtrOp::verify() {
  return success(getMemRefType().getElementType() ==
                 getResult().getType().getPointeeType());
}

} // namespace mlir::ttm

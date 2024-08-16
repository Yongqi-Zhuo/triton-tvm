#include "triton-tvm/Dialect/TVM/IR/Dialect.h"

using namespace mlir;
using namespace mlir::tvm;

void TVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-tvm/Dialect/TVM/IR/Ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton-tvm/Dialect/TVM/IR/Types.cpp.inc"
      >();
}

#include "triton-tvm/Dialect/TVM/IR/Dialect.cpp.inc"

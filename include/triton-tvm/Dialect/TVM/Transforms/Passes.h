#ifndef MLIR_DIALECT_TVM_PASSES_H
#define MLIR_DIALECT_TVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::tvm {

#define GEN_PASS_DECL
#include "triton-tvm/Dialect/TVM/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-tvm/Dialect/TVM/Transforms/Passes.h.inc"

} // namespace mlir::tvm

#endif // MLIR_DIALECT_TVM_PASSES_H

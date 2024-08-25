#ifndef MLIR_DIALECT_TVM_IR_DIALECT_H
#define MLIR_DIALECT_TVM_IR_DIALECT_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::tvm {

constexpr inline char kAttrForKindName[] = "tvm.for_kind";
constexpr inline char kAttrForThreadName[] = "tvm.for_thread";

} // namespace mlir::tvm

#include "triton-tvm/Dialect/TVM/IR/Attributes.h.inc"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "triton-tvm/Dialect/TVM/IR/Ops.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-tvm/Dialect/TVM/IR/Types.h.inc"

namespace mlir::tvm {

// For convenience.
struct ForOp {
  // Forward params to scf::ForOp, and add attributes.
  static inline scf::ForOp create(
      OpBuilder &builder, Location loc, Value lowerBound, Value upperBound,
      ForKindAttr kind, std::optional<StringAttr> thread = std::nullopt,
      function_ref<void(OpBuilder &, Location, Value)> bodyBuilder = nullptr) {
    auto iterType = lowerBound.getType();
    auto c1 = arith::ConstantOp::materialize(
        builder, builder.getIntegerAttr(iterType, 1), iterType, loc);
    auto op = builder.create<scf::ForOp>(
        loc, lowerBound, upperBound, c1, std::nullopt,
        bodyBuilder ? [bodyBuilder](OpBuilder &builder, Location loc, Value inductionVar,
                      ValueRange) {
          bodyBuilder(builder, loc, inductionVar);
          builder.create<scf::YieldOp>(loc);
        } : scf::ForOp::BodyBuilderFn{});
    op->setAttr(builder.getStringAttr(kAttrForKindName), kind);
    if (thread) {
      op->setAttr(builder.getStringAttr(tvm::kAttrForThreadName), *thread);
    }
    return op;
  }
};

} // namespace mlir::tvm

#endif // MLIR_DIALECT_TVM_IR_DIALECT_H

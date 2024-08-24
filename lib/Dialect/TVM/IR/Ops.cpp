#include "mlir/Interfaces/FunctionImplementation.h"

#include "triton-tvm/Dialect/TVM/IR/Dialect.h"

using namespace mlir;
using namespace tvm;

#define GET_OP_CLASSES
#include "triton-tvm/Dialect/TVM/IR/Ops.cpp.inc"

void MatchBufferOp::build(OpBuilder &b, OperationState &result,
                          MemRefType resultType, Value source,
                          ArrayRef<OpFoldResult> sizes,
                          ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  result.addAttributes(attrs);
  build(b, result, resultType, source, dynamicSizes,
        b.getDenseI64ArrayAttr(staticSizes));
}

void BlockOp::build(OpBuilder &builder, OperationState &result,
                    function_ref<void(OpBuilder &, Location)> bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  if (bodyBuilder) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location);
  }
}

void InitOp::build(OpBuilder &builder, OperationState &result,
                   function_ref<void(OpBuilder &, Location)> bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  if (bodyBuilder) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location);
  }
}

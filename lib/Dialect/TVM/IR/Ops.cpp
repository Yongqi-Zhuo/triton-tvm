#include "triton-tvm/Dialect/TVM/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton-tvm/Dialect/TVM/IR/Ops.cpp.inc"

namespace mlir::tvm {

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

ReadOp BlockOp::getReadOp() {
  for (auto &op : *getBody())
    if (auto r = dyn_cast<ReadOp>(op))
      return r;
  return nullptr;
}

WriteOp BlockOp::getWriteOp() {
  for (auto &op : *getBody())
    if (auto w = dyn_cast<WriteOp>(op))
      return w;
  return nullptr;
}

AssignOp BlockOp::getAssignOp() {
  for (auto &op : *getBody())
    if (auto a = dyn_cast<AssignOp>(op))
      return a;
  return nullptr;
}

InitOp BlockOp::getInitOp() {
  for (auto &op : *getBody())
    if (auto i = dyn_cast<InitOp>(op))
      return i;
  return nullptr;
}

BlockOp AssignOp::getBlockOp() {
  auto *parent = getOperation()->getParentOp();
  if (isa<InitOp>(parent)) {
    parent = parent->getParentOp();
  }
  return cast<BlockOp>(parent);
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

} // namespace mlir::tvm

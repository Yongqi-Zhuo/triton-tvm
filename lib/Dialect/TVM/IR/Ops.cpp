#include "mlir/Interfaces/FunctionImplementation.h"

#include "triton-tvm/Dialect/TVM/IR/Dialect.h"

using namespace mlir;
using namespace tvm;

#define GET_OP_CLASSES
#include "triton-tvm/Dialect/TVM/IR/Ops.cpp.inc"

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

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

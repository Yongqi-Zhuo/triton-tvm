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

void ForOp::build(
    OpBuilder &builder, OperationState &result, IntegerAttr start,
    IntegerAttr stop, ForKindAttr kind, std::optional<StringAttr> thread,
    function_ref<void(OpBuilder &, Location, Value)> bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute("start", start);
  result.addAttribute("stop", stop);
  result.addAttribute("kind", kind);
  if (thread)
    result.addAttribute("thread", *thread);

  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  bodyBlock->addArgument(builder.getIndexType(), result.location);

  if (bodyBuilder) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock->getArgument(0));
  }
}

void ForOp::print(OpAsmPrinter &p) {
  p << ' ' << getInductionVar() << " = " << getKind();
  if (getThread()) {
    p << '(';
    p.printString(*getThread());
    p << ')';
  }
  p << ' ' << getStart() << " to " << getStop() << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"start", "stop", "kind", "thread"});
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parse the induction variable followed by '='.
  OpAsmParser::Argument inductionVariable;
  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual())
    return failure();
  inductionVariable.type = builder.getIndexType();

  // Parse kind.
  StringRef kindString;
  if (parser.parseKeyword(&kindString))
    return failure();
  ForKind kind;
  if (auto k = symbolizeForKind(kindString); k) {
    kind = *k;
  } else {
    return parser.emitError(parser.getNameLoc(), "unknown kind: ")
           << kindString;
  }
  result.addAttribute("kind", builder.getAttr<ForKindAttr>(kind));

  // Parse optional thread attribute.
  bool hasThread = succeeded(parser.parseOptionalLParen());
  if (hasThread) {
    std::string thread;
    if (parser.parseString(&thread) || parser.parseRParen())
      return failure();
    result.addAttribute("thread", builder.getStringAttr(std::move(thread)));
  }

  // Parse loop bounds.
  int64_t start, stop;
  if (parser.parseInteger(start) || parser.parseKeyword("to") ||
      parser.parseInteger(stop)) {
    return failure();
  }
  result.addAttribute("start", builder.getIndexAttr(start));
  result.addAttribute("stop", builder.getIndexAttr(stop));

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {inductionVariable}))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
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

#ifndef TVM_ATTRIBUTES
#define TVM_ATTRIBUTES

include "mlir/IR/EnumAttr.td"


def TVM_ForKind : I32EnumAttr<
  "ForKind", "",
  [
    I32EnumAttrCase<"SERIAL", 0, "serial">,
    I32EnumAttrCase<"PARALLEL", 1, "parallel">,
    I32EnumAttrCase<"VECTORIZED", 2, "vectorized">,
    I32EnumAttrCase<"UNROLL", 3, "unroll">,
    I32EnumAttrCase<"THREAD_BINDING", 4, "thread_binding">,
  ]> {
  let cppNamespace = "::mlir::tvm";
}

def TVM_AxisKind : I32EnumAttr<
  "AxisKind", "",
  [
    I32EnumAttrCase<"SPATIAL", 0, "spatial">,
    I32EnumAttrCase<"REDUCE", 1, "reduce">,
    I32EnumAttrCase<"SCAN", 2, "scan">,
    I32EnumAttrCase<"OPAQUE", 3, "opaque">,
  ]> {
  let cppNamespace = "::mlir::tvm";
}

#endif

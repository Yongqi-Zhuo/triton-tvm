#ifndef TVM_DIALECT
#define TVM_DIALECT

include "mlir/IR/OpBase.td"


def TVM_Dialect : Dialect {
  let name = "tvm";
  let summary = "An MLIR dialect for TVM TensorIR.";
  let description = [{
    Representing TVM TensorIR in MLIR,
    without interacting with the Python DSL.
  }];
  let cppNamespace = "::mlir::tvm";
}

#endif // TVM_DIALECT

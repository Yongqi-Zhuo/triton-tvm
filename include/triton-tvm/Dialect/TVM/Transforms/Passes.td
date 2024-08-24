#ifndef TVM_PASSES
#define TVM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertToTVMScript : Pass<"convert-to-tvmscript", "::mlir::ModuleOp"> {

  let summary = "Convert TVM dialect to TVMScript";

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::tvm::TVMDialect",
  ];

  let options = [
    Option<"outputFilename", "output", "std::string", /*default*/"\"kernel_tvm.py\"",
           "output filename">,
  ];

}

#endif

#ifndef TRITONGPU_TO_TVM_CONVERSION_PASSES
#define TRITONGPU_TO_TVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUToTVM : Pass<"convert-tritongpu-to-tvm", "mlir::ModuleOp"> {
  let summary = "Convert TritonGPU to TVM dialect";
}

#endif

#ifndef TRITONGPU_TO_TVM_CONVERSION_PASSES
#define TRITONGPU_TO_TVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def RewriteSPMDToLoops : Pass<"rewrite-spmd-to-loops", "::mlir::triton::FuncOp"> {

  let summary = "Rewrite SPMD to loops";

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::triton::TritonDialect",
    "::mlir::tvm::TVMDialect",
  ];

  let options = [
    ListOption<"gridDim", "grid-dim", "int",
               "gridDim of the kernel">,
  ];

}

def TritonGPUToTVM : Pass<"convert-tritongpu-to-tvm", "::mlir::ModuleOp"> {

  let summary = "Convert TritonGPU to TVM dialect";

  let constructor = "::mlir::triton::gpu::createConvertTritonGPUToTVMPass()";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::triton::gpu::TritonGPUDialect",
    "::mlir::tvm::TVMDialect",
  ];

}

#endif

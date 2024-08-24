#ifndef TRITONGPU_TO_TVM_CONVERSION_PASSES
#define TRITONGPU_TO_TVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def RewriteSPMDToLoops : Pass<"rewrite-spmd-to-loops", "::mlir::triton::FuncOp"> {

  let summary = "Rewrite SPMD to loops";

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::triton::TritonDialect",
  ];

  let options = [
    ListOption<"gridDim", "grid-dim", "int",
               "gridDim of the kernel">,
  ];

}

def LowerToTensorIdioms : Pass<"lower-to-tensor-idioms", "::mlir::triton::FuncOp"> {

  let summary = "Lower tt.splat, tt.make_range, tt.addptr and tt.broadcast to tensor.generate, for later canonicalization";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::triton::TritonDialect",
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

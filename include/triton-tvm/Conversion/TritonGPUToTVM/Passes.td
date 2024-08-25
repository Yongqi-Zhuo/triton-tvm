#ifndef TRITONGPU_TO_TVM_CONVERSION_PASSES
#define TRITONGPU_TO_TVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LowerToTensorIdioms : Pass<"lower-to-tensor-idioms", "::mlir::triton::FuncOp"> {

  let summary = "Lower tt.splat, tt.make_range, tt.addptr, tt.expand_dims, tt.broadcast and lifted tensor operations to tensor.generate, for later canonicalization.";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::triton::TritonDialect",
  ];

}

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

def ReplaceTritonPointersWithMemRefs : Pass<"replace-triton-pointers-with-memrefs", "::mlir::ModuleOp"> {

  let summary = "Eliminate Triton pointers, replacing them with memrefs. Require canonicalization passes to be run first, so that tt.addptr can be folded. This eliminates all tt.addptr operations.";

  let constructor = "::mlir::triton::createReplaceTritonPointersWithMemRefs()";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::triton::TritonDialect",
    "::mlir::ttm::TritonMemRefDialect",
  ];

}

def TritonGPUToTVM : Pass<"convert-tritongpu-to-tvm", "::mlir::ModuleOp"> {

  let summary = "Convert TritonGPU to TVM dialect";

  let constructor = "::mlir::triton::createConvertTritonGPUToTVMPass()";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::triton::gpu::TritonGPUDialect",
    "::mlir::ttm::TritonMemRefDialect",
    "::mlir::tvm::TVMDialect",
  ];

}

#endif

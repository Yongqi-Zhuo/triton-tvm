#ifndef TRITON_GPU_TO_TVM_CONVERSION_PASSES_H
#define TRITON_GPU_TO_TVM_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createReplaceTritonPointersWithMemRefs();

std::unique_ptr<OperationPass<ModuleOp>> createReplaceTritonPointersWithMemRefs(
    SmallVector<SmallVector<int>> tensorShapes,
    SmallVector<SmallVector<int>> tensorStrides);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToTVMPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToTVMPass(SmallVector<int> gridDim,
                                SmallVector<SmallVector<int>> tensorShapes,
                                SmallVector<SmallVector<int>> tensorStrides);

#define GEN_PASS_DECL
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

} // namespace mlir::triton

#endif

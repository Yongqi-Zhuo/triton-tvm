#ifndef TRITON_GPU_TO_TVM_CONVERSION_PASSES_H
#define TRITON_GPU_TO_TVM_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToTVMPass(SmallVector<int> gridDim,
                                SmallVector<SmallVector<int>> tensorShapes,
                                SmallVector<SmallVector<int>> tensorStrides);

#define GEN_PASS_DECL
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif

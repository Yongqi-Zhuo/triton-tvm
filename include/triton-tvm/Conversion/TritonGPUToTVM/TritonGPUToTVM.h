#ifndef TRITON_CONVERSION_TRITONGPUTOTVM_TRITONGPUTOTVM_H
#define TRITON_CONVERSION_TRITONGPUTOTVM_TRITONGPUTOTVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToTVMPass(std::array<int, 3> gridDim,
                                std::vector<std::vector<int>> shapes,
                                std::vector<std::vector<int>> strides);

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONGPUTOTVM_TRITONGPUTOTVM_H

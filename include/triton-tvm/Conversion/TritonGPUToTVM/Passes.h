#ifndef TRITON_GPU_TO_TVM_CONVERSION_PASSES_H
#define TRITON_GPU_TO_TVM_CONVERSION_PASSES_H

#include "triton-tvm/Conversion/TritonGPUToTVM/TritonGPUToTVM.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif

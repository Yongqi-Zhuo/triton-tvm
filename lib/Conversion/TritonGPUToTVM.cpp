#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h"
#include "triton-tvm/Dialect/TVM/Transforms/Passes.h"
#include "triton-tvm/Dialect/TritonMemRef/TritonMemRef.h"

#define DEBUG_TYPE "tritongpu-to-tvm"

using namespace mlir;

struct StolenDims {
  SmallVector<int> gridDim;
  SmallVector<SmallVector<int>> tensorShapes;
  SmallVector<SmallVector<int>> tensorStrides;
};

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONGPUTOTVM
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

class TritonGPUToTVMPass : public impl::TritonGPUToTVMBase<TritonGPUToTVMPass> {
  StolenDims stolenDims;

public:
  TritonGPUToTVMPass(StolenDims stolenDims)
      : stolenDims(std::move(stolenDims)) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();

    const auto &gridDim = stolenDims.gridDim;
    const auto &tensorShapes = stolenDims.tensorShapes;
    const auto &tensorStrides = stolenDims.tensorStrides;

    llvm::errs() << "gridDim { ";
    for (int dim : gridDim) {
      llvm::errs() << dim << ", ";
    }
    llvm::errs() << "}\n";
    for (const auto &sizes : tensorShapes) {
      llvm::errs() << "sizes { ";
      for (int size : sizes) {
        llvm::errs() << size << ", ";
      }
      llvm::errs() << "}\n";
    }
    for (const auto &strides : tensorStrides) {
      llvm::errs() << "strides { ";
      for (int stride : strides) {
        llvm::errs() << stride << ", ";
      }
      llvm::errs() << "}\n";
    }

    moduleOp.walk([&](triton::FuncOp func) {
      // Make sure all integer parameters are constants, because we currently
      // only support constant integer values.
      for (auto argType : func.getArgumentTypes()) {
        if (!isa<triton::PointerType>(argType)) {
          func.emitError("Only pointer arguments are supported.")
              << " Got " << argType;
          return signalPassFailure();
        }
      }
      // Also check that we have gathered all the tensor shapes and strides.
      if (func.getNumArguments() != tensorShapes.size() ||
          func.getNumArguments() != tensorStrides.size()) {
        func.emitError("Number of arguments does not match the number of "
                       "tensor shapes and strides.");
        return signalPassFailure();
      }
      // Check that tt.return has no arguments.
      func.walk([&](triton::ReturnOp returnOp) {
        if (returnOp.getNumOperands() > 0) {
          returnOp.emitError("tt.return with operands not supported");
          return signalPassFailure();
        }
      });

      PassManager pm(&getContext(), func.getOperationName());
      pm.addPass(createLowerToTensorIdioms());
      pm.addPass(createRewriteSPMDToLoops({gridDim}));
      if (failed(runPipeline(pm, func))) {
        signalPassFailure();
      }

      // Use func.func instead of tt.func.
      OpBuilder builder(func);
      auto name = func.getName();
      auto type = func.getFunctionType();
      // Copy attributes.
      SmallVector<DictionaryAttr> argAttrs, resAttrs;
      func.getAllArgAttrs(argAttrs);
      func.getAllResultAttrs(resAttrs);
      // Create the new function.
      auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, type);
      funcFunc.setAllArgAttrs(argAttrs);
      funcFunc.setAllResultAttrs(resAttrs);
      // Clone the body.
      auto &funcFuncBody = funcFunc.getBody();
      auto &funcBody = func.getBody();
      IRMapping map;
      funcBody.cloneInto(&funcFuncBody, map);
      // Replace the old function with the new one.
      for (Block &block : funcFuncBody.getBlocks()) {
        auto *term = block.getTerminator();
        builder.setInsertionPoint(term);
        builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
        term->erase();
      }
      func.erase();
    });

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(
        createReplaceTritonPointersWithMemRefs(tensorShapes, tensorStrides));
    pm.addPass(createMaterializeTensorsToTVMBuffers());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createLoopInvariantCodeMotionPass());
    if (failed(runPipeline(pm, moduleOp))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToTVMPass() {
  return std::make_unique<TritonGPUToTVMPass>(StolenDims{});
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToTVMPass(SmallVector<int> gridDim,
                                SmallVector<SmallVector<int>> tensorShapes,
                                SmallVector<SmallVector<int>> tensorStrides) {
  return std::make_unique<TritonGPUToTVMPass>(
      StolenDims{.gridDim = std::move(gridDim),
                 .tensorShapes = std::move(tensorShapes),
                 .tensorStrides = std::move(tensorStrides)});
}

} // namespace mlir::triton

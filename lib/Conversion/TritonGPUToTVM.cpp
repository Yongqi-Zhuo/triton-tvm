#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include <optional>

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h"
#include "triton-tvm/Conversion/TritonGPUToTVM/TritonGPUToTVM.h"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h"

#define DEBUG_TYPE "tritongpu-to-tvm"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

struct StolenDims {
  std::array<int, 3> gridDim;
  std::vector<std::vector<int>> tensorShapes;
  std::vector<std::vector<int>> tensorStrides;
};

namespace mlir::triton::gpu {

class TritonGPUToTVMPass : public TritonGPUToTVMBase<TritonGPUToTVMPass> {
  StolenDims stolenDims;

public:
  TritonGPUToTVMPass(StolenDims stolenDims)
      : stolenDims(std::move(stolenDims)) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();

    const auto &gridDim = stolenDims.gridDim;
    const auto &tensorShapes = stolenDims.tensorShapes;
    const auto &tensorStrides = stolenDims.tensorStrides;

    llvm::errs() << "gridDim { .X = " << gridDim[0] << ", .Y = " << gridDim[1]
                 << ", .Z = " << gridDim[2] << " }\n";
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

      OpBuilder b(func);

      constexpr int64_t kNumRows = 128, numCols = 512, numWarps = 4,
                        warpSize = 32, numThreads = numWarps * warpSize;

      std::string name = (func.getName() + "_tvm").str();
      auto type =
          b.getFunctionType({UnrankedMemRefType::get(b.getF32Type(), {})}, {});

      Location loc = func.getLoc();

      auto tvmFunc = b.create<tvm::FuncOp>(loc, name, type);

      Block &entryBlock = tvmFunc.front();
      b.setInsertionPointToStart(&entryBlock);

      auto numRows = b.create<tvm::VarOp>(loc, b.getI32Type()).getResult();

      auto arg = tvmFunc.getArgument(0);
      auto shapedArg = b.create<tvm::MatchBufferOp>(
          loc, MemRefType::get({ShapedType::kDynamic, numCols}, b.getF32Type()),
          arg, ArrayRef<OpFoldResult>{numRows, b.getI32IntegerAttr(numCols)});

      auto max = b.create<tvm::AllocBufferOp>(
          loc, MemRefType::get({ShapedType::kDynamic}, b.getF32Type()),
          "shared");

      auto c0 = arith::ConstantOp::materialize(b, b.getI32IntegerAttr(0),
                                               b.getI32Type(), loc);
      auto cNumThreads = arith::ConstantOp::materialize(
          b, b.getI32IntegerAttr(numThreads), b.getI32Type(), loc);
      auto cNumGroups = arith::ConstantOp::materialize(
          b, b.getI32IntegerAttr(numCols / numThreads), b.getI32Type(), loc);
      auto for0 = tvm::ForOp::create(
          b, loc, c0, numRows,
          b.getAttr<tvm::ForKindAttr>(tvm::ForKind::THREAD_BINDING),
          b.getStringAttr("blockIdx.x"),
          [&](OpBuilder &b, Location loc, Value varRows) {
            auto for1 = tvm::ForOp::create(
                b, loc, c0, cNumThreads,
                b.getAttr<tvm::ForKindAttr>(tvm::ForKind::THREAD_BINDING),
                b.getStringAttr("threadIdx.x"),
                [&](OpBuilder &b, Location loc, Value varCols0) {
                  auto for2 = tvm::ForOp::create(
                      b, loc, c0, cNumGroups,
                      b.getAttr<tvm::ForKindAttr>(tvm::ForKind::UNROLL),
                      std::nullopt,
                      [&](OpBuilder &b, Location loc, Value varCols1) {
                        b.create<tvm::BlockOp>(loc, [&](OpBuilder &b,
                                                        Location loc) {
                          auto arithMultiplier = arith::ConstantOp::materialize(
                              b, b.getI32IntegerAttr(numThreads),
                              b.getI32Type(), loc);
                          auto arithMultiplied = b.create<arith::MulIOp>(
                              loc, varCols1, arithMultiplier.getResult());
                          auto arithAdded = b.create<arith::AddIOp>(
                              loc, varCols0, arithMultiplied.getResult());
                          auto arithBound = arith::ConstantOp::materialize(
                              b, b.getI32IntegerAttr(numCols), b.getI32Type(),
                              loc);
                          auto arithCmped = b.create<arith::CmpIOp>(
                              loc, arith::CmpIPredicate::ult,
                              arithAdded.getResult(), arithBound);
                          b.create<tvm::WhereOp>(loc, arithCmped.getResult());
                          tvm::AxisOp iRows = b.create<tvm::AxisOp>(
                              loc,
                              b.getAttr<tvm::AxisKindAttr>(
                                  tvm::AxisKind::SPATIAL),
                              varRows);
                          tvm::AxisOp iCols = b.create<tvm::AxisOp>(
                              loc,
                              b.getAttr<tvm::AxisKindAttr>(
                                  tvm::AxisKind::REDUCTION),
                              arithAdded.getResult());
                          auto refRead = b.create<tvm::RefOp>(
                              loc, shapedArg, ValueRange{iRows, iCols});
                          auto refWrite =
                              b.create<tvm::RefOp>(loc, max, ValueRange{iRows});
                          b.create<tvm::ReadOp>(loc, ValueRange{refRead});
                          b.create<tvm::WriteOp>(loc, ValueRange{refWrite});
                          b.create<tvm::InitOp>(loc, [&](OpBuilder &b,
                                                         Location loc) {
                            auto arithMin = arith::ConstantOp::materialize(
                                b,
                                b.getF32FloatAttr(
                                    -std::numeric_limits<float>::infinity()),
                                b.getF32Type(), loc);
                            b.create<tvm::AssignOp>(loc, refWrite, arithMin);
                          });
                          b.create<tvm::AssignOp>(loc, refWrite, refRead);
                        });
                      });
                });
          });

      b.setInsertionPointToEnd(&entryBlock);
      b.create<tvm::ReturnOp>(loc);

      // func.erase();
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToTVMPass(std::array<int, 3> gridDim,
                                std::vector<std::vector<int>> tensorShapes,
                                std::vector<std::vector<int>> tensorStrides) {
  return std::make_unique<TritonGPUToTVMPass>(
      StolenDims{.gridDim = std::move(gridDim),
                 .tensorShapes = std::move(tensorShapes),
                 .tensorStrides = std::move(tensorStrides)});
}

} // namespace mlir::triton::gpu

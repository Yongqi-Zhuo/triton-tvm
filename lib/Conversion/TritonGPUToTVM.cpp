#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include "triton-tvm/Conversion/TritonGPUToTVM/TritonGPUToTVM.h"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h"

#define DEBUG_TYPE "tritongpu-to-tvm"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

namespace {

class TritonGPUToTVMPass : public TritonGPUToTVMBase<TritonGPUToTVMPass> {
  std::array<int, 3> gridDim;

public:
  TritonGPUToTVMPass(int gridDimX, int gridDimY, int gridDimZ)
      : gridDim{gridDimX, gridDimY, gridDimZ} {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    triton::gpu::TritonGPUDialect, tvm::TVMDialect>();
  }
  void runOnOperation() override {
    auto moduleOp = getOperation();

    llvm::errs() << "gridDim { .X = " << gridDim[0] << ", .Y = " << gridDim[1]
                 << ", .Z = " << gridDim[2] << " }\n";

    moduleOp.walk([&](triton::FuncOp func) {
      // make sure all integer parameters are constants, because TVM TensorIR
      // only supports constant integer values
      for (auto argType : func.getArgumentTypes()) {
        if (!isa<triton::PointerType>(argType)) {
          func.emitError("only pointer arguments are supported");
          return signalPassFailure();
        }
      }

      OpBuilder b(func);

      constexpr int64_t numRows = 128, numCols = 512, numWarps = 4,
                        warpSize = 32, numThreads = numWarps * warpSize;

      std::string name = (func.getName() + "_tvm").str();
      auto type = b.getFunctionType(
          {MemRefType::get({numRows, numCols}, b.getF32Type())}, {});

      Location loc = func.getLoc();

      auto tvmFunc = b.create<tvm::FuncOp>(loc, name, type);

      Block &entryBlock = tvmFunc.front();
      b.setInsertionPointToStart(&entryBlock);

      Value arg = tvmFunc.getArgument(0);

      auto max = b.create<tvm::AllocBufferOp>(
          loc, MemRefType::get({numRows}, b.getF32Type()), "shared");

      b.create<tvm::ForOp>(
          loc, b.getIndexAttr(0), b.getIndexAttr(numRows),
          b.getAttr<tvm::ForKindAttr>(tvm::ForKind::THREAD_BINDING),
          b.getStringAttr("blockIdx.x"),
          [&](OpBuilder &b, Location loc, Value varRows) {
            b.create<tvm::ForOp>(
                loc, b.getIndexAttr(0), b.getIndexAttr(numThreads),
                b.getAttr<tvm::ForKindAttr>(tvm::ForKind::THREAD_BINDING),
                b.getStringAttr("threadIdx.x"),
                [&](OpBuilder &b, Location loc, Value varCols0) {
                  b.create<tvm::ForOp>(
                      loc, b.getIndexAttr(0),
                      b.getIndexAttr(numCols / numThreads),
                      b.getAttr<tvm::ForKindAttr>(tvm::ForKind::UNROLL),
                      std::nullopt,
                      [&](OpBuilder &b, Location loc, Value varCols1) {
                        b.create<tvm::BlockOp>(loc, [&](OpBuilder &b,
                                                        Location loc) {
                          auto arithMultiplier = arith::ConstantOp::materialize(
                              b, b.getIndexAttr(numThreads), b.getIndexType(),
                              loc);
                          auto arithMultiplied = b.create<arith::MulIOp>(
                              loc, varCols1, arithMultiplier.getResult());
                          auto arithAdded = b.create<arith::AddIOp>(
                              loc, varCols0, arithMultiplied.getResult());
                          auto arithBound = arith::ConstantOp::materialize(
                              b, b.getIndexAttr(numCols), b.getIndexType(),
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
                              loc, arg, ValueRange{iRows, iCols});
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

      func.erase();
    });
  }
};

} // namespace

namespace mlir::triton::gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToTVMPass(int gridDimX, int gridDimY, int gridDimZ) {
  return std::make_unique<TritonGPUToTVMPass>(gridDimX, gridDimY, gridDimZ);
}

} // namespace mlir::triton::gpu

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h"
#include "triton-tvm/Utils/Builder.h"

#define DEBUG_TYPE "rewrite-spmd-to-loops"

using namespace mlir;

namespace mlir::triton {

#define GEN_PASS_DEF_REWRITESPMDTOLOOPS
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

namespace {

struct GetProgramIDConverter
    : public OpConversionPattern<triton::GetProgramIdOp> {

  const SmallVectorImpl<Value> &programIds;

  GetProgramIDConverter(MLIRContext *context,
                        const SmallVectorImpl<Value> &programIds)
      : OpConversionPattern(context), programIds(programIds) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = static_cast<uint32_t>(op.getAxis());
    if (axis >= programIds.size()) {
      op.emitError("get_program_id axis ")
          << axis << " exceeds grid rank " << programIds.size();
      return failure();
    }
    rewriter.replaceOp(op, programIds[axis]);
    return success();
  }
};

struct GetNumProgramsConverter
    : public OpConversionPattern<triton::GetNumProgramsOp> {
  const SmallVectorImpl<int> &gridDim;

  GetNumProgramsConverter(MLIRContext *context,
                          const SmallVectorImpl<int> &gridDim)
      : OpConversionPattern(context), gridDim(gridDim) {}

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = static_cast<uint32_t>(op.getAxis());
    if (axis >= gridDim.size()) {
      op.emitError("get_num_programs axis ")
          << axis << " exceeds grid rank " << gridDim.size();
      return failure();
    }
    auto gridDimValue = arith::ConstantOp::materialize(
        rewriter, rewriter.getI32IntegerAttr(gridDim[axis]),
        rewriter.getI32Type(), op.getLoc());
    rewriter.replaceOp(op, gridDimValue);
    return success();
  }
};

} // namespace

class RewriteSPMDToLoopsPass
    : public impl::RewriteSPMDToLoopsBase<RewriteSPMDToLoopsPass> {
public:
  using RewriteSPMDToLoopsBase::RewriteSPMDToLoopsBase;

  scf::ForOp wrapInForLoop(Region &region, int staticExtent,
                           StringAttr thread) {
    assert(region.hasOneBlock() && "expected region to have one block");
    // First add a for loop to it.
    Block &entryBlock = region.front();
    auto builder = OpBuilder(&region);
    builder.setInsertionPointToStart(&entryBlock);
    auto loc = region.getLoc();
    auto c0 = tvm::utils::getConstantOpI32(builder, loc, 0);
    auto cExtent = tvm::utils::getConstantOpI32(builder, loc, staticExtent);
    auto forLoop = tvm::ForOp::create(
        builder, loc, c0, cExtent,
        builder.getAttr<tvm::ForKindAttr>(tvm::ForKind::THREAD_BINDING),
        thread);

    // Now move the operations from the rest of the entry block to the for loop
    // body.
    auto &forLoopOps = forLoop.getBody()->getOperations();
    forLoopOps.splice(
        // Insert to the beginning of the for loop body.
        forLoopOps.begin(),
        // Move from the entry block.
        entryBlock.getOperations(),
        // Start moving from the next op of for loop, because we have been
        // inserting ops to the start of the entry block.
        ++Block::iterator(forLoop),
        // Stop before the terminator.
        --entryBlock.end());
    return forLoop;
  }

  void runOnOperation() override {
    if (this->gridDim.empty()) {
      // No grid.
      return;
    }
    const SmallVector<int> gridDim(this->gridDim.begin(), this->gridDim.end());

    auto funcOp = cast<triton::FuncOp>(getOperation());

    // Check that tt.return has no arguments.
    funcOp.walk([&](triton::ReturnOp returnOp) {
      if (returnOp.getNumOperands() > 0) {
        returnOp.emitError("tt.return with operands not supported");
        return signalPassFailure();
      }
    });

    OpBuilder builder(funcOp);
    SmallVector<Value> inductionVars;

    // Add the loops.
    scf::ForOp gridX = wrapInForLoop(funcOp.getBody(), gridDim[0],
                                     builder.getStringAttr("blockIdx.x"));
    inductionVars.push_back(gridX.getInductionVar());
    if (gridDim.size() > 1) {
      scf::ForOp gridY = wrapInForLoop(gridX.getRegion(), gridDim[1],
                                       builder.getStringAttr("blockIdx.y"));
      inductionVars.push_back(gridY.getInductionVar());
      if (gridDim.size() > 2) {
        scf::ForOp gridZ = wrapInForLoop(gridY.getRegion(), gridDim[2],
                                         builder.getStringAttr("blockIdx.z"));
        inductionVars.push_back(gridZ.getInductionVar());
      }
    }

    // Replace tt.get_program_id with for iterators.
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<triton::GetProgramIdOp, triton::GetNumProgramsOp>();
    patterns.add<GetProgramIDConverter>(patterns.getContext(), inductionVars);
    patterns.add<GetNumProgramsConverter>(patterns.getContext(), gridDim);
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp.emitError("Error rewriting SPMD to loops");
      return signalPassFailure();
    }
  }
};

} // namespace mlir::triton

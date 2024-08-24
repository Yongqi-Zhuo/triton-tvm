#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h"
#include "triton-tvm/Utils/Builder.h"

#define DEBUG_TYPE "lower-to-tensor-idioms"

using namespace mlir;

namespace mlir::triton {

#define GEN_PASS_DEF_REPLACETRITONPOINTERSWITHMEMREFS
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

namespace {

struct AddPtrFolder : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;
  LogicalResult match(triton::AddPtrOp op) const override {
    // If the base pointer is also an addptr, we can fold the offsets.
    return success(op.getPtr().getDefiningOp<triton::AddPtrOp>());
  }
  void rewrite(triton::AddPtrOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    // addptr(addptr(base, offset_1), offset_2)
    //        ^^^^^^^^^^^^^^^^^^^^^^
    auto innerAddPtr = cast<triton::AddPtrOp>(adaptor.getPtr().getDefiningOp());

    // addptr(addptr(base, offset_1), offset_2)
    //                     ^^^^^^^^
    auto innerOffset = innerAddPtr.getOffset();

    // addptr(addptr(base, offset_1), offset_2)
    //                                ^^^^^^^^
    auto outerOffset = adaptor.getOffset();

    // addptr(base, offset_1 + offset_2)
    //              ^^^^^^^^^^^^^^^^^^^
    auto newOffset =
        rewriter.create<arith::AddIOp>(op.getLoc(), innerOffset, outerOffset);

    // addptr(base, offset_1 + offset_2)
    auto newAddPtr = rewriter.create<triton::AddPtrOp>(
        op.getLoc(), op.getType(), innerAddPtr.getPtr(), newOffset);

    rewriter.replaceOp(op, newAddPtr);
  }
};

} // namespace

class ReplaceTritonPointersWithMemRefsPass
    : public impl::ReplaceTritonPointersWithMemRefsBase<
          ReplaceTritonPointersWithMemRefsPass> {
  SmallVector<SmallVector<int>> tensorShapes;
  SmallVector<SmallVector<int>> tensorStrides;

public:
  ReplaceTritonPointersWithMemRefsPass(
      SmallVector<SmallVector<int>> tensorShapes,
      SmallVector<SmallVector<int>> tensorStrides)
      : tensorShapes(std::move(tensorShapes)),
        tensorStrides(std::move(tensorStrides)) {}

  // First we need to rewrite tt.addptr:
  //   addptr(addptr(base, offset_1), offset_2)
  //    --->  addptr(base, offset_1 + offset_2)
  // Keep folding until the base pointer is a function argument.
  void foldAddPtr() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect>();
    target.addDynamicallyLegalOp<triton::AddPtrOp>([](triton::AddPtrOp op) {
      if (op.getPtr().getDefiningOp()) {
        // This is not a function argument.
        return false;
      }
      auto blockArg = cast<BlockArgument>(op.getPtr());
      auto *owningBlock = blockArg.getOwner();
      auto *owningOp = owningBlock->getParentOp();
      return
          // The block argument is in a function.
          isa<func::FuncOp>(owningOp) &&
          // The block is the entry block.
          (&cast<func::FuncOp>(owningOp).getBlocks().front() == owningBlock);
    });
    patterns.add<AddPtrFolder>(patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      moduleOp.emitError(
          "Failed to rewrite addptr operations. It seems the current "
          "implementation is unable to match this pattern.");
      return signalPassFailure();
    }
  }

  void runOnOperation() override { foldAddPtr(); }
};

std::unique_ptr<OperationPass<ModuleOp>>
createReplaceTritonPointersWithMemRefs() {
  return std::make_unique<ReplaceTritonPointersWithMemRefsPass>(
      SmallVector<SmallVector<int>>{}, SmallVector<SmallVector<int>>{});
}

std::unique_ptr<OperationPass<ModuleOp>> createReplaceTritonPointersWithMemRefs(
    SmallVector<SmallVector<int>> tensorShapes,
    SmallVector<SmallVector<int>> tensorStrides) {
  return std::make_unique<ReplaceTritonPointersWithMemRefsPass>(
      std::move(tensorShapes), std::move(tensorStrides));
}

} // namespace mlir::triton

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h"
#include "triton-tvm/Dialect/TritonMemRef/TritonMemRef.h"
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

struct TensorSpec {
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;

  TensorSpec() = default;
  template <typename Sizes, typename Strides>
  TensorSpec(Sizes &&sizes, Strides &&strides)
      : sizes(std::forward<Sizes>(sizes)),
        strides(std::forward<Strides>(strides)) {}

  MemRefType toMemRefType(Type elementType) const {
    return MemRefType::get(
        sizes, elementType,
        StridedLayoutAttr::get(elementType.getContext(), 0, strides));
  }
};

struct FuncArgToMemRefConverter : public OpConversionPattern<func::FuncOp> {
  const SmallVectorImpl<TensorSpec> &tensorSpec;
  FuncArgToMemRefConverter(MLIRContext *ctx,
                           const SmallVectorImpl<TensorSpec> &tensorSpec)
      : OpConversionPattern(ctx), tensorSpec(tensorSpec) {}

  LogicalResult match(func::FuncOp op) const override {
    return success(
        llvm::any_of(op.getFunctionType().getInputs(), [](Type argType) {
          return isa<triton::PointerType>(argType);
        }));
  }

  void rewrite(func::FuncOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto funcType = op.getFunctionType();

    // Substitute all pointer arguments with memref types.
    TypeConverter::SignatureConversion newSignature(funcType.getNumInputs());
    SmallVector<Type> newInputs;
    int64_t tensorIndex = 0;
    for (auto argTypeEnumerated : llvm::enumerate(funcType.getInputs())) {
      auto argNo = argTypeEnumerated.index();
      auto argType = argTypeEnumerated.value();
      if (auto ptrType = dyn_cast<triton::PointerType>(argType)) {
        auto tensorType =
            tensorSpec[tensorIndex].toMemRefType(ptrType.getPointeeType());
        newInputs.push_back(tensorType);
        ++tensorIndex;
      } else {
        newInputs.push_back(argType);
      }
      newSignature.addInputs(argNo, {newInputs.back()});
    }
    if (tensorIndex != tensorSpec.size()) {
      op.emitError("Number of captured tensors does not match the number of "
                   "pointer arguments.");
      return;
    }

    // Apply the signature conversion at the entry block.
    rewriter.applySignatureConversion(&op.getFunctionBody().front(),
                                      newSignature);

    // Update the function type.
    auto newFuncType = FunctionType::get(rewriter.getContext(), newInputs,
                                         funcType.getResults());
    rewriter.modifyOpInPlace(op, [&] { op.setType(newFuncType); });
  }
};

struct UseMemRefInAddPtrConverter
    : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memref = adaptor.getPtr();
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType) {
      // We have not applied the conversion pattern to the function arguments.
      return failure();
    }
    // Here, the function argument is already converted to a memref<*x?>.
    // Use ttm.memref_to_ptr %memref[%offset0, %offset1, ...] to perform
    // indexing. To do this, we delinearize the offset.
    auto indices = tvm::utils::delinearizeIndexWithStrides(
        rewriter, op->getLoc(), adaptor.getOffset(),
        cast<StridedLayoutAttr>(memrefType.getLayout()).getStrides());
    auto newPtr = rewriter.create<ttm::MemRefToPtrOp>(
        op.getLoc(), op.getType(), memref, ValueRange(indices));
    rewriter.replaceOp(op, newPtr);
    return success();
  }
};

} // namespace

class ReplaceTritonPointersWithMemRefsPass
    : public impl::ReplaceTritonPointersWithMemRefsBase<
          ReplaceTritonPointersWithMemRefsPass> {
  SmallVector<TensorSpec> tensorSpecs;

public:
  ReplaceTritonPointersWithMemRefsPass(SmallVector<TensorSpec> tensorSpecs)
      : tensorSpecs(std::move(tensorSpecs)) {}

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

  void runOnOperation() override {
    // First fold addptr operations.
    foldAddPtr();

    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, ttm::TritonMemRefDialect>();
    target.addIllegalOp<triton::AddPtrOp>();
    patterns.add<UseMemRefInAddPtrConverter>(patterns.getContext());

    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
      return llvm::none_of(op.getArgumentTypes(), [](Type argType) {
        return isa<triton::PointerType>(argType);
      });
    });
    patterns.add<FuncArgToMemRefConverter>(patterns.getContext(), tensorSpecs);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      moduleOp.emitError("Failed to convert Triton pointers to memrefs.");
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createReplaceTritonPointersWithMemRefs() {
  return std::make_unique<ReplaceTritonPointersWithMemRefsPass>(
      SmallVector<TensorSpec>{});
}

std::unique_ptr<OperationPass<ModuleOp>> createReplaceTritonPointersWithMemRefs(
    SmallVector<SmallVector<int>> tensorShapes,
    SmallVector<SmallVector<int>> tensorStrides) {
  SmallVector<TensorSpec> tensorSpecs;
  for (auto dims : llvm::zip(tensorShapes, tensorStrides)) {
    const auto &[sizes, strides] = dims;
    SmallVector<int64_t> sizesI64(sizes.begin(), sizes.end());
    SmallVector<int64_t> stridesI64(strides.begin(), strides.end());
    tensorSpecs.emplace_back(std::move(sizesI64), std::move(stridesI64));
  }
  return std::make_unique<ReplaceTritonPointersWithMemRefsPass>(
      std::move(tensorSpecs));
}

} // namespace mlir::triton

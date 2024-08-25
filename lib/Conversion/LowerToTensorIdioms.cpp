#include <tuple>

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

#define GEN_PASS_DEF_LOWERTOTENSORIDIOMS
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

namespace {

struct SplatToTensorConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<RankedTensorType>(op.getResult().getType());
    auto src = adaptor.getSrc();
    // We cannot use tensor.splat here, because it rejects pointer types.
    auto tensorGenerate = rewriter.create<tensor::GenerateOp>(
        op.getLoc(), type,
        // All static dimensions
        ValueRange{},
        // Just emit the src value
        [&](OpBuilder &b, Location loc, ValueRange args) {
          b.create<tensor::YieldOp>(loc, src);
        });
    tensorGenerate->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, tensorGenerate);
    return success();
  }
};

struct MakeRangeToTensorConverter
    : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<RankedTensorType>(op.getResult().getType());
    assert(op.getStart() == 0 && "only support start=0 for now");
    auto tensorGenerate = rewriter.create<tensor::GenerateOp>(
        op.getLoc(), type,
        // All static dimensions
        ValueRange{}, [&](OpBuilder &b, Location loc, ValueRange args) {
          assert(args.size() == 1 && "expected 1-D for make_range");
          Value iter = args[0];
          if (!type.getElementType().isIndex()) {
            // We need to cast to i32, because Triton sucks.
            iter =
                b.create<arith::IndexCastOp>(loc, type.getElementType(), iter);
          }
          b.create<tensor::YieldOp>(loc, iter);
        });
    tensorGenerate->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, tensorGenerate);
    return success();
  }
};

struct AddPointerToTensorConverter
    : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;
  LogicalResult match(triton::AddPtrOp op) const override {
    // We only lift addition of tensors to tensors of addition.
    return success(isa<RankedTensorType>(op.getOperand(0).getType()));
  }
  void rewrite(triton::AddPtrOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    auto typeLhs = cast<RankedTensorType>(lhs.getType());
    auto typeRhs = cast<RankedTensorType>(rhs.getType());
    // Note that their element types may be different, because of !tt.ptr.
    assert(typeLhs.getShape() == typeRhs.getShape() &&
           "expected same shape for add");
    auto tensorGenerate = rewriter.create<tensor::GenerateOp>(
        op.getLoc(), typeLhs,
        // All static dimensions
        ValueRange{}, [&](OpBuilder &b, Location loc, ValueRange args) {
          Value scalarLhs = b.create<tensor::ExtractOp>(loc, lhs, args);
          Value scalarRhs = b.create<tensor::ExtractOp>(loc, rhs, args);
          Value scalar;
          // Now consider the scalar type. !tt.ptr just sucks.
          auto scalarLhsType = scalarLhs.getType();
          if (isa<triton::PointerType>(scalarLhsType)) {
            scalar = b.create<triton::AddPtrOp>(loc, scalarLhsType, scalarLhs,
                                                scalarRhs);
          } else {
            scalar = b.create<arith::AddIOp>(loc, scalarLhs, scalarRhs);
          }
          b.create<tensor::YieldOp>(loc, scalar);
        });
    tensorGenerate->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, tensorGenerate);
  }
};

// We do not like lifted elementwise ops, so we convert them to tensor.generate
// with scalar ops.
template <typename OpTy>
struct ElementwiseToTensorConverter : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult match(OpTy op) const override {
    // We only lift elementwise ops.
    return success(isa<RankedTensorType>(op.getResult().getType()));
  }
  void rewrite(OpTy op, typename OpConversionPattern<OpTy>::OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto type = cast<RankedTensorType>(op.getResult().getType());
    auto tensorGenerate = rewriter.create<tensor::GenerateOp>(
        op.getLoc(), type,
        // All static dimensions
        ValueRange{}, [&](OpBuilder &b, Location loc, ValueRange args) {
          SmallVector<Value> scalarOperands;
          for (auto operand : adaptor.getOperands()) {
            scalarOperands.push_back(
                b.create<tensor::ExtractOp>(loc, operand, args));
          }
          Value scalar =
              b.create<OpTy>(loc, ValueRange(scalarOperands), op->getAttrs());
          b.create<tensor::YieldOp>(loc, scalar);
        });
    rewriter.replaceOp(op, tensorGenerate);
  }
};

} // namespace

class LowerToTensorIdiomsPass
    : public impl::LowerToTensorIdiomsBase<LowerToTensorIdiomsPass> {
public:
  using LowerToTensorIdiomsBase::LowerToTensorIdiomsBase;

  template <typename... OpTy>
  static void addElementwiseToTensorConverter(RewritePatternSet &patterns,
                                              std::tuple<OpTy...> *) {
    (patterns.add<ElementwiseToTensorConverter<OpTy>>(patterns.getContext()),
     ...);
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<tensor::TensorDialect>();
    auto isNotLiftedTensorOp = [](Operation *op) {
      if (op->getNumOperands() > 0 && op->getNumResults() == 1) {
        auto resultType = op->getResult(0).getType();
        return !isa<RankedTensorType>(resultType);
      }
      return true;
    };
    target.addDynamicallyLegalDialect<arith::ArithDialect>(isNotLiftedTensorOp);
    target.addDynamicallyLegalDialect<math::MathDialect>(isNotLiftedTensorOp);
    target.addIllegalOp<triton::SplatOp, triton::MakeRangeOp>();
    target.addDynamicallyLegalOp<triton::AddPtrOp>([](triton::AddPtrOp op) {
      // Scalars.
      return !isa<RankedTensorType>(op.getOperand(0).getType());
    });

    // We do not convert load/store ops, because they should be left to later
    // passes, where we determine which tensors to materialize.

    patterns.add<SplatToTensorConverter>(patterns.getContext());
    patterns.add<MakeRangeToTensorConverter>(patterns.getContext());
    patterns.add<AddPointerToTensorConverter>(patterns.getContext());
    // TODO: ExpandDimsOp.
    // TODO: BroadcastOp.
    using LiftedElementwiseOps = std::tuple<
        arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::CeilDivSIOp,
        arith::CeilDivUIOp, arith::CmpFOp, arith::CmpIOp, arith::DivFOp,
        arith::DivSIOp, arith::DivUIOp, arith::FloorDivSIOp, arith::MaximumFOp,
        arith::MaxNumFOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinimumFOp,
        arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp, arith::MulFOp,
        arith::MulIOp, arith::NegFOp, arith::OrIOp, arith::RemFOp,
        arith::RemSIOp, arith::RemUIOp, arith::ShLIOp, arith::ShRSIOp,
        arith::ShRUIOp, arith::SubFOp, arith::SubIOp, arith::XOrIOp,
        math::AbsFOp, math::AbsIOp, math::CbrtOp, math::CeilOp, math::CosOp,
        math::ExpOp, math::FloorOp, math::LogOp, math::Log2Op, math::SinOp,
        math::SqrtOp, math::TanOp>;
    addElementwiseToTensorConverter(
        patterns, static_cast<LiftedElementwiseOps *>(nullptr));

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp.emitError("Error lowering to tensor idioms");
      return signalPassFailure();
    }
  }
};

} // namespace mlir::triton

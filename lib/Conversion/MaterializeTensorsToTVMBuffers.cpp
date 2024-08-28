#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h"
#include "triton-tvm/Dialect/TritonMemRef/TritonMemRef.h"
#include "triton-tvm/Utils/Builder.h"

#define DEBUG_TYPE "materialize-tensors-to-tvm-buffers"

namespace mlir::triton {

#define GEN_PASS_DEF_MATERIALIZETENSORSTOTVMBUFFERS
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h.inc"

namespace {

// blockIdx has been bound to the loop induction var
enum class ThreadBinding {
  THREAD_IDX_X,
  THREAD_IDX_Y,
  THREAD_IDX_Z,
};
StringAttr threadBindingToAttr(OpBuilder &b, ThreadBinding binding) {
  switch (binding) {
  case ThreadBinding::THREAD_IDX_X:
    return b.getStringAttr("threadIdx.x");
  case ThreadBinding::THREAD_IDX_Y:
    return b.getStringAttr("threadIdx.y");
  case ThreadBinding::THREAD_IDX_Z:
    return b.getStringAttr("threadIdx.z");
  default:
    llvm_unreachable("unexpected thread binding");
  }
}
std::optional<StringAttr>
threadBindingToAttr(OpBuilder &b, std::optional<ThreadBinding> binding) {
  if (binding)
    return threadBindingToAttr(b, *binding);
  return std::nullopt;
}

struct LoopSpec {
  unsigned extent;
  tvm::ForKind kind;
  std::optional<ThreadBinding> thread;
  // Auxiliary
  unsigned dim;
};

SmallVector<scf::ForOp> generateLoopNestFromSpecs(OpBuilder &b, Location loc,
                                                  ArrayRef<LoopSpec> specs) {
  SmallVector<scf::ForOp> loops;
  if (specs.empty())
    return loops;
  auto zero = tvm::utils::getConstantOpI32(b, loc, 0);
  for (const auto &[extent, kind, thread, _] : specs) {
    auto loop = tvm::ForOp::create(
        b, loc, zero, tvm::utils::getConstantOpI32(b, loc, extent),
        b.getAttr<tvm::ForKindAttr>(kind), threadBindingToAttr(b, thread));
    loops.push_back(loop);
    b.setInsertionPointToStart(loop.getBody());
  }
  b.setInsertionPointAfter(loops.front());
  return loops;
}

struct LoopNest {
  SmallVector<scf::ForOp> loops;
  // The induction vars are recovered to tensor indices.
  SmallVector<Value> tensorIndices;
  // If no contraction, this is the tensor shape. Otherwise, an additional
  // reduction dim is appended.
  SmallVector<int64_t> tensorShape;

  void setInsertingToInnermost(OpBuilder &b) {
    if (loops.empty())
      return;
    b.setInsertionPoint(loops.back().getBody()->getTerminator());
  }
};

// reductionDim is the dimension to be reduced.
// If no reduction is required, reductionDim = -1.
// If the reduction is not in the tensor type, e.g., tt.dot where matmul
// contracts a dimension, reductionDim = rank.
class ReductionDim {
  int rank;        // rank of tensor
  int dim;         // the reduction dim
  unsigned extent; // the extent of the additional loop
  ReductionDim(int rank, int dim, int extent)
      : rank(rank), dim(dim), extent(extent) {}

public:
  static ReductionDim createSpatial(unsigned rank) {
    return ReductionDim(rank, -1, 0);
  }
  static ReductionDim createReduction(unsigned rank, unsigned dim) {
    assert(dim < rank && "reduction dim out of range");
    return ReductionDim(rank, dim, 0);
  }
  static ReductionDim createContraction(unsigned rank, unsigned extent) {
    return ReductionDim(rank, rank, extent);
  }
  bool isSpatial() const { return dim < 0; }
  bool isReduction() const { return dim >= 0 && dim < rank; }
  bool isContraction() const { return dim == rank; }
  unsigned getRank() const { return rank; }
  unsigned get() const {
    assert(!isSpatial() && "no reduction");
    return dim;
  }
  bool isDimReduced(unsigned d) const { return d == dim; }
  unsigned getContractionExtent() const {
    assert(isContraction() && "not a contraction");
    return extent;
  }
  template <typename T>
  inline T match(T spatial, T reduction, T contraction) const {
    if (isSpatial())
      return spatial;
    if (isReduction())
      return reduction;
    if (isContraction())
      return contraction;
    llvm_unreachable("unexpected reduction dim");
  }
};

// TODO: add vectorize option.
LoopNest generateLoopNest(OpBuilder &b, Location loc, RankedTensorType tensor,
                          ReductionDim rDim) {
  const auto layout =
      cast<triton::gpu::BlockedEncodingAttr>(tensor.getEncoding());
  // Ignore Hopper. CGA size is always 1.

  SmallVector<LoopSpec> specs;

  // Here is our loop nest:
  // for warpsPerCTA[order[r-1]]:
  //  ...
  //   for warpsPerBlock[order[0]]:
  //    for threadsPerWarp[order[r-1]]:
  //     ...
  //      for threadsPerBlock[order[0]]:
  //       for shape[order[r-1]]/totalPerCTA[order[r-1]]: # unroll
  //        ...
  //         for shape[order[0]]/totalPerBlock[order[0]]: # unroll
  //          for sizePerThread[order[r-1]]:
  //           ...
  //            for sizePerThread[order[0]]: # vectorize
  enum Stage {
    STAGE_WARPS_PER_CTA = 0,
    STAGE_THREADS_PER_WARP,
    STAGE_CTA_TILE,
    STAGE_SIZE_PER_THREAD,
    STAGE_COUNT,
  };
  const auto layoutOrder = layout.getOrder();
  const auto layoutWarpsPerCTA = layout.getWarpsPerCTA();
  const auto layoutThreadsPerWarp = layout.getThreadsPerWarp();
  const auto layoutSizePerThread = layout.getSizePerThread();

  // For reduction, perform some basic checks.
  if (rDim.isReduction()) {
    assert(layoutOrder[0] == rDim.get() &&
           "reduction dim must be the most contiguous dimension");
    assert(layoutSizePerThread[rDim.get()] == 1 &&
           "cannot vectorize a cross-thread reduction");
  }

  for (Stage stage = static_cast<Stage>(0); stage < STAGE_COUNT;
       stage = static_cast<Stage>(stage + 1)) {
    // From the least contiguous dimension to the most contiguous dimension.
    for (int order = static_cast<int>(layoutOrder.size()) - 1; order >= 0;
         --order) {
      unsigned dim = layoutOrder[order];
      LoopSpec spec{0, tvm::ForKind::SERIAL, std::nullopt, dim};
      switch (stage) {
      case STAGE_WARPS_PER_CTA:
        spec.extent = layoutWarpsPerCTA[dim];
        spec.kind = tvm::ForKind::THREAD_BINDING;
        spec.thread = ThreadBinding::THREAD_IDX_X;
        break;
      case STAGE_THREADS_PER_WARP:
        spec.extent = layoutThreadsPerWarp[dim];
        spec.kind = tvm::ForKind::THREAD_BINDING;
        spec.thread = ThreadBinding::THREAD_IDX_X;
        break;
      case STAGE_CTA_TILE:
        spec.extent = tensor.getDimSize(dim) /
                      (layoutWarpsPerCTA[dim] * layoutThreadsPerWarp[dim] *
                       layoutSizePerThread[dim]);
        // TODO: Will this work?
        spec.kind = rDim.match(tvm::ForKind::UNROLL, tvm::ForKind::SERIAL,
                               tvm::ForKind::UNROLL);
        break;
      case STAGE_SIZE_PER_THREAD:
        spec.extent = layoutSizePerThread[dim];
        spec.kind = rDim.match(order == 0 ? tvm::ForKind::VECTORIZED
                                          : tvm::ForKind::UNROLL,
                               tvm::ForKind::UNROLL, // unreachable
                               tvm::ForKind::UNROLL);
        break;
      default:
        llvm_unreachable("unexpected stage");
      }
      if (spec.extent > 1)
        specs.push_back(spec);
    }
  }
  if (rDim.isContraction()) {
    specs.push_back({rDim.getContractionExtent(), tvm::ForKind::SERIAL,
                     std::nullopt, rDim.get()});
  }
  auto loops = generateLoopNestFromSpecs(b, loc, specs);

  // Then recover the tensor indices.
  OpBuilder::InsertionGuard guard(b);
  if (!loops.empty())
    b.setInsertionPointToStart(loops.back().getBody());
  auto zero = tvm::utils::getConstantOpI32(b, loc, 0);
  SmallVector<Value> tensorIndices(tensor.getRank() + rDim.isContraction(),
                                   zero);
  SmallVector<unsigned> strides(tensor.getRank() + rDim.isContraction(), 1);
  for (int forLoopId = static_cast<int>(specs.size()) - 1; forLoopId >= 0;
       --forLoopId) {
    Value inductionVar = loops[forLoopId].getInductionVar();
    unsigned dim = specs[forLoopId].dim;
    Value offset = b.create<arith::MulIOp>(
        loc, inductionVar, tvm::utils::getConstantOpI32(b, loc, strides[dim]));
    tensorIndices[dim] =
        b.create<arith::AddIOp>(loc, tensorIndices[dim], offset);
    strides[dim] *= specs[forLoopId].extent;
  }
  SmallVector<int64_t> tensorShape(tensor.getShape());
  if (rDim.isContraction()) {
    tensorShape.push_back(rDim.getContractionExtent());
  }
  return LoopNest{std::move(loops), std::move(tensorIndices),
                  std::move(tensorShape)};
}

// Traverse the loop nest and collect.
SmallVector<scf::ForOp> collectLoopsInScope(Block *block) {
  SmallVector<scf::ForOp> loops;
  while (block) {
    auto *parentOp = block->getParentOp();
    if (!parentOp)
      break;
    if (llvm::TypeSwitch<Operation *, bool>(parentOp)
            .Case([&](scf::ForOp forOp) {
              loops.push_back(forOp);
              return true;
            })
            .Case([&](func::FuncOp) { return false; })
            .Default([&](Operation *) { return true; })) {
      block = parentOp->getBlock();
    } else {
      break;
    }
  }
  std::reverse(loops.begin(), loops.end());
  return loops;
}

SmallVector<Value>
getInductionVarsFromLoopNest(SmallVectorImpl<scf::ForOp> &loops) {
  return llvm::map_to_vector(
      loops, [](scf::ForOp loop) { return loop.getInductionVar(); });
}

SmallVector<Value> getExtentsFromLoopNest(SmallVectorImpl<scf::ForOp> &loops) {
  return llvm::map_to_vector(
      loops, [](scf::ForOp loop) { return loop.getUpperBound(); });
}

SmallVector<Value> mapVarsToAxes(OpBuilder &b, ValueRange extents,
                                 ValueRange vars, ReductionDim rDim) {
  SmallVector<Value> axes;
  for (auto [dim, extent, var] : llvm::enumerate(extents, vars)) {
    axes.push_back(b.create<tvm::AxisOp>(
        var.getLoc(),
        b.getAttr<tvm::AxisKindAttr>(rDim.isDimReduced(dim)
                                         ? tvm::AxisKind::REDUCTION
                                         : tvm::AxisKind::SPATIAL),
        extent, var));
  }
  return axes;
}

// Returns true if the op is cloned.
bool inlineRecursivelyImpl(OpBuilder &b, Value value, IRMapping &mapper,
                           bool needsCloning = false) {
  if (auto newValue = mapper.lookupOrNull(value)) {
    // Already visited. Reuse the result.
    return newValue != value;
  }
  // Have not visited. Try inlining.
  auto *op = value.getDefiningOp();
  if (!op) {
    // This is a block argument that needs not to be rewritten.
    // (Because previous lookup returned nullptr.)
    return false;
  }
  for (Value operand : op->getOperands()) {
    needsCloning |= inlineRecursivelyImpl(b, operand, mapper);
  }
  if (needsCloning) {
    // At least one of the oper.ands was rewritten. Clone the op.
    // Check that the op is pure.
    assert(isPure(op) && "expected pure op");
    // Check that we are not cloning a region.
    assert(op->getNumRegions() == 0 && "expected no regions");
    op = b.cloneWithoutRegions(*op, mapper);
    // The results are implicitly mapped.
    return true;
  }
  // No need to clone. Just update the mapping.
  mapper.map(op->getResults(), op->getResults());
  return false;
}

// Return yielded value.
SmallVector<Value, 1> inlineRecursively(OpBuilder &b, Block &block,
                                        IRMapping &mapper) {
  // It is possible that there are some variables defined in the block that we
  // have to use, which may not dominate the current insertion point, so we must
  // explicitly copy the operations in the block.
  for (auto &op : block.without_terminator()) {
    if (op.getNumResults() == 0)
      continue; // seems to be no-op.
    inlineRecursivelyImpl(b, op.getResult(0), mapper, /*needsCloning=*/true);
  }
  auto *terminator = block.getTerminator();
  assert(terminator->hasTrait<OpTrait::ReturnLike>());
  SmallVector<Value, 1> results;
  for (Value operand : terminator->getOperands()) {
    inlineRecursivelyImpl(b, operand, mapper);
    results.push_back(mapper.lookup(operand));
  }
  return results;
}

struct Computation {
  // The created computation block.
  tvm::BlockOp block;
  // The outer loop nest has already been mapped.
  IRMapping mapper;
  // With tvm.axis applied.
  SmallVector<Value> innerLoopsAxes;
  // The reduction dim.
  ReductionDim rDim;
  // The outer loops axes.
  SmallVector<Value> outerLoopsAxes;
  // The allocated buffer.
  tvm::AllocBufferOp buffer;

  [[nodiscard]] OpBuilder::InsertionGuard enterBlock(OpBuilder &b) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(block.getBody());
    return guard;
  }

  static SmallVector<Value> getArrayAsIndexType(OpBuilder &b,
                                                ValueRange array) {
    auto type = b.getIndexType();
    return llvm::map_to_vector(array, [&](Value value) {
      return b.create<arith::IndexCastOp>(value.getLoc(), type, value)
          .getResult();
    });
  }

  static SmallVector<Value> getIndexArrayAsType(OpBuilder &b, ValueRange array,
                                                Type type) {
    return llvm::map_to_vector(array, [&](Value value) {
      // Fast path: if the value is already of the right type, just return it.
      if (auto indexCastOp = value.getDefiningOp<arith::IndexCastOp>())
        if (indexCastOp.getIn().getType() == type)
          return indexCastOp.getIn();
      return b.create<arith::IndexCastOp>(value.getLoc(), type, value)
          .getResult();
    });
  }

  // Because we currently uses i32 as indices, we have to cast innerLoopsAxes to
  // index so that they can be used to inline the generator.
  SmallVector<Value> getInnerLoopsAxesAsIndexType(OpBuilder &b) const {
    return getArrayAsIndexType(b, innerLoopsAxes);
  }

  // We simply return the indices wrapped in the axes. This can be used in
  // tvm.where statements.
  SmallVector<Value> getInnerLoopsIndices() const {
    return llvm::map_to_vector(innerLoopsAxes, [](Value axis) {
      return static_cast<Value>(axis.getDefiningOp<tvm::AxisOp>().getBinding());
    });
  }

  SmallVector<Value> getInnerLoopsIndicesAsIndexType(OpBuilder &b) const {
    return getArrayAsIndexType(b, getInnerLoopsIndices());
  }

  SmallVector<Value> getOuterLoopsIndices() const {
    return llvm::map_to_vector(outerLoopsAxes, [](Value axis) {
      return static_cast<Value>(axis.getDefiningOp<tvm::AxisOp>().getBinding());
    });
  }

  SmallVector<Value> getOuterLoopsExtents() const {
    return llvm::map_to_vector(outerLoopsAxes, [](Value axis) {
      return static_cast<Value>(axis.getDefiningOp<tvm::AxisOp>().getExtent());
    });
  }

  SmallVector<Value> inlineFunction(OpBuilder &b, Operation *op,
                                    ValueRange args) {
    auto &generator = op->getRegion(0).front();
    mapper.map(generator.getArguments(), args);
    return inlineRecursively(b, generator, mapper);
  }

  // Here, we access the buffer with the outer loop indices.
  tvm::RefOp getDest(OpBuilder &b, Location loc) {
    auto axes = innerLoopsAxes;
    if (!rDim.isSpatial()) {
      axes.erase(axes.begin() + rDim.get());
    }
    std::copy(outerLoopsAxes.begin(), outerLoopsAxes.end(),
              std::inserter(axes, axes.begin()));
    return b.create<tvm::RefOp>(loc, buffer, axes);
  }
};

SmallVector<Value>
getOuterLoopsAxesInBlockFromVars(tvm::BlockOp blockOp,
                                 SmallVectorImpl<Value> &outerLoopsVars) {
  IRMapping mapper;
  // Get all axes definitions in the computation block.
  for (auto &op : *blockOp.getBody()) {
    if (auto axis = dyn_cast<tvm::AxisOp>(op)) {
      mapper.map(axis.getBinding(), axis);
    }
  }
  return llvm::map_to_vector(outerLoopsVars,
                             [&](Value var) { return mapper.lookup(var); });
}

Block::iterator getPointInBlockWhereAllAxesAreDefined(tvm::BlockOp blockOp) {
  auto *block = blockOp.getBody();
  Block::iterator ret = block->begin();
  for (auto it = block->begin(), e = block->end(); it != e; ++it) {
    if (isa<tvm::AxisOp>(it)) {
      ret = std::next(it);
    }
  }
  return ret;
}

Computation computeTensor(OpBuilder &b, Location loc,
                          RankedTensorType tensorType, ReductionDim rDim,
                          RankedTensorType allocType) {
  Computation result{.rDim = rDim};

  // Collect outer loops.
  auto outerLoopNest = collectLoopsInScope(b.getBlock());
  auto outerLoopsExtents = getExtentsFromLoopNest(outerLoopNest);
  auto outerLoopsVars = getInductionVarsFromLoopNest(outerLoopNest);

  if (allocType) {
    // 1. Allocate buffer.
    auto shape =
        // Get outer loops extents. For the time being, we require outer loops
        // to have constant extent.
        llvm::map_to_vector(outerLoopsExtents, [&](Value extent) {
          return cast<IntegerAttr>(
                     extent.getDefiningOp<arith::ConstantOp>().getValue())
              .getValue()
              .getSExtValue();
        });
    // Get result tensor shape.
    auto tensorShape = allocType.getShape();
    // Prepend the result tensor shape with the outer loops extents. This is
    // required by TVM.
    std::copy(tensorShape.begin(), tensorShape.end(),
              std::back_inserter(shape));
    auto buffer = b.create<tvm::AllocBufferOp>(
        loc, MemRefType::get(shape, allocType.getElementType()),
        b.getStringAttr("shared"));
    result.buffer = buffer;

    // // TODO: make sure this really works and does not break the rewriter.
    // if (!outerLoopNest.empty()) {
    //   // Promote this allocation to the root scope.
    //   buffer->moveBefore(outerLoopNest.front());
    // }
  }

  // 2. Get into the loop nest and create tvm.block.
  auto innerLoopNest = generateLoopNest(b, loc, tensorType, rDim);
  OpBuilder::InsertionGuard guard(b);
  innerLoopNest.setInsertingToInnermost(b);
  result.block = b.create<tvm::BlockOp>(loc, [&](OpBuilder &b, Location loc) {
    // 3. Map every induction var to tvm.axis.
    // All outer loops are parallel loops.
    result.outerLoopsAxes =
        mapVarsToAxes(b, outerLoopsExtents, outerLoopsVars,
                      ReductionDim::createSpatial(tensorType.getRank()));
    // We can consolidate multiple inner loops into one induction variable.
    result.innerLoopsAxes = mapVarsToAxes(
        b,
        llvm::map_to_vector(
            innerLoopNest.tensorShape,
            [&](int64_t extent) {
              return tvm::utils::getConstantOpI32(b, loc, extent).getResult();
            }),
        innerLoopNest.tensorIndices, rDim);
    result.mapper.map(outerLoopsVars, result.outerLoopsAxes);
  });

  // Do not forget to apply bufferization.to_tensor.
  return result;
}

Operation *materializeIntermediateTensor(OpBuilder &b,
                                         tensor::GenerateOp tensor) {
  auto tensorType = tensor.getType();
  auto loc = tensor.getLoc();

  auto computation = computeTensor(
      b, loc, tensorType, ReductionDim::createSpatial(tensorType.getRank()),
      tensorType);
  auto &block = computation.block;
  auto &mapper = computation.mapper;
  auto &innerLoopsAxes = computation.innerLoopsAxes;
  auto &buffer = computation.buffer;

  {
    // Enter the block.
    auto guard = computation.enterBlock(b);
    auto innerLoopsAxesAsIndexType =
        computation.getInnerLoopsAxesAsIndexType(b);

    // 4. Copy tensor.generate contents.
    auto rhs =
        computation.inlineFunction(b, tensor, innerLoopsAxesAsIndexType)[0];

    // 5. tvm.write and tvm.assign the tensor.
    auto lhs = computation.getDest(b, loc);
    b.create<tvm::WriteOp>(loc, ValueRange{lhs});
    b.create<tvm::AssignOp>(loc, lhs, rhs);
  }

  // 6. bufferization.to_tensor
  return b.create<ttm::MemRefToTensorOp>(loc, tensorType, buffer);
}

// This rewrites all tensor.generate ops. So make sure to run DCE before this so
// we do not materialize pointers, masks, and constants, e.t.c..
struct GenerateOpConverter : public OpConversionPattern<tensor::GenerateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::GenerateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, materializeIntermediateTensor(rewriter, op));
    return success();
  }
};

Operation *materializeLoadOp(OpBuilder &b, triton::LoadOp load) {
  auto tensorType = cast<RankedTensorType>(load.getType());
  auto loc = load.getLoc();

  auto computation = computeTensor(
      b, loc, tensorType, ReductionDim::createSpatial(tensorType.getRank()),
      tensorType);
  auto &block = computation.block;
  auto &mapper = computation.mapper;
  auto &innerLoopsAxes = computation.innerLoopsAxes;
  auto &buffer = computation.buffer;

  {
    // Enter the block.
    auto guard = computation.enterBlock(b);
    auto innerLoopsAxesAsIndexType =
        computation.getInnerLoopsAxesAsIndexType(b);

    auto ptrGeneratorOp =
        cast<tensor::GenerateOp>(load.getPtr().getDefiningOp());

    // 4. Copy tensor.generate contents.
    auto ptr = computation.inlineFunction(b, ptrGeneratorOp,
                                          innerLoopsAxesAsIndexType)[0];

    // In the previous pass, we have already eliminated pointer arithmetic. Now
    // we have a memref with indices.
    auto rhsMemRefPtr = ptr.getDefiningOp<ttm::MemRefToPtrOp>();
    Value rhs =
        b.create<tvm::RefOp>(rhsMemRefPtr->getLoc(), rhsMemRefPtr.getMemRef(),
                             rhsMemRefPtr.getIndices());

    // Since we add tvm.read according to tensor.extract, and all inputs are
    // already memrefs, we should insert tvm.read in advance.
    b.create<tvm::ReadOp>(loc, ValueRange{rhs});

    // Add tvm.if_then_else on demand.
    if (load.getMask()) {
      auto maskGeneratorOp =
          cast<tensor::GenerateOp>(load.getMask().getDefiningOp());
      auto otherGeneratorOp =
          cast<tensor::GenerateOp>(load.getOther().getDefiningOp());
      auto mask = computation.inlineFunction(b, maskGeneratorOp,
                                             innerLoopsAxesAsIndexType)[0];
      auto other = computation.inlineFunction(b, otherGeneratorOp,
                                              innerLoopsAxesAsIndexType)[0];
      rhs = b.create<tvm::IfThenElseOp>(loc, mask, rhs, other);
    }

    // 5. tvm.write and tvm.assign the tensor.
    auto lhs = computation.getDest(b, loc);
    b.create<tvm::WriteOp>(loc, ValueRange{lhs});
    b.create<tvm::AssignOp>(loc, lhs, rhs);
  }

  // 6. bufferization.to_tensor
  return b.create<ttm::MemRefToTensorOp>(loc, tensorType, buffer);
}

struct LoadOpConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, materializeLoadOp(rewriter, op));
    return success();
  }
};

void materializeStoreOp(OpBuilder &b, triton::StoreOp store) {
  auto tensorType = cast<RankedTensorType>(store.getPtr().getType());
  auto loc = store.getLoc();

  // 1. No need to allocate buffer, because we materialize into a destination
  // parameter.

  auto computation = computeTensor(
      b, loc, tensorType, ReductionDim::createSpatial(tensorType.getRank()),
      /*allocType=*/nullptr);
  auto &block = computation.block;
  auto &mapper = computation.mapper;
  auto &innerLoopsAxes = computation.innerLoopsAxes;

  {
    // Enter the block.
    auto guard = computation.enterBlock(b);
    auto innerLoopsAxesAsIndexType =
        computation.getInnerLoopsAxesAsIndexType(b);
    auto innerLoopsIndicesAsIndexType =
        computation.getInnerLoopsIndicesAsIndexType(b);

    auto ptrGeneratorOp =
        cast<tensor::GenerateOp>(store.getPtr().getDefiningOp());

    // 4. Copy tensor.generate contents.
    auto ptr = computation.inlineFunction(b, ptrGeneratorOp,
                                          innerLoopsAxesAsIndexType)[0];

    // Similar to load, we have a memref
    auto lhsMemRefPtr = ptr.getDefiningOp<ttm::MemRefToPtrOp>();
    Value lhs =
        b.create<tvm::RefOp>(lhsMemRefPtr->getLoc(), lhsMemRefPtr.getMemRef(),
                             lhsMemRefPtr.getIndices());

    // Add tvm.where on demand.
    if (store.getMask()) {
      auto maskGeneratorOp =
          cast<tensor::GenerateOp>(store.getMask().getDefiningOp());
      // Note that here is a bit different, because in tvm.where we do not use
      // axes, but indices.
      auto mask = computation.inlineFunction(b, maskGeneratorOp,
                                             innerLoopsIndicesAsIndexType)[0];
      b.create<tvm::WhereOp>(loc, mask);
    }

    // Extract the value to be stored.
    auto rhs = b.create<tensor::ExtractOp>(loc, store.getValue(),
                                           innerLoopsAxesAsIndexType);

    // 5. tvm.write and tvm.assign the tensor.
    b.create<tvm::WriteOp>(loc, ValueRange{lhs});
    b.create<tvm::AssignOp>(loc, lhs, rhs);
  }

  // 6. No need to bufferization.to_tensor.
}

struct StoreOpConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    materializeStoreOp(rewriter, op);
    rewriter.eraseOp(op);
    return success();
  }
};

Operation *materializeReduceOp(OpBuilder &b, triton::ReduceOp reduce) {
  assert(reduce.getNumOperands() == 1 && "expected single operand to reduce");
  auto input = reduce.getOperand(0);
  auto inputType = cast<RankedTensorType>(input.getType());
  auto scalarType = inputType.getElementType();
  auto resultType = reduce.getResultTypes()[0];
  auto loc = reduce.getLoc();

  // Consider that the result type may be a scalar. This inconsistency sucks.
  bool isResultScalar = !isa<RankedTensorType>(resultType);
  RankedTensorType resultTensorType;
  if (isResultScalar) {
    [[maybe_unused]] auto encoding =
        b.getAttr<triton::gpu::BlockedEncodingAttr>(
            ArrayRef<unsigned>{}, ArrayRef<unsigned>{}, ArrayRef<unsigned>{},
            ArrayRef<unsigned>{},
            triton::gpu::CTALayoutAttr::getDefault(b.getContext(), 0));
    resultTensorType = RankedTensorType::get({}, resultType, encoding);
  } else {
    resultTensorType = cast<RankedTensorType>(resultType);
  }

  // We do not support general reductions, because Triton implements reduction
  // as intra-warp operation, where no identity is needed. So we have to match
  // the reduction function and find the identity.
  auto reductionOps =
      llvm::map_to_vector(reduce.getBody()->without_terminator(),
                          [&](Operation &op) { return &op; });
  assert(reductionOps.size() == 1 &&
         "expected single reduction op in tt.reduce");
  auto *reductionOp = reductionOps.front();
  auto identity =
      llvm::TypeSwitch<Operation *, Value>(reductionOp)
          .Case([&](arith::AddFOp) {
            return arith::ConstantOp::materialize(
                b, b.getFloatAttr(scalarType, 0.0f), scalarType, loc);
          })
          .Case([&](arith::AddIOp) {
            return arith::ConstantOp::materialize(
                b, b.getIntegerAttr(scalarType, 0), scalarType, loc);
          })
          .Case<arith::MaximumFOp, arith::MaxNumFOp, arith::MaxSIOp,
                arith::MaxUIOp>(
              [&](auto) { return b.create<tvm::MinValueOp>(loc, scalarType); })
          .Case<arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
                arith::MinUIOp>(
              [&](auto) { return b.create<tvm::MaxValueOp>(loc, scalarType); })
          .Default([&](Operation *) {
            llvm_unreachable("This reduction is not supported");
            return nullptr;
          });

  auto computation = computeTensor(
      b, loc, inputType,
      ReductionDim::createReduction(inputType.getRank(), reduce.getAxis()),
      /*allocType=*/resultTensorType);
  auto &block = computation.block;
  auto &mapper = computation.mapper;
  auto &innerLoopsAxes = computation.innerLoopsAxes;
  auto &rDim = computation.rDim;
  auto &buffer = computation.buffer;

  {
    // Enter the block.
    auto guard = computation.enterBlock(b);
    auto innerLoopsAxesAsIndexType =
        computation.getInnerLoopsAxesAsIndexType(b);

    auto acc = computation.getDest(b, loc);

    // 5. tvm.write
    b.create<tvm::WriteOp>(loc, ValueRange{acc});

    // Insert init block.
    b.create<tvm::InitOp>(loc, [&](OpBuilder &b, Location loc) {
      b.create<tvm::AssignOp>(loc, acc, identity);
    });

    // 4. Copy reduce op contents.
    auto cur =
        b.create<tensor::ExtractOp>(loc, input, innerLoopsAxesAsIndexType);
    auto rhs = computation.inlineFunction(b, reduce, {acc, cur})[0];

    // 5. tvm.assign
    b.create<tvm::AssignOp>(loc, acc, rhs);
  }

  // 6. bufferization.to_tensor
  auto unbufferized =
      b.create<ttm::MemRefToTensorOp>(loc, resultTensorType, buffer);
  // if the result is scalar, we have to extract the value.
  if (isResultScalar) {
    auto scalar = b.create<tensor::ExtractOp>(loc, unbufferized, ValueRange{});
    return scalar;
  }
  return unbufferized;
}

struct ReduceOpConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, materializeReduceOp(rewriter, op));
    return success();
  }
};

tvm::BlockOp getOwnerBlockOp(Operation *op) {
  while (op) {
    if (auto blockOp = dyn_cast<tvm::BlockOp>(op))
      return blockOp;
    op = op->getParentOp();
  }
  return nullptr;
}

func::FuncOp getOwnerFuncOp(Operation *op) {
  while (op) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op))
      return funcOp;
    op = op->getParentOp();
  }
  return nullptr;
}

} // namespace

class MaterializeTensorsToTVMBuffers
    : public impl::MaterializeTensorsToTVMBuffersBase<
          MaterializeTensorsToTVMBuffers> {

public:
  using MaterializeTensorsToTVMBuffersBase::MaterializeTensorsToTVMBuffersBase;
  void runOnOperation() override {
    // Creating A tvm.block requires these:
    // - tvm.alloc_buffer for the materialized tensor.
    // - Map every induction var with tvm.axis to get axes.
    // - tvm.read all loaded tensor.
    // - tvm.write all stored tensor.
    // - tvm.init for reduction.
    // - tvm.if_then_else for load guard.
    // - tvm.where for store guard.
    // In this pass, we materialize tensors. So, we need to
    // 1. Create tvm.alloc_buffer for writing.
    // 2. Create a loop nest and tvm.block.
    // 3. Bind axes with tvm.axis.
    // 4. Copy contents of tensor.generate. Use IRMapping to map induction
    //    vars to the axes, also modifying all the address computing ops.
    //    * The algorithm is simple: recursively visit operands of Ops, upon
    //      induction variable, replace with axes. Otherwise, return the
    //      original Value.
    // 5. tvm.write and tvm.assign the tensor.
    // 6. Apply bufferization.to_tensor for each materialized tensor, because we
    //    still have tensor.extract.
    // 7. Then add tvm.read for each tensor.extract.
    // ** For each kind of computation,
    //    - tt.load & tt.store. Extract from ttm.memref_to_ptr memref and
    //      indices. Use tvm.if_then_else and tvm.where for guards.
    //    - tt.reduce. Note that we have to use tvm.axis reduce and tvm.init.
    //      Note that there may be scalar output, so we have to also convert
    //      that to a tensor.
    //    - tt.dot. Also tvm.axis reduce.
    //    - Others. All tensor.generate. Note that we can reject materializing
    //      pointer tensors for the time being, because we choose not to support
    //      indirection.

    lowerLoadAndStore();
    lowerAllTensors();
    readProducers();
  }

  void lowerLoadAndStore() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, math::MathDialect,
                           scf::SCFDialect, tensor::TensorDialect,
                           ttm::TritonMemRefDialect, tvm::TVMDialect>();
    target.addIllegalOp<triton::LoadOp, triton::StoreOp>();
    patterns.add<LoadOpConverter, StoreOpConverter>(&getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      moduleOp.emitError("Failed to lower tt.load and tt.store.");
      signalPassFailure();
    }
  }

  void lowerAllTensors() {
    auto moduleOp = getOperation();

    PassManager pm(&getContext(), moduleOp.getOperationName());
    // This is necessary because we need to eliminate the tensors used only by
    // tt.load and tt.store.
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, moduleOp))) {
      signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, math::MathDialect,
                           scf::SCFDialect, ttm::TritonMemRefDialect,
                           tvm::TVMDialect>();
    target.addDynamicallyLegalDialect<tensor::TensorDialect>(
        [](Operation *op) { return !isa<tensor::GenerateOp>(op); });
    target.addIllegalOp<tensor::GenerateOp, triton::ReduceOp>();
    patterns.add<GenerateOpConverter, ReduceOpConverter>(&getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      moduleOp.emitError("Failed to lower tensor.generate and tt.reduce.");
      signalPassFailure();
    }
  }

  void readProducers() {
    auto moduleOp = getOperation();

    // For each consumer BlockOp, collect the producers.
    DenseMap<tvm::BlockOp, SmallVector<Value>> producers;

    // Each producer is a tensor.extract.
    moduleOp.walk([&](tensor::ExtractOp extractOp) {
      auto memrefBackedTensor =
          extractOp.getTensor().getDefiningOp<ttm::MemRefToTensorOp>();
      assert(memrefBackedTensor &&
             "Expected all tensors to be results of memref_to_tensor");
      auto allocatedMemRef = memrefBackedTensor.getMemRef();

      // Compute the consumers. All consumers are in the forward slice.
      // For later use, we compute the owner block of each Op in the slice.
      SmallVector<std::pair<Operation *, tvm::BlockOp>> sliceAndOwner;
      {
        SetVector<Operation *> slice;
        getForwardSlice(extractOp.getOperation(), &slice);
        std::transform(slice.begin(), slice.end(),
                       std::back_inserter(sliceAndOwner), [&](Operation *op) {
                         return std::make_pair(op, getOwnerBlockOp(op));
                       });
      }

      auto inlineOpsNotInBlockAndUpdateUses =
          [&](OpBuilder &b, tvm::BlockOp blockOp,
              SetVector<Operation *> &required, IRMapping &mapper) {
            // Since the slice is already in topological order, we only need to
            // inline one by one.
            for (auto [op, ownerBlockOp] : sliceAndOwner) {
              if (required.contains(op) && ownerBlockOp != blockOp) {
                assert(isPure(op) && "expected pure op");
                assert(op->getNumRegions() == 0 && "expected no regions");
                b.cloneWithoutRegions(*op, mapper);
              }
            }
            // Then update all uses.
            for (auto [op, ownerBlockOp] : sliceAndOwner) {
              if (required.contains(op) && ownerBlockOp == blockOp) {
                SmallVector<Value> remappedOperands;
                for (auto operand : op->getOperands()) {
                  remappedOperands.push_back(mapper.lookupOrDefault(operand));
                }
                op->setOperands(remappedOperands);
              }
            }
          };

      OpBuilder b(extractOp);
      // Note that we have to cast the indices to i32.
      auto indices = Computation::getIndexArrayAsType(b, extractOp.getIndices(),
                                                      b.getI32Type());

      auto allocation = allocatedMemRef.getDefiningOp<tvm::AllocBufferOp>();
      auto outerLoopNest = collectLoopsInScope(allocation->getBlock());
      auto outerLoopsVars = getInductionVarsFromLoopNest(outerLoopNest);

      // For each corresponding BlockOp, add its producer.
      for (auto [op, ownerBlockOp] : sliceAndOwner) {
        if (auto assignOp = dyn_cast<tvm::AssignOp>(op)) {
          auto blockOp = assignOp.getBlockOp();
          // Note that we have to use outer loops axes to access the buffer.
          auto fullIndices =
              getOuterLoopsAxesInBlockFromVars(blockOp, outerLoopsVars);
          std::copy(indices.begin(), indices.end(),
                    std::back_inserter(fullIndices));
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPoint(blockOp.getBody(),
                              getPointInBlockWhereAllAxesAreDefined(blockOp));
          // For later use. Only ops that are in the backward slice of the
          // assign op are required.
          SetVector<Operation *> required;
          getBackwardSlice(assignOp, &required);
          // Now that the slice is collected, we can erase the extract op,
          // and replace with standard tvm.ref, which has no
          // consumer/producer information.
          auto refOp = b.create<tvm::RefOp>(extractOp.getLoc(), allocatedMemRef,
                                            fullIndices);
          producers[blockOp].push_back(refOp);
          // And we have to replace all uses of the extract op with the ref op.
          IRMapping mapper;
          mapper.map(extractOp.getResult(), refOp.getResult());
          inlineOpsNotInBlockAndUpdateUses(b, blockOp, required, mapper);
        }
      }
    });

    // Then add tvm.read directives for the producers.
    moduleOp.walk([&](tvm::BlockOp blockOp) {
      if (auto it = producers.find(blockOp); it != producers.end()) {
        auto &refs = it->second;
        auto assignOp = blockOp.getAssignOp();
        assert(!blockOp.getReadOp() && "Expected no existing read op.");
        OpBuilder b(assignOp);
        b.create<tvm::ReadOp>(assignOp.getLoc(), refs);
      }
    });

    // Finally, promote the buffers to the outermost scope.
    SmallVector<tvm::AllocBufferOp> allocations;
    moduleOp.walk(
        [&](tvm::AllocBufferOp alloc) { allocations.push_back(alloc); });
    for (auto alloc : allocations) {
      auto func = getOwnerFuncOp(alloc);
      auto &entryBlock = func.getFunctionBody().front();
      alloc->moveBefore(&entryBlock, entryBlock.begin());
    }
  }
};

} // namespace mlir::triton

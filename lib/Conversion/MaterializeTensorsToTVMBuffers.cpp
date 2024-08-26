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

  void setInsertingToInnermost(OpBuilder &b) {
    if (loops.empty())
      return;
    b.setInsertionPoint(loops.back().getBody()->getTerminator());
  }
};

// TODO: add vectorize option.
LoopNest generateLoopNest(OpBuilder &b, Location loc, RankedTensorType tensor) {
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
        spec.kind = tvm::ForKind::UNROLL;
        break;
      case STAGE_SIZE_PER_THREAD:
        spec.extent = layoutSizePerThread[dim];
        spec.kind =
            order == 0 ? tvm::ForKind::VECTORIZED : tvm::ForKind::UNROLL;
        break;
      default:
        llvm_unreachable("unexpected stage");
      }
      if (spec.extent > 1)
        specs.push_back(spec);
    }
  }
  auto loops = generateLoopNestFromSpecs(b, loc, specs);

  // Then recover the tensor indices.
  OpBuilder::InsertionGuard guard(b);
  if (!loops.empty())
    b.setInsertionPointToStart(loops.back().getBody());
  auto zero = tvm::utils::getConstantOpI32(b, loc, 0);
  SmallVector<Value> tensorIndices(tensor.getRank(), zero);
  SmallVector<unsigned> strides(tensor.getRank(), 1);
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
  return LoopNest{std::move(loops), std::move(tensorIndices)};
}

// Traverse the loop nest and collect.
SmallVector<scf::ForOp> collectLoopsInScope(OpBuilder &b) {
  SmallVector<scf::ForOp> loops;
  auto *block = b.getBlock();
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
  SmallVector<Value> inductionVars;
  for (auto loop : loops) {
    inductionVars.push_back(loop.getInductionVar());
  }
  return inductionVars;
}

SmallVector<Value> getExtentsFromLoopNest(SmallVectorImpl<scf::ForOp> &loops) {
  SmallVector<Value> extents;
  for (auto loop : loops) {
    extents.push_back(loop.getUpperBound());
  }
  return extents;
}

SmallVector<Value> mapVarsToSpatialAxes(OpBuilder &b, ValueRange extents,
                                        ValueRange vars) {
  SmallVector<Value> axes;
  for (auto [extent, var] : llvm::zip(extents, vars)) {
    axes.push_back(b.create<tvm::AxisOp>(
        var.getLoc(), b.getAttr<tvm::AxisKindAttr>(tvm::AxisKind::SPATIAL),
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

// The first 3 steps are done for you. Now carry on.
struct Materialization {
  // Allocated buffer.
  tvm::AllocBufferOp buffer;
  // The created computation block.
  tvm::BlockOp block;
  // The outer loop nest has already been mapped.
  IRMapping mapper;
  // With tvm.axis applied.
  SmallVector<Value> innerLoopsAxes;

  // Because we currently uses i32 as indices, we have to cast innerLoopsAxes to
  // index so that they can be used to inline the generator.
  SmallVector<Value> getInnerLoopsAxesAsIndexType(OpBuilder &b) const {
    SmallVector<Value> innerLoopsAxesAsIndexType;
    for (auto axis : innerLoopsAxes) {
      innerLoopsAxesAsIndexType.push_back(
          b.create<arith::IndexCastOp>(axis.getLoc(), b.getIndexType(), axis));
    }
    return innerLoopsAxesAsIndexType;
  }

  SmallVector<Value>
  inlineGenerator(OpBuilder &b, tensor::GenerateOp generateOp,
                  ArrayRef<Value> innerLoopsAxesAsIndexType) {
    auto &generator = generateOp.getBody().front();
    mapper.map(generator.getArguments(), innerLoopsAxesAsIndexType);
    return inlineRecursively(b, generator, mapper);
  }

  tvm::RefOp getDest(OpBuilder &b, Location loc) {
    return b.create<tvm::RefOp>(loc, buffer, innerLoopsAxes);
  }
};

Materialization materializeTensor(OpBuilder &b, Location loc,
                                  RankedTensorType tensorType) {
  Materialization result;

  // 1. Allocate buffer (and apply bufferization.to_tensor later).
  result.buffer = b.create<tvm::AllocBufferOp>(
      loc, MemRefType::get(tensorType.getShape(), tensorType.getElementType()),
      b.getStringAttr("shared"));

  // 2. Get into the loop nest and create tvm.block.
  {
    OpBuilder::InsertionGuard guard(b);
    auto outerLoopNest = collectLoopsInScope(b);
    auto outerLoopVars = getInductionVarsFromLoopNest(outerLoopNest);
    auto innerLoopNest = generateLoopNest(b, loc, tensorType);
    innerLoopNest.setInsertingToInnermost(b);
    result.block = b.create<tvm::BlockOp>(loc, [&](OpBuilder &b, Location loc) {
      // 3. Map every induction var to tvm.axis.
      // Now we pretend there are only parallel dims. TODO: support reduction
      // dims.
      auto outerLoopsAxes = mapVarsToSpatialAxes(
          b, outerLoopVars, getExtentsFromLoopNest(outerLoopNest));
      // We can consolidate multiple inner loops into one induction variable.
      result.innerLoopsAxes =
          mapVarsToSpatialAxes(b, innerLoopNest.tensorIndices,
                               getExtentsFromLoopNest(innerLoopNest.loops));
      result.mapper.map(outerLoopVars, outerLoopsAxes);
    });
  }

  // Do not forget to apply bufferization.to_tensor.
  return result;
}

void materializeIntermediateTensor(OpBuilder &b, tensor::GenerateOp tensor) {
  auto tensorType = tensor.getType();
  auto loc = tensor.getLoc();
  auto materialization = materializeTensor(b, loc, tensorType);
  auto &[buffer, block, mapper, innerLoopsAxes] = materialization;

  {
    // Enter the block.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(block.getBody());
    auto innerLoopsAxesAsIndexType =
        materialization.getInnerLoopsAxesAsIndexType(b);

    // 4. Copy tensor.generate contents.
    auto rhs = materialization.inlineGenerator(b, tensor,
                                               innerLoopsAxesAsIndexType)[0];

    // 5. tvm.write and tvm.assign the tensor.
    auto lhs = materialization.getDest(b, loc);
    b.create<tvm::WriteOp>(loc, ValueRange{lhs});
    b.create<tvm::AssignOp>(loc, lhs, rhs);
  }

  // 6. bufferization.to_tensor
  auto unbufferized =
      b.create<bufferization::ToTensorOp>(loc, buffer, b.getUnitAttr());
  tensor->replaceAllUsesWith(unbufferized->getResults());
}

void materializeLoadOp(OpBuilder &b, triton::LoadOp load) {
  auto tensorType = cast<RankedTensorType>(load.getType());
  auto loc = load.getLoc();
  auto materialization = materializeTensor(b, loc, tensorType);
  auto &[buffer, block, mapper, innerLoopsAxes] = materialization;

  {
    // Enter the block.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(block.getBody());
    auto innerLoopsAxesAsIndexType =
        materialization.getInnerLoopsAxesAsIndexType(b);

    auto ptrGeneratorOp =
        cast<tensor::GenerateOp>(load.getPtr().getDefiningOp());

    // 4. Copy tensor.generate contents.
    auto ptr = materialization.inlineGenerator(b, ptrGeneratorOp,
                                               innerLoopsAxesAsIndexType)[0];

    // In the previous pass, we have already eliminated pointer arithmetic. Now
    // we have a memref with indices.
    auto rhsMemRefPtr = cast<ttm::MemRefToPtrOp>(ptr.getDefiningOp());
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
      auto mask = materialization.inlineGenerator(b, maskGeneratorOp,
                                                  innerLoopsAxesAsIndexType)[0];
      auto other = materialization.inlineGenerator(
          b, otherGeneratorOp, innerLoopsAxesAsIndexType)[0];
      rhs = b.create<tvm::IfThenElseOp>(loc, mask, rhs, other);
    }

    // 5. tvm.write and tvm.assign the tensor.
    auto lhs = materialization.getDest(b, loc);
    b.create<tvm::WriteOp>(loc, ValueRange{lhs});
    b.create<tvm::AssignOp>(loc, lhs, rhs);
  }

  // 6. bufferization.to_tensor
  auto unbufferized =
      b.create<bufferization::ToTensorOp>(loc, buffer, b.getUnitAttr());
  load->replaceAllUsesWith(unbufferized->getResults());
  // Remove the load.
  load->erase();
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
    // ** For each kind of materialization,
    //    - tt.load & tt.store. Extract from ttm.memref_to_ptr memref and
    //      indices. Use tvm.if_then_else and tvm.where for guards.
    //    - tt.reduce. Note that we have to use tvm.axis reduction and tvm.init.
    //      Note that there may be scalar output, so we have to also convert
    //      that to a tensor.
    //    - tt.dot. Also tvm.axis reduction.
    //    - Others. All tensor.generate. Note that we can reject materializing
    //      pointer tensors for the time being, because we choose not to support
    //      indirection.
    auto moduleOp = getOperation();
    moduleOp.walk([&](tensor::GenerateOp tensor) {
      for (auto *user : tensor->getUsers()) {
        if (isa<triton::LoadOp>(user) || isa<triton::StoreOp>(user))
          return;
      }
      OpBuilder builder(&getContext());
      builder.setInsertionPointAfter(tensor);
      materializeIntermediateTensor(builder, tensor);
    });
    moduleOp.walk([&](triton::LoadOp load) {
      OpBuilder builder(&getContext());
      builder.setInsertionPointAfter(load);
      materializeLoadOp(builder, load);
    });
  }
};

} // namespace mlir::triton

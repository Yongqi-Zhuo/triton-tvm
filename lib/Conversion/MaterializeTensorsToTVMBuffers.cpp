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

SmallVector<tvm::AxisOp> mapVarsToSpatialAxes(OpBuilder &b, ValueRange vars) {
  SmallVector<tvm::AxisOp> axes;
  for (Value var : vars) {
    axes.push_back(b.create<tvm::AxisOp>(
        var.getLoc(), b.getAttr<tvm::AxisKindAttr>(tvm::AxisKind::SPATIAL),
        var));
  }
  return axes;
}

// Returns true if the op is cloned.
bool inlineRecursivelyImpl(OpBuilder &b, Value value, IRMapping &mapper) {
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
  bool needsCloning = false;
  for (Value operand : op->getOperands()) {
    needsCloning |= inlineRecursivelyImpl(b, operand, mapper);
  }
  if (needsCloning) {
    // At least one of the operands was rewritten. Clone the op.
    // The results are implicitly mapped.
    op = b.cloneWithoutRegions(*op, mapper);
    return true;
  }
  // No need to clone. Just update the mapping.
  mapper.map(op->getResults(), op->getResults());
  return false;
}

// Return yielded value.
SmallVector<Value> inlineRecursively(OpBuilder &b, Block &block,
                                     IRMapping &mapper) {
  auto *terminator = block.getTerminator();
  assert(terminator->hasTrait<OpTrait::ReturnLike>());
  SmallVector<Value> results;
  for (Value operand : terminator->getOperands()) {
    inlineRecursivelyImpl(b, operand, mapper);
    results.push_back(mapper.lookup(operand));
  }
  return results;
}

void materializeGenerateOp(OpBuilder &b, tensor::GenerateOp tensor) {
  auto tensorType = tensor.getType();
  // 1. Allocate buffer (and apply bufferization.to_tensor later).
  auto buffer = b.create<tvm::AllocBufferOp>(
      tensor.getLoc(),
      MemRefType::get(tensorType.getShape(), tensorType.getElementType()),
      b.getStringAttr("shared"));

  // 2. Get into the loop nest and create tvm.block.
  {
    OpBuilder::InsertionGuard guard(b);
    auto outerLoopNest = collectLoopsInScope(b);
    auto outerLoopVars = getInductionVarsFromLoopNest(outerLoopNest);
    auto innerLoopNest = generateLoopNest(b, tensor.getLoc(), tensorType);
    innerLoopNest.setInsertingToInnermost(b);
    b.create<tvm::BlockOp>(tensor.getLoc(), [&](OpBuilder &b, Location loc) {
      // 3. Map every induction var to tvm.axis.
      // Now we pretend there are only parallel dims. TODO: support reduction
      // dims.
      auto outerLoopAxes = mapVarsToSpatialAxes(b, outerLoopVars);
      // We can consolidate multiple inner loops into one induction variable.
      auto innerLoopAxes = mapVarsToSpatialAxes(b, innerLoopNest.tensorIndices);

      // 4. Copy tensor.generate contents.
      // The algorithm is simple: recursively visit operands of Ops, upon
      // induction variable, replace with axes. Otherwise, return the original
      // Value.
      IRMapping mapper;
      mapper.map(outerLoopVars, outerLoopAxes);
      mapper.map(tensor.getBody().getArguments(), innerLoopAxes);
      inlineRecursively(b, tensor.getBody().front(), mapper);
    });
  }

  // Do not forget to apply bufferization.to_tensor.
  auto unbufferized = b.create<bufferization::ToTensorOp>(
      tensor.getLoc(), buffer, b.getUnitAttr());
  tensor->replaceAllUsesWith(unbufferized->getResults());
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
    // 1. Create tvm.alloc_buffer for writing, and bufferization.to_tensor so it
    //    can still be read.
    // 2. Create a loop nest and tvm.block.
    // 3. Bind axes with tvm.axis.
    // 4. Copy contents of tensor.generate. Use IRMapping to map induction
    //    vars to the axes, also modifying all the address computing ops.
    // 5. tvm.write the tensor.
    // 6. For each kind of materialization,
    //    - tt.load & tt.store. Extract from ttm.memref_to_ptr memref and
    //      indices. Use tvm.if_then_else and tvm.where for guards.
    //    - tt.reduce. Note that we have to use tvm.axis reduction and tvm.init.
    //    - tt.dot. Also tvm.axis reduction. Note that there may be scalar
    //      output, so we have to also convert that to a tensor.
    //    - Others. All tensor.generate. Note that we can reject materializing
    //      pointer tensors for the time being, because we choose not to support
    //      indirection.
    // 7. Then add tvm.read for each tensor.extract.
  }
};

} // namespace mlir::triton

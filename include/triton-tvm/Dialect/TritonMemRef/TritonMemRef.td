#ifndef TRITON_MEMREF_DIALECT
#define TRITON_MEMREF_DIALECT

include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "triton/Dialect/Triton/IR/TritonTypes.td"


def TritonMemRef_Dialect : Dialect {
  let name = "ttm";
  let summary = "Bridge between Triton and MemRef.";
  let description = [{
    This dialect is only for conversion.
  }];
  let cppNamespace = "::mlir::ttm";
}

def TritonMemRef_MemRefToPtrOp : Op<
  TritonMemRef_Dialect,
  "memref_to_ptr",
  [Pure]
> {

  let summary = "Converts a memref access to a Triton pointer.";

  let description = [{
    Used in when rewriting tt.addptr operands.
  }];

  // TODO: Do not use i32! Instead, promote all I32 to Index.
  let arguments = (ins AnyRankedOrUnrankedMemRef:$memref,
                       Variadic<I32>:$indices);

  let results = (outs TT_Ptr:$result);

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    BaseMemRefType getMemRefType() {
      return ::llvm::cast<BaseMemRefType>(getMemRef().getType());
    }
  }];

  let assemblyFormat = [{
    $memref `[` $indices `]` attr-dict `:`
    type($memref) `to` type($result)
  }];

  let hasVerifier = 1;

}

def TritonMemRef_MemRefToTensorOp : Op<
  TritonMemRef_Dialect,
  "memref_to_tensor",
  [
    Pure,
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
  ]
> {

  let summary = "Converts a memref to a tensor.";

  let description = [{
    bufferization.to_tensor imposes too strong requirements on the result tensor,
    that it cannot keep any encoding attributes. This op relaxes such requirements.
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$memref);

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $memref attr-dict `:` type($memref) `to` type($result)
  }];

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(); }
    void setMemRef(Value value) { setOperand(value); }
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemRef().getType());
    }
  }];

}

#endif // TRITON_MEMREF_DIALECT

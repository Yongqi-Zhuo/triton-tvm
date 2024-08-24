#ifndef TVM_OPS
#define TVM_OPS

include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "triton-tvm/Dialect/TVM/IR/Attributes.td"
include "triton-tvm/Dialect/TVM/IR/Dialect.td"
include "triton-tvm/Dialect/TVM/IR/Types.td"


class TVM_Op<string mnemonic, list<Trait> traits = []>
    : Op<TVM_Dialect, mnemonic, traits>;

def TVM_FuncOp : TVM_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {

  let summary = "Equivalent to T.prim_func in TVM TensorIR.";

  let description = [{
    Declares a function with the specified name, arguments and region.

    Example:
    ```mlir
      tvm.func @main(%A: memref<5x10xf32>, %B: memref<5x10xf32>) {
        // Region
      }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;

}

def ReturnOp : TVM_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {

  let summary = "The return operation.";

  let description = [{
    This produces no results.

    Example:
    ```mlir
      tvm.func @foo() {
        ...
        tvm.return
      }
    ```
  }];

  let arguments = (ins);

  let assemblyFormat = "attr-dict";

}

def TVM_VarOp : TVM_Op<"var", [
  MemoryEffects<[MemWrite<DefaultResource>]>,
]> {

  let summary = "Equivalent to T.var() in TVM TensorIR.";

  let description = [{
    Declares a variable.

    Example:
    ```mlir
      %0 = tvm.var : index
    ```
  }];

  let arguments = (ins);

  let results = (outs AnyInteger:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

}

def TVM_MatchBufferOp : TVM_Op<"match_buffer", [
  MemoryEffects<[MemRead<DefaultResource>]>,
  MemoryEffects<[MemWrite<DefaultResource>]>,
]> {

  let summary = "Equivalent to T.match_buffer in TVM TensorIR.";

  let description = [{
    Matches a buffer.

    Example:
    ```mlir
      %2 = tvm.match_buffer %0 to [%1, 128] : memref<?x128xf32>
    ```
  }];

  // TODO: Do not use i32! Instead, promote all I32 to Index.
  let arguments = (ins AnyRankedOrUnrankedMemRef:$source,
                       Variadic<I32>:$sizes,
                       DenseI64ArrayAttr:$static_sizes);

  let results = (outs AnyMemRef:$result);

  let builders = [
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$sizes,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let assemblyFormat = [{
    $source `to`
    custom<DynamicIndexList>($sizes, $static_sizes)
    attr-dict `:` type($source) `to` type($result)
  }];

}

def TVM_AllocBufferOp : TVM_Op<"alloc_buffer"> {

  let summary = "Equivalent to T.alloc_buffer in TVM TensorIR.";

  let description = [{
    Allocates a buffer in the memory space specified by the scope attribute.

    Example:
    ```mlir
      %0 = tvm.alloc_buffer() {scope = "global"} : memref<5x10xf32>
    ```
  }];

  let arguments = (ins StrAttr:$scope);

  let results = (outs Res<AnyMemRef, "",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>:$memref);

  let extraClassDeclaration = [{
    MemRefType getType() { return ::llvm::cast<MemRefType>(getResult().getType()); }
  }];

  let assemblyFormat = "`(` `)` attr-dict `:` type($memref)";

}

def TVM_BlockOp : TVM_Op<"block", [
  NoTerminator,
  SingleBlock,
  RecursiveMemoryEffects,
]> {

  let summary = "Equivalent to T.block in TVM TensorIR.";

  let description = [{
    Declares a block for computation.

    Example:
    ```mlir
      tvm.block {
        // Region
      }
    ```
  }];

  let arguments = (ins);

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
    CArg<"function_ref<void(OpBuilder &, Location)>", "nullptr">:$bodyBuilder
  )>];

  let assemblyFormat = "regions attr-dict";

}

def TVM_WhereOp : TVM_Op<"where", [
  MemoryEffects<[MemRead<DefaultResource>]>,
  MemoryEffects<[MemWrite<DefaultResource>]>,
]> {

  let summary = "Equivalent to T.where in TVM TensorIR.";

  let description = [{
    Declares a where predicate.

    Example:
    ```mlir
      tvm.where %0
    ```
  }];

  let arguments = (ins I1:$condition);

  let assemblyFormat = "$condition attr-dict";

}

def TVM_AxisOp : TVM_Op<"axis", [Pure, SameOperandsAndResultType]> {

  let summary = "Equivalent to T.axis in TVM TensorIR.";

  let description = [{
    Bind an axis.

    Example:
    ```mlir
      %2 = tvm.axis spatial %0 : index
      %3 = tvm.axis reduce %1 : index
    ```
  }];

  let arguments = (ins TVM_AxisKind:$axis_kind, AnyInteger:$binding);

  let results = (outs AnyInteger:$result);

  let assemblyFormat = "$axis_kind $binding attr-dict `:` type($result)";

}

def TVM_RefOp : TVM_Op<"ref", [
  Pure,
  TypesMatchWith<
    "result type matches element type of 'memref'",
    "memref", "result",
    "::llvm::cast<MemRefType>($_self).getElementType()"
  >,
]> {

  let summary = "Equivalent to tensor access in TVM TensorIR.";

  let description = [{
    Access a tensor. This can later be used in read/write operations.

    Example:
    ```mlir
      %2 = tvm.ref %0[%1] : memref<5x10xf32>
    ```
  }];

  // TODO: Do not use i32! Instead, promote all I32 to Index.
  let arguments = (ins AnyMemRef:$memref,
                       Variadic<I32>:$indices);
  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemRef().getType());
    }
  }];

  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)";

}

def TVM_IfThenElseOp : TVM_Op<"if_then_else", [
  Pure,
  AllTypesMatch<["true_value", "false_value", "result"]>
]> {

  let summary = "Equivalent to T.if_then_else in TVM TensorIR.";

  let description = [{
    Conditional expression.

    Example:
    ```mlir
      %3 = tvm.if_then_else %0, %1, %2 : f32
    ```
  }];

  let arguments = (ins I1:$condition, AnyType:$true_value, AnyType:$false_value);

  let results = (outs AnyType:$result);

  let assemblyFormat = "operands attr-dict `:` type($result)";

}

def TVM_ReadOp : TVM_Op<"read", [
  MemoryEffects<[MemRead<DefaultResource>]>,
  MemoryEffects<[MemWrite<DefaultResource>]>,
]> {

  let summary = "Equivalent to T.reads in TVM TensorIR.";

  let description = [{
    Load from a tensor.

    Example:
    ```mlir
      tvm.read %0 : f32
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$refs);

  let assemblyFormat = "attr-dict ($refs^ `:` type($refs))?";

  // TODO: Verify that the operand is a ref.

}

def TVM_WriteOp : TVM_Op<"write", [
  MemoryEffects<[MemRead<DefaultResource>]>,
  MemoryEffects<[MemWrite<DefaultResource>]>,
]> {

  let summary = "Equivalent to T.writes in TVM TensorIR.";

  let description = [{
    Store to a tensor.

    Example:
    ```mlir
      tvm.write %0 : f32
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$refs);

  let assemblyFormat = "attr-dict ($refs^ `:` type($refs))?";

  // TODO: Verify that the operand is a ref.

}

def TVM_AssignOp : TVM_Op<"assign", [
  AllTypesMatch<["lhs", "rhs"]>,
  MemoryEffects<[MemRead<DefaultResource>]>,
  MemoryEffects<[MemWrite<DefaultResource>]>,
]> {

  let summary = "Equivalent to an assignment in TVM TensorIR.";

  let description = [{
    Make an assignment.

    Example:
    ```mlir
      tvm.assign %0 = %1 : f32
    ```
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let assemblyFormat = "attr-dict $lhs `=` $rhs `:` type($lhs)";

}

def TVM_InitOp : TVM_Op<"init", [
  NoTerminator,
  SingleBlock,
  RecursiveMemoryEffects,
]> {

  let summary = "Equivalent to T.init in TVM TensorIR.";

  let description = [{
    Declares an init block for initialization.

    Example:
    ```mlir
      tvm.init {
        // Region
      }
    ```
  }];

  let arguments = (ins);

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
    CArg<"function_ref<void(OpBuilder &, Location)>", "nullptr">:$bodyBuilder
  )>];

  let assemblyFormat = "regions attr-dict";

}

#endif

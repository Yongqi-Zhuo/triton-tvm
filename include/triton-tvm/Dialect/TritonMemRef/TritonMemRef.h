#ifndef MLIR_DIALECT_TRITONMEMREF_DIALECT_H
#define MLIR_DIALECT_TRITONMEMREF_DIALECT_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-tvm/Dialect/TritonMemRef/TritonMemRefDialect.h.inc"

#define GET_OP_CLASSES
#include "triton-tvm/Dialect/TritonMemRef/TritonMemRefOps.h.inc"

#endif // MLIR_DIALECT_TRITONMEMREF_DIALECT_H

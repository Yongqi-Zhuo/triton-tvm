#ifndef MLIR_DIALECT_TVM_IR_DIALECT_H
#define MLIR_DIALECT_TVM_IR_DIALECT_H

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

#include "mlir/IR/Dialect.h"

#include "triton-tvm/Dialect/TVM/IR/Attributes.h.inc"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "triton-tvm/Dialect/TVM/IR/Ops.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-tvm/Dialect/TVM/IR/Types.h.inc"

#endif // MLIR_DIALECT_TVM_IR_DIALECT_H

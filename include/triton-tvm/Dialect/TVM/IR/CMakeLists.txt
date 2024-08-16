set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(Attributes.h.inc -gen-enum-decls)
mlir_tablegen(Attributes.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tvm)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tvm)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=tvm)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=tvm)

add_public_tablegen_target(TVMTableGen)

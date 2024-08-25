set(LLVM_TARGET_DEFINITIONS TritonMemRef.td)
mlir_tablegen(TritonMemRefDialect.h.inc -gen-dialect-decls -dialect=ttm)
mlir_tablegen(TritonMemRefDialect.cpp.inc -gen-dialect-defs -dialect=ttm)

set(LLVM_TARGET_DEFINITIONS TritonMemRef.td)
mlir_tablegen(TritonMemRefOps.h.inc -gen-op-decls)
mlir_tablegen(TritonMemRefOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(TritonMemRefTableGen)

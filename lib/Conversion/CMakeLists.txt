add_triton_library(TritonGPUToTVM
  LowerToTensorIdioms.cpp
  ReplaceTritonPointersWithMemRefs.cpp
  RewriteSPMDToLoops.cpp
  TritonGPUToTVM.cpp

  DEPENDS
  TritonMemRefTableGen
  TVMTableGen
  TritonGPUToTVMConversionPassIncGen
  TVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonMemRef
  TVMIR
  TVMTransforms
)

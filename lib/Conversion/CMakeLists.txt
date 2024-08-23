add_triton_library(TritonGPUToTVM
  RewriteSPMDToLoops.cpp
  TritonGPUToTVM.cpp

  DEPENDS
  TVMTableGen
  TritonGPUToTVMConversionPassIncGen

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
)

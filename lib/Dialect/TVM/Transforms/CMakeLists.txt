add_triton_library(TVMTransforms
  ConvertToTVMScript.cpp

  DEPENDS
  TVMTableGen
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
)

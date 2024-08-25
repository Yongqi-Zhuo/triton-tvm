add_triton_library(TritonMemRef
  TritonMemRef.cpp

  DEPENDS
  TritonMemRefTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)

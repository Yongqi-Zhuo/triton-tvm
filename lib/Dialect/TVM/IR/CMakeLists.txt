add_triton_library(TVMIR
  Attributes.cpp
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TVMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)

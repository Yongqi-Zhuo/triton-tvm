option(TRITON_TVM_BUILD_BACKEND "Build triton-tvm backend" ON)

set(TRITON_TVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_TVM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)

if (TRITON_TVM_BUILD_BACKEND)
    add_triton_plugin(TritonTVM ${CMAKE_CURRENT_SOURCE_DIR}/triton_tvm.cpp LINK_LIBS TritonGPUToTVM TVMIR)
endif()

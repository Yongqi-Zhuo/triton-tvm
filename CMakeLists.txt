option(TRITON_TVM_BUILD_BACKEND "Build triton-tvm backend" ON)

set(TRITON_TVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_TVM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-deprecated-declarations")
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)

if (TRITON_TVM_BUILD_BACKEND)
    add_triton_plugin(TritonTVM ${CMAKE_CURRENT_SOURCE_DIR}/triton_tvm.cpp LINK_LIBS TritonGPUToTVM TVMTransforms TVMIR)
    target_link_libraries(TritonTVM PRIVATE Python3::Module pybind11::headers)
endif()

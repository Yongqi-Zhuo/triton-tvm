#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton-tvm/Conversion/TritonGPUToTVM/TritonGPUToTVM.h"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_tvm_passes_ttgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_convert_to_tvm",
                     mlir::triton::gpu::createConvertTritonGPUToTVMPass);
}

void init_triton_triton_tvm(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_tvm_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::tvm::TVMDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}

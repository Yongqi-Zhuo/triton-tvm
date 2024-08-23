#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton-tvm/Conversion/TritonGPUToTVM/Passes.h"
#include "triton-tvm/Dialect/TVM/IR/Dialect.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_tvm_passes_ttgpuir(py::module &&m) {
  m.def("add_convert_to_tvm", [](mlir ::PassManager &pm, std::vector<int> val0,
                                 std::vector<std::vector<int>> val1,
                                 std::vector<std::vector<int>> val2) {
    llvm::SmallVector<int> gridDim(val0.begin(), val0.end());
    llvm::SmallVector<llvm::SmallVector<int>> tensorShapes;
    for (auto &v : val1) {
      tensorShapes.emplace_back(llvm::SmallVector<int>(v.begin(), v.end()));
    }
    llvm::SmallVector<llvm::SmallVector<int>> tensorStrides;
    for (auto &v : val2) {
      tensorStrides.emplace_back(llvm::SmallVector<int>(v.begin(), v.end()));
    }
    pm.addPass(mlir ::triton ::gpu ::createConvertTritonGPUToTVMPass(
        gridDim, tensorShapes, tensorStrides));
  });
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

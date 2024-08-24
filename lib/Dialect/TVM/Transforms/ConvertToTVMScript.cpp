#include <fstream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "triton-tvm/Dialect/TVM/IR/Dialect.h"
#include "triton-tvm/Dialect/TVM/Transforms/Passes.h"
#include "triton-tvm/Utils/Python.hpp"

#define DEBUG_TYPE "convert-to-tvmscript"

namespace mlir::tvm {

#define GEN_PASS_DEF_CONVERTTOTVMSCRIPT
#include "triton-tvm/Dialect/TVM/Transforms/Passes.h.inc"

class ConvertToTVMScriptPass
    : public impl::ConvertToTVMScriptBase<ConvertToTVMScriptPass> {

  using Printer =
      decltype(PythonCodePrinter(std::declval<std::ofstream &>(), 0));

  DenseMap<Value, std::string> valueNames;

  llvm::SmallDenseMap<Type, std::string> typeNames;

  std::string getTypeName(Type type) const { return typeNames.at(type); }

  std::string getValueName(Value value) {
    auto it = valueNames.find(value);
    if (it != valueNames.end())
      return it->second;
    std::string name = "v" + std::to_string(valueNames.size());
    valueNames.try_emplace(value, name);
    return name;
  }

  void printStaticShape(Printer &printer, MemRefType type) {
    for (auto dim : type.getShape()) {
      printer.write("T.int32({}), ", dim);
    }
  }

  void printArgType(Printer &printer, Type type) {
    if (!isa<MemRefType>(type)) {
      return signalPassFailure();
    }
    auto memrefType = cast<MemRefType>(type);
    if (memrefType.hasStaticShape()) {
      // Static shape, example: T.Buffer((T.int32(10), T.int32(20)), "float32")
      printer.write("T.Buffer");
      printer.parens([&] {
        printer.parens([&] { printStaticShape(printer, memrefType); });
        printer.write(", \"{}\"", getTypeName(memrefType.getElementType()));
      });
    } else {
      // Dynamic shape
      printer.write("T.handle");
    }
  }

  void visitVarOp(Printer &printer, tvm::VarOp varOp) {}

  void visitMatchBufferOp(Printer &printer, tvm::MatchBufferOp matchOp) {}

  void visitAllocBufferOp(Printer &printer, tvm::AllocBufferOp allocOp) {}

  void visitForOp(Printer &printer, scf::ForOp forOp) {}

  void visitBlockOp(Printer &printer, tvm::BlockOp blockOp) {}

  void visitWhereOp(Printer &printer, tvm::WhereOp whereOp) {}

  void visitAxisOp(Printer &printer, tvm::AxisOp axisOp) {}

  void visitRefOp(Printer &printer, tvm::RefOp refOp) {}

  void visitIfThenElseOp(Printer &printer, tvm::IfThenElseOp ifOp) {}

  void visitReadOp(Printer &printer, tvm::ReadOp readOp) {}

  void visitWriteOp(Printer &printer, tvm::WriteOp writeOp) {}

  void visitAssignOp(Printer &printer, tvm::AssignOp assignOp) {}

  void visitInitOp(Printer &printer, tvm::InitOp initOp) {}

public:
  using ConvertToTVMScriptBase::ConvertToTVMScriptBase;

  void runOnOperation() override {
    typeNames = {
        {IndexType::get(&getContext()), "int64"},
        {IntegerType::get(&getContext(), 64), "int64"},
        {IntegerType::get(&getContext(), 32), "int32"},
        {IntegerType::get(&getContext(), 16), "int16"},
        {FloatType::getF64(&getContext()), "float64"},
        {FloatType::getF32(&getContext()), "float32"},
        {FloatType::getF16(&getContext()), "float16"},
    };

    auto moduleOp = getOperation();
    auto ofstream = std::ofstream(outputFilename);
    Printer printer(ofstream, 0);

    printer.writeLn("import tvm");
    printer.writeLn("import tvm.script");
    printer.writeLn("from tvm.script import tir as T");
    printer.writeLn();
    printer.writeLn("@tvm.script.ir_module");
    printer.writeLn("class Module:");
    printer.indent([&] {
      for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
        printer.writeLn("@T.prim_func");
        printer.write("def {}", funcOp.getName().str());
        printer.parens([&] {
          for (auto arg : funcOp.getArguments()) {
            printer.write("{}: ", getValueName(arg));
            printArgType(printer, arg.getType());
            printer.writeLn(", ");
          }
        });
        printer.writeLn(":");
        printer.indent([&] {
          printer.writeLn(R"code(T.func_attr({{"tir.noalias": True}}))code");
          if (!funcOp.getBody().hasOneBlock()) {
            funcOp.emitError("expected function to have a single block");
            return signalPassFailure();
          }
          auto &entryBlock = funcOp.getBody().front();
          for (Operation &op : entryBlock) {
            if (isa<tvm::VarOp>(op)) {
              visitVarOp(printer, cast<tvm::VarOp>(op));
            }
            if (isa<tvm::AllocBufferOp>(op)) {
              visitAllocBufferOp(printer, cast<tvm::AllocBufferOp>(op));
            }
            if (isa<tvm::MatchBufferOp>(op)) {
              visitMatchBufferOp(printer, cast<tvm::MatchBufferOp>(op));
            }
            if (isa<scf::ForOp>(op)) {
              visitForOp(printer, cast<scf::ForOp>(op));
            }
          }
        });
      }
    });
  }
};

} // namespace mlir::tvm

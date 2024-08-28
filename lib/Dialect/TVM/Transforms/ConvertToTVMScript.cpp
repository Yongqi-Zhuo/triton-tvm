#include <fstream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"

#include "triton-tvm/Dialect/TVM/IR/Dialect.h"
#include "triton-tvm/Dialect/TVM/Transforms/Passes.h"
#include "triton-tvm/Utils/Python.hpp"

#define DEBUG_TYPE "convert-to-tvmscript"

namespace mlir::tvm {

#define GEN_PASS_DEF_CONVERTTOTVMSCRIPT
#include "triton-tvm/Dialect/TVM/Transforms/Passes.h.inc"

class ConvertToTVMScriptPass
    : public impl::ConvertToTVMScriptBase<ConvertToTVMScriptPass> {

  using Self = ConvertToTVMScriptPass;

  std::optional<PythonCodePrinter> printer;

  DenseMap<Value, std::string> varNames;

  llvm::SmallDenseMap<Type, StringRef> typeNames;

  inline StringRef typeName(Type type) const { return typeNames.at(type); }

  inline std::string createVar(Value value,
                               [[maybe_unused]] StringRef prefix = "v") {
    auto it = varNames.find(value);
    if (it != varNames.end())
      return it->second;
    std::string name = (prefix + Twine(varNames.size())).str();
    varNames.try_emplace(value, name);
    return name;
  }

  template <typename... Args>
  inline void write(const char *format, Args &&...args) {
    printer->write(format, std::forward<Args>(args)...);
  }

  inline void writeLn() { printer->writeLn(); }

  template <typename... Args>
  inline void writeLn(const char *format, Args &&...args) {
    printer->writeLn(format, std::forward<Args>(args)...);
  }

  inline void writeRaw(StringRef str) { printer->writeRaw(str); }

  inline void writeRawLn(StringRef str) { printer->writeRawLn(str); }

  template <typename R, typename F>
  inline void printRange(R &&range, F &&each) {
    using std::begin;
    using std::end;
    for (auto it = begin(std::forward<R>(range)),
              endIt = end(std::forward<R>(range));
         it != endIt; ++it) {
      if constexpr (std::is_invocable_v<F, decltype(*it)>) {
        std::invoke(std::forward<F>(each), *it);
      } else {
        // For member function pointers.
        std::invoke(std::forward<F>(each), this, *it);
      }
      // Should not use join, because Python needs trailing commas to identify
      // tuples.
      writeRaw(", ");
    }
  }

  template <typename R, typename F>
  inline void printParenRange(R &&range, F &&each) {
    printer->parens(
        [&] { printRange(std::forward<R>(range), std::forward<F>(each)); });
  }

  template <typename R, typename F>
  inline void printBracketRange(R &&range, F &&each) {
    printer->brackets(
        [&] { printRange(std::forward<R>(range), std::forward<F>(each)); });
  }

  inline void printStaticShape(ArrayRef<int64_t> shape) {
    printer->brackets([&] {
      printRange(shape, [&](int64_t dim) { write("T.int32({0})", dim); });
    });
  }

  inline void printQuotedTypeName(Type type) {
    write("\"{0}\"", typeName(type));
  }

  inline void printArgType(Type type) {
    if (!isa<MemRefType>(type)) {
      return signalPassFailure();
    }
    auto memrefType = cast<MemRefType>(type);
    if (memrefType.hasStaticShape()) {
      // Static shape.
      // Example: T.Buffer((T.int32(10), T.int32(20), ), "float32")
      writeRaw("T.Buffer");
      printer->parens(
          [&] { printStaticShape(memrefType.getShape()); },
          [&] { printQuotedTypeName(memrefType.getElementType()); });
    } else {
      // Dynamic shape.
      writeRaw("T.handle");
    }
  }

  inline void printAttr(Attribute attr) {
    llvm::TypeSwitch<Attribute>(attr)
        .Case<IntegerAttr>([&](IntegerAttr intAttr) {
          write("T.{0}({1})", typeName(intAttr.getType()), intAttr.getValue());
        })
        .Case<FloatAttr>([&](FloatAttr floatAttr) {
          write("T.{0}({1})", typeName(floatAttr.getType()),
                floatAttr.getValue().convertToDouble());
        })
        .Case<StringAttr>(
            [&](StringAttr strAttr) { write("\"{0}\"", strAttr.getValue()); })
        .Default([&](auto) {
          emitError(UnknownLoc()) << "unsupported attribute: " << attr;
          signalPassFailure();
        });
  }

  inline void printExpr(Value value) {
    // TODO!!!
    if (auto op = value.getDefiningOp<arith::ConstantOp>()) {
      printAttr(op.getValue());
    } else {
      write("{0}", value);
    }
  }

  void visit(VarOp op) {
    // Example: var0 = T.var("int32")
    writeLn("{0} = T.var(\"{1}\")", createVar(op, "var"),
            typeName(op.getType()));
  }

  void visit(MatchBufferOp op) {
    // Example: buffer0 = T.match_buffer(v0, (v1, v2, ), "float32")
    write("{0} = T.match_buffer", createVar(op, "buffer"));
    printer->parens([&] { printExpr(op); },
                    [&] { printParenRange(op.getSizes(), &Self::printExpr); },
                    [&] { printQuotedTypeName(op.getType()); });
    writeLn();
  }

  void visit(AllocBufferOp op) {
    // Example: alloc0 = T.alloc_buffer([T.int32(10), ], dtype="float32",
    // scope="global")
    write("{0} = T.alloc_buffer", createVar(op, "alloc"));
    // TODO: Add dynamic shape.
    printer->parens([&] { printStaticShape(op.getType().getShape()); },
                    [&] {
                      write("dtype=\"{0}\"",
                            typeName(op.getType().getElementType()));
                    },
                    [&] { write("scope=\"{0}\"", op.getScope()); });
    writeLn();
  }

  void visit(scf::ForOp op) {
    // Example: for index0 in T.serial(T.int32(0), T.int32(10)):
    auto forKind = op->getAttrOfType<ForKindAttr>(kAttrForKindName).getValue();
    auto forThread = op->getAttrOfType<StringAttr>(kAttrForThreadName);
    auto forKindStr = stringifyForKind(forKind);
    assert(op.getConstantStep().value() == 1 &&
           "only support for loop with step 1");
    write("for {0} in T.{1}", createVar(op.getInductionVar(), "index"),
          forKindStr);
    printer->parens([&] {
      printExpr(op.getLowerBound());
      writeRaw(", ");
      printExpr(op.getUpperBound());
      if (forKind == ForKind::THREAD_BINDING) {
        write(", thread=\"{0}\"", forThread.getValue());
      }
    });
    writeRawLn(":");
    auto &entryBlock = op.getRegion().front();
    printer->indent([&] {
      for (Operation &op : entryBlock) {
        visitOp(&op);
      }
    });
  }

  void visit(BlockOp op) {
    // Example: with T.block():
    writeRawLn("with T.block():");
    printer->indent([&] {
      for (Operation &op : op.getRegion().front()) {
        visitOp(&op);
      }
    });
  }

  void visit(WhereOp op) {
    // Example: T.where(v0 > T.int32(1))
    writeRaw("T.where");
    printer->parens([&] { printExpr(op.getCondition()); });
    writeLn();
  }

  void visit(AxisOp op) {
    // Example: axis0 = T.axis.spatial(T.int32(5), index0)
    write("{0} = T.axis.{1}", createVar(op, "axis"),
          stringifyAxisKind(op.getAxisKind()));
    printer->parens([&] { printExpr(op.getExtent()); },
                    [&] { printExpr(op.getBinding()); });
    writeLn();
  }

  void visit(IfThenElseOp op) {
    // Example: T.if_then_else(v0 > T.int32(1), arg0[axis0], T.min_value(0))
    writeRaw("T.if_then_else");
    printer->parens([&] { printExpr(op.getCondition()); },
                    [&] { printExpr(op.getTrueValue()); },
                    [&] { printExpr(op.getFalseValue()); });
    writeLn();
  }

  void visit(ReadOp op) {
    // Example: T.reads([buffer0[axis0]])
    writeRaw("T.reads");
    printer->parens([&] { printBracketRange(op.getRefs(), &Self::printExpr); });
    writeLn();
  }

  void visit(WriteOp op) {
    // Example: T.writes([buffer0[axis0]])
    writeRaw("T.writes");
    printer->parens([&] { printBracketRange(op.getRefs(), &Self::printExpr); });
    writeLn();
  }

  void visit(AssignOp op) {
    // Example: buffer0[axis0] = T.float32(1.0)
    printExpr(op.getLhs());
    writeRaw(" = ");
    printExpr(op.getRhs());
    writeLn();
  }

  void visit(InitOp op) {
    // Example: with T.init():
    writeRawLn("with T.init():");
    printer->indent([&] {
      for (Operation &op : op.getRegion().front()) {
        visitOp(&op);
      }
    });
  }

  inline void visitOp(Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<VarOp>([&](VarOp op) { visit(op); })
        .Case<MatchBufferOp>([&](MatchBufferOp op) { visit(op); })
        .Case<AllocBufferOp>([&](AllocBufferOp op) { visit(op); })
        .Case<scf::ForOp>([&](scf::ForOp op) { visit(op); })
        .Case<BlockOp>([&](BlockOp op) { visit(op); })
        .Case<WhereOp>([&](WhereOp op) { visit(op); })
        .Case<AxisOp>([&](AxisOp op) { visit(op); })
        .Case<IfThenElseOp>([&](IfThenElseOp op) { visit(op); })
        .Case<ReadOp>([&](ReadOp op) { visit(op); })
        .Case<WriteOp>([&](WriteOp op) { visit(op); })
        .Case<AssignOp>([&](AssignOp op) { visit(op); })
        .Case<InitOp>([&](InitOp op) { visit(op); })
        .Default([&](auto op) {
          // Not important. Just skip.
          // Note: tvm.ref, tvm.min_value and tvm.max_value should be handled by
          // printExpr.
        });
  }

public:
  using ConvertToTVMScriptBase<ConvertToTVMScriptPass>::ConvertToTVMScriptBase;

  void runOnOperation() override {
    typeNames = {
        {IndexType::get(&getContext()), "int64"},
        {IntegerType::get(&getContext(), 64), "int64"},
        {IntegerType::get(&getContext(), 32), "int32"},
        {IntegerType::get(&getContext(), 16), "int16"},
        {IntegerType::get(&getContext(), 1), "bool"},
        {FloatType::getF64(&getContext()), "float64"},
        {FloatType::getF32(&getContext()), "float32"},
        {FloatType::getF16(&getContext()), "float16"},
    };

    auto moduleOp = getOperation();
    auto ofstream = std::ofstream(outputFilename);
    auto rawOfstream = llvm::raw_os_ostream(ofstream);
    printer.emplace(rawOfstream, 0);

    writeRawLn("import tvm");
    writeRawLn("import tvm.script");
    writeRawLn("from tvm.script import tir as T");
    writeLn();
    writeRawLn("@tvm.script.ir_module");
    writeRawLn("class Module:");
    printer->indent([&] {
      for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
        writeRawLn("@T.prim_func");
        write("def {0}", funcOp.getName());
        printParenRange(funcOp.getArguments(), [&](BlockArgument arg) {
          write("{0}: ", createVar(arg, "arg"));
          printArgType(arg.getType());
        });
        writeRawLn(":");
        printer->indent([&] {
          writeRawLn(R"code(T.func_attr({"tir.noalias": True}))code");
          if (!funcOp.getBody().hasOneBlock()) {
            funcOp.emitError("expected function to have a single block");
            return signalPassFailure();
          }
          auto &entryBlock = funcOp.getBody().front();
          for (Operation &op : entryBlock) {
            visitOp(&op);
          }
        });
      }
    });
  }
};

} // namespace mlir::tvm

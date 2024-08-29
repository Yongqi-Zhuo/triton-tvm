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
#include "triton-tvm/Utils/Python.h"

#define DEBUG_TYPE "convert-to-tvmscript"

namespace mlir::tvm {

#define GEN_PASS_DEF_CONVERTTOTVMSCRIPT
#include "triton-tvm/Dialect/TVM/Transforms/Passes.h.inc"

namespace {

std::string escapeString(StringRef value) {
  std::string ret;
  llvm::raw_string_ostream os(ret);
  os.write_escaped(value);
  return os.str();
}

struct PyExpr {
  enum Precedence : int {
    // Literals, parenthesized expressions, ...
    ATOM = 0,
    // Function calls, subscriptions, ...
    CALL,
    // Multiplication, division, modulo, ...
    MUL,
    // Addition, subtraction
    ADD,
    // Bitwise shift
    SHIFT,
    // Bitwise AND
    BITAND,
    // Bitwise XOR
    BITXOR,
    // Bitwise OR
    BITOR,
    // Comparison operators
    COMP,
    // Boolean NOT
    NOT,
    // Boolean AND
    AND,
    // Boolean OR
    OR,
  };
  std::string str;
  Precedence prec;

  static PyExpr atom(std::string value) {
    return {.str = std::move(value), .prec = ATOM};
  }

  static PyExpr literal(StringRef value) {
    std::string ret;
    llvm::raw_string_ostream os(ret);
    os << '"';
    os.write_escaped(value);
    os << '"';
    return atom(os.str());
  }

  static PyExpr unary(const PyExpr &operand, StringRef op, Precedence prec) {
    PyExpr result{.prec = prec};
    result.str = (op + (operand.prec > prec ? ("(" + Twine(operand.str) + ")")
                                            : Twine(operand.str)))
                     .str();
    return result;
  }

  static PyExpr binary(const PyExpr &lhs, const PyExpr &rhs, StringRef op,
                       Precedence prec) {
    PyExpr result{.prec = prec};
    // Parenthesize if necessary.
    result.str =
        (
            // lhs
            (lhs.prec > prec ? ("(" + Twine(lhs.str) + ")") : Twine(lhs.str)) +
            // op
            " " + op + " " +
            // rhs
            (rhs.prec >= prec ? ("(" + Twine(rhs.str) + ")") : Twine(rhs.str)))
            .str();
    return result;
  }

  template <char Left, char Right>
  static PyExpr callLike(const PyExpr &func, ArrayRef<PyExpr> args) {
    PyExpr result{.prec = CALL};
    result.str =
        (func.prec > CALL ? ("(" + Twine(func.str) + ")") : Twine(func.str))
            .str();
    result.str += Left;
    StringRef separator = "";
    for (const PyExpr &arg : args) {
      result.str += separator;
      result.str += arg.str;
      separator = ", ";
    }
    result.str += Right;
    return result;
  }

  static PyExpr call(const PyExpr &func, ArrayRef<PyExpr> args) {
    return callLike<'(', ')'>(func, args);
  }

  static PyExpr subscription(const PyExpr &base, ArrayRef<PyExpr> indices) {
    return callLike<'[', ']'>(base, indices);
  }
};

StringRef getCmpFPredicate(arith::CmpFPredicate predicate) {
  switch (predicate) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    return "==";
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    return ">";
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return ">=";
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    return "<";
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    return "<=";
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return "!=";
  default:
    llvm_unreachable("unsupported predicate");
  }
}

StringRef getCmpIPredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return "==";
  case arith::CmpIPredicate::ne:
    return "!=";
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return "<";
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return "<=";
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return ">";
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return ">=";
  default:
    llvm_unreachable("unsupported predicate");
  }
}

template <typename Op>
struct BinOpTraits {};
template <PyExpr::Precedence Prec, char... OpStr>
struct OpImpl {
  static constexpr const char opStr[] = {OpStr..., '\0'};
  static constexpr PyExpr::Precedence prec = Prec;
};
template <char... OpStr>
struct CallImpl {
  static constexpr const char opStr[] = {'T', '.', OpStr..., '\0'};
};

template <>
struct BinOpTraits<arith::AddFOp> : OpImpl<PyExpr::ADD, '+'> {};
template <>
struct BinOpTraits<arith::AddIOp> : OpImpl<PyExpr::ADD, '+'> {};
template <>
struct BinOpTraits<arith::DivFOp> : OpImpl<PyExpr::MUL, '/'> {};
template <>
struct BinOpTraits<arith::MulFOp> : OpImpl<PyExpr::MUL, '*'> {};
template <>
struct BinOpTraits<arith::MulIOp> : OpImpl<PyExpr::MUL, '*'> {};
template <>
struct BinOpTraits<arith::RemFOp> : OpImpl<PyExpr::MUL, '%'> {};
// TODO: Make sure the '%' here always follows the same semantics.
template <>
struct BinOpTraits<arith::RemSIOp> : OpImpl<PyExpr::MUL, '%'> {};
template <>
struct BinOpTraits<arith::RemUIOp> : OpImpl<PyExpr::MUL, '%'> {};
template <>
struct BinOpTraits<arith::ShLIOp> : OpImpl<PyExpr::SHIFT, '<', '<'> {};
template <>
struct BinOpTraits<arith::ShRSIOp> : OpImpl<PyExpr::SHIFT, '>', '>'> {};
template <>
struct BinOpTraits<arith::ShRUIOp> : OpImpl<PyExpr::SHIFT, '>', '>'> {};
template <>
struct BinOpTraits<arith::SubFOp> : OpImpl<PyExpr::ADD, '-'> {};
template <>
struct BinOpTraits<arith::SubIOp> : OpImpl<PyExpr::ADD, '-'> {};
template <>
struct BinOpTraits<arith::XOrIOp> : OpImpl<PyExpr::BITXOR, '^'> {};
// TODO: Make sure the divisions here always follow the same semantics.
template <>
struct BinOpTraits<arith::CeilDivSIOp>
    : CallImpl<'c', 'e', 'i', 'l', 'd', 'i', 'v'> {};
template <>
struct BinOpTraits<arith::CeilDivUIOp>
    : CallImpl<'c', 'e', 'i', 'l', 'd', 'i', 'v'> {};
template <>
struct BinOpTraits<arith::DivSIOp>
    : CallImpl<'t', 'r', 'u', 'n', 'c', 'd', 'i', 'v'> {};
template <>
struct BinOpTraits<arith::DivUIOp>
    : CallImpl<'t', 'r', 'u', 'n', 'c', 'd', 'i', 'v'> {};
template <>
struct BinOpTraits<arith::FloorDivSIOp>
    : CallImpl<'f', 'l', 'o', 'o', 'r', 'd', 'i', 'v'> {};
template <>
struct BinOpTraits<arith::MaximumFOp> : CallImpl<'m', 'a', 'x'> {};
template <>
struct BinOpTraits<arith::MaxNumFOp> : CallImpl<'m', 'a', 'x'> {};
template <>
struct BinOpTraits<arith::MaxSIOp> : CallImpl<'m', 'a', 'x'> {};
template <>
struct BinOpTraits<arith::MaxUIOp> : CallImpl<'m', 'a', 'x'> {};
template <>
struct BinOpTraits<arith::MinimumFOp> : CallImpl<'m', 'i', 'n'> {};
template <>
struct BinOpTraits<arith::MinNumFOp> : CallImpl<'m', 'i', 'n'> {};
template <>
struct BinOpTraits<arith::MinSIOp> : CallImpl<'m', 'i', 'n'> {};
template <>
struct BinOpTraits<arith::MinUIOp> : CallImpl<'m', 'i', 'n'> {};
template <>
struct BinOpTraits<math::AbsFOp> : CallImpl<'a', 'b', 's'> {};
template <>
struct BinOpTraits<math::AbsIOp> : CallImpl<'a', 'b', 's'> {};
template <>
struct BinOpTraits<math::CeilOp> : CallImpl<'c', 'e', 'i', 'l'> {};
template <>
struct BinOpTraits<math::CosOp> : CallImpl<'c', 'o', 's'> {};
template <>
struct BinOpTraits<math::ExpOp> : CallImpl<'e', 'x', 'p'> {};
template <>
struct BinOpTraits<math::FloorOp> : CallImpl<'f', 'l', 'o', 'o', 'r'> {};
template <>
struct BinOpTraits<math::LogOp> : CallImpl<'l', 'o', 'g'> {};
template <>
struct BinOpTraits<math::Log2Op> : CallImpl<'l', 'o', 'g', '2'> {};
template <>
struct BinOpTraits<math::SinOp> : CallImpl<'s', 'i', 'n'> {};
template <>
struct BinOpTraits<math::SqrtOp> : CallImpl<'s', 'q', 'r', 't'> {};
template <>
struct BinOpTraits<math::TanOp> : CallImpl<'t', 'a', 'n'> {};

struct TypeNameMapper {
  llvm::SmallDenseMap<Type, StringRef> names;
  TypeNameMapper(MLIRContext *ctx)
      : names{
            {IndexType::get(ctx), "int64"},
            {IntegerType::get(ctx, 64), "int64"},
            {IntegerType::get(ctx, 32), "int32"},
            {IntegerType::get(ctx, 16), "int16"},
            {IntegerType::get(ctx, 1), "bool"},
            {FloatType::getF64(ctx), "float64"},
            {FloatType::getF32(ctx), "float32"},
            {FloatType::getF16(ctx), "float16"},
        } {}
  StringRef get(Type type) const { return names.at(type); }
};

class PyEval {
  const TypeNameMapper &typeNames;
  DenseMap<Value, PyExpr> exprs;

public:
  PyEval(const TypeNameMapper &typeNames) : typeNames(typeNames) {}

  std::string createVar(Value value, StringRef prefix = "v") {
    auto var = PyExpr::atom((prefix + Twine(exprs.size())).str());
    auto [it, inserted] = exprs.try_emplace(value, std::move(var));
    assert(inserted && "value already exists");
    return it->second.str;
  }

  PyExpr makeConstant(Attribute attr) {
    return llvm::TypeSwitch<Attribute, PyExpr>(attr)
        .Case<IntegerAttr>([&](IntegerAttr intAttr) {
          return PyExpr{.str = llvm::formatv("T.{0}({1})",
                                             typeNames.get(intAttr.getType()),
                                             intAttr.getValue()),
                        .prec = PyExpr::CALL};
        })
        .Case<FloatAttr>([&](FloatAttr floatAttr) {
          assert(!floatAttr.getValue().isNaN());
          assert(!floatAttr.getValue().isInfinity() &&
                 "infinity should be handled by convertInfinity()");
          return PyExpr{.str = llvm::formatv("T.{0}({1})",
                                             typeNames.get(floatAttr.getType()),
                                             floatAttr.getValueAsDouble()),
                        .prec = PyExpr::CALL};
        })
        .Case<StringAttr>([&](StringAttr strAttr) {
          return PyExpr::atom(escapeString(strAttr.getValue()));
        })
        .Default(
            [&](auto) -> PyExpr { llvm_unreachable("unsupported attribute"); });
  }

  template <typename... Operands>
  PyExpr makeCall(std::string f, Operands... operands) {
    return PyExpr::call(PyExpr::atom(std::move(f)), {get(operands)...});
  }

  PyExpr makeUnary(Value operand, StringRef op, PyExpr::Precedence prec) {
    return PyExpr::unary(get(operand), op, prec);
  }

  PyExpr makeBinary(Value lhs, Value rhs, StringRef op,
                    PyExpr::Precedence prec) {
    return PyExpr::binary(get(lhs), get(rhs), op, prec);
  }

  template <typename... Ops>
  void matchUnaryOp(llvm::TypeSwitch<Operation *, PyExpr> &switcher) {
    (switcher.Case<Ops>([&](Ops op) {
      return makeUnary(op.getOperand(), BinOpTraits<Ops>::opStr,
                       BinOpTraits<Ops>::prec);
    }),
     ...);
  }

  template <typename... Ops>
  void matchBinaryOp(llvm::TypeSwitch<Operation *, PyExpr> &switcher) {
    (switcher.Case<Ops>([&](Ops op) {
      return makeBinary(op.getLhs(), op.getRhs(), BinOpTraits<Ops>::opStr,
                        BinOpTraits<Ops>::prec);
    }),
     ...);
  }

  template <typename... Ops>
  void matchUnaryCall(llvm::TypeSwitch<Operation *, PyExpr> &switcher) {
    (switcher.Case<Ops>([&](Ops op) {
      return makeCall(BinOpTraits<Ops>::opStr, op.getOperand());
    }),
     ...);
  }

  template <typename... Ops>
  void matchBinaryCall(llvm::TypeSwitch<Operation *, PyExpr> &switcher) {
    (switcher.Case<Ops>([&](Ops op) {
      return makeCall(BinOpTraits<Ops>::opStr, op.getLhs(), op.getRhs());
    }),
     ...);
  }

  PyExpr makeSubscription(Value base, ValueRange subscripts) {
    auto target = get(base);
    auto indices = llvm::map_to_vector(subscripts,
                                       [&](Value index) { return get(index); });
    return PyExpr::subscription(target, indices);
  }

  PyExpr get(Value value) {
    if (auto it = exprs.find(value); it != exprs.end()) {
      return it->second;
    }
    auto *op = value.getDefiningOp();
    assert(op && "BlockArgument is not defined. Remember to call createVar().");
    assert(op->hasTrait<OpTrait::OneResult>() &&
           "Expression Ops should always return only one result.");
    llvm::TypeSwitch<Operation *, PyExpr> switcher(op);
    matchBinaryOp<arith::AddFOp, arith::AddIOp, arith::DivFOp, arith::MulFOp,
                  arith::MulIOp, arith::RemFOp, arith::RemSIOp, arith::RemUIOp,
                  arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp, arith::SubFOp,
                  arith::SubIOp, arith::XOrIOp>(switcher);
    matchBinaryCall<arith::CeilDivSIOp, arith::CeilDivUIOp, arith::DivSIOp,
                    arith::DivUIOp, arith::FloorDivSIOp, arith::MaximumFOp,
                    arith::MaxNumFOp, arith::MaxSIOp, arith::MaxUIOp,
                    arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
                    arith::MinUIOp>(switcher);
    matchUnaryCall<math::AbsFOp, math::AbsIOp, math::CeilOp, math::CosOp,
                   math::ExpOp, math::FloorOp, math::LogOp, math::Log2Op,
                   math::SinOp, math::SqrtOp, math::TanOp>(switcher);
    // TODO: arith::NegFOp
    switcher
        .Case<arith::AndIOp>([&](arith::AndIOp op) {
          StringRef opStr = "&";
          PyExpr::Precedence prec = PyExpr::BITAND;
          if (op.getType().isInteger(1)) {
            opStr = "and";
            prec = PyExpr::AND;
          }
          return makeBinary(op.getLhs(), op.getRhs(), opStr, prec);
        })
        .Case<arith::CmpFOp>([&](auto op) {
          return makeBinary(op.getLhs(), op.getRhs(),
                            getCmpFPredicate(op.getPredicate()), PyExpr::COMP);
        })
        .Case<arith::CmpIOp>([&](auto op) {
          return makeBinary(op.getLhs(), op.getRhs(),
                            getCmpIPredicate(op.getPredicate()), PyExpr::COMP);
        })
        .Case<arith::ConstantOp>(
            [&](arith::ConstantOp op) { return makeConstant(op.getValue()); })
        .Case<arith::OrIOp>([&](arith::OrIOp op) {
          StringRef opStr = "|";
          PyExpr::Precedence prec = PyExpr::BITOR;
          if (op.getType().isInteger(1)) {
            opStr = "or";
            prec = PyExpr::OR;
          }
          return makeBinary(op.getLhs(), op.getRhs(), opStr, prec);
        })
        .Case<tvm::IfThenElseOp>([&](tvm::IfThenElseOp op) {
          return makeCall("T.if_then_else", op.getCondition(),
                          op.getTrueValue(), op.getFalseValue());
        })
        .Case<tvm::MaxValueOp>([&](tvm::MaxValueOp op) {
          return PyExpr::call(PyExpr::atom("T.max_value"),
                              PyExpr::literal(typeNames.get(op.getType())));
        })
        .Case<tvm::MinValueOp>([&](tvm::MinValueOp op) {
          return PyExpr::call(PyExpr::atom("T.min_value"),
                              PyExpr::literal(typeNames.get(op.getType())));
        })
        .Case<tvm::RefOp>([&](tvm::RefOp op) {
          return makeSubscription(op.getMemRef(), op.getIndices());
        });
    auto result = switcher.Default(
        [&](auto) -> PyExpr { llvm_unreachable("unsupported op"); });
    auto [it, inserted] = exprs.try_emplace(value, std::move(result));
    assert(inserted && "Cyclic dependencies in values");
    return it->second;
  }
};

LogicalResult convertInfinityConstant(arith::ConstantOp op,
                                      PatternRewriter &rewriter) {
  auto attr = dyn_cast<FloatAttr>(op.getValue());
  if (!attr)
    return failure();
  auto value = attr.getValue();
  if (value.isPosInfinity()) {
    rewriter.replaceOpWithNewOp<tvm::MaxValueOp>(op, op.getType());
  } else if (value.isNegInfinity()) {
    rewriter.replaceOpWithNewOp<tvm::MinValueOp>(op, op.getType());
  } else {
    return failure();
  }
  return success();
}

} // namespace

class ConvertToTVMScriptPass
    : public impl::ConvertToTVMScriptBase<ConvertToTVMScriptPass> {

  using Self = ConvertToTVMScriptPass;

  std::optional<TypeNameMapper> typeNames;

  std::optional<PythonCodePrinter> printer;

  std::optional<PyEval> exprs;

  inline StringRef typeName(Type type) const { return typeNames->get(type); }

  inline std::string createVar(Value value, StringRef prefix = "v") {
    return exprs->createVar(value, prefix);
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

  inline void printExpr(Value value) { writeRaw(exprs->get(value).str); }

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
        .Case<ReadOp>([&](ReadOp op) { visit(op); })
        .Case<WriteOp>([&](WriteOp op) { visit(op); })
        .Case<AssignOp>([&](AssignOp op) { visit(op); })
        .Case<InitOp>([&](InitOp op) { visit(op); })
        .Default([&](auto op) {
          // Not important. Just skip.
          // Note: tvm.if_then_else, tvm.ref, tvm.min_value and tvm.max_value
          // should be handled by printExpr.
        });
  }

public:
  using ConvertToTVMScriptBase<ConvertToTVMScriptPass>::ConvertToTVMScriptBase;

  // Replace infinity constants with tvm.min_value and tvm.max_value.
  void replaceInfinity() {
    RewritePatternSet patterns(&getContext());
    patterns.add(convertInfinityConstant);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void runOnOperation() override {
    replaceInfinity();

    auto moduleOp = getOperation();

    typeNames.emplace(&getContext());
    auto ofstream = std::ofstream(outputFilename);
    auto rawOfstream = llvm::raw_os_ostream(ofstream);
    printer.emplace(rawOfstream, 0);
    exprs.emplace(*typeNames);

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

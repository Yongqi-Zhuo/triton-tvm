#pragma once

#include "mlir/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::tvm {

class PythonCodePrinter {
  llvm::raw_ostream &os;
  bool isNewLine = true;
  std::size_t indentLevel;

  inline void writeIndent() {
    for (std::size_t i = 0; i < indentLevel; ++i)
      os << '\t';
  }

public:
  PythonCodePrinter(llvm::raw_ostream &os, std::size_t indentLevel)
      : os{os}, indentLevel{indentLevel} {}
  template <typename F>
  inline void indent(F &&f) {
    const std::size_t oldIndentLevel = indentLevel++;
    std::invoke(std::forward<F>(f));
    indentLevel = oldIndentLevel;
  };
  template <typename... Fs>
  inline void join(Fs &&...fs) {
    const char *separator = "";
    ((writeRaw(separator), std::invoke(std::forward<Fs>(fs)), separator = ", "),
     ...);
  }
  template <typename... Fs>
  inline void parens(Fs &&...fs) {
    writeRaw("(");
    join(std::forward<Fs>(fs)...);
    writeRaw(")");
  };
  template <typename... Fs>
  inline void brackets(Fs &&...fs) {
    writeRaw("[");
    join(std::forward<Fs>(fs)...);
    writeRaw("]");
  };
  template <typename... Args>
  inline void write(const char *format, Args &&...args) {
    if (isNewLine) {
      writeIndent();
      isNewLine = false;
    }
    llvm::formatv(format, std::forward<Args>(args)...).format(os);
  }
  inline void writeLn() {
    os << "\n";
    isNewLine = true;
  }
  template <typename... Args>
  inline void writeLn(const char *format, Args &&...args) {
    write(format, std::forward<Args>(args)...);
    writeLn();
  }
  inline void writeRaw(StringRef str) {
    if (isNewLine) {
      writeIndent();
      isNewLine = false;
    }
    os << str;
  }
  inline void writeRawLn(StringRef str) {
    writeRaw(str);
    writeLn();
  }
};

} // namespace mlir::tvm

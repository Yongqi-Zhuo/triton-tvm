#pragma once

#include <functional>
#include <ostream>

#include <fmt/format.h>

namespace mlir::tvm {

template <typename _CharT, typename _Traits>
class PythonCodePrinter {
  std::basic_ostream<_CharT, _Traits> &oss;
  bool isNewLine = true;
  std::size_t indentLevel;
  void writeIndent() {
    for (std::size_t i = 0; i < indentLevel; ++i)
      oss << '\t';
  }

public:
  template <typename F>
  void indent(F &&f) {
    const std::size_t oldIndentLevel = indentLevel++;
    std::invoke(f);
    indentLevel = oldIndentLevel;
  };
  template <bool CommaAndNewLine = false, typename F>
  void parens(F &&f) {
    writeLn("(");
    indent(std::forward<F>(f));
    if constexpr (CommaAndNewLine) {
      writeLn("),");
    } else {
      write(")");
    }
  };
  PythonCodePrinter(std::basic_ostream<_CharT, _Traits> &oss,
                    std::size_t indentLevel)
      : oss{oss}, indentLevel{indentLevel} {}
  template <typename... Args>
  void write(fmt::format_string<Args...> format, Args &&...args) {
    if (isNewLine) {
      writeIndent();
      isNewLine = false;
    }
    fmt::format_to(std::ostreambuf_iterator(oss), format,
                   std::forward<Args>(args)...);
  }
  void writeLn() {
    oss << "\n";
    isNewLine = true;
  }
  template <typename... Args>
  void writeLn(fmt::format_string<Args...> format, Args &&...args) {
    write(format, std::forward<Args>(args)...);
    writeLn();
  }
};

} // namespace mlir::tvm

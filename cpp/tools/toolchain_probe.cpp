#include <iostream>
#include <string_view>

namespace {

constexpr std::string_view compiler_id() {
#if defined(__apple_build_version__)
  return "AppleClang";
#elif defined(__clang__)
  return "Clang";
#elif defined(__GNUC__)
  return "GNU";
#elif defined(_MSC_VER)
  return "MSVC";
#else
  return "Unknown";
#endif
}

constexpr std::string_view architecture() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#else
  return "unknown";
#endif
}

constexpr std::string_view compiler_version() {
#if defined(__clang_version__)
  return __clang_version__;
#elif defined(__VERSION__)
  return __VERSION__;
#else
  return "unknown";
#endif
}

void write_json_string(std::string_view value) {
  std::cout << '"';
  for (const char character : value) {
    switch (character) {
      case '"':
        std::cout << "\\\"";
        break;
      case '\\':
        std::cout << "\\\\";
        break;
      case '\n':
        std::cout << "\\n";
        break;
      case '\r':
        std::cout << "\\r";
        break;
      case '\t':
        std::cout << "\\t";
        break;
      default:
        std::cout << character;
        break;
    }
  }
  std::cout << '"';
}

}  // namespace

int main() {
  std::cout << "{\"architecture\":";
  write_json_string(architecture());
  std::cout << ",\"compiler_id\":";
  write_json_string(compiler_id());
  std::cout << ",\"compiler_version\":";
  write_json_string(compiler_version());
  std::cout << ",\"cxx_standard\":" << __cplusplus << "}\n";
  return 0;
}

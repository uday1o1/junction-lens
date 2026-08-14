#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "contract/validation.hpp"

namespace {

struct Options {
  std::filesystem::path input;
  std::size_t runs = 5000U;
  std::uint64_t seed = 20260814U;
  std::size_t max_length = 1024U * 1024U;
};

[[nodiscard]] std::uint64_t Next(std::uint64_t& state) {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

template <typename Integer>
[[nodiscard]] Integer ParseInteger(std::string_view value, std::string_view name) {
  Integer parsed{};
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
    throw std::invalid_argument("invalid value for " + std::string(name));
  }
  return parsed;
}

[[nodiscard]] Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("every option requires a value");
    }
    const std::string_view name(argv[index]);
    const std::string_view value(argv[index + 1]);
    if (name == "--input") {
      options.input = value;
    } else if (name == "--runs") {
      options.runs = ParseInteger<std::size_t>(value, name);
    } else if (name == "--seed") {
      options.seed = ParseInteger<std::uint64_t>(value, name);
    } else if (name == "--max-length") {
      options.max_length = ParseInteger<std::size_t>(value, name);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(name));
    }
  }
  if (options.input.empty() || options.runs == 0U || options.max_length == 0U ||
      options.max_length > junctionlens::contract::kMaximumSerializedBytes) {
    throw std::invalid_argument("input, runs, and bounded max-length are required");
  }
  return options;
}

[[nodiscard]] std::vector<std::uint8_t> ReadSeed(const Options& options) {
  std::ifstream stream(options.input, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("unable to open seed input");
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(stream)),
                                  std::istreambuf_iterator<char>());
  if (stream.bad() || bytes.empty() || bytes.size() > options.max_length) {
    throw std::runtime_error("seed input is empty, unreadable, or over max-length");
  }
  return bytes;
}

void Mutate(std::vector<std::uint8_t>& bytes, std::uint64_t& state, std::size_t max_length) {
  const std::size_t operations = 1U + static_cast<std::size_t>(Next(state) % 8U);
  for (std::size_t operation = 0; operation < operations; ++operation) {
    const std::uint64_t choice = Next(state) % 4U;
    if (choice == 0U && !bytes.empty()) {
      const std::size_t offset = static_cast<std::size_t>(Next(state) % bytes.size());
      bytes[offset] ^= static_cast<std::uint8_t>(1U << (Next(state) % 8U));
    } else if (choice == 1U && !bytes.empty()) {
      const std::size_t offset = static_cast<std::size_t>(Next(state) % bytes.size());
      bytes[offset] = static_cast<std::uint8_t>(Next(state) & 0xffU);
    } else if (choice == 2U && !bytes.empty()) {
      const std::size_t offset = static_cast<std::size_t>(Next(state) % bytes.size());
      bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(offset));
    } else if (bytes.size() < max_length) {
      const std::size_t offset =
          bytes.empty() ? 0U : static_cast<std::size_t>(Next(state) % (bytes.size() + 1U));
      bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                   static_cast<std::uint8_t>(Next(state) & 0xffU));
    }
  }
}

[[nodiscard]] bool Exercise(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  junctionlens::v1::SceneControlGraphEnvelope envelope;
  if (!envelope.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
    return false;
  }
  static_cast<void>(junctionlens::contract::Validate(envelope));
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfinity);
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfinity);
    const Options options = ParseOptions(argc, argv);
    const std::vector<std::uint8_t> seed_bytes = ReadSeed(options);
    std::uint64_t state = options.seed == 0U ? 1U : options.seed;
    std::size_t parsed = 0U;
    for (std::size_t run = 0; run < options.runs; ++run) {
      std::vector<std::uint8_t> candidate = seed_bytes;
      Mutate(candidate, state, options.max_length);
      if (Exercise(candidate)) {
        ++parsed;
      }
    }
    std::cout << "{\"runs\":" << options.runs << ",\"seed\":" << options.seed
              << ",\"parsed\":" << parsed << ",\"status\":\"passed\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fuzz smoke failed: " << error.what() << '\n';
    return 2;
  }
}

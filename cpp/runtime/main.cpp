#include "junctionlens/infer/runtime.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "contract/validation.hpp"

namespace {

using junctionlens::infer::BufferPool;
using junctionlens::infer::BufferState;
using junctionlens::infer::CpuRuntime;
using junctionlens::infer::ProducerOptions;
using junctionlens::infer::RuntimeError;
using junctionlens::infer::RuntimeOptions;
namespace v1 = junctionlens::v1;

struct Arguments {
  std::filesystem::path model;
  std::string expected_profile_sha256;
  std::filesystem::path input_list;
  std::filesystem::path asset_root;
  std::filesystem::path output_directory;
  std::string git_commit;
  bool git_dirty = false;
  std::string configuration_sha256;
  std::string runtime_build_sha256;
  std::size_t repeat_loads = 1U;
  std::size_t buffer_slots = 2U;
};

[[nodiscard]] std::size_t ParseSize(const std::string& value, const std::string& label,
                                    const std::size_t minimum, const std::size_t maximum) {
  std::size_t consumed = 0U;
  const unsigned long parsed = std::stoul(value, &consumed);
  if (consumed != value.size() || parsed < minimum || parsed > maximum) {
    throw std::invalid_argument(label + " must be within [" + std::to_string(minimum) + ", " +
                                std::to_string(maximum) + "]");
  }
  return static_cast<std::size_t>(parsed);
}

[[noreturn]] void PrintHelp() {
  std::cout << "Usage: junctionlens-runtime infer --model PATH --expected-profile-sha256 HEX "
               "--input-list PATH --asset-root DIR --output-dir DIR --git-commit HEX "
               "--configuration-sha256 HEX --runtime-build-sha256 HEX "
               "[--git-dirty] [--repeat-loads 1..100] [--buffer-slots 1..1024]\n";
  std::exit(EXIT_SUCCESS);
}

[[nodiscard]] Arguments ParseArguments(const int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    PrintHelp();
  }
  if (argc < 2 || std::string_view(argv[1]) != "infer") {
    throw std::invalid_argument("the required runtime command is 'infer'");
  }
  Arguments result;
  for (int index = 2; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help") {
      PrintHelp();
    }
    if (argument == "--git-dirty") {
      result.git_dirty = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value after " + argument);
    }
    const std::string value(argv[++index]);
    if (argument == "--model") {
      result.model = value;
    } else if (argument == "--expected-profile-sha256") {
      result.expected_profile_sha256 = value;
    } else if (argument == "--input-list") {
      result.input_list = value;
    } else if (argument == "--asset-root") {
      result.asset_root = value;
    } else if (argument == "--output-dir") {
      result.output_directory = value;
    } else if (argument == "--git-commit") {
      result.git_commit = value;
    } else if (argument == "--configuration-sha256") {
      result.configuration_sha256 = value;
    } else if (argument == "--runtime-build-sha256") {
      result.runtime_build_sha256 = value;
    } else if (argument == "--repeat-loads") {
      result.repeat_loads = ParseSize(value, "repeat loads", 1U, 100U);
    } else if (argument == "--buffer-slots") {
      result.buffer_slots = ParseSize(value, "buffer slots", 1U, 1024U);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (result.model.empty() || result.expected_profile_sha256.empty() || result.input_list.empty() ||
      result.asset_root.empty() || result.output_directory.empty() || result.git_commit.empty() ||
      result.configuration_sha256.empty() || result.runtime_build_sha256.empty()) {
    throw std::invalid_argument("all documented infer arguments are required");
  }
  return result;
}

[[nodiscard]] std::string Trim(std::string value) {
  const auto not_space = [](const unsigned char character) { return std::isspace(character) == 0; };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), [not_space](const char character) {
                return not_space(static_cast<unsigned char>(character));
              }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [not_space](const char character) {
                             return not_space(static_cast<unsigned char>(character));
                           })
                  .base(),
              value.end());
  return value;
}

[[nodiscard]] std::vector<std::filesystem::path> ReadInputList(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw RuntimeError("RUNTIME_INPUT_LIST", "input list could not be opened");
  }
  std::vector<std::filesystem::path> result;
  std::string line;
  std::size_t byte_count = 0U;
  while (std::getline(input, line)) {
    byte_count += line.size() + 1U;
    if (byte_count > 1024U * 1024U) {
      throw RuntimeError("RUNTIME_INPUT_LIST", "input list exceeds 1 MiB");
    }
    line = Trim(std::move(line));
    if (line.empty() || line.starts_with('#')) {
      continue;
    }
    std::filesystem::path item(line);
    if (item.is_relative()) {
      item = path.parent_path() / item;
    }
    result.push_back(std::move(item));
    if (result.size() > 10000U) {
      throw RuntimeError("RUNTIME_INPUT_LIST", "input list exceeds 10,000 frames");
    }
  }
  if (!input.eof()) {
    throw RuntimeError("RUNTIME_INPUT_LIST", "input list could not be read");
  }
  if (result.empty()) {
    throw RuntimeError("RUNTIME_INPUT_LIST", "input list contains no frames");
  }
  return result;
}

[[nodiscard]] RuntimeOptions MakeOptions(const Arguments& arguments) {
  return {
      arguments.model,
      arguments.expected_profile_sha256,
      ProducerOptions{arguments.git_commit, arguments.git_dirty, arguments.configuration_sha256,
                      arguments.runtime_build_sha256},
      0.5,
      0.5,
  };
}

[[nodiscard]] std::unique_ptr<CpuRuntime> LoadRuntime(const Arguments& arguments) {
  for (std::size_t iteration = 1U; iteration < arguments.repeat_loads; ++iteration) {
    auto probe = std::make_unique<CpuRuntime>(MakeOptions(arguments));
    probe.reset();
  }
  return std::make_unique<CpuRuntime>(MakeOptions(arguments));
}

[[nodiscard]] v1::SensorFrame ParseSensorFrame(const std::filesystem::path& path) {
  v1::SceneControlGraphEnvelope envelope;
  const auto result = junctionlens::contract::ParseFile(path, envelope);
  if (!result.valid) {
    throw RuntimeError("RUNTIME_INPUT_CONTRACT",
                       result.reason_code + " at " + result.path + ": " + result.detail);
  }
  if (!envelope.graph().has_sensor_frame()) {
    throw RuntimeError("RUNTIME_SENSOR_FRAME", "input graph does not contain a sensor frame");
  }
  if (envelope.graph().frame_key().SerializeAsString() !=
      envelope.graph().sensor_frame().frame_key().SerializeAsString()) {
    throw RuntimeError("RUNTIME_FRAME_KEY", "graph and sensor frame keys differ");
  }
  return envelope.graph().sensor_frame();
}

void WriteAtomically(const std::filesystem::path& path,
                     const v1::SceneControlGraphEnvelope& envelope) {
  if (std::filesystem::exists(path)) {
    throw RuntimeError("RUNTIME_OUTPUT_EXISTS",
                       "runtime refuses to overwrite output: " + path.string());
  }
  const std::filesystem::path temporary = path.string() + ".tmp";
  if (std::filesystem::exists(temporary)) {
    throw RuntimeError("RUNTIME_OUTPUT_EXISTS", "stale temporary output already exists");
  }
  try {
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output || !envelope.SerializeToOstream(&output)) {
        throw RuntimeError("RUNTIME_OUTPUT_IO", "protobuf output could not be serialized");
      }
      output.flush();
      if (!output) {
        throw RuntimeError("RUNTIME_OUTPUT_IO", "protobuf output could not be flushed");
      }
    }
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

[[nodiscard]] bool SameSequence(const v1::SensorFrame& previous, const v1::SensorFrame& current) {
  const auto& first = previous.frame_key();
  const auto& second = current.frame_key();
  return first.dataset_id() == second.dataset_id() &&
         first.dataset_version() == second.dataset_version() &&
         first.split_id() == second.split_id() && first.segment_id() == second.segment_id() &&
         first.timestamp_ns() < second.timestamp_ns();
}

int Run(const Arguments& arguments) {
  const auto inputs = ReadInputList(arguments.input_list);
  std::filesystem::create_directories(arguments.output_directory);
  if (!std::filesystem::is_directory(arguments.output_directory)) {
    throw RuntimeError("RUNTIME_OUTPUT_DIR", "output path is not a directory");
  }
  auto runtime = LoadRuntime(arguments);
  BufferPool pool(arguments.buffer_slots);
  std::optional<v1::SensorFrame> previous;
  std::size_t processed = 0U;
  for (const auto& input_path : inputs) {
    auto lease = pool.Acquire();
    v1::SensorFrame current = ParseSensorFrame(input_path);
    const v1::SensorFrame* paired_previous =
        previous.has_value() && SameSequence(*previous, current) ? &*previous : nullptr;
    auto preprocessed =
        junctionlens::infer::Preprocess(paired_previous, current, arguments.asset_root, lease);
    lease.Advance(BufferState::kInference);
    auto output = runtime->Infer(preprocessed, lease);
    const std::filesystem::path output_path =
        arguments.output_directory /
        (std::to_string(processed) + "-" + std::to_string(current.frame_key().timestamp_ns()) +
         ".prediction.pb");
    lease.Advance(BufferState::kSerializing);
    WriteAtomically(output_path, output);
    lease.Release();
    previous = std::move(current);
    ++processed;
  }
  runtime.reset();
  if (!pool.all_free()) {
    throw RuntimeError("RUNTIME_BUFFER_LEAK", "buffer slots were not released at clean shutdown");
  }
  std::cout << "{\"schema_version\":\"junctionlens.runtime-batch.v1\",\"status\":\"PASSED\","
               "\"provider\":\"CPUExecutionProvider\",\"processed_frames\":"
            << processed << ",\"repeat_loads\":" << arguments.repeat_loads
            << ",\"buffer_capacity\":" << pool.capacity()
            << ",\"buffer_high_water_mark\":" << pool.high_water_mark()
            << ",\"all_slots_free\":true}\n";
  return EXIT_SUCCESS;
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    return Run(ParseArguments(argc, argv));
  } catch (const RuntimeError& error) {
    std::cerr << "runtime error [" << error.reason_code() << "]: " << error.what() << '\n';
  } catch (const Ort::Exception& error) {
    std::cerr << "runtime error [RUNTIME_ONNX]: " << error.what() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "runtime error [RUNTIME_ARGUMENT]: " << error.what() << '\n';
  }
  return EXIT_FAILURE;
}

#include "junctionlens/infer/runtime.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "contract/validation.hpp"
#include "junctionlens/infer/instrumentation.hpp"

namespace {

using junctionlens::infer::BufferPool;
using junctionlens::infer::BufferState;
using junctionlens::infer::CpuRuntime;
using junctionlens::infer::ExecutionProviderProfile;
using junctionlens::infer::ProducerOptions;
using junctionlens::infer::ProviderOptions;
using junctionlens::infer::RuntimeError;
using junctionlens::infer::RuntimeMemoryHighWater;
using junctionlens::infer::RuntimeOptions;
using junctionlens::infer::RuntimePhaseTiming;
namespace v1 = junctionlens::v1;

struct Arguments {
  std::string command;
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
  ProviderOptions provider;
  std::filesystem::path provider_log_output;
  std::filesystem::path timing_output;
  std::size_t warmup_frames = 0U;
  std::size_t measured_frames = 0U;
  std::size_t stability_frames = 0U;
  std::size_t memory_sample_period = 100U;
  std::string input_profile = "full-file";
  bool profiler_run = false;
  std::filesystem::path onnx_profile_prefix;
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
  std::cout << "Usage:\n"
               "  junctionlens-runtime infer --model PATH --expected-profile-sha256 HEX "
               "--input-list PATH --asset-root DIR --output-dir DIR --git-commit HEX "
               "--configuration-sha256 HEX --runtime-build-sha256 HEX "
               "[--provider-profile cpu-reference|cuda|tensorrt] "
               "[--provider-log-output PATH] [provider options] "
               "[--git-dirty] [--repeat-loads 1..100] [--buffer-slots 1..1024]\n"
               "  junctionlens-runtime doctor --model PATH --expected-profile-sha256 HEX "
               "[--provider-profile cpu-reference|cuda|tensorrt] "
               "[--provider-log-output PATH] [provider options]\n"
               "  junctionlens-runtime benchmark --model PATH --expected-profile-sha256 HEX "
               "--input-list PATH --asset-root DIR --timing-output PATH --git-commit HEX "
               "--configuration-sha256 HEX --runtime-build-sha256 HEX "
               "--warmup-frames N --measured-frames N --stability-frames N "
               "[--input-profile full-file|predecoded] [--memory-sample-period N] "
               "[--profiler-run] [provider options]\n"
               "Provider options: --device-id N --provider-cache-root DIR "
               "--gpu-compute-capability TEXT --cuda-version TEXT "
               "--driver-compatibility-class TEXT --tensorrt-version TEXT\n";
  std::exit(EXIT_SUCCESS);
}

[[nodiscard]] ExecutionProviderProfile ParseProviderProfile(const std::string& value) {
  if (value == "cpu-reference") {
    return ExecutionProviderProfile::kCpuReference;
  }
  if (value == "cuda") {
    return ExecutionProviderProfile::kCuda;
  }
  if (value == "tensorrt") {
    return ExecutionProviderProfile::kTensorRt;
  }
  throw std::invalid_argument("provider profile must be cpu-reference, cuda, or tensorrt");
}

[[nodiscard]] Arguments ParseArguments(const int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    PrintHelp();
  }
  if (argc < 2 || (std::string_view(argv[1]) != "infer" && std::string_view(argv[1]) != "doctor" &&
                   std::string_view(argv[1]) != "benchmark")) {
    throw std::invalid_argument("the runtime command must be 'infer', 'doctor', or 'benchmark'");
  }
  Arguments result;
  result.command = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help") {
      PrintHelp();
    }
    if (argument == "--git-dirty") {
      result.git_dirty = true;
      continue;
    }
    if (argument == "--profiler-run") {
      result.profiler_run = true;
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
    } else if (argument == "--provider-profile") {
      result.provider.profile = ParseProviderProfile(value);
    } else if (argument == "--device-id") {
      result.provider.device_id = static_cast<int>(ParseSize(value, "device ID", 0U, 1024U));
    } else if (argument == "--provider-cache-root") {
      result.provider.cache_root = value;
    } else if (argument == "--gpu-compute-capability") {
      result.provider.gpu_compute_capability = value;
    } else if (argument == "--cuda-version") {
      result.provider.cuda_version = value;
    } else if (argument == "--driver-compatibility-class") {
      result.provider.driver_compatibility_class = value;
    } else if (argument == "--tensorrt-version") {
      result.provider.tensorrt_version = value;
    } else if (argument == "--provider-log-output") {
      result.provider_log_output = value;
    } else if (argument == "--timing-output") {
      result.timing_output = value;
    } else if (argument == "--warmup-frames") {
      result.warmup_frames = ParseSize(value, "warmup frames", 0U, 100000U);
    } else if (argument == "--measured-frames") {
      result.measured_frames = ParseSize(value, "measured frames", 1U, 100000U);
    } else if (argument == "--stability-frames") {
      result.stability_frames = ParseSize(value, "stability frames", 0U, 100000U);
    } else if (argument == "--memory-sample-period") {
      result.memory_sample_period = ParseSize(value, "memory sample period", 1U, 10000U);
    } else if (argument == "--input-profile") {
      result.input_profile = value;
    } else if (argument == "--onnx-profile-prefix") {
      result.onnx_profile_prefix = value;
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (result.model.empty() || result.expected_profile_sha256.empty()) {
    throw std::invalid_argument("model and expected profile digest are required");
  }
  if (result.command == "infer" &&
      (result.input_list.empty() || result.asset_root.empty() || result.output_directory.empty() ||
       result.git_commit.empty() || result.configuration_sha256.empty() ||
       result.runtime_build_sha256.empty())) {
    throw std::invalid_argument("all documented infer arguments are required");
  }
  if (result.command == "benchmark" &&
      (result.input_list.empty() || result.asset_root.empty() || result.timing_output.empty() ||
       result.git_commit.empty() || result.configuration_sha256.empty() ||
       result.runtime_build_sha256.empty() || result.measured_frames == 0U)) {
    throw std::invalid_argument("all documented benchmark arguments are required");
  }
  if (result.input_profile != "full-file" && result.input_profile != "predecoded") {
    throw std::invalid_argument("input profile must be full-file or predecoded");
  }
  if (!result.onnx_profile_prefix.empty() &&
      (result.command != "benchmark" || !result.profiler_run)) {
    throw std::invalid_argument("ONNX profiling is allowed only for explicit profiler runs");
  }
  if (result.command == "doctor") {
    result.git_commit = std::string(40U, '0');
    result.configuration_sha256 = std::string(64U, '0');
    result.runtime_build_sha256 = std::string(64U, '0');
  }
  if (result.provider.profile != ExecutionProviderProfile::kCpuReference &&
      result.provider_log_output.empty()) {
    throw std::invalid_argument("accelerated profiles require --provider-log-output");
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
      arguments.provider,
      arguments.onnx_profile_prefix,
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

[[nodiscard]] std::string JsonString(const std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const char raw_character : value) {
    const auto character = static_cast<unsigned char>(raw_character);
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
  return output.str();
}

void WriteTextAtomically(const std::filesystem::path& path, const std::string_view content) {
  if (path.empty()) {
    return;
  }
  if (std::filesystem::exists(path)) {
    throw RuntimeError("RUNTIME_OUTPUT_EXISTS", "runtime refuses to overwrite provider log output");
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const std::filesystem::path temporary = path.string() + ".tmp";
  if (std::filesystem::exists(temporary)) {
    throw RuntimeError("RUNTIME_OUTPUT_EXISTS", "stale provider log temporary file exists");
  }
  try {
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      output.write(content.data(), static_cast<std::streamsize>(content.size()));
      output.flush();
      if (!output) {
        throw RuntimeError("RUNTIME_OUTPUT_IO", "provider log could not be written");
      }
    }
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

[[nodiscard]] std::string PrimaryProvider(const ExecutionProviderProfile profile) {
  if (profile == ExecutionProviderProfile::kCpuReference) {
    return "CPUExecutionProvider";
  }
  if (profile == ExecutionProviderProfile::kCuda) {
    return "CUDAExecutionProvider";
  }
  return "TensorrtExecutionProvider";
}

void WriteProviderLog(const Arguments& arguments, const CpuRuntime& runtime) {
  WriteTextAtomically(arguments.provider_log_output, runtime.diagnostics().provider_log);
}

int RunDoctor(const Arguments& arguments) {
  auto runtime = std::make_unique<CpuRuntime>(MakeOptions(arguments));
  WriteProviderLog(arguments, *runtime);
  const auto& diagnostics = runtime->diagnostics();
  std::cout << "{\"schema_version\":\"junctionlens.runtime-doctor.v1\",\"status\":\"PASSED\","
               "\"provider_profile\":"
            << JsonString(std::string(
                   junctionlens::infer::ExecutionProviderProfileName(arguments.provider.profile)))
            << ",\"available_providers\":[";
  for (std::size_t index = 0; index < diagnostics.available_providers.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << JsonString(diagnostics.available_providers[index]);
  }
  std::cout << "],\"onnxruntime_version\":" << JsonString(diagnostics.ort_version)
            << ",\"onnxruntime_library_sha256\":" << JsonString(diagnostics.ort_build_sha256)
            << ",\"model_sha256\":" << JsonString(diagnostics.model_sha256) << ",\"inputs\":[";
  for (std::size_t index = 0; index < diagnostics.input_names.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << JsonString(diagnostics.input_names[index]);
  }
  std::cout << "],\"outputs\":[";
  for (std::size_t index = 0; index < diagnostics.output_names.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << JsonString(diagnostics.output_names[index]);
  }
  std::cout << "],\"provider_assignment\":{\"node_counts\":{";
  std::size_t index = 0U;
  for (const auto& [provider, count] : diagnostics.provider_assignment.node_counts) {
    if (index++ != 0U) {
      std::cout << ',';
    }
    std::cout << JsonString(provider) << ':' << count;
  }
  std::cout << "},\"raw_log_sha256\":" << JsonString(diagnostics.provider_assignment.raw_log_sha256)
            << ",\"canonical_sha256\":"
            << JsonString(diagnostics.provider_assignment.canonical_sha256)
            << "},\"io_binding_enabled\":" << (diagnostics.io_binding_enabled ? "true" : "false")
            << ",\"provider_cache_key\":" << JsonString(diagnostics.provider_cache_key)
            << ",\"gpu\":";
  if (diagnostics.gpu_name.empty()) {
    std::cout << "null";
  } else {
    std::cout << "{\"name\":" << JsonString(diagnostics.gpu_name)
              << ",\"uuid\":" << JsonString(diagnostics.gpu_uuid) << ",\"compute_capability\":"
              << JsonString(std::to_string(diagnostics.gpu_compute_capability_major) + "." +
                            std::to_string(diagnostics.gpu_compute_capability_minor))
              << ",\"memory_bytes\":" << diagnostics.gpu_memory_bytes << '}';
  }
  std::cout << "}\n";
  return EXIT_SUCCESS;
}

struct BenchmarkSample {
  std::size_t iteration;
  std::string kind;
  RuntimePhaseTiming timing;
  RuntimeMemoryHighWater memory;
};

[[nodiscard]] std::string TimingJson(const RuntimePhaseTiming& timing) {
  std::ostringstream output;
  output << std::setprecision(17) << "{\"decode_ms\":" << timing.decode_ms
         << ",\"preprocess_ms\":" << timing.preprocess_ms
         << ",\"host_to_device_ms\":" << timing.host_to_device_ms
         << ",\"inference_ms\":" << timing.inference_ms
         << ",\"device_to_host_ms\":" << timing.device_to_host_ms
         << ",\"postprocess_ms\":" << timing.postprocess_ms << ",\"track_ms\":" << timing.track_ms
         << ",\"serialize_ms\":" << timing.serialize_ms
         << ",\"end_to_end_ms\":" << timing.end_to_end_ms << '}';
  return output.str();
}

void WriteBenchmarkReport(const Arguments& arguments, const CpuRuntime& runtime,
                          const double startup_ms, const std::vector<BenchmarkSample>& samples,
                          const std::size_t stability_processed,
                          const std::filesystem::path& onnx_profile_path) {
  const auto& diagnostics = runtime.diagnostics();
  std::ostringstream output;
  output << std::setprecision(17)
         << "{\"schema_version\":\"junctionlens.runtime-benchmark-raw.v1\","
            "\"status\":\"MEASURED_UNQUALIFIED\",\"publishable\":"
         << (arguments.profiler_run ? "false" : "true")
         << ",\"profiler_run\":" << (arguments.profiler_run ? "true" : "false")
         << ",\"onnx_profile_file\":"
         << (onnx_profile_path.empty() ? "null" : JsonString(onnx_profile_path.filename().string()))
         << ",\"clock_source\":"
         << JsonString(std::string(junctionlens::infer::MonotonicClockSource()))
         << ",\"input_profile\":" << JsonString(arguments.input_profile) << ",\"provider_profile\":"
         << JsonString(std::string(
                junctionlens::infer::ExecutionProviderProfileName(arguments.provider.profile)))
         << ",\"model_sha256\":" << JsonString(diagnostics.model_sha256)
         << ",\"provider_assignment_sha256\":"
         << JsonString(diagnostics.provider_assignment.canonical_sha256)
         << ",\"provider_node_counts\":{";
  std::size_t provider_index = 0U;
  for (const auto& [provider, count] : diagnostics.provider_assignment.node_counts) {
    if (provider_index++ != 0U) {
      output << ',';
    }
    output << JsonString(provider) << ':' << count;
  }
  output << "},\"startup_ms\":" << startup_ms << ",\"warmup_frames\":" << arguments.warmup_frames
         << ",\"measured_frames\":" << arguments.measured_frames
         << ",\"stability_frames\":" << arguments.stability_frames
         << ",\"stability_frames_processed\":" << stability_processed
         << ",\"memory_sample_period\":" << arguments.memory_sample_period << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    const BenchmarkSample& sample = samples[index];
    output << "{\"iteration\":" << sample.iteration
           << ",\"sample_kind\":" << JsonString(sample.kind)
           << ",\"phases\":" << TimingJson(sample.timing)
           << ",\"peak_resident_host_bytes\":" << sample.memory.peak_resident_host_bytes
           << ",\"current_device_bytes\":" << sample.memory.current_device_bytes
           << ",\"peak_device_bytes\":" << sample.memory.peak_device_bytes << '}';
  }
  output << "]}\n";
  WriteTextAtomically(arguments.timing_output, output.str());
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
  WriteProviderLog(arguments, *runtime);
  const auto provider_assignment = runtime->diagnostics().provider_assignment;
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
  std::cout
      << "{\"schema_version\":\"junctionlens.runtime-batch.v1\",\"status\":\"PASSED\","
         "\"provider\":"
      << JsonString(PrimaryProvider(arguments.provider.profile))
      << ",\"provider_assignment_sha256\":" << JsonString(provider_assignment.canonical_sha256)
      << ",\"provider_log_sha256\":" << JsonString(provider_assignment.raw_log_sha256)
      << ",\"io_binding_enabled\":"
      << (arguments.provider.profile == ExecutionProviderProfile::kCpuReference ? "false" : "true")
      << ",\"processed_frames\":" << processed << ",\"repeat_loads\":" << arguments.repeat_loads
      << ",\"buffer_capacity\":" << pool.capacity()
      << ",\"buffer_high_water_mark\":" << pool.high_water_mark() << ",\"all_slots_free\":true}\n";
  return EXIT_SUCCESS;
}

int RunBenchmark(const Arguments& arguments) {
  const auto input_paths = ReadInputList(arguments.input_list);
  const std::uint64_t startup_started = junctionlens::infer::MonotonicNowNanoseconds();
  auto runtime = LoadRuntime(arguments);
  const double startup_ms = junctionlens::infer::ElapsedMilliseconds(
      startup_started, junctionlens::infer::MonotonicNowNanoseconds());
  WriteProviderLog(arguments, *runtime);
  BufferPool pool(arguments.buffer_slots);
  std::vector<v1::SensorFrame> cached_frames;
  std::vector<junctionlens::infer::PreprocessedInputs> cached_inputs;
  if (arguments.input_profile == "predecoded") {
    cached_frames.reserve(input_paths.size());
    cached_inputs.reserve(input_paths.size());
    std::optional<v1::SensorFrame> previous;
    for (const auto& input_path : input_paths) {
      cached_frames.push_back(ParseSensorFrame(input_path));
      const v1::SensorFrame& current = cached_frames.back();
      const v1::SensorFrame* paired_previous =
          previous.has_value() && SameSequence(*previous, current) ? &*previous : nullptr;
      auto lease = pool.Acquire();
      cached_inputs.push_back(
          junctionlens::infer::Preprocess(paired_previous, current, arguments.asset_root, lease));
      lease.Release();
      previous = current;
    }
  }
  const std::size_t total_frames =
      arguments.warmup_frames + arguments.measured_frames + arguments.stability_frames;
  std::vector<BenchmarkSample> samples;
  samples.reserve(arguments.warmup_frames + arguments.measured_frames +
                  (arguments.stability_frames / arguments.memory_sample_period) + 2U);
  std::optional<v1::SensorFrame> previous;
  std::size_t stability_processed = 0U;
  for (std::size_t iteration = 0U; iteration < total_frames; ++iteration) {
    if (iteration == arguments.warmup_frames + arguments.measured_frames) {
      runtime->SetDeviceMemoryTracking(true);
    }
    const std::size_t input_index = iteration % input_paths.size();
    const std::uint64_t end_to_end_started = junctionlens::infer::MonotonicNowNanoseconds();
    auto lease = pool.Acquire();
    std::optional<v1::SensorFrame> loaded;
    const junctionlens::infer::PreprocessedInputs* selected = nullptr;
    std::optional<junctionlens::infer::PreprocessedInputs> transient;
    if (arguments.input_profile == "predecoded") {
      selected = &cached_inputs[input_index];
      lease.Advance(BufferState::kPreprocessing);
    } else {
      loaded = ParseSensorFrame(input_paths[input_index]);
      const v1::SensorFrame* paired_previous =
          previous.has_value() && SameSequence(*previous, *loaded) ? &*previous : nullptr;
      transient =
          junctionlens::infer::Preprocess(paired_previous, *loaded, arguments.asset_root, lease);
      selected = &*transient;
    }
    lease.Advance(BufferState::kInference);
    RuntimePhaseTiming timing;
    auto output = runtime->Infer(*selected, lease, &timing);
    if (arguments.input_profile == "predecoded") {
      timing.decode_ms = 0.0;
      timing.preprocess_ms = 0.0;
    }
    const std::uint64_t track_started = junctionlens::infer::MonotonicNowNanoseconds();
    {
      const junctionlens::infer::NvtxRange range("track");
      if (loaded.has_value()) {
        previous = *loaded;
      }
    }
    timing.track_ms = junctionlens::infer::ElapsedMilliseconds(
        track_started, junctionlens::infer::MonotonicNowNanoseconds());
    const std::uint64_t serialize_started = junctionlens::infer::MonotonicNowNanoseconds();
    std::string serialized;
    {
      const junctionlens::infer::NvtxRange range("serialize");
      if (!output.SerializeToString(&serialized) || serialized.empty()) {
        throw RuntimeError("RUNTIME_OUTPUT_IO", "benchmark output could not be serialized");
      }
    }
    timing.serialize_ms = junctionlens::infer::ElapsedMilliseconds(
        serialize_started, junctionlens::infer::MonotonicNowNanoseconds());
    timing.end_to_end_ms = junctionlens::infer::ElapsedMilliseconds(
        end_to_end_started, junctionlens::infer::MonotonicNowNanoseconds());
    lease.Release();
    const RuntimeMemoryHighWater memory = runtime->memory_high_water();
    const bool warmup = iteration < arguments.warmup_frames;
    const bool measured =
        !warmup && iteration < arguments.warmup_frames + arguments.measured_frames;
    if (warmup || measured) {
      samples.push_back({warmup ? iteration : iteration - arguments.warmup_frames,
                         warmup ? "warmup" : "measured", timing, memory});
    } else {
      ++stability_processed;
      if (stability_processed == 1U || stability_processed % arguments.memory_sample_period == 0U ||
          stability_processed == arguments.stability_frames) {
        samples.push_back({stability_processed - 1U, "stability", timing, memory});
      }
    }
  }
  if (!pool.all_free()) {
    throw RuntimeError("RUNTIME_BUFFER_LEAK", "benchmark buffer slots were not released");
  }
  const std::filesystem::path onnx_profile_path = runtime->EndProfiling();
  WriteBenchmarkReport(arguments, *runtime, startup_ms, samples, stability_processed,
                       onnx_profile_path);
  const RuntimeMemoryHighWater memory = runtime->memory_high_water();
  std::cout << "{\"schema_version\":\"junctionlens.runtime-benchmark-receipt.v1\","
               "\"status\":\"MEASURED_UNQUALIFIED\",\"timing_output\":"
            << JsonString(arguments.timing_output.string())
            << ",\"warmup_frames\":" << arguments.warmup_frames
            << ",\"measured_frames\":" << arguments.measured_frames
            << ",\"stability_frames\":" << stability_processed
            << ",\"peak_resident_host_bytes\":" << memory.peak_resident_host_bytes
            << ",\"peak_device_bytes\":" << memory.peak_device_bytes << "}\n";
  return EXIT_SUCCESS;
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    const Arguments arguments = ParseArguments(argc, argv);
    if (arguments.command == "doctor") {
      return RunDoctor(arguments);
    }
    return arguments.command == "benchmark" ? RunBenchmark(arguments) : Run(arguments);
  } catch (const RuntimeError& error) {
    std::cerr << "runtime error [" << error.reason_code() << "]: " << error.what() << '\n';
  } catch (const Ort::Exception& error) {
    std::cerr << "runtime error [RUNTIME_ONNX]: " << error.what() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "runtime error [RUNTIME_ARGUMENT]: " << error.what() << '\n';
  }
  return EXIT_FAILURE;
}

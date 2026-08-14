#include "junctionlens/infer/runtime.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace junctionlens::infer {
namespace {

constexpr std::string_view kQualifiedOrtVersion = "1.25.0";

[[nodiscard]] bool IsLowerHex(const std::string_view value, const std::size_t size) {
  return value.size() == size && std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

[[nodiscard]] std::string TrimRight(std::string value) {
  while (!value.empty() && (value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
    value.pop_back();
  }
  return value;
}

[[nodiscard]] std::size_t RequiredCount(const std::map<std::string, std::size_t>& counts,
                                        const std::string_view provider) {
  const auto found = counts.find(std::string(provider));
  return found == counts.end() ? 0U : found->second;
}

void ValidateProfileAssignment(const ProviderAssignment& assignment,
                               const ExecutionProviderProfile profile) {
  const std::size_t cpu = RequiredCount(assignment.node_counts, "CPUExecutionProvider");
  const std::size_t cuda = RequiredCount(assignment.node_counts, "CUDAExecutionProvider");
  const std::size_t tensorrt = RequiredCount(assignment.node_counts, "TensorrtExecutionProvider");
  if (profile == ExecutionProviderProfile::kCpuReference) {
    if (assignment.node_counts.size() != 1U || cpu == 0U) {
      throw RuntimeError("RUNTIME_PROVIDER_ASSIGNMENT",
                         "CPU profile did not assign every model node to CPU");
    }
    return;
  }
  if (cpu != 0U) {
    throw RuntimeError("RUNTIME_PROVIDER_FALLBACK",
                       "accelerated profile assigned model nodes to CPU");
  }
  if (profile == ExecutionProviderProfile::kCuda) {
    if (assignment.node_counts.size() != 1U || cuda == 0U) {
      throw RuntimeError("RUNTIME_PROVIDER_ASSIGNMENT",
                         "CUDA profile did not assign every model node to CUDA");
    }
    return;
  }
  if (tensorrt == 0U) {
    throw RuntimeError("RUNTIME_PROVIDER_ASSIGNMENT",
                       "TensorRT profile did not assign any model node to TensorRT");
  }
  for (const auto& [provider, count] : assignment.node_counts) {
    static_cast<void>(count);
    if (provider != "TensorrtExecutionProvider" && provider != "CUDAExecutionProvider" &&
        provider != "CPUExecutionProvider") {
      throw RuntimeError("RUNTIME_PROVIDER_ASSIGNMENT",
                         "TensorRT profile used an undeclared execution provider");
    }
  }
}

}  // namespace

std::string_view ExecutionProviderProfileName(const ExecutionProviderProfile profile) noexcept {
  switch (profile) {
    case ExecutionProviderProfile::kCpuReference:
      return "cpu-reference";
    case ExecutionProviderProfile::kCuda:
      return "cuda";
    case ExecutionProviderProfile::kTensorRt:
      return "tensorrt";
  }
  return "unknown";
}

ProviderAssignment ParseProviderAssignmentLog(const std::string_view raw_log,
                                              const std::string_view ort_version,
                                              const std::string_view ort_build_sha256,
                                              const ExecutionProviderProfile profile) {
  if (ort_version != kQualifiedOrtVersion || !IsLowerHex(ort_build_sha256, 64U)) {
    throw RuntimeError("RUNTIME_PROVIDER_PARSER_UNQUALIFIED",
                       "provider parser requires exact ONNX Runtime 1.25.0 build identity");
  }
  static const std::regex assignment_pattern(
      R"(^\s*(?:All nodes|Node\(s\)) placed on \[([A-Za-z0-9_]+)\]\. Number of nodes: ([0-9]+)$)");
  ProviderAssignment result;
  result.ort_version = std::string(ort_version);
  result.ort_build_sha256 = std::string(ort_build_sha256);
  result.raw_log_sha256 = Sha256Text(raw_log);
  std::istringstream lines{std::string(raw_log)};
  std::string line;
  std::string current_provider;
  bool saw_boundary = false;
  while (std::getline(lines, line)) {
    line = TrimRight(std::move(line));
    if (line == "Node placements") {
      saw_boundary = true;
      current_provider.clear();
      continue;
    }
    if (!saw_boundary) {
      continue;
    }
    std::smatch match;
    if (std::regex_match(line, match, assignment_pattern)) {
      current_provider = match[1].str();
      std::size_t count = 0U;
      const std::string count_text = match[2].str();
      const auto parsed =
          std::from_chars(count_text.data(), count_text.data() + count_text.size(), count);
      if (parsed.ec != std::errc{} || parsed.ptr != count_text.data() + count_text.size() ||
          count == 0U || result.node_counts.contains(current_provider)) {
        throw RuntimeError("RUNTIME_PROVIDER_LOG",
                           "provider assignment count is malformed or duplicated");
      }
      result.node_counts.emplace(current_provider, count);
      continue;
    }
    if (line.starts_with("  ") && !current_provider.empty()) {
      if (current_provider == "CPUExecutionProvider") {
        result.cpu_nodes.push_back(line.substr(2U));
      }
      continue;
    }
    current_provider.clear();
  }
  if (!saw_boundary || result.node_counts.empty()) {
    throw RuntimeError("RUNTIME_PROVIDER_LOG",
                       "qualified provider-assignment boundary is absent from the raw log");
  }
  ValidateProfileAssignment(result, profile);
  std::ostringstream canonical;
  canonical << "junctionlens-provider-assignment-v1\n"
            << result.ort_version << '\n'
            << result.ort_build_sha256 << '\n'
            << ExecutionProviderProfileName(profile) << '\n'
            << result.raw_log_sha256 << '\n';
  for (const auto& [provider, count] : result.node_counts) {
    canonical << provider << '=' << count << '\n';
  }
  for (const std::string& node : result.cpu_nodes) {
    canonical << "cpu-node=" << node << '\n';
  }
  result.canonical_sha256 = Sha256Text(canonical.str());
  return result;
}

std::string ProviderCacheKey(const std::string_view model_sha256, const ProviderOptions& options) {
  if (!IsLowerHex(model_sha256, 64U)) {
    throw RuntimeError("RUNTIME_CACHE_KEY", "model digest must be lowercase SHA-256");
  }
  if (options.profile == ExecutionProviderProfile::kTensorRt &&
      (options.gpu_compute_capability.empty() || options.cuda_version.empty() ||
       options.driver_compatibility_class.empty() || options.tensorrt_version.empty())) {
    throw RuntimeError("RUNTIME_CACHE_KEY",
                       "TensorRT cache identity is missing a compatibility dimension");
  }
  std::ostringstream canonical;
  canonical << "junctionlens-provider-cache-v1\n"
            << "model_sha256=" << model_sha256 << '\n'
            << "profile=" << ExecutionProviderProfileName(options.profile) << '\n'
            << "device_id=" << options.device_id << '\n'
            << "gpu_compute_capability=" << options.gpu_compute_capability << '\n'
            << "cuda_version=" << options.cuda_version << '\n'
            << "driver_compatibility_class=" << options.driver_compatibility_class << '\n'
            << "tensorrt_version=" << options.tensorrt_version << '\n'
            << "shape_profile=b1-t2-c8-rgb3-h384-w640-fixed-v1\n"
            << "cuda_options=arena:kNextPowerOfTwo,cudnn:EXHAUSTIVE,default_copy_stream:1,"
               "max_workspace:1\n"
            << "tensorrt_options=fp16:1,workspace:2147483648,builder_opt:3,aux_streams:0,"
               "timing_cache:1,force_timing_cache:0\n";
  return Sha256Text(canonical.str());
}

}  // namespace junctionlens::infer

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "junctionlens/v1/scene_control_graph.pb.h"

namespace junctionlens::infer {

class RuntimeError final : public std::runtime_error {
 public:
  RuntimeError(std::string reason_code, std::string detail);

  [[nodiscard]] const std::string& reason_code() const noexcept;

 private:
  std::string reason_code_;
};

enum class BufferState {
  kFree,
  kDecoding,
  kPreprocessing,
  kInference,
  kPostprocessing,
  kSerializing,
};

[[nodiscard]] std::string_view BufferStateName(BufferState state) noexcept;

class BufferPool;

class BufferLease final {
 public:
  BufferLease() noexcept = default;
  BufferLease(const BufferLease&) = delete;
  BufferLease& operator=(const BufferLease&) = delete;
  BufferLease(BufferLease&& other) noexcept;
  BufferLease& operator=(BufferLease&& other) noexcept;
  ~BufferLease();

  void Advance(BufferState next);
  void Release();
  [[nodiscard]] std::size_t slot_index() const;
  [[nodiscard]] BufferState state() const;

 private:
  friend class BufferPool;
  BufferLease(BufferPool* pool, std::size_t slot_index) noexcept;

  BufferPool* pool_ = nullptr;
  std::size_t slot_index_ = 0;
};

class BufferPool final {
 public:
  explicit BufferPool(std::size_t capacity);

  [[nodiscard]] BufferLease Acquire();
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t current_depth() const noexcept;
  [[nodiscard]] std::size_t high_water_mark() const noexcept;
  [[nodiscard]] bool all_free() const noexcept;
  [[nodiscard]] BufferState state(std::size_t slot_index) const;

 private:
  friend class BufferLease;
  void Advance(std::size_t slot_index, BufferState next);
  void Release(std::size_t slot_index) noexcept;

  std::vector<BufferState> states_;
  std::size_t current_depth_ = 0;
  std::size_t high_water_mark_ = 0;
};

struct ProducerOptions {
  std::string git_commit;
  bool git_dirty = false;
  std::string configuration_sha256;
  std::string runtime_build_sha256;
};

enum class ExecutionProviderProfile {
  kCpuReference,
  kCuda,
  kTensorRt,
};

[[nodiscard]] std::string_view ExecutionProviderProfileName(
    ExecutionProviderProfile profile) noexcept;

struct ProviderOptions {
  ExecutionProviderProfile profile = ExecutionProviderProfile::kCpuReference;
  int device_id = 0;
  std::filesystem::path cache_root;
  std::string gpu_compute_capability;
  std::string cuda_version;
  std::string driver_compatibility_class;
  std::string tensorrt_version;
};

struct ProviderAssignment {
  std::string ort_version;
  std::string ort_build_sha256;
  std::string raw_log_sha256;
  std::map<std::string, std::size_t> node_counts;
  std::vector<std::string> cpu_nodes;
  std::string canonical_sha256;
};

struct RuntimeDiagnostics {
  std::string ort_version;
  std::string ort_build_sha256;
  std::vector<std::string> available_providers;
  std::string model_sha256;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::string provider_log;
  ProviderAssignment provider_assignment;
  std::string provider_cache_key;
  bool io_binding_enabled = false;
  std::string gpu_name;
  std::string gpu_uuid;
  int gpu_compute_capability_major = 0;
  int gpu_compute_capability_minor = 0;
  std::uint64_t gpu_memory_bytes = 0U;
};

struct RuntimePhaseTiming {
  double decode_ms = 0.0;
  double preprocess_ms = 0.0;
  double host_to_device_ms = 0.0;
  double inference_ms = 0.0;
  double device_to_host_ms = 0.0;
  double postprocess_ms = 0.0;
  double track_ms = 0.0;
  double serialize_ms = 0.0;
  double end_to_end_ms = 0.0;
};

struct RuntimeMemoryHighWater {
  std::uint64_t peak_resident_host_bytes = 0U;
  std::uint64_t current_device_bytes = 0U;
  std::uint64_t peak_device_bytes = 0U;
};

struct RuntimeOptions {
  std::filesystem::path model_path;
  std::string expected_profile_sha256;
  ProducerOptions producer;
  double node_threshold = 0.5;
  double edge_threshold = 0.5;
  ProviderOptions provider;
  std::filesystem::path onnx_profile_prefix;
};

struct PreprocessedInputs {
  std::vector<float> images;
  std::vector<std::uint8_t> camera_valid;
  std::vector<float> intrinsics;
  std::vector<float> t_vehicle_camera;
  std::vector<float> ego_motion_previous_to_current;
  std::vector<std::uint8_t> temporal_valid;
  v1::SensorFrame output_sensor_frame;
  RuntimePhaseTiming timing;
};

[[nodiscard]] PreprocessedInputs Preprocess(const v1::SensorFrame* previous,
                                            const v1::SensorFrame& current,
                                            const std::filesystem::path& asset_root,
                                            BufferLease& lease);

class CpuRuntime final {
 public:
  explicit CpuRuntime(RuntimeOptions options);
  CpuRuntime(const CpuRuntime&) = delete;
  CpuRuntime& operator=(const CpuRuntime&) = delete;
  CpuRuntime(CpuRuntime&&) noexcept;
  CpuRuntime& operator=(CpuRuntime&&) noexcept;
  ~CpuRuntime();

  [[nodiscard]] v1::SceneControlGraphEnvelope Infer(const PreprocessedInputs& inputs,
                                                    BufferLease& lease,
                                                    RuntimePhaseTiming* timing = nullptr) const;
  [[nodiscard]] const RuntimeOptions& options() const noexcept;
  [[nodiscard]] const RuntimeDiagnostics& diagnostics() const noexcept;
  [[nodiscard]] RuntimeMemoryHighWater memory_high_water() const;
  void SetDeviceMemoryTracking(bool enabled) noexcept;
  [[nodiscard]] std::filesystem::path EndProfiling();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string Sha256File(const std::filesystem::path& path);
[[nodiscard]] std::string Sha256Text(std::string_view value);
[[nodiscard]] ProviderAssignment ParseProviderAssignmentLog(std::string_view raw_log,
                                                            std::string_view ort_version,
                                                            std::string_view ort_build_sha256,
                                                            ExecutionProviderProfile profile);
[[nodiscard]] std::string ProviderCacheKey(std::string_view model_sha256,
                                           const ProviderOptions& options);

}  // namespace junctionlens::infer

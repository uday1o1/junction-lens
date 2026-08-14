#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
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

struct RuntimeOptions {
  std::filesystem::path model_path;
  std::string expected_profile_sha256;
  ProducerOptions producer;
  double node_threshold = 0.5;
  double edge_threshold = 0.5;
};

struct PreprocessedInputs {
  std::vector<float> images;
  std::vector<std::uint8_t> camera_valid;
  std::vector<float> intrinsics;
  std::vector<float> t_vehicle_camera;
  std::vector<float> ego_motion_previous_to_current;
  std::vector<std::uint8_t> temporal_valid;
  v1::SensorFrame output_sensor_frame;
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
                                                    BufferLease& lease) const;
  [[nodiscard]] const RuntimeOptions& options() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string Sha256File(const std::filesystem::path& path);
[[nodiscard]] std::string Sha256Text(std::string_view value);

}  // namespace junctionlens::infer

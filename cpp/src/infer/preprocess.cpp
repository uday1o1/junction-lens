#include "junctionlens/infer/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace junctionlens::infer {
namespace {

constexpr int kTimestampCount = 2;
constexpr int kCameraCount = 8;
constexpr int kChannels = 3;
constexpr int kModelHeight = 384;
constexpr int kModelWidth = 640;
constexpr std::array<float, kChannels> kMean = {0.485F, 0.456F, 0.406F};
constexpr std::array<float, kChannels> kStandardDeviation = {0.229F, 0.224F, 0.225F};

using Matrix3f = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
using Matrix4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

[[nodiscard]] bool IsLowerHex(const std::string_view value, const std::size_t size) {
  return value.size() == size && std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

void RequireFrameKey(const v1::FrameKey& key, const std::string& label) {
  for (const auto& value :
       {key.dataset_id(), key.dataset_version(), key.split_id(), key.segment_id()}) {
    if (value.empty() || value.size() > 4096U) {
      throw RuntimeError("RUNTIME_FRAME_KEY", label + " contains an invalid identity string");
    }
  }
  if (key.timestamp_ns() < 0 || key.source_domain() == v1::SOURCE_DOMAIN_UNSPECIFIED ||
      !IsLowerHex(key.calibration_sha256(), 64U) || !IsLowerHex(key.frame_manifest_sha256(), 64U)) {
    throw RuntimeError("RUNTIME_FRAME_KEY", label + " contains invalid time or provenance");
  }
}

[[nodiscard]] std::size_t ImageIndex(const int timestamp, const int camera, const int channel,
                                     const int row, const int column) {
  return static_cast<std::size_t>(
      (((((timestamp * kCameraCount) + camera) * kChannels + channel) * kModelHeight + row) *
       kModelWidth) +
      column);
}

[[nodiscard]] std::size_t CameraIndex(const int timestamp, const int camera) {
  return static_cast<std::size_t>(timestamp * kCameraCount + camera);
}

[[nodiscard]] std::size_t MatrixIndex(const int timestamp, const int camera, const int width,
                                      const int row, const int column) {
  return static_cast<std::size_t>((((timestamp * kCameraCount) + camera) * width + row) * width +
                                  column);
}

[[nodiscard]] Matrix3f ReadMatrix3(const v1::Matrix3d& matrix, const std::string& label) {
  if (matrix.values_size() != 9) {
    throw RuntimeError("RUNTIME_INTRINSIC_SHAPE", label + " must contain 9 values");
  }
  Matrix3f result;
  for (int index = 0; index < 9; ++index) {
    const double value = matrix.values(index);
    if (!std::isfinite(value)) {
      throw RuntimeError("RUNTIME_NONFINITE_INPUT", label + " contains a nonfinite value");
    }
    result.data()[index] = static_cast<float>(value);
  }
  return result;
}

[[nodiscard]] Matrix4f ReadMatrix4(const v1::Matrix4d& matrix, const std::string& label) {
  if (matrix.values_size() != 16) {
    throw RuntimeError("RUNTIME_TRANSFORM_SHAPE", label + " must contain 16 values");
  }
  Matrix4f result;
  for (int index = 0; index < 16; ++index) {
    const double value = matrix.values(index);
    if (!std::isfinite(value)) {
      throw RuntimeError("RUNTIME_NONFINITE_INPUT", label + " contains a nonfinite value");
    }
    result.data()[index] = static_cast<float>(value);
  }
  return result;
}

void RequireCameraLayout(const v1::SensorFrame& frame, const std::string& label) {
  RequireFrameKey(frame.frame_key(), label + " frame key");
  if (frame.adapter_version().empty() || frame.adapter_version().size() > 4096U) {
    throw RuntimeError("RUNTIME_SENSOR_FRAME", label + " adapter version is invalid");
  }
  if (frame.cameras_size() != kCameraCount) {
    throw RuntimeError("RUNTIME_CAMERA_LAYOUT", label + " must contain exactly 8 cameras");
  }
  for (int index = 0; index < kCameraCount; ++index) {
    if (frame.cameras(index).slot() != index + 1) {
      throw RuntimeError("RUNTIME_CAMERA_LAYOUT", label + " cameras are not in canonical order");
    }
    const auto& camera = frame.cameras(index);
    if (camera.capture_timestamp_ns() < 0 ||
        camera.distortion_model() == v1::DISTORTION_MODEL_UNSPECIFIED) {
      throw RuntimeError("RUNTIME_CAMERA_LAYOUT", label + " contains invalid camera metadata");
    }
  }
}

[[nodiscard]] bool IsWithin(const std::filesystem::path& root,
                            const std::filesystem::path& candidate) {
  auto root_part = root.begin();
  auto candidate_part = candidate.begin();
  while (root_part != root.end() && candidate_part != candidate.end() &&
         *root_part == *candidate_part) {
    ++root_part;
    ++candidate_part;
  }
  return root_part == root.end();
}

[[nodiscard]] std::filesystem::path ResolveImage(const std::filesystem::path& asset_root,
                                                 const v1::ArtifactRef& artifact) {
  if (artifact.kind() != v1::ARTIFACT_KIND_SOURCE_IMAGE || !IsLowerHex(artifact.sha256(), 64U) ||
      artifact.media_type().empty() || artifact.license_id().empty()) {
    throw RuntimeError("RUNTIME_IMAGE_ARTIFACT", "source image artifact metadata is invalid");
  }
  const std::filesystem::path relative(artifact.relative_uri());
  if (relative.empty() || relative.is_absolute()) {
    throw RuntimeError("RUNTIME_IMAGE_PATH", "image artifact URI must be relative");
  }
  for (const auto& component : relative) {
    if (component == "..") {
      throw RuntimeError("RUNTIME_IMAGE_PATH", "image artifact URI cannot traverse parents");
    }
  }
  std::error_code error;
  const auto root = std::filesystem::weakly_canonical(asset_root, error);
  if (error) {
    throw RuntimeError("RUNTIME_ASSET_ROOT", "asset root could not be resolved");
  }
  const auto candidate = std::filesystem::weakly_canonical(root / relative, error);
  if (error || !IsWithin(root, candidate) || !std::filesystem::is_regular_file(candidate)) {
    throw RuntimeError("RUNTIME_IMAGE_PATH", "image artifact is absent or outside asset root");
  }
  if (artifact.byte_size() != std::filesystem::file_size(candidate)) {
    throw RuntimeError("RUNTIME_IMAGE_SIZE", "image byte size differs from its artifact reference");
  }
  if (Sha256File(candidate) != artifact.sha256()) {
    throw RuntimeError("RUNTIME_IMAGE_DIGEST", "image digest differs from its artifact reference");
  }
  return candidate;
}

struct Letterbox {
  int resized_width;
  int resized_height;
  int pad_left;
  int pad_top;
  float scale_x;
  float scale_y;
};

[[nodiscard]] Letterbox ComputeLetterbox(const int width, const int height) {
  if (width <= 0 || height <= 0) {
    throw RuntimeError("RUNTIME_IMAGE_DIMENSIONS", "decoded image dimensions must be positive");
  }
  const double scale = std::min(static_cast<double>(kModelWidth) / static_cast<double>(width),
                                static_cast<double>(kModelHeight) / static_cast<double>(height));
  const int resized_width =
      std::clamp(static_cast<int>(std::lround(static_cast<double>(width) * scale)), 1, kModelWidth);
  const int resized_height = std::clamp(
      static_cast<int>(std::lround(static_cast<double>(height) * scale)), 1, kModelHeight);
  return {
      resized_width,
      resized_height,
      (kModelWidth - resized_width) / 2,
      (kModelHeight - resized_height) / 2,
      static_cast<float>(resized_width) / static_cast<float>(width),
      static_cast<float>(resized_height) / static_cast<float>(height),
  };
}

void StoreMatrix(const Matrix3f& matrix, const int timestamp, const int camera,
                 std::vector<float>& values) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      values[MatrixIndex(timestamp, camera, 3, row, column)] = matrix(row, column);
    }
  }
}

void StoreMatrix(const Matrix4f& matrix, const int timestamp, const int camera,
                 std::vector<float>& values) {
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      values[MatrixIndex(timestamp, camera, 4, row, column)] = matrix(row, column);
    }
  }
}

void WriteImageTransform(const Letterbox& box, v1::CameraFrame& camera) {
  auto* transform = camera.mutable_image_transform();
  transform->set_resized_width(static_cast<std::uint32_t>(box.resized_width));
  transform->set_resized_height(static_cast<std::uint32_t>(box.resized_height));
  transform->set_crop_left(0U);
  transform->set_crop_top(0U);
  transform->set_pad_left(static_cast<std::uint32_t>(box.pad_left));
  transform->set_pad_top(static_cast<std::uint32_t>(box.pad_top));
  auto* matrix = transform->mutable_original_to_model();
  matrix->clear_values();
  for (const double value :
       {static_cast<double>(box.scale_x), 0.0, static_cast<double>(box.pad_left), 0.0,
        static_cast<double>(box.scale_y), static_cast<double>(box.pad_top), 0.0, 0.0, 1.0}) {
    matrix->add_values(value);
  }
}

[[nodiscard]] cv::Mat DecodeCamera(const v1::CameraFrame& source,
                                   const std::filesystem::path& asset_root) {
  if (!source.has_original_image()) {
    throw RuntimeError("RUNTIME_IMAGE_REQUIRED", "valid camera has no source image artifact");
  }
  const std::filesystem::path path = ResolveImage(asset_root, source.original_image());
  cv::Mat decoded = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (decoded.empty() || decoded.type() != CV_8UC3) {
    throw RuntimeError("RUNTIME_IMAGE_DECODE", "image could not be decoded as 8-bit color");
  }
  if (decoded.cols != static_cast<int>(source.original_width()) ||
      decoded.rows != static_cast<int>(source.original_height())) {
    throw RuntimeError("RUNTIME_IMAGE_DIMENSIONS",
                       "decoded image dimensions differ from the sensor contract");
  }
  return decoded;
}

void MaterializeCamera(const cv::Mat& decoded, const int timestamp, const int camera_index,
                       std::vector<float>& images, Matrix3f& intrinsic,
                       v1::CameraFrame* output_camera) {
  const Letterbox box = ComputeLetterbox(decoded.cols, decoded.rows);
  cv::Mat resized;
  cv::resize(decoded, resized, cv::Size(box.resized_width, box.resized_height), 0.0, 0.0,
             cv::INTER_LINEAR);
  for (int channel = 0; channel < kChannels; ++channel) {
    const float padding = -kMean[static_cast<std::size_t>(channel)] /
                          kStandardDeviation[static_cast<std::size_t>(channel)];
    for (int row = 0; row < kModelHeight; ++row) {
      for (int column = 0; column < kModelWidth; ++column) {
        images[ImageIndex(timestamp, camera_index, channel, row, column)] = padding;
      }
    }
  }
  for (int row = 0; row < box.resized_height; ++row) {
    for (int column = 0; column < box.resized_width; ++column) {
      const cv::Vec3b pixel = resized.at<cv::Vec3b>(row, column);
      for (int channel = 0; channel < kChannels; ++channel) {
        const float rgb = static_cast<float>(pixel[2 - channel]) / 255.0F;
        images[ImageIndex(timestamp, camera_index, channel, row + box.pad_top,
                          column + box.pad_left)] =
            (rgb - kMean[static_cast<std::size_t>(channel)]) /
            kStandardDeviation[static_cast<std::size_t>(channel)];
      }
    }
  }
  Matrix3f image_transform = Matrix3f::Identity();
  image_transform(0, 0) = box.scale_x;
  image_transform(1, 1) = box.scale_y;
  image_transform(0, 2) = static_cast<float>(box.pad_left);
  image_transform(1, 2) = static_cast<float>(box.pad_top);
  intrinsic = image_transform * intrinsic;
  if (output_camera != nullptr) {
    WriteImageTransform(box, *output_camera);
  }
}

[[nodiscard]] bool TemporalPairIsValid(const v1::SensorFrame* previous,
                                       const v1::SensorFrame& current) {
  if (previous == nullptr || !previous->pose_valid() || !current.pose_valid()) {
    return false;
  }
  const auto& first = previous->frame_key();
  const auto& second = current.frame_key();
  return first.dataset_id() == second.dataset_id() &&
         first.dataset_version() == second.dataset_version() &&
         first.split_id() == second.split_id() && first.segment_id() == second.segment_id() &&
         first.timestamp_ns() < second.timestamp_ns();
}

}  // namespace

PreprocessedInputs Preprocess(const v1::SensorFrame* previous, const v1::SensorFrame& current,
                              const std::filesystem::path& asset_root, BufferLease& lease) {
  cv::setNumThreads(1);
  RequireCameraLayout(current, "current sensor frame");
  const bool temporal_valid = TemporalPairIsValid(previous, current);
  if (previous != nullptr) {
    RequireCameraLayout(*previous, "previous sensor frame");
  }
  const v1::SensorFrame& first = temporal_valid ? *previous : current;
  const std::array<const v1::SensorFrame*, kTimestampCount> frames = {&first, &current};
  std::array<std::array<cv::Mat, kCameraCount>, kTimestampCount> decoded;
  for (int timestamp = 0; timestamp < kTimestampCount; ++timestamp) {
    for (int camera_index = 0; camera_index < kCameraCount; ++camera_index) {
      const auto& camera = frames[static_cast<std::size_t>(timestamp)]->cameras(camera_index);
      if (camera.valid()) {
        decoded[static_cast<std::size_t>(timestamp)][static_cast<std::size_t>(camera_index)] =
            DecodeCamera(camera, asset_root);
      }
    }
  }
  lease.Advance(BufferState::kPreprocessing);
  PreprocessedInputs result{
      std::vector<float>(static_cast<std::size_t>(kTimestampCount * kCameraCount * kChannels *
                                                  kModelHeight * kModelWidth),
                         0.0F),
      std::vector<std::uint8_t>(static_cast<std::size_t>(kTimestampCount * kCameraCount), 0U),
      std::vector<float>(static_cast<std::size_t>(kTimestampCount * kCameraCount * 9), 0.0F),
      std::vector<float>(static_cast<std::size_t>(kTimestampCount * kCameraCount * 16), 0.0F),
      std::vector<float>(16U, 0.0F),
      std::vector<std::uint8_t>(1U, temporal_valid ? 1U : 0U),
      current,
  };
  for (int timestamp = 0; timestamp < kTimestampCount; ++timestamp) {
    for (int camera_index = 0; camera_index < kCameraCount; ++camera_index) {
      const auto& camera = frames[static_cast<std::size_t>(timestamp)]->cameras(camera_index);
      Matrix3f intrinsic = ReadMatrix3(camera.intrinsic(), "camera intrinsic");
      const Matrix4f transform = ReadMatrix4(camera.t_vehicle_camera(), "camera transform");
      result.camera_valid[CameraIndex(timestamp, camera_index)] = camera.valid() ? 1U : 0U;
      if (camera.valid()) {
        v1::CameraFrame* output_camera =
            timestamp == 1 ? result.output_sensor_frame.mutable_cameras(camera_index) : nullptr;
        MaterializeCamera(
            decoded[static_cast<std::size_t>(timestamp)][static_cast<std::size_t>(camera_index)],
            timestamp, camera_index, result.images, intrinsic, output_camera);
        StoreMatrix(intrinsic, timestamp, camera_index, result.intrinsics);
        StoreMatrix(transform, timestamp, camera_index, result.t_vehicle_camera);
      }
    }
  }
  Matrix4f ego_motion = Matrix4f::Identity();
  if (temporal_valid) {
    const Matrix4f t_world_previous = ReadMatrix4(previous->t_world_vehicle(), "previous pose");
    const Matrix4f t_world_current = ReadMatrix4(current.t_world_vehicle(), "current pose");
    ego_motion = t_world_current.inverse() * t_world_previous;
  }
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      result.ego_motion_previous_to_current[static_cast<std::size_t>(row * 4 + column)] =
          ego_motion(row, column);
    }
  }
  return result;
}

}  // namespace junctionlens::infer

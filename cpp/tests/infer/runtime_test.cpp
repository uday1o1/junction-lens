#include "junctionlens/infer/runtime.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace junctionlens::infer {
namespace {

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    for (std::size_t suffix = 0U; suffix < 1000U; ++suffix) {
      path_ = std::filesystem::temp_directory_path() /
              ("junctionlens-infer-test-" + std::to_string(suffix));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
    }
    throw std::runtime_error("could not create inference test directory");
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

void AddIdentity(v1::Matrix3d& matrix) {
  for (const double value : {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}) {
    matrix.add_values(value);
  }
}

void AddIdentity(v1::Matrix4d& matrix) {
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      matrix.add_values(row == column ? 1.0 : 0.0);
    }
  }
}

[[nodiscard]] v1::SensorFrame MakeFrame(const std::filesystem::path& image_path) {
  v1::SensorFrame frame;
  auto* key = frame.mutable_frame_key();
  key->set_dataset_id("runtime-test");
  key->set_dataset_version("1");
  key->set_split_id("test");
  key->set_segment_id("segment");
  key->set_timestamp_ns(100);
  key->set_source_domain(v1::SOURCE_DOMAIN_SYNTHETIC);
  key->set_calibration_sha256(std::string(64U, 'a'));
  key->set_frame_manifest_sha256(std::string(64U, 'b'));
  frame.set_pose_valid(true);
  frame.set_adapter_version("runtime-test-v1");
  AddIdentity(*frame.mutable_t_world_vehicle());
  for (int index = 0; index < 8; ++index) {
    auto* camera = frame.add_cameras();
    camera->set_slot(static_cast<v1::CameraSlot>(index + 1));
    camera->set_valid(true);
    camera->set_capture_timestamp_ns(100);
    camera->set_original_width(640U);
    camera->set_original_height(384U);
    camera->set_distortion_model(v1::DISTORTION_MODEL_NONE);
    AddIdentity(*camera->mutable_intrinsic());
    AddIdentity(*camera->mutable_t_vehicle_camera());
    AddIdentity(*camera->mutable_image_transform()->mutable_original_to_model());
    auto* artifact = camera->mutable_original_image();
    artifact->set_kind(v1::ARTIFACT_KIND_SOURCE_IMAGE);
    artifact->set_sha256(Sha256File(image_path));
    artifact->set_byte_size(std::filesystem::file_size(image_path));
    artifact->set_media_type("image/x-portable-pixmap");
    artifact->set_relative_uri(image_path.filename().string());
    artifact->set_license_id("CC0-1.0");
  }
  return frame;
}

TEST(Sha256, MatchesPublishedVectors) {
  EXPECT_EQ(Sha256Text(""),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(Sha256Text("abc"),
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad");
}

TEST(ProviderAudit, ParsesQualifiedCpuAndCudaAssignments) {
  const std::string build_hash(64U, 'a');
  const auto cpu = ParseProviderAssignmentLog(
      "Node placements\n All nodes placed on [CPUExecutionProvider]. Number of nodes: 376\n",
      "1.25.0", build_hash, ExecutionProviderProfile::kCpuReference);
  EXPECT_EQ(cpu.node_counts.at("CPUExecutionProvider"), 376U);
  EXPECT_EQ(cpu.raw_log_sha256.size(), 64U);
  EXPECT_EQ(cpu.canonical_sha256.size(), 64U);

  const auto cuda = ParseProviderAssignmentLog(
      "Node placements\n All nodes placed on [CUDAExecutionProvider]. Number of nodes: 376\n",
      "1.25.0", build_hash, ExecutionProviderProfile::kCuda);
  EXPECT_EQ(cuda.node_counts.at("CUDAExecutionProvider"), 376U);
}

TEST(ProviderAudit, RejectsFallbackAndUnqualifiedBuilds) {
  const std::string raw =
      "Node placements\n"
      " Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 375\n"
      " Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1\n"
      "  shape_helper (Shape)\n";
  EXPECT_THROW(static_cast<void>(ParseProviderAssignmentLog(raw, "1.25.0", std::string(64U, 'a'),
                                                            ExecutionProviderProfile::kCuda)),
               RuntimeError);
  EXPECT_THROW(static_cast<void>(ParseProviderAssignmentLog(raw, "1.25.1", std::string(64U, 'a'),
                                                            ExecutionProviderProfile::kCuda)),
               RuntimeError);
}

TEST(ProviderAudit, RecordsPartialTensorRtAndContentAddressesCache) {
  const auto assignment = ParseProviderAssignmentLog(
      "Node placements\n"
      " Node(s) placed on [TensorrtExecutionProvider]. Number of nodes: 300\n"
      " Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 76\n",
      "1.25.0", std::string(64U, 'b'), ExecutionProviderProfile::kTensorRt);
  EXPECT_EQ(assignment.node_counts.at("TensorrtExecutionProvider"), 300U);
  EXPECT_EQ(assignment.node_counts.at("CUDAExecutionProvider"), 76U);
  ProviderOptions options;
  options.profile = ExecutionProviderProfile::kTensorRt;
  options.device_id = 0;
  options.gpu_compute_capability = "8.9";
  options.cuda_version = "12.8.1";
  options.driver_compatibility_class = "570";
  options.tensorrt_version = "10.14.1.48";
  const std::string first = ProviderCacheKey(std::string(64U, 'c'), options);
  options.driver_compatibility_class = "575";
  EXPECT_NE(first, ProviderCacheKey(std::string(64U, 'c'), options));
}

TEST(BufferPool, EnforcesOwnershipLifecycle) {
  BufferPool pool(2U);
  auto first = pool.Acquire();
  auto second = pool.Acquire();
  EXPECT_EQ(pool.high_water_mark(), 2U);
  EXPECT_THROW(static_cast<void>(pool.Acquire()), RuntimeError);
  first.Advance(BufferState::kPreprocessing);
  first.Advance(BufferState::kInference);
  first.Advance(BufferState::kPostprocessing);
  first.Advance(BufferState::kSerializing);
  first.Release();
  second.Release();
  EXPECT_TRUE(pool.all_free());
}

TEST(BufferPool, RejectsSkippedStageAndCleansUpErrors) {
  BufferPool pool(1U);
  {
    auto lease = pool.Acquire();
    EXPECT_THROW(lease.Advance(BufferState::kInference), RuntimeError);
  }
  EXPECT_TRUE(pool.all_free());
}

TEST(Preprocess, DecodesRgbNormalizesAndDuplicatesFirstFrame) {
  TemporaryDirectory directory;
  const auto image_path = directory.path() / "solid.ppm";
  {
    std::ofstream image(image_path, std::ios::binary);
    image << "P6\n640 384\n255\n";
    const std::array<char, 3> red = {static_cast<char>(255), 0, 0};
    for (std::size_t pixel = 0U; pixel < 640U * 384U; ++pixel) {
      image.write(red.data(), static_cast<std::streamsize>(red.size()));
    }
  }
  const v1::SensorFrame frame = MakeFrame(image_path);
  BufferPool pool(1U);
  auto lease = pool.Acquire();
  const auto inputs = Preprocess(nullptr, frame, directory.path(), lease);
  EXPECT_EQ(inputs.images.size(), 2U * 8U * 3U * 384U * 640U);
  EXPECT_EQ(inputs.temporal_valid, std::vector<std::uint8_t>({0U}));
  EXPECT_NEAR(inputs.images[0], (1.0F - 0.485F) / 0.229F, 1.0e-6F);
  EXPECT_NEAR(inputs.images[384U * 640U], -0.456F / 0.224F, 1.0e-6F);
  EXPECT_EQ(inputs.images[0], inputs.images[8U * 3U * 384U * 640U]);
  EXPECT_EQ(inputs.output_sensor_frame.cameras(0).image_transform().resized_width(), 640U);
  EXPECT_EQ(inputs.output_sensor_frame.cameras(0).image_transform().pad_top(), 0U);
}

TEST(Preprocess, RejectsArtifactDimensionMismatch) {
  TemporaryDirectory directory;
  const auto image_path = directory.path() / "one.ppm";
  {
    std::ofstream image(image_path, std::ios::binary);
    image << "P6\n1 1\n255\n";
    const std::array<char, 3> black = {0, 0, 0};
    image.write(black.data(), static_cast<std::streamsize>(black.size()));
  }
  const v1::SensorFrame frame = MakeFrame(image_path);
  BufferPool pool(1U);
  auto lease = pool.Acquire();
  EXPECT_THROW(static_cast<void>(Preprocess(nullptr, frame, directory.path(), lease)),
               RuntimeError);
}

}  // namespace
}  // namespace junctionlens::infer

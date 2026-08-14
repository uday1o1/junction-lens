#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace junctionlens::geometry {

using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix4d;
using Vector3d = Eigen::Vector3d;
using PointMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using PixelMatrix = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using CostMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Box = std::array<double, 4>;

class GeometryError final : public std::invalid_argument {
 public:
  GeometryError(std::string reason_code, std::string detail);

  [[nodiscard]] const std::string& reason_code() const noexcept;

 private:
  std::string reason_code_;
};

struct ProjectionResult {
  PixelMatrix pixels;
  std::vector<std::uint8_t> valid;
};

struct EndpointFeatures {
  double displacement_x;
  double displacement_y;
  double displacement_z;
  double distance;
  double tangent_cosine;
  double source_forward_cosine;
  double target_forward_cosine;

  [[nodiscard]] std::array<double, 7> AsArray() const noexcept;
};

[[nodiscard]] Matrix4d RigidTransform(const Matrix3d& rotation, const Vector3d& translation);
[[nodiscard]] Matrix4d ValidateTransform(const Matrix4d& transform);
[[nodiscard]] Matrix4d InvertTransform(const Matrix4d& transform);
[[nodiscard]] Matrix4d ComposeTransforms(const std::vector<Matrix4d>& transforms);
[[nodiscard]] PointMatrix TransformPoints(const Matrix4d& transform, const PointMatrix& points);
[[nodiscard]] Matrix4d OpenLaneToJunctionLens();
[[nodiscard]] PointMatrix AlignPointsBetweenVehicleFrames(const PointMatrix& previous_points,
                                                          const Matrix4d& t_world_previous_vehicle,
                                                          const Matrix4d& t_world_current_vehicle);

[[nodiscard]] ProjectionResult ProjectVehiclePoints(const PointMatrix& points_vehicle,
                                                    const Matrix3d& intrinsic,
                                                    const Matrix4d& t_vehicle_camera);
[[nodiscard]] PointMatrix BackprojectPixelsToPlane(const PixelMatrix& pixels,
                                                   const Matrix3d& intrinsic,
                                                   const Matrix4d& t_vehicle_camera,
                                                   double plane_z = 0.0);

[[nodiscard]] Matrix3d ImageTransform(double scale_x, double scale_y, double crop_left,
                                      double crop_top, double pad_left, double pad_top);
[[nodiscard]] Box TransformBox(const Box& box, const Matrix3d& transform);
[[nodiscard]] double HalfOpenIou(const Box& first, const Box& second);

[[nodiscard]] PointMatrix ResamplePolyline(const PointMatrix& points, std::size_t sample_count);
[[nodiscard]] double DiscreteFrechetDistance(const PointMatrix& first, const PointMatrix& second);
[[nodiscard]] double ChamferDistance(const PointMatrix& first, const PointMatrix& second);
[[nodiscard]] EndpointFeatures ComputeEndpointFeatures(const PointMatrix& source,
                                                       const PointMatrix& target);

[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> DeterministicHungarian(
    const CostMatrix& costs);

void ValidateLaneBoundaryOrientation(const PointMatrix& centerline,
                                     const PointMatrix& left_boundary,
                                     const PointMatrix& right_boundary);
void ValidateStrictlyIncreasingTimestamps(const std::vector<std::int64_t>& timestamps_ns);

}  // namespace junctionlens::geometry

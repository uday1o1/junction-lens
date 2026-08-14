#include "junctionlens/geometry/geometry.hpp"

#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace junctionlens::geometry {
namespace {

template <typename Derived>
void RequireFinite(const Eigen::MatrixBase<Derived>& value, const std::string& label) {
  if (!value.allFinite()) {
    throw GeometryError("GEOMETRY_NONFINITE", label + " contains a nonfinite value");
  }
}

void RequirePolyline(const PointMatrix& points, const std::string& label) {
  if (points.rows() < 2) {
    throw GeometryError("GEOMETRY_POLYLINE_SHAPE", label + " requires at least two points");
  }
  RequireFinite(points, label);
}

[[nodiscard]] Vector3d EndpointTangent(const PointMatrix& points, bool from_end) {
  if (from_end) {
    for (Eigen::Index index = points.rows() - 1; index > 0; --index) {
      const Vector3d delta = points.row(index).transpose() - points.row(index - 1).transpose();
      if (delta.norm() > 0.0) {
        return delta.normalized();
      }
    }
  } else {
    for (Eigen::Index index = 0; index + 1 < points.rows(); ++index) {
      const Vector3d delta = points.row(index + 1).transpose() - points.row(index).transpose();
      if (delta.norm() > 0.0) {
        return delta.normalized();
      }
    }
  }
  throw GeometryError("GEOMETRY_ENDPOINT_TANGENT",
                      "polyline endpoint tangent is undefined for duplicate-only points");
}

[[nodiscard]] std::vector<std::size_t> RowsToColumns(const CostMatrix& costs) {
  const std::size_t rows = static_cast<std::size_t>(costs.rows());
  const std::size_t columns = static_cast<std::size_t>(costs.cols());
  std::vector<double> row_potential(rows + 1U, 0.0);
  std::vector<double> column_potential(columns + 1U, 0.0);
  std::vector<std::size_t> column_row(columns + 1U, 0U);
  std::vector<std::size_t> predecessor(columns + 1U, 0U);
  for (std::size_t row = 1U; row <= rows; ++row) {
    column_row[0] = row;
    std::vector<double> minimum(columns + 1U, std::numeric_limits<double>::infinity());
    std::vector<bool> used(columns + 1U, false);
    std::size_t column = 0U;
    do {
      used[column] = true;
      const std::size_t active_row = column_row[column];
      double delta = std::numeric_limits<double>::infinity();
      std::size_t next_column = 0U;
      for (std::size_t candidate = 1U; candidate <= columns; ++candidate) {
        if (used[candidate]) {
          continue;
        }
        const double reduced = costs(static_cast<Eigen::Index>(active_row - 1U),
                                     static_cast<Eigen::Index>(candidate - 1U)) -
                               row_potential[active_row] - column_potential[candidate];
        if (reduced < minimum[candidate]) {
          minimum[candidate] = reduced;
          predecessor[candidate] = column;
        }
        if (minimum[candidate] < delta ||
            (minimum[candidate] == delta && candidate < next_column)) {
          delta = minimum[candidate];
          next_column = candidate;
        }
      }
      if (!std::isfinite(delta)) {
        throw GeometryError("GEOMETRY_ASSIGNMENT", "assignment has no finite augmenting path");
      }
      for (std::size_t candidate = 0U; candidate <= columns; ++candidate) {
        if (used[candidate]) {
          row_potential[column_row[candidate]] += delta;
          column_potential[candidate] -= delta;
        } else {
          minimum[candidate] -= delta;
        }
      }
      column = next_column;
    } while (column_row[column] != 0U);
    do {
      const std::size_t previous = predecessor[column];
      column_row[column] = column_row[previous];
      column = previous;
    } while (column != 0U);
  }
  std::vector<std::size_t> result(rows, 0U);
  for (std::size_t column = 1U; column <= columns; ++column) {
    if (column_row[column] != 0U) {
      result[column_row[column] - 1U] = column - 1U;
    }
  }
  return result;
}

}  // namespace

GeometryError::GeometryError(std::string reason_code, std::string detail)
    : std::invalid_argument(reason_code + ": " + detail), reason_code_(std::move(reason_code)) {}

const std::string& GeometryError::reason_code() const noexcept { return reason_code_; }

std::array<double, 7> EndpointFeatures::AsArray() const noexcept {
  return {
      displacement_x, displacement_y,        displacement_z,        distance,
      tangent_cosine, source_forward_cosine, target_forward_cosine,
  };
}

Matrix4d RigidTransform(const Matrix3d& rotation, const Vector3d& translation) {
  RequireFinite(rotation, "rotation");
  RequireFinite(translation, "translation");
  const double identity_error =
      (rotation.transpose() * rotation - Matrix3d::Identity()).cwiseAbs().maxCoeff();
  if (identity_error > 1.0e-6 || std::abs(rotation.determinant() - 1.0) > 1.0e-6) {
    throw GeometryError("GEOMETRY_TRANSFORM_RIGID", "rotation is not right-handed orthonormal");
  }
  Matrix4d result = Matrix4d::Identity();
  result.topLeftCorner<3, 3>() = rotation;
  result.topRightCorner<3, 1>() = translation;
  return result;
}

Matrix4d ValidateTransform(const Matrix4d& transform) {
  RequireFinite(transform, "transform");
  if ((transform.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).cwiseAbs().maxCoeff() > 1.0e-12) {
    throw GeometryError("GEOMETRY_TRANSFORM_AFFINE", "homogeneous final row is invalid");
  }
  return RigidTransform(transform.topLeftCorner<3, 3>(), transform.topRightCorner<3, 1>());
}

Matrix4d InvertTransform(const Matrix4d& transform) {
  const Matrix4d source = ValidateTransform(transform);
  Matrix4d result = Matrix4d::Identity();
  result.topLeftCorner<3, 3>() = source.topLeftCorner<3, 3>().transpose();
  result.topRightCorner<3, 1>() = -result.topLeftCorner<3, 3>() * source.topRightCorner<3, 1>();
  return result;
}

Matrix4d ComposeTransforms(const std::vector<Matrix4d>& transforms) {
  Matrix4d result = Matrix4d::Identity();
  for (const auto& transform : transforms) {
    result *= ValidateTransform(transform);
  }
  return ValidateTransform(result);
}

PointMatrix TransformPoints(const Matrix4d& transform, const PointMatrix& points) {
  const Matrix4d matrix = ValidateTransform(transform);
  RequireFinite(points, "points");
  PointMatrix result(points.rows(), 3);
  for (Eigen::Index index = 0; index < points.rows(); ++index) {
    result.row(index) = (matrix.topLeftCorner<3, 3>() * points.row(index).transpose() +
                         matrix.topRightCorner<3, 1>())
                            .transpose();
  }
  return result;
}

Matrix4d OpenLaneToJunctionLens() {
  Matrix4d result = Matrix4d::Identity();
  result.topLeftCorner<3, 3>() << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  return result;
}

PointMatrix AlignPointsBetweenVehicleFrames(const PointMatrix& previous_points,
                                            const Matrix4d& t_world_previous_vehicle,
                                            const Matrix4d& t_world_current_vehicle) {
  return TransformPoints(
      ComposeTransforms({InvertTransform(t_world_current_vehicle), t_world_previous_vehicle}),
      previous_points);
}

ProjectionResult ProjectVehiclePoints(const PointMatrix& points_vehicle, const Matrix3d& intrinsic,
                                      const Matrix4d& t_vehicle_camera) {
  RequireFinite(intrinsic, "intrinsic");
  if (std::abs(intrinsic.determinant()) <= 1.0e-12) {
    throw GeometryError("GEOMETRY_INTRINSIC", "intrinsic matrix is singular");
  }
  const PointMatrix camera_points =
      TransformPoints(InvertTransform(t_vehicle_camera), points_vehicle);
  ProjectionResult result{
      PixelMatrix::Constant(camera_points.rows(), 2, std::numeric_limits<double>::quiet_NaN()),
      std::vector<std::uint8_t>(static_cast<std::size_t>(camera_points.rows()), 0U),
  };
  for (Eigen::Index index = 0; index < camera_points.rows(); ++index) {
    if (camera_points(index, 2) <= 0.0) {
      continue;
    }
    const Vector3d projected = intrinsic * camera_points.row(index).transpose();
    result.pixels(index, 0) = projected.x() / projected.z();
    result.pixels(index, 1) = projected.y() / projected.z();
    result.valid[static_cast<std::size_t>(index)] = 1U;
  }
  return result;
}

PointMatrix BackprojectPixelsToPlane(const PixelMatrix& pixels, const Matrix3d& intrinsic,
                                     const Matrix4d& t_vehicle_camera, double plane_z) {
  RequireFinite(pixels, "pixels");
  RequireFinite(intrinsic, "intrinsic");
  if (!std::isfinite(plane_z) || std::abs(intrinsic.determinant()) <= 1.0e-12) {
    throw GeometryError("GEOMETRY_BACKPROJECT", "plane or intrinsic is invalid");
  }
  const Matrix4d transform = ValidateTransform(t_vehicle_camera);
  const Matrix3d inverse_intrinsic = intrinsic.inverse();
  const Vector3d origin = transform.topRightCorner<3, 1>();
  PointMatrix result(pixels.rows(), 3);
  for (Eigen::Index index = 0; index < pixels.rows(); ++index) {
    const Vector3d camera_ray =
        inverse_intrinsic * Vector3d(pixels(index, 0), pixels(index, 1), 1.0);
    const Vector3d vehicle_ray = transform.topLeftCorner<3, 3>() * camera_ray;
    if (std::abs(vehicle_ray.z()) < 1.0e-12) {
      throw GeometryError("GEOMETRY_BACKPROJECT", "camera ray is parallel to the target plane");
    }
    const double scale = (plane_z - origin.z()) / vehicle_ray.z();
    if (scale <= 0.0) {
      throw GeometryError("GEOMETRY_BACKPROJECT", "target plane lies behind a camera ray");
    }
    result.row(index) = (origin + scale * vehicle_ray).transpose();
  }
  return result;
}

Matrix3d ImageTransform(double scale_x, double scale_y, double crop_left, double crop_top,
                        double pad_left, double pad_top) {
  const std::array<double, 6> values{scale_x, scale_y, crop_left, crop_top, pad_left, pad_top};
  if (!std::all_of(values.begin(), values.end(),
                   [](double value) { return std::isfinite(value); }) ||
      scale_x <= 0.0 || scale_y <= 0.0) {
    throw GeometryError("GEOMETRY_IMAGE_TRANSFORM", "image transform values are invalid");
  }
  Matrix3d result = Matrix3d::Identity();
  result(0, 0) = scale_x;
  result(1, 1) = scale_y;
  result(0, 2) = pad_left - crop_left;
  result(1, 2) = pad_top - crop_top;
  return result;
}

Box TransformBox(const Box& box, const Matrix3d& transform) {
  RequireFinite(transform, "image transform");
  if (!std::all_of(box.begin(), box.end(), [](double value) { return std::isfinite(value); }) ||
      box[0] > box[2] || box[1] > box[3]) {
    throw GeometryError("GEOMETRY_BOX", "box is nonfinite or unordered");
  }
  const std::array<Vector3d, 4> corners{
      Vector3d(box[0], box[1], 1.0),
      Vector3d(box[0], box[3], 1.0),
      Vector3d(box[2], box[1], 1.0),
      Vector3d(box[2], box[3], 1.0),
  };
  Box result{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
  };
  for (const auto& corner : corners) {
    const Vector3d projected = transform * corner;
    if (std::abs(projected.z()) < 1.0e-12) {
      throw GeometryError("GEOMETRY_BOX", "box transform has a zero homogeneous coordinate");
    }
    const double horizontal = projected.x() / projected.z();
    const double vertical = projected.y() / projected.z();
    if (!std::isfinite(horizontal) || !std::isfinite(vertical)) {
      throw GeometryError("GEOMETRY_BOX", "box transform produced a nonfinite coordinate");
    }
    result[0] = std::min(result[0], horizontal);
    result[1] = std::min(result[1], vertical);
    result[2] = std::max(result[2], horizontal);
    result[3] = std::max(result[3], vertical);
  }
  return result;
}

double HalfOpenIou(const Box& first, const Box& second) {
  const auto area = [](const Box& box) {
    if (!std::all_of(box.begin(), box.end(), [](double value) { return std::isfinite(value); }) ||
        box[0] > box[2] || box[1] > box[3]) {
      throw GeometryError("GEOMETRY_BOX", "box is nonfinite or unordered");
    }
    return (box[2] - box[0]) * (box[3] - box[1]);
  };
  const double intersection_width =
      std::max(0.0, std::min(first[2], second[2]) - std::max(first[0], second[0]));
  const double intersection_height =
      std::max(0.0, std::min(first[3], second[3]) - std::max(first[1], second[1]));
  const double intersection = intersection_width * intersection_height;
  const double union_area = area(first) + area(second) - intersection;
  return union_area > 0.0 ? intersection / union_area : 0.0;
}

PointMatrix ResamplePolyline(const PointMatrix& points, std::size_t sample_count) {
  RequirePolyline(points, "polyline");
  if (sample_count < 2U ||
      sample_count > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max())) {
    throw GeometryError("GEOMETRY_SAMPLE_COUNT", "sample count must be at least two");
  }
  std::vector<double> lengths(static_cast<std::size_t>(points.rows() - 1), 0.0);
  for (Eigen::Index index = 0; index + 1 < points.rows(); ++index) {
    lengths[static_cast<std::size_t>(index)] = (points.row(index + 1) - points.row(index)).norm();
  }
  const double total = std::accumulate(lengths.begin(), lengths.end(), 0.0);
  if (total <= 0.0) {
    throw GeometryError("GEOMETRY_POLYLINE_LENGTH", "polyline has no distinct points");
  }
  std::vector<double> cumulative(lengths.size() + 1U, 0.0);
  std::partial_sum(lengths.begin(), lengths.end(), cumulative.begin() + 1);
  PointMatrix result(static_cast<Eigen::Index>(sample_count), 3);
  std::size_t segment = 0U;
  for (std::size_t index = 0U; index < sample_count; ++index) {
    const double distance =
        total * static_cast<double>(index) / static_cast<double>(sample_count - 1U);
    while (segment + 1U < lengths.size() && distance > cumulative[segment + 1U]) {
      ++segment;
    }
    while (segment < lengths.size() && lengths[segment] == 0.0) {
      ++segment;
    }
    if (segment == lengths.size()) {
      result.row(static_cast<Eigen::Index>(index)) = points.row(points.rows() - 1);
      continue;
    }
    const double fraction =
        std::clamp((distance - cumulative[segment]) / lengths[segment], 0.0, 1.0);
    result.row(static_cast<Eigen::Index>(index)) =
        points.row(static_cast<Eigen::Index>(segment)) +
        fraction * (points.row(static_cast<Eigen::Index>(segment + 1U)) -
                    points.row(static_cast<Eigen::Index>(segment)));
  }
  result.row(0) = points.row(0);
  result.row(result.rows() - 1) = points.row(points.rows() - 1);
  return result;
}

double DiscreteFrechetDistance(const PointMatrix& first, const PointMatrix& second) {
  RequirePolyline(first, "first");
  RequirePolyline(second, "second");
  CostMatrix dynamic(first.rows(), second.rows());
  for (Eigen::Index row = 0; row < first.rows(); ++row) {
    for (Eigen::Index column = 0; column < second.rows(); ++column) {
      const double distance = (first.row(row) - second.row(column)).norm();
      if (row == 0 && column == 0) {
        dynamic(row, column) = distance;
      } else if (row == 0) {
        dynamic(row, column) = std::max(dynamic(row, column - 1), distance);
      } else if (column == 0) {
        dynamic(row, column) = std::max(dynamic(row - 1, column), distance);
      } else {
        dynamic(row, column) =
            std::max(distance, std::min({dynamic(row - 1, column), dynamic(row - 1, column - 1),
                                         dynamic(row, column - 1)}));
      }
    }
  }
  return dynamic(dynamic.rows() - 1, dynamic.cols() - 1);
}

double ChamferDistance(const PointMatrix& first, const PointMatrix& second) {
  RequirePolyline(first, "first");
  RequirePolyline(second, "second");
  double left_sum = 0.0;
  for (Eigen::Index row = 0; row < first.rows(); ++row) {
    double minimum = std::numeric_limits<double>::infinity();
    for (Eigen::Index column = 0; column < second.rows(); ++column) {
      minimum = std::min(minimum, (first.row(row) - second.row(column)).norm());
    }
    left_sum += minimum;
  }
  double right_sum = 0.0;
  for (Eigen::Index column = 0; column < second.rows(); ++column) {
    double minimum = std::numeric_limits<double>::infinity();
    for (Eigen::Index row = 0; row < first.rows(); ++row) {
      minimum = std::min(minimum, (second.row(column) - first.row(row)).norm());
    }
    right_sum += minimum;
  }
  return 0.5 * (left_sum / static_cast<double>(first.rows()) +
                right_sum / static_cast<double>(second.rows()));
}

EndpointFeatures ComputeEndpointFeatures(const PointMatrix& source, const PointMatrix& target) {
  RequirePolyline(source, "source");
  RequirePolyline(target, "target");
  const Vector3d source_tangent = EndpointTangent(source, true);
  const Vector3d target_tangent = EndpointTangent(target, false);
  const Vector3d displacement =
      target.row(0).transpose() - source.row(source.rows() - 1).transpose();
  const double distance = displacement.norm();
  const Vector3d direction = distance > 0.0 ? displacement / distance : source_tangent;
  return {
      displacement.x(),
      displacement.y(),
      displacement.z(),
      distance,
      std::clamp(source_tangent.dot(target_tangent), -1.0, 1.0),
      std::clamp(source_tangent.dot(direction), -1.0, 1.0),
      std::clamp(target_tangent.dot(direction), -1.0, 1.0),
  };
}

std::vector<std::pair<std::size_t, std::size_t>> DeterministicHungarian(const CostMatrix& costs) {
  RequireFinite(costs, "cost matrix");
  if (costs.rows() == 0 || costs.cols() == 0) {
    return {};
  }
  std::vector<std::pair<std::size_t, std::size_t>> result;
  if (costs.rows() <= costs.cols()) {
    const auto columns = RowsToColumns(costs);
    result.reserve(columns.size());
    for (std::size_t row = 0U; row < columns.size(); ++row) {
      result.emplace_back(row, columns[row]);
    }
  } else {
    CostMatrix transposed = costs.transpose();
    const auto rows = RowsToColumns(transposed);
    result.reserve(rows.size());
    for (std::size_t column = 0U; column < rows.size(); ++column) {
      result.emplace_back(rows[column], column);
    }
    std::sort(result.begin(), result.end());
  }
  return result;
}

void ValidateLaneBoundaryOrientation(const PointMatrix& centerline,
                                     const PointMatrix& left_boundary,
                                     const PointMatrix& right_boundary) {
  RequirePolyline(centerline, "centerline");
  RequirePolyline(left_boundary, "left boundary");
  RequirePolyline(right_boundary, "right boundary");
  if (centerline.rows() != left_boundary.rows() || centerline.rows() != right_boundary.rows()) {
    throw GeometryError("GEOMETRY_LANE_SHAPE", "lane polylines must have matching point counts");
  }
  for (Eigen::Index index = 0; index + 1 < centerline.rows(); ++index) {
    const Eigen::Vector2d tangent =
        (centerline.row(index + 1).head<2>() - centerline.row(index).head<2>()).transpose();
    if (tangent.norm() <= 0.0) {
      throw GeometryError("GEOMETRY_LANE_DIRECTION", "centerline has a zero-length segment");
    }
    const Eigen::Vector2d direction = tangent.normalized();
    const Eigen::Vector2d center_mid =
        0.5 * (centerline.row(index).head<2>() + centerline.row(index + 1).head<2>()).transpose();
    const Eigen::Vector2d left_mid =
        0.5 *
        (left_boundary.row(index).head<2>() + left_boundary.row(index + 1).head<2>()).transpose();
    const Eigen::Vector2d right_mid =
        0.5 *
        (right_boundary.row(index).head<2>() + right_boundary.row(index + 1).head<2>()).transpose();
    const Eigen::Vector2d left_offset = left_mid - center_mid;
    const Eigen::Vector2d right_offset = right_mid - center_mid;
    const double left_cross = direction.x() * left_offset.y() - direction.y() * left_offset.x();
    const double right_cross = direction.x() * right_offset.y() - direction.y() * right_offset.x();
    if (left_cross <= 0.0 || right_cross >= 0.0) {
      throw GeometryError("GEOMETRY_LANE_ORIENTATION", "boundaries violate left/right convention");
    }
  }
}

void ValidateStrictlyIncreasingTimestamps(const std::vector<std::int64_t>& timestamps_ns) {
  if (timestamps_ns.empty()) {
    throw GeometryError("GEOMETRY_TIMESTAMPS", "timestamp sequence is empty");
  }
  std::int64_t previous = -1;
  for (const auto timestamp : timestamps_ns) {
    if (timestamp < 0 || timestamp <= previous) {
      throw GeometryError("GEOMETRY_TIMESTAMPS", "timestamps are not strictly increasing");
    }
    previous = timestamp;
  }
}

}  // namespace junctionlens::geometry

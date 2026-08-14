#include "junctionlens/geometry/geometry.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace junctionlens::geometry {
namespace {

[[nodiscard]] PointMatrix Points(std::initializer_list<std::array<double, 3>> values) {
  PointMatrix result(static_cast<Eigen::Index>(values.size()), 3);
  Eigen::Index row = 0;
  for (const auto& point : values) {
    result.row(row++) << point[0], point[1], point[2];
  }
  return result;
}

TEST(Transform, InverseCompositionMeetsIdentityGate) {
  const double angle = 0.73;
  Matrix3d rotation;
  rotation << std::cos(angle), -std::sin(angle), 0.0, std::sin(angle), std::cos(angle), 0.0, 0.0,
      0.0, 1.0;
  const Matrix4d transform = RigidTransform(rotation, Vector3d(12.0, -3.0, 1.5));
  const Matrix4d identity = ComposeTransforms({transform, InvertTransform(transform)});
  EXPECT_LE((identity - Matrix4d::Identity()).cwiseAbs().maxCoeff(), 1.0e-6);
}

TEST(Transform, OpenLaneBasisAxesAndRoundTripAreExact) {
  const PointMatrix source = Points({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
  const PointMatrix expected = Points({{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}});
  const auto canonical = TransformPoints(OpenLaneToJunctionLens(), source);
  EXPECT_TRUE(canonical.isApprox(expected, 0.0));
  const auto round_trip = TransformPoints(InvertTransform(OpenLaneToJunctionLens()), canonical);
  EXPECT_LE((round_trip - source).cwiseAbs().maxCoeff(), 1.0e-5);
}

TEST(Transform, EgoMotionAlignmentUsesWorldPoseDirection) {
  const PointMatrix previous = Points({{5.0, 0.0, 0.0}});
  const Matrix4d world_previous = Matrix4d::Identity();
  const Matrix4d world_current = RigidTransform(Matrix3d::Identity(), Vector3d(2.0, 0.0, 0.0));
  const auto aligned = AlignPointsBetweenVehicleFrames(previous, world_previous, world_current);
  EXPECT_TRUE(aligned.isApprox(Points({{3.0, 0.0, 0.0}}), 1.0e-12));
}

TEST(Projection, KnownPixelsAndGroundPlaneRoundTripMeetGates) {
  Matrix3d rotation;
  rotation << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
  const Matrix4d vehicle_camera = RigidTransform(rotation, Vector3d(0.0, 0.0, 1.0));
  Matrix3d intrinsic;
  intrinsic << 100.0, 0.0, 320.0, 0.0, 100.0, 192.0, 0.0, 0.0, 1.0;
  const PointMatrix expected = Points({{5.0, 0.0, 0.0}, {10.0, -2.0, 0.0}});
  const auto projection = ProjectVehiclePoints(expected, intrinsic, vehicle_camera);
  ASSERT_EQ(projection.valid, (std::vector<std::uint8_t>{1U, 1U}));
  PixelMatrix pixel_golden(2, 2);
  pixel_golden << 320.0, 212.0, 340.0, 202.0;
  EXPECT_LE((projection.pixels - pixel_golden).cwiseAbs().maxCoeff(), 0.25);
  const auto observed = BackprojectPixelsToPlane(projection.pixels, intrinsic, vehicle_camera);
  EXPECT_LE((observed - expected).rowwise().norm().maxCoeff(), 1.0e-5);
}

TEST(ImageGeometry, ResizeCropPadAndHalfOpenBoxesAreExact) {
  const Matrix3d transform = ImageTransform(0.5, 0.5, 20.0, 10.0, 4.0, 8.0);
  const Box source{40.0, 30.0, 80.0, 70.0};
  const Box model = TransformBox(source, transform);
  const Box round_trip = TransformBox(model, transform.inverse());
  for (std::size_t index = 0; index < source.size(); ++index) {
    EXPECT_NEAR(round_trip[index], source[index], 1.0e-9);
  }
  EXPECT_DOUBLE_EQ(HalfOpenIou({0.0, 0.0, 10.0, 10.0}, {10.0, 0.0, 20.0, 10.0}), 0.0);
  EXPECT_DOUBLE_EQ(HalfOpenIou({0.0, 0.0, 100.0, 80.0}, {0.0, 0.0, 100.0, 80.0}), 1.0);
  EXPECT_DOUBLE_EQ(HalfOpenIou({4.0, 5.0, 5.0, 6.0}, {4.0, 5.0, 5.0, 6.0}), 1.0);
}

TEST(Curves, InterpolationDistancesAndEndpointFeaturesMatchGoldens) {
  const PointMatrix first = Points({{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {2.0, 2.0, 0.0}});
  const auto sampled = ResamplePolyline(first, 5U);
  EXPECT_TRUE(sampled.isApprox(Points({
                                   {0.0, 0.0, 0.0},
                                   {1.0, 0.0, 0.0},
                                   {2.0, 0.0, 0.0},
                                   {2.0, 1.0, 0.0},
                                   {2.0, 2.0, 0.0},
                               }),
                               1.0e-12));
  const PointMatrix translated = first.rowwise() + Eigen::RowVector3d(0.0, 1.0, 0.0);
  EXPECT_NEAR(DiscreteFrechetDistance(first, translated), 1.0, 1.0e-12);
  EXPECT_NEAR(ChamferDistance(first, translated), 1.0, 1.0e-12);
  const auto features = ComputeEndpointFeatures(Points({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}),
                                                Points({{2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}}));
  EXPECT_EQ(features.AsArray(), (std::array<double, 7>{1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0}));
}

TEST(Matching, RectangularOptimumAndTiesAreDeterministic) {
  CostMatrix costs(2, 3);
  costs << 4.0, 1.0, 1.0, 2.0, 0.0, 5.0;
  EXPECT_EQ(DeterministicHungarian(costs),
            (std::vector<std::pair<std::size_t, std::size_t>>{{0U, 2U}, {1U, 1U}}));
  const CostMatrix ties = CostMatrix::Zero(3, 2);
  const auto expected = std::vector<std::pair<std::size_t, std::size_t>>{{0U, 0U}, {1U, 1U}};
  for (int repeat = 0; repeat < 10; ++repeat) {
    EXPECT_EQ(DeterministicHungarian(ties), expected);
  }
}

TEST(Conventions, LaneOrientationAndTimestampsFailClosed) {
  const PointMatrix center = Points({{0.0, 0.0, 0.0}, {5.0, 0.0, 0.0}});
  const PointMatrix left = Points({{0.0, 1.0, 0.0}, {5.0, 1.0, 0.0}});
  const PointMatrix right = Points({{0.0, -1.0, 0.0}, {5.0, -1.0, 0.0}});
  EXPECT_NO_THROW(ValidateLaneBoundaryOrientation(center, left, right));
  EXPECT_THROW(ValidateLaneBoundaryOrientation(center, right, left), GeometryError);
  EXPECT_NO_THROW(ValidateStrictlyIncreasingTimestamps({1, 2, 4}));
  EXPECT_THROW(ValidateStrictlyIncreasingTimestamps({1, 1}), GeometryError);
  EXPECT_THROW(ValidateStrictlyIncreasingTimestamps({2, 1}), GeometryError);
}

TEST(Malformed, ShapesNonfiniteAndDegeneracyAreRejected) {
  PointMatrix one_point = Points({{0.0, 0.0, 0.0}});
  EXPECT_THROW(static_cast<void>(ResamplePolyline(one_point, 2U)), GeometryError);
  PointMatrix nonfinite = Points({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}});
  nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(static_cast<void>(ChamferDistance(nonfinite, nonfinite)), GeometryError);
  EXPECT_THROW(static_cast<void>(HalfOpenIou({2.0, 0.0, 1.0, 1.0}, {0.0, 0.0, 1.0, 1.0})),
               GeometryError);
  CostMatrix invalid(1, 1);
  invalid(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(static_cast<void>(DeterministicHungarian(invalid)), GeometryError);
}

}  // namespace
}  // namespace junctionlens::geometry

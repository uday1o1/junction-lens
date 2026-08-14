"""Geometry-core goldens, properties, and malformed-input controls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn
from scipy.optimize import linear_sum_assignment

from junctionlens.geometry import (
    GeometryError,
    align_points_between_vehicle_frames,
    chamfer_distance,
    deterministic_hungarian,
    discrete_frechet_distance,
    endpoint_features,
    half_open_iou,
    image_transform,
    resample_polyline,
    rigid_transform,
    transform_box,
    validate_lane_boundary_orientation,
    validate_strictly_increasing_timestamps,
)


@st.composite
def _cost_matrices(draw: DrawFn) -> np.ndarray:
    rows = draw(st.integers(min_value=1, max_value=5))
    columns = draw(st.integers(min_value=1, max_value=5))
    values = draw(
        st.lists(
            st.integers(min_value=-20, max_value=20),
            min_size=rows * columns,
            max_size=rows * columns,
        )
    )
    return np.asarray(values, dtype=np.float64).reshape(rows, columns)


@st.composite
def _translated_polylines(draw: DrawFn) -> tuple[np.ndarray, np.ndarray]:
    point_count = draw(st.integers(min_value=2, max_value=8))
    values = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=-20, max_value=20),
                st.integers(min_value=-20, max_value=20),
                st.integers(min_value=-4, max_value=4),
            ),
            min_size=point_count,
            max_size=point_count,
        )
    )
    first = np.asarray(values, dtype=np.float64)
    offset = np.asarray(
        [
            draw(st.integers(min_value=-5, max_value=5)),
            draw(st.integers(min_value=-5, max_value=5)),
            draw(st.integers(min_value=-2, max_value=2)),
        ],
        dtype=np.float64,
    )
    return first, first + offset


def test_polyline_interpolation_distances_and_endpoint_feature_goldens() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    np.testing.assert_allclose(
        resample_polyline(source, 5),
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
        atol=1e-12,
        rtol=0.0,
    )
    translated = source + np.asarray([0.0, 1.0, 0.0])
    assert discrete_frechet_distance(source, translated) == pytest.approx(1.0)
    assert chamfer_distance(source, translated) == pytest.approx(1.0)
    features = endpoint_features(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(features.as_array(), [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])


@given(_translated_polylines())
def test_polyline_distances_are_symmetric_and_nonnegative(
    polylines: tuple[np.ndarray, np.ndarray],
) -> None:
    first, second = polylines
    frechet = discrete_frechet_distance(first, second)
    chamfer = chamfer_distance(first, second)
    assert frechet >= 0.0
    assert chamfer >= 0.0
    assert frechet == pytest.approx(discrete_frechet_distance(second, first))
    assert chamfer == pytest.approx(chamfer_distance(second, first))


@given(_cost_matrices())
def test_hungarian_wrapper_matches_scipy_optimum(costs: np.ndarray) -> None:
    observed = deterministic_hungarian(costs)
    rows, columns = linear_sum_assignment(costs)
    expected_cost = float(costs[rows, columns].sum())
    observed_cost = sum(float(costs[row, column]) for row, column in observed)
    assert len(observed) == min(costs.shape)
    assert len({row for row, _ in observed}) == len(observed)
    assert len({column for _, column in observed}) == len(observed)
    assert observed_cost == pytest.approx(expected_cost)


def test_hungarian_ties_are_repeatable_and_lexicographically_stable() -> None:
    costs = np.zeros((3, 2), dtype=np.float64)
    expected = ((0, 0), (1, 1))
    assert all(deterministic_hungarian(costs) == expected for _ in range(20))


def test_ego_alignment_lane_orientation_image_transform_and_time_goldens() -> None:
    identity = rigid_transform(np.eye(3), [0.0, 0.0, 0.0], label="previous")
    current = rigid_transform(np.eye(3), [2.0, 0.0, 0.0], label="current")
    aligned = align_points_between_vehicle_frames([[5.0, 0.0, 0.0]], identity, current)
    np.testing.assert_allclose(aligned, [[3.0, 0.0, 0.0]], atol=1e-12, rtol=0.0)

    center = np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    left = center + np.asarray([0.0, 1.0, 0.0])
    right = center + np.asarray([0.0, -1.0, 0.0])
    validate_lane_boundary_orientation(center, left, right)
    with pytest.raises(GeometryError):
        validate_lane_boundary_orientation(center, right, left)

    pixels = image_transform(
        0.5,
        0.5,
        crop_left=20.0,
        crop_top=10.0,
        pad_left=4.0,
        pad_top=8.0,
    )
    source_box = (40.0, 30.0, 80.0, 70.0)
    np.testing.assert_allclose(
        transform_box(transform_box(source_box, pixels), np.linalg.inv(pixels)),
        source_box,
        atol=1e-9,
        rtol=0.0,
    )
    validate_strictly_increasing_timestamps([0, 1, 4])
    with pytest.raises(GeometryError):
        validate_strictly_increasing_timestamps([1, 1])


@pytest.mark.parametrize(
    "call",
    [
        lambda: discrete_frechet_distance([[0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]),
        lambda: chamfer_distance([[0.0, 0.0], [np.nan, 1.0]], [[0.0, 0.0], [1.0, 1.0]]),
        lambda: deterministic_hungarian([[np.inf]]),
        lambda: half_open_iou((0.0, 0.0, np.nan, 1.0), (0.0, 0.0, 1.0, 1.0)),
        lambda: image_transform(0.0, 1.0),
        lambda: validate_strictly_increasing_timestamps([]),
        lambda: validate_strictly_increasing_timestamps([2, 1]),
    ],
)
def test_malformed_inputs_fail_closed(call: Callable[[], object]) -> None:
    with pytest.raises(GeometryError):
        call()

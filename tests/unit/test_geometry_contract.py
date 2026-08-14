"""Coordinate and image convention goldens and properties."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from junctionlens.data.geometry import (
    backproject_pixels_to_plane,
    compose_transforms,
    denormalize_half_open_box,
    half_open_iou,
    invert_transform,
    junctionlens_points_to_openlane,
    letterbox_transform,
    normalize_half_open_box,
    openlane_points_to_junctionlens,
    project_vehicle_points,
    rigid_transform,
    transform_box,
)


def test_openlane_basis_change_axis_goldens() -> None:
    """Source right/forward/up becomes canonical left/forward/up exactly."""
    source = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    observed = openlane_points_to_junctionlens(source)
    expected = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(observed, expected)


@given(
    st.lists(
        st.tuples(
            st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
            st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
            st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_basis_change_inverse_round_trip(points: list[tuple[float, float, float]]) -> None:
    """The declared source and canonical bases are exact inverses."""
    source = np.asarray(points, dtype=np.float64)
    round_trip = junctionlens_points_to_openlane(openlane_points_to_junctionlens(source))
    np.testing.assert_allclose(round_trip, source, atol=1e-12, rtol=0.0)


@given(st.floats(-math.pi, math.pi), st.floats(-20.0, 20.0), st.floats(-20.0, 20.0))
def test_rigid_inverse_and_composition(angle: float, x: float, y: float) -> None:
    """Rigid transforms close to identity well below the 1e-6 gate."""
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = rigid_transform(rotation, [x, y, 2.0], label="fixture")
    identity = compose_transforms(transform, invert_transform(transform))
    np.testing.assert_allclose(identity, np.eye(4), atol=1e-10, rtol=0.0)


def test_camera_projection_and_ground_plane_round_trip() -> None:
    """A declared camera ray returns to its ground-plane point within 1e-5 meter."""
    rotation = np.asarray([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    t_vehicle_camera = rigid_transform(rotation, [0.0, 0.0, 1.0], label="camera")
    intrinsic = np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 192.0], [0.0, 0.0, 1.0]])
    expected = np.asarray([[5.0, 0.0, 0.0], [10.0, -2.0, 0.0]])
    pixels, valid = project_vehicle_points(expected, intrinsic, t_vehicle_camera)
    assert valid.tolist() == [True, True]
    observed = backproject_pixels_to_plane(pixels, intrinsic, t_vehicle_camera)
    np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=0.0)


def test_resize_pad_and_box_round_trip() -> None:
    """Letterbox geometry and normalized half-open boxes are lossless in floating point."""
    transform = letterbox_transform(1920, 1080)
    source_box = (0.0, 0.0, 1920.0, 1080.0)
    model_box = transform_box(source_box, transform)
    round_trip = transform_box(model_box, np.linalg.inv(transform))
    np.testing.assert_allclose(round_trip, source_box, atol=1e-9, rtol=0.0)
    normalized = normalize_half_open_box((0.0, 0.0, 1.0, 1.0), 1920, 1080)
    assert denormalize_half_open_box(normalized, 1920, 1080) == (0.0, 0.0, 1.0, 1.0)


def test_half_open_box_iou_goldens() -> None:
    """Touching borders have no area while full-image and one-pixel boxes are exact."""
    assert half_open_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert half_open_iou((0, 0, 100, 80), (0, 0, 100, 80)) == 1.0
    assert half_open_iou((4, 5, 5, 6), (4, 5, 5, 6)) == 1.0

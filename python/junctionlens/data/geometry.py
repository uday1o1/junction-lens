"""Frozen coordinate, projection, and image-box conventions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

OPENLANE_TO_JUNCTIONLENS: FloatArray = np.asarray(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class GeometryError(ValueError):
    """Raised when a source transform or geometry violates the contract."""


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise GeometryError(f"{label} must have shape {shape}, observed {array.shape}")
    if not np.isfinite(array).all():
        raise GeometryError(f"{label} contains a nonfinite value")
    return array


def rigid_transform(rotation: object, translation: object, *, label: str) -> FloatArray:
    """Construct and validate one right-handed rigid transform."""
    rotation_array = _finite_array(rotation, (3, 3), f"{label}.rotation")
    translation_array = _finite_array(translation, (3,), f"{label}.translation")
    identity_error = float(
        np.max(np.abs(rotation_array.T @ rotation_array - np.eye(3, dtype=np.float64)))
    )
    determinant = float(np.linalg.det(rotation_array))
    if identity_error > 1e-6 or not math.isclose(determinant, 1.0, abs_tol=1e-6):
        raise GeometryError(
            f"{label}.rotation is not rigid: identity_error={identity_error}, det={determinant}"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_array
    transform[:3, 3] = translation_array
    return transform


def validate_transform(transform: object, *, label: str) -> FloatArray:
    """Validate one homogeneous rigid transform and return a copy."""
    array = _finite_array(transform, (4, 4), label)
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise GeometryError(f"{label} has an invalid homogeneous bottom row")
    return rigid_transform(array[:3, :3], array[:3, 3], label=label)


def invert_transform(transform: object) -> FloatArray:
    """Invert a validated rigid transform without a generic matrix inverse."""
    source = validate_transform(transform, label="transform")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = source[:3, :3].T
    result[:3, 3] = -(source[:3, :3].T @ source[:3, 3])
    return result


def compose_transforms(*transforms: object) -> FloatArray:
    """Compose transforms left to right under T_target_source notation."""
    result = np.eye(4, dtype=np.float64)
    for index, transform in enumerate(transforms):
        result = result @ validate_transform(transform, label=f"transform[{index}]")
    return validate_transform(result, label="composed_transform")


def transform_points(transform: object, points: object) -> FloatArray:
    """Apply a column-vector homogeneous transform to N by 3 points."""
    matrix = validate_transform(transform, label="point_transform")
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3 or not np.isfinite(point_array).all():
        raise GeometryError("points must be a finite N by 3 array")
    homogeneous = np.concatenate(
        [point_array, np.ones((point_array.shape[0], 1), dtype=np.float64)], axis=1
    )
    return cast(FloatArray, (matrix @ homogeneous.T).T[:, :3])


def openlane_points_to_junctionlens(points: object) -> FloatArray:
    """Map OpenLane x-right, y-forward, z-up points into JunctionLens axes."""
    return transform_points(OPENLANE_TO_JUNCTIONLENS, points)


def junctionlens_points_to_openlane(points: object) -> FloatArray:
    """Map JunctionLens points back into OpenLane axes."""
    return transform_points(invert_transform(OPENLANE_TO_JUNCTIONLENS), points)


def openlane_camera_to_junctionlens_vehicle(rotation: object, translation: object) -> FloatArray:
    """Convert upstream camera-to-OpenLane-vehicle extrinsics into canonical axes."""
    t_openlane_camera = rigid_transform(rotation, translation, label="openlane_extrinsic")
    return compose_transforms(OPENLANE_TO_JUNCTIONLENS, t_openlane_camera)


def openlane_pose_to_junctionlens_world(rotation: object, translation: object) -> FloatArray:
    """Apply the source-to-canonical basis change on both sides of an ego pose."""
    t_source_world_vehicle = rigid_transform(rotation, translation, label="openlane_pose")
    return compose_transforms(
        OPENLANE_TO_JUNCTIONLENS,
        t_source_world_vehicle,
        invert_transform(OPENLANE_TO_JUNCTIONLENS),
    )


def project_vehicle_points(
    points_vehicle: object,
    intrinsic: object,
    t_vehicle_camera: object,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Project vehicle-frame points and return image pixels plus positive-depth mask."""
    points = np.asarray(points_vehicle, dtype=np.float64)
    camera_points = transform_points(invert_transform(t_vehicle_camera), points)
    valid = camera_points[:, 2] > 0.0
    pixels = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    if bool(valid.any()):
        calibration = _finite_array(intrinsic, (3, 3), "intrinsic")
        projected = (calibration @ camera_points[valid].T).T
        pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def backproject_pixels_to_plane(
    pixels: object,
    intrinsic: object,
    t_vehicle_camera: object,
    *,
    plane_z: float = 0.0,
) -> FloatArray:
    """Back-project pixels to a horizontal vehicle-frame plane."""
    pixel_array = np.asarray(pixels, dtype=np.float64)
    if pixel_array.ndim != 2 or pixel_array.shape[1] != 2 or not np.isfinite(pixel_array).all():
        raise GeometryError("pixels must be a finite N by 2 array")
    calibration = _finite_array(intrinsic, (3, 3), "intrinsic")
    transform = validate_transform(t_vehicle_camera, label="t_vehicle_camera")
    camera_rays = (
        np.linalg.inv(calibration)
        @ np.concatenate([pixel_array, np.ones((len(pixel_array), 1), dtype=np.float64)], axis=1).T
    ).T
    vehicle_rays = (transform[:3, :3] @ camera_rays.T).T
    origin = transform[:3, 3]
    denominator = vehicle_rays[:, 2]
    if np.any(np.abs(denominator) < 1e-12):
        raise GeometryError("a camera ray is parallel to the target plane")
    scales = (plane_z - origin[2]) / denominator
    if np.any(scales <= 0.0):
        raise GeometryError("the target plane lies behind a camera ray")
    return cast(FloatArray, origin[None, :] + scales[:, None] * vehicle_rays)


def letterbox_transform(
    source_width: int,
    source_height: int,
    *,
    target_width: int = 640,
    target_height: int = 384,
) -> FloatArray:
    """Return the deterministic source-pixel to model-pixel homography."""
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise GeometryError("image dimensions must be positive")
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = source_width * scale
    resized_height = source_height * scale
    pad_x = (target_width - resized_width) / 2.0
    pad_y = (target_height - resized_height) / 2.0
    return np.asarray(
        [[scale, 0.0, pad_x], [0.0, scale, pad_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def transform_box(box: Sequence[float], homography: object) -> tuple[float, float, float, float]:
    """Transform a continuous half-open axis-aligned box."""
    if len(box) != 4:
        raise GeometryError("a box must contain four coordinates")
    u_min, v_min, u_max, v_max = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (u_min, v_min, u_max, v_max)):
        raise GeometryError("box coordinates must be finite")
    if u_min > u_max or v_min > v_max:
        raise GeometryError("box minima cannot exceed maxima")
    matrix = _finite_array(homography, (3, 3), "image_homography")
    corners = np.asarray([[u_min, v_min, 1.0], [u_max, v_max, 1.0]], dtype=np.float64)
    transformed = (matrix @ corners.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    return (
        float(transformed[0, 0]),
        float(transformed[0, 1]),
        float(transformed[1, 0]),
        float(transformed[1, 1]),
    )


def normalize_half_open_box(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Normalize by width and height, never width minus one or height minus one."""
    if image_width <= 0 or image_height <= 0:
        raise GeometryError("image dimensions must be positive")
    u_min, v_min, u_max, v_max = (float(value) for value in box)
    if not (0.0 <= u_min <= u_max <= image_width and 0.0 <= v_min <= v_max <= image_height):
        raise GeometryError("box lies outside the declared image extent")
    return (
        u_min / image_width,
        v_min / image_height,
        u_max / image_width,
        v_max / image_height,
    )


def denormalize_half_open_box(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert a normalized model box back to continuous source pixels once."""
    if image_width <= 0 or image_height <= 0:
        raise GeometryError("image dimensions must be positive")
    u_min, v_min, u_max, v_max = (float(value) for value in box)
    if not (0.0 <= u_min <= u_max <= 1.0 and 0.0 <= v_min <= v_max <= 1.0):
        raise GeometryError("normalized box lies outside zero to one")
    return (
        u_min * image_width,
        v_min * image_height,
        u_max * image_width,
        v_max * image_height,
    )


def half_open_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute continuous IoU for half-open boxes, including border-touching boxes."""
    a_u0, a_v0, a_u1, a_v1 = (float(value) for value in first)
    b_u0, b_v0, b_u1, b_v1 = (float(value) for value in second)
    if min(a_u1 - a_u0, a_v1 - a_v0, b_u1 - b_u0, b_v1 - b_v0) < 0.0:
        raise GeometryError("box minima cannot exceed maxima")
    intersection_width = max(0.0, min(a_u1, b_u1) - max(a_u0, b_u0))
    intersection_height = max(0.0, min(a_v1, b_v1) - max(a_v0, b_v0))
    intersection = intersection_width * intersection_height
    union = (a_u1 - a_u0) * (a_v1 - a_v0) + (b_u1 - b_u0) * (b_v1 - b_v0) - intersection
    return intersection / union if union > 0.0 else 0.0

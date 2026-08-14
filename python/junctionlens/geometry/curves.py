"""Polyline interpolation, distance, and endpoint features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from junctionlens.data.geometry import GeometryError

FloatArray = npt.NDArray[np.float64]


def _polyline(value: object, label: str) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] not in {2, 3}:
        raise GeometryError(f"{label} must be a finite N by 2 or N by 3 polyline with N >= 2")
    if not np.isfinite(points).all():
        raise GeometryError(f"{label} contains a nonfinite value")
    return points


def resample_polyline(points: object, sample_count: int) -> FloatArray:
    """Resample a polyline at equal arc-length intervals including both endpoints."""
    source = _polyline(points, "polyline")
    if sample_count < 2:
        raise GeometryError("sample_count must be at least two")
    segment_lengths = np.linalg.norm(np.diff(source, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 0.0:
        raise GeometryError("polyline must contain at least two distinct points")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, total_length, sample_count, dtype=np.float64)
    result = np.empty((sample_count, source.shape[1]), dtype=np.float64)
    segment = 0
    for index, distance in enumerate(targets):
        while segment + 1 < len(segment_lengths) and distance > cumulative[segment + 1]:
            segment += 1
        length = segment_lengths[segment]
        if length == 0.0:
            next_segment = segment
            while next_segment < len(segment_lengths) and segment_lengths[next_segment] == 0.0:
                next_segment += 1
            if next_segment == len(segment_lengths):
                result[index] = source[-1]
                continue
            segment = next_segment
            length = segment_lengths[segment]
        fraction = min(1.0, max(0.0, (distance - cumulative[segment]) / length))
        result[index] = source[segment] + fraction * (source[segment + 1] - source[segment])
    result[0] = source[0]
    result[-1] = source[-1]
    return result


def discrete_frechet_distance(first: object, second: object) -> float:
    """Compute the standard discrete Fréchet distance between ordered polylines."""
    left = _polyline(first, "first")
    right = _polyline(second, "second")
    if left.shape[1] != right.shape[1]:
        raise GeometryError("polylines must have the same coordinate dimension")
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    dynamic = np.empty_like(distances)
    dynamic[0, 0] = distances[0, 0]
    for row in range(1, len(left)):
        dynamic[row, 0] = max(dynamic[row - 1, 0], distances[row, 0])
    for column in range(1, len(right)):
        dynamic[0, column] = max(dynamic[0, column - 1], distances[0, column])
    for row in range(1, len(left)):
        for column in range(1, len(right)):
            dynamic[row, column] = max(
                distances[row, column],
                min(
                    dynamic[row - 1, column],
                    dynamic[row - 1, column - 1],
                    dynamic[row, column - 1],
                ),
            )
    return float(dynamic[-1, -1])


def chamfer_distance(first: object, second: object) -> float:
    """Return the symmetric mean Euclidean nearest-point Chamfer distance."""
    left = _polyline(first, "first")
    right = _polyline(second, "second")
    if left.shape[1] != right.shape[1]:
        raise GeometryError("polylines must have the same coordinate dimension")
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2.0)


def _endpoint_tangent(points: FloatArray, *, from_end: bool) -> FloatArray:
    indices = range(len(points) - 1, 0, -1) if from_end else range(len(points) - 1)
    for index in indices:
        delta = points[index] - points[index - 1] if from_end else points[index + 1] - points[index]
        length = float(np.linalg.norm(delta))
        if length > 0.0:
            return cast(FloatArray, delta / length)
    raise GeometryError("polyline endpoint tangent is undefined for duplicate-only points")


@dataclass(frozen=True, slots=True)
class EndpointFeatures:
    """Frozen successor features from a source end to a target start."""

    displacement_x: float
    displacement_y: float
    displacement_z: float
    distance: float
    tangent_cosine: float
    source_forward_cosine: float
    target_forward_cosine: float

    def as_array(self) -> FloatArray:
        """Return features in their frozen model order."""
        return np.asarray(
            [
                self.displacement_x,
                self.displacement_y,
                self.displacement_z,
                self.distance,
                self.tangent_cosine,
                self.source_forward_cosine,
                self.target_forward_cosine,
            ],
            dtype=np.float64,
        )


def endpoint_features(source: object, target: object) -> EndpointFeatures:
    """Compute deterministic lane-successor endpoint geometry features."""
    source_points = _polyline(source, "source")
    target_points = _polyline(target, "target")
    if source_points.shape[1] != target_points.shape[1]:
        raise GeometryError("polylines must have the same coordinate dimension")
    if source_points.shape[1] == 2:
        source_points = np.pad(source_points, ((0, 0), (0, 1)))
        target_points = np.pad(target_points, ((0, 0), (0, 1)))
    source_tangent = _endpoint_tangent(source_points, from_end=True)
    target_tangent = _endpoint_tangent(target_points, from_end=False)
    displacement = target_points[0] - source_points[-1]
    distance = float(np.linalg.norm(displacement))
    direction = displacement / distance if distance > 0.0 else source_tangent
    return EndpointFeatures(
        displacement_x=float(displacement[0]),
        displacement_y=float(displacement[1]),
        displacement_z=float(displacement[2]),
        distance=distance,
        tangent_cosine=float(np.clip(source_tangent @ target_tangent, -1.0, 1.0)),
        source_forward_cosine=float(np.clip(source_tangent @ direction, -1.0, 1.0)),
        target_forward_cosine=float(np.clip(target_tangent @ direction, -1.0, 1.0)),
    )

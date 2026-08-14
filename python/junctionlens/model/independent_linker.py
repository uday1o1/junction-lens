"""Frozen geometric topology rules for the E0 independent baseline."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from junctionlens.model.e0_profile import E0Profile
from junctionlens.registry.store import canonical_json_bytes


class IndependentLinkerError(ValueError):
    """Raised when geometric-linker inputs or fit provenance are invalid."""


@dataclass(frozen=True, slots=True)
class RuleObservation:
    """One training-only candidate edge and its two frozen geometric features."""

    stable_id: str
    distance: float
    heading_difference_deg: float
    outcome: int

    def __post_init__(self) -> None:
        if (
            not self.stable_id
            or not math.isfinite(self.distance)
            or not math.isfinite(self.heading_difference_deg)
            or self.distance < 0.0
            or not 0.0 <= self.heading_difference_deg <= 180.0
            or self.outcome not in {0, 1}
        ):
            raise IndependentLinkerError("rule observation is invalid")


@dataclass(frozen=True, slots=True)
class FittedThreshold:
    distance: float
    heading_difference_deg: float
    true_positive: int
    false_positive: int
    false_negative: int
    f1: float


@dataclass(frozen=True, slots=True)
class IndependentLinkerArtifact:
    schema_version: str
    algorithm: str
    fit_partition: str
    training_split_manifest_sha256: str
    profile_sha256: str
    successor: FittedThreshold
    control: FittedThreshold
    observation_sha256: str

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(asdict(self)) + b"\n"


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fit(
    observations: tuple[RuleObservation, ...],
    distances: tuple[float, ...],
    headings: tuple[float, ...],
) -> FittedThreshold:
    if not observations or not any(item.outcome for item in observations):
        raise IndependentLinkerError("threshold fitting requires observations and a positive edge")
    if len({item.stable_id for item in observations}) != len(observations):
        raise IndependentLinkerError("rule observation IDs are duplicated")
    candidates: list[FittedThreshold] = []
    for distance, heading in itertools.product(distances, headings):
        predicted = [
            item.distance <= distance and item.heading_difference_deg <= heading
            for item in observations
        ]
        true_positive = sum(
            prediction and item.outcome == 1
            for prediction, item in zip(predicted, observations, strict=True)
        )
        false_positive = sum(
            prediction and item.outcome == 0
            for prediction, item in zip(predicted, observations, strict=True)
        )
        false_negative = sum(
            not prediction and item.outcome == 1
            for prediction, item in zip(predicted, observations, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
        candidates.append(
            FittedThreshold(
                distance,
                heading,
                true_positive,
                false_positive,
                false_negative,
                f1,
            )
        )
    return min(candidates, key=lambda item: (-item.f1, item.distance, item.heading_difference_deg))


def fit_independent_linker(
    profile: E0Profile,
    successor_observations: tuple[RuleObservation, ...],
    control_observations: tuple[RuleObservation, ...],
    *,
    partition: str,
    training_split_manifest_sha256: str,
) -> IndependentLinkerArtifact:
    """Fit both rule pairs exclusively from the declared model-training partition."""
    if partition != profile.independent_linker.fit_partition:
        raise IndependentLinkerError("independent linker may be fit only on model_training")
    if len(training_split_manifest_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in training_split_manifest_sha256
    ):
        raise IndependentLinkerError("training split manifest hash is invalid")
    search = profile.independent_linker
    observations = {
        "successor": [
            asdict(item) for item in sorted(successor_observations, key=lambda x: x.stable_id)
        ],
        "control": [
            asdict(item) for item in sorted(control_observations, key=lambda x: x.stable_id)
        ],
    }
    return IndependentLinkerArtifact(
        schema_version="junctionlens.independent-linker.v1",
        algorithm="bounded-grid-max-f1-smallest-threshold-tie-v1",
        fit_partition=partition,
        training_split_manifest_sha256=training_split_manifest_sha256,
        profile_sha256=profile.canonical_sha256(),
        successor=_fit(
            successor_observations,
            search.successor_distance_candidates_m,
            search.successor_heading_candidates_deg,
        ),
        control=_fit(
            control_observations,
            search.control_endpoint_distance_candidates_px,
            search.control_heading_candidates_deg,
        ),
        observation_sha256=_sha256(observations),
    )


def _heading_deg(vector: npt.NDArray[np.float64]) -> float:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise IndependentLinkerError("cannot compute heading from a degenerate segment")
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _angle_difference_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def successor_rule_features(
    source: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> tuple[float, float]:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise IndependentLinkerError("lane centerlines must have shape [points, 3]")
    if (
        len(source) < 2
        or len(target) < 2
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise IndependentLinkerError("lane centerlines are nonfinite or too short")
    distance = float(np.linalg.norm(source[-1] - target[0]))
    heading = _angle_difference_deg(
        _heading_deg(source[-1, :2] - source[-2, :2]),
        _heading_deg(target[1, :2] - target[0, :2]),
    )
    return distance, heading


def successor_edges(
    centerlines: tuple[npt.NDArray[np.float64], ...], threshold: FittedThreshold
) -> tuple[tuple[int, int], ...]:
    result = []
    for source_index, source in enumerate(centerlines):
        for target_index, target in enumerate(centerlines):
            if source_index == target_index:
                continue
            distance, heading = successor_rule_features(source, target)
            if distance <= threshold.distance and heading <= threshold.heading_difference_deg:
                result.append((source_index, target_index))
    return tuple(result)


def _project_visible(
    centerline: npt.NDArray[np.float64],
    intrinsic: npt.NDArray[np.float64],
    t_vehicle_camera: npt.NDArray[np.float64],
    width: int,
    height: int,
) -> npt.NDArray[np.float64]:
    if intrinsic.shape != (3, 3) or t_vehicle_camera.shape != (4, 4):
        raise IndependentLinkerError("control-rule calibration shapes are invalid")
    if (
        width <= 1
        or height <= 1
        or not all(np.isfinite(value).all() for value in (centerline, intrinsic, t_vehicle_camera))
    ):
        raise IndependentLinkerError("control-rule geometry is invalid")
    rotation = t_vehicle_camera[:3, :3]
    translation = t_vehicle_camera[:3, 3]
    camera = (centerline - translation) @ rotation
    pixels_h = camera @ intrinsic.T
    valid_depth = camera[:, 2] > 1.0e-6
    pixels = pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1.0e-6)
    visible = valid_depth
    visible &= (pixels[:, 0] >= 0.0) & (pixels[:, 0] <= width - 1.0)
    visible &= (pixels[:, 1] >= 0.0) & (pixels[:, 1] <= height - 1.0)
    return cast(npt.NDArray[np.float64], pixels[visible])


def control_rule_features(
    centerline: npt.NDArray[np.float64],
    box: tuple[float, float, float, float],
    intrinsic: npt.NDArray[np.float64],
    t_vehicle_camera: npt.NDArray[np.float64],
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """Measure last-visible lane point to box and image-up heading difference."""
    visible = _project_visible(centerline, intrinsic, t_vehicle_camera, width, height)
    if len(visible) < 2:
        return None
    x_min, y_min, x_max, y_max = box
    if not all(math.isfinite(value) for value in box) or not 0.0 <= x_min <= x_max <= width:
        raise IndependentLinkerError("control box is invalid")
    if not 0.0 <= y_min <= y_max <= height:
        raise IndependentLinkerError("control box is invalid")
    endpoint = visible[-1]
    closest = np.asarray(
        (np.clip(endpoint[0], x_min, x_max), np.clip(endpoint[1], y_min, y_max)),
        dtype=np.float64,
    )
    distance = float(np.linalg.norm(endpoint - closest))
    heading = _angle_difference_deg(_heading_deg(visible[-1] - visible[-2]), -90.0)
    return distance, heading


def control_edges(
    centerlines: tuple[npt.NDArray[np.float64], ...],
    controls: tuple[
        tuple[
            tuple[float, float, float, float],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
            int,
        ],
        ...,
    ],
    threshold: FittedThreshold,
) -> tuple[tuple[int, int], ...]:
    """Return canonical control-major, lane-minor edge index pairs."""
    result = []
    for control_index, (box, intrinsic, transform, width, height) in enumerate(controls):
        for lane_index, centerline in enumerate(centerlines):
            features = control_rule_features(centerline, box, intrinsic, transform, width, height)
            if features is None:
                continue
            distance, heading = features
            if distance <= threshold.distance and heading <= threshold.heading_difference_deg:
                result.append((control_index, lane_index))
    return tuple(result)


__all__ = [
    "FittedThreshold",
    "IndependentLinkerArtifact",
    "IndependentLinkerError",
    "RuleObservation",
    "control_edges",
    "control_rule_features",
    "fit_independent_linker",
    "successor_edges",
    "successor_rule_features",
]

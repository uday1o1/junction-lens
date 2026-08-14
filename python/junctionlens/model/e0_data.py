"""Licensed-data preparation with fail-closed partition isolation for E0."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from junctionlens.data.contracts import AdaptedFrame
from junctionlens.data.manifests import (
    audit_split_manifest,
    load_split_manifest,
    load_split_policy,
)
from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.model.e0_losses import E0Targets
from junctionlens.model.e0_profile import E0Profile
from junctionlens.model.independent_linker import (
    RuleObservation,
    control_rule_features,
    successor_rule_features,
)


class E0DataError(ValueError):
    """Raised when a dataset or partition could contaminate E0 training."""


@dataclass(frozen=True, slots=True)
class E0Inputs:
    images: Tensor
    camera_valid: Tensor
    intrinsics: Tensor
    t_vehicle_camera: Tensor
    ego_motion_previous_to_current: Tensor
    temporal_valid: Tensor

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.images,
            self.camera_valid,
            self.intrinsics,
            self.t_vehicle_camera,
            self.ego_motion_previous_to_current,
            self.temporal_valid,
        )


@dataclass(frozen=True, slots=True)
class PartitionIsolation:
    split_manifest_sha256: str
    partition: str
    segment_ids: frozenset[str]
    forbidden_segment_ids: frozenset[str]
    source_dataset_manifest_sha256: str
    source_frame_manifest_sha256: str
    source_frame_records_sha256: str

    def require_segment(self, segment_id: str) -> None:
        if segment_id not in self.segment_ids or segment_id in self.forbidden_segment_ids:
            raise E0DataError(f"segment {segment_id} is not authorized for {self.partition}")


def load_partition_isolation(
    split_manifest_path: Path,
    policy_path: Path,
    *,
    partition: str,
    statistics: bool,
) -> PartitionIsolation:
    manifest_bytes = split_manifest_path.read_bytes()
    manifest = load_split_manifest(split_manifest_path)
    policy = load_split_policy(policy_path)
    audit_split_manifest(manifest, policy)
    if partition not in {"model_training", "model_selection"}:
        raise E0DataError("E0 may read only model_training or model_selection")
    if statistics and partition != "model_training":
        raise E0DataError("training statistics may read only model_training")
    raw_partitions = cast(Mapping[str, Any], manifest["partitions"])
    selected = cast(Mapping[str, Any], raw_partitions[partition])
    segment_ids = frozenset(cast(list[str], selected["segments"]))
    forbidden = frozenset(
        segment
        for name, raw in raw_partitions.items()
        if name != partition
        for segment in cast(list[str], cast(Mapping[str, Any], raw)["segments"])
    )
    if segment_ids & forbidden:
        raise E0DataError("partition isolation contains segment overlap")
    return PartitionIsolation(
        split_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        partition=partition,
        segment_ids=segment_ids,
        forbidden_segment_ids=forbidden,
        source_dataset_manifest_sha256=str(manifest["source_dataset_manifest_sha256"]),
        source_frame_manifest_sha256=str(manifest["source_frame_manifest_sha256"]),
        source_frame_records_sha256=str(manifest["source_frame_records_sha256"]),
    )


def iter_partition_frames(
    adapter: OpenLaneAdapter, isolation: PartitionIsolation
) -> Iterator[AdaptedFrame]:
    observed_segments: set[str] = set()
    for frame in adapter.iter_frames("full"):
        if frame.key.split_id != "train" or frame.key.segment_id not in isolation.segment_ids:
            continue
        isolation.require_segment(frame.key.segment_id)
        if not frame.annotations_valid:
            raise E0DataError("an authorized E0 frame has no annotations")
        observed_segments.add(frame.key.segment_id)
        yield frame
    if observed_segments != set(isolation.segment_ids):
        missing = sorted(set(isolation.segment_ids) - observed_segments)
        raise E0DataError(f"dataset is missing {len(missing)} authorized segment(s)")


def _resample_curve(points: tuple[tuple[float, float, float], ...], count: int) -> Tensor:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2 or not np.isfinite(values).all():
        raise E0DataError("curve target is invalid")
    distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(distances)))
    if cumulative[-1] <= 1.0e-9:
        raise E0DataError("curve target is degenerate")
    keep = np.concatenate((np.asarray([True]), np.diff(cumulative) > 1.0e-9))
    cumulative = cumulative[keep]
    values = values[keep]
    samples = np.linspace(0.0, cumulative[-1], count)
    result = np.stack(
        [np.interp(samples, cumulative, values[:, coordinate]) for coordinate in range(3)],
        axis=1,
    )
    return torch.from_numpy(result.astype(np.float32))


def _area_target(
    points: tuple[tuple[float, float, float], ...], count: int
) -> tuple[Tensor, Tensor]:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3 or not np.isfinite(values).all():
        raise E0DataError("area target is invalid")
    output = np.zeros((count, 3), dtype=np.float32)
    valid = np.zeros(count, dtype=np.bool_)
    if len(values) <= count:
        output[: len(values)] = values
        output[len(values) :] = values[-1]
        valid[: len(values)] = True
    else:
        indices = np.linspace(0, len(values) - 1, count)
        for coordinate in range(3):
            output[:, coordinate] = np.interp(
                indices, np.arange(len(values)), values[:, coordinate]
            )
        valid[:] = True
    return torch.from_numpy(output), torch.from_numpy(valid)


def frame_targets(frame: AdaptedFrame, profile: E0Profile) -> E0Targets:
    architecture = profile.architecture
    areas = [_area_target(item.points, architecture.area_points) for item in frame.road_areas]
    target = E0Targets(
        lane_centerline=torch.stack(
            [_resample_curve(item.centerline, architecture.lane_points) for item in frame.lanes]
        )
        if frame.lanes
        else torch.empty(0, architecture.lane_points, 3),
        lane_left_boundary=torch.stack(
            [_resample_curve(item.left_boundary, architecture.lane_points) for item in frame.lanes]
        )
        if frame.lanes
        else torch.empty(0, architecture.lane_points, 3),
        lane_right_boundary=torch.stack(
            [_resample_curve(item.right_boundary, architecture.lane_points) for item in frame.lanes]
        )
        if frame.lanes
        else torch.empty(0, architecture.lane_points, 3),
        lane_left_type=torch.tensor(
            [item.left_boundary_type for item in frame.lanes], dtype=torch.int64
        ),
        lane_right_type=torch.tensor(
            [item.right_boundary_type for item in frame.lanes], dtype=torch.int64
        ),
        lane_connector=torch.tensor(
            [item.is_intersection_or_connector for item in frame.lanes], dtype=torch.bool
        ),
        traffic_boxes=torch.tensor(
            [item.normalized_half_open_box for item in frame.traffic_controls], dtype=torch.float32
        ).reshape(-1, 4),
        traffic_category=torch.tensor(
            [item.category - 1 for item in frame.traffic_controls], dtype=torch.int64
        ),
        traffic_attribute=torch.tensor(
            [item.attribute for item in frame.traffic_controls], dtype=torch.int64
        ),
        area_points=torch.stack([item[0] for item in areas])
        if areas
        else torch.empty(0, architecture.area_points, 3),
        area_valid=torch.stack([item[1] for item in areas])
        if areas
        else torch.empty(0, architecture.area_points, dtype=torch.bool),
        area_category=torch.tensor(
            [item.category - 1 for item in frame.road_areas], dtype=torch.int64
        ),
    )
    target.validate()
    if len(frame.lanes) > architecture.lane_queries:
        raise E0DataError("lane labels exceed frozen query capacity")
    if len(frame.traffic_controls) > architecture.traffic_queries:
        raise E0DataError("traffic-control labels exceed frozen query capacity")
    if len(frame.road_areas) > architecture.area_queries:
        raise E0DataError("road-area labels exceed frozen query capacity")
    return target


def frame_inputs(adapter: OpenLaneAdapter, frame: AdaptedFrame, profile: E0Profile) -> E0Inputs:
    current = adapter.model_camera_inputs(frame)
    images = torch.zeros(
        1,
        profile.input.timestamps,
        profile.input.cameras,
        profile.input.channels,
        profile.input.height,
        profile.input.width,
    )
    valid = torch.zeros(1, profile.input.timestamps, profile.input.cameras, dtype=torch.bool)
    intrinsics = torch.zeros(1, profile.input.timestamps, profile.input.cameras, 3, 3)
    transforms = torch.zeros(1, profile.input.timestamps, profile.input.cameras, 4, 4)
    current_index = profile.input.current_timestamp_index
    images[0, current_index] = torch.from_numpy(current.images.copy())
    valid[0, current_index] = torch.from_numpy(current.camera_valid.copy())
    intrinsics[0, current_index] = torch.from_numpy(current.intrinsics.copy())
    transforms[0, current_index] = torch.from_numpy(current.t_vehicle_camera.copy())
    return E0Inputs(
        images,
        valid,
        intrinsics,
        transforms,
        torch.eye(4).reshape(1, 4, 4),
        torch.zeros(1, dtype=torch.bool),
    )


def frame_rule_observations(
    frame: AdaptedFrame,
) -> tuple[tuple[RuleObservation, ...], tuple[RuleObservation, ...]]:
    lanes = tuple(np.asarray(item.centerline, dtype=np.float64) for item in frame.lanes)
    successor = []
    for source, target in np.ndindex((len(lanes), len(lanes))):
        if source == target:
            continue
        distance, heading = successor_rule_features(lanes[source], lanes[target])
        successor.append(
            RuleObservation(
                f"{frame.key.segment_id}:{frame.key.timestamp_ns}:ll:{source}:{target}",
                distance,
                heading,
                frame.topology_lane_lane[source][target],
            )
        )
    camera_by_slot = {camera.slot: camera for camera in frame.cameras}
    control = []
    for control_index, element in enumerate(frame.traffic_controls):
        camera = camera_by_slot[element.source_camera]
        if not camera.valid:
            raise E0DataError("annotated control references an invalid camera")
        normalized = element.normalized_half_open_box
        box = (
            normalized[0] * camera.original_width,
            normalized[1] * camera.original_height,
            normalized[2] * camera.original_width,
            normalized[3] * camera.original_height,
        )
        for lane_index, lane in enumerate(lanes):
            features = control_rule_features(
                lane,
                box,
                np.asarray(camera.intrinsic, dtype=np.float64),
                np.asarray(camera.t_vehicle_camera, dtype=np.float64),
                camera.original_width,
                camera.original_height,
            )
            if features is None:
                continue
            distance, heading = features
            control.append(
                RuleObservation(
                    f"{frame.key.segment_id}:{frame.key.timestamp_ns}:lt:{control_index}:{lane_index}",
                    distance,
                    heading,
                    frame.topology_lane_traffic[lane_index][control_index],
                )
            )
    return tuple(successor), tuple(control)


__all__ = [
    "E0DataError",
    "E0Inputs",
    "PartitionIsolation",
    "frame_inputs",
    "frame_rule_observations",
    "frame_targets",
    "iter_partition_frames",
    "load_partition_isolation",
]

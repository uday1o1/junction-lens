"""Capacity and source-identity audits for normalized OpenLane frames."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from junctionlens.data.contracts import AdaptedFrame, Point3
from junctionlens.data.geometry import FloatArray, transform_points


@dataclass(frozen=True, slots=True)
class CapacityTypeAudit:
    """One frozen-query capacity result."""

    capacity: int
    frame_count: int
    maximum_count: int
    exceeding_frame_count: int
    coverage: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class IdentityTypeAudit:
    """Per-node-type temporal source-ID evidence."""

    object_type: str
    object_count: int
    unique_source_id_count: int
    persistent_source_id_count: int
    within_frame_collision_count: int
    signature_reuse_count: int
    continuity_violation_count: int
    maximum_continuity_delta: float
    temporal_kpi_state: str


def audit_capacities(
    frames: Sequence[AdaptedFrame],
    capacities: Mapping[str, int],
    *,
    required_coverage: float,
) -> Mapping[str, CapacityTypeAudit]:
    """Measure complete per-frame label distributions without clipping."""
    if not frames:
        raise ValueError("capacity audit requires at least one frame")
    counts = {
        "lane_segment": [len(frame.lanes) for frame in frames],
        "traffic_element": [len(frame.traffic_controls) for frame in frames],
        "area": [len(frame.road_areas) for frame in frames],
    }
    result: dict[str, CapacityTypeAudit] = {}
    for object_type, values in counts.items():
        capacity = capacities[object_type]
        exceeding = sum(value > capacity for value in values)
        coverage = 1.0 - exceeding / len(values)
        result[object_type] = CapacityTypeAudit(
            capacity=capacity,
            frame_count=len(values),
            maximum_count=max(values),
            exceeding_frame_count=exceeding,
            coverage=coverage,
            accepted=coverage >= required_coverage,
        )
    return result


def _world_centroid(frame: AdaptedFrame, points: Iterable[Point3]) -> FloatArray:
    point_array = np.asarray(tuple(points), dtype=np.float64)
    transformed = transform_points(frame.t_world_vehicle, point_array)
    return np.asarray(transformed.mean(axis=0), dtype=np.float64)


def _identity_records(
    frames: Sequence[AdaptedFrame], object_type: str
) -> Iterable[tuple[AdaptedFrame, str, tuple[int, ...], FloatArray]]:
    for frame in frames:
        yield from _frame_identity_records(frame, object_type)


def _frame_identity_records(
    frame: AdaptedFrame, object_type: str
) -> Iterable[tuple[AdaptedFrame, str, tuple[int, ...], FloatArray]]:
    if object_type == "lane_segment":
        for lane in frame.lanes:
            yield (
                frame,
                lane.source_object_id,
                (lane.left_boundary_type, lane.right_boundary_type),
                _world_centroid(frame, lane.centerline),
            )
    elif object_type == "traffic_element":
        for control in frame.traffic_controls:
            box = control.normalized_half_open_box
            yield (
                frame,
                control.source_object_id,
                (control.category, control.attribute),
                np.asarray(
                    [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0],
                    dtype=np.float64,
                ),
            )
    elif object_type == "area":
        for area in frame.road_areas:
            yield (
                frame,
                area.source_object_id,
                (area.category,),
                _world_centroid(frame, area.points),
            )
    else:
        raise ValueError(f"unsupported identity type: {object_type}")


def audit_identities(
    frames: Sequence[AdaptedFrame],
    continuity_thresholds: Mapping[str, float],
) -> Mapping[str, IdentityTypeAudit]:
    """Audit collisions, signature reuse, persistence, and geometric continuity by type."""
    result: dict[str, IdentityTypeAudit] = {}
    for object_type in ("lane_segment", "traffic_element", "area"):
        records = list(_identity_records(frames, object_type))
        occurrences: dict[tuple[str, str, str], list[tuple[int, tuple[int, ...], FloatArray]]] = (
            defaultdict(list)
        )
        within_frame_counts: dict[tuple[str, str, int, str], int] = defaultdict(int)
        for frame, source_id, signature, centroid in records:
            identity = (frame.key.split_id, frame.key.segment_id, source_id)
            occurrences[identity].append((frame.key.timestamp_ns, signature, centroid))
            within_frame_counts[
                (frame.key.split_id, frame.key.segment_id, frame.key.timestamp_ns, source_id)
            ] += 1
        collisions = sum(count - 1 for count in within_frame_counts.values() if count > 1)
        persistent = 0
        signature_reuse = 0
        continuity_violations = 0
        maximum_delta = 0.0
        threshold = continuity_thresholds[object_type]
        for history in occurrences.values():
            ordered = sorted(history, key=lambda item: item[0])
            if len(ordered) > 1:
                persistent += 1
            if len({signature for _, signature, _ in ordered}) > 1:
                signature_reuse += 1
            for previous, current in pairwise(ordered):
                delta = float(np.linalg.norm(current[2] - previous[2]))
                if not math.isfinite(delta):
                    continuity_violations += 1
                    continue
                maximum_delta = max(maximum_delta, delta)
                if delta > threshold:
                    continuity_violations += 1
        if collisions or signature_reuse:
            state = "DISABLED_IDENTITY_CONTRACT_FAILURE"
        elif continuity_violations:
            state = "DISABLED_GEOMETRIC_DISCONTINUITY"
        elif persistent == 0:
            state = "DISABLED_NO_PERSISTENT_SOURCE_IDS"
        else:
            state = "ENABLED_SOURCE_IDENTITY"
        result[object_type] = IdentityTypeAudit(
            object_type=object_type,
            object_count=len(records),
            unique_source_id_count=len(occurrences),
            persistent_source_id_count=persistent,
            within_frame_collision_count=collisions,
            signature_reuse_count=signature_reuse,
            continuity_violation_count=continuity_violations,
            maximum_continuity_delta=maximum_delta,
            temporal_kpi_state=state,
        )
    return result


def audit_report(
    frames: Iterable[AdaptedFrame],
    capacities: Mapping[str, int],
    continuity_thresholds: Mapping[str, float],
    *,
    required_coverage: float,
) -> Mapping[str, Any]:
    """Create one deterministic report in one pass without retaining full frames."""
    capacity_counts = {
        object_type: {"maximum": 0, "exceeding": 0}
        for object_type in ("lane_segment", "traffic_element", "area")
    }
    identity_states = {
        object_type: _StreamingIdentityState(
            object_type=object_type,
            continuity_threshold=continuity_thresholds[object_type],
        )
        for object_type in ("lane_segment", "traffic_element", "area")
    }
    frame_count = 0
    previous_frame_key: tuple[str, str, int] | None = None
    for frame in frames:
        frame_key = (frame.key.split_id, frame.key.segment_id, frame.key.timestamp_ns)
        if previous_frame_key is not None and frame_key <= previous_frame_key:
            raise ValueError("capacity audit frames must be in strict source order")
        previous_frame_key = frame_key
        frame_count += 1
        counts = {
            "lane_segment": len(frame.lanes),
            "traffic_element": len(frame.traffic_controls),
            "area": len(frame.road_areas),
        }
        for object_type, count in counts.items():
            state = capacity_counts[object_type]
            state["maximum"] = max(state["maximum"], count)
            state["exceeding"] += int(count > capacities[object_type])
            identity_states[object_type].consume(frame)
    if frame_count == 0:
        raise ValueError("capacity audit requires at least one frame")
    capacity = {
        object_type: CapacityTypeAudit(
            capacity=capacities[object_type],
            frame_count=frame_count,
            maximum_count=state["maximum"],
            exceeding_frame_count=state["exceeding"],
            coverage=1.0 - state["exceeding"] / frame_count,
            accepted=(1.0 - state["exceeding"] / frame_count) >= required_coverage,
        )
        for object_type, state in capacity_counts.items()
    }
    identities = {object_type: state.result() for object_type, state in identity_states.items()}
    return {
        "schema_version": "1.0.0",
        "frame_count": frame_count,
        "capacity": {name: asdict(value) for name, value in capacity.items()},
        "identity": {name: asdict(value) for name, value in identities.items()},
        "capacity_gate_accepted": all(value.accepted for value in capacity.values()),
    }


@dataclass(slots=True)
class _IdentityHistory:
    occurrence_count: int
    last_signature: tuple[int, ...]
    last_centroid: FloatArray
    signature_reuse_recorded: bool = False


@dataclass(slots=True)
class _StreamingIdentityState:
    object_type: str
    continuity_threshold: float
    object_count: int = 0
    unique_source_id_count: int = 0
    persistent_source_id_count: int = 0
    within_frame_collision_count: int = 0
    signature_reuse_count: int = 0
    continuity_violation_count: int = 0
    maximum_continuity_delta: float = 0.0
    current_group: tuple[str, str] | None = None
    history: dict[str, _IdentityHistory] | None = None

    def consume(self, frame: AdaptedFrame) -> None:
        """Consume one source-ordered frame and retain only its segment history."""
        group = (frame.key.split_id, frame.key.segment_id)
        if self.current_group != group:
            self.current_group = group
            self.history = {}
        if self.history is None:
            raise AssertionError("identity history was not initialized")
        frame_counts: dict[str, int] = defaultdict(int)
        for _, source_id, signature, centroid in _frame_identity_records(frame, self.object_type):
            self.object_count += 1
            frame_counts[source_id] += 1
            previous = self.history.get(source_id)
            if previous is None:
                self.unique_source_id_count += 1
                self.history[source_id] = _IdentityHistory(1, signature, centroid)
                continue
            if previous.occurrence_count == 1:
                self.persistent_source_id_count += 1
            previous.occurrence_count += 1
            if signature != previous.last_signature and not previous.signature_reuse_recorded:
                self.signature_reuse_count += 1
                previous.signature_reuse_recorded = True
            delta = float(np.linalg.norm(centroid - previous.last_centroid))
            if not math.isfinite(delta) or delta > self.continuity_threshold:
                self.continuity_violation_count += 1
            if math.isfinite(delta):
                self.maximum_continuity_delta = max(self.maximum_continuity_delta, delta)
            previous.last_signature = signature
            previous.last_centroid = centroid
        self.within_frame_collision_count += sum(
            count - 1 for count in frame_counts.values() if count > 1
        )

    def result(self) -> IdentityTypeAudit:
        """Freeze accumulated identity evidence and its temporal-KPI state."""
        if self.within_frame_collision_count or self.signature_reuse_count:
            state = "DISABLED_IDENTITY_CONTRACT_FAILURE"
        elif self.continuity_violation_count:
            state = "DISABLED_GEOMETRIC_DISCONTINUITY"
        elif self.persistent_source_id_count == 0:
            state = "DISABLED_NO_PERSISTENT_SOURCE_IDS"
        else:
            state = "ENABLED_SOURCE_IDENTITY"
        return IdentityTypeAudit(
            object_type=self.object_type,
            object_count=self.object_count,
            unique_source_id_count=self.unique_source_id_count,
            persistent_source_id_count=self.persistent_source_id_count,
            within_frame_collision_count=self.within_frame_collision_count,
            signature_reuse_count=self.signature_reuse_count,
            continuity_violation_count=self.continuity_violation_count,
            maximum_continuity_delta=self.maximum_continuity_delta,
            temporal_kpi_state=state,
        )

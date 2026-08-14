"""Independent invariant checks for transformed fault-lab bundles."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from typing import Any

import numpy as np
from google.protobuf.message import Message

from junctionlens.contract.validation import ContractViolation, validate_envelope
from junctionlens.faults.models import FAULT_REASON_CODES, FaultKind, PredictionBundle
from junctionlens.faults.transforms import decode_envelopes
from junctionlens.v1 import scene_control_graph_pb2 as scg


class FaultAnalysisError(RuntimeError):
    """Raised when a parent and child cannot form an exact fault comparison."""


def _frame_maps(
    parent: PredictionBundle, child: PredictionBundle
) -> tuple[
    dict[str, scg.SceneControlGraphEnvelope],
    dict[str, scg.SceneControlGraphEnvelope],
]:
    parent_frames = dict(decode_envelopes(parent, validate=True))
    child_frames = dict(decode_envelopes(child, validate=False))
    if set(parent_frames) != set(child_frames):
        raise FaultAnalysisError("fault transformation changed the frame-token set")
    return parent_frames, child_frames


def _message_bytes(message: object) -> bytes:
    return message.SerializeToString(deterministic=True)  # type: ignore[attr-defined, no-any-return]


def _replace_sorted(container: Any, values: Sequence[Message]) -> None:
    copies = []
    for value in values:
        clone = value.__class__()
        clone.CopyFrom(value)
        copies.append(clone)
    del container[:]
    container.extend(copies)


def _normalized_graph_bytes(envelope: scg.SceneControlGraphEnvelope) -> bytes:
    clone = deepcopy(envelope)
    graph = clone.graph
    for collection, key in (
        (graph.lanes, lambda item: item.node_id),
        (graph.traffic_controls, lambda item: item.node_id),
        (graph.road_areas, lambda item: item.node_id),
        (graph.edges, lambda item: item.edge_id),
        (graph.tracks, lambda item: item.track_id),
    ):
        _replace_sorted(collection, sorted(collection, key=key))
    return _message_bytes(clone)


def _node_signature(envelope: scg.SceneControlGraphEnvelope) -> str:
    digest = hashlib.sha256()
    for label, nodes in (
        (b"lane", envelope.graph.lanes),
        (b"control", envelope.graph.traffic_controls),
        (b"area", envelope.graph.road_areas),
    ):
        for node in sorted(nodes, key=lambda item: item.node_id):
            digest.update(label)
            digest.update(_message_bytes(node))
    return digest.hexdigest()


def _node_map(envelope: scg.SceneControlGraphEnvelope) -> dict[tuple[str, int], bytes]:
    result = {}
    for label, nodes in (
        ("lane", envelope.graph.lanes),
        ("control", envelope.graph.traffic_controls),
        ("area", envelope.graph.road_areas),
    ):
        result.update({(label, node.node_id): _message_bytes(node) for node in nodes})
    return result


def _edges(envelope: scg.SceneControlGraphEnvelope, edge_type: int) -> set[tuple[int, int]]:
    return {
        (edge.source_node_id, edge.target_node_id)
        for edge in envelope.graph.edges
        if edge.edge_type == edge_type and edge.binary_decision
    }


def _all_valid(frames: Mapping[str, scg.SceneControlGraphEnvelope]) -> bool:
    try:
        for envelope in frames.values():
            validate_envelope(envelope)
    except ContractViolation:
        return False
    return True


def _first_violation(
    frames: Mapping[str, scg.SceneControlGraphEnvelope],
) -> ContractViolation | None:
    for token in sorted(frames):
        try:
            validate_envelope(frames[token])
        except ContractViolation as error:
            return error
    return None


def _find_changed_pair(
    parent_frames: Mapping[str, scg.SceneControlGraphEnvelope],
    child_frames: Mapping[str, scg.SceneControlGraphEnvelope],
    predicate: Callable[[scg.SceneControlGraphEnvelope, scg.SceneControlGraphEnvelope], bool],
) -> bool:
    return any(predicate(parent_frames[token], child_frames[token]) for token in parent_frames)


def _control_stats(
    parent_frames: Mapping[str, scg.SceneControlGraphEnvelope],
    child_frames: Mapping[str, scg.SceneControlGraphEnvelope],
) -> tuple[int, int, int]:
    expected = 0
    retained = 0
    predicted = 0
    for token, parent in parent_frames.items():
        parent_edges = _edges(parent, scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE)
        child_edges = _edges(child_frames[token], scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE)
        expected += len(parent_edges)
        retained += len(parent_edges & child_edges)
        predicted += len(child_edges)
    return expected, retained, predicted


def _successor_stats(
    parent_frames: Mapping[str, scg.SceneControlGraphEnvelope],
    child_frames: Mapping[str, scg.SceneControlGraphEnvelope],
) -> tuple[set[tuple[str, int, int]], set[tuple[str, int, int]]]:
    parent = {
        (token, source, target)
        for token, envelope in parent_frames.items()
        for source, target in _edges(envelope, scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR)
    }
    child = {
        (token, source, target)
        for token, envelope in child_frames.items()
        for source, target in _edges(envelope, scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR)
    }
    return parent, child


def _lane_points(
    envelope: scg.SceneControlGraphEnvelope,
) -> dict[tuple[int, str, int], tuple[float, ...]]:
    result: dict[tuple[int, str, int], tuple[float, ...]] = {}
    for lane in envelope.graph.lanes:
        for name in ("centerline", "left_boundary", "right_boundary"):
            polyline = getattr(lane, name)
            for index, point in enumerate(polyline.points):
                result[(lane.node_id, name, index)] = (point.x, point.y, point.z)
    return result


def _extrinsics(envelope: scg.SceneControlGraphEnvelope) -> tuple[tuple[float, ...], ...]:
    if not envelope.graph.HasField("sensor_frame"):
        return ()
    return tuple(
        tuple(camera.t_vehicle_camera.values) for camera in envelope.graph.sensor_frame.cameras
    )


def _uncertainties(envelope: scg.SceneControlGraphEnvelope) -> list[float]:
    result: list[float] = []
    for lane in envelope.graph.lanes:
        scales = list(lane.centerline_laplace_scale_m)
        for polyline in (lane.centerline, lane.left_boundary, lane.right_boundary):
            scales.extend(polyline.point_uncertainty)
        result.extend(value for scale in scales for value in (scale.x, scale.y, scale.z))
    for area in envelope.graph.road_areas:
        scales = [*area.geometry_uncertainty, *area.geometry.point_uncertainty]
        result.extend(value for scale in scales for value in (scale.x, scale.y, scale.z))
    return result


def _probabilities(envelope: scg.SceneControlGraphEnvelope) -> dict[str, float]:
    values: dict[str, float] = {}
    for lane in envelope.graph.lanes:
        prefix = f"lane:{lane.node_id}"
        values[f"{prefix}:existence"] = lane.existence_confidence
        values[f"{prefix}:intersection"] = lane.intersection_or_connector_probability
    for control in envelope.graph.traffic_controls:
        prefix = f"control:{control.node_id}"
        values[f"{prefix}:existence"] = control.existence_confidence
        values[f"{prefix}:class"] = control.calibrated_class_confidence
        values[f"{prefix}:attribute"] = control.calibrated_attribute_confidence
    for area in envelope.graph.road_areas:
        values[f"area:{area.node_id}:existence"] = area.existence_confidence
    for edge in envelope.graph.edges:
        values[f"edge:{edge.edge_id}:raw"] = edge.raw_probability
        values[f"edge:{edge.edge_id}:calibrated"] = edge.calibrated_probability
    return values


def _strict_order_preserved(parent: Mapping[str, float], child: Mapping[str, float]) -> bool:
    keys = sorted(parent)
    for first_index, first in enumerate(keys):
        for second in keys[first_index + 1 :]:
            parent_delta = parent[first] - parent[second]
            child_delta = child[first] - child[second]
            if parent_delta > 0.0 and child_delta <= 0.0:
                return False
            if parent_delta < 0.0 and child_delta >= 0.0:
                return False
    return True


def _transition_count(
    frames: Mapping[str, scg.SceneControlGraphEnvelope],
    extractor: Callable[[scg.SceneControlGraphEnvelope], set[object]],
) -> int:
    groups: dict[str, list[scg.SceneControlGraphEnvelope]] = defaultdict(list)
    for envelope in frames.values():
        groups[envelope.graph.frame_key.segment_id].append(envelope)
    changes = 0
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item.graph.frame_key.timestamp_ns)
        for first, second in pairwise(ordered):
            changes += int(extractor(first) != extractor(second))
    return changes


def _active_edges(envelope: scg.SceneControlGraphEnvelope) -> set[object]:
    return {
        (edge.edge_type, edge.source_node_id, edge.target_node_id)
        for edge in envelope.graph.edges
        if edge.binary_decision
    }


def _lane_presence(envelope: scg.SceneControlGraphEnvelope) -> set[object]:
    return {lane.node_id for lane in envelope.graph.lanes}


def _track_reuse_count(frames: Mapping[str, scg.SceneControlGraphEnvelope]) -> int:
    groups: dict[str, list[scg.SceneControlGraphEnvelope]] = defaultdict(list)
    for envelope in frames.values():
        groups[envelope.graph.frame_key.segment_id].append(envelope)
    count = 0
    for group in groups.values():
        terminated: set[int] = set()
        for envelope in sorted(group, key=lambda item: item.graph.frame_key.timestamp_ns):
            for track in envelope.graph.tracks:
                if (
                    track.track_id in terminated
                    and track.termination_reason == scg.TRACK_TERMINATION_REASON_ACTIVE
                ):
                    count += 1
                if track.termination_reason not in (
                    scg.TRACK_TERMINATION_REASON_ACTIVE,
                    scg.TRACK_TERMINATION_REASON_SEGMENT_END,
                ):
                    terminated.add(track.track_id)
    return count


def _p95(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95, method="linear"))


def verify_clean_bundle(bundle: PredictionBundle) -> Mapping[str, object]:
    """Verify the nearby clean control before any seeded transformation."""
    frames = dict(decode_envelopes(bundle, validate=False))
    checks = {
        "contract_valid": _all_valid(frames),
        "gpu_provider_only": bundle.runtime.provider_node_counts.get("CPUExecutionProvider", 0)
        == 0,
        "postprocess_budget": _p95(bundle.runtime.postprocess_latency_ms) <= 100.0,
        "bounded_device_memory": max(bundle.runtime.device_memory_bytes)
        - min(bundle.runtime.device_memory_bytes)
        <= 1024 * 1024,
        "no_fault_history": not bundle.fault_history,
    }
    return {
        "schema_version": "junctionlens.fault-control.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "reason_code": "FAULT_CONTROL_CLEAN",
        "checks": checks,
    }


def analyze_fault(parent: PredictionBundle, child: PredictionBundle) -> Mapping[str, object]:
    """Detect a declared transform from actual parent-child invariant changes."""
    if len(child.fault_history) != 1:
        raise FaultAnalysisError("derived bundle must contain exactly one fault declaration")
    declaration = child.fault_history[0]
    kind = declaration.kind
    expected_reason = FAULT_REASON_CODES[kind]
    parent_frames, child_frames = _frame_maps(parent, child)
    checks: dict[str, bool] = {"frame_set_preserved": True}
    details: dict[str, Any] = {}
    violation = _first_violation(child_frames)
    valid_faults = {
        FaultKind.DUPLICATE_NODE_ID,
        FaultKind.DANGLING_EDGE,
        FaultKind.ZERO_UNCERTAINTY,
        FaultKind.INJECT_NAN,
    }
    if kind in valid_faults:
        checks["expected_contract_rejection"] = (
            violation is not None and violation.reason_code == expected_reason
        )
        details["contract_reason_code"] = None if violation is None else violation.reason_code
    else:
        checks["contract_valid"] = violation is None

    node_signatures_match = all(
        _node_signature(parent_frames[token]) == _node_signature(child_frames[token])
        for token in parent_frames
    )
    if kind == FaultKind.SWAP_CONTROL_EDGES:
        expected, retained, predicted = _control_stats(parent_frames, child_frames)
        checks.update(
            {
                "node_tensors_unchanged": node_signatures_match,
                "control_edge_count_preserved": expected == predicted and expected >= 2,
                "control_assignments_changed": retained < expected,
            }
        )
        details.update(
            {
                "DET_l_delta": 0.0 if node_signatures_match else None,
                "DET_t_delta": 0.0 if node_signatures_match else None,
                "node_geometry_unchanged": node_signatures_match,
                "control_edge_recall": retained / expected if expected else None,
                "wrong_control_assignment_rate": (
                    (predicted - retained) / predicted if predicted else None
                ),
                "lane_control_fault_cell": {
                    "cell_id": "overall.control_edge_recall",
                    "status": "FAIL_REGRESSION" if retained < expected else "PASS",
                    "reason_code": "FAULT_CONTROL_ASSIGNMENT_CHANGED",
                    "v1_release_acceptance_run": False,
                },
            }
        )
    elif kind == FaultKind.DROP_CONTROL_EDGES:
        expected, retained, predicted = _control_stats(parent_frames, child_frames)
        checks.update(
            {
                "node_tensors_unchanged": node_signatures_match,
                "control_edges_strictly_dropped": 0 <= predicted < expected,
                "retained_edges_are_parent_edges": retained == predicted,
            }
        )
        details["control_edge_recall"] = retained / expected if expected else None
    elif kind in {FaultKind.DROP_SUCCESSOR_CHAIN, FaultKind.ADD_SPURIOUS_SUCCESSORS}:
        parent_edges, child_edges = _successor_stats(parent_frames, child_frames)
        if kind == FaultKind.DROP_SUCCESSOR_CHAIN:
            checks["successor_strict_subset"] = child_edges < parent_edges
        else:
            checks["successor_strict_superset"] = child_edges > parent_edges
        checks["node_tensors_unchanged"] = node_signatures_match
    elif kind == FaultKind.PERMUTE_NODES_CORRECTLY:
        checks["normalized_graph_invariant"] = all(
            _normalized_graph_bytes(parent_frames[token])
            == _normalized_graph_bytes(child_frames[token])
            for token in parent_frames
        )
        checks["wire_order_changed"] = any(
            _message_bytes(parent_frames[token]) != _message_bytes(child_frames[token])
            for token in parent_frames
        )
    elif kind == FaultKind.PERMUTE_NODES_WITHOUT_EDGES:
        checks["node_identity_population_preserved"] = all(
            set(_node_map(parent_frames[token])) == set(_node_map(child_frames[token]))
            for token in parent_frames
        )
        checks["node_semantics_misaligned"] = any(
            _node_map(parent_frames[token]) != _node_map(child_frames[token])
            for token in parent_frames
        )
        parent_edges, child_edges = _successor_stats(parent_frames, child_frames)
        checks["edge_axes_left_unchanged"] = parent_edges == child_edges
    elif kind == FaultKind.JITTER_LANES:
        checks["lane_geometry_changed"] = _find_changed_pair(
            parent_frames,
            child_frames,
            lambda parent_frame, child_frame: _lane_points(parent_frame)
            != _lane_points(child_frame),
        )
        parent_edges, child_edges = _successor_stats(parent_frames, child_frames)
        checks["topology_unchanged"] = parent_edges == child_edges
    elif kind == FaultKind.FLIP_BOUNDARIES:

        def flipped(
            parent_frame: scg.SceneControlGraphEnvelope,
            child_frame: scg.SceneControlGraphEnvelope,
        ) -> bool:
            child_by_id = {lane.node_id: lane for lane in child_frame.graph.lanes}
            return any(
                lane.node_id in child_by_id
                and _message_bytes(lane.left_boundary)
                == _message_bytes(child_by_id[lane.node_id].right_boundary)
                and _message_bytes(lane.right_boundary)
                == _message_bytes(child_by_id[lane.node_id].left_boundary)
                for lane in parent_frame.graph.lanes
            )

        checks["left_right_boundaries_swapped"] = _find_changed_pair(
            parent_frames, child_frames, flipped
        )
    elif kind == FaultKind.CORRUPT_EXTRINSIC:
        checks["camera_extrinsic_changed"] = _find_changed_pair(
            parent_frames,
            child_frames,
            lambda parent_frame, child_frame: _extrinsics(parent_frame) != _extrinsics(child_frame),
        )
        checks["node_tensors_unchanged"] = node_signatures_match
    elif kind == FaultKind.INFLATE_UNCERTAINTY:
        parent_uncertainties = [
            value for frame in parent_frames.values() for value in _uncertainties(frame)
        ]
        child_uncertainties = [
            value for frame in child_frames.values() for value in _uncertainties(frame)
        ]
        checks["uncertainty_population_preserved"] = (
            len(parent_uncertainties) == len(child_uncertainties) > 0
        )
        checks["uncertainty_strictly_inflated"] = all(
            child_value > parent_value
            for parent_value, child_value in zip(
                parent_uncertainties, child_uncertainties, strict=True
            )
        )
    elif kind == FaultKind.TEMPERATURE_COLLAPSE:
        parent_probabilities = {
            f"{token}:{key}": value
            for token, frame in parent_frames.items()
            for key, value in _probabilities(frame).items()
        }
        child_probabilities = {
            f"{token}:{key}": value
            for token, frame in child_frames.items()
            for key, value in _probabilities(frame).items()
        }
        checks["probability_population_preserved"] = set(parent_probabilities) == set(
            child_probabilities
        )
        checks["predictive_rank_preserved"] = _strict_order_preserved(
            parent_probabilities, child_probabilities
        )
        checks["probabilities_changed"] = any(
            parent_probabilities[key] != child_probabilities[key] for key in parent_probabilities
        )
        details["mean_confidence_extremity_before"] = sum(
            abs(value - 0.5) for value in parent_probabilities.values()
        ) / len(parent_probabilities)
        details["mean_confidence_extremity_after"] = sum(
            abs(value - 0.5) for value in child_probabilities.values()
        ) / len(child_probabilities)
        checks["confidence_extremity_increased"] = (
            details["mean_confidence_extremity_after"] > details["mean_confidence_extremity_before"]
        )
    elif kind == FaultKind.ALTERNATE_EDGE_CONFIDENCE:
        parent_changes = _transition_count(parent_frames, _active_edges)
        child_changes = _transition_count(child_frames, _active_edges)
        checks["edge_flip_count_increased"] = child_changes > parent_changes
        details.update({"parent_edge_flips": parent_changes, "child_edge_flips": child_changes})
    elif kind == FaultKind.ALTERNATE_NODE_PRESENCE:
        parent_changes = _transition_count(parent_frames, _lane_presence)
        child_changes = _transition_count(child_frames, _lane_presence)
        checks["presence_flicker_increased"] = child_changes > parent_changes
        details.update(
            {"parent_presence_changes": parent_changes, "child_presence_changes": child_changes}
        )
    elif kind == FaultKind.REUSE_TRACK_ID:
        parent_reuse = _track_reuse_count(parent_frames)
        child_reuse = _track_reuse_count(child_frames)
        checks["terminated_track_id_reused"] = child_reuse > parent_reuse
        details.update({"parent_track_reuse": parent_reuse, "child_track_reuse": child_reuse})
    elif kind == FaultKind.FORCE_PROVIDER_FALLBACK:
        checks["unexpected_cpu_nodes_added"] = (
            parent.runtime.provider_node_counts.get("CPUExecutionProvider", 0) == 0
            and child.runtime.provider_node_counts.get("CPUExecutionProvider", 0) > 0
        )
    elif kind == FaultKind.DELAY_POSTPROCESS:
        parent_p95 = _p95(parent.runtime.postprocess_latency_ms)
        child_p95 = _p95(child.runtime.postprocess_latency_ms)
        checks["p95_crossed_absolute_budget"] = parent_p95 <= 100.0 < child_p95
        details.update({"parent_p95_latency_ms": parent_p95, "child_p95_latency_ms": child_p95})
    elif kind == FaultKind.LEAK_BUFFER:
        parent_growth = max(parent.runtime.device_memory_bytes) - min(
            parent.runtime.device_memory_bytes
        )
        child_growth = max(child.runtime.device_memory_bytes) - min(
            child.runtime.device_memory_bytes
        )
        checks["unbounded_memory_growth_added"] = parent_growth <= 1024 * 1024 < child_growth
        details.update(
            {"parent_device_growth_bytes": parent_growth, "child_device_growth_bytes": child_growth}
        )

    detected = all(checks.values())
    return {
        "schema_version": "junctionlens.fault-detection.v1",
        "fault_kind": kind.value,
        "status": (
            "CONTROL_PASSED"
            if detected and kind == FaultKind.PERMUTE_NODES_CORRECTLY
            else "DETECTED"
            if detected
            else "FAILED_TO_DETECT"
        ),
        "primary_reason_code": expected_reason if detected else "FAULT_NOT_DETECTED",
        "expected_reason_code": expected_reason,
        "checks": checks,
        "details": details,
        "detection_rate": 1.0 if detected else 0.0,
    }


__all__ = ["FaultAnalysisError", "analyze_fault", "verify_clean_bundle"]

"""Deterministic protobuf and runtime transformations for every V1 fault."""

from __future__ import annotations

import base64
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Any

from google.protobuf.message import Message

from junctionlens.contract.codec import parse_binary
from junctionlens.contract.ids import edge_id
from junctionlens.contract.validation import validate_envelope
from junctionlens.faults.models import (
    FaultDeclaration,
    FaultKind,
    PredictionBundle,
    PredictionFrame,
    RuntimeFixture,
)
from junctionlens.faults.runtime import inject_bounded_latency, run_allocator_fixture
from junctionlens.v1 import scene_control_graph_pb2 as scg


class FaultTransformError(ValueError):
    """Raised when a declared fault has no eligible target in its input bundle."""


def decode_envelopes(
    bundle: PredictionBundle, *, validate: bool
) -> list[tuple[str, scg.SceneControlGraphEnvelope]]:
    result = []
    for frame in bundle.frames:
        payload = base64.b64decode(frame.envelope_pb_base64, validate=True)
        if validate:
            envelope = parse_binary(payload)
        else:
            envelope = scg.SceneControlGraphEnvelope()
            envelope.ParseFromString(payload)
        result.append((frame.frame_token, envelope))
    return result


def _encode_envelopes(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]],
) -> tuple[PredictionFrame, ...]:
    return tuple(
        PredictionFrame(
            frame_token=token,
            envelope_pb_base64=base64.b64encode(
                envelope.SerializeToString(deterministic=True)
            ).decode("ascii"),
        )
        for token, envelope in envelopes
    )


def _replace[MessageT: Message](container: Any, retained: Iterable[MessageT]) -> None:
    copies = []
    for item in retained:
        clone = item.__class__()
        clone.CopyFrom(item)
        copies.append(clone)
    del container[:]
    container.extend(copies)


def _edge_sets(graph: scg.SceneControlGraph, edge_type: int) -> list[scg.GraphEdge]:
    return [edge for edge in graph.edges if edge.edge_type == edge_type]


def _reidentify_edge(graph: scg.SceneControlGraph, edge: scg.GraphEdge) -> None:
    edge.edge_id = edge_id(
        graph.frame_key,
        edge.edge_type,
        edge.source_node_id,
        edge.target_node_id,
    )


def _swap_control_edges(envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]]) -> None:
    for _, envelope in envelopes:
        edges = _edge_sets(envelope.graph, scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE)
        for first_index, first in enumerate(edges):
            for second in edges[first_index + 1 :]:
                if (
                    first.source_node_id != second.source_node_id
                    and first.target_node_id != second.target_node_id
                ):
                    first.target_node_id, second.target_node_id = (
                        second.target_node_id,
                        first.target_node_id,
                    )
                    _reidentify_edge(envelope.graph, first)
                    _reidentify_edge(envelope.graph, second)
                    return
    raise FaultTransformError("swap-control-edges requires two controls governing distinct lanes")


def _drop_control_edges(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]], fraction: float
) -> None:
    candidates = [
        (envelope.graph, edge.edge_id)
        for _, envelope in envelopes
        for edge in envelope.graph.edges
        if edge.edge_type == scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE
    ]
    if not candidates:
        raise FaultTransformError("drop-control-edges requires a control edge")
    count = max(1, math.ceil(len(candidates) * fraction))
    selected = {identity for _, identity in candidates[:count]}
    for _, envelope in envelopes:
        _replace(
            envelope.graph.edges,
            [edge for edge in envelope.graph.edges if edge.edge_id not in selected],
        )


def _find_successor_path(graph: scg.SceneControlGraph) -> list[int] | None:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge in graph.edges:
        if edge.edge_type == scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR:
            adjacency[edge.source_node_id].append(edge.target_node_id)
    for values in adjacency.values():
        values.sort()
    for start in sorted(adjacency):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) == 4:
                return path
            for target in reversed(adjacency.get(node, [])):
                if target not in path:
                    stack.append((target, [*path, target]))
    return None


def _drop_successor_chain(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]],
) -> None:
    for _, envelope in envelopes:
        path = _find_successor_path(envelope.graph)
        if path is None:
            continue
        source, target = path[1], path[2]
        retained = [
            edge
            for edge in envelope.graph.edges
            if not (
                edge.edge_type == scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR
                and edge.source_node_id == source
                and edge.target_node_id == target
            )
        ]
        _replace(envelope.graph.edges, retained)
        return
    raise FaultTransformError("drop-successor-chain requires a reachable three-hop lane path")


def _endpoint_gap(source: scg.LaneSegment, target: scg.LaneSegment) -> float:
    first = source.centerline.points[-1]
    second = target.centerline.points[0]
    return math.sqrt(
        (first.x - second.x) ** 2 + (first.y - second.y) ** 2 + (first.z - second.z) ** 2
    )


def _add_spurious_successor(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]],
) -> None:
    for _, envelope in envelopes:
        graph = envelope.graph
        existing = {
            (edge.source_node_id, edge.target_node_id)
            for edge in graph.edges
            if edge.edge_type == scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR
        }
        candidates = sorted(
            (
                _endpoint_gap(source, target),
                source.node_id,
                target.node_id,
            )
            for source in graph.lanes
            for target in graph.lanes
            if source.node_id != target.node_id and (source.node_id, target.node_id) not in existing
        )
        if not candidates:
            continue
        _, source_id, target_id = candidates[0]
        edge = graph.edges.add(
            edge_type=scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR,
            source_node_id=source_id,
            target_node_id=target_id,
            raw_probability=0.99,
            calibrated_probability=0.99,
            binary_decision=True,
        )
        edge.uncertainty.standard_deviation = 0.01
        edge.uncertainty.method = "fault-lab-seeded"
        _reidentify_edge(graph, edge)
        return
    raise FaultTransformError("add-spurious-successors requires two lane nodes")


def _permute_correctly(envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]]) -> None:
    target = next((envelope for _, envelope in envelopes if len(envelope.graph.lanes) >= 2), None)
    if target is None:
        raise FaultTransformError("permute-nodes-correctly requires at least two lanes")
    graph = target.graph
    for collection in (
        graph.lanes,
        graph.traffic_controls,
        graph.road_areas,
        graph.edges,
        graph.tracks,
    ):
        _replace(collection, reversed(list(collection)))


def _permute_without_edges(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]],
) -> None:
    target = next((envelope for _, envelope in envelopes if len(envelope.graph.lanes) >= 2), None)
    if target is None:
        raise FaultTransformError("permute-nodes-without-edges requires at least two lanes")
    first = deepcopy(target.graph.lanes[0])
    second = deepcopy(target.graph.lanes[1])
    first_identity = (target.graph.lanes[0].node_id, target.graph.lanes[0].track_id)
    second_identity = (target.graph.lanes[1].node_id, target.graph.lanes[1].track_id)
    target.graph.lanes[0].CopyFrom(second)
    target.graph.lanes[1].CopyFrom(first)
    target.graph.lanes[0].node_id, target.graph.lanes[0].track_id = first_identity
    target.graph.lanes[1].node_id, target.graph.lanes[1].track_id = second_identity


def _duplicate_node_id(envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]]) -> None:
    for _, envelope in envelopes:
        graph = envelope.graph
        nodes = [*graph.lanes, *graph.traffic_controls, *graph.road_areas]
        if len(nodes) >= 2:
            nodes[1].node_id = nodes[0].node_id
            return
    raise FaultTransformError("duplicate-node-id requires two nodes")


def _dangling_edge(envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]]) -> None:
    for _, envelope in envelopes:
        if envelope.graph.edges:
            used = {
                node.node_id
                for nodes in (
                    envelope.graph.lanes,
                    envelope.graph.traffic_controls,
                    envelope.graph.road_areas,
                )
                for node in nodes
            }
            missing = max(used) + 1
            while missing in used:
                missing += 1
            envelope.graph.edges[0].target_node_id = missing
            return
    raise FaultTransformError("dangling-edge requires an edge")


def _lane_points(lane: scg.LaneSegment) -> Iterable[scg.Point3d]:
    for polyline in (lane.centerline, lane.left_boundary, lane.right_boundary):
        yield from polyline.points


def _jitter_lanes(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]], seed: int
) -> None:
    generator = random.Random(seed)  # noqa: S311 - reproducible perturbations, not security
    changed = False
    for _, envelope in envelopes:
        for lane in envelope.graph.lanes:
            for point in _lane_points(lane):
                point.x += generator.uniform(-0.35, 0.35)
                point.y += generator.uniform(-0.35, 0.35)
                changed = True
    if not changed:
        raise FaultTransformError("jitter-lanes requires lane geometry")


def _flip_boundaries(envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]]) -> None:
    for _, envelope in envelopes:
        if not envelope.graph.lanes:
            continue
        lane = envelope.graph.lanes[0]
        left = deepcopy(lane.left_boundary)
        left_type = deepcopy(lane.left_boundary_type)
        lane.left_boundary.CopyFrom(lane.right_boundary)
        lane.left_boundary_type.CopyFrom(lane.right_boundary_type)
        lane.right_boundary.CopyFrom(left)
        lane.right_boundary_type.CopyFrom(left_type)
        return
    raise FaultTransformError("flip-boundaries requires a lane")


def _corrupt_extrinsic(envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]]) -> None:
    for _, envelope in envelopes:
        if not envelope.graph.HasField("sensor_frame"):
            continue
        for camera in envelope.graph.sensor_frame.cameras:
            if camera.valid and len(camera.t_vehicle_camera.values) == 16:
                camera.t_vehicle_camera.values[3] += 0.75
                return
    raise FaultTransformError("corrupt-extrinsic requires a valid calibrated camera")


def _scales(envelope: scg.SceneControlGraphEnvelope) -> Iterable[scg.LaplaceScale3d]:
    for lane in envelope.graph.lanes:
        yield from lane.centerline_laplace_scale_m
        for polyline in (lane.centerline, lane.left_boundary, lane.right_boundary):
            yield from polyline.point_uncertainty
    for area in envelope.graph.road_areas:
        yield from area.geometry_uncertainty
        yield from area.geometry.point_uncertainty


def _set_uncertainty(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]], value: float
) -> None:
    changed = False
    for _, envelope in envelopes:
        for scale in _scales(envelope):
            scale.x = scale.y = scale.z = value
            changed = True
    if not changed:
        raise FaultTransformError("uncertainty fault requires geometry scale evidence")


def _temperature_probability(value: float, temperature: float = 0.1) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    logit = math.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + math.exp(-logit))


def _temperature_distribution(values: Sequence[float]) -> list[float]:
    logits = [math.log(max(value, 1e-12)) / 0.1 for value in values]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _collapse_temperature(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]],
) -> None:
    changed = False
    for _, envelope in envelopes:
        graph = envelope.graph
        for lane in graph.lanes:
            for polyline in (lane.centerline, lane.left_boundary, lane.right_boundary):
                polyline.confidence = _temperature_probability(polyline.confidence)
            for distribution in (lane.left_boundary_type, lane.right_boundary_type):
                distribution.probabilities[:] = _temperature_distribution(
                    distribution.probabilities
                )
            lane.existence_confidence = _temperature_probability(lane.existence_confidence)
            lane.intersection_or_connector_probability = _temperature_probability(
                lane.intersection_or_connector_probability
            )
            changed = True
        for control in graph.traffic_controls:
            for distribution in (
                control.category_distribution,
                control.attribute_distribution,
            ):
                distribution.probabilities[:] = _temperature_distribution(
                    distribution.probabilities
                )
            for field in (
                "existence_confidence",
                "calibrated_class_confidence",
                "calibrated_attribute_confidence",
            ):
                setattr(control, field, _temperature_probability(float(getattr(control, field))))
            changed = True
        for area in graph.road_areas:
            area.category_distribution.probabilities[:] = _temperature_distribution(
                area.category_distribution.probabilities
            )
            area.existence_confidence = _temperature_probability(area.existence_confidence)
            changed = True
        for edge in graph.edges:
            edge.raw_probability = _temperature_probability(edge.raw_probability)
            edge.calibrated_probability = _temperature_probability(edge.calibrated_probability)
            changed = True
    if not changed:
        raise FaultTransformError("temperature-collapse requires probability evidence")


def _inject_nan(envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]]) -> None:
    for _, envelope in envelopes:
        if envelope.graph.lanes and envelope.graph.lanes[0].centerline.points:
            envelope.graph.lanes[0].centerline.points[0].x = math.nan
            return
    raise FaultTransformError("inject-nan requires a lane point")


def _temporal_group(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]],
) -> list[scg.SceneControlGraphEnvelope]:
    groups: dict[str, list[scg.SceneControlGraphEnvelope]] = defaultdict(list)
    for _, envelope in envelopes:
        groups[envelope.graph.frame_key.segment_id].append(envelope)
    eligible = [group for group in groups.values() if len(group) >= 3]
    if not eligible:
        raise FaultTransformError("temporal fault requires at least three frames in one segment")
    return sorted(
        eligible,
        key=lambda group: (group[0].graph.frame_key.segment_id, len(group)),
    )[0]


def _alternate_edge_confidence(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]],
) -> None:
    group = sorted(_temporal_group(envelopes), key=lambda item: item.graph.frame_key.timestamp_ns)
    common = set.intersection(
        *[
            {
                (edge.edge_type, edge.source_node_id, edge.target_node_id)
                for edge in envelope.graph.edges
            }
            for envelope in group
        ]
    )
    if not common:
        raise FaultTransformError("alternate-edge-confidence requires a persistent edge")
    key = sorted(common)[0]
    for index, envelope in enumerate(group):
        edge = next(
            item
            for item in envelope.graph.edges
            if (item.edge_type, item.source_node_id, item.target_node_id) == key
        )
        edge.raw_probability = edge.calibrated_probability = 0.9 if index % 2 == 0 else 0.1
        edge.binary_decision = index % 2 == 0


def _remove_node(graph: scg.SceneControlGraph, node_id: int) -> None:
    _replace(graph.lanes, [node for node in graph.lanes if node.node_id != node_id])
    _replace(
        graph.edges,
        [
            edge
            for edge in graph.edges
            if edge.source_node_id != node_id and edge.target_node_id != node_id
        ],
    )
    _replace(graph.tracks, [track for track in graph.tracks if track.current_node_id != node_id])


def _alternate_node_presence(
    envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]],
) -> None:
    group = sorted(_temporal_group(envelopes), key=lambda item: item.graph.frame_key.timestamp_ns)
    common = set.intersection(*[{lane.node_id for lane in item.graph.lanes} for item in group])
    if not common:
        raise FaultTransformError("alternate-node-presence requires a persistent lane")
    node_id = min(common)
    for index, envelope in enumerate(group):
        if index % 2 == 1:
            _remove_node(envelope.graph, node_id)


def _reuse_track_id(envelopes: Sequence[tuple[str, scg.SceneControlGraphEnvelope]]) -> None:
    group = sorted(_temporal_group(envelopes), key=lambda item: item.graph.frame_key.timestamp_ns)
    common = set.intersection(*[{track.track_id for track in item.graph.tracks} for item in group])
    if not common:
        raise FaultTransformError("reuse-track-id requires a persistent track")
    track_id = min(common)
    first = next(track for track in group[0].graph.tracks if track.track_id == track_id)
    first.termination_reason = scg.TRACK_TERMINATION_REASON_MAX_MISSES
    for envelope in group[1:]:
        track = next(item for item in envelope.graph.tracks if item.track_id == track_id)
        track.termination_reason = scg.TRACK_TERMINATION_REASON_ACTIVE


def _runtime_transform(runtime: RuntimeFixture, kind: FaultKind) -> RuntimeFixture:
    payload = runtime.model_dump(mode="json")
    if kind == FaultKind.FORCE_PROVIDER_FALLBACK:
        payload["provider_node_counts"]["CPUExecutionProvider"] = 1
    elif kind == FaultKind.DELAY_POSTPROCESS:
        payload["postprocess_latency_ms"] = inject_bounded_latency(
            runtime.postprocess_latency_ms,
            added_delay_ms=120.0,
        )
    elif kind == FaultKind.LEAK_BUFFER:
        fixture = run_allocator_fixture(
            frame_count=len(runtime.device_memory_bytes),
            buffer_bytes=64 * 1024,
            leak=True,
            baseline_bytes=runtime.device_memory_bytes[0],
        )
        payload["device_memory_bytes"] = fixture.memory_samples_bytes
    return RuntimeFixture.model_validate(payload)


def apply_fault(
    bundle: PredictionBundle,
    kind: FaultKind,
    *,
    seed: int,
    fraction: float,
) -> PredictionBundle:
    """Apply exactly one deterministic transformation without mutating the input bundle."""
    if bundle.fault_history:
        raise FaultTransformError("V1 fault injection requires an original clean bundle")
    envelopes = [
        (token, deepcopy(envelope)) for token, envelope in decode_envelopes(bundle, validate=True)
    ]
    if kind == FaultKind.SWAP_CONTROL_EDGES:
        _swap_control_edges(envelopes)
    elif kind == FaultKind.DROP_CONTROL_EDGES:
        _drop_control_edges(envelopes, fraction)
    elif kind == FaultKind.DROP_SUCCESSOR_CHAIN:
        _drop_successor_chain(envelopes)
    elif kind == FaultKind.ADD_SPURIOUS_SUCCESSORS:
        _add_spurious_successor(envelopes)
    elif kind == FaultKind.PERMUTE_NODES_CORRECTLY:
        _permute_correctly(envelopes)
    elif kind == FaultKind.PERMUTE_NODES_WITHOUT_EDGES:
        _permute_without_edges(envelopes)
    elif kind == FaultKind.DUPLICATE_NODE_ID:
        _duplicate_node_id(envelopes)
    elif kind == FaultKind.DANGLING_EDGE:
        _dangling_edge(envelopes)
    elif kind == FaultKind.JITTER_LANES:
        _jitter_lanes(envelopes, seed)
    elif kind == FaultKind.FLIP_BOUNDARIES:
        _flip_boundaries(envelopes)
    elif kind == FaultKind.CORRUPT_EXTRINSIC:
        _corrupt_extrinsic(envelopes)
    elif kind == FaultKind.ZERO_UNCERTAINTY:
        _set_uncertainty(envelopes, 0.0)
    elif kind == FaultKind.INFLATE_UNCERTAINTY:
        _set_uncertainty(envelopes, 100.0)
    elif kind == FaultKind.TEMPERATURE_COLLAPSE:
        _collapse_temperature(envelopes)
    elif kind == FaultKind.INJECT_NAN:
        _inject_nan(envelopes)
    elif kind == FaultKind.ALTERNATE_EDGE_CONFIDENCE:
        _alternate_edge_confidence(envelopes)
    elif kind == FaultKind.ALTERNATE_NODE_PRESENCE:
        _alternate_node_presence(envelopes)
    elif kind == FaultKind.REUSE_TRACK_ID:
        _reuse_track_id(envelopes)
    runtime = _runtime_transform(bundle.runtime, kind)
    derived = PredictionBundle(
        schema_version="junctionlens.prediction-bundle.v1",
        bundle_id=f"{bundle.bundle_id}.{kind.value}.seed-{seed}",
        frames=_encode_envelopes(envelopes),
        runtime=runtime,
        fault_history=(FaultDeclaration(kind=kind, seed=seed, fraction=fraction),),
    )
    if kind not in {
        FaultKind.DUPLICATE_NODE_ID,
        FaultKind.DANGLING_EDGE,
        FaultKind.ZERO_UNCERTAINTY,
        FaultKind.INJECT_NAN,
    }:
        for _, envelope in decode_envelopes(derived, validate=False):
            validate_envelope(envelope)
    return derived


__all__ = ["FaultTransformError", "apply_fault", "decode_envelopes"]

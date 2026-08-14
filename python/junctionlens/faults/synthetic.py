"""Repository-owned clean synthetic bundle that exercises every V1 fault target."""

from __future__ import annotations

import base64
from copy import deepcopy
from itertools import pairwise

from junctionlens.contract.ids import edge_id, predicted_node_id
from junctionlens.contract.validation import validate_envelope
from junctionlens.faults.models import PredictionBundle, PredictionFrame, RuntimeFixture
from junctionlens.synthetic import generate_scene_frames
from junctionlens.v1 import scene_control_graph_pb2 as scg

_FRAME_INTERVAL_NS = 100_000_000


def _set_distribution(distribution: scg.ClassDistribution) -> None:
    values = list(distribution.probabilities)
    if len(values) < 2:
        return
    winner = max(range(len(values)), key=values.__getitem__)
    remainder = 0.2 / (len(values) - 1)
    distribution.probabilities[:] = [
        0.8 if index == winner else remainder for index in range(len(values))
    ]


def _soften_probabilities(envelope: scg.SceneControlGraphEnvelope) -> None:
    graph = envelope.graph
    for lane in graph.lanes:
        lane.existence_confidence = 0.8
        lane.intersection_or_connector_probability = (
            0.8 if lane.intersection_or_connector_probability >= 0.5 else 0.2
        )
        for polyline in (lane.centerline, lane.left_boundary, lane.right_boundary):
            polyline.confidence = 0.8
        _set_distribution(lane.left_boundary_type)
        _set_distribution(lane.right_boundary_type)
    for control in graph.traffic_controls:
        control.existence_confidence = 0.8
        control.calibrated_class_confidence = 0.8
        control.calibrated_attribute_confidence = 0.8
        _set_distribution(control.category_distribution)
        _set_distribution(control.attribute_distribution)
    for area in graph.road_areas:
        area.existence_confidence = 0.8
        area.geometry.confidence = 0.8
        _set_distribution(area.category_distribution)
    for edge in graph.edges:
        edge.raw_probability = 0.8
        edge.calibrated_probability = 0.8
        edge.binary_decision = True


def _new_track_id(graph: scg.SceneControlGraph) -> int:
    return max((track.track_id for track in graph.tracks), default=0) + 1


def _enhance_controls(envelope: scg.SceneControlGraphEnvelope) -> None:
    graph = envelope.graph
    if not graph.traffic_controls or len(graph.lanes) < 2:
        raise ValueError("synthetic control fixture lacks required nodes")
    original = graph.traffic_controls[0]
    control = graph.traffic_controls.add()
    control.CopyFrom(original)
    control.node_id = predicted_node_id(scg.NODE_TYPE_TRAFFIC_CONTROL, 1)
    control.track_id = _new_track_id(graph)
    control.decoder_query_index = 1
    control.normalized_half_open_box.x_min = 0.55
    control.normalized_half_open_box.x_max = 0.63
    if control.HasField("source_pixel_box"):
        width = control.source_pixel_box.image_width
        control.source_pixel_box.x0 = 0.55 * width
        control.source_pixel_box.x1 = 0.63 * width
    track = graph.tracks.add()
    track.CopyFrom(next(item for item in graph.tracks if item.current_node_id == original.node_id))
    track.track_id = control.track_id
    track.current_node_id = control.node_id
    edge = graph.edges.add(
        edge_type=scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE,
        source_node_id=control.node_id,
        target_node_id=graph.lanes[1].node_id,
        raw_probability=0.8,
        calibrated_probability=0.8,
        binary_decision=True,
    )
    edge.uncertainty.standard_deviation = 0.05
    edge.uncertainty.method = "synthetic-fault-control"
    edge.edge_id = edge_id(
        graph.frame_key, edge.edge_type, edge.source_node_id, edge.target_node_id
    )


def _enhance_successor_chain(envelope: scg.SceneControlGraphEnvelope) -> None:
    graph = envelope.graph
    if len(graph.lanes) < 3:
        raise ValueError("synthetic successor fixture lacks required lanes")
    lane = graph.lanes.add()
    lane.CopyFrom(graph.lanes[-2])
    lane.node_id = predicted_node_id(scg.NODE_TYPE_LANE_SEGMENT, 3)
    lane.track_id = _new_track_id(graph)
    lane.decoder_query_index = 3
    for polyline in (lane.centerline, lane.left_boundary, lane.right_boundary):
        for point in polyline.points:
            point.x += 25.0
            point.y += 2.0
    track = graph.tracks.add()
    track.CopyFrom(
        next(item for item in graph.tracks if item.current_node_id == graph.lanes[1].node_id)
    )
    track.track_id = lane.track_id
    track.current_node_id = lane.node_id
    retained = [
        deepcopy(edge)
        for edge in graph.edges
        if edge.edge_type != scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR
    ]
    del graph.edges[:]
    graph.edges.extend(retained)
    lane_ids = [item.node_id for item in graph.lanes]
    for source, target in pairwise(lane_ids):
        edge = graph.edges.add(
            edge_type=scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR,
            source_node_id=source,
            target_node_id=target,
            raw_probability=0.8,
            calibrated_probability=0.8,
            binary_decision=True,
        )
        edge.uncertainty.standard_deviation = 0.05
        edge.uncertainty.method = "synthetic-fault-chain"
        edge.edge_id = edge_id(
            graph.frame_key, edge.edge_type, edge.source_node_id, edge.target_node_id
        )


def _retime(envelope: scg.SceneControlGraphEnvelope, index: int) -> None:
    graph = envelope.graph
    timestamp = graph.frame_key.timestamp_ns + index * _FRAME_INTERVAL_NS
    graph.frame_key.timestamp_ns = timestamp
    if graph.HasField("sensor_frame"):
        graph.sensor_frame.frame_key.CopyFrom(graph.frame_key)
        graph.sensor_frame.t_world_vehicle.values[3] += index * 0.5
    for track in graph.tracks:
        track.last_timestamp_ns = timestamp
        track.age_observed_frames += index
    for edge in graph.edges:
        edge.edge_id = edge_id(
            graph.frame_key,
            edge.edge_type,
            edge.source_node_id,
            edge.target_node_id,
        )


def _frame(token: str, envelope: scg.SceneControlGraphEnvelope) -> PredictionFrame:
    validate_envelope(envelope)
    return PredictionFrame(
        frame_token=token,
        envelope_pb_base64=base64.b64encode(envelope.SerializeToString(deterministic=True)).decode(
            "ascii"
        ),
    )


def build_synthetic_fault_bundle() -> PredictionBundle:
    """Build a deterministic clean sequence with every mandatory fault precondition."""
    frames = generate_scene_frames()
    intersection = deepcopy(
        next(
            item.perfect_prediction
            for item in frames
            if item.scene_kind.value == "intersection-crosswalk" and item.frame_index == 0
        )
    )
    merge = deepcopy(
        next(
            item.perfect_prediction
            for item in frames
            if item.scene_kind.value == "merge" and item.frame_index == 0
        )
    )
    _enhance_controls(intersection)
    _soften_probabilities(intersection)
    _enhance_successor_chain(merge)
    _soften_probabilities(merge)
    temporal = []
    for index in range(3):
        envelope = deepcopy(merge)
        _retime(envelope, index)
        temporal.append(_frame(f"merge-temporal-{index}", envelope))
    runtime = RuntimeFixture(
        provider_node_counts={"CUDAExecutionProvider": 128, "CPUExecutionProvider": 0},
        postprocess_latency_ms=tuple(5.0 for _ in range(20)),
        device_memory_bytes=tuple(512 * 1024 * 1024 for _ in range(100)),
    )
    return PredictionBundle(
        schema_version="junctionlens.prediction-bundle.v1",
        bundle_id="synthetic-fault-lab-v1",
        frames=(_frame("intersection-control", intersection), *temporal),
        runtime=runtime,
        fault_history=(),
    )


__all__ = ["build_synthetic_fault_bundle"]

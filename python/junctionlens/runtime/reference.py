"""Independent Python oracle for native C++ graph postprocessing parity."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from junctionlens.contract.ids import edge_id, predicted_node_id
from junctionlens.contract.validation import validate_envelope
from junctionlens.v1 import scene_control_graph_pb2 as scg

FloatArray = npt.NDArray[np.float32]


def _sigmoid(value: float | np.float32) -> float:
    number = float(value)
    if number >= 0.0:
        exponent = math.exp(-number)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(number)
    return exponent / (1.0 + exponent)


def _softmax(values: Sequence[float | np.float32]) -> tuple[float, ...]:
    maximum = max(float(value) for value in values)
    exponentials = tuple(math.exp(float(value) - maximum) for value in values)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _quantize(value: float | np.float32, step: float) -> int:
    scaled = float(value) / step
    return math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)


def _ordered_box(values: FloatArray) -> tuple[float, float, float, float]:
    x_min, x_max = sorted((float(values[0]), float(values[2])))
    y_min, y_max = sorted((float(values[1]), float(values[3])))
    x_min, x_max = min(1.0, max(0.0, x_min)), min(1.0, max(0.0, x_max))
    y_min, y_max = min(1.0, max(0.0, y_min)), min(1.0, max(0.0, y_max))
    if x_min == x_max:
        x_min, x_max = max(0.0, x_min - 1e-7), min(1.0, x_max + 1e-7)
    if y_min == y_max:
        y_min, y_max = max(0.0, y_min - 1e-7), min(1.0, y_max + 1e-7)
    return x_min, y_min, x_max, y_max


def _area_indices(logits: FloatArray) -> tuple[int, ...]:
    retained = tuple(index for index, value in enumerate(logits) if _sigmoid(value) >= 0.5)
    if len(retained) >= 2:
        return retained
    highest = sorted(range(20), key=lambda index: (-float(logits[index]), index))[:2]
    return tuple(sorted(highest))


def _add_point(line: scg.Polyline3d, values: FloatArray, index: int) -> None:
    line.points.add(
        x=float(values[index, 0]),
        y=float(values[index, 1]),
        z=float(values[index, 2]),
    )


def _add_scale(
    destination: Any,
    values: FloatArray,
    index: int,
) -> None:
    destination.add(
        x=float(values[index, 0]),
        y=float(values[index, 1]),
        z=float(values[index, 2]),
    )


def reference_postprocess(
    outputs: Mapping[str, FloatArray],
    sensor_frame: scg.SensorFrame,
    producer: scg.ProducerInfo,
    *,
    node_threshold: float = 0.5,
    edge_threshold: float = 0.5,
) -> scg.SceneControlGraphEnvelope:
    """Convert exact ORT outputs through the frozen language-neutral V1 policy."""
    required = {
        "lane_existence_logits",
        "lane_centerline",
        "lane_left_boundary",
        "lane_right_boundary",
        "lane_left_boundary_logits",
        "lane_right_boundary_logits",
        "lane_connector_logits",
        "lane_geometry_scales",
        "traffic_existence_logits",
        "traffic_boxes",
        "traffic_category_logits",
        "traffic_attribute_logits",
        "area_existence_logits",
        "area_category_logits",
        "area_points",
        "area_valid_logits",
        "area_geometry_scales",
        "lane_successor_logits",
        "control_lane_logits",
    }
    if not required.issubset(outputs):
        raise ValueError("reference postprocessor output mapping is incomplete")
    if any(not np.isfinite(value).all() for value in outputs.values()):
        raise ValueError("reference postprocessor received a nonfinite model output")
    lane_records: list[tuple[float, tuple[int, ...], int]] = []
    for query in range(96):
        confidence = _sigmoid(outputs["lane_existence_logits"][0, query])
        if confidence >= node_threshold:
            key = tuple(
                _quantize(value, 0.001)
                for name in ("lane_centerline", "lane_left_boundary", "lane_right_boundary")
                for value in outputs[name][0, query].flat
            )
            lane_records.append((confidence, key, query))
    traffic_records: list[tuple[float, tuple[int, ...], int]] = []
    for query in range(64):
        confidence = _sigmoid(outputs["traffic_existence_logits"][0, query])
        if confidence >= node_threshold:
            box = _ordered_box(outputs["traffic_boxes"][0, query])
            traffic_records.append(
                (confidence, tuple(_quantize(value, 0.000001) for value in box), query)
            )
    area_records: list[tuple[float, tuple[int, ...], int]] = []
    area_points: dict[int, tuple[int, ...]] = {}
    for query in range(32):
        confidence = _sigmoid(outputs["area_existence_logits"][0, query])
        if confidence >= node_threshold:
            indices = _area_indices(outputs["area_valid_logits"][0, query])
            area_points[query] = indices
            key = tuple(
                _quantize(value, 0.001)
                for index in indices
                for value in outputs["area_points"][0, query, index]
            )
            area_records.append((confidence, key, query))
    for records in (lane_records, traffic_records, area_records):
        records.sort(key=lambda item: (-item[0], item[1], item[2]))

    envelope = scg.SceneControlGraphEnvelope(schema_major=1, schema_minor=0)
    envelope.producer.CopyFrom(producer)
    graph = envelope.graph
    graph.role = scg.GRAPH_ROLE_PREDICTION
    graph.frame_key.CopyFrom(sensor_frame.frame_key)
    graph.sensor_frame.CopyFrom(sensor_frame)
    lane_ids: dict[int, int] = {}
    traffic_ids: dict[int, int] = {}
    for ordinal, (confidence, _key, query) in enumerate(lane_records):
        node_id = predicted_node_id(scg.NODE_TYPE_LANE_SEGMENT, ordinal)
        lane_ids[query] = node_id
        lane = graph.lanes.add(
            node_id=node_id,
            decoder_query_index=query,
            existence_confidence=confidence,
            intersection_or_connector_probability=_sigmoid(
                outputs["lane_connector_logits"][0, query]
            ),
        )
        for line, name, kind in (
            (lane.centerline, "lane_centerline", 0),
            (lane.left_boundary, "lane_left_boundary", 1),
            (lane.right_boundary, "lane_right_boundary", 2),
        ):
            line.confidence = confidence
            values = outputs[name][0, query]
            scales = outputs["lane_geometry_scales"][0, query, kind]
            for point in range(11):
                _add_point(line, values, point)
                _add_scale(line.point_uncertainty, scales, point)
                if kind == 0:
                    _add_scale(lane.centerline_laplace_scale_m, scales, point)
        lane.left_boundary_type.probabilities.extend(
            _softmax(outputs["lane_left_boundary_logits"][0, query])
        )
        lane.right_boundary_type.probabilities.extend(
            _softmax(outputs["lane_right_boundary_logits"][0, query])
        )
    for ordinal, (confidence, _key, query) in enumerate(traffic_records):
        node_id = predicted_node_id(scg.NODE_TYPE_TRAFFIC_CONTROL, ordinal)
        traffic_ids[query] = node_id
        control = graph.traffic_controls.add(
            node_id=node_id,
            decoder_query_index=query,
            source_camera=scg.CAMERA_SLOT_FRONT_CENTER,
            existence_confidence=confidence,
        )
        box = _ordered_box(outputs["traffic_boxes"][0, query])
        (
            control.normalized_half_open_box.x_min,
            control.normalized_half_open_box.y_min,
            control.normalized_half_open_box.x_max,
            control.normalized_half_open_box.y_max,
        ) = box
        category = _softmax(outputs["traffic_category_logits"][0, query])
        attributes = _softmax(outputs["traffic_attribute_logits"][0, query])
        control.category_distribution.probabilities.extend(category)
        control.attribute_distribution.probabilities.extend(attributes)
        control.calibrated_class_confidence = max(category)
        control.calibrated_attribute_confidence = max(attributes)
    for ordinal, (confidence, _key, query) in enumerate(area_records):
        area = graph.road_areas.add(
            node_id=predicted_node_id(scg.NODE_TYPE_ROAD_AREA, ordinal),
            decoder_query_index=query,
            existence_confidence=confidence,
        )
        area.category_distribution.probabilities.extend(
            _softmax(outputs["area_category_logits"][0, query])
        )
        area.geometry.confidence = confidence
        values = outputs["area_points"][0, query]
        scales = outputs["area_geometry_scales"][0, query]
        for point in area_points[query]:
            _add_point(area.geometry, values, point)
            _add_scale(area.geometry.point_uncertainty, scales, point)
            _add_scale(area.geometry_uncertainty, scales, point)
    edges: list[tuple[int, int, int, float]] = []
    for source_query, source_id in lane_ids.items():
        for target_query, target_id in lane_ids.items():
            probability = _sigmoid(outputs["lane_successor_logits"][0, source_query, target_query])
            if probability >= edge_threshold:
                edges.append(
                    (scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR, source_id, target_id, probability)
                )
    for source_query, source_id in traffic_ids.items():
        for target_query, target_id in lane_ids.items():
            probability = _sigmoid(outputs["control_lane_logits"][0, source_query, target_query])
            if probability >= edge_threshold:
                edges.append(
                    (
                        scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE,
                        source_id,
                        target_id,
                        probability,
                    )
                )
    for edge_type, source, target, probability in sorted(edges):
        edge = graph.edges.add(
            edge_type=edge_type,
            source_node_id=source,
            target_node_id=target,
            raw_probability=probability,
            calibrated_probability=probability,
            binary_decision=True,
        )
        edge.edge_id = edge_id(graph.frame_key, edge_type, source, target)
    validate_envelope(envelope)
    return envelope

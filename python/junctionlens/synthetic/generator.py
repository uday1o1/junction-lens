"""Deterministic generation of calibrated synthetic V1 graph truth."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from junctionlens.contract import validate_envelope
from junctionlens.contract.ids import edge_id, predicted_node_id
from junctionlens.data.geometry import validate_lane_boundary_orientation
from junctionlens.geometry import resample_polyline
from junctionlens.synthetic.calibration import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    calibration_sha256,
    camera_calibrations,
)
from junctionlens.synthetic.models import (
    CorruptionKind,
    LaneSpec,
    SceneKind,
    SceneSpec,
    scene_specs,
)
from junctionlens.synthetic.render import render_camera_svg
from junctionlens.v1 import scene_control_graph_pb2 as scg

FloatArray = npt.NDArray[np.float64]
BASE_TIMESTAMP_NS = 1_725_000_000_000_000_000
FRAME_INTERVAL_NS = 100_000_000
EGO_STEP_METERS = 2.0
LANE_POINT_COUNT = 11
LANE_HALF_WIDTH_METERS = 1.75
IDENTITY3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class GeneratedSceneFrame:
    """One generated sensor frame with truth, a perfect prediction, and source images."""

    scene_kind: SceneKind
    frame_index: int
    ground_truth: scg.SceneControlGraphEnvelope
    perfect_prediction: scg.SceneControlGraphEnvelope
    camera_images: dict[str, bytes]


@dataclass(frozen=True, slots=True)
class GeneratedCorruption:
    """One valid prediction with a single controlled semantic fault."""

    scene_kind: SceneKind
    frame_index: int
    corruption: CorruptionKind
    prediction: scg.SceneControlGraphEnvelope


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return value or 1


def _configuration_sha256() -> str:
    payload = json.dumps(
        [asdict(specification) for specification in scene_specs()],
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_text(payload)


def _seed_lateral_offset(seed: int, kind: SceneKind) -> float:
    digest = hashlib.sha256(f"junctionlens-synthetic-v1|{seed}|{kind}".encode()).digest()
    unit = int.from_bytes(digest[:4], "big") / float((1 << 32) - 1)
    return (unit - 0.5) * 0.5


def _current_points(
    points: tuple[tuple[float, float, float], ...],
    *,
    ego_x: float,
    lateral_offset: float,
) -> FloatArray:
    result = np.asarray(points, dtype=np.float64).copy()
    result[:, 0] -= ego_x
    result[:, 1] += lateral_offset
    return result


def _lane_geometry(
    specification: LaneSpec,
    *,
    ego_x: float,
    lateral_offset: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    anchors = _current_points(
        specification.anchors,
        ego_x=ego_x,
        lateral_offset=lateral_offset,
    )
    centerline = resample_polyline(anchors, LANE_POINT_COUNT)
    tangent = np.empty((len(centerline), 2), dtype=np.float64)
    tangent[0] = centerline[1, :2] - centerline[0, :2]
    tangent[-1] = centerline[-1, :2] - centerline[-2, :2]
    tangent[1:-1] = centerline[2:, :2] - centerline[:-2, :2]
    lengths = np.linalg.norm(tangent, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError(f"synthetic lane {specification.key} has an undefined tangent")
    tangent /= lengths[:, None]
    left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left = centerline.copy()
    right = centerline.copy()
    left[:, :2] += LANE_HALF_WIDTH_METERS * left_normal
    right[:, :2] -= LANE_HALF_WIDTH_METERS * left_normal
    validate_lane_boundary_orientation(centerline, left, right)
    return centerline, left, right


def _copy_polyline(target: scg.Polyline3d, points: FloatArray, *, scale: float) -> None:
    target.confidence = 1.0
    for x_coordinate, y_coordinate, z_coordinate in points:
        target.points.add(x=x_coordinate, y=y_coordinate, z=z_coordinate)
        target.point_uncertainty.add(x=scale, y=scale, z=scale)


def _populate_producer(
    envelope: scg.SceneControlGraphEnvelope,
    *,
    seed: int,
    profile: str,
    model_artifact_sha256: str = "",
) -> None:
    envelope.schema_major = 1
    envelope.schema_minor = 0
    envelope.producer.git_commit = "0" * 40
    envelope.producer.git_dirty = False
    envelope.producer.model_artifact_sha256 = model_artifact_sha256
    envelope.producer.configuration_sha256 = _configuration_sha256()
    envelope.producer.runtime_build_sha256 = _sha256_text("junctionlens-synthetic-generator-v1")
    envelope.producer.execution_provider_profile = profile
    envelope.producer.provider_assignment_digest = _sha256_text(f"{profile}|no-runtime-nodes")
    envelope.producer.random_seed = seed


def _populate_frame_key(
    frame_key: scg.FrameKey,
    *,
    specification: SceneSpec,
    seed: int,
    frame_index: int,
) -> None:
    frame_key.dataset_id = "junctionlens-synthetic"
    frame_key.dataset_version = "v1"
    frame_key.split_id = "fixture"
    frame_key.segment_id = specification.kind.value
    frame_key.timestamp_ns = BASE_TIMESTAMP_NS + frame_index * FRAME_INTERVAL_NS
    frame_key.source_domain = scg.SOURCE_DOMAIN_SYNTHETIC
    frame_key.calibration_sha256 = calibration_sha256()
    frame_key.frame_manifest_sha256 = _sha256_text(
        f"junctionlens-synthetic-frame-v1|{seed}|{specification.kind}|{frame_index}"
    )


def _populate_sensor(
    graph: scg.SceneControlGraph,
    *,
    specification: SceneSpec,
    frame_index: int,
) -> None:
    sensor = graph.sensor_frame
    sensor.frame_key.CopyFrom(graph.frame_key)
    ego_x = frame_index * EGO_STEP_METERS
    sensor.t_world_vehicle.values.extend(
        (1.0, 0.0, 0.0, ego_x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    )
    sensor.pose_valid = True
    sensor.adapter_version = "junctionlens-synthetic-v1"
    for calibration in camera_calibrations():
        camera = sensor.cameras.add(slot=calibration.slot, valid=True)
        camera.capture_timestamp_ns = graph.frame_key.timestamp_ns
        camera.original_width = IMAGE_WIDTH
        camera.original_height = IMAGE_HEIGHT
        camera.intrinsic.values.extend(calibration.intrinsic.reshape(-1).tolist())
        camera.t_vehicle_camera.values.extend(calibration.t_vehicle_camera.reshape(-1).tolist())
        camera.distortion_model = scg.DISTORTION_MODEL_NONE
        camera.image_transform.original_to_model.values.extend(IDENTITY3)
        camera.image_transform.resized_width = IMAGE_WIDTH
        camera.image_transform.resized_height = IMAGE_HEIGHT
        camera.original_image.kind = scg.ARTIFACT_KIND_SOURCE_IMAGE
        camera.original_image.sha256 = "0" * 64
        camera.original_image.media_type = "image/svg+xml"
        camera.original_image.relative_uri = (
            f"renderings/{specification.kind.value}/frame-{frame_index:02d}/{calibration.slug}.svg"
        )
        camera.original_image.license_id = "LicenseRef-JunctionLens-Synthetic"


def _populate_lane(
    graph: scg.SceneControlGraph,
    specification: SceneSpec,
    lane_specification: LaneSpec,
    *,
    ordinal: int,
    frame_index: int,
    lateral_offset: float,
) -> int:
    node_id = _stable_u64("ground-truth", specification.kind, "lane", lane_specification.key)
    track_id = _stable_u64("track", specification.kind, "lane", lane_specification.key)
    centerline, left_boundary, right_boundary = _lane_geometry(
        lane_specification,
        ego_x=frame_index * EGO_STEP_METERS,
        lateral_offset=lateral_offset,
    )
    lane = graph.lanes.add(
        node_id=node_id,
        track_id=track_id,
        decoder_query_index=ordinal,
        intersection_or_connector_probability=(
            1.0 if lane_specification.intersection_or_connector else 0.0
        ),
        existence_confidence=1.0,
    )
    _copy_polyline(lane.centerline, centerline, scale=0.05)
    _copy_polyline(lane.left_boundary, left_boundary, scale=0.05)
    _copy_polyline(lane.right_boundary, right_boundary, scale=0.05)
    lane.left_boundary_type.probabilities.extend((1.0, 0.0))
    lane.right_boundary_type.probabilities.extend((1.0, 0.0))
    for _ in lane.centerline.points:
        lane.centerline_laplace_scale_m.add(x=0.05, y=0.05, z=0.05)
    lane.adapter_metadata.source_object_id = lane_specification.key
    lane.adapter_metadata.source_namespace = f"synthetic/{specification.kind.value}/lane"
    return node_id


def _populate_control(
    graph: scg.SceneControlGraph,
    specification: SceneSpec,
    *,
    ordinal: int,
) -> int:
    control_specification = specification.controls[ordinal]
    node_id = _stable_u64("ground-truth", specification.kind, "control", control_specification.key)
    track_id = _stable_u64("track", specification.kind, "control", control_specification.key)
    x_min, y_min, x_max, y_max = control_specification.normalized_box
    control = graph.traffic_controls.add(
        node_id=node_id,
        track_id=track_id,
        source_camera=scg.CAMERA_SLOT_FRONT_CENTER,
        existence_confidence=1.0,
        calibrated_class_confidence=1.0,
        calibrated_attribute_confidence=1.0,
        decoder_query_index=ordinal,
    )
    control.source_pixel_box.x0 = x_min * IMAGE_WIDTH
    control.source_pixel_box.y0 = y_min * IMAGE_HEIGHT
    control.source_pixel_box.x1 = x_max * IMAGE_WIDTH
    control.source_pixel_box.y1 = y_max * IMAGE_HEIGHT
    control.source_pixel_box.convention = scg.SOURCE_BOX_CONVENTION_XYXY_HALF_OPEN
    control.source_pixel_box.image_width = IMAGE_WIDTH
    control.source_pixel_box.image_height = IMAGE_HEIGHT
    control.normalized_half_open_box.x_min = x_min
    control.normalized_half_open_box.y_min = y_min
    control.normalized_half_open_box.x_max = x_max
    control.normalized_half_open_box.y_max = y_max
    control.category_distribution.probabilities.extend((1.0, 0.0, 0.0))
    control.attribute_distribution.probabilities.extend((1.0, 0.0, 0.0))
    control.adapter_metadata.source_object_id = control_specification.key
    control.adapter_metadata.source_namespace = f"synthetic/{specification.kind.value}/control"
    return node_id


def _populate_area(
    graph: scg.SceneControlGraph,
    specification: SceneSpec,
    *,
    ordinal: int,
    frame_index: int,
    lateral_offset: float,
) -> int:
    area_specification = specification.areas[ordinal]
    node_id = _stable_u64("ground-truth", specification.kind, "area", area_specification.key)
    track_id = _stable_u64("track", specification.kind, "area", area_specification.key)
    geometry = _current_points(
        area_specification.points,
        ego_x=frame_index * EGO_STEP_METERS,
        lateral_offset=lateral_offset,
    )
    area = graph.road_areas.add(
        node_id=node_id,
        track_id=track_id,
        existence_confidence=1.0,
        decoder_query_index=ordinal,
    )
    distribution = [0.0, 0.0]
    distribution[area_specification.category_index] = 1.0
    area.category_distribution.probabilities.extend(distribution)
    _copy_polyline(area.geometry, geometry, scale=0.10)
    for _ in area.geometry.points:
        area.geometry_uncertainty.add(x=0.10, y=0.10, z=0.10)
    area.adapter_metadata.source_object_id = area_specification.key
    area.adapter_metadata.source_namespace = f"synthetic/{specification.kind.value}/area"
    return node_id


def _add_edge(
    graph: scg.SceneControlGraph,
    edge_type: int,
    source_node_id: int,
    target_node_id: int,
) -> None:
    edge = graph.edges.add(
        edge_type=edge_type,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        raw_probability=1.0,
        calibrated_probability=1.0,
        binary_decision=True,
    )
    edge.edge_id = edge_id(graph.frame_key, edge_type, source_node_id, target_node_id)
    edge.uncertainty.standard_deviation = 0.0
    edge.uncertainty.method = "synthetic-exact"


def _populate_tracks(graph: scg.SceneControlGraph, frame_index: int) -> None:
    first_timestamp = graph.frame_key.timestamp_ns - frame_index * FRAME_INTERVAL_NS
    collections = (
        (graph.lanes, scg.NODE_TYPE_LANE_SEGMENT),
        (graph.traffic_controls, scg.NODE_TYPE_TRAFFIC_CONTROL),
        (graph.road_areas, scg.NODE_TYPE_ROAD_AREA),
    )
    for nodes, node_type in collections:
        for node in nodes:
            graph.tracks.add(
                track_id=node.track_id,
                node_type=node_type,
                current_node_id=node.node_id,
                first_timestamp_ns=first_timestamp,
                last_timestamp_ns=graph.frame_key.timestamp_ns,
                age_observed_frames=frame_index + 1,
                missed_frame_count=0,
                termination_reason=scg.TRACK_TERMINATION_REASON_ACTIVE,
            )


def _attach_images(
    envelope: scg.SceneControlGraphEnvelope,
) -> dict[str, bytes]:
    images: dict[str, bytes] = {}
    calibrations = {calibration.slot: calibration for calibration in camera_calibrations()}
    for camera in envelope.graph.sensor_frame.cameras:
        image = render_camera_svg(envelope.graph, calibrations[camera.slot])
        camera.original_image.sha256 = hashlib.sha256(image).hexdigest()
        camera.original_image.byte_size = len(image)
        images[camera.original_image.relative_uri] = image
    return images


def _ground_truth(
    specification: SceneSpec,
    *,
    seed: int,
    frame_index: int,
) -> tuple[scg.SceneControlGraphEnvelope, dict[str, bytes]]:
    envelope = scg.SceneControlGraphEnvelope()
    _populate_producer(envelope, seed=seed, profile="synthetic-ground-truth")
    graph = envelope.graph
    graph.role = scg.GRAPH_ROLE_GROUND_TRUTH
    _populate_frame_key(
        graph.frame_key,
        specification=specification,
        seed=seed,
        frame_index=frame_index,
    )
    _populate_sensor(graph, specification=specification, frame_index=frame_index)
    lateral_offset = _seed_lateral_offset(seed, specification.kind)
    lane_ids = {
        lane_specification.key: _populate_lane(
            graph,
            specification,
            lane_specification,
            ordinal=ordinal,
            frame_index=frame_index,
            lateral_offset=lateral_offset,
        )
        for ordinal, lane_specification in enumerate(specification.lanes)
    }
    control_ids = {
        control_specification.key: _populate_control(graph, specification, ordinal=ordinal)
        for ordinal, control_specification in enumerate(specification.controls)
    }
    for ordinal in range(len(specification.areas)):
        _populate_area(
            graph,
            specification,
            ordinal=ordinal,
            frame_index=frame_index,
            lateral_offset=lateral_offset,
        )
    for source_key, target_key in specification.successors:
        _add_edge(
            graph,
            scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR,
            lane_ids[source_key],
            lane_ids[target_key],
        )
    for control_specification in specification.controls:
        for target_key in control_specification.applies_to:
            _add_edge(
                graph,
                scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE,
                control_ids[control_specification.key],
                lane_ids[target_key],
            )
    _populate_tracks(graph, frame_index)
    images = _attach_images(envelope)
    validate_envelope(envelope)
    return envelope, images


def _perfect_prediction(
    ground_truth: scg.SceneControlGraphEnvelope,
    *,
    seed: int,
) -> scg.SceneControlGraphEnvelope:
    prediction = deepcopy(ground_truth)
    _populate_producer(
        prediction,
        seed=seed,
        profile="synthetic-perfect-prediction",
        model_artifact_sha256=_sha256_text("junctionlens-synthetic-perfect-model-v1"),
    )
    prediction.graph.role = scg.GRAPH_ROLE_PREDICTION
    identity_map: dict[int, int] = {}
    for node_type, nodes in (
        (scg.NODE_TYPE_LANE_SEGMENT, prediction.graph.lanes),
        (scg.NODE_TYPE_TRAFFIC_CONTROL, prediction.graph.traffic_controls),
        (scg.NODE_TYPE_ROAD_AREA, prediction.graph.road_areas),
    ):
        for ordinal, node in enumerate(nodes):
            old_id = node.node_id
            node.node_id = predicted_node_id(node_type, ordinal)
            node.ClearField("adapter_metadata")
            identity_map[old_id] = node.node_id
    for edge in prediction.graph.edges:
        edge.source_node_id = identity_map[edge.source_node_id]
        edge.target_node_id = identity_map[edge.target_node_id]
        edge.edge_id = edge_id(
            prediction.graph.frame_key,
            edge.edge_type,
            edge.source_node_id,
            edge.target_node_id,
        )
    for track in prediction.graph.tracks:
        track.current_node_id = identity_map[track.current_node_id]
    validate_envelope(prediction)
    return prediction


def generate_scene_frames(seed: int = 20_260_813) -> tuple[GeneratedSceneFrame, ...]:
    """Generate every ordered scene frame and its byte-stable perfect prediction."""
    if seed < 0 or seed >= 1 << 64:
        raise ValueError("synthetic seed must fit an unsigned 64-bit integer")
    result: list[GeneratedSceneFrame] = []
    for specification in scene_specs():
        for frame_index in range(specification.frame_count):
            ground_truth, images = _ground_truth(
                specification,
                seed=seed,
                frame_index=frame_index,
            )
            result.append(
                GeneratedSceneFrame(
                    scene_kind=specification.kind,
                    frame_index=frame_index,
                    ground_truth=ground_truth,
                    perfect_prediction=_perfect_prediction(ground_truth, seed=seed),
                    camera_images=images,
                )
            )
    return tuple(result)


def _replace_edges(graph: scg.SceneControlGraph, retained: list[scg.GraphEdge]) -> None:
    copies: list[scg.GraphEdge] = []
    for edge in retained:
        clone = scg.GraphEdge()
        clone.CopyFrom(edge)
        copies.append(clone)
    del graph.edges[:]
    graph.edges.extend(copies)


def _replace_tracks(graph: scg.SceneControlGraph, retained: list[scg.TemporalTrack]) -> None:
    copies: list[scg.TemporalTrack] = []
    for track in retained:
        clone = scg.TemporalTrack()
        clone.CopyFrom(track)
        copies.append(clone)
    del graph.tracks[:]
    graph.tracks.extend(copies)


def _drop_control(prediction: scg.SceneControlGraphEnvelope) -> None:
    graph = prediction.graph
    if not graph.traffic_controls:
        raise ValueError("drop-control corruption requires a traffic control")
    control = graph.traffic_controls[0]
    control_node_id = control.node_id
    control_track_id = control.track_id
    del graph.traffic_controls[0]
    _replace_edges(
        graph,
        [edge for edge in graph.edges if edge.source_node_id != control_node_id],
    )
    _replace_tracks(
        graph,
        [track for track in graph.tracks if track.track_id != control_track_id],
    )


def _break_topology(prediction: scg.SceneControlGraphEnvelope) -> None:
    graph = prediction.graph
    removed = False
    retained: list[scg.GraphEdge] = []
    for edge in graph.edges:
        if not removed and edge.edge_type == scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR:
            removed = True
            continue
        retained.append(edge)
    if not removed:
        raise ValueError("break-topology corruption requires a lane-successor edge")
    _replace_edges(graph, retained)


def _shift_lane(prediction: scg.SceneControlGraphEnvelope) -> None:
    if not prediction.graph.lanes:
        raise ValueError("shift-lane corruption requires a lane")
    lane = prediction.graph.lanes[0]
    for polyline in (lane.centerline, lane.left_boundary, lane.right_boundary):
        for point in polyline.points:
            point.y += 2.5


def generate_corruptions(
    frames: tuple[GeneratedSceneFrame, ...],
) -> tuple[GeneratedCorruption, ...]:
    """Generate one valid single-fault prediction for each controlled corruption family."""
    plans = (
        (SceneKind.STRAIGHT_CONTROL, CorruptionKind.DROP_CONTROL, _drop_control),
        (SceneKind.MERGE, CorruptionKind.BREAK_TOPOLOGY, _break_topology),
        (SceneKind.INTERSECTION_CROSSWALK, CorruptionKind.SHIFT_LANE, _shift_lane),
    )
    by_scene = {(frame.scene_kind, frame.frame_index): frame for frame in frames}
    result: list[GeneratedCorruption] = []
    for scene_kind, corruption, transform in plans:
        prediction = deepcopy(by_scene[(scene_kind, 0)].perfect_prediction)
        transform(prediction)
        validate_envelope(prediction)
        result.append(
            GeneratedCorruption(
                scene_kind=scene_kind,
                frame_index=0,
                corruption=corruption,
                prediction=prediction,
            )
        )
    return tuple(result)

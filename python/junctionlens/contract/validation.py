"""Semantic and resource validation for the V1 graph contract."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NoReturn, cast

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

from junctionlens.contract.ids import edge_id, predicted_node_type
from junctionlens.contract.limits import (
    CAMERAS_PER_FRAME,
    MAX_ARTIFACTS_PER_FRAME,
    MAX_EDGES_PER_FRAME,
    MAX_LANES_PER_FRAME,
    MAX_POINTS_PER_POLYLINE,
    MAX_ROAD_AREAS_PER_FRAME,
    MAX_STRING_BYTES,
    MAX_TRACKS_PER_FRAME,
    MAX_TRAFFIC_CONTROLS_PER_FRAME,
    MAX_WARNINGS_PER_FRAME,
    SCHEMA_MAJOR,
)
from junctionlens.v1 import scene_control_graph_pb2 as scg

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CONCRETE_SLOTS = tuple(range(scg.CAMERA_SLOT_FRONT_CENTER, scg.CAMERA_SLOT_REAR_RIGHT + 1))


@dataclass(frozen=True, slots=True)
class ContractViolation(ValueError):
    """A stable machine-readable validation failure."""

    reason_code: str
    path: str
    detail: str

    def __str__(self) -> str:
        return f"{self.reason_code} at {self.path}: {self.detail}"


def _fail(reason_code: str, path: str, detail: str) -> NoReturn:
    raise ContractViolation(reason_code, path, detail)


def _walk_scalars(message: Message, path: str) -> None:
    for descriptor, value in message.ListFields():
        child_path = f"{path}.{descriptor.name}"
        if descriptor.is_repeated:
            values = list(value)
            for index, item in enumerate(values):
                item_path = f"{child_path}[{index}]"
                if descriptor.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
                    _walk_scalars(item, item_path)
                else:
                    _validate_scalar(descriptor, item, item_path)
        elif descriptor.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
            _walk_scalars(value, child_path)
        else:
            _validate_scalar(descriptor, value, child_path)


def _validate_scalar(descriptor: FieldDescriptor, value: object, path: str) -> None:
    if descriptor.cpp_type in {
        FieldDescriptor.CPPTYPE_DOUBLE,
        FieldDescriptor.CPPTYPE_FLOAT,
    } and not math.isfinite(cast(float, value)):
        _fail("CONTRACT_NONFINITE", path, "floating-point values must be finite")
    if descriptor.cpp_type == FieldDescriptor.CPPTYPE_STRING:
        encoded = (
            cast(bytes, value)
            if descriptor.type == FieldDescriptor.TYPE_BYTES
            else cast(str, value).encode()
        )
        if len(encoded) > MAX_STRING_BYTES:
            _fail("CONTRACT_STRING_LIMIT", path, f"value exceeds {MAX_STRING_BYTES} bytes")


def _required_text(value: str, path: str) -> None:
    if not value:
        _fail("CONTRACT_REQUIRED", path, "value must not be empty")


def _sha(value: str, path: str, *, required: bool = True) -> None:
    if not value and not required:
        return
    if not _SHA256.fullmatch(value):
        _fail("CONTRACT_SHA256", path, "expected 64 lowercase hexadecimal characters")


def _probability(value: float, path: str) -> None:
    if value < 0.0 or value > 1.0:
        _fail("CONTRACT_PROBABILITY", path, "probability must be within [0, 1]")


def _distribution(distribution: scg.ClassDistribution, path: str) -> None:
    if not distribution.probabilities:
        _fail("CONTRACT_DISTRIBUTION", path, "distribution must not be empty")
    for index, probability in enumerate(distribution.probabilities):
        _probability(probability, f"{path}.probabilities[{index}]")
    if not math.isclose(sum(distribution.probabilities), 1.0, abs_tol=1e-6):
        _fail("CONTRACT_DISTRIBUTION", path, "probabilities must sum to one")


def _matrix(values: object, size: int, path: str) -> list[float]:
    result = [float(value) for value in values]  # type: ignore[attr-defined]
    if len(result) != size:
        _fail("CONTRACT_TRANSFORM_SHAPE", path, f"matrix requires exactly {size} values")
    return result


def _rigid_transform(matrix: scg.Matrix4d, path: str) -> None:
    values = _matrix(matrix.values, 16, path)
    if any(
        not math.isclose(values[12 + index], expected, abs_tol=1e-8)
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        _fail("CONTRACT_TRANSFORM_AFFINE", path, "last row must be [0, 0, 0, 1]")
    rotation = [values[0:3], values[4:7], values[8:11]]
    for row in range(3):
        for column in range(3):
            dot = sum(rotation[row][index] * rotation[column][index] for index in range(3))
            expected = 1.0 if row == column else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-6):
                _fail("CONTRACT_TRANSFORM_RIGID", path, "rotation must be orthonormal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-6):
        _fail("CONTRACT_TRANSFORM_RIGID", path, "rotation determinant must be +1")


def _frame_key(frame: scg.FrameKey, path: str) -> None:
    for field in ("dataset_id", "dataset_version", "split_id", "segment_id"):
        _required_text(str(getattr(frame, field)), f"{path}.{field}")
    if frame.timestamp_ns < 0:
        _fail("CONTRACT_TIMESTAMP", f"{path}.timestamp_ns", "timestamp must be nonnegative")
    if frame.source_domain == scg.SOURCE_DOMAIN_UNSPECIFIED:
        _fail("CONTRACT_ENUM_UNSPECIFIED", f"{path}.source_domain", "source domain is required")
    _sha(frame.calibration_sha256, f"{path}.calibration_sha256")
    _sha(frame.frame_manifest_sha256, f"{path}.frame_manifest_sha256")


def _artifact(artifact: scg.ArtifactRef, path: str) -> None:
    if artifact.kind == scg.ARTIFACT_KIND_UNSPECIFIED:
        _fail("CONTRACT_ENUM_UNSPECIFIED", f"{path}.kind", "artifact kind is required")
    _sha(artifact.sha256, f"{path}.sha256")
    _required_text(artifact.media_type, f"{path}.media_type")
    _required_text(artifact.relative_uri, f"{path}.relative_uri")
    if artifact.relative_uri.startswith(("/", "\\")) or ".." in artifact.relative_uri.split("/"):
        _fail("CONTRACT_ARTIFACT_URI", f"{path}.relative_uri", "URI must be repository-relative")
    _required_text(artifact.license_id, f"{path}.license_id")


def _polyline(polyline: scg.Polyline3d, path: str) -> None:
    if len(polyline.points) < 2:
        _fail("CONTRACT_POLYLINE_POINTS", f"{path}.points", "polyline requires at least two points")
    if len(polyline.points) > MAX_POINTS_PER_POLYLINE:
        _fail("CONTRACT_POINTS_LIMIT", f"{path}.points", "polyline point limit exceeded")
    _probability(polyline.confidence, f"{path}.confidence")
    if polyline.point_uncertainty and len(polyline.point_uncertainty) != len(polyline.points):
        _fail(
            "CONTRACT_UNCERTAINTY_SHAPE",
            f"{path}.point_uncertainty",
            "uncertainty must align with points",
        )
    _scales(polyline.point_uncertainty, f"{path}.point_uncertainty")


def _scales(scales: Iterable[scg.LaplaceScale3d], path: str) -> None:
    for index, scale in enumerate(scales):
        if min(scale.x, scale.y, scale.z) <= 0.0:
            _fail(
                "CONTRACT_UNCERTAINTY_SCALE",
                f"{path}[{index}]",
                "scales must be positive",
            )


def _source_box(box: scg.SourcePixelBox, path: str) -> None:
    if box.convention == scg.SOURCE_BOX_CONVENTION_UNSPECIFIED:
        _fail("CONTRACT_ENUM_UNSPECIFIED", f"{path}.convention", "source convention is required")
    if box.image_width == 0 or box.image_height == 0:
        _fail("CONTRACT_BOX_DIMENSIONS", path, "source image dimensions must be positive")
    if not (
        0.0 <= box.x0 <= box.x1 <= box.image_width and 0.0 <= box.y0 <= box.y1 <= box.image_height
    ):
        _fail("CONTRACT_SOURCE_BOX", path, "source box must be ordered and inside the source image")


def _normalized_box(box: scg.NormalizedBox, path: str) -> None:
    if not (0.0 <= box.x_min < box.x_max <= 1.0 and 0.0 <= box.y_min < box.y_max <= 1.0):
        _fail(
            "CONTRACT_NORMALIZED_BOX", path, "half-open box must have positive area inside [0, 1]"
        )


def _sensor_frame(sensor: scg.SensorFrame, graph_key: scg.FrameKey, path: str) -> None:
    if sensor.frame_key != graph_key:
        _fail(
            "CONTRACT_FRAME_KEY_MISMATCH", f"{path}.frame_key", "sensor and graph frame keys differ"
        )
    slots = tuple(camera.slot for camera in sensor.cameras)
    if len(slots) != CAMERAS_PER_FRAME or slots != _CONCRETE_SLOTS:
        _fail(
            "CONTRACT_CAMERA_SLOTS",
            f"{path}.cameras",
            "exactly one camera in canonical slot order is required",
        )
    _rigid_transform(sensor.t_world_vehicle, f"{path}.t_world_vehicle")
    _required_text(sensor.adapter_version, f"{path}.adapter_version")
    for index, camera in enumerate(sensor.cameras):
        camera_path = f"{path}.cameras[{index}]"
        _matrix(camera.intrinsic.values, 9, f"{camera_path}.intrinsic")
        _rigid_transform(camera.t_vehicle_camera, f"{camera_path}.t_vehicle_camera")
        transform = _matrix(
            camera.image_transform.original_to_model.values,
            9,
            f"{camera_path}.image_transform.original_to_model",
        )
        if not all(
            math.isclose(value, expected, abs_tol=1e-8)
            for value, expected in zip(transform[6:9], (0.0, 0.0, 1.0), strict=True)
        ):
            _fail(
                "CONTRACT_TRANSFORM_AFFINE",
                f"{camera_path}.image_transform",
                "image transform last row must be [0, 0, 1]",
            )
        if camera.distortion_model == scg.DISTORTION_MODEL_UNSPECIFIED:
            _fail(
                "CONTRACT_ENUM_UNSPECIFIED",
                f"{camera_path}.distortion_model",
                "distortion model is required",
            )
        if camera.valid:
            if not camera.HasField("original_image"):
                _fail(
                    "CONTRACT_REQUIRED",
                    f"{camera_path}.original_image",
                    "valid camera requires an image",
                )
            _artifact(camera.original_image, f"{camera_path}.original_image")
            if camera.original_width == 0 or camera.original_height == 0:
                _fail(
                    "CONTRACT_IMAGE_DIMENSIONS",
                    camera_path,
                    "valid camera dimensions must be positive",
                )


def _producer(producer: scg.ProducerInfo, path: str) -> None:
    if not _GIT_COMMIT.fullmatch(producer.git_commit):
        _fail(
            "CONTRACT_GIT_COMMIT", f"{path}.git_commit", "expected a 40-character lowercase Git SHA"
        )
    _sha(producer.model_artifact_sha256, f"{path}.model_artifact_sha256", required=False)
    _sha(producer.configuration_sha256, f"{path}.configuration_sha256")
    _sha(producer.runtime_build_sha256, f"{path}.runtime_build_sha256")
    _required_text(producer.execution_provider_profile, f"{path}.execution_provider_profile")
    _sha(producer.provider_assignment_digest, f"{path}.provider_assignment_digest")


def _validate_nodes(graph: scg.SceneControlGraph) -> dict[int, int]:
    counts = (
        (len(graph.lanes), MAX_LANES_PER_FRAME, "lanes"),
        (len(graph.traffic_controls), MAX_TRAFFIC_CONTROLS_PER_FRAME, "traffic_controls"),
        (len(graph.road_areas), MAX_ROAD_AREAS_PER_FRAME, "road_areas"),
    )
    for count, limit, name in counts:
        if count > limit:
            _fail("CONTRACT_NODE_LIMIT", f"graph.{name}", f"node count exceeds {limit}")
    node_types: dict[int, int] = {}
    collections = (
        (graph.lanes, scg.NODE_TYPE_LANE_SEGMENT, "lanes"),
        (graph.traffic_controls, scg.NODE_TYPE_TRAFFIC_CONTROL, "traffic_controls"),
        (graph.road_areas, scg.NODE_TYPE_ROAD_AREA, "road_areas"),
    )
    for nodes, node_type, name in collections:
        for index, node in enumerate(nodes):
            path = f"graph.{name}[{index}]"
            if node.node_id == 0:
                _fail("CONTRACT_NODE_ID_ZERO", f"{path}.node_id", "node ID must be nonzero")
            if node.node_id in node_types:
                _fail("CONTRACT_NODE_ID_DUPLICATE", f"{path}.node_id", "node IDs are frame-unique")
            if (
                graph.role == scg.GRAPH_ROLE_PREDICTION
                and predicted_node_type(node.node_id) != node_type
            ):
                _fail(
                    "CONTRACT_NODE_ID_TYPE",
                    f"{path}.node_id",
                    "encoded type does not match node type",
                )
            node_types[node.node_id] = node_type
    for index, lane in enumerate(graph.lanes):
        path = f"graph.lanes[{index}]"
        _polyline(lane.centerline, f"{path}.centerline")
        _polyline(lane.left_boundary, f"{path}.left_boundary")
        _polyline(lane.right_boundary, f"{path}.right_boundary")
        _distribution(lane.left_boundary_type, f"{path}.left_boundary_type")
        _distribution(lane.right_boundary_type, f"{path}.right_boundary_type")
        _probability(
            lane.intersection_or_connector_probability,
            f"{path}.intersection_or_connector_probability",
        )
        _probability(lane.existence_confidence, f"{path}.existence_confidence")
        if lane.centerline_laplace_scale_m and len(lane.centerline_laplace_scale_m) != len(
            lane.centerline.points
        ):
            _fail(
                "CONTRACT_UNCERTAINTY_SHAPE",
                f"{path}.centerline_laplace_scale_m",
                "scales must align with centerline",
            )
        _scales(lane.centerline_laplace_scale_m, f"{path}.centerline_laplace_scale_m")
    for index, control in enumerate(graph.traffic_controls):
        path = f"graph.traffic_controls[{index}]"
        if control.source_camera not in _CONCRETE_SLOTS:
            _fail("CONTRACT_ENUM_UNSPECIFIED", f"{path}.source_camera", "source camera is required")
        if control.HasField("source_pixel_box"):
            _source_box(control.source_pixel_box, f"{path}.source_pixel_box")
        _normalized_box(control.normalized_half_open_box, f"{path}.normalized_half_open_box")
        _distribution(control.category_distribution, f"{path}.category_distribution")
        _distribution(control.attribute_distribution, f"{path}.attribute_distribution")
        for field in (
            "existence_confidence",
            "calibrated_class_confidence",
            "calibrated_attribute_confidence",
        ):
            _probability(float(getattr(control, field)), f"{path}.{field}")
    for index, area in enumerate(graph.road_areas):
        path = f"graph.road_areas[{index}]"
        _distribution(area.category_distribution, f"{path}.category_distribution")
        _polyline(area.geometry, f"{path}.geometry")
        _probability(area.existence_confidence, f"{path}.existence_confidence")
        if area.geometry_uncertainty and len(area.geometry_uncertainty) != len(
            area.geometry.points
        ):
            _fail(
                "CONTRACT_UNCERTAINTY_SHAPE",
                f"{path}.geometry_uncertainty",
                "uncertainty must align with geometry",
            )
        _scales(area.geometry_uncertainty, f"{path}.geometry_uncertainty")
    return node_types


def _validate_edges(graph: scg.SceneControlGraph, node_types: dict[int, int]) -> None:
    if len(graph.edges) > MAX_EDGES_PER_FRAME:
        _fail("CONTRACT_EDGE_LIMIT", "graph.edges", "edge count limit exceeded")
    seen: set[int] = set()
    for index, edge in enumerate(graph.edges):
        path = f"graph.edges[{index}]"
        if edge.edge_id == 0:
            _fail("CONTRACT_EDGE_ID_ZERO", f"{path}.edge_id", "edge ID must be nonzero")
        if edge.edge_id in seen:
            _fail("CONTRACT_EDGE_ID_DUPLICATE", f"{path}.edge_id", "edge IDs must be unique")
        seen.add(edge.edge_id)
        if edge.source_node_id not in node_types or edge.target_node_id not in node_types:
            _fail("CONTRACT_EDGE_DANGLING", path, "edge endpoint does not exist")
        types = (node_types[edge.source_node_id], node_types[edge.target_node_id])
        expected_types = {
            scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR: (
                scg.NODE_TYPE_LANE_SEGMENT,
                scg.NODE_TYPE_LANE_SEGMENT,
            ),
            scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE: (
                scg.NODE_TYPE_TRAFFIC_CONTROL,
                scg.NODE_TYPE_LANE_SEGMENT,
            ),
        }.get(edge.edge_type)
        if expected_types is None:
            _fail("CONTRACT_ENUM_UNSPECIFIED", f"{path}.edge_type", "edge type is required")
        if types != expected_types:
            _fail("CONTRACT_EDGE_TYPES", path, "edge endpoint types do not match edge type")
        expected_id = edge_id(
            graph.frame_key, edge.edge_type, edge.source_node_id, edge.target_node_id
        )
        if edge.edge_id != expected_id:
            _fail(
                "CONTRACT_EDGE_ID",
                f"{path}.edge_id",
                "edge ID does not match its deterministic identity",
            )
        _probability(edge.raw_probability, f"{path}.raw_probability")
        _probability(edge.calibrated_probability, f"{path}.calibrated_probability")
        if edge.HasField("uncertainty") and edge.uncertainty.standard_deviation < 0.0:
            _fail(
                "CONTRACT_UNCERTAINTY_SCALE",
                f"{path}.uncertainty",
                "standard deviation must be nonnegative",
            )


def _validate_tracks(graph: scg.SceneControlGraph, node_types: dict[int, int]) -> None:
    if len(graph.tracks) > MAX_TRACKS_PER_FRAME:
        _fail("CONTRACT_TRACK_LIMIT", "graph.tracks", "track count limit exceeded")
    seen: set[int] = set()
    tracks_by_id: dict[int, scg.TemporalTrack] = {}
    for index, track in enumerate(graph.tracks):
        path = f"graph.tracks[{index}]"
        if track.track_id == 0:
            _fail("CONTRACT_TRACK_ID_ZERO", f"{path}.track_id", "track ID must be nonzero")
        if track.track_id in seen:
            _fail("CONTRACT_TRACK_ID_DUPLICATE", f"{path}.track_id", "track IDs must be unique")
        seen.add(track.track_id)
        tracks_by_id[track.track_id] = track
        if track.current_node_id not in node_types:
            _fail(
                "CONTRACT_TRACK_DANGLING", f"{path}.current_node_id", "current node does not exist"
            )
        if node_types[track.current_node_id] != track.node_type:
            _fail("CONTRACT_TRACK_NODE_TYPE", path, "track and current node types differ")
        if (
            track.node_type == scg.NODE_TYPE_UNSPECIFIED
            or track.termination_reason == scg.TRACK_TERMINATION_REASON_UNSPECIFIED
        ):
            _fail(
                "CONTRACT_ENUM_UNSPECIFIED", path, "track type and termination reason are required"
            )
        if track.first_timestamp_ns > track.last_timestamp_ns:
            _fail("CONTRACT_TRACK_TIMESTAMPS", path, "first timestamp follows last timestamp")
        if track.age_observed_frames == 0:
            _fail(
                "CONTRACT_TRACK_AGE", f"{path}.age_observed_frames", "observed age must be positive"
            )
    referenced = {
        int(node.track_id)
        for nodes in (graph.lanes, graph.traffic_controls, graph.road_areas)
        for node in nodes
        if node.HasField("track_id")
    }
    if not referenced.issubset(seen):
        _fail("CONTRACT_TRACK_REFERENCE", "graph", "node references a missing track")
    for nodes in (graph.lanes, graph.traffic_controls, graph.road_areas):
        for node in nodes:
            if (
                node.HasField("track_id")
                and tracks_by_id[node.track_id].current_node_id != node.node_id
            ):
                _fail(
                    "CONTRACT_TRACK_REFERENCE",
                    "graph",
                    "node track reference does not identify that current node",
                )


def validate_envelope(envelope: scg.SceneControlGraphEnvelope) -> None:
    """Validate one complete envelope or raise a stable violation."""
    _walk_scalars(envelope, "envelope")
    if envelope.schema_major != SCHEMA_MAJOR:
        _fail("CONTRACT_SCHEMA_MAJOR", "schema_major", f"expected {SCHEMA_MAJOR}")
    _producer(envelope.producer, "producer")
    graph = envelope.graph
    if graph.role == scg.GRAPH_ROLE_UNSPECIFIED:
        _fail("CONTRACT_ENUM_UNSPECIFIED", "graph.role", "graph role is required")
    _frame_key(graph.frame_key, "graph.frame_key")
    if graph.HasField("sensor_frame"):
        _sensor_frame(graph.sensor_frame, graph.frame_key, "graph.sensor_frame")
    if len(graph.raw_tensor_artifacts) > MAX_ARTIFACTS_PER_FRAME:
        _fail(
            "CONTRACT_ARTIFACT_LIMIT", "graph.raw_tensor_artifacts", "artifact count limit exceeded"
        )
    for index, artifact in enumerate(graph.raw_tensor_artifacts):
        _artifact(artifact, f"graph.raw_tensor_artifacts[{index}]")
    if len(graph.warnings) > MAX_WARNINGS_PER_FRAME:
        _fail("CONTRACT_WARNING_LIMIT", "graph.warnings", "warning count limit exceeded")
    node_types = _validate_nodes(graph)
    _validate_edges(graph, node_types)
    _validate_tracks(graph, node_types)

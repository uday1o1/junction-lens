"""Lossless OpenLane frame conversion for private model-selection evidence."""

from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

from junctionlens.contract.ids import edge_id
from junctionlens.contract.validation import validate_envelope
from junctionlens.data.contracts import AdaptedFrame, CameraFrame, CameraSlot
from junctionlens.v1 import scene_control_graph_pb2 as scg

FloatArray = npt.NDArray[np.float32]

_CAMERA_ENUM = {
    CameraSlot.FRONT_CENTER: scg.CAMERA_SLOT_FRONT_CENTER,
    CameraSlot.FRONT_LEFT: scg.CAMERA_SLOT_FRONT_LEFT,
    CameraSlot.FRONT_RIGHT: scg.CAMERA_SLOT_FRONT_RIGHT,
    CameraSlot.SIDE_LEFT: scg.CAMERA_SLOT_SIDE_LEFT,
    CameraSlot.SIDE_RIGHT: scg.CAMERA_SLOT_SIDE_RIGHT,
    CameraSlot.REAR_LEFT: scg.CAMERA_SLOT_REAR_LEFT,
    CameraSlot.REAR_CENTER: scg.CAMERA_SLOT_REAR_CENTER,
    CameraSlot.REAR_RIGHT: scg.CAMERA_SLOT_REAR_RIGHT,
}


class EvaluationGraphError(ValueError):
    """Raised when a private evaluation graph cannot preserve source identity."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _domain_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_node_id(frame: AdaptedFrame, node_type: str, source_id: str) -> int:
    digest = hashlib.sha256()
    for value in (
        "junctionlens-source-node-v1",
        frame.key.dataset_id,
        frame.key.segment_id,
        str(frame.key.timestamp_ns),
        node_type,
        source_id,
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") or 1


def _fill_frame_key(destination: scg.FrameKey, frame: AdaptedFrame) -> None:
    destination.dataset_id = frame.key.dataset_id
    destination.dataset_version = frame.key.dataset_version
    destination.split_id = frame.key.split_id
    destination.segment_id = frame.key.segment_id
    destination.timestamp_ns = frame.key.timestamp_ns
    destination.source_domain = scg.SOURCE_DOMAIN_OPENLANE_V2
    destination.calibration_sha256 = frame.key.calibration_sha256
    destination.frame_manifest_sha256 = frame.key.frame_manifest_sha256


def _distortion_model(value: str) -> scg.DistortionModel:
    normalized = value.strip().upper().replace("-", "_")
    if normalized in {"NONE", "PINHOLE", "UPSTREAM_UNSPECIFIED"}:
        return scg.DISTORTION_MODEL_NONE
    if "FISH" in normalized:
        return scg.DISTORTION_MODEL_FISHEYE
    return scg.DISTORTION_MODEL_BROWN_CONRADY


def _fill_camera(
    destination: scg.CameraFrame,
    camera: CameraFrame,
    dataset_root: Path,
) -> None:
    destination.slot = _CAMERA_ENUM[camera.slot]
    destination.valid = camera.valid
    destination.capture_timestamp_ns = camera.capture_timestamp_ns
    destination.original_width = camera.original_width
    destination.original_height = camera.original_height
    destination.intrinsic.values.extend(value for row in camera.intrinsic for value in row)
    destination.t_vehicle_camera.values.extend(
        value for row in camera.t_vehicle_camera for value in row
    )
    destination.distortion_model = _distortion_model(camera.distortion_model)
    destination.distortion_coefficients.extend(camera.distortion_coefficients)
    transform = camera.image_transform
    destination.image_transform.original_to_model.values.extend(
        value for row in transform for value in row
    )
    destination.image_transform.resized_width = max(
        1, round(camera.original_width * transform[0][0])
    )
    destination.image_transform.resized_height = max(
        1, round(camera.original_height * transform[1][1])
    )
    destination.image_transform.pad_left = max(0, round(transform[0][2]))
    destination.image_transform.pad_top = max(0, round(transform[1][2]))
    if not camera.valid:
        return
    if camera.image_relative_path is None:
        raise EvaluationGraphError("valid camera has no source image path")
    relative = Path(camera.image_relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise EvaluationGraphError("camera image path is unsafe")
    path = (dataset_root / relative).resolve(strict=True)
    try:
        path.relative_to(dataset_root)
    except ValueError as error:
        raise EvaluationGraphError("camera image escapes the dataset root") from error
    if path.is_symlink() or not path.is_file():
        raise EvaluationGraphError("camera image must be a regular file")
    artifact = destination.original_image
    artifact.kind = scg.ARTIFACT_KIND_SOURCE_IMAGE
    artifact.sha256 = _hash_file(path)
    artifact.byte_size = path.stat().st_size
    artifact.media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    artifact.relative_uri = relative.as_posix()
    artifact.license_id = "CC-BY-NC-SA-4.0-and-source-dataset-terms"


def sensor_frame(frame: AdaptedFrame, dataset_root: Path) -> scg.SensorFrame:
    """Convert one adapted frame into the canonical private sensor contract."""
    root = dataset_root.resolve(strict=True)
    result = scg.SensorFrame()
    _fill_frame_key(result.frame_key, frame)
    for camera in frame.cameras:
        _fill_camera(result.cameras.add(), camera, root)
    result.t_world_vehicle.values.extend(value for row in frame.t_world_vehicle for value in row)
    result.pose_valid = frame.pose_valid
    result.adapter_version = frame.adapter_version
    return result


def producer_info(
    *,
    source_commit: str,
    model_sha256: str,
    configuration_sha256: str,
    provider_profile: str,
    seed: int,
) -> scg.ProducerInfo:
    """Create a complete producer identity without machine-local values."""
    result = scg.ProducerInfo(
        git_commit=source_commit,
        model_artifact_sha256=model_sha256,
        configuration_sha256=configuration_sha256,
        runtime_build_sha256=_domain_hash(f"junctionlens-python-runtime:{source_commit}"),
        execution_provider_profile=provider_profile,
        provider_assignment_digest=_domain_hash(provider_profile),
        random_seed=seed,
    )
    return result


def _one_hot(size: int, selected: int) -> tuple[float, ...]:
    if not 0 <= selected < size:
        raise EvaluationGraphError("class index exceeds its frozen distribution")
    return tuple(1.0 if index == selected else 0.0 for index in range(size))


def _add_points(
    destination: scg.Polyline3d, points: tuple[tuple[float, float, float], ...]
) -> None:
    destination.confidence = 1.0
    for x, y, z in points:
        destination.points.add(x=x, y=y, z=z)


def ground_truth_envelope(
    frame: AdaptedFrame,
    dataset_root: Path,
    *,
    source_commit: str,
    configuration_sha256: str,
) -> scg.SceneControlGraphEnvelope:
    """Build a validated ground-truth graph without changing source geometry or IDs."""
    envelope = scg.SceneControlGraphEnvelope(schema_major=1, schema_minor=0)
    envelope.producer.CopyFrom(
        producer_info(
            source_commit=source_commit,
            model_sha256="",
            configuration_sha256=configuration_sha256,
            provider_profile="openlane-adapter-ground-truth",
            seed=0,
        )
    )
    graph = envelope.graph
    graph.role = scg.GRAPH_ROLE_GROUND_TRUTH
    _fill_frame_key(graph.frame_key, frame)
    graph.sensor_frame.CopyFrom(sensor_frame(frame, dataset_root))
    lane_ids: list[int] = []
    control_ids: list[int] = []
    for lane_source in frame.lanes:
        node_id = _source_node_id(frame, "lane", lane_source.source_object_id)
        lane_ids.append(node_id)
        lane = graph.lanes.add(
            node_id=node_id,
            intersection_or_connector_probability=float(lane_source.is_intersection_or_connector),
            existence_confidence=1.0,
        )
        lane.adapter_metadata.source_object_id = lane_source.source_object_id
        lane.adapter_metadata.source_namespace = "openlane-v2"
        _add_points(lane.centerline, lane_source.centerline)
        _add_points(lane.left_boundary, lane_source.left_boundary)
        _add_points(lane.right_boundary, lane_source.right_boundary)
        lane.left_boundary_type.probabilities.extend(_one_hot(3, lane_source.left_boundary_type))
        lane.right_boundary_type.probabilities.extend(_one_hot(3, lane_source.right_boundary_type))
    for control_source in frame.traffic_controls:
        node_id = _source_node_id(frame, "control", control_source.source_object_id)
        control_ids.append(node_id)
        control = graph.traffic_controls.add(
            node_id=node_id,
            source_camera=_CAMERA_ENUM[control_source.source_camera],
            existence_confidence=1.0,
            calibrated_class_confidence=1.0,
            calibrated_attribute_confidence=1.0,
        )
        control.adapter_metadata.source_object_id = control_source.source_object_id
        control.adapter_metadata.source_namespace = "openlane-v2"
        pixel_source = control_source.source_pixel_box
        control.source_pixel_box.x0, control.source_pixel_box.y0 = pixel_source.points[0]
        control.source_pixel_box.x1, control.source_pixel_box.y1 = pixel_source.points[1]
        control.source_pixel_box.convention = scg.SOURCE_BOX_CONVENTION_TWO_CORNERS
        control.source_pixel_box.image_width = pixel_source.image_width
        control.source_pixel_box.image_height = pixel_source.image_height
        (
            control.normalized_half_open_box.x_min,
            control.normalized_half_open_box.y_min,
            control.normalized_half_open_box.x_max,
            control.normalized_half_open_box.y_max,
        ) = control_source.normalized_half_open_box
        control.category_distribution.probabilities.extend(_one_hot(2, control_source.category - 1))
        control.attribute_distribution.probabilities.extend(_one_hot(13, control_source.attribute))
    for area_source in frame.road_areas:
        area = graph.road_areas.add(
            node_id=_source_node_id(frame, "area", area_source.source_object_id),
            existence_confidence=1.0,
        )
        area.adapter_metadata.source_object_id = area_source.source_object_id
        area.adapter_metadata.source_namespace = "openlane-v2"
        area.category_distribution.probabilities.extend(_one_hot(2, area_source.category - 1))
        _add_points(area.geometry, area_source.points)
    edges: list[tuple[int, int, int]] = []
    for source_index, row in enumerate(frame.topology_lane_lane):
        for target_index, value in enumerate(row):
            if value:
                edges.append(
                    (
                        scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR,
                        lane_ids[source_index],
                        lane_ids[target_index],
                    )
                )
    for lane_index, row in enumerate(frame.topology_lane_traffic):
        for control_index, value in enumerate(row):
            if value:
                edges.append(
                    (
                        scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE,
                        control_ids[control_index],
                        lane_ids[lane_index],
                    )
                )
    for edge_type, edge_source, edge_target in sorted(edges):
        item = graph.edges.add(
            edge_type=edge_type,
            source_node_id=edge_source,
            target_node_id=edge_target,
            raw_probability=1.0,
            calibrated_probability=1.0,
            binary_decision=True,
        )
        item.edge_id = edge_id(graph.frame_key, edge_type, edge_source, edge_target)
    validate_envelope(envelope)
    return envelope


def numpy_outputs(outputs: Mapping[str, object]) -> dict[str, FloatArray]:
    """Copy a tensor-like output mapping into finite float32 NumPy arrays."""
    result: dict[str, FloatArray] = {}
    for name, value in outputs.items():
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise EvaluationGraphError(f"model output {name} contains a nonfinite value")
        result[name] = array
    return result


__all__ = [
    "EvaluationGraphError",
    "ground_truth_envelope",
    "numpy_outputs",
    "producer_info",
    "sensor_frame",
]

"""Strict OpenLane-V2 v2.1.0 lane-segment source adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from PIL import Image

from junctionlens.data.contracts import (
    CAMERA_SLOTS,
    AdaptedFrame,
    CameraFrame,
    CameraSlot,
    FrameKey,
    LaneSegment,
    Matrix3,
    Matrix4,
    Point2,
    Point3,
    RoadArea,
    SourcePixelBox,
    TrafficControl,
)
from junctionlens.data.geometry import (
    FloatArray,
    GeometryError,
    letterbox_transform,
    normalize_half_open_box,
    openlane_camera_to_junctionlens_vehicle,
    openlane_points_to_junctionlens,
    openlane_pose_to_junctionlens_world,
)

_MAX_METADATA_BYTES = 16 * 1024 * 1024
_IDENTITY_3 = np.eye(3, dtype=np.float64)
_IDENTITY_4 = np.eye(4, dtype=np.float64)


class OpenLaneAdapterError(ValueError):
    """Raised for malformed, ambiguous, or unsafe OpenLane source data."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(payload.encode("utf-8"))


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise OpenLaneAdapterError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _as_sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise OpenLaneAdapterError(f"{label} must be an array")
    return cast(Sequence[Any], value)


def _required(value: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in value or value[key] is None:
        raise OpenLaneAdapterError(f"{label}.{key} is required")
    return value[key]


def _required_int(value: Mapping[str, Any], key: str, label: str) -> int:
    raw = _required(value, key, label)
    if isinstance(raw, bool):
        raise OpenLaneAdapterError(f"{label}.{key} must be an integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as error:
        raise OpenLaneAdapterError(f"{label}.{key} must be an integer") from error


def _matrix3_tuple(value: FloatArray) -> Matrix3:
    return (
        (float(value[0, 0]), float(value[0, 1]), float(value[0, 2])),
        (float(value[1, 0]), float(value[1, 1]), float(value[1, 2])),
        (float(value[2, 0]), float(value[2, 1]), float(value[2, 2])),
    )


def _matrix4_tuple(value: FloatArray) -> Matrix4:
    return (
        (float(value[0, 0]), float(value[0, 1]), float(value[0, 2]), float(value[0, 3])),
        (float(value[1, 0]), float(value[1, 1]), float(value[1, 2]), float(value[1, 3])),
        (float(value[2, 0]), float(value[2, 1]), float(value[2, 2]), float(value[2, 3])),
        (float(value[3, 0]), float(value[3, 1]), float(value[3, 2]), float(value[3, 3])),
    )


def _point3_tuple(value: object, label: str) -> tuple[Point3, ...]:
    try:
        transformed = openlane_points_to_junctionlens(value)
    except (GeometryError, TypeError, ValueError) as error:
        raise OpenLaneAdapterError(f"{label}: {error}") from error
    if len(transformed) < 2:
        raise OpenLaneAdapterError(f"{label} must contain at least two points")
    return tuple((float(point[0]), float(point[1]), float(point[2])) for point in transformed)


def _topology(
    value: object,
    rows: int,
    columns: int,
    label: str,
) -> tuple[tuple[int, ...], ...]:
    raw_rows = _as_sequence(value, label)
    if len(raw_rows) != rows:
        raise OpenLaneAdapterError(f"{label} must contain {rows} rows")
    result: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(raw_rows):
        values = _as_sequence(raw_row, f"{label}[{row_index}]")
        if len(values) != columns:
            raise OpenLaneAdapterError(f"{label}[{row_index}] must contain {columns} columns")
        row = tuple(int(item) for item in values)
        if any(item not in {0, 1} for item in row):
            raise OpenLaneAdapterError(f"{label} ground-truth entries must be zero or one")
        result.append(row)
    return tuple(result)


class OpenLaneAdapter:
    """Read deterministic raw `-ls.json` frames without importing the pickle devkit."""

    def __init__(self, root: Path, config_path: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)
        with config_path.open(encoding="utf-8") as source:
            raw_config = yaml.safe_load(source)
        self.config = _as_mapping(raw_config, "adapter config")
        self.adapter_version = str(self.config["adapter_version"])
        raw_mappings = _as_mapping(self.config["source_camera_mappings"], "camera mappings")
        self.camera_mappings = {
            str(domain): {
                str(source): CameraSlot(str(target))
                for source, target in _as_mapping(mapping, f"camera mappings.{domain}").items()
            }
            for domain, mapping in raw_mappings.items()
        }

    def iter_identifiers(self, profile: str) -> Iterator[tuple[str, str, str]]:
        """Yield source identifiers in stable split, segment, and timestamp order."""
        manifest_name = {
            "sample": "data_dict_example.json",
            "full": "data_dict_subset_A.json",
        }.get(profile)
        if manifest_name is None:
            raise OpenLaneAdapterError(f"unsupported OpenLane profile: {profile}")
        manifest = self._load_json(self.root / manifest_name, f"{profile} manifest")
        for split_id in sorted(manifest):
            segments = _as_mapping(manifest[split_id], f"manifest.{split_id}")
            for segment_id in sorted(segments):
                timestamps = _as_sequence(segments[segment_id], f"manifest.{split_id}.{segment_id}")
                normalized = sorted(
                    (Path(str(timestamp)).stem for timestamp in timestamps),
                    key=lambda value: int(value),
                )
                if len(normalized) != len(set(normalized)):
                    raise OpenLaneAdapterError(
                        f"manifest.{split_id}.{segment_id} contains duplicate timestamps"
                    )
                for timestamp in normalized:
                    yield str(split_id), str(segment_id), timestamp

    def iter_frames(self, profile: str) -> Iterator[AdaptedFrame]:
        """Adapt every declared profile frame without loading all metadata at once."""
        for identifier in self.iter_identifiers(profile):
            yield self.load_frame(*identifier)

    def load_frame(self, split_id: str, segment_id: str, timestamp: str) -> AdaptedFrame:
        """Load and validate one raw lane-segment metadata frame."""
        info_root = self._safe_path(Path(split_id) / segment_id / "info", require_file=False)
        lane_segment_path = info_root / f"{timestamp}-ls.json"
        if not lane_segment_path.is_file():
            raise OpenLaneAdapterError(
                f"missing lane-segment metadata {lane_segment_path}; install the Map Element Bucket"
            )
        raw_bytes = lane_segment_path.read_bytes()
        if len(raw_bytes) > _MAX_METADATA_BYTES:
            raise OpenLaneAdapterError(f"metadata file exceeds {_MAX_METADATA_BYTES} bytes")
        try:
            raw_value = json.loads(raw_bytes)
        except json.JSONDecodeError as error:
            raise OpenLaneAdapterError(f"invalid JSON in {lane_segment_path}: {error}") from error
        metadata = _as_mapping(raw_value, "frame metadata")
        declared_segment = str(metadata.get("segment_id", segment_id))
        declared_timestamp = int(metadata.get("timestamp", timestamp))
        if declared_segment != segment_id or declared_timestamp != int(timestamp):
            raise OpenLaneAdapterError("manifest and frame identity disagree")

        meta_data = _as_mapping(metadata.get("meta_data", {}), "meta_data")
        source_domain = str(meta_data.get("source", "unknown")).lower()
        cameras = self._adapt_cameras(metadata, source_domain, declared_timestamp)
        calibration_sha256 = _canonical_sha256(
            [
                {
                    "slot": camera.slot.value,
                    "valid": camera.valid,
                    "intrinsic": camera.intrinsic,
                    "t_vehicle_camera": camera.t_vehicle_camera,
                }
                for camera in cameras
            ]
        )
        pose, pose_valid = self._adapt_pose(metadata)
        lanes, controls, areas, topology_ll, topology_lt = self._adapt_annotations(
            metadata, cameras
        )
        return AdaptedFrame(
            key=FrameKey(
                dataset_id="openlane-v2-v2.1",
                dataset_version="2.1",
                split_id=split_id,
                segment_id=segment_id,
                timestamp_ns=declared_timestamp,
                source_domain=source_domain,
                calibration_sha256=calibration_sha256,
                frame_manifest_sha256=_sha256_bytes(raw_bytes),
            ),
            cameras=cameras,
            t_world_vehicle=_matrix4_tuple(pose),
            pose_valid=pose_valid,
            adapter_version=self.adapter_version,
            lanes=lanes,
            traffic_controls=controls,
            road_areas=areas,
            topology_lane_lane=topology_ll,
            topology_lane_traffic=topology_lt,
        )

    def _load_json(self, path: Path, label: str) -> Mapping[str, Any]:
        safe_path = self._safe_path(path.relative_to(self.root))
        if safe_path.stat().st_size > _MAX_METADATA_BYTES:
            raise OpenLaneAdapterError(f"{label} exceeds {_MAX_METADATA_BYTES} bytes")
        try:
            with safe_path.open(encoding="utf-8") as source:
                value = json.load(source)
        except json.JSONDecodeError as error:
            raise OpenLaneAdapterError(f"invalid {label}: {error}") from error
        return _as_mapping(value, label)

    def _safe_path(self, relative: Path, *, require_file: bool = True) -> Path:
        if relative.is_absolute() or ".." in relative.parts:
            raise OpenLaneAdapterError(f"unsafe dataset path: {relative}")
        candidate = (self.root / relative).resolve(strict=True)
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise OpenLaneAdapterError(f"dataset path escapes root: {relative}") from error
        if require_file and not candidate.is_file():
            raise OpenLaneAdapterError(f"dataset path is not a file: {relative}")
        return candidate

    def _domain_mapping(
        self, source_domain: str, source_cameras: Sequence[str]
    ) -> Mapping[str, CameraSlot]:
        normalized = {
            "argoverse": "argoverse2",
            "argoverse_v2": "argoverse2",
            "nuscenes": "nuscenes",
        }.get(source_domain, source_domain)
        if normalized not in self.camera_mappings:
            if all(camera.startswith("CAM_") for camera in source_cameras):
                normalized = "nuscenes"
            elif all(camera.startswith("ring_") for camera in source_cameras):
                normalized = "argoverse2"
            else:
                raise OpenLaneAdapterError(
                    f"cannot select a camera mapping for source domain {source_domain!r}"
                )
        return self.camera_mappings[normalized]

    def _adapt_cameras(
        self,
        metadata: Mapping[str, Any],
        source_domain: str,
        timestamp_ns: int,
    ) -> tuple[CameraFrame, ...]:
        sensors = _as_mapping(metadata.get("sensor"), "sensor")
        source_cameras = [str(name) for name in sensors if name != "sd_map"]
        mapping = self._domain_mapping(source_domain, source_cameras)
        by_slot: dict[CameraSlot, CameraFrame] = {}
        for source_camera in source_cameras:
            if source_camera not in mapping:
                raise OpenLaneAdapterError(f"unmapped source camera: {source_camera}")
            slot = mapping[source_camera]
            if slot in by_slot:
                raise OpenLaneAdapterError(f"multiple source cameras map to {slot.value}")
            sensor = _as_mapping(sensors[source_camera], f"sensor.{source_camera}")
            intrinsic = _as_mapping(sensor.get("intrinsic"), f"sensor.{source_camera}.intrinsic")
            extrinsic = _as_mapping(sensor.get("extrinsic"), f"sensor.{source_camera}.extrinsic")
            intrinsic_array = np.asarray(intrinsic.get("K"), dtype=np.float64)
            if intrinsic_array.shape != (3, 3) or not np.isfinite(intrinsic_array).all():
                raise OpenLaneAdapterError(
                    f"sensor.{source_camera}.intrinsic.K is not finite 3 by 3"
                )
            try:
                transform = openlane_camera_to_junctionlens_vehicle(
                    extrinsic.get("rotation"), extrinsic.get("translation")
                )
            except (GeometryError, TypeError, ValueError) as error:
                raise OpenLaneAdapterError(f"sensor.{source_camera}.extrinsic: {error}") from error
            image_relative = Path(str(sensor.get("image_path", "")))
            image_path = self._safe_path(image_relative)
            declared_width = sensor.get("image_width")
            declared_height = sensor.get("image_height")
            if declared_width is None or declared_height is None:
                with Image.open(image_path) as image:
                    width, height = image.size
            else:
                width, height = int(declared_width), int(declared_height)
            if width <= 0 or height <= 0:
                raise OpenLaneAdapterError(f"sensor.{source_camera} has invalid image dimensions")
            distortion_raw = intrinsic.get("distortion", [])
            distortion = tuple(float(value) for value in _as_sequence(distortion_raw, "distortion"))
            if not all(np.isfinite(distortion)):
                raise OpenLaneAdapterError(
                    f"sensor.{source_camera}.distortion contains nonfinite values"
                )
            by_slot[slot] = CameraFrame(
                slot=slot,
                valid=True,
                source_camera=source_camera,
                image_relative_path=image_relative.as_posix(),
                capture_timestamp_ns=timestamp_ns,
                original_width=width,
                original_height=height,
                intrinsic=_matrix3_tuple(intrinsic_array),
                t_vehicle_camera=_matrix4_tuple(transform),
                distortion_model=str(intrinsic.get("model", "upstream-unspecified")),
                distortion_coefficients=distortion,
                image_transform=_matrix3_tuple(letterbox_transform(width, height)),
            )
        return tuple(
            by_slot.get(
                slot,
                CameraFrame(
                    slot=slot,
                    valid=False,
                    source_camera=None,
                    image_relative_path=None,
                    capture_timestamp_ns=timestamp_ns,
                    original_width=0,
                    original_height=0,
                    intrinsic=_matrix3_tuple(_IDENTITY_3),
                    t_vehicle_camera=_matrix4_tuple(_IDENTITY_4),
                    distortion_model="NONE",
                    distortion_coefficients=(),
                    image_transform=_matrix3_tuple(_IDENTITY_3),
                ),
            )
            for slot in CAMERA_SLOTS
        )

    def _adapt_pose(self, metadata: Mapping[str, Any]) -> tuple[FloatArray, bool]:
        raw_pose = metadata.get("pose")
        if raw_pose is None:
            return _IDENTITY_4.copy(), False
        pose = _as_mapping(raw_pose, "pose")
        try:
            return (
                openlane_pose_to_junctionlens_world(
                    pose.get("rotation"),
                    pose.get("translation"),
                ),
                True,
            )
        except (GeometryError, TypeError, ValueError) as error:
            raise OpenLaneAdapterError(f"pose: {error}") from error

    def _adapt_annotations(
        self,
        metadata: Mapping[str, Any],
        cameras: tuple[CameraFrame, ...],
    ) -> tuple[
        tuple[LaneSegment, ...],
        tuple[TrafficControl, ...],
        tuple[RoadArea, ...],
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
    ]:
        annotation = _as_mapping(metadata.get("annotation"), "annotation")
        raw_lanes = _as_sequence(annotation.get("lane_segment"), "annotation.lane_segment")
        lanes = tuple(self._adapt_lane(raw, index) for index, raw in enumerate(raw_lanes))
        raw_controls = _as_sequence(annotation.get("traffic_element"), "annotation.traffic_element")
        front = cameras[CAMERA_SLOTS.index(CameraSlot.FRONT_CENTER)]
        if raw_controls and not front.valid:
            raise OpenLaneAdapterError("traffic elements require a valid FRONT_CENTER camera")
        controls = tuple(
            self._adapt_control(raw, index, front) for index, raw in enumerate(raw_controls)
        )
        raw_areas = _as_sequence(annotation.get("area"), "annotation.area")
        areas = tuple(self._adapt_area(raw, index) for index, raw in enumerate(raw_areas))
        topology_ll = _topology(
            annotation.get("topology_lsls"), len(lanes), len(lanes), "annotation.topology_lsls"
        )
        topology_lt = _topology(
            annotation.get("topology_lste"),
            len(lanes),
            len(controls),
            "annotation.topology_lste",
        )
        return lanes, controls, areas, topology_ll, topology_lt

    def _adapt_lane(self, raw: object, index: int) -> LaneSegment:
        lane = _as_mapping(raw, f"lane_segment[{index}]")
        label = f"lane_segment[{index}]"
        return LaneSegment(
            source_object_id=str(_required(lane, "id", label)),
            centerline=_point3_tuple(lane.get("centerline"), f"lane_segment[{index}].centerline"),
            left_boundary=_point3_tuple(
                lane.get("left_laneline"), f"lane_segment[{index}].left_laneline"
            ),
            right_boundary=_point3_tuple(
                lane.get("right_laneline"), f"lane_segment[{index}].right_laneline"
            ),
            left_boundary_type=_required_int(lane, "left_laneline_type", label),
            right_boundary_type=_required_int(lane, "right_laneline_type", label),
            is_intersection_or_connector=bool(lane.get("is_intersection_or_connector")),
        )

    def _adapt_control(
        self,
        raw: object,
        index: int,
        front: CameraFrame,
    ) -> TrafficControl:
        control = _as_mapping(raw, f"traffic_element[{index}]")
        label = f"traffic_element[{index}]"
        points_array = np.asarray(control.get("points"), dtype=np.float64)
        if points_array.shape != (2, 2) or not np.isfinite(points_array).all():
            raise OpenLaneAdapterError(f"traffic_element[{index}].points must be finite 2 by 2")
        points: tuple[Point2, Point2] = (
            (float(points_array[0, 0]), float(points_array[0, 1])),
            (float(points_array[1, 0]), float(points_array[1, 1])),
        )
        normalized = normalize_half_open_box(
            (
                points[0][0],
                points[0][1],
                points[1][0],
                points[1][1],
            ),
            front.original_width,
            front.original_height,
        )
        return TrafficControl(
            source_object_id=str(_required(control, "id", label)),
            category=_required_int(control, "category", label),
            attribute=_required_int(control, "attribute", label),
            source_camera=CameraSlot.FRONT_CENTER,
            source_pixel_box=SourcePixelBox(
                points=points,
                convention="OPENLANE_V2_TOP_LEFT_BOTTOM_RIGHT_SOURCE_NUMERIC",
                image_width=front.original_width,
                image_height=front.original_height,
            ),
            normalized_half_open_box=normalized,
        )

    def _adapt_area(self, raw: object, index: int) -> RoadArea:
        area = _as_mapping(raw, f"area[{index}]")
        label = f"area[{index}]"
        return RoadArea(
            source_object_id=str(_required(area, "id", label)),
            category=_required_int(area, "category", label),
            points=_point3_tuple(area.get("points"), f"area[{index}].points"),
        )

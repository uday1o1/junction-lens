"""Strict OpenLane-V2 v2.1.0 lane-segment source adapter."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
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
    ModelCameraInputs,
    Point2,
    Point3,
    RoadArea,
    SourceDomainMetadata,
    SourcePixelBox,
    TrafficControl,
)
from junctionlens.data.geometry import (
    FloatArray,
    GeometryError,
    image_transform,
    normalize_half_open_box,
    openlane_camera_to_junctionlens_vehicle,
    openlane_points_to_junctionlens,
    openlane_pose_to_junctionlens_world,
)
from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object,
    load_json_object_path,
    load_yaml_object_path,
    read_bounded_file,
)

_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_IDENTITY_3 = np.eye(3, dtype=np.float64)
_IDENTITY_4 = np.eye(4, dtype=np.float64)
_ALLOWED_BOUNDARY_TYPES = frozenset({0, 1, 2})
_ALLOWED_TRAFFIC_CATEGORIES = frozenset({1, 2})
_ALLOWED_TRAFFIC_ATTRIBUTES = frozenset(range(13))
_ALLOWED_AREA_CATEGORIES = frozenset({1, 2})


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


def _bounded_category(
    value: Mapping[str, Any], key: str, label: str, allowed: frozenset[int]
) -> int:
    result = _required_int(value, key, label)
    if result not in allowed:
        raise OpenLaneAdapterError(
            f"{label}.{key} must be one of {sorted(allowed)}; observed {result}"
        )
    return result


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


def _point3_tuple(value: object, label: str, *, minimum_points: int = 2) -> tuple[Point3, ...]:
    try:
        transformed = openlane_points_to_junctionlens(value)
    except (GeometryError, TypeError, ValueError) as error:
        raise OpenLaneAdapterError(f"{label}: {error}") from error
    if len(transformed) < minimum_points:
        raise OpenLaneAdapterError(f"{label} must contain at least {minimum_points} points")
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
        row_values: list[int] = []
        for column_index, item in enumerate(values):
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise OpenLaneAdapterError(
                    f"{label}[{row_index}][{column_index}] must be numeric zero or one"
                )
            numeric = float(item)
            if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
                raise OpenLaneAdapterError(
                    f"{label} ground-truth entries must be exactly zero or one"
                )
            row_values.append(int(numeric))
        row = tuple(row_values)
        result.append(row)
    return tuple(result)


class OpenLaneAdapter:
    """Read deterministic raw `-ls.json` frames without importing the pickle devkit."""

    def __init__(self, root: Path, config_path: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)
        try:
            self.config = load_yaml_object_path(
                config_path,
                "adapter config",
                ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
            )
        except ParseBoundaryError as error:
            raise OpenLaneAdapterError(str(error)) from error
        self.adapter_version = str(self.config["adapter_version"])
        self.schema_mode = str(self.config["schema_mode"])
        if self.schema_mode != "lane-segment-v2.1":
            raise OpenLaneAdapterError(f"unsupported schema mode: {self.schema_mode}")
        raw_mappings = _as_mapping(self.config["source_camera_mappings"], "camera mappings")
        self.camera_mappings = {
            str(domain): {
                str(source): CameraSlot(str(target))
                for source, target in _as_mapping(mapping, f"camera mappings.{domain}").items()
            }
            for domain, mapping in raw_mappings.items()
        }
        raw_dimensions = _as_mapping(self.config["source_image_dimensions"], "image dimensions")
        self.source_image_dimensions = {
            str(domain): {
                str(camera): self._parse_dimensions(
                    dimensions, f"image dimensions.{domain}.{camera}"
                )
                for camera, dimensions in _as_mapping(
                    domain_dimensions, f"image dimensions.{domain}"
                ).items()
            }
            for domain, domain_dimensions in raw_dimensions.items()
        }
        model_image = _as_mapping(self.config["model_image"], "model image")
        self.model_height = _required_int(model_image, "height", "model image")
        self.model_width = _required_int(model_image, "width", "model image")
        if (self.model_height, self.model_width) != (384, 640):
            raise OpenLaneAdapterError("model image dimensions must be exactly 384 by 640")
        if model_image.get("color_order") != "RGB":
            raise OpenLaneAdapterError("only RGB model images are supported")
        if model_image.get("resize_policy") != "letterbox-fit":
            raise OpenLaneAdapterError("only letterbox-fit resizing is supported")
        if model_image.get("interpolation") != "bilinear":
            raise OpenLaneAdapterError("only bilinear image interpolation is supported")
        self.pad_value = _required_int(model_image, "pad_value", "model image")
        if not 0 <= self.pad_value <= 255:
            raise OpenLaneAdapterError("model image pad value must be between zero and 255")
        self.mean = self._finite_triplet(model_image.get("mean"), "model image.mean")
        self.std = self._finite_triplet(model_image.get("std"), "model image.std")
        if any(value <= 0.0 for value in self.std):
            raise OpenLaneAdapterError("model image standard deviations must be positive")

    @staticmethod
    def _parse_dimensions(value: object, label: str) -> tuple[int, int]:
        dimensions = _as_sequence(value, label)
        if len(dimensions) != 2:
            raise OpenLaneAdapterError(f"{label} must contain width and height")
        try:
            width, height = (int(item) for item in dimensions)
        except (TypeError, ValueError) as error:
            raise OpenLaneAdapterError(f"{label} must contain integer dimensions") from error
        if min(width, height) <= 0:
            raise OpenLaneAdapterError(f"{label} dimensions must be positive")
        return width, height

    @staticmethod
    def _finite_triplet(value: object, label: str) -> tuple[float, float, float]:
        values = _as_sequence(value, label)
        if len(values) != 3:
            raise OpenLaneAdapterError(f"{label} must contain three values")
        result = tuple(float(item) for item in values)
        if not all(math.isfinite(item) for item in result):
            raise OpenLaneAdapterError(f"{label} must contain finite values")
        return cast(tuple[float, float, float], result)

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
        try:
            lane_segment_path = self._safe_path(
                Path(split_id) / segment_id / "info" / f"{timestamp}-ls.json"
            )
        except (OSError, OpenLaneAdapterError) as error:
            raise OpenLaneAdapterError(
                "missing or unsafe lane-segment metadata; install the Map Element Bucket"
            ) from error
        try:
            raw_bytes = read_bounded_file(
                lane_segment_path,
                "frame metadata",
                _MAX_METADATA_BYTES,
            )
            metadata = load_json_object(
                raw_bytes,
                "frame metadata",
                ParseLimits(
                    max_bytes=_MAX_METADATA_BYTES,
                    max_depth=32,
                    max_nodes=1_000_000,
                    max_container_items=100_000,
                    max_string_bytes=4 * 1024 * 1024,
                ),
            )
        except ParseBoundaryError as error:
            raise OpenLaneAdapterError(str(error)) from error
        metadata_version = str(_required(metadata, "version", "frame metadata"))
        if not metadata_version or len(metadata_version) > 64:
            raise OpenLaneAdapterError("OpenLane metadata version must be a short nonempty string")
        declared_segment = str(metadata.get("segment_id", segment_id))
        declared_timestamp = int(metadata.get("timestamp", timestamp))
        if declared_segment != segment_id or declared_timestamp != int(timestamp):
            raise OpenLaneAdapterError("manifest and frame identity disagree")

        meta_data = _as_mapping(metadata.get("meta_data", {}), "meta_data")
        source_name = str(_required(meta_data, "source", "meta_data"))
        source_segment_id = str(_required(meta_data, "source_id", "meta_data"))
        if not source_name or not source_segment_id:
            raise OpenLaneAdapterError("source name and source segment ID must be nonempty")
        source_domain = self._domain_key(source_name)
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
            source_metadata=SourceDomainMetadata(
                source_name=source_name,
                source_segment_id=source_segment_id,
                normalized_domain=source_domain,
                metadata_version=metadata_version,
                schema_mode=self.schema_mode,
            ),
            cameras=cameras,
            t_world_vehicle=_matrix4_tuple(pose),
            pose_valid=pose_valid,
            annotations_valid="annotation" in metadata,
            adapter_version=self.adapter_version,
            lanes=lanes,
            traffic_controls=controls,
            road_areas=areas,
            topology_lane_lane=topology_ll,
            topology_lane_traffic=topology_lt,
        )

    def load_camera_rgb(self, camera: CameraFrame) -> npt.NDArray[np.uint8]:
        """Decode exactly one valid camera image on explicit caller demand."""
        if not camera.valid or camera.image_relative_path is None:
            raise OpenLaneAdapterError(f"camera slot {camera.slot.value} has no source image")
        image_path = self._safe_path(Path(camera.image_relative_path))
        try:
            image_bytes = read_bounded_file(
                image_path,
                f"camera image {camera.slot.value}",
                _MAX_IMAGE_BYTES,
            )
            with Image.open(BytesIO(image_bytes)) as source:
                source.load()
                if source.size != (camera.original_width, camera.original_height):
                    raise OpenLaneAdapterError(
                        f"image dimensions disagree for {camera.image_relative_path}: "
                        f"expected {camera.original_width}x{camera.original_height}, "
                        f"observed {source.width}x{source.height}"
                    )
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        except (Image.DecompressionBombError, OSError, ParseBoundaryError) as error:
            raise OpenLaneAdapterError(
                f"cannot decode camera image {camera.image_relative_path}: {error}"
            ) from error
        if rgb.shape != (camera.original_height, camera.original_width, 3):
            raise OpenLaneAdapterError(
                f"decoded camera image has an invalid RGB shape: {rgb.shape}"
            )
        rgb.setflags(write=False)
        return rgb

    def model_camera_inputs(self, frame: AdaptedFrame) -> ModelCameraInputs:
        """Decode and normalize one frame into the frozen model camera contract."""
        images = np.zeros(
            (len(CAMERA_SLOTS), 3, self.model_height, self.model_width),
            dtype=np.float32,
        )
        camera_valid = np.zeros((len(CAMERA_SLOTS),), dtype=np.bool_)
        intrinsics = np.zeros((len(CAMERA_SLOTS), 3, 3), dtype=np.float32)
        transforms = np.zeros((len(CAMERA_SLOTS), 4, 4), dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, 3)
        for index, camera in enumerate(frame.cameras):
            if not camera.valid:
                continue
            rgb = self.load_camera_rgb(camera)
            resized_width, resized_height, pad_left, pad_top = self._letterbox_plan(
                camera.original_width, camera.original_height
            )
            source_image = Image.fromarray(rgb)
            resized = source_image.resize(
                (resized_width, resized_height),
                resample=Image.Resampling.BILINEAR,
            )
            canvas = Image.new(
                "RGB",
                (self.model_width, self.model_height),
                color=(self.pad_value, self.pad_value, self.pad_value),
            )
            canvas.paste(resized, (pad_left, pad_top))
            pixels = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
            normalized = (pixels - mean) / std
            images[index] = np.transpose(normalized, (2, 0, 1))
            camera_valid[index] = True
            original_to_model = np.asarray(camera.image_transform, dtype=np.float64)
            intrinsics[index] = (
                original_to_model @ np.asarray(camera.intrinsic, dtype=np.float64)
            ).astype(np.float32)
            transforms[index] = np.asarray(camera.t_vehicle_camera, dtype=np.float32)
        return ModelCameraInputs(
            images=images,
            camera_valid=camera_valid,
            intrinsics=intrinsics,
            t_vehicle_camera=transforms,
        )

    def _load_json(self, path: Path, label: str) -> Mapping[str, Any]:
        safe_path = self._safe_path(path.relative_to(self.root))
        try:
            return load_json_object_path(
                safe_path,
                label,
                ParseLimits(
                    max_bytes=_MAX_METADATA_BYTES,
                    max_depth=32,
                    max_nodes=2_000_000,
                    max_container_items=1_000_000,
                    max_string_bytes=4 * 1024 * 1024,
                ),
            )
        except ParseBoundaryError as error:
            raise OpenLaneAdapterError(str(error)) from error

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

    @staticmethod
    def _domain_key(source_domain: str) -> str:
        normalized = source_domain.strip().lower().replace(" ", "_").replace("-", "_")
        return {
            "argoverse": "argoverse2",
            "argoverse_2": "argoverse2",
            "argoverse_v2": "argoverse2",
            "nuscenes": "nuscenes",
            "nu_scenes": "nuscenes",
        }.get(normalized, normalized)

    def _domain_mapping(
        self, source_domain: str, source_cameras: Sequence[str]
    ) -> Mapping[str, CameraSlot]:
        normalized = self._domain_key(source_domain)
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

    def _source_dimensions(
        self,
        source_domain: str,
        source_camera: str,
        sensor: Mapping[str, Any],
    ) -> tuple[int, int]:
        declared_width = sensor.get("image_width")
        declared_height = sensor.get("image_height")
        if (declared_width is None) != (declared_height is None):
            raise OpenLaneAdapterError("source image width and height must be declared together")
        if declared_width is not None and declared_height is not None:
            try:
                width, height = int(declared_width), int(declared_height)
            except (TypeError, ValueError) as error:
                raise OpenLaneAdapterError("source image dimensions must be integers") from error
            if min(width, height) <= 0:
                raise OpenLaneAdapterError("source image dimensions must be positive")
            return width, height
        normalized = self._domain_key(source_domain)
        try:
            return self.source_image_dimensions[normalized][source_camera]
        except KeyError as error:
            raise OpenLaneAdapterError(
                "metadata omits image dimensions and no dimensions are pinned for "
                f"{normalized!r}.{source_camera}"
            ) from error

    def _letterbox_plan(self, source_width: int, source_height: int) -> tuple[int, int, int, int]:
        scale = min(self.model_width / source_width, self.model_height / source_height)
        resized_width = max(1, min(self.model_width, math.floor(source_width * scale + 0.5)))
        resized_height = max(1, min(self.model_height, math.floor(source_height * scale + 0.5)))
        return (
            resized_width,
            resized_height,
            (self.model_width - resized_width) // 2,
            (self.model_height - resized_height) // 2,
        )

    def _letterbox_transform(self, source_width: int, source_height: int) -> FloatArray:
        resized_width, resized_height, pad_left, pad_top = self._letterbox_plan(
            source_width, source_height
        )
        return image_transform(
            resized_width / source_width,
            resized_height / source_height,
            pad_left=float(pad_left),
            pad_top=float(pad_top),
        )

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
            self._safe_path(image_relative)
            width, height = self._source_dimensions(source_domain, source_camera, sensor)
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
                image_transform=_matrix3_tuple(self._letterbox_transform(width, height)),
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
        if "annotation" not in metadata:
            return (), (), (), (), ()
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
        for label, identifiers in {
            "lane_segment": [lane.source_object_id for lane in lanes],
            "traffic_element": [control.source_object_id for control in controls],
            "area": [area.source_object_id for area in areas],
        }.items():
            if len(identifiers) != len(set(identifiers)):
                raise OpenLaneAdapterError(f"annotation.{label} contains duplicate IDs")
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
        connector = _required(lane, "is_intersection_or_connector", label)
        if not isinstance(connector, bool):
            raise OpenLaneAdapterError(f"{label}.is_intersection_or_connector must be a boolean")
        return LaneSegment(
            source_object_id=str(_required(lane, "id", label)),
            centerline=_point3_tuple(lane.get("centerline"), f"lane_segment[{index}].centerline"),
            left_boundary=_point3_tuple(
                lane.get("left_laneline"), f"lane_segment[{index}].left_laneline"
            ),
            right_boundary=_point3_tuple(
                lane.get("right_laneline"), f"lane_segment[{index}].right_laneline"
            ),
            left_boundary_type=_bounded_category(
                lane, "left_laneline_type", label, _ALLOWED_BOUNDARY_TYPES
            ),
            right_boundary_type=_bounded_category(
                lane, "right_laneline_type", label, _ALLOWED_BOUNDARY_TYPES
            ),
            is_intersection_or_connector=connector,
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
        if points[1][0] <= points[0][0] or points[1][1] <= points[0][1]:
            raise OpenLaneAdapterError(
                f"traffic_element[{index}].points must be a positive-area XYXY box"
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
            category=_bounded_category(control, "category", label, _ALLOWED_TRAFFIC_CATEGORIES),
            attribute=_bounded_category(control, "attribute", label, _ALLOWED_TRAFFIC_ATTRIBUTES),
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
        category = _bounded_category(area, "category", label, _ALLOWED_AREA_CATEGORIES)
        return RoadArea(
            source_object_id=str(_required(area, "id", label)),
            category=category,
            points=_point3_tuple(
                area.get("points"),
                f"area[{index}].points",
                minimum_points=3 if category == 1 else 2,
            ),
        )

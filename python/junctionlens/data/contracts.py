"""Immutable M0 data boundaries used before protobuf ownership in M1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


class CameraSlot(StrEnum):
    """Canonical camera slots in stable tensor order."""

    FRONT_CENTER = "FRONT_CENTER"
    FRONT_LEFT = "FRONT_LEFT"
    FRONT_RIGHT = "FRONT_RIGHT"
    SIDE_LEFT = "SIDE_LEFT"
    SIDE_RIGHT = "SIDE_RIGHT"
    REAR_LEFT = "REAR_LEFT"
    REAR_CENTER = "REAR_CENTER"
    REAR_RIGHT = "REAR_RIGHT"


CAMERA_SLOTS: tuple[CameraSlot, ...] = tuple(CameraSlot)


@dataclass(frozen=True, slots=True)
class FrameKey:
    """Lossless frame identity and source provenance."""

    dataset_id: str
    dataset_version: str
    split_id: str
    segment_id: str
    timestamp_ns: int
    source_domain: str
    calibration_sha256: str
    frame_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One canonical camera entry, including explicit missing slots."""

    slot: CameraSlot
    valid: bool
    source_camera: str | None
    image_relative_path: str | None
    capture_timestamp_ns: int
    original_width: int
    original_height: int
    intrinsic: Matrix3
    t_vehicle_camera: Matrix4
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    image_transform: Matrix3


@dataclass(frozen=True, slots=True)
class SourcePixelBox:
    """Unrounded OpenLane coordinates retained for official evaluation."""

    points: tuple[Point2, Point2]
    convention: str
    image_width: int
    image_height: int


@dataclass(frozen=True, slots=True)
class LaneSegment:
    """Normalized lane-segment annotation in JunctionLens axes."""

    source_object_id: str
    centerline: tuple[Point3, ...]
    left_boundary: tuple[Point3, ...]
    right_boundary: tuple[Point3, ...]
    left_boundary_type: int
    right_boundary_type: int
    is_intersection_or_connector: bool


@dataclass(frozen=True, slots=True)
class TrafficControl:
    """Traffic element with lossless source and normalized box forms."""

    source_object_id: str
    category: int
    attribute: int
    source_camera: CameraSlot
    source_pixel_box: SourcePixelBox
    normalized_half_open_box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class RoadArea:
    """Normalized pedestrian-crossing or road-boundary annotation."""

    source_object_id: str
    category: int
    points: tuple[Point3, ...]


@dataclass(frozen=True, slots=True)
class AdaptedFrame:
    """One immutable normalized frame produced from OpenLane source JSON."""

    key: FrameKey
    cameras: tuple[CameraFrame, ...]
    t_world_vehicle: Matrix4
    pose_valid: bool
    adapter_version: str
    lanes: tuple[LaneSegment, ...]
    traffic_controls: tuple[TrafficControl, ...]
    road_areas: tuple[RoadArea, ...]
    topology_lane_lane: tuple[tuple[int, ...], ...]
    topology_lane_traffic: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if tuple(camera.slot for camera in self.cameras) != CAMERA_SLOTS:
            raise ValueError("camera entries do not match canonical slot order")
        lane_count = len(self.lanes)
        traffic_count = len(self.traffic_controls)
        if len(self.topology_lane_lane) != lane_count or any(
            len(row) != lane_count for row in self.topology_lane_lane
        ):
            raise ValueError("lane-lane topology shape does not match lane count")
        if len(self.topology_lane_traffic) != lane_count or any(
            len(row) != traffic_count for row in self.topology_lane_traffic
        ):
            raise ValueError("lane-traffic topology shape does not match node counts")

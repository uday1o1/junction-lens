"""Private visual overlays and aggregate-only OpenLane data audits."""

from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image, ImageDraw

from junctionlens.data.contracts import AdaptedFrame, CameraFrame, Point3
from junctionlens.data.geometry import project_vehicle_points
from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
from junctionlens.registry.store import canonical_json_bytes

FloatArray = npt.NDArray[np.float64]
_AUDIT_POLICY_KEYS = {
    "bev_render",
    "canonical_bev_range_m",
    "control_pixel_area_buckets",
    "dataset_id",
    "dataset_version",
    "frozen_frames",
    "hard_geometry_range_m",
    "lane_count_buckets",
    "lane_curvature_buckets_per_m",
    "long_range_x_m",
    "low_luminance_threshold",
    "policy_id",
    "schema_version",
    "slice_registry",
}
_FRAME_SELECTOR_KEYS = {"segment_id", "split_id", "timestamp"}
_TRAFFIC_ATTRIBUTE_GROUPS = {
    0: "unknown",
    1: "signal-state",
    2: "signal-state",
    3: "signal-state",
    4: "required-movement",
    5: "required-movement",
    6: "required-movement",
    7: "prohibited-movement",
    8: "prohibited-movement",
    9: "required-movement",
    10: "prohibited-movement",
    11: "required-movement",
    12: "required-movement",
}
_SLICE_IDS = (
    "source_domain",
    "intersection_or_connector_presence",
    "merge_or_split_topology",
    "lane_graph_degree_bucket",
    "lane_curvature_bucket",
    "traffic_control_pixel_area_bucket",
    "long_range_projection_proxy",
    "crosswalk_presence",
    "traffic_control_attribute_group",
    "low_luminance_proxy",
    "camera_availability_pattern",
    "lane_count_complexity_bucket",
)


class VisualAuditError(RuntimeError):
    """Raised when audit policy, geometry, rendering, or persistence is invalid."""


@dataclass(frozen=True, slots=True)
class AuditPolicy:
    """Frozen rendering, range, bucket, and frame-selection policy."""

    policy_id: str
    hard_geometry_range_m: Mapping[str, tuple[float, float]]
    canonical_bev_range_m: Mapping[str, tuple[float, float]]
    lane_curvature_buckets_per_m: tuple[float, float]
    control_pixel_area_buckets: tuple[float, float]
    lane_count_buckets: tuple[int, int]
    long_range_x_m: float
    low_luminance_threshold: float
    bev_width: int
    bev_height: int
    frozen_frames: tuple[tuple[str, str, str], ...]
    slice_registry_sha256: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class RenderedOverlay:
    """One deterministic PNG overlay and its projection evidence."""

    png: bytes
    projected_point_count: int
    visible_point_count: int


@dataclass(frozen=True, slots=True)
class AuditBundleReceipt:
    """Stable identity and bounded public evidence for a private audit bundle."""

    bundle_manifest_sha256: str
    file_count: int
    selected_frame_count: int
    rendered_camera_count: int
    range_gate_accepted: bool


def _pair(
    value: object,
    label: str,
    *,
    integer: bool = False,
) -> tuple[float, float] | tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise VisualAuditError(f"{label} must contain exactly two values")
    if integer:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise VisualAuditError(f"{label} must contain integers")
        result_int = (int(value[0]), int(value[1]))
        if result_int[0] <= 0 or result_int[0] >= result_int[1]:
            raise VisualAuditError(f"{label} must be positive and increasing")
        return result_int
    try:
        result = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as error:
        raise VisualAuditError(f"{label} must contain numbers") from error
    if not all(math.isfinite(item) for item in result) or result[0] >= result[1]:
        raise VisualAuditError(f"{label} must be finite and increasing")
    return result


def load_audit_policy(path: Path) -> AuditPolicy:
    """Load the exact V1 aggregate and manual-inspection policy."""
    raw = path.read_bytes()
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise VisualAuditError(f"invalid audit policy YAML: {error}") from error
    if not isinstance(value, dict) or set(value) != _AUDIT_POLICY_KEYS:
        raise VisualAuditError("audit policy has invalid top-level keys")
    if (
        value["schema_version"] != "junctionlens.openlane-audit-policy.v1"
        or value["policy_id"] != "openlane-v2-v2.1-audit-v1"
        or value["dataset_id"] != "openlane-v2-v2.1"
        or value["dataset_version"] != "2.1"
    ):
        raise VisualAuditError("audit policy identity differs from V1")
    registry_contract = value["slice_registry"]
    if not isinstance(registry_contract, dict) or set(registry_contract) != {"path", "sha256"}:
        raise VisualAuditError("audit policy slice registry contract is incomplete")
    registry_relative = Path(str(registry_contract["path"]))
    if registry_relative.is_absolute():
        raise VisualAuditError("slice registry path must be relative to the audit policy")
    policy_path = path.resolve(strict=True)
    config_root = policy_path.parent.parent
    registry_path = (policy_path.parent / registry_relative).resolve(strict=True)
    try:
        registry_path.relative_to(config_root)
    except ValueError as error:
        raise VisualAuditError("slice registry path escapes the configuration root") from error
    registry_bytes = registry_path.read_bytes()
    registry_sha256 = hashlib.sha256(registry_bytes).hexdigest()
    if registry_contract["sha256"] != registry_sha256:
        raise VisualAuditError("slice registry differs from its audit-policy hash")
    try:
        registry = yaml.safe_load(registry_bytes)
    except yaml.YAMLError as error:
        raise VisualAuditError(f"invalid slice registry YAML: {error}") from error
    if (
        not isinstance(registry, dict)
        or set(registry) != {"registry_id", "schema_version", "slices"}
        or registry["schema_version"] != "junctionlens.slice-registry.v1"
        or registry["registry_id"] != "junctionlens-openlane-slices-v1"
        or not isinstance(registry["slices"], list)
    ):
        raise VisualAuditError("slice registry contract is invalid")
    slice_ids: list[str] = []
    for index, item in enumerate(registry["slices"]):
        if not isinstance(item, dict) or set(item) != {"definition", "id", "provenance"}:
            raise VisualAuditError(f"slice registry item {index} is invalid")
        if not all(isinstance(item[key], str) and item[key] for key in item):
            raise VisualAuditError(f"slice registry item {index} has an empty field")
        slice_ids.append(str(item["id"]))
    if tuple(slice_ids) != _SLICE_IDS:
        raise VisualAuditError("slice registry IDs or order differ from V1")
    hard_ranges = value["hard_geometry_range_m"]
    bev_ranges = value["canonical_bev_range_m"]
    render = value["bev_render"]
    if (
        not isinstance(hard_ranges, dict)
        or set(hard_ranges) != {"x", "y", "z"}
        or not isinstance(bev_ranges, dict)
        or set(bev_ranges) != {"x", "y"}
        or not isinstance(render, dict)
        or set(render) != {"height", "width"}
    ):
        raise VisualAuditError("audit range or render configuration is incomplete")
    frames = value["frozen_frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= 32:
        raise VisualAuditError("audit policy must select between one and 32 frames")
    frozen_frames: list[tuple[str, str, str]] = []
    for index, item in enumerate(frames):
        if not isinstance(item, dict) or set(item) != _FRAME_SELECTOR_KEYS:
            raise VisualAuditError(f"audit frame selector {index} has invalid keys")
        identifier = tuple(str(item[key]) for key in ("split_id", "segment_id", "timestamp"))
        if any(not part or Path(part).name != part for part in identifier):
            raise VisualAuditError(f"audit frame selector {index} is unsafe")
        try:
            int(identifier[2])
        except ValueError as error:
            raise VisualAuditError(f"audit frame selector {index} has invalid timestamp") from error
        frozen_frames.append(cast(tuple[str, str, str], identifier))
    if len(frozen_frames) != len(set(frozen_frames)):
        raise VisualAuditError("audit policy repeats a frozen frame")
    curvature = cast(
        tuple[float, float],
        _pair(value["lane_curvature_buckets_per_m"], "curvature buckets"),
    )
    area = cast(
        tuple[float, float],
        _pair(value["control_pixel_area_buckets"], "control area buckets"),
    )
    lane_counts = cast(
        tuple[int, int],
        _pair(value["lane_count_buckets"], "lane count buckets", integer=True),
    )
    long_range = float(value["long_range_x_m"])
    luminance = float(value["low_luminance_threshold"])
    width, height = int(render["width"]), int(render["height"])
    if (
        not math.isfinite(long_range)
        or long_range <= 0.0
        or not 0.0 < luminance < 1.0
        or width < 320
        or height < 240
        or width > 4096
        or height > 4096
    ):
        raise VisualAuditError("audit scalar or render configuration is invalid")
    return AuditPolicy(
        policy_id=str(value["policy_id"]),
        hard_geometry_range_m={
            axis: cast(tuple[float, float], _pair(hard_ranges[axis], f"hard range {axis}"))
            for axis in ("x", "y", "z")
        },
        canonical_bev_range_m={
            axis: cast(tuple[float, float], _pair(bev_ranges[axis], f"BEV range {axis}"))
            for axis in ("x", "y")
        },
        lane_curvature_buckets_per_m=curvature,
        control_pixel_area_buckets=area,
        lane_count_buckets=lane_counts,
        long_range_x_m=long_range,
        low_luminance_threshold=luminance,
        bev_width=width,
        bev_height=height,
        frozen_frames=tuple(frozen_frames),
        slice_registry_sha256=registry_sha256,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _polyline_array(points: Sequence[Point3]) -> FloatArray:
    return np.asarray(points, dtype=np.float64)


def project_points(
    frame: AdaptedFrame, camera: CameraFrame, points: Sequence[Point3]
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Project canonical vehicle points through one normalized source calibration."""
    if not camera.valid:
        raise VisualAuditError("cannot project through an invalid camera")
    try:
        return project_vehicle_points(
            _polyline_array(points),
            np.asarray(camera.intrinsic, dtype=np.float64),
            np.asarray(camera.t_vehicle_camera, dtype=np.float64),
        )
    except (TypeError, ValueError) as error:
        raise VisualAuditError(
            f"cannot project frame {frame.key.segment_id}/{frame.key.timestamp_ns}: {error}"
        ) from error


def _visible_runs(
    projected: FloatArray,
    valid: npt.NDArray[np.bool_],
    width: int,
    height: int,
) -> Iterable[list[tuple[float, float]]]:
    current: list[tuple[float, float]] = []
    for point, is_valid in zip(projected, valid, strict=True):
        inside = bool(is_valid) and 0.0 <= point[0] < width and 0.0 <= point[1] < height
        if inside:
            current.append((float(point[0]), float(point[1])))
        elif current:
            yield current
            current = []
    if current:
        yield current


def render_camera_overlay(
    adapter: OpenLaneAdapter,
    frame: AdaptedFrame,
    camera: CameraFrame,
) -> RenderedOverlay:
    """Render a private source-image overlay with calibrated ground labels and boxes."""
    image = Image.fromarray(adapter.load_camera_rgb(camera)).copy()
    draw = ImageDraw.Draw(image)
    projected_count = 0
    visible_count = 0
    geometry: list[tuple[Sequence[Point3], tuple[int, int, int], int]] = []
    for lane in frame.lanes:
        geometry.extend(
            (
                (lane.centerline, (36, 214, 126), 3),
                (lane.left_boundary, (92, 196, 255), 2),
                (lane.right_boundary, (92, 196, 255), 2),
            )
        )
    geometry.extend((area.points, (255, 209, 102), 3) for area in frame.road_areas)
    for points, color, line_width in geometry:
        projected, valid = project_points(frame, camera, points)
        projected_count += len(projected)
        for run in _visible_runs(projected, valid, camera.original_width, camera.original_height):
            visible_count += len(run)
            if len(run) == 1:
                horizontal, vertical = run[0]
                radius = max(2, line_width)
                draw.ellipse(
                    (
                        horizontal - radius,
                        vertical - radius,
                        horizontal + radius,
                        vertical + radius,
                    ),
                    outline=color,
                    width=line_width,
                )
            else:
                draw.line(run, fill=color, width=line_width, joint="curve")
    for control in frame.traffic_controls:
        if control.source_camera != camera.slot:
            continue
        (x_min, y_min), (x_max, y_max) = control.source_pixel_box.points
        draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 72, 72), width=3)
    intrinsic = np.asarray(camera.intrinsic, dtype=np.float64)
    center = (float(intrinsic[0, 2]), float(intrinsic[1, 2]))
    cross = max(4, min(camera.original_width, camera.original_height) // 80)
    draw.line(
        (center[0] - cross, center[1], center[0] + cross, center[1]),
        fill=(0, 255, 255),
        width=2,
    )
    draw.line(
        (center[0], center[1] - cross, center[0], center[1] + cross),
        fill=(0, 255, 255),
        width=2,
    )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return RenderedOverlay(output.getvalue(), projected_count, visible_count)


def _bev_pixel(policy: AuditPolicy, point: Sequence[float]) -> tuple[float, float]:
    x_min, x_max = policy.canonical_bev_range_m["x"]
    y_min, y_max = policy.canonical_bev_range_m["y"]
    horizontal = (y_max - float(point[1])) / (y_max - y_min) * policy.bev_width
    vertical = (x_max - float(point[0])) / (x_max - x_min) * policy.bev_height
    return horizontal, vertical


def _svg_polyline(
    policy: AuditPolicy,
    points: Sequence[Point3],
    *,
    color: str,
    width: float,
    identity: str,
) -> str:
    coordinates = " ".join(
        f"{horizontal:.3f},{vertical:.3f}"
        for horizontal, vertical in (_bev_pixel(policy, point) for point in points)
    )
    return (
        f'<polyline data-id="{escape(identity)}" points="{coordinates}" '
        f'fill="none" stroke="{color}" stroke-width="{width:.1f}" />'
    )


def render_bev_svg(frame: AdaptedFrame, policy: AuditPolicy) -> bytes:
    """Render canonical BEV labels and positive topology in a deterministic SVG."""
    elements: list[str] = []
    for index in range(1, 5):
        horizontal = policy.bev_width * index / 5
        vertical = policy.bev_height * index / 5
        elements.append(
            f'<line x1="{horizontal:.1f}" y1="0" x2="{horizontal:.1f}" '
            f'y2="{policy.bev_height}" stroke="#263341" stroke-width="1" />'
        )
        elements.append(
            f'<line x1="0" y1="{vertical:.1f}" x2="{policy.bev_width}" '
            f'y2="{vertical:.1f}" stroke="#263341" stroke-width="1" />'
        )
    for lane in frame.lanes:
        elements.extend(
            (
                _svg_polyline(
                    policy,
                    lane.left_boundary,
                    color="#5cc4ff",
                    width=1.5,
                    identity=f"lane-{lane.source_object_id}-left",
                ),
                _svg_polyline(
                    policy,
                    lane.right_boundary,
                    color="#5cc4ff",
                    width=1.5,
                    identity=f"lane-{lane.source_object_id}-right",
                ),
                _svg_polyline(
                    policy,
                    lane.centerline,
                    color="#24d67e",
                    width=3.0,
                    identity=f"lane-{lane.source_object_id}-center",
                ),
            )
        )
    for area in frame.road_areas:
        elements.append(
            _svg_polyline(
                policy,
                area.points,
                color="#ffd166",
                width=3.0,
                identity=f"area-{area.source_object_id}",
            )
        )
    for source_index, row in enumerate(frame.topology_lane_lane):
        for target_index, present in enumerate(row):
            if not present:
                continue
            start = _bev_pixel(policy, frame.lanes[source_index].centerline[-1])
            end = _bev_pixel(policy, frame.lanes[target_index].centerline[0])
            elements.append(
                f'<line data-edge="lane-lane-{source_index}-{target_index}" '
                f'x1="{start[0]:.3f}" y1="{start[1]:.3f}" '
                f'x2="{end[0]:.3f}" y2="{end[1]:.3f}" '
                'stroke="#ff70a6" stroke-width="2" stroke-dasharray="5 4" />'
            )
    for lane_index, row in enumerate(frame.topology_lane_traffic):
        if not any(row):
            continue
        horizontal, vertical = _bev_pixel(policy, frame.lanes[lane_index].centerline[-1])
        radius = 4 + 2 * sum(row)
        elements.append(
            f'<circle data-edge="lane-control-{lane_index}" cx="{horizontal:.3f}" '
            f'cy="{vertical:.3f}" r="{radius}" fill="none" '
            'stroke="#ff4848" stroke-width="2" />'
        )
    body = "\n  ".join(elements)
    scene = escape(f"{frame.key.split_id}/{frame.key.segment_id}/{frame.key.timestamp_ns}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{policy.bev_width}" '
        f'height="{policy.bev_height}" viewBox="0 0 {policy.bev_width} {policy.bev_height}" '
        f'data-frame="{scene}">\n'
        f'  <rect width="{policy.bev_width}" height="{policy.bev_height}" fill="#101820" />\n'
        f"  {body}\n"
        "</svg>\n"
    ).encode()


def _curvature(points: Sequence[Point3]) -> float:
    array = _polyline_array(points)
    if len(array) < 3:
        return 0.0
    differences = np.diff(array[:, :2], axis=0)
    lengths = np.linalg.norm(differences, axis=1)
    valid = lengths > 1e-9
    if np.count_nonzero(valid) < 2:
        return 0.0
    headings = np.unwrap(np.arctan2(differences[valid, 1], differences[valid, 0]))
    changes = np.abs(np.diff(headings))
    scales = (lengths[valid][:-1] + lengths[valid][1:]) / 2.0
    return float(np.max(changes / np.maximum(scales, 1e-9), initial=0.0))


def _bucket(value: float, boundaries: tuple[float, float], names: tuple[str, str, str]) -> str:
    if value < boundaries[0]:
        return names[0]
    if value < boundaries[1]:
        return names[1]
    return names[2]


def slice_values(
    frame: AdaptedFrame,
    policy: AuditPolicy,
    luminance: str,
) -> Mapping[str, str]:
    lane_count = len(frame.lanes)
    in_degree = [0] * lane_count
    out_degree = [0] * lane_count
    for source, row in enumerate(frame.topology_lane_lane):
        for target, present in enumerate(row):
            if present:
                out_degree[source] += 1
                in_degree[target] += 1
    has_merge = any(degree >= 2 for degree in in_degree)
    has_split = any(degree >= 2 for degree in out_degree)
    topology_shape = (
        "merge-and-split"
        if has_merge and has_split
        else "merge"
        if has_merge
        else "split"
        if has_split
        else "neither"
    )
    maximum_degree = max([*in_degree, *out_degree], default=0)
    maximum_curvature = max((_curvature(lane.centerline) for lane in frame.lanes), default=0.0)
    maximum_control_area = max(
        (
            (control.normalized_half_open_box[2] - control.normalized_half_open_box[0])
            * (control.normalized_half_open_box[3] - control.normalized_half_open_box[1])
            for control in frame.traffic_controls
        ),
        default=-1.0,
    )
    attribute_groups = sorted(
        {_TRAFFIC_ATTRIBUTE_GROUPS[control.attribute] for control in frame.traffic_controls}
    )
    camera_pattern = "".join("1" if camera.valid else "0" for camera in frame.cameras)
    return {
        "source_domain": frame.key.source_domain,
        "intersection_or_connector_presence": (
            "present"
            if any(lane.is_intersection_or_connector for lane in frame.lanes)
            else "absent"
        ),
        "merge_or_split_topology": topology_shape,
        "lane_graph_degree_bucket": (
            "degree-0-or-1"
            if maximum_degree <= 1
            else "degree-2"
            if maximum_degree == 2
            else "degree-3plus"
        ),
        "lane_curvature_bucket": _bucket(
            maximum_curvature,
            policy.lane_curvature_buckets_per_m,
            ("low", "medium", "high"),
        ),
        "traffic_control_pixel_area_bucket": (
            "none"
            if maximum_control_area < 0.0
            else _bucket(
                maximum_control_area,
                policy.control_pixel_area_buckets,
                ("small", "medium", "large"),
            )
        ),
        "long_range_projection_proxy": (
            "present"
            if any(
                point[0] >= policy.long_range_x_m
                for lane in frame.lanes
                for point in lane.centerline
            )
            else "absent"
        ),
        "crosswalk_presence": (
            "present" if any(area.category == 1 for area in frame.road_areas) else "absent"
        ),
        "traffic_control_attribute_group": "+".join(attribute_groups) or "none",
        "low_luminance_proxy": luminance,
        "camera_availability_pattern": camera_pattern,
        "lane_count_complexity_bucket": _bucket(
            float(lane_count),
            (float(policy.lane_count_buckets[0]), float(policy.lane_count_buckets[1])),
            ("low", "medium", "high"),
        ),
    }


def _mean_luminance(adapter: OpenLaneAdapter, frame: AdaptedFrame, policy: AuditPolicy) -> str:
    front = frame.cameras[0]
    if not front.valid:
        return "unavailable"
    rgb = adapter.load_camera_rgb(front).astype(np.float32)
    luminance = float(
        np.mean(
            rgb[..., 0] * np.float32(0.2126)
            + rgb[..., 1] * np.float32(0.7152)
            + rgb[..., 2] * np.float32(0.0722)
        )
        / 255.0
    )
    return "low" if luminance < policy.low_luminance_threshold else "not-low"


def audit_dataset(
    adapter: OpenLaneAdapter,
    profile: str,
    policy: AuditPolicy,
    *,
    compute_luminance: bool = True,
) -> Mapping[str, Any]:
    """Stream complete aggregate distributions without retaining frames or source images."""
    frame_count = 0
    class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    capacity_histograms: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    camera_patterns: dict[str, int] = defaultdict(int)
    geometry_min = np.full((3,), np.inf, dtype=np.float64)
    geometry_max = np.full((3,), -np.inf, dtype=np.float64)
    geometry_point_count = 0
    outside_hard_range_count = 0
    canonical_bev_point_count = 0
    topology: dict[str, int] = defaultdict(int)
    slice_frames: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    slice_segments: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    capacities = cast(Mapping[str, int], adapter.config["query_capacities"])
    capacity_exceeding: dict[str, int] = defaultdict(int)
    for frame in adapter.iter_frames(profile):
        frame_count += 1
        frame_counts = {
            "lane_segment": len(frame.lanes),
            "traffic_element": len(frame.traffic_controls),
            "area": len(frame.road_areas),
        }
        for object_type, count in frame_counts.items():
            capacity_histograms[object_type][count] += 1
            capacity_exceeding[object_type] += int(count > capacities[object_type])
        for lane in frame.lanes:
            class_counts["left_boundary_type"][str(lane.left_boundary_type)] += 1
            class_counts["right_boundary_type"][str(lane.right_boundary_type)] += 1
            class_counts["connector"][str(lane.is_intersection_or_connector).lower()] += 1
        for control in frame.traffic_controls:
            class_counts["traffic_category"][str(control.category)] += 1
            class_counts["traffic_attribute"][str(control.attribute)] += 1
        for area in frame.road_areas:
            class_counts["area_category"][str(area.category)] += 1
        camera_pattern = "".join("1" if camera.valid else "0" for camera in frame.cameras)
        camera_patterns[camera_pattern] += 1
        point_groups = [
            *(lane.centerline for lane in frame.lanes),
            *(lane.left_boundary for lane in frame.lanes),
            *(lane.right_boundary for lane in frame.lanes),
            *(area.points for area in frame.road_areas),
        ]
        for points in point_groups:
            array = _polyline_array(points)
            geometry_min = np.minimum(geometry_min, array.min(axis=0))
            geometry_max = np.maximum(geometry_max, array.max(axis=0))
            geometry_point_count += len(array)
            inside_hard = np.ones((len(array),), dtype=np.bool_)
            for axis_index, axis in enumerate(("x", "y", "z")):
                lower, upper = policy.hard_geometry_range_m[axis]
                inside_hard &= (array[:, axis_index] >= lower) & (array[:, axis_index] <= upper)
            outside_hard_range_count += int(np.count_nonzero(~inside_hard))
            bev_x = policy.canonical_bev_range_m["x"]
            bev_y = policy.canonical_bev_range_m["y"]
            inside_bev = (
                (array[:, 0] >= bev_x[0])
                & (array[:, 0] < bev_x[1])
                & (array[:, 1] >= bev_y[0])
                & (array[:, 1] < bev_y[1])
            )
            canonical_bev_point_count += int(np.count_nonzero(inside_bev))
        lane_count = len(frame.lanes)
        control_count = len(frame.traffic_controls)
        topology["lane_lane_possible_off_diagonal"] += lane_count * max(0, lane_count - 1)
        topology["lane_lane_positive"] += sum(sum(row) for row in frame.topology_lane_lane)
        topology["lane_lane_self_positive"] += sum(
            frame.topology_lane_lane[index][index] for index in range(lane_count)
        )
        topology["lane_control_possible"] += lane_count * control_count
        topology["lane_control_positive"] += sum(sum(row) for row in frame.topology_lane_traffic)
        luminance = _mean_luminance(adapter, frame, policy) if compute_luminance else "not-computed"
        for slice_name, value in slice_values(frame, policy, luminance).items():
            slice_frames[slice_name][value] += 1
            slice_segments[slice_name][value].add((frame.key.split_id, frame.key.segment_id))
    if frame_count == 0 or geometry_point_count == 0:
        raise VisualAuditError("dataset audit requires frames with ground geometry")
    capacity_report = {
        object_type: {
            "capacity": capacities[object_type],
            "histogram": {
                str(count): frequency
                for count, frequency in sorted(capacity_histograms[object_type].items())
            },
            "exceeding_frame_count": capacity_exceeding[object_type],
            "coverage": 1.0 - capacity_exceeding[object_type] / frame_count,
        }
        for object_type in ("lane_segment", "traffic_element", "area")
    }
    slices = {
        slice_name: {
            value: {
                "frame_count": slice_frames[slice_name][value],
                "segment_count": len(segments),
            }
            for value, segments in sorted(values.items())
        }
        for slice_name, values in sorted(slice_segments.items())
    }
    return {
        "schema_version": "junctionlens.openlane-statistical-audit.v1",
        "policy_id": policy.policy_id,
        "profile": profile,
        "slice_registry_sha256": policy.slice_registry_sha256,
        "frame_count": frame_count,
        "class_distributions": {
            name: dict(sorted(values.items())) for name, values in sorted(class_counts.items())
        },
        "capacity_distributions": capacity_report,
        "missing_camera_patterns": dict(sorted(camera_patterns.items())),
        "geometry_ranges_m": {
            axis: {"minimum": float(geometry_min[index]), "maximum": float(geometry_max[index])}
            for index, axis in enumerate(("x", "y", "z"))
        },
        "geometry_point_count": geometry_point_count,
        "outside_hard_range_point_count": outside_hard_range_count,
        "canonical_bev_point_fraction": canonical_bev_point_count / geometry_point_count,
        "topology_support": dict(sorted(topology.items())),
        "slice_support_preview": slices,
        "range_gate_accepted": outside_hard_range_count == 0,
    }


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _bundle_index(visual_paths: Sequence[str]) -> bytes:
    items = "\n".join(
        f'<li><a href="{escape(path)}">{escape(path)}</a><br><img src="{escape(path)}" '
        'loading="lazy" style="max-width:900px;max-height:700px"></li>'
        for path in visual_paths
    )
    return (
        "<!doctype html>\n"
        '<html lang="en"><meta charset="utf-8">\n'
        "<title>JunctionLens private data audit</title>\n"
        "<body><h1>Private licensed-data audit</h1>\n"
        "<p>Do not publish source-image overlays or label renderings without review.</p>\n"
        f"<ol>{items}</ol></body></html>\n"
    ).encode()


def write_audit_bundle(
    adapter: OpenLaneAdapter,
    profile: str,
    policy: AuditPolicy,
    output_root: Path,
) -> AuditBundleReceipt:
    """Write a frozen private visual and aggregate bundle through one bounded workflow."""
    if output_root.exists() or output_root.is_symlink():
        raise VisualAuditError("audit output already exists; choose a new frozen output path")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    records: list[dict[str, Any]] = []
    visual_paths: list[str] = []
    rendered_camera_count = 0
    try:
        statistical_report = audit_dataset(adapter, profile, policy)
        for split_id, segment_id, timestamp in policy.frozen_frames:
            try:
                frame = adapter.load_frame(split_id, segment_id, timestamp)
            except OpenLaneAdapterError as error:
                raise VisualAuditError(str(error)) from error
            frame_root = Path(split_id) / segment_id / timestamp
            bev_relative = (frame_root / "bev.svg").as_posix()
            bev = render_bev_svg(frame, policy)
            _write(temp / bev_relative, bev)
            records.append(
                {
                    "path": bev_relative,
                    "media_type": "image/svg+xml",
                    "byte_size": len(bev),
                    "sha256": hashlib.sha256(bev).hexdigest(),
                    "privacy": "PRIVATE_DERIVED_LABEL_GEOMETRY",
                }
            )
            visual_paths.append(bev_relative)
            for camera in frame.cameras:
                if not camera.valid:
                    continue
                overlay = render_camera_overlay(adapter, frame, camera)
                relative = (frame_root / f"camera-{camera.slot.value.lower()}.png").as_posix()
                _write(temp / relative, overlay.png)
                records.append(
                    {
                        "path": relative,
                        "media_type": "image/png",
                        "byte_size": len(overlay.png),
                        "sha256": hashlib.sha256(overlay.png).hexdigest(),
                        "privacy": "PRIVATE_LICENSED_SOURCE_IMAGE",
                        "projected_point_count": overlay.projected_point_count,
                        "visible_point_count": overlay.visible_point_count,
                    }
                )
                visual_paths.append(relative)
                rendered_camera_count += 1
        summary = canonical_json_bytes(statistical_report) + b"\n"
        _write(temp / "summary.json", summary)
        index = _bundle_index(visual_paths)
        _write(temp / "index.html", index)
        for path, media_type, payload in (
            ("summary.json", "application/json", summary),
            ("index.html", "text/html", index),
        ):
            records.append(
                {
                    "path": path,
                    "media_type": media_type,
                    "byte_size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "privacy": "AGGREGATE_ONLY" if path == "summary.json" else "PRIVATE_INDEX",
                }
            )
        bundle_manifest = {
            "schema_version": "junctionlens.openlane-audit-bundle.v1",
            "policy_id": policy.policy_id,
            "policy_config_sha256": policy.config_sha256,
            "slice_registry_sha256": policy.slice_registry_sha256,
            "profile": profile,
            "selected_frames": [
                {"split_id": split, "segment_id": segment, "timestamp": timestamp}
                for split, segment, timestamp in policy.frozen_frames
            ],
            "files": sorted(records, key=lambda record: str(record["path"])),
            "manual_review_state": "PENDING_HUMAN_INSPECTION",
        }
        manifest_bytes = canonical_json_bytes(bundle_manifest) + b"\n"
        _write(temp / "manifest.json", manifest_bytes)
        temp.rename(output_root)
        descriptor = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return AuditBundleReceipt(
        bundle_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        file_count=len(records) + 1,
        selected_frame_count=len(policy.frozen_frames),
        rendered_camera_count=rendered_camera_count,
        range_gate_accepted=bool(statistical_report["range_gate_accepted"]),
    )


__all__ = [
    "AuditBundleReceipt",
    "AuditPolicy",
    "RenderedOverlay",
    "VisualAuditError",
    "audit_dataset",
    "load_audit_policy",
    "project_points",
    "render_bev_svg",
    "render_camera_overlay",
    "slice_values",
    "write_audit_bundle",
]

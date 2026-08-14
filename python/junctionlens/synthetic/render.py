"""Deterministic SVG projection of repository-owned synthetic graphs."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape

import numpy as np
import numpy.typing as npt

from junctionlens.data.geometry import project_vehicle_points
from junctionlens.synthetic.calibration import IMAGE_HEIGHT, IMAGE_WIDTH, CameraCalibration
from junctionlens.v1 import scene_control_graph_pb2 as scg

FloatArray = npt.NDArray[np.float64]


def _points(polyline: scg.Polyline3d) -> FloatArray:
    return np.asarray([(point.x, point.y, point.z) for point in polyline.points], dtype=np.float64)


def project_polyline(
    polyline: scg.Polyline3d,
    calibration: CameraCalibration,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Project one persisted polyline through the declared camera calibration."""
    return project_vehicle_points(
        _points(polyline),
        calibration.intrinsic,
        calibration.t_vehicle_camera,
    )


def _polyline_element(
    projected: FloatArray,
    valid: npt.NDArray[np.bool_],
    *,
    color: str,
    width: float,
    identity: str,
) -> str | None:
    visible = projected[valid]
    if len(visible) < 2:
        return None
    coordinates = " ".join(f"{horizontal:.3f},{vertical:.3f}" for horizontal, vertical in visible)
    return (
        f'<polyline data-id="{escape(identity)}" points="{coordinates}" '
        f'fill="none" stroke="{color}" stroke-width="{width:.1f}" />'
    )


def _lane_elements(
    graph: scg.SceneControlGraph,
    calibration: CameraCalibration,
) -> Iterable[str]:
    styles = (
        ("centerline", "#45d483", 2.5),
        ("left_boundary", "#8bd6ff", 1.5),
        ("right_boundary", "#8bd6ff", 1.5),
    )
    for lane in graph.lanes:
        for field, color, width in styles:
            projected, valid = project_polyline(getattr(lane, field), calibration)
            element = _polyline_element(
                projected,
                valid,
                color=color,
                width=width,
                identity=f"lane-{lane.node_id}-{field}",
            )
            if element is not None:
                yield element


def _area_elements(
    graph: scg.SceneControlGraph,
    calibration: CameraCalibration,
) -> Iterable[str]:
    for area in graph.road_areas:
        projected, valid = project_polyline(area.geometry, calibration)
        element = _polyline_element(
            projected,
            valid,
            color="#ffd166",
            width=3.0,
            identity=f"area-{area.node_id}",
        )
        if element is not None:
            yield element


def _control_elements(graph: scg.SceneControlGraph, camera_slot: int) -> Iterable[str]:
    for control in graph.traffic_controls:
        if control.source_camera != camera_slot:
            continue
        box = control.normalized_half_open_box
        horizontal = box.x_min * IMAGE_WIDTH
        vertical = box.y_min * IMAGE_HEIGHT
        width = (box.x_max - box.x_min) * IMAGE_WIDTH
        height = (box.y_max - box.y_min) * IMAGE_HEIGHT
        yield (
            f'<rect data-id="control-{control.node_id}" x="{horizontal:.3f}" '
            f'y="{vertical:.3f}" width="{width:.3f}" height="{height:.3f}" '
            'fill="none" stroke="#ff5c5c" stroke-width="2.0" />'
        )


def render_camera_svg(
    graph: scg.SceneControlGraph,
    calibration: CameraCalibration,
) -> bytes:
    """Render one deterministic unrestricted camera overlay as UTF-8 SVG."""
    elements = [
        *_lane_elements(graph, calibration),
        *_area_elements(graph, calibration),
        *_control_elements(graph, calibration.slot),
    ]
    body = "\n  ".join(elements)
    body_block = f"  {body}\n" if body else ""
    scene = escape(graph.frame_key.segment_id)
    payload = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{IMAGE_WIDTH}" '
        f'height="{IMAGE_HEIGHT}" viewBox="0 0 {IMAGE_WIDTH} {IMAGE_HEIGHT}" '
        f'data-scene="{scene}" data-camera="{escape(calibration.slug)}">\n'
        f'  <rect width="{IMAGE_WIDTH}" height="{IMAGE_HEIGHT}" fill="#101820" />\n'
        f"{body_block}"
        "</svg>\n"
    )
    return payload.encode("utf-8")

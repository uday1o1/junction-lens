#!/usr/bin/env python3
"""Project frozen OpenLane frames through the pinned official devkit API."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from openlanev2.lanesegment.dataset.frame import Frame
from openlanev2.lanesegment.io import io

MAX_SELECTOR_BYTES = 64 * 1024
MAX_FRAMES = 32


class ParityRunnerError(ValueError):
    """Raised when a parity selector or official frame is invalid."""


def _primitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, np.ndarray)):  # noqa: UP038 - Python 3.8
        return [_primitive(item) for item in value]
    if isinstance(value, (np.integer, int)):  # noqa: UP038 - Python 3.8
        return int(value)
    if isinstance(value, (np.floating, float)):  # noqa: UP038 - Python 3.8
        result = float(value)
        if not math.isfinite(result):
            raise ParityRunnerError("official frame contains a nonfinite number")
        return result
    if value is None or isinstance(value, (str, bool)):  # noqa: UP038 - Python 3.8
        return value
    raise ParityRunnerError(f"unsupported official value type: {type(value).__name__}")


def _selector(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if len(raw) > MAX_SELECTOR_BYTES:
        raise ParityRunnerError("selector exceeds the byte limit")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"frames", "schema_version"}:
        raise ParityRunnerError("selector has invalid top-level keys")
    if value["schema_version"] != "junctionlens.openlane-parity-selector.v1":
        raise ParityRunnerError("selector schema is unsupported")
    frames = value["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ParityRunnerError("selector must contain between one and 32 frames")
    result: list[dict[str, str]] = []
    for index, item in enumerate(frames):
        if not isinstance(item, dict) or set(item) != {"segment_id", "split_id", "timestamp"}:
            raise ParityRunnerError(f"selector frame {index} has invalid keys")
        parsed = {key: str(item[key]) for key in ("split_id", "segment_id", "timestamp")}
        if any(not value or Path(value).name != value for value in parsed.values()):
            raise ParityRunnerError(f"selector frame {index} has an unsafe identifier")
        int(parsed["timestamp"])
        result.append(parsed)
    if len({tuple(sorted(item.items())) for item in result}) != len(result):
        raise ParityRunnerError("selector contains duplicate frames")
    return result


def _safe_frame_path(root: Path, identifier: dict[str, str]) -> Path:
    relative = (
        Path(identifier["split_id"])
        / identifier["segment_id"]
        / "info"
        / f"{identifier['timestamp']}-ls.json"
    )
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ParityRunnerError("selected metadata escapes the dataset root") from error
    if not candidate.is_file():
        raise ParityRunnerError("selected metadata is not a regular file")
    return candidate


def _camera_projection(frame: Frame, root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera in frame.get_camera_list():
        if camera == "sd_map":
            continue
        image_path = Path(frame.get_image_path(camera)).resolve(strict=False)
        try:
            relative_image = image_path.relative_to(root).as_posix()
        except ValueError as error:
            raise ParityRunnerError("official image path escapes the dataset root") from error
        intrinsic = frame.get_intrinsic(camera)
        extrinsic = frame.get_extrinsic(camera)
        result[camera] = {
            "image_path": relative_image,
            "intrinsic": {
                "K": _primitive(intrinsic["K"]),
                "distortion": _primitive(intrinsic.get("distortion", [])),
                "model": str(intrinsic.get("model", "upstream-unspecified")),
            },
            "extrinsic": {
                "rotation": _primitive(extrinsic["rotation"]),
                "translation": _primitive(extrinsic["translation"]),
            },
        }
    return result


def _annotation_projection(frame: Frame) -> dict[str, Any] | None:
    annotation = frame.get_annotations()
    if annotation is None:
        return None
    return {
        "lane_segment": [
            {
                "id": str(lane["id"]),
                "centerline": _primitive(lane["centerline"]),
                "left_laneline": _primitive(lane["left_laneline"]),
                "right_laneline": _primitive(lane["right_laneline"]),
                "left_laneline_type": int(lane["left_laneline_type"]),
                "right_laneline_type": int(lane["right_laneline_type"]),
                "is_intersection_or_connector": bool(lane["is_intersection_or_connector"]),
            }
            for lane in frame.get_annotations_lane_segments()
        ],
        "traffic_element": [
            {
                "id": str(control["id"]),
                "category": int(control["category"]),
                "attribute": int(control["attribute"]),
                "points": _primitive(control["points"]),
            }
            for control in frame.get_annotations_traffic_elements()
        ],
        "area": [
            {
                "id": str(area["id"]),
                "category": int(area["category"]),
                "points": _primitive(area["points"]),
            }
            for area in frame.get_annotations_areas()
        ],
        "topology_lsls": _primitive(frame.get_annotations_topology_lsls()),
        "topology_lste": _primitive(frame.get_annotations_topology_lste()),
    }


def _project(root: Path, identifier: dict[str, str]) -> dict[str, Any]:
    metadata = io.json_load(str(_safe_frame_path(root, identifier)))
    frame = Frame(str(root), metadata)
    pose = frame.get_pose()
    return {
        "identifier": identifier,
        "metadata_version": str(metadata["version"]),
        "segment_id": str(metadata["segment_id"]),
        "timestamp": int(metadata["timestamp"]),
        "source_name": str(metadata["meta_data"]["source"]),
        "source_segment_id": str(metadata["meta_data"]["source_id"]),
        "cameras": _camera_projection(frame, root),
        "pose": {
            "rotation": _primitive(pose["rotation"]),
            "translation": _primitive(pose["translation"]),
        },
        "annotation": _annotation_projection(frame),
    }


def run(root: Path, selector_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ParityRunnerError("dataset root is not a directory")
    return {
        "schema_version": "junctionlens.openlane-official-projection.v1",
        "devkit_version": "2.1.0",
        "frames": [_project(root, identifier) for identifier in _selector(selector_path)],
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: adapter_parity_runner.py DATASET_ROOT SELECTOR.json", file=sys.stderr)
        return 2
    try:
        result = run(Path(sys.argv[1]), Path(sys.argv[2]))
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"adapter parity runner error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

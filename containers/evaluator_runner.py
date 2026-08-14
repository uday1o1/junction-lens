#!/usr/bin/env python3
"""Run untouched OpenLane-V2 v2.1 metrics on one validated JSON request."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy
import shapely
from custom_match import CustomMatchError, run_custom_match
from evaluator_payload import EvaluatorPayloadError, parse_payload_bytes
from openlanev2.lanesegment.evaluation import evaluate
from ortools import __version__ as ortools_version


def _numpy_annotations(annotation: dict[str, Any]) -> dict[str, Any]:
    for lane in annotation["lane_segment"]:
        lane["centerline"] = np.asarray(lane["centerline"], dtype=np.float64)
        lane["left_laneline"] = np.asarray(lane["left_laneline"], dtype=np.float64)
        lane["right_laneline"] = np.asarray(lane["right_laneline"], dtype=np.float64)
        if "confidence" in lane:
            lane["confidence"] = np.float32(lane["confidence"])
    for traffic in annotation["traffic_element"]:
        traffic["points"] = np.asarray(traffic["points"], dtype=np.float64)
        if "confidence" in traffic:
            traffic["confidence"] = np.float32(traffic["confidence"])
    for area in annotation["area"]:
        area["points"] = np.asarray(area["points"], dtype=np.float64)
        if "confidence" in area:
            area["confidence"] = np.float32(area["confidence"])
    lane_count = len(annotation["lane_segment"])
    traffic_count = len(annotation["traffic_element"])
    annotation["topology_lsls"] = np.asarray(annotation["topology_lsls"], dtype=np.float64).reshape(
        (lane_count, lane_count)
    )
    annotation["topology_lste"] = np.asarray(annotation["topology_lste"], dtype=np.float64).reshape(
        (lane_count, traffic_count)
    )
    return annotation


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, np.ndarray)):  # noqa: UP038 - Python 3.8
        return [_normalize(item) for item in value]
    if isinstance(value, (np.integer, int)):  # noqa: UP038 - Python 3.8
        return int(value)
    if isinstance(value, (np.floating, float)):  # noqa: UP038 - Python 3.8
        result = float(value)
        return result if math.isfinite(result) else None
    if value is None or isinstance(value, (str, bool)):  # noqa: UP038 - Python 3.8
        return value
    raise TypeError(f"unsupported evaluator result type: {type(value).__name__}")


def _matching(ground_truth: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for token, frame in sorted(predictions["results"].items()):
        annotation = frame["predictions"]
        truth = ground_truth[token]["annotation"]
        frame_matches: dict[str, Any] = {}
        for object_type, thresholds in {
            "lane_segment": (1.0, 2.0, 3.0),
            "traffic_element": (0.75,),
        }.items():
            object_matches: dict[str, Any] = {}
            for threshold in thresholds:
                prefix = f"{object_type}_{threshold}"
                object_matches[str(threshold)] = {
                    "confidence": annotation[f"{prefix}_confidence"],
                    "confidence_thresholds": annotation[f"{prefix}_confidence_thresholds"],
                    "ground_truth_ids": [item["id"] for item in truth[object_type]],
                    "idx_match_gt": annotation[f"{prefix}_idx_match_gt"],
                    "prediction_ids": [item["id"] for item in annotation[object_type]],
                }
            frame_matches[object_type] = object_matches
        frames[token] = frame_matches
    return _normalize(
        {
            "frames": frames,
            "schema_version": "openlane-v2.v2.1.threshold-matching.v1",
            "thresholds": {
                "lane_segment": [1.0, 2.0, 3.0],
                "traffic_element": [0.75],
            },
        }
    )


def run(path: Path) -> dict[str, Any]:
    raw_bytes = path.read_bytes()
    payload = parse_payload_bytes(raw_bytes)
    ground_truth = payload["ground_truth"]
    predictions = payload["predictions"]
    for frame in ground_truth.values():
        _numpy_annotations(frame["annotation"])
    for frame in predictions["results"].values():
        _numpy_annotations(frame["predictions"])
    official = evaluate(ground_truth, predictions, verbose=False)["OpenLane-V2 UniScore"]
    return {
        "environment": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_distribution": f"opencv-python-headless=={version('opencv-python-headless')}",
            "openlane_v2": "2.1.0",
            "ortools": ortools_version,
            "python": platform.python_version(),
            "scipy": scipy.__version__,
            "shapely": shapely.__version__,
        },
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "matching": _matching(ground_truth, predictions),
        "metrics": {
            "DET_a": _normalize(official["DET_a"]),
            "DET_l": _normalize(official["DET_l"]),
            "DET_t": _normalize(official["DET_t"]),
            "OLUS": _normalize(official["score"]),
            "TOP_ll": _normalize(official["TOP_ll"]),
            "TOP_lt": _normalize(official["TOP_lt"]),
        },
        "schema_version": "junctionlens.official-evaluator-output.v1",
    }


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--custom-match":
        try:
            output = run_custom_match(Path(sys.argv[2]))
        except (CustomMatchError, OSError, KeyError, TypeError, ValueError) as error:
            print(f"custom match error: {error}", file=sys.stderr)
            return 2
        print(json.dumps(output, allow_nan=False, separators=(",", ":"), sort_keys=True))
        return 0
    if len(sys.argv) != 2:
        print("usage: evaluator_runner.py [--custom-match] INPUT.json", file=sys.stderr)
        return 2
    try:
        output = run(Path(sys.argv[1]))
    except (EvaluatorPayloadError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"evaluator error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(output, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

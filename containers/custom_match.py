"""Strict CustomMatchV1 execution using pinned OpenLane-V2 distance primitives."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from openlanev2.lanesegment.evaluation.distance import (
    area_distance,
    lane_segment_distance,
    traffic_element_distance,
)

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 1_000
MAX_ITEMS = 256
THRESHOLDS = {"area": 1.0, "lane_segment": 2.0, "traffic_element": 0.75}
QUANTIZATION = {"area": 0.001, "lane_segment": 0.001, "traffic_element": 0.000001}


class CustomMatchError(ValueError):
    """Raised when CustomMatchV1 input or output cannot be trusted."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CustomMatchError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CustomMatchError(f"{path} must be an object with string keys")
    return value


def _exact(value: dict[str, Any], keys: set, path: str) -> None:
    if set(value) != keys:
        raise CustomMatchError(f"{path} has unexpected keys")


def _number(
    value: object,
    path: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CustomMatchError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CustomMatchError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise CustomMatchError(f"{path} is below its minimum")
    if maximum is not None and result > maximum:
        raise CustomMatchError(f"{path} exceeds its maximum")
    return result


def _integer(value: object, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CustomMatchError(f"{path} must be an integer in range")
    return value


def _points(value: object, path: str, dimensions: int, minimum: int) -> list[list[float]]:
    if not isinstance(value, list) or not minimum <= len(value) <= 256:
        raise CustomMatchError(f"{path} has an invalid point count")
    result = []
    for point_index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != dimensions:
            raise CustomMatchError(f"{path}[{point_index}] has invalid dimensions")
        result.append([_number(component, f"{path}[{point_index}]") for component in point])
    return result


def _geometry(value: object, object_type: str, path: str) -> dict[str, Any]:
    geometry = _object(value, path)
    if object_type == "lane_segment":
        _exact(geometry, {"centerline", "left_laneline", "right_laneline"}, path)
        return {
            key: np.asarray(_points(geometry[key], f"{path}.{key}", 3, 2), dtype=np.float64)
            for key in sorted(geometry)
        }
    if object_type == "traffic_element":
        _exact(geometry, {"points"}, path)
        points = _points(geometry["points"], f"{path}.points", 2, 2)
        if len(points) != 2 or points[1][0] <= points[0][0] or points[1][1] <= points[0][1]:
            raise CustomMatchError(f"{path} must contain one positive-area XYXY box")
        return {"points": np.asarray(points, dtype=np.float64)}
    _exact(geometry, {"points"}, path)
    return {
        "points": np.asarray(_points(geometry["points"], f"{path}.points", 3, 3), dtype=np.float64)
    }


def _source_id(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(ord(item) < 32 for item in value)
    ):
        raise CustomMatchError(f"{path} must be a nonempty printable source ID")
    return value


def _items(value: object, object_type: str, prediction: bool, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_ITEMS:
        raise CustomMatchError(f"{path} exceeds its bounded capacity")
    result = []
    identifiers = set()
    for index, raw in enumerate(value):
        item_path = f"{path}[{index}]"
        item = _object(raw, item_path)
        keys = (
            {"geometry", "prediction_id", "raw_confidence", "decoder_query_index"}
            if prediction
            else {"geometry", "source_id"}
        )
        _exact(item, keys, item_path)
        parsed = {"geometry": _geometry(item["geometry"], object_type, item_path + ".geometry")}
        if prediction:
            identifier = str(
                _integer(item["prediction_id"], item_path + ".prediction_id", 1, 2**64 - 1)
            )
            parsed.update(
                {
                    "prediction_id": identifier,
                    "raw_confidence": _number(
                        item["raw_confidence"], item_path + ".raw_confidence", 0.0, 1.0
                    ),
                    "decoder_query_index": _integer(
                        item["decoder_query_index"],
                        item_path + ".decoder_query_index",
                        0,
                        2**32 - 1,
                    ),
                }
            )
        else:
            identifier = _source_id(item["source_id"], item_path + ".source_id")
            parsed["source_id"] = identifier
        if identifier in identifiers:
            raise CustomMatchError(f"{path} contains duplicate identifiers")
        identifiers.add(identifier)
        result.append(parsed)
    return result


def _parse(raw_bytes: bytes) -> dict[str, Any]:
    if len(raw_bytes) > MAX_INPUT_BYTES:
        raise CustomMatchError("custom match input exceeds the byte limit")
    try:
        value = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                CustomMatchError(f"nonfinite JSON constant {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CustomMatchError(f"custom match input is not strict UTF-8 JSON: {error}") from error
    payload = _object(value, "payload")
    _exact(payload, {"frames", "schema_version"}, "payload")
    if payload["schema_version"] != "junctionlens.custom-match-input.v1":
        raise CustomMatchError("custom match input schema is unsupported")
    frames = payload["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise CustomMatchError("custom match input has an invalid frame count")
    tokens = set()
    parsed_frames = []
    for index, raw_frame in enumerate(frames):
        path = f"payload.frames[{index}]"
        frame = _object(raw_frame, path)
        _exact(frame, {"frame_token", "ground_truth", "predictions"}, path)
        token = _source_id(frame["frame_token"], path + ".frame_token")
        if token in tokens:
            raise CustomMatchError("custom match input contains duplicate frame tokens")
        tokens.add(token)
        parsed = {"frame_token": token}
        for side, prediction in (("ground_truth", False), ("predictions", True)):
            collection = _object(frame[side], path + "." + side)
            _exact(collection, set(THRESHOLDS), path + "." + side)
            parsed[side] = {
                object_type: _items(
                    collection[object_type],
                    object_type,
                    prediction,
                    path + "." + side + "." + object_type,
                )
                for object_type in sorted(THRESHOLDS)
            }
        parsed_frames.append(parsed)
    return {"frames": parsed_frames, "schema_version": payload["schema_version"]}


def _quantize(value: float, step: float) -> int:
    scaled = value / step
    return math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)


def _geometry_key(item: dict[str, Any], object_type: str) -> tuple[int, ...]:
    geometry = item["geometry"]
    keys = (
        ("centerline", "left_laneline", "right_laneline")
        if object_type == "lane_segment"
        else ("points",)
    )
    step = QUANTIZATION[object_type]
    return tuple(
        _quantize(float(value), step) for key in keys for value in geometry[key].reshape(-1)
    )


def _distance(ground_truth: dict[str, Any], prediction: dict[str, Any], object_type: str) -> float:
    function = {
        "area": area_distance,
        "lane_segment": lane_segment_distance,
        "traffic_element": traffic_element_distance,
    }[object_type]
    result = float(function(ground_truth["geometry"], prediction["geometry"]))
    if not math.isfinite(result) or result < 0.0:
        raise CustomMatchError("official distance primitive returned an invalid cost")
    return result


def _match_type(
    ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]], object_type: str
) -> dict[str, Any]:
    threshold = THRESHOLDS[object_type]
    ordered = sorted(
        predictions,
        key=lambda item: (
            -item["raw_confidence"],
            _geometry_key(item, object_type),
            item["decoder_query_index"],
        ),
    )
    available = {item["source_id"] for item in ground_truth}
    records = []
    for sorted_index, prediction in enumerate(ordered):
        costs = sorted(
            (
                (item["source_id"], _distance(item, prediction, object_type))
                for item in ground_truth
            ),
            key=lambda pair: (pair[1], pair[0]),
        )
        selected = None
        for source_id, cost in costs:
            if cost < threshold and source_id in available:
                selected = source_id
                available.remove(source_id)
                break
        candidates = []
        for source_id, cost in costs:
            if source_id == selected:
                reason = "SELECTED"
            elif cost >= threshold:
                reason = "OUTSIDE_THRESHOLD"
            elif source_id not in available:
                reason = "GROUND_TRUTH_ALREADY_MATCHED"
            else:
                reason = "HIGHER_COST_THAN_SELECTED"
            candidates.append(
                {
                    "cost": cost,
                    "ground_truth_source_id": source_id,
                    "passes_threshold": cost < threshold,
                    "rejection_reason": reason,
                }
            )
        if selected is not None:
            unmatched_reason = None
        elif not any(cost < threshold for _, cost in costs):
            unmatched_reason = "NO_CANDIDATE_INSIDE_THRESHOLD"
        else:
            unmatched_reason = "ALL_ELIGIBLE_GROUND_TRUTH_TAKEN"
        records.append(
            {
                "candidates": candidates,
                "decoder_query_index": prediction["decoder_query_index"],
                "geometry_key": list(_geometry_key(prediction, object_type)),
                "prediction_id": prediction["prediction_id"],
                "raw_confidence": prediction["raw_confidence"],
                "selected_ground_truth_source_id": selected,
                "sorted_index": sorted_index,
                "unmatched_reason": unmatched_reason,
            }
        )
    selected_sources = {
        record["selected_ground_truth_source_id"]
        for record in records
        if record["selected_ground_truth_source_id"] is not None
    }
    return {
        "ground_truth": [
            {
                "source_id": item["source_id"],
                "unmatched_reason": None
                if item["source_id"] in selected_sources
                else "NO_PREDICTION_SELECTED",
            }
            for item in sorted(ground_truth, key=lambda item: item["source_id"])
        ],
        "predictions": records,
        "threshold": threshold,
    }


def run_custom_match(path: Path) -> dict[str, Any]:
    raw_bytes = path.read_bytes()
    payload = _parse(raw_bytes)
    return {
        "frames": {
            frame["frame_token"]: {
                object_type: _match_type(
                    frame["ground_truth"][object_type],
                    frame["predictions"][object_type],
                    object_type,
                )
                for object_type in sorted(THRESHOLDS)
            }
            for frame in payload["frames"]
        },
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "policy": {
            "cost_owner": "OpenLane-V2 v2.1.0 pinned official distance primitives",
            "geometry_quantization": QUANTIZATION,
            "prediction_order": [
                "descending_raw_confidence",
                "quantized_geometry_key",
                "decoder_query_index",
            ],
            "threshold_comparison": "strictly_less_than",
            "thresholds": THRESHOLDS,
        },
        "schema_version": "junctionlens.custom-match.v1",
    }


__all__ = ["CustomMatchError", "run_custom_match"]

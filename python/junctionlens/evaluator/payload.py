"""Strict JSON contract shared with the isolated OpenLane-V2 evaluator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 1_000
MAX_LANES = 256
MAX_TRAFFIC_ELEMENTS = 256
MAX_AREAS = 128
MAX_POLYLINE_POINTS = 256


class EvaluatorPayloadError(ValueError):
    """Raised when evaluator input is unsafe or violates the frozen schema."""


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvaluatorPayloadError(f"{path} must be an object with string keys")
    return value


def _exact_keys(value: dict[str, Any], keys: set[str], path: str) -> None:
    if set(value) != keys:
        raise EvaluatorPayloadError(
            f"{path} keys must be exactly {sorted(keys)}; observed {sorted(value)}"
        )


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):  # noqa: UP038
        raise EvaluatorPayloadError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluatorPayloadError(f"{path} must be finite")
    return result


def _integer(value: object, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluatorPayloadError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise EvaluatorPayloadError(f"{path} must be between {minimum} and {maximum}")
    return value


def _points(
    value: object,
    path: str,
    *,
    dimensions: int,
    minimum_points: int,
    maximum_points: int = MAX_POLYLINE_POINTS,
) -> list[list[float]]:
    if not isinstance(value, list) or not minimum_points <= len(value) <= maximum_points:
        raise EvaluatorPayloadError(
            f"{path} must contain between {minimum_points} and {maximum_points} points"
        )
    result: list[list[float]] = []
    for point_index, raw_point in enumerate(value):
        if not isinstance(raw_point, list) or len(raw_point) != dimensions:
            raise EvaluatorPayloadError(f"{path}[{point_index}] must have {dimensions} values")
        result.append(
            [
                _finite_number(component, f"{path}[{point_index}][{component_index}]")
                for component_index, component in enumerate(raw_point)
            ]
        )
    return result


def _confidence(value: object, path: str) -> float:
    result = _finite_number(value, path)
    if not 0.0 <= result <= 1.0:
        raise EvaluatorPayloadError(f"{path} must be between zero and one")
    return result


def _matrix(
    value: object, rows: int, columns: int, path: str, prediction: bool
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise EvaluatorPayloadError(f"{path} must have {rows} rows")
    result: list[list[float]] = []
    for row_index, raw_row in enumerate(value):
        if not isinstance(raw_row, list) or len(raw_row) != columns:
            raise EvaluatorPayloadError(f"{path}[{row_index}] must have {columns} columns")
        row: list[float] = []
        for column_index, raw_value in enumerate(raw_row):
            current_path = f"{path}[{row_index}][{column_index}]"
            if prediction:
                row.append(_confidence(raw_value, current_path))
            else:
                row.append(_integer(raw_value, current_path, 0, 1))
        result.append(row)
    return result


def _lane(value: object, path: str, prediction: bool) -> dict[str, Any]:
    lane = _object(value, path)
    keys = {"id", "centerline", "left_laneline", "right_laneline"}
    if prediction:
        keys.add("confidence")
    _exact_keys(lane, keys, path)
    lane_id = _integer(lane["id"], f"{path}.id", 0, 2**63 - 1)
    result: dict[str, Any] = {
        "id": lane_id,
        "centerline": _points(
            lane["centerline"], f"{path}.centerline", dimensions=3, minimum_points=2
        ),
        "left_laneline": _points(
            lane["left_laneline"], f"{path}.left_laneline", dimensions=3, minimum_points=2
        ),
        "right_laneline": _points(
            lane["right_laneline"], f"{path}.right_laneline", dimensions=3, minimum_points=2
        ),
    }
    if prediction:
        result["confidence"] = _confidence(lane["confidence"], f"{path}.confidence")
    return result


def _traffic(value: object, path: str, prediction: bool) -> dict[str, Any]:
    traffic = _object(value, path)
    keys = {"id", "attribute", "points"}
    if prediction:
        keys.add("confidence")
    _exact_keys(traffic, keys, path)
    result: dict[str, Any] = {
        "id": _integer(traffic["id"], f"{path}.id", 0, 2**63 - 1),
        "attribute": _integer(traffic["attribute"], f"{path}.attribute", 0, 12),
        "points": _points(
            traffic["points"], f"{path}.points", dimensions=2, minimum_points=2, maximum_points=2
        ),
    }
    (x1, y1), (x2, y2) = result["points"]
    if x2 <= x1 or y2 <= y1:
        raise EvaluatorPayloadError(f"{path}.points must be a positive-area XYXY box")
    if prediction:
        result["confidence"] = _confidence(traffic["confidence"], f"{path}.confidence")
    return result


def _area(value: object, path: str, prediction: bool) -> dict[str, Any]:
    area = _object(value, path)
    keys = {"id", "category", "points"}
    if prediction:
        keys.add("confidence")
    _exact_keys(area, keys, path)
    result: dict[str, Any] = {
        "id": _integer(area["id"], f"{path}.id", 0, 2**63 - 1),
        "category": _integer(area["category"], f"{path}.category", 1, 2),
        "points": _points(area["points"], f"{path}.points", dimensions=3, minimum_points=3),
    }
    if prediction:
        result["confidence"] = _confidence(area["confidence"], f"{path}.confidence")
    return result


def _unique_ids(items: list[dict[str, Any]], path: str) -> None:
    identifiers = [item["id"] for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise EvaluatorPayloadError(f"{path} contains duplicate IDs")


def _annotation(value: object, path: str, prediction: bool) -> dict[str, Any]:
    annotation = _object(value, path)
    _exact_keys(
        annotation,
        {"lane_segment", "traffic_element", "area", "topology_lsls", "topology_lste"},
        path,
    )
    limits = {
        "lane_segment": MAX_LANES,
        "traffic_element": MAX_TRAFFIC_ELEMENTS,
        "area": MAX_AREAS,
    }
    parsed: dict[str, list[dict[str, Any]]] = {}
    parsers = {"lane_segment": _lane, "traffic_element": _traffic, "area": _area}
    for object_type, parser in parsers.items():
        raw_items = annotation[object_type]
        if not isinstance(raw_items, list) or len(raw_items) > limits[object_type]:
            raise EvaluatorPayloadError(f"{path}.{object_type} exceeds its bounded capacity")
        parsed[object_type] = [
            parser(raw_item, f"{path}.{object_type}[{index}]", prediction)
            for index, raw_item in enumerate(raw_items)
        ]
        _unique_ids(parsed[object_type], f"{path}.{object_type}")
    lane_count = len(parsed["lane_segment"])
    traffic_count = len(parsed["traffic_element"])
    return {
        **parsed,
        "topology_lsls": _matrix(
            annotation["topology_lsls"], lane_count, lane_count, f"{path}.topology_lsls", prediction
        ),
        "topology_lste": _matrix(
            annotation["topology_lste"],
            lane_count,
            traffic_count,
            f"{path}.topology_lste",
            prediction,
        ),
    }


def validate_payload(value: object) -> dict[str, Any]:
    """Validate and copy a bounded evaluator request into primitive Python values."""
    payload = _object(value, "payload")
    _exact_keys(payload, {"schema_version", "ground_truth", "predictions"}, "payload")
    if payload["schema_version"] != "junctionlens.official-evaluator-input.v1":
        raise EvaluatorPayloadError("payload.schema_version is unsupported")
    raw_ground_truth = _object(payload["ground_truth"], "payload.ground_truth")
    raw_predictions = _object(payload["predictions"], "payload.predictions")
    _exact_keys(raw_predictions, {"results"}, "payload.predictions")
    raw_results = _object(raw_predictions["results"], "payload.predictions.results")
    if set(raw_ground_truth) != set(raw_results):
        raise EvaluatorPayloadError("ground-truth and prediction frame keys differ")
    if not 1 <= len(raw_ground_truth) <= MAX_FRAMES:
        raise EvaluatorPayloadError(f"payload must contain between one and {MAX_FRAMES} frames")
    ground_truth: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    for token in sorted(raw_ground_truth):
        if not token or len(token) > 256 or any(ord(character) < 0x20 for character in token):
            raise EvaluatorPayloadError("frame tokens must be nonempty printable strings")
        ground_frame = _object(raw_ground_truth[token], f"payload.ground_truth[{token!r}]")
        prediction_frame = _object(raw_results[token], f"payload.predictions.results[{token!r}]")
        _exact_keys(ground_frame, {"annotation"}, f"payload.ground_truth[{token!r}]")
        _exact_keys(prediction_frame, {"predictions"}, f"payload.predictions.results[{token!r}]")
        ground_truth[token] = {
            "annotation": _annotation(
                ground_frame["annotation"], f"payload.ground_truth[{token!r}].annotation", False
            )
        }
        predictions[token] = {
            "predictions": _annotation(
                prediction_frame["predictions"],
                f"payload.predictions.results[{token!r}].predictions",
                True,
            )
        }
    return {
        "schema_version": payload["schema_version"],
        "ground_truth": ground_truth,
        "predictions": {"results": predictions},
    }


def load_payload(path: Path) -> dict[str, Any]:
    """Read one bounded UTF-8 JSON file and validate its complete structure."""
    stat = path.stat()
    if not path.is_file() or stat.st_size > MAX_INPUT_BYTES:
        raise EvaluatorPayloadError(
            f"input must be a regular file no larger than {MAX_INPUT_BYTES} bytes"
        )
    return parse_payload_bytes(path.read_bytes())


def parse_payload_bytes(raw_bytes: bytes) -> dict[str, Any]:
    """Parse and validate the exact bounded bytes passed to the evaluator."""
    if len(raw_bytes) > MAX_INPUT_BYTES:
        raise EvaluatorPayloadError(f"input must be no larger than {MAX_INPUT_BYTES} bytes")
    try:
        source = raw_bytes.decode("utf-8")
        value = json.loads(
            source,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: _reject_constant(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluatorPayloadError(f"input is not valid strict UTF-8 JSON: {error}") from error
    return validate_payload(value)


def _reject_constant(value: str) -> None:
    raise EvaluatorPayloadError(f"JSON constant {value} is not permitted")


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluatorPayloadError(f"duplicate JSON object key is not permitted: {key}")
        result[key] = value
    return result

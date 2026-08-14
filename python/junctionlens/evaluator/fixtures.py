"""Frozen evaluator fixture materialization and expected-component checks."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from junctionlens.evaluator.payload import validate_payload
from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object_path,
    load_yaml_object_path,
)


class FixtureError(ValueError):
    """Raised when a frozen evaluator fixture declaration is invalid."""


_METRICS = {"DET_a", "DET_l", "DET_t", "OLUS", "TOP_ll", "TOP_lt"}
_TRANSFORMS = {
    "adversarial_high_confidence_false_positive",
    "duplicate_confidence",
    "empty_predictions",
    "empty_scene",
    "identity",
    "partial_predictions",
    "permute_predictions",
    "permute_predictions_without_topology",
    "replacements",
}


def _decode_pointer_token(token: str) -> str:
    result = token.replace("~1", "/").replace("~0", "~")
    if "~" in result:
        raise FixtureError("JSON pointer contains an invalid escape")
    return result


def _replace(payload: object, pointer: str, value: object) -> None:
    if not pointer.startswith("/"):
        raise FixtureError("replacement path must be an absolute JSON pointer")
    tokens = [_decode_pointer_token(token) for token in pointer[1:].split("/")]
    if not tokens:
        raise FixtureError("replacement path cannot target the document root")
    parent: object = payload
    for token in tokens[:-1]:
        if isinstance(parent, dict):
            if token not in parent:
                raise FixtureError(f"replacement path component does not exist: {token}")
            parent = parent[token]
        elif isinstance(parent, list):
            try:
                parent = parent[int(token)]
            except (ValueError, IndexError) as error:
                raise FixtureError(f"replacement list index is invalid: {token}") from error
        else:
            raise FixtureError("replacement path descends through a scalar")
    final = tokens[-1]
    if isinstance(parent, dict):
        if final not in parent:
            raise FixtureError(f"replacement target does not exist: {final}")
        parent[final] = copy.deepcopy(value)
    elif isinstance(parent, list):
        try:
            parent[int(final)] = copy.deepcopy(value)
        except (ValueError, IndexError) as error:
            raise FixtureError(f"replacement list index is invalid: {final}") from error
    else:
        raise FixtureError("replacement target parent is a scalar")


def _only_frame(payload: dict[str, Any], section: str) -> dict[str, Any]:
    frames = payload[section]
    if section == "predictions":
        frames = frames["results"]
    if not isinstance(frames, dict) or len(frames) != 1:
        raise FixtureError(f"fixture transform requires exactly one {section} frame")
    frame = next(iter(frames.values()))
    annotation_key = "annotation" if section == "ground_truth" else "predictions"
    return cast(dict[str, Any], frame[annotation_key])


def _empty_annotation(annotation: dict[str, Any]) -> None:
    annotation["lane_segment"] = []
    annotation["traffic_element"] = []
    annotation["area"] = []
    annotation["topology_lsls"] = []
    annotation["topology_lste"] = []


def _permute_predictions(annotation: dict[str, Any], *, topology: bool) -> None:
    lane_order = list(reversed(range(len(annotation["lane_segment"]))))
    traffic_order = list(reversed(range(len(annotation["traffic_element"]))))
    area_order = list(reversed(range(len(annotation["area"]))))
    lanes = annotation["lane_segment"]
    traffic = annotation["traffic_element"]
    areas = annotation["area"]
    lane_topology = annotation["topology_lsls"]
    control_topology = annotation["topology_lste"]
    annotation["lane_segment"] = [lanes[index] for index in lane_order]
    annotation["traffic_element"] = [traffic[index] for index in traffic_order]
    annotation["area"] = [areas[index] for index in area_order]
    if topology:
        annotation["topology_lsls"] = [
            [lane_topology[source][target] for target in lane_order] for source in lane_order
        ]
        annotation["topology_lste"] = [
            [control_topology[source][target] for target in traffic_order] for source in lane_order
        ]


def _partial_predictions(annotation: dict[str, Any]) -> None:
    annotation["lane_segment"] = annotation["lane_segment"][:1]
    annotation["traffic_element"] = annotation["traffic_element"][:7]
    annotation["area"] = annotation["area"][:1]
    annotation["topology_lsls"] = [[annotation["topology_lsls"][0][0]]]
    annotation["topology_lste"] = [annotation["topology_lste"][0][:7]]


def _adversarial_false_positive(annotation: dict[str, Any]) -> None:
    false_lane = copy.deepcopy(annotation["lane_segment"][0])
    false_lane["id"] = 1999
    false_lane["confidence"] = 1.0
    for geometry in ("centerline", "left_laneline", "right_laneline"):
        false_lane[geometry] = [
            [point[0] + 50.0, point[1] + 50.0, point[2]] for point in false_lane[geometry]
        ]
    old_lane_topology = annotation["topology_lsls"]
    old_control_topology = annotation["topology_lste"]
    annotation["lane_segment"] = [false_lane, *annotation["lane_segment"]]
    annotation["topology_lsls"] = [
        [0.01, 0.01, 0.01],
        [0.01, *old_lane_topology[0]],
        [0.01, *old_lane_topology[1]],
    ]
    annotation["topology_lste"] = [
        [0.01] * len(annotation["traffic_element"]),
        *old_control_topology,
    ]


def _apply_transform(payload: dict[str, Any], transform: object) -> None:
    if not isinstance(transform, dict) or "kind" not in transform:
        raise FixtureError("fixture transform must declare a kind")
    kind = transform["kind"]
    if not isinstance(kind, str) or kind not in _TRANSFORMS:
        raise FixtureError(f"unsupported fixture transform: {kind}")
    expected_keys = {"kind", "replacements"} if kind == "replacements" else {"kind"}
    if set(transform) != expected_keys:
        raise FixtureError(f"fixture transform {kind} has invalid keys")
    prediction = _only_frame(payload, "predictions")
    if kind == "identity":
        return
    if kind == "replacements":
        replacements = transform["replacements"]
        if not isinstance(replacements, list):
            raise FixtureError("replacement transform must contain a list")
        for replacement in replacements:
            if not isinstance(replacement, dict) or set(replacement) != {"path", "value"}:
                raise FixtureError("fixture case has an invalid replacement")
            _replace(payload, replacement["path"], replacement["value"])
        return
    if kind == "empty_scene":
        _empty_annotation(_only_frame(payload, "ground_truth"))
        _empty_annotation(prediction)
    elif kind == "empty_predictions":
        _empty_annotation(prediction)
    elif kind == "partial_predictions":
        _partial_predictions(prediction)
    elif kind == "duplicate_confidence":
        for object_type in ("lane_segment", "traffic_element", "area"):
            for item in prediction[object_type]:
                item["confidence"] = 0.75
    elif kind == "permute_predictions":
        _permute_predictions(prediction, topology=True)
    elif kind == "permute_predictions_without_topology":
        _permute_predictions(prediction, topology=False)
    elif kind == "adversarial_high_confidence_false_positive":
        _adversarial_false_positive(prediction)


def load_cases(fixtures_root: Path) -> Mapping[str, Mapping[str, Any]]:
    """Load and validate the declarative fixed case set."""
    try:
        manifest = load_yaml_object_path(
            fixtures_root / "cases.yaml",
            "evaluator fixture manifest",
            ParseLimits(max_bytes=1024 * 1024, max_depth=24, max_nodes=100_000),
        )
    except ParseBoundaryError as error:
        raise FixtureError(str(error)) from error
    if set(manifest) != {"base", "cases", "schema_version"}:
        raise FixtureError("fixture manifest has invalid top-level keys")
    if manifest["schema_version"] != "junctionlens.official-evaluator-cases.v1":
        raise FixtureError("fixture manifest schema is unsupported")
    base_name = manifest["base"]
    if not isinstance(base_name, str) or Path(base_name).name != base_name:
        raise FixtureError("fixture base must be one local filename")
    try:
        base = load_json_object_path(
            fixtures_root / base_name,
            "evaluator fixture base",
            ParseLimits(
                max_bytes=16 * 1024 * 1024,
                max_depth=32,
                max_nodes=1_000_000,
                max_container_items=100_000,
            ),
        )
    except ParseBoundaryError as error:
        raise FixtureError(str(error)) from error
    validate_payload(base)
    raw_cases = manifest["cases"]
    if not isinstance(raw_cases, dict) or not raw_cases:
        raise FixtureError("fixture manifest has no cases")
    result: dict[str, Mapping[str, Any]] = {}
    for name, raw_case in raw_cases.items():
        if not isinstance(name, str) or not isinstance(raw_case, dict):
            raise FixtureError("fixture case declaration is invalid")
        if set(raw_case) != {"expected_changed", "expected_metrics", "transform"}:
            raise FixtureError(f"fixture case {name} has invalid keys")
        expected_changed = raw_case["expected_changed"]
        expected_metrics = raw_case["expected_metrics"]
        if not isinstance(expected_changed, list) or not isinstance(expected_metrics, dict):
            raise FixtureError(f"fixture case {name} has invalid declarations")
        if (
            not all(isinstance(item, str) for item in expected_changed)
            or not set(expected_changed) <= _METRICS
            or set(expected_metrics) != _METRICS
            or any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, int | float))
                for value in expected_metrics.values()
            )
        ):
            raise FixtureError(f"fixture case {name} has invalid metric expectations")
        payload = copy.deepcopy(base)
        _apply_transform(payload, raw_case["transform"])
        result[name] = {
            "expected_changed": tuple(expected_changed),
            "expected_metrics": dict(expected_metrics),
            "payload": validate_payload(payload),
        }
    return result

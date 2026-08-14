"""Frozen evaluator fixture materialization and expected-component checks."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from junctionlens.evaluator.payload import validate_payload


class FixtureError(ValueError):
    """Raised when a frozen evaluator fixture declaration is invalid."""


_METRICS = {"DET_a", "DET_l", "DET_t", "OLUS", "TOP_ll", "TOP_lt"}


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


def load_cases(fixtures_root: Path) -> Mapping[str, Mapping[str, Any]]:
    """Load and validate the declarative fixed case set."""
    with (fixtures_root / "cases.yaml").open(encoding="utf-8") as source:
        manifest = yaml.safe_load(source)
    if not isinstance(manifest, dict) or set(manifest) != {"base", "cases", "schema_version"}:
        raise FixtureError("fixture manifest has invalid top-level keys")
    if manifest["schema_version"] != "junctionlens.official-evaluator-cases.v1":
        raise FixtureError("fixture manifest schema is unsupported")
    base_name = manifest["base"]
    if not isinstance(base_name, str) or Path(base_name).name != base_name:
        raise FixtureError("fixture base must be one local filename")
    with (fixtures_root / base_name).open(encoding="utf-8") as source:
        base = json.load(source)
    validate_payload(base)
    raw_cases = manifest["cases"]
    if not isinstance(raw_cases, dict) or not raw_cases:
        raise FixtureError("fixture manifest has no cases")
    result: dict[str, Mapping[str, Any]] = {}
    for name, raw_case in raw_cases.items():
        if not isinstance(name, str) or not isinstance(raw_case, dict):
            raise FixtureError("fixture case declaration is invalid")
        if set(raw_case) != {"expected_changed", "expected_metrics", "replacements"}:
            raise FixtureError(f"fixture case {name} has invalid keys")
        replacements = raw_case["replacements"]
        expected_changed = raw_case["expected_changed"]
        expected_metrics = raw_case["expected_metrics"]
        if (
            not isinstance(replacements, list)
            or not isinstance(expected_changed, list)
            or not isinstance(expected_metrics, dict)
        ):
            raise FixtureError(f"fixture case {name} has invalid declarations")
        if (
            not all(isinstance(item, str) for item in expected_changed)
            or not set(expected_changed) <= _METRICS
            or set(expected_metrics) != _METRICS
            or any(
                isinstance(value, bool) or not isinstance(value, int | float)
                for value in expected_metrics.values()
            )
        ):
            raise FixtureError(f"fixture case {name} has invalid metric expectations")
        payload = copy.deepcopy(base)
        for replacement in replacements:
            if not isinstance(replacement, dict) or set(replacement) != {"path", "value"}:
                raise FixtureError(f"fixture case {name} has an invalid replacement")
            _replace(payload, replacement["path"], replacement["value"])
        result[name] = {
            "expected_changed": tuple(expected_changed),
            "expected_metrics": dict(expected_metrics),
            "payload": validate_payload(payload),
        }
    return result

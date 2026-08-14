#!/usr/bin/env python3
"""Exercise fixed official-evaluator fixtures through the public CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PYTHON_SOURCE = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from junctionlens.evaluator.fixtures import load_cases  # noqa: E402
from junctionlens.evaluator.official import (  # noqa: E402
    evaluator_container_command,
    inspect_evaluator_image,
    load_evaluator_image_contract,
    validate_evaluator_output,
)


class EvaluatorVerificationError(RuntimeError):
    """Raised when a fixed fixture violates its declared behavior."""


def _run_cli(root: Path, request: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(root / ".tools/bin/uv"),
            "run",
            "--locked",
            "junctionlens",
            "evaluate",
            "--input",
            str(request),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise EvaluatorVerificationError(
            f"public evaluator CLI failed ({result.returncode}): {result.stderr[:2048]}"
        )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise EvaluatorVerificationError("public evaluator CLI did not return an object")
    return value


def _run_direct(
    root: Path,
    request: Path,
    payload: dict[str, Any],
    reference: str,
) -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        raise EvaluatorVerificationError("Docker CLI is unavailable")
    result = subprocess.run(
        evaluator_container_command(docker, reference, request),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise EvaluatorVerificationError(
            f"direct compatibility evaluator failed ({result.returncode}): {result.stderr[:2048]}"
        )
    return validate_evaluator_output(
        result.stdout,
        hashlib.sha256(request.read_bytes()).hexdigest(),
        payload,
    )


def _assert_same_returned_json(
    direct: object,
    wrapper: object,
    path: str,
) -> None:
    if direct is None or wrapper is None:
        if direct is not wrapper:
            raise EvaluatorVerificationError(f"{path} differs in normalized NaN behavior")
        return
    if (
        isinstance(direct, int | float)
        and not isinstance(direct, bool)
        and isinstance(wrapper, int | float)
        and not isinstance(wrapper, bool)
    ):
        if not math.isfinite(float(direct)) or not math.isfinite(float(wrapper)):
            raise EvaluatorVerificationError(f"{path} contains a nonfinite JSON number")
        if abs(float(direct) - float(wrapper)) > 1e-12:
            raise EvaluatorVerificationError(
                f"{path} differs between compatibility and modern wrapper outputs"
            )
        return
    if isinstance(direct, dict) and isinstance(wrapper, dict):
        if set(direct) != set(wrapper):
            raise EvaluatorVerificationError(f"{path} object keys differ")
        for key in sorted(direct):
            _assert_same_returned_json(direct[key], wrapper[key], f"{path}.{key}")
        return
    if isinstance(direct, list) and isinstance(wrapper, list):
        if len(direct) != len(wrapper):
            raise EvaluatorVerificationError(f"{path} array lengths differ")
        for index, (direct_item, wrapper_item) in enumerate(zip(direct, wrapper, strict=True)):
            _assert_same_returned_json(direct_item, wrapper_item, f"{path}[{index}]")
        return
    if type(direct) is not type(wrapper) or direct != wrapper:
        raise EvaluatorVerificationError(f"{path} differs between returned JSON values")


def _matching_sha256(output: dict[str, Any]) -> str:
    encoded = json.dumps(
        output["matching"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolved_matches(
    output: dict[str, Any],
    object_type: str,
    threshold: str,
) -> set[tuple[int, int | None]]:
    frames = output["matching"]["frames"]
    if len(frames) != 1:
        raise EvaluatorVerificationError("association control requires one frozen frame")
    frame = next(iter(frames.values()))
    artifact = frame[object_type][threshold]
    ground_ids = artifact["ground_truth_ids"]
    return {
        (
            prediction_id,
            None if index is None else ground_ids[int(index)],
        )
        for prediction_id, index in zip(
            artifact["prediction_ids"], artifact["idx_match_gt"], strict=True
        )
    }


def verify() -> dict[str, Any]:
    """Prove exact official outputs, wrapper parity, and permutation behavior."""
    root = Path(__file__).resolve().parents[2]
    cases = load_cases(root / "tests/fixtures/evaluator")
    contract = load_evaluator_image_contract(root)
    reference = str(contract["local_reference"])
    inspect_evaluator_image(
        root,
        reference,
        str(contract["config_sha256"]),
        str(contract["platform_manifest_sha256"]),
    )
    results: dict[str, Any] = {}
    cache_root = root / ".cache/evaluator-fixtures"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="junctionlens-evaluator-", dir=cache_root) as temp:
        temp_root = Path(temp)
        for name, case in cases.items():
            request = temp_root / f"{name}.json"
            request.write_text(
                json.dumps(case["payload"], sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            direct = _run_direct(root, request, case["payload"], reference)
            wrapper = _run_cli(root, request)
            _assert_same_returned_json(direct, wrapper, f"case.{name}")
            results[name] = wrapper
    perfect = results["perfect"]["metrics"]
    if set(perfect.values()) != {1.0}:
        raise EvaluatorVerificationError(f"perfect fixture is not optimal: {perfect!r}")
    for name, case in cases.items():
        metrics = results[name]["metrics"]
        for component, expected in case["expected_metrics"].items():
            observed = metrics[component]
            if expected is None or observed is None:
                if expected is not observed:
                    raise EvaluatorVerificationError(
                        f"{name} metric {component} normalized empty behavior drifted"
                    )
            elif abs(observed - expected) > 1e-12:
                raise EvaluatorVerificationError(
                    f"{name} metric {component} drifted: expected {expected}, observed {observed}"
                )
        if name == "perfect":
            continue
        for component in case["expected_changed"]:
            if not metrics[component] < perfect[component]:
                raise EvaluatorVerificationError(
                    f"{name} did not reduce expected component {component}: {metrics!r}"
                )
    detection_metrics = {"DET_a", "DET_l", "DET_t"}
    for name in ("corrupt_lane_topology", "corrupt_control_topology"):
        metrics = results[name]["metrics"]
        if any(metrics[component] != perfect[component] for component in detection_metrics):
            raise EvaluatorVerificationError(f"{name} unexpectedly changed detection: {metrics!r}")
    for name in ("duplicate_confidence", "permuted_order"):
        if results[name]["metrics"] != perfect:
            raise EvaluatorVerificationError(f"{name} is not metric invariant")
    for object_type, threshold in (("lane_segment", "2.0"), ("traffic_element", "0.75")):
        if _resolved_matches(results["permuted_order"], object_type, threshold) != (
            _resolved_matches(results["perfect"], object_type, threshold)
        ):
            raise EvaluatorVerificationError(
                f"correct {object_type} permutation changed resolved upstream associations"
            )
    broken = results["adversarial_unpermuted_topology"]["metrics"]
    if any(broken[component] != perfect[component] for component in detection_metrics):
        raise EvaluatorVerificationError("unpermuted topology control changed node detection")
    if not broken["TOP_ll"] < perfect["TOP_ll"] or not broken["TOP_lt"] < perfect["TOP_lt"]:
        raise EvaluatorVerificationError("unpermuted topology control did not degrade topology")
    adversarial_lane = next(
        iter(results["adversarial_high_confidence_false_positive"]["matching"]["frames"].values())
    )["lane_segment"]["2.0"]
    if adversarial_lane["idx_match_gt"][0] is not None:
        raise EvaluatorVerificationError("high-confidence false positive was unexpectedly matched")
    return {
        "cases": {name: result["metrics"] for name, result in sorted(results.items())},
        "matching_sha256": {
            name: _matching_sha256(result) for name, result in sorted(results.items())
        },
        "parity": {
            "absolute_tolerance": 1e-12,
            "compatibility_container_checks": len(cases),
            "modern_wrapper_checks": len(cases),
            "nan_normalization": "JSON null",
            "order_invariance_control": "PASSED",
            "unpermuted_topology_control": "PASSED",
        },
        "status": "PASSED",
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        arguments = _arguments()
        report = json.dumps(verify(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(report, encoding="utf-8")
        print(report, end="")
    except (EvaluatorVerificationError, OSError, ValueError) as error:
        print(f"evaluator verification error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

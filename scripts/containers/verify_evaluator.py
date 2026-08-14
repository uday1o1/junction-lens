#!/usr/bin/env python3
"""Exercise fixed official-evaluator fixtures through the public CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PYTHON_SOURCE = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from junctionlens.evaluator.fixtures import load_cases  # noqa: E402


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


def verify() -> dict[str, Any]:
    """Run every fixed case and check its expected metric component effects."""
    root = Path(__file__).resolve().parents[2]
    cases = load_cases(root / "tests/fixtures/evaluator")
    results: dict[str, Any] = {}
    cache_root = root / ".cache/evaluator-fixtures"
    cache_root.mkdir(parents=True, exist_ok=True)
    benchmark_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="junctionlens-evaluator-", dir=cache_root) as temp:
        temp_root = Path(temp)
        for name, case in cases.items():
            request = temp_root / f"{name}.json"
            request.write_text(
                json.dumps(case["payload"], sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            results[name] = _run_cli(root, request)
    benchmark_seconds = time.perf_counter() - benchmark_start
    perfect = results["perfect"]["metrics"]
    if set(perfect.values()) != {1.0}:
        raise EvaluatorVerificationError(f"perfect fixture is not optimal: {perfect!r}")
    for name, case in cases.items():
        metrics = results[name]["metrics"]
        for component, expected in case["expected_metrics"].items():
            if abs(metrics[component] - expected) > 1e-12:
                raise EvaluatorVerificationError(
                    f"{name} metric {component} drifted: expected {expected}, "
                    f"observed {metrics[component]}"
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
    return {
        "benchmark": {
            "clock": "time.perf_counter",
            "duration_seconds": benchmark_seconds,
            "profile": "public-cli-cold-container-per-segment",
            "segments": len(cases),
            "segments_per_second": len(cases) / benchmark_seconds,
        },
        "cases": {name: result["metrics"] for name, result in sorted(results.items())},
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

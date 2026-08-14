#!/usr/bin/env python3
"""Capture bounded non-benchmark Nsight and ONNX Runtime profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from junctionlens.model.profile import load_m0_profile
from junctionlens.runtime import load_runtime_qualification
from scripts.gpu.qualify_runtime import write_synthetic_runtime_fixture


class ProfilerError(RuntimeError):
    """Raised when profiler evidence is unavailable or ambiguous."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact(value: str, replacements: dict[str, str]) -> str:
    result = value
    for sensitive, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if sensitive:
            result = result.replace(sensitive, replacement)
    return result


def _runtime_command(arguments: argparse.Namespace, input_list: Path, raw: Path) -> list[str]:
    config = load_runtime_qualification(arguments.config)
    profile = load_m0_profile(arguments.profile)
    return [
        str(arguments.runtime.resolve()),
        "benchmark",
        "--model",
        str(arguments.model.resolve()),
        "--expected-profile-sha256",
        profile.canonical_sha256(),
        "--input-list",
        str(input_list.resolve()),
        "--asset-root",
        str(input_list.parent.resolve()),
        "--timing-output",
        str(raw.resolve()),
        "--git-commit",
        arguments.source_commit,
        "--configuration-sha256",
        profile.canonical_sha256(),
        "--runtime-build-sha256",
        _sha256(arguments.runtime),
        "--provider-profile",
        "cuda",
        "--provider-log-output",
        str((raw.parent / "provider.raw.log").resolve()),
        "--device-id",
        "0",
        "--warmup-frames",
        str(config.profiling.warmup_frames),
        "--measured-frames",
        str(config.profiling.measured_frames),
        "--stability-frames",
        "0",
        "--memory-sample-period",
        "1",
        "--input-profile",
        "predecoded",
        "--profiler-run",
        "--onnx-profile-prefix",
        str((raw.parent / "onnx-profile").resolve()),
    ]


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run one bounded profiler capture that cannot be used as benchmark evidence."""
    nsys = shutil.which("nsys")
    if nsys is None:
        raise ProfilerError("nsys is required for the target profiler phase")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "fixture"
    fixture.mkdir()
    input_list = write_synthetic_runtime_fixture(fixture)
    raw = output / "profiler-native-raw.json"
    trace_prefix = output / "junctionlens-runtime"
    config = load_runtime_qualification(arguments.config)
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=false",
        "--duration",
        str(config.profiling.maximum_duration_seconds),
        "--output",
        str(trace_prefix),
        *_runtime_command(arguments, input_list, raw),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = arguments.gpu_uuid
    completed = subprocess.run(
        command,
        cwd=arguments.project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=config.profiling.maximum_duration_seconds + 60,
    )
    replacements = {
        str(arguments.project_root.resolve()): "<SOURCE_ROOT>",
        str(output): "<OUTPUT_ROOT>",
        str(Path.home()): "<REMOTE_HOME>",
    }
    (output / "nsys.stdout.log").write_text(
        _redact(completed.stdout, replacements), encoding="utf-8"
    )
    (output / "nsys.stderr.log").write_text(
        _redact(completed.stderr, replacements), encoding="utf-8"
    )
    traces = sorted(output.glob("junctionlens-runtime*.nsys-rep"))
    onnx_profiles = sorted(output.glob("onnx-profile*.json"))
    if completed.returncode != 0 or len(traces) != 1 or len(onnx_profiles) != 1:
        raise ProfilerError("profiler run did not produce exactly one Nsight and ONNX trace")
    raw_report = json.loads(raw.read_text(encoding="utf-8"))
    if raw_report.get("profiler_run") is not True or raw_report.get("publishable") is not False:
        raise ProfilerError("profiler run was not isolated from benchmark evidence")
    report = {
        "nsight_systems_sha256": _sha256(traces[0]),
        "onnx_profile_sha256": _sha256(onnx_profiles[0]),
        "profiled_benchmark_publishable": False,
        "schema_version": "junctionlens.runtime-profiler-evidence.v1",
        "status": "PASSED",
        "trace_configuration": {
            "cpuctxsw": "none",
            "sample": "none",
            "trace": ["cuda", "nvtx"],
        },
    }
    (output / "profile-evidence.json").write_text(
        json.dumps(report, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run profiler automation from the command line."""
    try:
        report = run(_parse_args())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"runtime profiler error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

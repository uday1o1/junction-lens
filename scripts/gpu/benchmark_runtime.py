#!/usr/bin/env python3
"""Run the native CUDA benchmark while auditing qualification validity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from junctionlens.model.profile import load_m0_profile
from junctionlens.runtime import analyze_native_benchmark, load_runtime_qualification
from scripts.gpu.qualify_runtime import write_synthetic_runtime_fixture


class GpuBenchmarkError(RuntimeError):
    """Raised when target benchmark execution or monitoring is incomplete."""


def _query(command: list[str], *, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)


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


def _float(value: str, label: str) -> float:
    try:
        result = float(value.strip())
    except ValueError as error:
        raise GpuBenchmarkError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise GpuBenchmarkError(f"{label} is not finite")
    return result


def _integer(value: str, label: str) -> int:
    number = _float(value, label)
    if number < 0.0 or not number.is_integer():
        raise GpuBenchmarkError(f"{label} is not a nonnegative integer")
    return int(number)


def _gpu_sample(nvidia_smi: str, gpu_uuid: str, benchmark_pid: int) -> dict[str, Any]:
    query = _query(
        [
            nvidia_smi,
            f"--id={gpu_uuid}",
            "--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used,"
            "clocks.current.graphics,clocks.current.memory,power.draw,power.limit,"
            "clocks_throttle_reasons.active,pstate",
            "--format=csv,noheader,nounits",
        ]
    )
    values = [value.strip() for value in query.stdout.strip().split(",")]
    if query.returncode != 0 or len(values) != 10 or values[0] != gpu_uuid:
        raise GpuBenchmarkError("nvidia-smi telemetry query failed")
    processes = _query(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if processes.returncode != 0:
        raise GpuBenchmarkError("nvidia-smi compute-process query failed")
    competitors: list[dict[str, Any]] = []
    for line in processes.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4 or fields[0] != gpu_uuid:
            continue
        pid = _integer(fields[1], "compute process PID")
        if pid != benchmark_pid:
            competitors.append(
                {
                    "pid": pid,
                    "process_name_sha256": hashlib.sha256(fields[2].encode()).hexdigest(),
                    "used_memory_mib": fields[3],
                }
            )
    return {
        "captured_monotonic_ns": time.monotonic_ns(),
        "clock_graphics_mhz": _float(values[4], "graphics clock"),
        "clock_memory_mhz": _float(values[5], "memory clock"),
        "competing_processes": competitors,
        "gpu_uuid": values[0],
        "memory_used_mib": _float(values[3], "GPU memory"),
        "power_draw_w": _float(values[6], "power draw"),
        "power_limit_w": _float(values[7], "power limit"),
        "pstate": values[9],
        "temperature_c": _integer(values[1], "GPU temperature"),
        "throttle_reason": values[8],
        "utilization_gpu_percent": _float(values[2], "GPU utilization"),
    }


def _ecc_errors(nvidia_smi: str, gpu_uuid: str) -> int | None:
    result = _query(
        [
            nvidia_smi,
            f"--id={gpu_uuid}",
            "--query-gpu=ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
        ]
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value or value in {"N/A", "[N/A]"}:
        return None
    return _integer(value, "volatile uncorrected ECC errors")


def _xid_count(started_epoch_seconds: float) -> int | None:
    journalctl = shutil.which("journalctl")
    if journalctl is None:
        return None
    result = _query(
        [journalctl, "--dmesg", "--since", f"@{int(started_epoch_seconds)}", "--no-pager"],
        timeout=30.0,
    )
    if result.returncode != 0:
        return None
    return sum("NVRM: Xid" in line or "Xid (PCI:" in line for line in result.stdout.splitlines())


def analyze_monitor(
    samples: list[dict[str, Any]],
    *,
    ecc_errors: int | None,
    xid_errors: int | None,
    validity: Any,
) -> dict[str, Any]:
    """Apply the frozen environment limits to raw GPU monitor samples."""
    failures: list[str] = []
    first_active = next(
        (
            index
            for index, sample in enumerate(samples)
            if float(sample.get("utilization_gpu_percent", 0.0)) > 0.0
        ),
        len(samples),
    )
    stabilized = samples[first_active:]
    if len(stabilized) < validity.minimum_monitor_samples:
        failures.append("MONITOR_SAMPLE_COUNT")
    temperatures = [int(sample["temperature_c"]) for sample in stabilized]
    clocks = [float(sample["clock_graphics_mhz"]) for sample in stabilized]
    power_fractions = [
        float(sample["power_draw_w"]) / float(sample["power_limit_w"]) for sample in stabilized
    ]
    competitor_count = max((len(sample["competing_processes"]) for sample in stabilized), default=0)
    throttle_reasons = sorted({str(sample["throttle_reason"]) for sample in stabilized})
    clock_cv = statistics.pstdev(clocks) / statistics.fmean(clocks) if clocks else 0.0
    if temperatures and max(temperatures) > validity.temperature_c_maximum:
        failures.append("GPU_TEMPERATURE_LIMIT")
    if clock_cv > validity.graphics_clock_coefficient_of_variation_maximum:
        failures.append("GPU_CLOCK_VARIATION")
    if power_fractions and max(power_fractions) > validity.power_limit_fraction_maximum:
        failures.append("GPU_POWER_LIMIT")
    if competitor_count > validity.competing_compute_processes_maximum:
        failures.append("GPU_COMPETING_PROCESS")
    if any(reason not in validity.allowed_active_throttle_reasons for reason in throttle_reasons):
        failures.append("GPU_THROTTLING_ACTIVE")
    if ecc_errors is not None and ecc_errors > validity.volatile_ecc_errors_maximum:
        failures.append("GPU_ECC_ERROR")
    if xid_errors is None:
        failures.append("GPU_XID_AUDIT_UNAVAILABLE")
    elif xid_errors > validity.xid_errors_maximum:
        failures.append("GPU_XID_ERROR")
    return {
        "clock_graphics_coefficient_of_variation": clock_cv,
        "competing_processes_maximum": competitor_count,
        "ecc_supported": ecc_errors is not None,
        "failure_reason_codes": failures,
        "maximum_power_limit_fraction": max(power_fractions, default=0.0),
        "maximum_temperature_c": max(temperatures, default=0),
        "sample_count": len(stabilized),
        "schema_version": "junctionlens.gpu-benchmark-monitor.v1",
        "status": "PASSED" if not failures else "BLOCKED_INFRASTRUCTURE",
        "throttle_reasons": throttle_reasons,
        "volatile_uncorrected_ecc_errors": ecc_errors,
        "xid_errors": xid_errors,
    }


def _runtime_command(
    arguments: argparse.Namespace, input_list: Path, raw_output: Path
) -> list[str]:
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
        str(raw_output.resolve()),
        "--git-commit",
        arguments.source_commit,
        "--configuration-sha256",
        profile.canonical_sha256(),
        "--runtime-build-sha256",
        _sha256(arguments.runtime),
        "--provider-profile",
        "cuda",
        "--provider-log-output",
        str((raw_output.parent / "provider.raw.log").resolve()),
        "--device-id",
        "0",
        "--warmup-frames",
        str(config.protocol.warmup_frames),
        "--measured-frames",
        str(config.protocol.measured_frames),
        "--stability-frames",
        str(config.protocol.stability_frames),
        "--memory-sample-period",
        str(config.protocol.memory_sample_period),
        "--input-profile",
        config.protocol.input_profile,
    ]


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Execute one exclusive target benchmark and persist redacted evidence."""
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "fixture"
    fixture.mkdir()
    input_list = write_synthetic_runtime_fixture(fixture)
    raw_output = output / "native-raw.json"
    stdout_path = output / "runtime.stdout.log"
    stderr_path = output / "runtime.stderr.log"
    command = _runtime_command(arguments, input_list, raw_output)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise GpuBenchmarkError("nvidia-smi is required")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = arguments.gpu_uuid
    started_epoch = time.time()
    samples: list[dict[str, Any]] = []
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=arguments.project_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        try:
            interval = load_runtime_qualification(
                arguments.config
            ).benchmark_validity.monitor_interval_seconds
            while process.poll() is None:
                samples.append(_gpu_sample(nvidia_smi, arguments.gpu_uuid, process.pid))
                time.sleep(interval)
            return_code = process.wait()
        except BaseException:
            process.terminate()
            process.wait(timeout=30)
            raise
    replacements = {
        str(arguments.project_root.resolve()): "<SOURCE_ROOT>",
        str(output): "<OUTPUT_ROOT>",
        str(Path.home()): "<REMOTE_HOME>",
    }
    stdout_path.write_text(
        _redact(stdout_path.read_text(encoding="utf-8", errors="replace"), replacements),
        encoding="utf-8",
    )
    stderr_path.write_text(
        _redact(stderr_path.read_text(encoding="utf-8", errors="replace"), replacements),
        encoding="utf-8",
    )
    config = load_runtime_qualification(arguments.config)
    monitor = analyze_monitor(
        samples,
        ecc_errors=_ecc_errors(nvidia_smi, arguments.gpu_uuid),
        xid_errors=_xid_count(started_epoch),
        validity=config.benchmark_validity,
    )
    (output / "monitor.json").write_text(
        json.dumps({**monitor, "samples": samples}, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if return_code != 0 or not raw_output.is_file():
        raise GpuBenchmarkError(f"native benchmark failed with exit code {return_code}")
    report = analyze_native_benchmark(
        raw_output,
        config,
        qualification_environment=True,
        monitor_report=monitor,
    )
    (output / "qualification.json").write_text(
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
    """Run the target benchmark CLI."""
    try:
        report = run(_parse_args())
    except (GpuBenchmarkError, OSError, RuntimeError, ValueError) as error:
        print(f"GPU benchmark error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if report["status"] == "PASSED":
        return 0
    return 3 if report["status"] == "BLOCKED_INFRASTRUCTURE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed analysis for native runtime benchmark and stability evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object_path,
    load_yaml_object_path,
)


class BenchmarkEvidenceError(RuntimeError):
    """Raised when runtime benchmark evidence is malformed or incomplete."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Protocol(_FrozenModel):
    warmup_frames: int = Field(ge=0, le=100_000)
    measured_frames: int = Field(ge=1, le=100_000)
    stability_frames: int = Field(ge=1, le=100_000)
    memory_sample_period: int = Field(ge=1, le=10_000)
    input_profile: Literal["predecoded", "full-file"]


class AbsoluteBudgets(_FrozenModel):
    throughput_per_second_minimum: float = Field(gt=0.0)
    p95_latency_ms_maximum: float = Field(gt=0.0)
    p99_latency_ms_maximum: float = Field(gt=0.0)
    peak_device_memory_bytes_maximum: int = Field(gt=0)
    unexpected_cpu_provider_nodes_maximum: int = Field(ge=0)


class StabilityLimits(_FrozenModel):
    host_growth_bytes_maximum: int = Field(ge=0)
    device_growth_bytes_maximum: int = Field(ge=0)
    host_slope_bytes_per_1000_frames_maximum: float = Field(ge=0.0)
    device_slope_bytes_per_1000_frames_maximum: float = Field(ge=0.0)


class BenchmarkValidity(_FrozenModel):
    monitor_interval_seconds: float = Field(gt=0.0, le=60.0)
    minimum_monitor_samples: int = Field(ge=2)
    temperature_c_maximum: int = Field(gt=0, le=120)
    graphics_clock_coefficient_of_variation_maximum: float = Field(ge=0.0, le=1.0)
    power_limit_fraction_maximum: float = Field(gt=0.0, le=2.0)
    competing_compute_processes_maximum: int = Field(ge=0)
    volatile_ecc_errors_maximum: int = Field(ge=0)
    xid_errors_maximum: int = Field(ge=0)
    allowed_active_throttle_reasons: tuple[str, ...]


class ProfilingProtocol(_FrozenModel):
    warmup_frames: int = Field(ge=1, le=10_000)
    measured_frames: int = Field(ge=1, le=10_000)
    maximum_duration_seconds: int = Field(ge=1, le=3600)


class RuntimeQualificationConfig(_FrozenModel):
    schema_version: Literal["junctionlens.runtime-qualification-config.v1"]
    protocol: Protocol
    absolute_budgets: AbsoluteBudgets
    stability: StabilityLimits
    benchmark_validity: BenchmarkValidity
    profiling: ProfilingProtocol


_PHASES = (
    "decode_ms",
    "preprocess_ms",
    "host_to_device_ms",
    "inference_ms",
    "device_to_host_ms",
    "postprocess_ms",
    "track_ms",
    "serialize_ms",
    "end_to_end_ms",
)
_RAW_KEYS = {
    "clock_source",
    "input_profile",
    "measured_frames",
    "memory_sample_period",
    "model_sha256",
    "onnx_profile_file",
    "profiler_run",
    "provider_assignment_sha256",
    "provider_node_counts",
    "provider_profile",
    "publishable",
    "samples",
    "schema_version",
    "stability_frames",
    "stability_frames_processed",
    "startup_ms",
    "status",
    "warmup_frames",
}
_MONITOR_KEYS = {
    "clock_graphics_coefficient_of_variation",
    "competing_processes_maximum",
    "ecc_supported",
    "failure_reason_codes",
    "maximum_power_limit_fraction",
    "maximum_temperature_c",
    "sample_count",
    "schema_version",
    "status",
    "throttle_reasons",
    "volatile_uncorrected_ecc_errors",
    "xid_errors",
}


def _strict_json(path: Path, *, maximum_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    try:
        return load_json_object_path(
            path,
            "benchmark evidence",
            ParseLimits(
                max_bytes=maximum_bytes,
                max_depth=32,
                max_nodes=2_000_000,
                max_container_items=1_000_000,
                max_string_bytes=4 * 1024 * 1024,
            ),
        )
    except ParseBoundaryError as error:
        raise BenchmarkEvidenceError(str(error)) from error


def load_runtime_qualification(path: Path) -> RuntimeQualificationConfig:
    """Load the exact runtime protocol with unknown-key rejection."""
    try:
        value = load_yaml_object_path(
            path,
            "runtime qualification config",
            ParseLimits(
                max_bytes=1024 * 1024,
                max_depth=16,
                max_nodes=10_000,
                max_container_items=1_000,
                max_string_bytes=64 * 1024,
            ),
        )
    except ParseBoundaryError as error:
        raise BenchmarkEvidenceError(str(error)) from error
    result = RuntimeQualificationConfig.model_validate(value)
    if result.protocol.memory_sample_period > result.protocol.stability_frames:
        raise BenchmarkEvidenceError("memory sample period exceeds the stability run")
    if (
        result.absolute_budgets.p99_latency_ms_maximum
        < result.absolute_budgets.p95_latency_ms_maximum
    ):
        raise BenchmarkEvidenceError("P99 latency budget is tighter than P95")
    reasons = result.benchmark_validity.allowed_active_throttle_reasons
    if (
        not reasons
        or len(set(reasons)) != len(reasons)
        or any(not reason.strip() for reason in reasons)
    ):
        raise BenchmarkEvidenceError("allowed throttle reasons are empty or duplicated")
    return result


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkEvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        raise BenchmarkEvidenceError(f"{label} is outside its finite range")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkEvidenceError(f"{label} must be an integer at least {minimum}")
    return value


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise BenchmarkEvidenceError("measured runtime population is empty")
    mean = math.fsum(values) / len(values)
    return {
        "count": len(values),
        "mean_ms": mean,
        "median_ms": _quantile(values, 0.5),
        "p90_ms": _quantile(values, 0.9),
        "p95_ms": _quantile(values, 0.95),
        "p99_ms": _quantile(values, 0.99),
        "maximum_ms": max(values),
    }


def _slope_per_1000(samples: Sequence[tuple[int, float]]) -> float:
    if len(samples) < 2:
        raise BenchmarkEvidenceError("stability evidence requires at least two memory samples")
    x_mean = math.fsum(float(item[0]) for item in samples) / len(samples)
    y_mean = math.fsum(item[1] for item in samples) / len(samples)
    denominator = math.fsum((float(x) - x_mean) ** 2 for x, _ in samples)
    if denominator == 0.0:
        raise BenchmarkEvidenceError("stability memory sample iterations are degenerate")
    numerator = math.fsum((float(x) - x_mean) * (y - y_mean) for x, y in samples)
    return max(0.0, numerator / denominator * 1000.0)


def _gate(
    gate_id: str,
    observed: float | int,
    limit: float | int,
    passed: bool,
    comparison: str,
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "id": gate_id,
        "limit": limit,
        "observed": observed,
        "status": "PASS" if passed else "FAIL",
    }


def _validate_identity(raw: Mapping[str, Any]) -> None:
    for key in ("model_sha256", "provider_assignment_sha256"):
        value = raw[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise BenchmarkEvidenceError(f"native benchmark {key} is malformed")
        try:
            int(value, 16)
        except ValueError as error:
            raise BenchmarkEvidenceError(f"native benchmark {key} is not hexadecimal") from error


def _validate_monitor(monitor: Mapping[str, Any], config: RuntimeQualificationConfig) -> None:
    if set(monitor) != _MONITOR_KEYS:
        raise BenchmarkEvidenceError("qualification monitor has unknown or missing keys")
    if (
        monitor["schema_version"] != "junctionlens.gpu-benchmark-monitor.v1"
        or monitor["status"] not in {"PASSED", "BLOCKED_INFRASTRUCTURE"}
        or not isinstance(monitor["failure_reason_codes"], list)
        or not all(isinstance(item, str) and item for item in monitor["failure_reason_codes"])
        or not isinstance(monitor["throttle_reasons"], list)
        or not all(isinstance(item, str) and item for item in monitor["throttle_reasons"])
        or not isinstance(monitor["ecc_supported"], bool)
    ):
        raise BenchmarkEvidenceError("qualification monitor identity is invalid")
    sample_count = _integer(monitor["sample_count"], "monitor sample count")
    _integer(monitor["competing_processes_maximum"], "monitor competing processes")
    _integer(monitor["maximum_temperature_c"], "monitor maximum temperature")
    _number(monitor["clock_graphics_coefficient_of_variation"], "monitor clock variation")
    _number(monitor["maximum_power_limit_fraction"], "monitor power fraction")
    if monitor["status"] == "PASSED":
        if sample_count < config.benchmark_validity.minimum_monitor_samples:
            raise BenchmarkEvidenceError("passing monitor sample population is incomplete")
        if monitor["failure_reason_codes"]:
            raise BenchmarkEvidenceError("passing monitor contains failure reason codes")
    elif not monitor["failure_reason_codes"]:
        raise BenchmarkEvidenceError("blocked monitor has no failure reason code")


def analyze_native_benchmark(
    raw_path: Path,
    config: RuntimeQualificationConfig,
    *,
    qualification_environment: bool,
    monitor_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and summarize a benchmark without inferring target acceptance locally."""
    raw = _strict_json(raw_path)
    if set(raw) != _RAW_KEYS:
        raise BenchmarkEvidenceError("native benchmark has unknown or missing root keys")
    if (
        raw["schema_version"] != "junctionlens.runtime-benchmark-raw.v1"
        or raw["status"] != "MEASURED_UNQUALIFIED"
        or raw["profiler_run"] is not False
        or raw["publishable"] is not True
        or raw["onnx_profile_file"] is not None
    ):
        raise BenchmarkEvidenceError("native benchmark is not publishable benchmark evidence")
    _validate_identity(raw)
    for key, minimum in (
        ("warmup_frames", 0),
        ("measured_frames", 1),
        ("stability_frames", 1),
        ("stability_frames_processed", 1),
        ("memory_sample_period", 1),
    ):
        _integer(raw[key], key, minimum=minimum)
    if (
        raw["input_profile"] not in {"predecoded", "full-file"}
        or raw["provider_profile"] not in {"cpu-reference", "cuda", "tensorrt"}
        or not isinstance(raw["clock_source"], str)
        or not raw["clock_source"]
    ):
        raise BenchmarkEvidenceError("native benchmark profile or clock identity is invalid")
    protocol = config.protocol
    observed_protocol = {
        "input_profile": raw["input_profile"],
        "measured_frames": raw["measured_frames"],
        "memory_sample_period": raw["memory_sample_period"],
        "stability_frames": raw["stability_frames"],
        "stability_frames_processed": raw["stability_frames_processed"],
        "warmup_frames": raw["warmup_frames"],
    }
    expected_protocol = {
        "input_profile": protocol.input_profile,
        "measured_frames": protocol.measured_frames,
        "memory_sample_period": protocol.memory_sample_period,
        "stability_frames": protocol.stability_frames,
        "stability_frames_processed": protocol.stability_frames,
        "warmup_frames": protocol.warmup_frames,
    }
    protocol_complete = observed_protocol == expected_protocol
    samples = raw["samples"]
    if not isinstance(samples, list) or not samples:
        raise BenchmarkEvidenceError("native benchmark samples are absent")
    populations: dict[str, dict[str, list[float]]] = {
        phase: {"warmup": [], "measured": []} for phase in _PHASES
    }
    identities: set[tuple[str, int]] = set()
    stability_host: list[tuple[int, float]] = []
    stability_device: list[tuple[int, float]] = []
    peak_host = 0
    peak_device = 0
    observed_iterations: dict[str, set[int]] = {
        "warmup": set(),
        "measured": set(),
        "stability": set(),
    }
    for item in samples:
        if not isinstance(item, dict) or set(item) != {
            "current_device_bytes",
            "iteration",
            "peak_device_bytes",
            "peak_resident_host_bytes",
            "phases",
            "sample_kind",
        }:
            raise BenchmarkEvidenceError("native benchmark sample schema is invalid")
        kind = item["sample_kind"]
        iteration = _integer(item["iteration"], "sample iteration")
        if kind not in {"warmup", "measured", "stability"}:
            raise BenchmarkEvidenceError("native benchmark sample kind is invalid")
        identity = (cast(str, kind), iteration)
        if identity in identities:
            raise BenchmarkEvidenceError("native benchmark sample identity is duplicated")
        identities.add(identity)
        observed_iterations[cast(str, kind)].add(iteration)
        host = _integer(item["peak_resident_host_bytes"], "peak host memory")
        current_device = _integer(item["current_device_bytes"], "current device memory")
        device = _integer(item["peak_device_bytes"], "peak device memory")
        if host <= 0 or current_device > device:
            raise BenchmarkEvidenceError("native benchmark memory sample is inconsistent")
        peak_host = max(peak_host, host)
        peak_device = max(peak_device, device)
        phases = item["phases"]
        if not isinstance(phases, dict) or set(phases) != set(_PHASES):
            raise BenchmarkEvidenceError("native benchmark phase schema is invalid")
        for phase in _PHASES:
            duration = _number(
                phases[phase],
                phase,
                positive=phase in {"inference_ms", "serialize_ms", "end_to_end_ms"},
            )
            if kind in {"warmup", "measured"}:
                populations[phase][cast(str, kind)].append(duration)
        phase_sum = math.fsum(float(phases[phase]) for phase in _PHASES if phase != "end_to_end_ms")
        if float(phases["end_to_end_ms"]) + 1.0 < phase_sum:
            raise BenchmarkEvidenceError("end-to-end timing is shorter than its sequential phases")
        if kind == "stability":
            stability_host.append((iteration, float(host)))
            stability_device.append((iteration, float(current_device)))
    if len(populations["end_to_end_ms"]["warmup"]) != raw["warmup_frames"]:
        raise BenchmarkEvidenceError("warmup sample count differs from its declaration")
    if len(populations["end_to_end_ms"]["measured"]) != raw["measured_frames"]:
        raise BenchmarkEvidenceError("measured sample count differs from its declaration")
    expected_stability_iterations = {0, int(raw["stability_frames"]) - 1}
    expected_stability_iterations.update(
        range(
            int(raw["memory_sample_period"]) - 1,
            int(raw["stability_frames"]),
            int(raw["memory_sample_period"]),
        )
    )
    if (
        observed_iterations["warmup"] != set(range(int(raw["warmup_frames"])))
        or observed_iterations["measured"] != set(range(int(raw["measured_frames"])))
        or observed_iterations["stability"] != expected_stability_iterations
    ):
        raise BenchmarkEvidenceError("native benchmark sample schedule is incomplete")
    distributions = {
        phase: {
            "measured": _distribution(populations[phase]["measured"]),
            "warmup": (
                None
                if not populations[phase]["warmup"]
                else _distribution(populations[phase]["warmup"])
            ),
        }
        for phase in _PHASES
    }
    e2e = cast(dict[str, float | int], distributions["end_to_end_ms"]["measured"])
    throughput = 1000.0 / float(e2e["mean_ms"])
    if len(stability_host) < 2 or len(stability_device) < 2:
        raise BenchmarkEvidenceError("native benchmark stability samples are incomplete")
    host_growth = max(0.0, stability_host[-1][1] - stability_host[0][1])
    device_growth = max(0.0, stability_device[-1][1] - stability_device[0][1])
    host_slope = _slope_per_1000(stability_host)
    device_slope = _slope_per_1000(stability_device)
    node_counts = raw["provider_node_counts"]
    if not isinstance(node_counts, dict) or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in node_counts.items()
    ):
        raise BenchmarkEvidenceError("provider node counts are malformed")
    if set(node_counts) - {
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
    }:
        raise BenchmarkEvidenceError("provider node counts contain an undeclared provider")
    if (
        raw["provider_profile"] == "cuda" and int(node_counts.get("CUDAExecutionProvider", 0)) <= 0
    ) or (
        raw["provider_profile"] == "tensorrt"
        and int(node_counts.get("TensorrtExecutionProvider", 0)) <= 0
    ):
        raise BenchmarkEvidenceError("accelerated provider node assignment is absent")
    cpu_nodes = int(node_counts.get("CPUExecutionProvider", 0))
    budgets = config.absolute_budgets
    limits = config.stability
    gates = [
        _gate(
            "throughput_per_second",
            throughput,
            budgets.throughput_per_second_minimum,
            throughput >= budgets.throughput_per_second_minimum,
            ">=",
        ),
        _gate(
            "p95_end_to_end_ms",
            float(e2e["p95_ms"]),
            budgets.p95_latency_ms_maximum,
            float(e2e["p95_ms"]) <= budgets.p95_latency_ms_maximum,
            "<=",
        ),
        _gate(
            "p99_end_to_end_ms",
            float(e2e["p99_ms"]),
            budgets.p99_latency_ms_maximum,
            float(e2e["p99_ms"]) <= budgets.p99_latency_ms_maximum,
            "<=",
        ),
        _gate(
            "peak_device_memory_bytes",
            peak_device,
            budgets.peak_device_memory_bytes_maximum,
            peak_device <= budgets.peak_device_memory_bytes_maximum,
            "<=",
        ),
        _gate(
            "unexpected_cpu_provider_nodes",
            cpu_nodes,
            budgets.unexpected_cpu_provider_nodes_maximum,
            cpu_nodes <= budgets.unexpected_cpu_provider_nodes_maximum,
            "<=",
        ),
        _gate(
            "host_memory_growth_bytes",
            host_growth,
            limits.host_growth_bytes_maximum,
            host_growth <= limits.host_growth_bytes_maximum,
            "<=",
        ),
        _gate(
            "device_memory_growth_bytes",
            device_growth,
            limits.device_growth_bytes_maximum,
            device_growth <= limits.device_growth_bytes_maximum,
            "<=",
        ),
        _gate(
            "host_memory_slope_bytes_per_1000_frames",
            host_slope,
            limits.host_slope_bytes_per_1000_frames_maximum,
            host_slope <= limits.host_slope_bytes_per_1000_frames_maximum,
            "<=",
        ),
        _gate(
            "device_memory_slope_bytes_per_1000_frames",
            device_slope,
            limits.device_slope_bytes_per_1000_frames_maximum,
            device_slope <= limits.device_slope_bytes_per_1000_frames_maximum,
            "<=",
        ),
    ]
    monitor_status = None if monitor_report is None else monitor_report.get("status")
    if qualification_environment:
        if raw["provider_profile"] not in {"cuda", "tensorrt"}:
            raise BenchmarkEvidenceError("qualification evidence requires an accelerated profile")
        if monitor_report is None:
            raise BenchmarkEvidenceError("qualification monitor evidence is incomplete")
        _validate_monitor(monitor_report, config)
        if not protocol_complete:
            status, reason_code = "FAILED", "RUNTIME_PROTOCOL_MISMATCH"
        elif monitor_status != "PASSED":
            status, reason_code = "BLOCKED_INFRASTRUCTURE", "BENCHMARK_ENVIRONMENT_INVALID"
        elif all(gate["status"] == "PASS" for gate in gates):
            status, reason_code = "PASSED", "RUNTIME_ABSOLUTE_BUDGETS_MET"
        else:
            status, reason_code = "FAILED", "RUNTIME_ABSOLUTE_BUDGET_FAILED"
    else:
        status, reason_code = "LOCAL_DIAGNOSTIC", "QUALIFICATION_HARDWARE_NOT_EXECUTED"
    return {
        "clock_source": raw["clock_source"],
        "distributions": distributions,
        "gates": gates,
        "input_profile": raw["input_profile"],
        "memory": {
            "device_growth_bytes": device_growth,
            "device_slope_bytes_per_1000_frames": device_slope,
            "host_growth_bytes": host_growth,
            "host_slope_bytes_per_1000_frames": host_slope,
            "peak_device_bytes": peak_device,
            "peak_resident_host_bytes": peak_host,
        },
        "model_sha256": raw["model_sha256"],
        "monitor_status": monitor_status,
        "protocol_complete": protocol_complete,
        "provider_assignment_sha256": raw["provider_assignment_sha256"],
        "provider_node_counts": dict(sorted(node_counts.items())),
        "reason_code": reason_code,
        "schema_version": "junctionlens.runtime-qualification.v1",
        "startup_ms": _number(raw["startup_ms"], "startup_ms"),
        "status": status,
        "throughput_per_second": throughput,
    }


__all__ = [
    "BenchmarkEvidenceError",
    "RuntimeQualificationConfig",
    "analyze_native_benchmark",
    "load_runtime_qualification",
]

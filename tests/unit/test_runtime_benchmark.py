from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from scripts.gpu.benchmark_runtime import analyze_monitor

from junctionlens.runtime.benchmark import (
    BenchmarkEvidenceError,
    analyze_native_benchmark,
    load_runtime_qualification,
)


def _raw_report(*, provider: str = "cuda", host_growth: int = 0) -> dict[str, object]:
    phases = {
        "decode_ms": 0.0,
        "preprocess_ms": 0.0,
        "host_to_device_ms": 1.0 if provider != "cpu-reference" else 0.0,
        "inference_ms": 40.0,
        "device_to_host_ms": 1.0 if provider != "cpu-reference" else 0.0,
        "postprocess_ms": 2.0,
        "track_ms": 0.01,
        "serialize_ms": 1.0,
        "end_to_end_ms": 45.0,
    }
    samples: list[dict[str, object]] = []
    for kind, count in (("warmup", 1), ("measured", 3)):
        for iteration in range(count):
            samples.append(
                {
                    "current_device_bytes": 1024,
                    "iteration": iteration,
                    "peak_device_bytes": 2048,
                    "peak_resident_host_bytes": 4096,
                    "phases": phases,
                    "sample_kind": kind,
                }
            )
    for iteration, host in ((0, 4096), (1, 4096 + host_growth)):
        samples.append(
            {
                "current_device_bytes": 1024,
                "iteration": iteration,
                "peak_device_bytes": 2048,
                "peak_resident_host_bytes": host,
                "phases": phases,
                "sample_kind": "stability",
            }
        )
    return {
        "clock_source": "CLOCK_MONOTONIC_RAW",
        "input_profile": "predecoded",
        "measured_frames": 3,
        "memory_sample_period": 1,
        "model_sha256": "1" * 64,
        "onnx_profile_file": None,
        "profiler_run": False,
        "provider_assignment_sha256": "2" * 64,
        "provider_node_counts": {
            "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider": 10
        },
        "provider_profile": provider,
        "publishable": True,
        "samples": samples,
        "schema_version": "junctionlens.runtime-benchmark-raw.v1",
        "stability_frames": 2,
        "stability_frames_processed": 2,
        "startup_ms": 10.0,
        "status": "MEASURED_UNQUALIFIED",
        "warmup_frames": 1,
    }


def _small_config(path: Path) -> object:
    config = load_runtime_qualification(path)
    return config.model_copy(
        update={
            "protocol": config.protocol.model_copy(
                update={
                    "warmup_frames": 1,
                    "measured_frames": 3,
                    "stability_frames": 2,
                    "memory_sample_period": 1,
                }
            )
        }
    )


def _passing_monitor() -> dict[str, object]:
    return {
        "clock_graphics_coefficient_of_variation": 0.0,
        "competing_processes_maximum": 0,
        "ecc_supported": False,
        "failure_reason_codes": [],
        "maximum_power_limit_fraction": 0.5,
        "maximum_temperature_c": 70,
        "sample_count": 10,
        "schema_version": "junctionlens.gpu-benchmark-monitor.v1",
        "status": "PASSED",
        "throttle_reasons": ["Not Active"],
        "volatile_uncorrected_ecc_errors": None,
        "xid_errors": 0,
    }


def test_target_runtime_budgets_pass_only_with_complete_monitor(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_raw_report()), encoding="utf-8")
    config = _small_config(Path("configs/runtime/qualification-v1.yaml"))
    report = analyze_native_benchmark(
        raw,
        config,  # type: ignore[arg-type]
        qualification_environment=True,
        monitor_report=_passing_monitor(),
    )
    assert report["status"] == "PASSED"
    assert report["reason_code"] == "RUNTIME_ABSOLUTE_BUDGETS_MET"
    assert all(gate["status"] == "PASS" for gate in report["gates"])


def test_runtime_absolute_budgets_match_acceptance_charter() -> None:
    config = load_runtime_qualification(Path("configs/runtime/qualification-v1.yaml"))
    charter = yaml.safe_load(
        Path("configs/gates/acceptance-v1.draft.yaml").read_text(encoding="utf-8")
    )["absolute_runtime"]
    assert (
        config.absolute_budgets.throughput_per_second_minimum
        == charter["throughput_per_second_minimum"]
    )
    assert config.absolute_budgets.p95_latency_ms_maximum == charter["p95_latency_ms_maximum"]
    assert config.absolute_budgets.p99_latency_ms_maximum == charter["p99_latency_ms_maximum"]
    assert (
        config.absolute_budgets.peak_device_memory_bytes_maximum
        == charter["peak_device_memory_bytes_maximum"]
    )
    assert config.protocol.stability_frames == charter["long_run_frames"]
    assert (
        config.absolute_budgets.unexpected_cpu_provider_nodes_maximum
        == charter["unexpected_cpu_provider_nodes_maximum"]
    )


def test_seeded_memory_leak_fails_while_nearby_control_passes(tmp_path: Path) -> None:
    config = _small_config(Path("configs/runtime/qualification-v1.yaml"))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_raw_report()), encoding="utf-8")
    assert (
        analyze_native_benchmark(
            control,
            config,  # type: ignore[arg-type]
            qualification_environment=True,
            monitor_report=_passing_monitor(),
        )["status"]
        == "PASSED"
    )
    fault = tmp_path / "fault.json"
    fault.write_text(json.dumps(_raw_report(host_growth=128 * 1024 * 1024)), encoding="utf-8")
    report = analyze_native_benchmark(
        fault,
        config,  # type: ignore[arg-type]
        qualification_environment=True,
        monitor_report=_passing_monitor(),
    )
    assert report["status"] == "FAILED"
    assert report["reason_code"] == "RUNTIME_ABSOLUTE_BUDGET_FAILED"
    assert any(
        gate["id"] == "host_memory_growth_bytes" and gate["status"] == "FAIL"
        for gate in report["gates"]
    )


def test_contaminated_monitor_blocks_instead_of_failing_model(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_raw_report()), encoding="utf-8")
    monitor = _passing_monitor()
    monitor.update(
        {
            "failure_reason_codes": ["GPU_COMPETING_PROCESS"],
            "sample_count": 0,
            "status": "BLOCKED_INFRASTRUCTURE",
        }
    )
    report = analyze_native_benchmark(
        raw,
        _small_config(Path("configs/runtime/qualification-v1.yaml")),  # type: ignore[arg-type]
        qualification_environment=True,
        monitor_report=monitor,
    )
    assert report["status"] == "BLOCKED_INFRASTRUCTURE"
    assert report["reason_code"] == "BENCHMARK_ENVIRONMENT_INVALID"


def test_profiler_measurements_are_rejected_as_benchmark_evidence(tmp_path: Path) -> None:
    payload = _raw_report()
    payload["profiler_run"] = True
    payload["publishable"] = False
    payload["onnx_profile_file"] = "onnx-profile.json"
    raw = tmp_path / "profile.json"
    raw.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkEvidenceError, match="not publishable"):
        analyze_native_benchmark(
            raw,
            _small_config(Path("configs/runtime/qualification-v1.yaml")),  # type: ignore[arg-type]
            qualification_environment=False,
        )


def test_monitor_detects_contention_and_nearby_control() -> None:
    validity = load_runtime_qualification(
        Path("configs/runtime/qualification-v1.yaml")
    ).benchmark_validity
    sample = {
        "clock_graphics_mhz": 1500.0,
        "competing_processes": [],
        "power_draw_w": 200.0,
        "power_limit_w": 300.0,
        "temperature_c": 70,
        "throttle_reason": "Not Active",
        "utilization_gpu_percent": 90.0,
    }
    control = analyze_monitor([sample] * 10, ecc_errors=None, xid_errors=0, validity=validity)
    assert control["status"] == "PASSED"
    contaminated = dict(sample)
    contaminated["competing_processes"] = [{"pid": 7}]
    fault = analyze_monitor([contaminated] * 10, ecc_errors=None, xid_errors=0, validity=validity)
    assert fault["status"] == "BLOCKED_INFRASTRUCTURE"
    assert "GPU_COMPETING_PROCESS" in fault["failure_reason_codes"]

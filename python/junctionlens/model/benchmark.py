"""M0 portability measurements and frozen-budget evidence."""

from __future__ import annotations

import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from junctionlens.model.budget import BudgetPlan
from junctionlens.model.profile import M0ModelProfile
from junctionlens.model.spike import INPUT_NAMES, OUTPUT_NAMES
from junctionlens.model.synthetic import make_micro_inputs
from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_json_object_path


class BenchmarkError(RuntimeError):
    """Raised when prerequisite evidence is absent or internally inconsistent."""


def _read_report(path: Path, required_status: str) -> dict[str, Any]:
    try:
        value = load_json_object_path(
            path,
            "M0 prerequisite report",
            ParseLimits(max_bytes=16 * 1024 * 1024, max_depth=24, max_nodes=500_000),
        )
    except (OSError, ParseBoundaryError, TypeError) as error:
        raise BenchmarkError(f"cannot read prerequisite report {path}") from error
    if value.get("status") != required_status:
        raise BenchmarkError(f"prerequisite report {path} is not {required_status}")
    return value


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _latency_summary(samples_ms: list[float], warmup: int) -> dict[str, float | int | str]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "clock": "time.perf_counter_ns",
        "count": len(samples_ms),
        "warmup_count": warmup,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "maximum_ms": float(values.max()),
    }


def run_m0_benchmark(
    profile: M0ModelProfile,
    budget: BudgetPlan,
    model_path: Path,
    micro_overfit_report_path: Path,
    evaluator_report_path: Path,
    artifact_root: Path,
    output_path: Path,
    *,
    warmup: int = 10,
    measured: int = 50,
) -> dict[str, Any]:
    """Measure only the local CPU path and keep target estimates explicitly deferred."""
    if warmup < 1 or measured < 10:
        raise BenchmarkError("benchmark requires at least one warmup and ten measured iterations")
    micro = _read_report(micro_overfit_report_path, "PASSED")
    evaluator = _read_report(evaluator_report_path, "PASSED")
    evaluator_benchmark = evaluator.get("benchmark")
    if not isinstance(evaluator_benchmark, dict):
        raise BenchmarkError("official evaluator report does not contain throughput evidence")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = make_micro_inputs(
        profile,
        torch.tensor([7], dtype=torch.int64),
        spatial_size=(profile.input.height, profile.input.width),
    )
    feed = {name: value.numpy() for name, value in zip(INPUT_NAMES, inputs, strict=True)}
    for _ in range(warmup):
        session.run(list(OUTPUT_NAMES), feed)
    samples_ms: list[float] = []
    output_values: list[np.ndarray[Any, Any]] = []
    for _ in range(measured):
        start = time.perf_counter_ns()
        output_values = session.run(list(OUTPUT_NAMES), feed)
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    input_bytes = sum(value.nbytes for value in feed.values())
    output_bytes = sum(value.nbytes for value in output_values)
    budget_summary = budget.summary()
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PASSED_LOCAL",
        "profile_id": profile.profile_id,
        "profile_sha256": profile.canonical_sha256(),
        "host_profile": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "claim_scope": "CPU_PORTABILITY_ONLY",
        },
        "cpu_prepacked_raw_inference": _latency_summary(samples_ms, warmup),
        "cpu_peak_resident_host_memory_bytes": _maximum_rss_bytes(),
        "input_tensor_bytes": input_bytes,
        "output_tensor_bytes": output_bytes,
        "onnx_model_bytes": model_path.stat().st_size,
        "micro_overfit": {
            "training_sequences_per_second": micro["training_sequences_per_second"],
            "peak_training_memory_bytes": micro["peak_training_memory_bytes"],
            "checkpoint_bytes": micro["checkpoint_bytes"],
            "steps": micro["steps_completed"],
        },
        "official_evaluator": evaluator_benchmark,
        "generated_m0_artifact_bytes": _directory_bytes(artifact_root),
        "expected_v1_artifact_growth_gib": budget_summary["planned_generated_artifacts_gib"],
        "budget": budget_summary,
        "accelerated_feasibility_gate": {
            "state": "DEFERRED_HARDWARE",
            "reason_code": "NVIDIA_TARGET_UNAVAILABLE",
            "qualification_profile": "ubuntu-24.04-x86_64-cuda-12.8",
            "final_reference_encoder": profile.model.final_reference_encoder,
            "final_reference_hidden_dimension": profile.model.final_reference_hidden_dimension,
            "required_projected_p95_ms_with_contingency": 80.0,
            "required_projected_peak_device_memory_gib_with_contingency": 4.8,
            "absolute_p95_budget_ms": 100.0,
            "absolute_peak_device_memory_gib": 6.0,
            "local_cpu_numbers_used_as_gpu_claim": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report

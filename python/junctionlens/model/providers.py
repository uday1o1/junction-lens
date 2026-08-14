"""Fail-closed execution-provider probing and node-assignment evidence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import onnxruntime as ort
import torch

from junctionlens.model.profile import M0ModelProfile
from junctionlens.model.spike import INPUT_NAMES, OUTPUT_NAMES
from junctionlens.model.synthetic import make_micro_inputs
from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_json_path


class ProviderProbeError(RuntimeError):
    """Raised when an available provider cannot meet its declared local gate."""


def _profile_session(
    model_path: Path,
    profile: M0ModelProfile,
    providers: list[str],
    trace_prefix: Path,
) -> tuple[dict[str, int], list[str], str]:
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(trace_prefix)
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    inputs = make_micro_inputs(
        profile,
        torch.tensor([7], dtype=torch.int64),
        spatial_size=(profile.input.height, profile.input.width),
    )
    session.run(
        list(OUTPUT_NAMES),
        {name: value.numpy() for name, value in zip(INPUT_NAMES, inputs, strict=True)},
    )
    trace_path = Path(session.end_profiling())
    try:
        raw_events = load_json_path(
            trace_path,
            "ONNX Runtime provider trace",
            ParseLimits(
                max_bytes=64 * 1024 * 1024,
                max_depth=32,
                max_nodes=2_000_000,
                max_container_items=1_000_000,
            ),
        )
    except ParseBoundaryError as error:
        raise ProviderProbeError(str(error)) from error
    if not isinstance(raw_events, list) or not all(isinstance(item, dict) for item in raw_events):
        raise ProviderProbeError("ONNX Runtime provider trace must be an object array")
    events = cast(list[dict[str, Any]], raw_events)
    provider_counts: Counter[str] = Counter()
    cpu_nodes: list[str] = []
    for event in events:
        arguments = event.get("args")
        if not isinstance(arguments, dict):
            continue
        provider = arguments.get("provider")
        if not isinstance(provider, str) or not provider:
            continue
        provider_counts[provider] += 1
        if provider == "CPUExecutionProvider":
            cpu_nodes.append(str(event.get("name", "unknown")))
    if not provider_counts:
        raise ProviderProbeError("ONNX Runtime trace contained no provider-assigned node events")
    return dict(sorted(provider_counts.items())), sorted(set(cpu_nodes)), str(trace_path)


def run_provider_probe(
    model_path: Path,
    profile: M0ModelProfile,
    output_path: Path,
) -> dict[str, Any]:
    """Trace CPU and every locally available accelerated provider."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    available = ort.get_available_providers()
    cpu_counts, cpu_nodes, cpu_trace = _profile_session(
        model_path,
        profile,
        ["CPUExecutionProvider"],
        output_path.parent / "ort-cpu",
    )
    if set(cpu_counts) != {"CPUExecutionProvider"} or not cpu_nodes:
        raise ProviderProbeError("the explicit CPU session did not assign every node to CPU")
    profiles: dict[str, dict[str, Any]] = {
        "cpu": {
            "state": "ACCEPTED",
            "reason_code": "CPU_ALL_NODES_ASSIGNED",
            "requested_providers": ["CPUExecutionProvider"],
            "provider_node_events": cpu_counts,
            "trace_path": cpu_trace,
        }
    }

    if "CUDAExecutionProvider" not in available:
        profiles["cuda"] = {
            "state": "DEFERRED_HARDWARE",
            "reason_code": "CUDA_PROVIDER_UNAVAILABLE",
            "requested_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "unexpected_cpu_nodes": None,
        }
    else:
        cuda_counts, cuda_cpu_nodes, cuda_trace = _profile_session(
            model_path,
            profile,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            output_path.parent / "ort-cuda",
        )
        profiles["cuda"] = {
            "state": "ACCEPTED" if not cuda_cpu_nodes else "FAILED",
            "reason_code": (
                "CUDA_NO_CPU_FALLBACK" if not cuda_cpu_nodes else "CUDA_UNEXPECTED_CPU_NODES"
            ),
            "requested_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "provider_node_events": cuda_counts,
            "unexpected_cpu_nodes": cuda_cpu_nodes,
            "trace_path": cuda_trace,
        }

    if "TensorrtExecutionProvider" not in available:
        profiles["tensorrt"] = {
            "state": "DEFERRED_HARDWARE",
            "reason_code": "TENSORRT_PROVIDER_UNAVAILABLE",
            "requested_providers": [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "partition_complete_assumed": False,
        }
    elif "CUDAExecutionProvider" not in available:
        profiles["tensorrt"] = {
            "state": "FAILED",
            "reason_code": "TENSORRT_WITHOUT_CUDA_PROVIDER",
            "partition_complete_assumed": False,
        }
    else:
        trt_counts, trt_cpu_nodes, trt_trace = _profile_session(
            model_path,
            profile,
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            output_path.parent / "ort-tensorrt",
        )
        profiles["tensorrt"] = {
            "state": "ACCEPTED",
            "reason_code": "TENSORRT_PARTITION_RECORDED",
            "requested_providers": [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "provider_node_events": trt_counts,
            "cpu_nodes": trt_cpu_nodes,
            "partition_complete_assumed": False,
            "trace_path": trt_trace,
        }

    failed = [name for name, value in profiles.items() if value["state"] == "FAILED"]
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "FAILED" if failed else "PASSED",
        "profile_id": profile.profile_id,
        "profile_sha256": profile.canonical_sha256(),
        "onnxruntime_version": ort.__version__,
        "available_providers": available,
        "profiles": profiles,
    }
    output_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if failed:
        raise ProviderProbeError(f"available provider profiles failed: {failed!r}")
    return report

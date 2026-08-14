"""Doctor report orchestration and readiness calculation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from junctionlens.doctor.models import (
    CapabilityEvidence,
    CapabilityRequirement,
    CapabilityState,
    DoctorReport,
    Readiness,
)
from junctionlens.doctor.probes import (
    host_evidence,
    probe_command,
    probe_container_runtime,
    probe_cpp_truth_file,
    probe_cuda,
    probe_dataset,
    probe_nvidia,
    probe_onnxruntime,
    probe_python,
    probe_tensorrt,
)


def _reference_probe() -> CapabilityEvidence:
    host = host_evidence()
    compatible = host.system == "Linux" and host.machine in {"x86_64", "AMD64"}
    return CapabilityEvidence(
        capability="linux_x86_64_reference",
        state=CapabilityState.AVAILABLE if compatible else CapabilityState.NOT_APPLICABLE,
        requirement=CapabilityRequirement.REFERENCE_ONLY,
        reason_code="REFERENCE_PLATFORM_AVAILABLE"
        if compatible
        else "REFERENCE_PLATFORM_NOT_EXECUTED",
        summary="running on the Linux x86-64 reference platform"
        if compatible
        else "this host is a portability profile, not the Linux x86-64 reference",
    )


def run_doctor(project_root: Path | None = None) -> DoctorReport:
    """Run every real probe and return a versioned immutable report."""
    root = (project_root or Path.cwd()).resolve()
    host = host_evidence()
    profile = (
        "linux-x86_64-reference"
        if host.system == "Linux" and host.machine in {"x86_64", "AMD64"}
        else "macos-arm64-portability"
        if host.system == "Darwin" and host.machine == "arm64"
        else "unsupported-portability"
    )
    capabilities = [
        probe_python(),
        probe_command(
            "uv",
            "uv",
            ["--version"],
            CapabilityRequirement.REQUIRED_LOCAL,
            "0.11.23",
        ),
        probe_command(
            "cmake",
            "cmake",
            ["--version"],
            CapabilityRequirement.REQUIRED_LOCAL,
            "3.31.8",
        ),
        probe_command(
            "ninja",
            "ninja",
            ["--version"],
            CapabilityRequirement.REQUIRED_LOCAL,
            "1.12.1",
        ),
        probe_command(
            "protoc",
            "protoc",
            ["--version"],
            CapabilityRequirement.REQUIRED_LOCAL,
            "31.1",
        ),
        probe_command(
            "node",
            "node",
            ["--version"],
            CapabilityRequirement.REQUIRED_LOCAL,
            "22.18.0",
        ),
        probe_command(
            "pnpm",
            "pnpm",
            ["--version"],
            CapabilityRequirement.REQUIRED_LOCAL,
            "10.17.1",
        ),
        probe_onnxruntime(),
        probe_cpp_truth_file(root),
        probe_container_runtime(),
        probe_nvidia(),
        probe_cuda(),
        probe_tensorrt(),
        probe_dataset(),
        _reference_probe(),
    ]
    required_local = [
        capability
        for capability in capabilities
        if capability.requirement is CapabilityRequirement.REQUIRED_LOCAL
    ]
    local_cpu = all(capability.state is CapabilityState.AVAILABLE for capability in required_local)
    reference = next(item for item in capabilities if item.capability == "linux_x86_64_reference")
    gpu = next(item for item in capabilities if item.capability == "nvidia_gpu")
    cuda = next(item for item in capabilities if item.capability == "cuda_toolkit")
    dataset = next(item for item in capabilities if item.capability == "openlane_v2_dataset")
    return DoctorReport(
        generated_at=datetime.now(UTC),
        profile=profile,
        host=host,
        readiness=Readiness(
            local_cpu=local_cpu,
            linux_x86_64_reference=reference.state is CapabilityState.AVAILABLE and local_cpu,
            accelerated_target=gpu.state is CapabilityState.AVAILABLE
            and cuda.state is CapabilityState.AVAILABLE,
            licensed_data=dataset.state is CapabilityState.AVAILABLE,
        ),
        capabilities=capabilities,
    )

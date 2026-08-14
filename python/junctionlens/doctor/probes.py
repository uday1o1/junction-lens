"""Real, bounded host capability probes."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from junctionlens.doctor.models import (
    CapabilityEvidence,
    CapabilityRequirement,
    CapabilityState,
    HostEvidence,
)
from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_json_object

_OUTPUT_LIMIT = 4096
_VERSION_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+){1,3})(?!\d)")


def host_evidence() -> HostEvidence:
    """Return actual host and interpreter identity."""
    return HostEvidence(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
    )


def _bounded(value: str) -> str:
    return value.strip()[:_OUTPUT_LIMIT]


def _run(command: Sequence[str], timeout_seconds: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def probe_command(
    capability: str,
    executable: str,
    version_args: Sequence[str],
    requirement: CapabilityRequirement,
    expected_version: str | None = None,
) -> CapabilityEvidence:
    """Probe one executable and optionally compare its observed version."""
    resolved = shutil.which(executable)
    command = [executable, *version_args]
    if resolved is None:
        return CapabilityEvidence(
            capability=capability,
            state=CapabilityState.ABSENT,
            requirement=requirement,
            reason_code=f"{capability.upper()}_NOT_FOUND",
            summary=f"{executable} is not available on PATH",
            expected_version=expected_version,
            command=command,
        )
    try:
        result = _run([resolved, *version_args])
    except subprocess.TimeoutExpired:
        return CapabilityEvidence(
            capability=capability,
            state=CapabilityState.ERROR,
            requirement=requirement,
            reason_code=f"{capability.upper()}_PROBE_TIMEOUT",
            summary=f"{executable} did not respond within the probe timeout",
            expected_version=expected_version,
            command=command,
            details={"resolved_path": resolved},
        )
    combined = _bounded(f"{result.stdout}\n{result.stderr}")
    if result.returncode != 0:
        return CapabilityEvidence(
            capability=capability,
            state=CapabilityState.ERROR,
            requirement=requirement,
            reason_code=f"{capability.upper()}_PROBE_FAILED",
            summary=f"{executable} returned exit code {result.returncode}",
            expected_version=expected_version,
            command=command,
            details={"resolved_path": resolved, "output": combined},
        )
    match = _VERSION_PATTERN.search(combined)
    observed = match.group(1) if match else None
    compatible = expected_version is None or observed == expected_version
    return CapabilityEvidence(
        capability=capability,
        state=CapabilityState.AVAILABLE if compatible else CapabilityState.INCOMPATIBLE,
        requirement=requirement,
        reason_code=f"{capability.upper()}_AVAILABLE"
        if compatible
        else f"{capability.upper()}_VERSION_MISMATCH",
        summary=f"{executable} reported {observed or 'an unparsed version'}",
        observed_version=observed,
        expected_version=expected_version,
        command=command,
        details={"resolved_path": resolved, "output": combined},
    )


def probe_python() -> CapabilityEvidence:
    """Inspect the running interpreter rather than a configured default."""
    observed = platform.python_version()
    compatible = sys.version_info[:2] == (3, 12)
    return CapabilityEvidence(
        capability="python",
        state=CapabilityState.AVAILABLE if compatible else CapabilityState.INCOMPATIBLE,
        requirement=CapabilityRequirement.REQUIRED_LOCAL,
        reason_code="PYTHON_AVAILABLE" if compatible else "PYTHON_VERSION_MISMATCH",
        summary=f"running on {platform.python_implementation()} {observed}",
        observed_version=observed,
        expected_version="3.12",
        command=[sys.executable, "--version"],
        details={"executable": sys.executable},
    )


def probe_onnxruntime() -> CapabilityEvidence:
    """Inspect the imported ONNX Runtime package and real provider list."""
    try:
        import onnxruntime
    except ImportError as error:
        return CapabilityEvidence(
            capability="onnxruntime",
            state=CapabilityState.ABSENT,
            requirement=CapabilityRequirement.REQUIRED_LOCAL,
            reason_code="ONNXRUNTIME_NOT_INSTALLED",
            summary="ONNX Runtime cannot be imported",
            expected_version="1.25.0",
            details={"error": _bounded(str(error))},
        )
    version = str(onnxruntime.__version__)
    providers = [str(provider) for provider in onnxruntime.get_available_providers()]
    compatible = version == "1.25.0" and "CPUExecutionProvider" in providers
    return CapabilityEvidence(
        capability="onnxruntime",
        state=CapabilityState.AVAILABLE if compatible else CapabilityState.INCOMPATIBLE,
        requirement=CapabilityRequirement.REQUIRED_LOCAL,
        reason_code="ONNXRUNTIME_CPU_AVAILABLE" if compatible else "ONNXRUNTIME_CPU_INCOMPATIBLE",
        summary=f"ONNX Runtime {version} exposes {', '.join(providers) or 'no providers'}",
        observed_version=version,
        expected_version="1.25.0",
        command=[sys.executable, "-c", "import onnxruntime"],
        details={"providers": providers},
    )


def probe_container_runtime() -> CapabilityEvidence:
    """Distinguish a missing Docker CLI from an inaccessible daemon."""
    executable = shutil.which("docker")
    if executable is None:
        return CapabilityEvidence(
            capability="docker",
            state=CapabilityState.ABSENT,
            requirement=CapabilityRequirement.OPTIONAL,
            reason_code="DOCKER_CLI_NOT_FOUND",
            summary="Docker CLI is not available on PATH",
            command=["docker", "info"],
        )
    try:
        result = _run([executable, "info", "--format", "{{json .ServerVersion}}"], 8.0)
    except subprocess.TimeoutExpired:
        return CapabilityEvidence(
            capability="docker",
            state=CapabilityState.INACCESSIBLE,
            requirement=CapabilityRequirement.OPTIONAL,
            reason_code="DOCKER_DAEMON_TIMEOUT",
            summary="Docker CLI exists but its daemon probe timed out",
            command=["docker", "info"],
            details={"resolved_path": executable},
        )
    if result.returncode != 0:
        return CapabilityEvidence(
            capability="docker",
            state=CapabilityState.INACCESSIBLE,
            requirement=CapabilityRequirement.OPTIONAL,
            reason_code="DOCKER_DAEMON_UNREACHABLE",
            summary="Docker CLI exists but the daemon is unreachable",
            command=["docker", "info"],
            details={"resolved_path": executable, "error": _bounded(result.stderr)},
        )
    version = _bounded(result.stdout).strip('"')
    return CapabilityEvidence(
        capability="docker",
        state=CapabilityState.AVAILABLE,
        requirement=CapabilityRequirement.OPTIONAL,
        reason_code="DOCKER_DAEMON_AVAILABLE",
        summary=f"Docker daemon {version} is reachable",
        observed_version=version,
        command=["docker", "info"],
        details={"resolved_path": executable},
    )


def probe_nvidia() -> CapabilityEvidence:
    """Inspect an NVIDIA device without assuming CUDA from configuration."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return CapabilityEvidence(
            capability="nvidia_gpu",
            state=CapabilityState.ABSENT,
            requirement=CapabilityRequirement.TARGET_ONLY,
            reason_code="NVIDIA_SMI_NOT_FOUND",
            summary="nvidia-smi is not available on PATH",
            command=["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        )
    result = _run(
        [executable, "--query-gpu=uuid,name", "--format=csv,noheader"],
        timeout_seconds=8.0,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        return CapabilityEvidence(
            capability="nvidia_gpu",
            state=CapabilityState.ABSENT,
            requirement=CapabilityRequirement.TARGET_ONLY,
            reason_code="NO_NVIDIA_DEVICE",
            summary="nvidia-smi is installed but no usable device was reported",
            command=["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            details={"error": _bounded(result.stderr)},
        )
    return CapabilityEvidence(
        capability="nvidia_gpu",
        state=CapabilityState.AVAILABLE,
        requirement=CapabilityRequirement.TARGET_ONLY,
        reason_code="NVIDIA_DEVICE_AVAILABLE",
        summary=f"{len(lines)} NVIDIA device(s) reported",
        command=["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        details={"devices": lines},
    )


def probe_cuda() -> CapabilityEvidence:
    """Inspect the CUDA compiler independently of GPU availability."""
    if shutil.which("nvcc") is None:
        return CapabilityEvidence(
            capability="cuda_toolkit",
            state=CapabilityState.ABSENT,
            requirement=CapabilityRequirement.TARGET_ONLY,
            reason_code="CUDA_TOOLKIT_NOT_FOUND",
            summary="nvcc is not available on PATH",
            expected_version="12.8",
            command=["nvcc", "--version"],
        )
    return probe_command(
        capability="cuda_toolkit",
        executable="nvcc",
        version_args=["--version"],
        requirement=CapabilityRequirement.TARGET_ONLY,
        expected_version="12.8",
    )


def probe_tensorrt() -> CapabilityEvidence:
    """Inspect TensorRT independently of ONNX Runtime provider availability."""
    executable = shutil.which("trtexec")
    if executable is None:
        return CapabilityEvidence(
            capability="tensorrt",
            state=CapabilityState.ABSENT,
            requirement=CapabilityRequirement.OPTIONAL,
            reason_code="TENSORRT_NOT_FOUND",
            summary="trtexec is not available on PATH",
            expected_version="10.14.1.48",
            command=["trtexec", "--version"],
        )
    return probe_command(
        capability="tensorrt",
        executable=executable,
        version_args=["--version"],
        requirement=CapabilityRequirement.OPTIONAL,
        expected_version="10.14.1.48",
    )


def probe_dataset() -> CapabilityEvidence:
    """Inspect dataset root and local license receipt as separate prerequisites."""
    root = os.environ.get("OPENLANE_V2_ROOT")
    if not root:
        return CapabilityEvidence(
            capability="openlane_v2_dataset",
            state=CapabilityState.ABSENT,
            requirement=CapabilityRequirement.TARGET_ONLY,
            reason_code="DATASET_ROOT_UNSET",
            summary="OPENLANE_V2_ROOT is not set",
        )
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return CapabilityEvidence(
            capability="openlane_v2_dataset",
            state=CapabilityState.ABSENT,
            requirement=CapabilityRequirement.TARGET_ONLY,
            reason_code="DATASET_ROOT_NOT_FOUND",
            summary="the configured dataset root does not exist",
            details={"configured": True},
        )
    receipt = Path(".junctionlens/license-acknowledgments/openlane-v2-v2.1.json")
    if not receipt.is_file():
        return CapabilityEvidence(
            capability="openlane_v2_dataset",
            state=CapabilityState.INACCESSIBLE,
            requirement=CapabilityRequirement.TARGET_ONLY,
            reason_code="DATA_LICENSE_ACK_REQUIRED",
            summary="the dataset root exists but the required local terms receipt is absent",
            details={"configured": True},
        )
    return CapabilityEvidence(
        capability="openlane_v2_dataset",
        state=CapabilityState.AVAILABLE,
        requirement=CapabilityRequirement.TARGET_ONLY,
        reason_code="DATASET_REGISTERED",
        summary="the dataset root and local terms receipt are present",
        details={"configured": True},
    )


def probe_cpp_truth_file(project_root: Path) -> CapabilityEvidence:
    """Read the built C++ truth probe when present."""
    candidate = project_root / "build" / "cpu" / "junctionlens-toolchain-probe"
    if not candidate.is_file():
        return CapabilityEvidence(
            capability="cpp_truth_probe",
            state=CapabilityState.ABSENT,
            requirement=CapabilityRequirement.REQUIRED_LOCAL,
            reason_code="CPP_TRUTH_PROBE_NOT_BUILT",
            summary="the C++ toolchain truth probe has not been built",
            command=[str(candidate)],
        )
    result = _run([str(candidate)])
    try:
        payload = load_json_object(
            result.stdout.encode(),
            "C++ truth probe output",
            ParseLimits(max_bytes=_OUTPUT_LIMIT, max_depth=8, max_nodes=128),
        )
    except (ParseBoundaryError, TypeError):
        return CapabilityEvidence(
            capability="cpp_truth_probe",
            state=CapabilityState.ERROR,
            requirement=CapabilityRequirement.REQUIRED_LOCAL,
            reason_code="CPP_TRUTH_PROBE_INVALID_JSON",
            summary="the C++ truth probe did not return valid JSON",
            command=[str(candidate)],
            details={"output": _bounded(result.stdout)},
        )
    return CapabilityEvidence(
        capability="cpp_truth_probe",
        state=CapabilityState.AVAILABLE,
        requirement=CapabilityRequirement.REQUIRED_LOCAL,
        reason_code="CPP_TRUTH_PROBE_AVAILABLE",
        summary="the built C++ probe returned observed toolchain metadata",
        command=[str(candidate)],
        details=payload,
    )

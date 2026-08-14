#!/usr/bin/env python3
"""Noninteractive resumable Linux GPU qualification runner."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from junctionlens.security.parsing import (  # noqa: E402
    ParseBoundaryError,
    ParseLimits,
    load_json_object_path,
)
from junctionlens.security.redaction import redact_sensitive_text  # noqa: E402

SCHEMA_VERSION = "junctionlens.remote-qualification.v1"
MINIMUM_DRIVER = (570, 26)
MINIMUM_DISK_BYTES = 50 * 1024**3
MINIMUM_INODES = 100_000


class QualificationError(RuntimeError):
    """Raised when a remote qualification invariant fails."""


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """One durable phase result."""

    name: str
    status: str
    reason_code: str
    required: bool


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as output:
        output.write(_canonical_json(value) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)+)(?!\d)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def _run_probe(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _redact(value: str, replacements: dict[str, str]) -> str:
    result = value
    for sensitive, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if sensitive:
            result = result.replace(sensitive, replacement)
    return redact_sensitive_text(result)


class Runner:
    """Own durable phase execution and top-level evidence generation."""

    def __init__(self, root: Path, result_root: Path, config: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.result_root = result_root.resolve()
        self.config = config
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.commands_path = self.result_root / "commands.jsonl"
        self.source_digest = str(config["source_content_sha256"])
        self.source_commit = str(config["source_commit"])
        self.profile = str(config["profile"])
        if self.profile not in {
            "m0.3",
            "runtime-cuda",
            "runtime-performance",
            "core",
            "full-v1",
        }:
            raise QualificationError("qualification profile is invalid")
        data_root_value = config.get("remote_data_root")
        self.data_root = Path(str(data_root_value)).resolve() if data_root_value else None
        self.selected_gpu_uuid = ""
        self.environment: dict[str, Any] = {}
        self.replacements = {
            str(self.root): "<SOURCE_ROOT>",
            str(self.result_root): "<RESULT_ROOT>",
            str(Path.home()): "<REMOTE_HOME>",
        }
        if self.data_root is not None:
            self.replacements[str(self.data_root)] = "<LICENSED_DATA_ROOT>"

    def _phase_input_sha256(self, name: str, command: list[str] | None) -> str:
        return _sha256_bytes(
            _canonical_json(
                {
                    "source_content_sha256": self.source_digest,
                    "source_commit": self.source_commit,
                    "profile": self.profile,
                    "phase": name,
                    "command": command,
                    "selected_gpu_uuid": self.selected_gpu_uuid,
                }
            )
        )

    def _phase_root(self, name: str, input_sha256: str) -> Path:
        return self.result_root / "phases" / f"{name}-{input_sha256[:12]}"

    def _reuse(self, name: str, input_sha256: str, required: bool) -> PhaseResult | None:
        phase_root = self._phase_root(name, input_sha256)
        status_path = phase_root / "status.json"
        if not status_path.is_file():
            return None
        try:
            status = load_json_object_path(
                status_path,
                "remote phase status",
                ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
            )
        except ParseBoundaryError:
            return None
        if status.get("input_sha256") != input_sha256 or status.get("status") != "PASSED":
            return None
        return PhaseResult(name, "PASSED", "REUSED_HASH_MATCH", required)

    def record_phase(
        self,
        name: str,
        *,
        status: str,
        reason_code: str,
        required: bool,
        input_sha256: str,
        started_at: str,
        duration_seconds: float,
        details: dict[str, Any] | None = None,
    ) -> PhaseResult:
        phase_root = self._phase_root(name, input_sha256)
        phase_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            phase_root / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "phase": name,
                "status": status,
                "reason_code": reason_code,
                "required": required,
                "input_sha256": input_sha256,
                "started_at": started_at,
                "ended_at": _utc_now(),
                "duration_seconds": duration_seconds,
                "details": details or {},
            },
        )
        return PhaseResult(name, status, reason_code, required)

    def run_command(
        self,
        name: str,
        command: list[str],
        *,
        required: bool = True,
        environment: dict[str, str] | None = None,
        timeout: int = 14_400,
        blocked_return_codes: frozenset[int] = frozenset(),
    ) -> PhaseResult:
        input_sha256 = self._phase_input_sha256(name, command)
        reused = self._reuse(name, input_sha256, required)
        if reused is not None:
            return reused
        phase_root = self._phase_root(name, input_sha256)
        phase_root.mkdir(parents=True, exist_ok=True)
        raw_stdout = phase_root / ".stdout.raw"
        raw_stderr = phase_root / ".stderr.raw"
        started_at = _utc_now()
        started = time.monotonic()
        selected_environment = os.environ.copy()
        selected_environment.update(environment or {})
        allowlist = {
            key: selected_environment[key]
            for key in ("CUDA_VISIBLE_DEVICES", "CI", "UV_CACHE_DIR")
            if key in selected_environment
        }
        with raw_stdout.open("wb") as stdout, raw_stderr.open("wb") as stderr:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.root,
                    env=selected_environment,
                    check=False,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                )
                return_code = completed.returncode
                reason_code = "PHASE_COMMAND_PASSED" if return_code == 0 else "PHASE_COMMAND_FAILED"
            except subprocess.TimeoutExpired:
                return_code = 124
                reason_code = "PHASE_COMMAND_TIMEOUT"
        duration = time.monotonic() - started
        stdout_text = raw_stdout.read_text(encoding="utf-8", errors="replace")
        stderr_text = raw_stderr.read_text(encoding="utf-8", errors="replace")
        (phase_root / "stdout.log").write_text(
            _redact(stdout_text, self.replacements), encoding="utf-8"
        )
        (phase_root / "stderr.log").write_text(
            _redact(stderr_text, self.replacements), encoding="utf-8"
        )
        raw_stdout.unlink(missing_ok=True)
        raw_stderr.unlink(missing_ok=True)
        command_record = {
            "phase": name,
            "command": [_redact(part, self.replacements) for part in command],
            "environment": {
                key: _redact(value, self.replacements) for key, value in allowlist.items()
            },
            "started_at": started_at,
            "duration_seconds": duration,
            "return_code": return_code,
        }
        with self.commands_path.open("a", encoding="utf-8") as commands:
            commands.write(json.dumps(command_record, sort_keys=True, allow_nan=False) + "\n")
            commands.flush()
            os.fsync(commands.fileno())
        status = (
            "PASSED"
            if return_code == 0
            else "BLOCKED"
            if return_code in blocked_return_codes
            else "FAILED"
        )
        if status == "BLOCKED":
            reason_code = "PHASE_COMMAND_BLOCKED_INFRASTRUCTURE"
        return self.record_phase(
            name,
            status=status,
            reason_code=reason_code,
            required=required,
            input_sha256=input_sha256,
            started_at=started_at,
            duration_seconds=duration,
            details={"return_code": return_code},
        )

    def blocked_phase(self, name: str, reason_code: str, *, required: bool = True) -> PhaseResult:
        started = _utc_now()
        input_sha256 = self._phase_input_sha256(name, None)
        return self.record_phase(
            name,
            status="BLOCKED",
            reason_code=reason_code,
            required=required,
            input_sha256=input_sha256,
            started_at=started,
            duration_seconds=0.0,
        )

    def preflight(self) -> PhaseResult:
        name = "01-preflight"
        input_sha256 = self._phase_input_sha256(name, None)
        reused = self._reuse(name, input_sha256, True)
        if reused is not None:
            status = load_json_object_path(
                self._phase_root(name, input_sha256) / "environment.json",
                "remote qualification environment",
                ParseLimits(max_bytes=4 * 1024 * 1024, max_depth=16, max_nodes=100_000),
            )
            self.environment = status
            self.selected_gpu_uuid = str(status["selected_gpu"]["uuid"])
            return reused
        started_at = _utc_now()
        started = time.monotonic()
        failures: list[str] = []
        os_release: dict[str, str] = {}
        release_path = Path("/etc/os-release")
        if release_path.is_file():
            for line in release_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator:
                    os_release[key] = value.strip('"')
        if os_release.get("ID") != "ubuntu" or os_release.get("VERSION_ID") != "24.04":
            failures.append("PREFLIGHT_UBUNTU_24_04_REQUIRED")
        if platform.machine() not in {"x86_64", "AMD64"}:
            failures.append("PREFLIGHT_X86_64_REQUIRED")
        nvidia_smi = shutil.which("nvidia-smi")
        gpu_rows: list[dict[str, Any]] = []
        driver_version = ""
        if nvidia_smi is None:
            failures.append("PREFLIGHT_NVIDIA_SMI_REQUIRED")
        else:
            driver = _run_probe(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader,nounits"]
            )
            driver_version = driver.stdout.splitlines()[0].strip() if driver.returncode == 0 else ""
            if _version_tuple(driver_version) < MINIMUM_DRIVER:
                failures.append("PREFLIGHT_DRIVER_BELOW_570_26")
            query = _run_probe(
                [
                    nvidia_smi,
                    "--query-gpu=uuid,name,compute_cap,memory.total,temperature.gpu,"
                    "utilization.gpu,utilization.memory,pstate",
                    "--format=csv,noheader,nounits",
                ]
            )
            if query.returncode != 0:
                failures.append("PREFLIGHT_GPU_QUERY_FAILED")
            else:
                for line in query.stdout.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if len(values) != 8:
                        continue
                    try:
                        memory_mib = int(values[3])
                        temperature = int(values[4])
                    except ValueError:
                        continue
                    gpu_rows.append(
                        {
                            "uuid": values[0],
                            "name": values[1],
                            "compute_capability": values[2],
                            "memory_mib": memory_mib,
                            "temperature_c": temperature,
                            "utilization_gpu_percent": values[5],
                            "utilization_memory_percent": values[6],
                            "pstate": values[7],
                            "healthy": memory_mib >= 6144 and temperature < 90,
                        }
                    )
        healthy = sorted((gpu for gpu in gpu_rows if gpu["healthy"]), key=lambda gpu: gpu["uuid"])
        override = self.config.get("gpu_uuid")
        if override:
            selected = next((gpu for gpu in healthy if gpu["uuid"] == override), None)
            selection_reason = "validated explicit override"
        else:
            selected = healthy[0] if healthy else None
            selection_reason = "lexicographically smallest healthy UUID"
        if selected is None:
            failures.append("PREFLIGHT_NO_QUALIFYING_GPU")
            selected = {"uuid": "", "name": "", "compute_capability": "", "memory_mib": 0}
        self.selected_gpu_uuid = str(selected["uuid"])
        nvcc = _run_probe(["nvcc", "--version"]) if shutil.which("nvcc") else None
        cuda_version = ""
        if nvcc is None or nvcc.returncode != 0:
            failures.append("PREFLIGHT_CUDA_TOOLKIT_REQUIRED")
        else:
            release = re.search(r"release\s+(\d+\.\d+)", nvcc.stdout + nvcc.stderr)
            cuda_version = release.group(1) if release else ""
            if cuda_version != "12.8":
                failures.append("PREFLIGHT_CUDA_12_8_REQUIRED")
        cudnn = _run_probe(["dpkg-query", "-W", "-f=${Version}", "libcudnn9-cuda-12"])
        cudnn_version = cudnn.stdout.strip() if cudnn.returncode == 0 else ""
        if not cudnn_version.startswith("9.14.0.64"):
            failures.append("PREFLIGHT_CUDNN_9_14_REQUIRED")
        tensorrt = _run_probe(["trtexec", "--version"]) if shutil.which("trtexec") else None
        disk = shutil.disk_usage(self.root)
        file_system = os.statvfs(self.root)
        free_inodes = file_system.f_favail
        if disk.free < MINIMUM_DISK_BYTES:
            failures.append("PREFLIGHT_DISK_SPACE_LOW")
        if free_inodes < MINIMUM_INODES:
            failures.append("PREFLIGHT_INODES_LOW")
        competing: list[dict[str, Any]] = []
        if nvidia_smi is not None:
            processes = _run_probe(
                [
                    nvidia_smi,
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            if processes.returncode != 0:
                failures.append("PREFLIGHT_COMPUTE_PROCESS_QUERY_FAILED")
            else:
                for line in processes.stdout.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if len(values) == 4:
                        competing.append(
                            {
                                "gpu_uuid": values[0],
                                "pid": values[1],
                                "process_name_sha256": _sha256_bytes(values[2].encode()),
                                "used_gpu_memory_mib": values[3],
                            }
                        )
        tools: dict[str, str] = {}
        for name, command in {
            "gcc": ["gcc", "--version"],
            "clang": ["clang", "--version"],
            "cmake": ["cmake", "--version"],
            "ninja": ["ninja", "--version"],
            "python": [sys.executable, "--version"],
            "node": ["node", "--version"],
            "docker": ["docker", "--version"],
            "nsys": ["nsys", "--version"],
        }.items():
            if shutil.which(command[0]) is None:
                tools[name] = "ABSENT"
            else:
                probe = _run_probe(command)
                tools[name] = (probe.stdout + probe.stderr).splitlines()[0].strip()
        if self.profile in {"runtime-performance", "core", "full-v1"} and tools["nsys"] == "ABSENT":
            failures.append("PREFLIGHT_NSYS_REQUIRED")
        self.environment = {
            "schema_version": SCHEMA_VERSION,
            "host": {
                "hostname_sha256": _sha256_bytes(socket.gethostname().encode()),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "os_release": os_release,
                "cpu_count": os.cpu_count(),
            },
            "gpus": gpu_rows,
            "selected_gpu": {**selected, "selection_reason": selection_reason},
            "driver_version": driver_version,
            "cuda_toolkit_version": cuda_version,
            "cudnn_package_version": cudnn_version,
            "tensorrt_probe": (tensorrt.stdout + tensorrt.stderr).strip()
            if tensorrt is not None
            else "ABSENT",
            "disk_free_bytes": disk.free,
            "free_inodes": free_inodes,
            "competing_gpu_processes": competing,
            "tools": tools,
            "dataset_configured": self.data_root is not None,
            "dataset_exists": self.data_root.is_dir() if self.data_root is not None else False,
            "failures": failures,
        }
        phase_root = self._phase_root(name, input_sha256)
        phase_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(phase_root / "environment.json", self.environment)
        _atomic_json(self.result_root / "environment.json", self.environment)
        duration = time.monotonic() - started
        return self.record_phase(
            name,
            status="PASSED" if not failures else "BLOCKED",
            reason_code="PREFLIGHT_PASSED" if not failures else failures[0],
            required=True,
            input_sha256=input_sha256,
            started_at=started_at,
            duration_seconds=duration,
            details={"failure_count": len(failures)},
        )

    def write_final(self, phases: list[PhaseResult]) -> str:
        required_failures = [
            phase for phase in phases if phase.required and phase.status != "PASSED"
        ]
        if any(phase.status == "BLOCKED" for phase in required_failures):
            status = "BLOCKED"
        elif required_failures:
            status = "FAILED"
        else:
            status = "PASSED"
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "profile": self.profile,
            "source_commit": self.source_commit,
            "source_content_sha256": self.source_digest,
            "selected_gpu_uuid": self.selected_gpu_uuid,
            "phases": [asdict(phase) for phase in phases],
            "generated_at": _utc_now(),
        }
        _atomic_json(self.result_root / "status.json", summary)
        if status != "PASSED":
            missing = "\n".join(
                f"- `{phase.reason_code}` in `{phase.name}`" for phase in required_failures
            )
            (self.result_root / "USER_ACTION_REQUIRED.md").write_text(
                "# User action required\n\n"
                "Remote qualification did not pass.\n"
                "Resolve the following prerequisites or failures, then rerun "
                "`./scripts/gpu/qualify_remote.sh`.\n\n"
                f"{missing}\n",
                encoding="utf-8",
            )
        if not (self.result_root / "benchmarks.json").exists():
            (self.result_root / "benchmarks.json").write_text(
                '{"schema_version":"junctionlens.benchmarks.v1","status":"DEFERRED_TO_M8_3"}\n',
                encoding="utf-8",
            )
        self._write_junit(phases)
        report_lines = [
            "# Remote qualification report",
            "",
            f"Status: `{status}`",
            "",
            f"Profile: `{self.profile}`",
            "",
            f"Source commit: `{self.source_commit}`",
            "",
            "## Phases",
            "",
        ]
        report_lines.extend(
            f"- `{phase.name}`: `{phase.status}` (`{phase.reason_code}`)" for phase in phases
        )
        (self.result_root / "REPORT.md").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        self._write_checksums()
        return status

    def _write_junit(self, phases: list[PhaseResult]) -> None:
        failures = sum(phase.required and phase.status != "PASSED" for phase in phases)
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="remote-qualification" tests="{len(phases)}" failures="{failures}">',
        ]
        for phase in phases:
            lines.append(f'  <testcase classname="remote" name="{phase.name}">')
            if phase.required and phase.status != "PASSED":
                lines.append(f'    <failure message="{phase.reason_code}"/>')
            lines.append("  </testcase>")
        lines.append("</testsuite>")
        (self.result_root / "junit.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_checksums(self) -> None:
        paths = sorted(
            path
            for path in self.result_root.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS" and not path.name.startswith(".")
        )
        lines = [
            f"{_sha256_file(path)}  {path.relative_to(self.result_root).as_posix()}"
            for path in paths
        ]
        (self.result_root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = load_json_object_path(
            path,
            "remote runner configuration",
            ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
        )
    except ParseBoundaryError as error:
        raise QualificationError(str(error)) from error
    required = {"profile", "source_commit", "source_content_sha256"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise QualificationError("remote runner configuration is invalid")
    return value


def _acquire_lock(lock_root: Path, source_digest: str) -> None:
    try:
        lock_root.mkdir()
    except FileExistsError:
        owner_path = lock_root / "owner.json"
        try:
            owner = load_json_object_path(
                owner_path,
                "remote runner lock owner",
                ParseLimits(max_bytes=64 * 1024, max_depth=8, max_nodes=1_000),
            )
            pid = int(owner["pid"])
            same_bundle = owner["source_content_sha256"] == source_digest
            os.kill(pid, 0)
        except (OSError, ValueError, KeyError, ParseBoundaryError):
            stale = lock_root.with_name(f"{lock_root.name}.stale-{int(time.time())}")
            lock_root.replace(stale)
            lock_root.mkdir()
        else:
            if same_bundle:
                raise QualificationError("an identical qualification runner is already live")
            raise QualificationError("a conflicting qualification runner is already live")
    _atomic_json(
        lock_root / "owner.json",
        {
            "pid": os.getpid(),
            "host_sha256": _sha256_bytes(socket.gethostname().encode()),
            "source_content_sha256": source_digest,
            "started_at": _utc_now(),
            "heartbeat": _utc_now(),
        },
    )


def run(root: Path, result_root: Path, config: dict[str, Any]) -> str:
    """Execute every available core GPU phase and persist a complete result."""
    runner = Runner(root, result_root, config)
    lock_root = result_root.parent / "qualification.lock"
    _acquire_lock(lock_root, runner.source_digest)
    phases: list[PhaseResult] = []
    try:
        preflight = runner.preflight()
        phases.append(preflight)
        if preflight.status != "PASSED":
            phases.extend(
                runner.blocked_phase(name, "BLOCKED_BY_PREFLIGHT")
                for name in (
                    "02-dependencies",
                    "03-gpu-build",
                    "04-linux-verification",
                    "05-cuda-provider-audit",
                    "06-cuda-parity",
                )
            )
            return runner.write_final(phases)
        common_environment = {
            "CUDA_VISIBLE_DEVICES": runner.selected_gpu_uuid,
            "CI": "true",
            "UV_CACHE_DIR": str(runner.root / ".cache" / "uv"),
            "JUNCTIONLENS_SOURCE_COMMIT": runner.source_commit,
        }
        dependencies = runner.run_command(
            "02-dependencies",
            ["./tools/jl", "bootstrap-cpu", "--no-sync"],
            environment=common_environment,
        )
        phases.append(dependencies)
        if dependencies.status == "PASSED":
            dependencies = runner.run_command(
                "02b-python-gpu-environment",
                [
                    "./.tools/bin/uv",
                    "sync",
                    "--locked",
                    "--python",
                    "3.12.13",
                    "--extra",
                    "cuda",
                    "--extra",
                    "analytics",
                    "--extra",
                    "service",
                ],
                environment=common_environment,
            )
            phases.append(dependencies)
        if dependencies.status == "PASSED":
            dependencies = runner.run_command(
                "02c-gpu-runtime",
                ["./.venv/bin/python", "scripts/bootstrap/bootstrap_gpu.py"],
                environment=common_environment,
            )
            phases.append(dependencies)
        if dependencies.status != "PASSED":
            phases.extend(
                runner.blocked_phase(name, "BLOCKED_BY_DEPENDENCIES")
                for name in (
                    "03-gpu-build",
                    "04-linux-verification",
                    "05-cuda-provider-audit",
                    "06-cuda-parity",
                )
            )
            return runner.write_final(phases)
        gpu_build = runner.run_command(
            "03-gpu-build",
            ["./tools/jl", "configure-gpu"],
            environment=common_environment,
        )
        phases.append(gpu_build)
        if gpu_build.status == "PASSED":
            gpu_build = runner.run_command(
                "03b-gpu-build",
                ["./tools/jl", "build-gpu"],
                environment=common_environment,
            )
            phases.append(gpu_build)
        linux_verification = (
            runner.run_command(
                "04-linux-verification",
                ["./tools/jl", "verify-m8-1-local"],
                environment=common_environment,
                timeout=28_800,
            )
            if gpu_build.status == "PASSED"
            else runner.blocked_phase("04-linux-verification", "BLOCKED_BY_GPU_BUILD")
        )
        phases.append(linux_verification)
        provider = runner.blocked_phase("05-cuda-provider-audit", "BLOCKED_BY_LINUX_VERIFICATION")
        if linux_verification.status == "PASSED":
            export_report = load_json_object_path(
                runner.root / "artifacts/m0/model-spike/model.export.json",
                "model export report",
                ParseLimits(max_bytes=4 * 1024 * 1024, max_depth=16, max_nodes=100_000),
            )
            profile_sha256 = str(export_report["profile_sha256"])
            provider_root = runner.result_root / "provider"
            provider_root.mkdir(exist_ok=True)
            provider = runner.run_command(
                "05-cuda-provider-audit",
                [
                    "./build/gpu/bin/junctionlens-runtime",
                    "doctor",
                    "--model",
                    "artifacts/m0/model-spike/model.onnx",
                    "--expected-profile-sha256",
                    profile_sha256,
                    "--provider-profile",
                    "cuda",
                    "--device-id",
                    "0",
                    "--provider-log-output",
                    str(provider_root / "cuda-provider.raw.log"),
                ],
                environment=common_environment,
            )
            phase_stdout = (
                runner._phase_root(
                    "05-cuda-provider-audit",
                    runner._phase_input_sha256(
                        "05-cuda-provider-audit",
                        [
                            "./build/gpu/bin/junctionlens-runtime",
                            "doctor",
                            "--model",
                            "artifacts/m0/model-spike/model.onnx",
                            "--expected-profile-sha256",
                            profile_sha256,
                            "--provider-profile",
                            "cuda",
                            "--device-id",
                            "0",
                            "--provider-log-output",
                            str(provider_root / "cuda-provider.raw.log"),
                        ],
                    ),
                )
                / "stdout.log"
            )
            if provider.status == "PASSED" and phase_stdout.is_file():
                shutil.copyfile(phase_stdout, runner.result_root / "provider-assignment.json")
        phases.append(provider)
        parity = (
            runner.run_command(
                "06-cuda-parity",
                [
                    "./.venv/bin/python",
                    "scripts/gpu/qualify_runtime.py",
                    "--model",
                    "artifacts/m0/model-spike/model.onnx",
                    "--profile",
                    "configs/model/m0-spike.yaml",
                    "--cpu-runtime",
                    "build/cpu/bin/junctionlens-runtime",
                    "--gpu-runtime",
                    "build/gpu/bin/junctionlens-runtime",
                    "--project-root",
                    str(runner.root),
                    "--source-commit",
                    runner.source_commit,
                    "--output",
                    str(runner.result_root / "cuda-parity"),
                ],
                environment=common_environment,
            )
            if provider.status == "PASSED"
            else runner.blocked_phase("06-cuda-parity", "BLOCKED_BY_PROVIDER_AUDIT")
        )
        phases.append(parity)
        if runner.profile in {"runtime-performance", "core", "full-v1"}:
            performance_root = runner.result_root / "performance"
            performance = (
                runner.run_command(
                    "13-accelerated-performance",
                    [
                        "./.venv/bin/python",
                        "-m",
                        "scripts.gpu.benchmark_runtime",
                        "--model",
                        "artifacts/m0/model-spike/model.onnx",
                        "--profile",
                        "configs/model/m0-spike.yaml",
                        "--runtime",
                        "build/gpu/bin/junctionlens-runtime",
                        "--project-root",
                        str(runner.root),
                        "--source-commit",
                        runner.source_commit,
                        "--gpu-uuid",
                        runner.selected_gpu_uuid,
                        "--config",
                        "configs/runtime/qualification-v1.yaml",
                        "--output",
                        str(performance_root),
                    ],
                    environment=common_environment,
                    timeout=14_400,
                    blocked_return_codes=frozenset({3}),
                )
                if parity.status == "PASSED"
                else runner.blocked_phase("13-accelerated-performance", "BLOCKED_BY_CUDA_PARITY")
            )
            phases.append(performance)
            if (
                performance.status == "PASSED"
                and (performance_root / "qualification.json").is_file()
            ):
                shutil.copyfile(
                    performance_root / "qualification.json",
                    runner.result_root / "benchmarks.json",
                )
            profiler_root = runner.result_root / "profiler"
            profiler = (
                runner.run_command(
                    "14-profiler-automation",
                    [
                        "./.venv/bin/python",
                        "-m",
                        "scripts.gpu.profile_runtime",
                        "--model",
                        "artifacts/m0/model-spike/model.onnx",
                        "--profile",
                        "configs/model/m0-spike.yaml",
                        "--runtime",
                        "build/gpu/bin/junctionlens-runtime",
                        "--project-root",
                        str(runner.root),
                        "--source-commit",
                        runner.source_commit,
                        "--gpu-uuid",
                        runner.selected_gpu_uuid,
                        "--config",
                        "configs/runtime/qualification-v1.yaml",
                        "--output",
                        str(profiler_root),
                    ],
                    environment=common_environment,
                    timeout=900,
                )
                if parity.status == "PASSED"
                else runner.blocked_phase("14-profiler-automation", "BLOCKED_BY_CUDA_PARITY")
            )
            phases.append(profiler)
        tensorrt = runner.record_phase(
            "07-tensorrt-conditional",
            status="PASSED",
            reason_code="TENSORRT_DEFERRED_UNTIL_CUDA_CORE_ACCEPTED"
            if parity.status != "PASSED"
            else "TENSORRT_CONDITIONAL_PROBE_PENDING",
            required=False,
            input_sha256=runner._phase_input_sha256("07-tensorrt-conditional", None),
            started_at=_utc_now(),
            duration_seconds=0.0,
            details={"partition_complete_assumed": False},
        )
        phases.append(tensorrt)
        if runner.profile in {"core", "full-v1"}:
            phases.append(
                runner.blocked_phase(
                    "98-core-profile",
                    "CORE_PROFILE_PHASES_NOT_IMPLEMENTED_AT_THIS_SOURCE_COMMIT",
                )
            )
        if runner.profile == "full-v1":
            phases.append(
                runner.blocked_phase(
                    "99-full-v1-profile",
                    "FULL_V1_PHASES_NOT_IMPLEMENTED_AT_THIS_SOURCE_COMMIT",
                )
            )
        return runner.write_final(phases)
    finally:
        owner = lock_root / "owner.json"
        if owner.is_file():
            owner.unlink()
        with suppress(OSError):
            lock_root.rmdir()


def _parse_args() -> tuple[Path, Path, Path]:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    return arguments.source_root, arguments.result_root, arguments.config


def main() -> int:
    """Run remote qualification from a verified source checkout."""
    source_root, result_root, config_path = _parse_args()
    try:
        status = run(source_root, result_root, _load_config(config_path))
    except (OSError, QualificationError, json.JSONDecodeError) as error:
        detail = redact_sensitive_text(
            str(error),
            (source_root, result_root, config_path),
        )
        result_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            result_root / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "FAILED",
                "reason_code": "REMOTE_RUNNER_INTERNAL_ERROR",
                "detail": detail,
                "generated_at": _utc_now(),
            },
        )
        print(f"remote qualification error: {detail}", file=sys.stderr)
        return 1
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

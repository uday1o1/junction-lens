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
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from junctionlens.data.license import install_acknowledgment  # noqa: E402
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


def _directory_sha256(path: Path) -> str:
    entries: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise QualificationError("declared output tree contains a symbolic link")
        relative = item.relative_to(path).as_posix()
        if item.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif item.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "byte_size": item.stat().st_size,
                    "sha256": _sha256_file(item),
                }
            )
        else:
            raise QualificationError("declared output tree contains a special file")
    return _sha256_bytes(_canonical_json(entries))


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
        self.qualification_sha256 = str(config["qualification_sha256"])
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
        acknowledgment = config.get("license_acknowledgment")
        self.license_acknowledgment = acknowledgment if isinstance(acknowledgment, dict) else None
        self.license_acknowledgment_sha256 = (
            _sha256_bytes(_canonical_json(self.license_acknowledgment))
            if self.license_acknowledgment is not None
            else None
        )
        visual_signoff = config.get("visual_audit_signoff")
        self.visual_audit_signoff = visual_signoff if isinstance(visual_signoff, dict) else None
        if self.visual_audit_signoff is not None:
            assertions = self.visual_audit_signoff.get("assertions")
            if (
                self.visual_audit_signoff.get("schema_version")
                != "junctionlens.visual-audit-signoff.v1"
                or self.visual_audit_signoff.get("dataset_id") != "openlane-v2-v2.1"
                or self.visual_audit_signoff.get("policy_id") != "openlane-v2-v2.1-audit-v1"
                or not isinstance(assertions, dict)
                or len(assertions) != 4
                or not all(item is True for item in assertions.values())
            ):
                raise QualificationError("visual audit signoff is invalid")
        self.visual_audit_signoff_sha256 = (
            _sha256_bytes(_canonical_json(self.visual_audit_signoff))
            if self.visual_audit_signoff is not None
            else None
        )
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
        data_sensitive = name.startswith(("01-", "01b-", "08", "10-", "11-", "15-"))
        visual_sensitive = name.startswith(("08c-", "10-", "11-", "15-"))
        return _sha256_bytes(
            _canonical_json(
                {
                    "source_content_sha256": self.source_digest,
                    "source_commit": self.source_commit,
                    "profile": self.profile,
                    "phase": name,
                    "command": command,
                    "selected_gpu_uuid": self.selected_gpu_uuid,
                    "remote_data_root_sha256": _sha256_bytes(str(self.data_root).encode())
                    if data_sensitive and self.data_root is not None
                    else None,
                    "license_acknowledgment_sha256": (
                        self.license_acknowledgment_sha256 if data_sensitive else None
                    ),
                    "visual_audit_signoff_sha256": (
                        self.visual_audit_signoff_sha256 if visual_sensitive else None
                    ),
                }
            )
        )

    def install_license_acknowledgment(self) -> PhaseResult:
        """Reconstruct the validated owner-only receipt without exposing its path."""
        name = "01b-license-acknowledgment"
        input_sha256 = self._phase_input_sha256(name, None)
        reused = self._reuse(name, input_sha256, True)
        if reused is not None:
            return reused
        started_at = _utc_now()
        started = time.monotonic()
        if self.license_acknowledgment is None:
            return self.record_phase(
                name,
                status="BLOCKED",
                reason_code="LICENSE_ACKNOWLEDGMENT_REQUIRED",
                required=True,
                input_sha256=input_sha256,
                started_at=started_at,
                duration_seconds=time.monotonic() - started,
            )
        try:
            receipt = install_acknowledgment(
                self.root / "configs/data/openlane-v2-v2.1.lock.yaml",
                self.root,
                self.license_acknowledgment,
            )
        except (OSError, ValueError) as error:
            return self.record_phase(
                name,
                status="FAILED",
                reason_code="LICENSE_ACKNOWLEDGMENT_INVALID",
                required=True,
                input_sha256=input_sha256,
                started_at=started_at,
                duration_seconds=time.monotonic() - started,
                details={"error_sha256": _sha256_bytes(str(error).encode())},
            )
        return self.record_phase(
            name,
            status="PASSED",
            reason_code="LICENSE_ACKNOWLEDGMENT_INSTALLED",
            required=True,
            input_sha256=input_sha256,
            started_at=started_at,
            duration_seconds=time.monotonic() - started,
            details={
                "dataset_id": receipt["dataset_id"],
                "license_acknowledgment_sha256": self.license_acknowledgment_sha256,
            },
        )

    def _phase_root(self, name: str, input_sha256: str) -> Path:
        return self.result_root / "phases" / f"{name}-{input_sha256[:12]}"

    def _output_record(self, path: Path) -> dict[str, Any]:
        candidate = path
        if candidate.is_symlink() or not candidate.exists():
            raise QualificationError("declared phase output is missing or symbolic")
        resolved = candidate.resolve(strict=True)
        scope: str
        relative: Path
        try:
            relative = resolved.relative_to(self.result_root)
            scope = "result"
        except ValueError:
            try:
                relative = resolved.relative_to(self.root)
                scope = "source"
            except ValueError as error:
                raise QualificationError("declared phase output escapes owned roots") from error
        if resolved.is_file():
            kind = "file"
            sha256 = _sha256_file(resolved)
        elif resolved.is_dir():
            kind = "directory"
            sha256 = _directory_sha256(resolved)
        else:
            raise QualificationError("declared phase output is not a regular file or directory")
        return {
            "scope": scope,
            "path": relative.as_posix(),
            "kind": kind,
            "sha256": sha256,
        }

    def _output_path(self, record: dict[str, Any]) -> Path:
        if set(record) != {"scope", "path", "kind", "sha256"}:
            raise QualificationError("declared phase output record is invalid")
        scope = record.get("scope")
        raw_path = record.get("path")
        if scope not in {"result", "source"} or not isinstance(raw_path, str):
            raise QualificationError("declared phase output scope is invalid")
        relative = Path(raw_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise QualificationError("declared phase output path is invalid")
        return (self.result_root if scope == "result" else self.root) / relative

    def _reuse(
        self,
        name: str,
        input_sha256: str,
        required: bool,
        declared_outputs: Sequence[Path] = (),
    ) -> PhaseResult | None:
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
        if declared_outputs:
            details = status.get("details")
            raw_records = details.get("declared_outputs") if isinstance(details, dict) else None
            if not isinstance(raw_records, list) or len(raw_records) != len(declared_outputs):
                return None
            try:
                records = [dict(record) for record in raw_records if isinstance(record, dict)]
                if len(records) != len(raw_records):
                    return None
                expected = [self._output_record(path) for path in declared_outputs]
                observed = [self._output_record(self._output_path(record)) for record in records]
            except (OSError, QualificationError, ValueError):
                return None
            if records != expected or records != observed:
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
        declared_outputs: Sequence[Path] = (),
    ) -> PhaseResult:
        input_sha256 = self._phase_input_sha256(name, command)
        reused = self._reuse(name, input_sha256, required, declared_outputs)
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
        output_records: list[dict[str, Any]] = []
        output_error_sha256: str | None = None
        if return_code == 0 and declared_outputs:
            try:
                output_records = [self._output_record(path) for path in declared_outputs]
            except (OSError, QualificationError) as error:
                return_code = 1
                reason_code = "PHASE_DECLARED_OUTPUT_INVALID"
                output_error_sha256 = _sha256_bytes(str(error).encode())
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
            details={
                "return_code": return_code,
                "declared_outputs": output_records,
                "output_error_sha256": output_error_sha256,
            },
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
        if self.profile in {"core", "full-v1"}:
            if self.data_root is None:
                failures.append("PREFLIGHT_LICENSED_DATA_ROOT_REQUIRED")
            elif not self.data_root.is_dir():
                failures.append("PREFLIGHT_LICENSED_DATA_ROOT_MISSING")
            if self.license_acknowledgment is None:
                failures.append("PREFLIGHT_LICENSE_ACKNOWLEDGMENT_REQUIRED")
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
                "physical_memory_bytes": (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")),
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
            "license_acknowledgment_configured": self.license_acknowledgment is not None,
            "license_acknowledgment_sha256": self.license_acknowledgment_sha256,
            "visual_audit_signoff_configured": self.visual_audit_signoff is not None,
            "visual_audit_signoff_sha256": self.visual_audit_signoff_sha256,
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
            "qualification_sha256": self.qualification_sha256,
            "selected_gpu_uuid": self.selected_gpu_uuid,
            "phases": [asdict(phase) for phase in phases],
            "generated_at": _utc_now(),
        }
        _atomic_json(self.result_root / "status.json", summary)
        user_action = self.result_root / "USER_ACTION_REQUIRED.md"
        if status == "PASSED":
            user_action.unlink(missing_ok=True)
        else:
            missing = "\n".join(
                f"- `{phase.reason_code}` in `{phase.name}`" for phase in required_failures
            )
            visual_procedure = ""
            if any(
                phase.reason_code == "LICENSED_VISUAL_AUDIT_REVIEW_REQUIRED"
                for phase in required_failures
            ):
                visual_procedure = (
                    "\n## Licensed visual review\n\n"
                    "Inspect every file below "
                    "`<downloaded-result>/licensed-data/private-visual-audit`.\n"
                    "If and only if all four assertions are true, run this command from the "
                    "repository root.\n\n"
                    "```bash\n"
                    "uv run --locked junctionlens data signoff-visual-audit \\\n"
                    "  --bundle <downloaded-result>/licensed-data/private-visual-audit \\\n"
                    "  --accept-camera-projection-alignment \\\n"
                    "  --accept-bev-geometry-alignment \\\n"
                    "  --accept-label-identity-and-topology \\\n"
                    "  --confirm-private-data-handling\n"
                    "./scripts/gpu/qualify_remote.sh --profile core\n"
                    "```\n"
                )
            user_action.write_text(
                "# User action required\n\n"
                "Remote qualification did not pass.\n"
                "Resolve the following prerequisites or failures, then rerun "
                "`./scripts/gpu/qualify_remote.sh`.\n\n"
                f"{missing}\n"
                f"{visual_procedure}",
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
    required = {
        "schema_version",
        "profile",
        "source_commit",
        "source_content_sha256",
        "remote_data_root",
        "gpu_uuid",
        "license_acknowledgment",
        "visual_audit_signoff",
        "qualification_sha256",
    }
    qualification_sha256 = value.get("qualification_sha256")
    unsigned = {key: item for key, item in value.items() if key != "qualification_sha256"}
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != "junctionlens.remote-qualification-config.v1"
        or not isinstance(qualification_sha256, str)
        or qualification_sha256 != _sha256_bytes(_canonical_json(unsigned))
    ):
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


@contextmanager
def _exclusive_gpu_lock(gpu_uuid: str, source_digest: str) -> Any:
    """Hold one cross-bundle GPU lease with a durable liveness heartbeat."""
    lock_parent_value = os.environ.get("JUNCTIONLENS_GPU_LOCK_ROOT")
    lock_parent = (
        Path(lock_parent_value).expanduser()
        if lock_parent_value
        else Path.home() / ".junctionlens/gpu-locks"
    )
    if lock_parent.is_symlink():
        raise QualificationError("GPU lock root cannot be a symbolic link")
    lock_parent.mkdir(parents=True, exist_ok=True)
    lock_root = lock_parent / _sha256_bytes(gpu_uuid.encode())
    _acquire_lock(lock_root, source_digest)
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(30.0):
            owner = lock_root / "owner.json"
            if not owner.is_file():
                return
            _atomic_json(
                owner,
                {
                    "pid": os.getpid(),
                    "host_sha256": _sha256_bytes(socket.gethostname().encode()),
                    "source_content_sha256": source_digest,
                    "started_at": started_at,
                    "heartbeat": _utc_now(),
                },
            )

    started_at = _utc_now()
    thread = threading.Thread(target=heartbeat, name="junctionlens-gpu-lock", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=5.0)
        owner = lock_root / "owner.json"
        owner.unlink(missing_ok=True)
        with suppress(OSError):
            lock_root.rmdir()


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
            blocked_names = [
                "02-dependencies",
                "03-gpu-build",
                "04-linux-verification",
                "05-cuda-provider-audit",
                "06-cuda-parity",
            ]
            if runner.profile in {"core", "full-v1"}:
                blocked_names.extend(
                    [
                        "08-dataset-adapter-evaluator",
                        "08b-dataset-evidence-audit",
                        "08c-licensed-visual-review",
                        "09-micro-overfit-audit",
                        "10-baseline-candidate-training",
                        "11-selected-study-charter-freeze",
                        "12-fault-lab",
                        "13-accelerated-performance",
                        "14-profiler-automation",
                        "15-core-evidence-report",
                    ]
                )
            phases.extend(
                runner.blocked_phase(name, "BLOCKED_BY_PREFLIGHT") for name in blocked_names
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
        license_acknowledgment: PhaseResult | None = None
        if dependencies.status == "PASSED" and runner.profile in {"core", "full-v1"}:
            license_acknowledgment = runner.install_license_acknowledgment()
            phases.append(license_acknowledgment)
        if dependencies.status != "PASSED":
            blocked_names = [
                "03-gpu-build",
                "04-linux-verification",
                "05-cuda-provider-audit",
                "06-cuda-parity",
            ]
            if runner.profile in {"core", "full-v1"}:
                blocked_names.extend(
                    [
                        "08-dataset-adapter-evaluator",
                        "08b-dataset-evidence-audit",
                        "08c-licensed-visual-review",
                        "09-micro-overfit-audit",
                        "10-baseline-candidate-training",
                        "11-selected-study-charter-freeze",
                        "12-fault-lab",
                        "13-accelerated-performance",
                        "14-profiler-automation",
                        "15-core-evidence-report",
                    ]
                )
            phases.extend(
                runner.blocked_phase(name, "BLOCKED_BY_DEPENDENCIES") for name in blocked_names
            )
            return runner.write_final(phases)
        core_data: PhaseResult | None = None
        visual_review: PhaseResult | None = None
        if runner.profile in {"core", "full-v1"}:
            data_evidence: dict[str, Any] = {}
            data_output = runner.result_root / "licensed-data"
            if license_acknowledgment is None or license_acknowledgment.status != "PASSED":
                core_data = runner.blocked_phase(
                    "08-dataset-adapter-evaluator",
                    "BLOCKED_BY_LICENSE_ACKNOWLEDGMENT",
                )
            elif runner.data_root is None:
                core_data = runner.blocked_phase(
                    "08-dataset-adapter-evaluator",
                    "LICENSED_DATA_ROOT_REQUIRED",
                )
            else:
                core_data = runner.run_command(
                    "08-dataset-adapter-evaluator",
                    [
                        "./.venv/bin/python",
                        "scripts/gpu/qualify_data.py",
                        "--project-root",
                        str(runner.root),
                        "--dataset-root",
                        str(runner.data_root),
                        "--output-root",
                        str(data_output),
                    ],
                    environment=common_environment,
                    timeout=28_800,
                    declared_outputs=(data_output,),
                )
            phases.append(core_data)
            if core_data.status == "PASSED":
                try:
                    data_evidence = load_json_object_path(
                        data_output / "qualification.json",
                        "remote licensed-data qualification",
                        ParseLimits(
                            max_bytes=1024 * 1024,
                            max_depth=16,
                            max_nodes=10_000,
                        ),
                    )
                    valid_data_evidence = (
                        data_evidence.get("mechanical_state") == "ACCEPTED"
                        and data_evidence.get("state") == "PENDING_HUMAN_INSPECTION"
                        and data_evidence.get("segment_count") == 700
                        and data_evidence.get("visual_audit_frame_count") == 12
                    )
                except (OSError, ParseBoundaryError):
                    valid_data_evidence = False
                data_audit = runner.record_phase(
                    "08b-dataset-evidence-audit",
                    status="PASSED" if valid_data_evidence else "FAILED",
                    reason_code=(
                        "DATASET_EVIDENCE_AUDIT_PASSED"
                        if valid_data_evidence
                        else "DATASET_EVIDENCE_AUDIT_FAILED"
                    ),
                    required=True,
                    input_sha256=runner._phase_input_sha256("08b-dataset-evidence-audit", None),
                    started_at=_utc_now(),
                    duration_seconds=0.0,
                    details={
                        "qualification_sha256": _sha256_file(data_output / "qualification.json")
                        if valid_data_evidence
                        else None
                    },
                )
            else:
                data_audit = runner.blocked_phase(
                    "08b-dataset-evidence-audit",
                    "BLOCKED_BY_DATASET_QUALIFICATION",
                )
            phases.append(data_audit)
            visual_review_name = "08c-licensed-visual-review"
            if data_audit.status != "PASSED":
                visual_review = runner.blocked_phase(
                    visual_review_name,
                    "BLOCKED_BY_DATASET_EVIDENCE_AUDIT",
                )
            elif runner.visual_audit_signoff is None:
                visual_review = runner.blocked_phase(
                    visual_review_name,
                    "LICENSED_VISUAL_AUDIT_REVIEW_REQUIRED",
                )
            else:
                signoff = runner.visual_audit_signoff
                valid_signoff = signoff.get("bundle_manifest_sha256") == data_evidence.get(
                    "visual_audit_manifest_sha256"
                ) and signoff.get("policy_id") == data_evidence.get("visual_audit_policy_id")
                visual_review = runner.record_phase(
                    visual_review_name,
                    status="PASSED" if valid_signoff else "FAILED",
                    reason_code=(
                        "LICENSED_VISUAL_AUDIT_REVIEW_ACCEPTED"
                        if valid_signoff
                        else "LICENSED_VISUAL_AUDIT_SIGNOFF_MISMATCH"
                    ),
                    required=True,
                    input_sha256=runner._phase_input_sha256(visual_review_name, None),
                    started_at=_utc_now(),
                    duration_seconds=0.0,
                    details={
                        "visual_audit_signoff_sha256": runner.visual_audit_signoff_sha256,
                    },
                )
            phases.append(visual_review)
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
        performance: PhaseResult | None = None
        profiler: PhaseResult | None = None
        if runner.profile in {"runtime-performance", "core", "full-v1"}:
            performance_root = runner.result_root / "performance"
            if parity.status == "PASSED":
                try:
                    with _exclusive_gpu_lock(runner.selected_gpu_uuid, runner.source_digest):
                        performance = runner.run_command(
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
                except QualificationError:
                    performance = runner.blocked_phase(
                        "13-accelerated-performance",
                        "GPU_EXCLUSIVE_LOCK_UNAVAILABLE",
                    )
            else:
                performance = runner.blocked_phase(
                    "13-accelerated-performance", "BLOCKED_BY_CUDA_PARITY"
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
            reason_code="TENSORRT_DEFERRED_CONDITIONAL",
            required=False,
            input_sha256=runner._phase_input_sha256("07-tensorrt-conditional", None),
            started_at=_utc_now(),
            duration_seconds=0.0,
            details={"partition_complete_assumed": False},
        )
        phases.append(tensorrt)
        if runner.profile in {"core", "full-v1"}:
            micro_name = "09-micro-overfit-audit"
            micro_path = runner.root / "artifacts/m0/model-spike/overfit/micro-overfit-report.json"
            micro_input_sha256 = runner._phase_input_sha256(micro_name, None)
            try:
                micro_report = load_json_object_path(
                    micro_path,
                    "remote micro-overfit report",
                    ParseLimits(max_bytes=4 * 1024 * 1024, max_depth=24, max_nodes=100_000),
                )
                valid_micro = (
                    linux_verification.status == "PASSED"
                    and micro_report.get("status") == "PASSED"
                    and micro_report.get("frames") == 32
                    and micro_report.get("nonfinite_count") == 0
                    and isinstance(micro_report.get("checkpoint_sha256"), str)
                )
            except (OSError, ParseBoundaryError):
                valid_micro = False
            micro = runner.record_phase(
                micro_name,
                status="PASSED" if valid_micro else "FAILED",
                reason_code=(
                    "MICRO_OVERFIT_GATE_AUDIT_PASSED"
                    if valid_micro
                    else "MICRO_OVERFIT_GATE_AUDIT_FAILED"
                ),
                required=True,
                input_sha256=micro_input_sha256,
                started_at=_utc_now(),
                duration_seconds=0.0,
                details={"report_sha256": _sha256_file(micro_path) if valid_micro else None},
            )
            phases.append(micro)
            models_root = runner.result_root / "core-models"
            models = (
                runner.run_command(
                    "10-baseline-candidate-training",
                    [
                        "./.venv/bin/python",
                        "scripts/gpu/qualify_models.py",
                        "--project-root",
                        str(runner.root),
                        "--dataset-root",
                        str(runner.data_root),
                        "--data-qualification-root",
                        str(runner.result_root / "licensed-data"),
                        "--output-root",
                        str(models_root),
                    ],
                    environment=common_environment,
                    timeout=432_000,
                    declared_outputs=(models_root,),
                )
                if visual_review is not None
                and visual_review.status == "PASSED"
                and parity.status == "PASSED"
                and micro.status == "PASSED"
                and runner.data_root is not None
                else runner.blocked_phase(
                    "10-baseline-candidate-training",
                    "BLOCKED_BY_CORE_DATA_OR_RUNTIME_GATE",
                )
            )
            phases.append(models)
            study_root = runner.result_root / "core-study"
            study = (
                runner.run_command(
                    "11-selected-study-charter-freeze",
                    [
                        "./.venv/bin/python",
                        "scripts/gpu/qualify_study.py",
                        "--project-root",
                        str(runner.root),
                        "--dataset-root",
                        str(runner.data_root),
                        "--data-qualification-root",
                        str(runner.result_root / "licensed-data"),
                        "--models-root",
                        str(models_root),
                        "--environment",
                        str(runner.result_root / "environment.json"),
                        "--performance",
                        str(runner.result_root / "performance/qualification.json"),
                        "--provider-assignment",
                        str(runner.result_root / "provider-assignment.json"),
                        "--output-root",
                        str(study_root),
                        "--source-commit",
                        runner.source_commit,
                    ],
                    environment=common_environment,
                    timeout=432_000,
                    declared_outputs=(study_root,),
                )
                if models.status == "PASSED"
                and performance is not None
                and performance.status == "PASSED"
                and provider.status == "PASSED"
                and runner.data_root is not None
                else runner.blocked_phase(
                    "11-selected-study-charter-freeze",
                    "BLOCKED_BY_MODEL_OR_HARDWARE_BASELINE",
                )
            )
            phases.append(study)
            faults_root = runner.result_root / "fault-lab"
            faults = (
                runner.run_command(
                    "12-fault-lab",
                    [
                        "./.venv/bin/python",
                        "scripts/gpu/qualify_faults.py",
                        "--project-root",
                        str(runner.root),
                        "--output-root",
                        str(faults_root),
                    ],
                    environment=common_environment,
                    timeout=28_800,
                    declared_outputs=(faults_root,),
                )
                if linux_verification.status == "PASSED"
                else runner.blocked_phase("12-fault-lab", "BLOCKED_BY_LINUX_VERIFICATION")
            )
            phases.append(faults)
            core_evidence_root = runner.result_root / "portfolio-core"
            core_evidence = (
                runner.run_command(
                    "15-core-evidence-report",
                    [
                        "./.venv/bin/python",
                        "scripts/gpu/qualify_core_evidence.py",
                        "--data",
                        str(runner.result_root / "licensed-data"),
                        "--micro",
                        str(micro_path),
                        "--models",
                        str(models_root),
                        "--study",
                        str(study_root),
                        "--parity",
                        str(runner.result_root / "cuda-parity/qualification.json"),
                        "--performance",
                        str(runner.result_root / "performance/qualification.json"),
                        "--profiler",
                        str(runner.result_root / "profiler/profile-evidence.json"),
                        "--provider",
                        str(runner.result_root / "provider-assignment.json"),
                        "--faults",
                        str(faults_root),
                        "--visual-signoff-sha256",
                        str(runner.visual_audit_signoff_sha256),
                        "--source-commit",
                        runner.source_commit,
                        "--output",
                        str(core_evidence_root),
                    ],
                    environment=common_environment,
                    timeout=28_800,
                    declared_outputs=(core_evidence_root,),
                )
                if study.status == "PASSED"
                and faults.status == "PASSED"
                and profiler is not None
                and profiler.status == "PASSED"
                and runner.visual_audit_signoff_sha256 is not None
                else runner.blocked_phase(
                    "15-core-evidence-report",
                    "BLOCKED_BY_INCOMPLETE_CORE_EVIDENCE",
                )
            )
            phases.append(core_evidence)
        if runner.profile == "full-v1":
            phases.append(
                runner.blocked_phase(
                    "99-full-v1-profile",
                    "EXTENDED_V1_REQUIRES_ACCEPTED_CORE_HANDOFF",
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

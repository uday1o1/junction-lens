"""Tests for durable remote GPU qualification state and public handoff behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.gpu.remote_runner import (
    PhaseResult,
    QualificationError,
    Runner,
    _exclusive_gpu_lock,
)
from scripts.gpu.remote_runner import run as run_remote
from scripts.gpu.source_bundle import create_bundle, create_remote_config

from junctionlens.data.license import acknowledge_licenses


def _runner(tmp_path: Path) -> Runner:
    root = Path(__file__).resolve().parents[2]
    return Runner(
        root,
        tmp_path / "results",
        {
            "schema_version": "junctionlens.remote-qualification-config.v1",
            "profile": "runtime-cuda",
            "source_commit": "a" * 40,
            "source_content_sha256": "b" * 64,
            "remote_data_root": None,
            "gpu_uuid": None,
            "license_acknowledgment": None,
            "visual_audit_signoff": None,
            "qualification_sha256": "c" * 64,
        },
    )


def test_final_result_is_fail_closed_and_self_hashing(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    status = runner.write_final(
        [
            PhaseResult("preflight", "PASSED", "PREFLIGHT_PASSED", True),
            PhaseResult("cuda-parity", "BLOCKED", "CUDA_PROVIDER_UNAVAILABLE", True),
            PhaseResult("tensorrt", "PASSED", "CONDITIONAL_REJECTED", False),
        ]
    )
    assert status == "BLOCKED"
    result = json.loads((runner.result_root / "status.json").read_text(encoding="utf-8"))
    assert result["status"] == "BLOCKED"
    assert (runner.result_root / "USER_ACTION_REQUIRED.md").is_file()
    checksum_paths = {
        line.split("  ", 1)[1]
        for line in (runner.result_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert {"status.json", "environment.json"}.intersection(checksum_paths) == {"status.json"}
    assert "USER_ACTION_REQUIRED.md" in checksum_paths


def test_remote_config_is_derived_from_verified_source_manifest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.json"
    create_bundle(
        root,
        tmp_path / "source.tar",
        manifest_path,
        require_clean=False,
    )
    config_path = tmp_path / "config.json"
    value = create_remote_config(
        manifest_path,
        config_path,
        profile="runtime-cuda",
        remote_data_root="/licensed/data",
        gpu_uuid="GPU-1234",
    )
    assert value["source_commit"]
    assert value["source_content_sha256"]
    assert json.loads(config_path.read_text(encoding="utf-8")) == value


def test_runtime_performance_profile_is_distinct_from_cuda_correctness(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.json"
    create_bundle(root, tmp_path / "source.tar", manifest_path, require_clean=False)
    value = create_remote_config(
        manifest_path,
        tmp_path / "config.json",
        profile="runtime-performance",
        remote_data_root=None,
        gpu_uuid="GPU-1234",
    )
    assert value["profile"] == "runtime-performance"
    performance = Runner(root, tmp_path / "results", value)
    assert performance.profile == "runtime-performance"


def test_public_handoff_requires_one_explicit_host_prerequisite() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("JUNCTIONLENS_GPU_HOST", None)
    completed = subprocess.run(
        [str(root / "scripts/gpu/qualify_remote.sh"), "--profile", "runtime-cuda"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert "Set JUNCTIONLENS_GPU_HOST to an SSH alias" in completed.stderr


def test_infrastructure_exit_is_recorded_as_blocked(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_command(
        "contaminated-benchmark",
        [sys.executable, "-c", "raise SystemExit(3)"],
        blocked_return_codes=frozenset({3}),
    )
    assert result.status == "BLOCKED"
    assert result.reason_code == "PHASE_COMMAND_BLOCKED_INFRASTRUCTURE"


def test_phase_reuse_rehashes_declared_outputs(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    output = runner.result_root / "evidence.json"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(output)!r}).write_text('accepted')",
    ]
    first = runner.run_command("declared-output", command, declared_outputs=(output,))
    second = runner.run_command("declared-output", command, declared_outputs=(output,))
    output.write_text("tampered", encoding="utf-8")
    third = runner.run_command("declared-output", command, declared_outputs=(output,))

    assert first.status == "PASSED"
    assert second.reason_code == "REUSED_HASH_MATCH"
    assert third.status == "PASSED"
    assert third.reason_code == "PHASE_COMMAND_PASSED"
    assert output.read_text(encoding="utf-8") == "accepted"


def test_phase_rejects_missing_declared_output(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    missing = runner.result_root / "missing.json"
    result = runner.run_command(
        "missing-output",
        [sys.executable, "-c", "pass"],
        declared_outputs=(missing,),
    )

    assert result.status == "FAILED"
    assert result.reason_code == "PHASE_DECLARED_OUTPUT_INVALID"


def test_visual_signoff_invalidates_only_review_and_downstream_phases(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    base: dict[str, Any] = {
        "schema_version": "junctionlens.remote-qualification-config.v1",
        "profile": "core",
        "source_commit": "a" * 40,
        "source_content_sha256": "b" * 64,
        "remote_data_root": "/licensed/data",
        "gpu_uuid": None,
        "license_acknowledgment": None,
        "visual_audit_signoff": None,
        "qualification_sha256": "c" * 64,
    }
    unsigned = Runner(root, tmp_path / "unsigned", base)
    signed_config = dict(base)
    signed_config["visual_audit_signoff"] = {
        "schema_version": "junctionlens.visual-audit-signoff.v1",
        "dataset_id": "openlane-v2-v2.1",
        "policy_id": "openlane-v2-v2.1-audit-v1",
        "bundle_manifest_sha256": "d" * 64,
        "reviewed_file_count": 24,
        "assertions": {
            "camera_projection_alignment_accepted": True,
            "bev_geometry_alignment_accepted": True,
            "label_identity_and_topology_accepted": True,
            "private_data_handling_confirmed": True,
        },
        "reviewed_at": "2026-08-14T00:00:00+00:00",
    }
    signed_config["qualification_sha256"] = "e" * 64
    signed = Runner(root, tmp_path / "signed", signed_config)

    assert unsigned._phase_input_sha256("02-dependencies", []) == signed._phase_input_sha256(
        "02-dependencies", []
    )
    assert unsigned._phase_input_sha256(
        "08-dataset-adapter-evaluator", []
    ) == signed._phase_input_sha256("08-dataset-adapter-evaluator", [])
    assert unsigned._phase_input_sha256(
        "08c-licensed-visual-review", None
    ) != signed._phase_input_sha256("08c-licensed-visual-review", None)
    assert unsigned._phase_input_sha256(
        "10-baseline-candidate-training", []
    ) != signed._phase_input_sha256("10-baseline-candidate-training", [])


def test_gpu_performance_lock_is_cross_bundle_exclusive_and_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUNCTIONLENS_GPU_LOCK_ROOT", str(tmp_path / "gpu-locks"))

    with (
        _exclusive_gpu_lock("GPU-TEST", "a" * 64),
        pytest.raises(QualificationError, match="already live"),
        _exclusive_gpu_lock("GPU-TEST", "b" * 64),
    ):
        pass

    with _exclusive_gpu_lock("GPU-TEST", "c" * 64):
        assert any((tmp_path / "gpu-locks").iterdir())


def test_core_profile_runs_mechanical_data_phase_before_visual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    lock = root / "configs/data/openlane-v2-v2.1.lock.yaml"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(Path("configs/data/openlane-v2-v2.1.lock.yaml").read_bytes())
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    acknowledgment = acknowledge_licenses(
        lock,
        root,
        ["CC-BY-NC-SA-4.0", "nuScenes-terms", "Argoverse-2-terms"],
        confirmed_restricted_noncommercial_use=True,
    )
    config = {
        "schema_version": "junctionlens.remote-qualification-config.v1",
        "profile": "core",
        "source_commit": "a" * 40,
        "source_content_sha256": "b" * 64,
        "remote_data_root": str(dataset),
        "gpu_uuid": None,
        "license_acknowledgment": acknowledgment,
        "visual_audit_signoff": None,
        "qualification_sha256": "c" * 64,
    }

    def fake_preflight(self: Runner) -> PhaseResult:
        self.selected_gpu_uuid = "GPU-TEST"
        return PhaseResult("01-preflight", "PASSED", "PREFLIGHT_PASSED", True)

    def fake_run_command(
        self: Runner,
        name: str,
        _command: list[str],
        **kwargs: object,
    ) -> PhaseResult:
        if name == "08-dataset-adapter-evaluator":
            declared = kwargs["declared_outputs"]
            assert isinstance(declared, tuple)
            output = declared[0]
            assert isinstance(output, Path)
            output.mkdir()
            (output / "qualification.json").write_text(
                json.dumps(
                    {
                        "mechanical_state": "ACCEPTED",
                        "state": "PENDING_HUMAN_INSPECTION",
                        "segment_count": 700,
                        "visual_audit_frame_count": 12,
                    }
                ),
                encoding="utf-8",
            )
            return PhaseResult(name, "PASSED", "PHASE_COMMAND_PASSED", True)
        if name == "03-gpu-build":
            return PhaseResult(name, "FAILED", "SEEDED_GPU_BUILD_FAILURE", True)
        return PhaseResult(name, "PASSED", "PHASE_COMMAND_PASSED", True)

    monkeypatch.setattr(Runner, "preflight", fake_preflight)
    monkeypatch.setattr(Runner, "run_command", fake_run_command)
    result_root = tmp_path / "results"

    status = run_remote(root, result_root, config)

    result = json.loads((result_root / "status.json").read_text(encoding="utf-8"))
    by_name = {phase["name"]: phase for phase in result["phases"]}
    assert status == "BLOCKED"
    assert by_name["08-dataset-adapter-evaluator"]["status"] == "PASSED"
    assert by_name["08b-dataset-evidence-audit"]["status"] == "PASSED"
    assert by_name["08c-licensed-visual-review"]["reason_code"] == (
        "LICENSED_VISUAL_AUDIT_REVIEW_REQUIRED"
    )

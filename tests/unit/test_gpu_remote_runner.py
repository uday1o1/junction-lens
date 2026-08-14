"""Tests for durable remote GPU qualification state and public handoff behavior."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from scripts.gpu.remote_runner import PhaseResult, Runner
from scripts.gpu.source_bundle import create_bundle, create_remote_config


def _runner(tmp_path: Path) -> Runner:
    root = Path(__file__).resolve().parents[2]
    return Runner(
        root,
        tmp_path / "results",
        {
            "profile": "runtime-cuda",
            "source_commit": "a" * 40,
            "source_content_sha256": "b" * 64,
            "remote_data_root": None,
            "gpu_uuid": None,
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

"""Unit tests for bounded capability probes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from junctionlens.doctor import probes
from junctionlens.doctor.models import CapabilityRequirement, CapabilityState


def test_missing_command_is_distinct_from_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing executable emits an absence reason code."""
    monkeypatch.setattr(probes.shutil, "which", lambda _name: None)
    evidence = probes.probe_command(
        "sample",
        "sample",
        ["--version"],
        CapabilityRequirement.REQUIRED_LOCAL,
        "1.2.3",
    )
    assert evidence.state is CapabilityState.ABSENT
    assert evidence.reason_code == "SAMPLE_NOT_FOUND"


def test_nonzero_command_is_a_probe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present but broken command does not masquerade as absence."""
    monkeypatch.setattr(probes.shutil, "which", lambda _name: "/bin/sample")
    monkeypatch.setattr(
        probes,
        "_run",
        lambda _command: subprocess.CompletedProcess([], 7, stdout="", stderr="broken"),
    )
    evidence = probes.probe_command(
        "sample",
        "sample",
        ["--version"],
        CapabilityRequirement.REQUIRED_LOCAL,
    )
    assert evidence.state is CapabilityState.ERROR
    assert evidence.reason_code == "SAMPLE_PROBE_FAILED"


def test_version_mismatch_is_incompatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured pins are compared with observed output."""
    monkeypatch.setattr(probes.shutil, "which", lambda _name: "/bin/sample")
    monkeypatch.setattr(
        probes,
        "_run",
        lambda _command: subprocess.CompletedProcess([], 0, stdout="sample 2.0.0", stderr=""),
    )
    evidence = probes.probe_command(
        "sample",
        "sample",
        ["--version"],
        CapabilityRequirement.REQUIRED_LOCAL,
        "1.2.3",
    )
    assert evidence.state is CapabilityState.INCOMPATIBLE
    assert evidence.observed_version == "2.0.0"


def test_cpp_truth_probe_rejects_malformed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed native probe output is a typed error."""
    probe = tmp_path / "build/cpu/junctionlens-toolchain-probe"
    probe.parent.mkdir(parents=True)
    probe.write_text("not executed", encoding="utf-8")
    monkeypatch.setattr(
        probes,
        "_run",
        lambda _command: subprocess.CompletedProcess([], 0, stdout="not json", stderr=""),
    )
    evidence = probes.probe_cpp_truth_file(tmp_path)
    assert evidence.reason_code == "CPP_TRUTH_PROBE_INVALID_JSON"


def test_dataset_requires_terms_after_root_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured root without a receipt remains inaccessible."""
    monkeypatch.setenv("OPENLANE_V2_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    evidence = probes.probe_dataset()
    assert evidence.state is CapabilityState.INACCESSIBLE
    assert evidence.reason_code == "DATA_LICENSE_ACK_REQUIRED"

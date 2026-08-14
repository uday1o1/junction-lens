"""Public CLI coverage for normalized OpenLane data paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from junctionlens.cli.main import app


def test_data_audit_runs_real_lazy_adapter_through_public_cli(
    openlane_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented audit command exercises real metadata normalization end to end."""
    monkeypatch.setattr(
        "junctionlens.cli.data._registered_root",
        lambda *_args, **_kwargs: openlane_root,
    )
    result = CliRunner().invoke(
        app,
        ["data", "audit", "--root", str(openlane_root), "--profile", "sample"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["frame_count"] == 2
    assert payload["capacity_gate_accepted"] is True
    assert payload["identity"]["lane_segment"]["temporal_kpi_state"] == ("ENABLED_SOURCE_IDENTITY")


def test_verify_adapter_public_cli_emits_only_bounded_evidence(
    openlane_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The official parity command returns evidence rather than restricted annotations."""
    report = {
        "schema_version": "junctionlens.openlane-adapter-parity-report.v1",
        "state": "ACCEPTED",
        "devkit_version": "2.1.0",
        "frame_count": 3,
        "maximum_absolute_numeric_error": 0.0,
        "tolerance": 1e-9,
        "official_projection_set_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        "junctionlens.cli.data._registered_root",
        lambda *_args, **_kwargs: openlane_root,
    )
    monkeypatch.setattr(
        "junctionlens.cli.data.verify_official_parity",
        lambda *_args, **_kwargs: report,
    )
    result = CliRunner().invoke(
        app,
        ["data", "verify-adapter", "--root", str(openlane_root)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == report

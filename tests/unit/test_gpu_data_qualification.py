"""Unit tests for the atomic licensed-data remote workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.gpu import qualify_data

HASH = "a" * 64


def _receipts() -> dict[str, dict[str, Any]]:
    return {
        "register": {
            "dataset_id": "openlane-v2-v2.1",
            "profile": "full",
            "root": "<registered-dataset-root>",
        },
        "audit": {"capacity_gate_accepted": True, "frame_count": 1_000},
        "verify-adapter": {
            "state": "ACCEPTED",
            "frame_count": 3,
            "maximum_absolute_numeric_error": 0.0,
            "tolerance": 1e-9,
        },
        "manifest": {"state": "ACCEPTED", "artifact_manifest_sha256": HASH},
        "split": {
            "state": "ACCEPTED",
            "segment_count": 700,
            "overlap_count": 0,
            "split_manifest_sha256": HASH,
        },
        "audit-splits": {"state": "ACCEPTED", "segment_count": 700, "overlap_count": 0},
        "visual-audit": {
            "state": "PENDING_HUMAN_INSPECTION",
            "selected_frame_count": 12,
            "range_gate_accepted": True,
            "bundle_manifest_sha256": HASH,
        },
    }


def test_qualify_runs_every_public_data_step_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    dataset = tmp_path / "dataset"
    output = tmp_path / "result"
    project.mkdir()
    dataset.mkdir()
    expected = _receipts()
    observed: list[str] = []

    def fake_run(
        label: str,
        arguments: list[str],
        *,
        staging_root: Path,
        **_kwargs: object,
    ) -> dict[str, Any]:
        observed.append(label)
        if label == "visual-audit":
            visual = Path(arguments[arguments.index("--output") + 1])
            visual.mkdir()
            manifest = visual / "manifest.json"
            manifest.write_text('{"policy_id":"openlane-v2-v2.1-audit-v1"}\n', encoding="utf-8")
            expected[label]["bundle_manifest_sha256"] = qualify_data._sha256_file(manifest)
        (staging_root / "receipts").mkdir(exist_ok=True)
        (staging_root / "receipts" / f"{label}.json").write_text(
            json.dumps(expected[label]), encoding="utf-8"
        )
        if label == "split":
            export = Path(arguments[arguments.index("--export") + 1])
            export.write_text("{}\n", encoding="utf-8")
        return expected[label]

    monkeypatch.setattr(qualify_data, "_run_cli", fake_run)
    result = qualify_data.qualify(project, dataset, output)

    assert observed == [
        "register",
        "audit",
        "verify-adapter",
        "manifest",
        "split",
        "audit-splits",
        "visual-audit",
    ]
    assert result["mechanical_state"] == "ACCEPTED"
    assert result["state"] == "PENDING_HUMAN_INSPECTION"
    assert (output / "qualification.json").is_file()
    assert not list(tmp_path.glob(".result-*"))


def test_qualify_removes_staging_after_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    dataset = tmp_path / "dataset"
    output = tmp_path / "result"
    project.mkdir()
    dataset.mkdir()
    expected = _receipts()
    expected["split"]["segment_count"] = 699

    def fake_run(
        label: str,
        _arguments: list[str],
        *,
        staging_root: Path,
        **_kwargs: object,
    ) -> dict[str, Any]:
        (staging_root / "receipts").mkdir(exist_ok=True)
        (staging_root / "receipts" / f"{label}.json").write_text(
            json.dumps(expected[label]), encoding="utf-8"
        )
        return expected[label]

    monkeypatch.setattr(qualify_data, "_run_cli", fake_run)
    with pytest.raises(qualify_data.DataQualificationError, match="700-segment"):
        qualify_data.qualify(project, dataset, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".result-*"))

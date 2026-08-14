"""Public CLI coverage for charter freeze and persisted gate decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.unit.test_gate_decision import _evidence, _write_charter
from typer.testing import CliRunner

import junctionlens.gate.charter as charter_module
from junctionlens.cli.main import app
from junctionlens.gate.charter import load_charter_draft
from junctionlens.registry import ContentAddressedStore
from junctionlens.registry.store import canonical_json_bytes

PROJECT_ROOT = Path(__file__).parents[2]
HASH = "a" * 64


def _baseline_evidence() -> dict[str, Any]:
    draft = load_charter_draft(PROJECT_ROOT / "configs/gates/acceptance-v1.draft.yaml")
    return {
        "schema_version": "junctionlens.baseline-freeze-evidence.v1",
        "experiment_id": "E0-independent",
        "source_partitions": ["model_training", "model_selection"],
        "internal_holdout_access_count": 0,
        "baseline_seed_checkpoint_sha256": {
            "20260813": HASH,
            "20260814": HASH,
            "20260815": HASH,
        },
        "baseline_variability": {
            cell.id: [0.4, 0.41, 0.39] for cell in draft.cells if cell.stage != "performance"
        },
        "m0_hardware_baseline_manifest_sha256": "b" * 64,
        "power_simulation": {
            "artifact_sha256": "c" * 64,
            "candidate_results_used": False,
            "internal_holdout_used": False,
            "source_partitions": ["model_training", "model_selection"],
        },
        "product_priorities_sha256": "d" * 64,
        "proposed_margins": {cell.id: cell.margin for cell in draft.cells},
    }


def test_gate_freeze_cli_uses_immutable_baseline_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(
        tmp_path / "artifacts", PROJECT_ROOT / "schemas/artifact-manifest-v1.schema.json"
    )
    receipt = store.put_bytes(
        canonical_json_bytes(_baseline_evidence()),
        kind="baseline_freeze_evidence",
        media_type="application/json",
        license_id="Apache-2.0",
        metadata={},
    )
    monkeypatch.setattr(charter_module, "_source_commit", lambda _: "e" * 40)
    output = tmp_path / "acceptance-v1.yaml"

    result = CliRunner().invoke(
        app,
        [
            "gate",
            "freeze",
            "--draft",
            str(PROJECT_ROOT / "configs/gates/acceptance-v1.draft.yaml"),
            "--baseline-run",
            f"artifacts://runs/{receipt.manifest_sha256}",
            "--output",
            str(output),
            "--signer",
            "release-owner",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--metrics",
            str(PROJECT_ROOT / "configs/metrics/v1.yaml"),
            "--slices",
            str(PROJECT_ROOT / "configs/slices/v1.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["state"] == "FROZEN"
    assert output.stat().st_mode & 0o222 == 0


def test_gate_decide_cli_persists_pass_and_refuses_replacement(tmp_path: Path) -> None:
    charter = tmp_path / "acceptance-v1.yaml"
    charter_hash = _write_charter(charter)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence(charter_hash)), encoding="utf-8")
    output = tmp_path / "decision.json"
    arguments = [
        "gate",
        "decide",
        "--charter",
        str(charter),
        "--evidence",
        str(evidence),
        "--output",
        str(output),
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "PASS"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    assert output.stat().st_mode & 0o222 == 0
    assert repeated.exit_code == 2
    assert "already exists" in repeated.stderr

"""Public fault command workflow coverage."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from junctionlens.cli.main import app
from junctionlens.faults.service import put_prediction_bundle
from junctionlens.faults.synthetic import build_synthetic_fault_bundle
from junctionlens.registry.service import EvidenceRegistry

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "schemas/artifact-manifest-v1.schema.json"


def test_fault_cli_derives_and_detects_without_modifying_parent(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    parent_hash = put_prediction_bundle(registry, build_synthetic_fault_bundle())
    parent_manifest_before = registry.store.object_path(parent_hash).read_bytes()

    result = CliRunner().invoke(
        app,
        [
            "fault",
            "--input",
            parent_hash,
            "--kind",
            "swap-control-edges",
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert receipt["state"] == "DETECTED"
    assert receipt["primary_reason_code"] == "FAULT_CONTROL_ASSIGNMENT_CHANGED"
    assert registry.store.object_path(parent_hash).read_bytes() == parent_manifest_before

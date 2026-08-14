"""Tests for one-way acceptance-charter freezing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import junctionlens.gate.charter as charter_module
from junctionlens.gate.charter import CharterError, freeze_charter, load_charter_draft
from junctionlens.registry import ContentAddressedStore
from junctionlens.registry.store import canonical_json_bytes

PROJECT_ROOT = Path(__file__).parents[2]
HASH = "a" * 64


def _freeze_evidence() -> dict[str, Any]:
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


def _store_evidence(tmp_path: Path, evidence: dict[str, Any]) -> tuple[Path, str]:
    root = tmp_path / "artifacts"
    store = ContentAddressedStore(root, PROJECT_ROOT / "schemas/artifact-manifest-v1.schema.json")
    receipt = store.put_bytes(
        canonical_json_bytes(evidence),
        kind="baseline_freeze_evidence",
        media_type="application/json",
        license_id="Apache-2.0",
        metadata={"experiment_id": "E0-independent"},
    )
    return root, receipt.manifest_sha256


def _freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: dict[str, Any]) -> Path:
    artifact_root, manifest_hash = _store_evidence(tmp_path, evidence)
    output = tmp_path / "acceptance-v1.yaml"
    monkeypatch.setattr(charter_module, "_source_commit", lambda _: "e" * 40)
    freeze_charter(
        PROJECT_ROOT / "configs/gates/acceptance-v1.draft.yaml",
        f"artifacts://runs/{manifest_hash}",
        output,
        artifact_root=artifact_root,
        project_root=PROJECT_ROOT,
        signer="release-owner",
        metrics_path=PROJECT_ROOT / "configs/metrics/v1.yaml",
        slices_path=PROJECT_ROOT / "configs/slices/v1.yaml",
    )
    return output


def test_freeze_binds_all_preholdout_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _freeze(tmp_path, monkeypatch, _freeze_evidence())
    frozen = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert frozen["frozen"] is True
    assert frozen["source_commit"] == "e" * 40
    assert frozen["freeze_evidence"]["m0_hardware_baseline_manifest_sha256"] == "b" * 64
    assert output.stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("case", ["holdout", "candidate", "loosened_margin"])
def test_freeze_rejects_postcandidate_policy_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    evidence = _freeze_evidence()
    if case == "holdout":
        evidence["internal_holdout_access_count"] = 1
    elif case == "candidate":
        evidence["power_simulation"]["candidate_results_used"] = True
    else:
        first = next(iter(evidence["proposed_margins"]))
        evidence["proposed_margins"][first] += 0.001
    artifact_root, manifest_hash = _store_evidence(tmp_path, evidence)
    monkeypatch.setattr(charter_module, "_source_commit", lambda _: "e" * 40)

    with pytest.raises(CharterError):
        freeze_charter(
            PROJECT_ROOT / "configs/gates/acceptance-v1.draft.yaml",
            f"artifacts://runs/{manifest_hash}",
            tmp_path / "acceptance-v1.yaml",
            artifact_root=artifact_root,
            project_root=PROJECT_ROOT,
            signer="release-owner",
            metrics_path=PROJECT_ROOT / "configs/metrics/v1.yaml",
            slices_path=PROJECT_ROOT / "configs/slices/v1.yaml",
        )


def test_frozen_charter_detects_content_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _freeze(tmp_path, monkeypatch, _freeze_evidence())
    frozen = yaml.safe_load(output.read_text(encoding="utf-8"))
    frozen["cells"][0]["margin"] += 0.001
    output.chmod(0o644)
    output.write_text(json.dumps(frozen), encoding="utf-8")

    with pytest.raises(CharterError, match="self-hash"):
        charter_module.load_frozen_charter(output)


def test_draft_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    draft = tmp_path / "duplicate.yaml"
    draft.write_text(
        "schema_version: junctionlens.acceptance-charter-draft.v1\n"
        "schema_version: junctionlens.acceptance-charter-draft.v1\n",
        encoding="utf-8",
    )

    with pytest.raises(CharterError, match="duplicate YAML key"):
        load_charter_draft(draft)


def test_source_bundle_commit_override_requires_gitless_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUNCTIONLENS_SOURCE_COMMIT", "f" * 40)

    assert charter_module._source_commit(tmp_path) == "f" * 40

    (tmp_path / ".git").mkdir()
    with pytest.raises(CharterError, match="clean source checkout|cannot resolve"):
        charter_module._source_commit(tmp_path)

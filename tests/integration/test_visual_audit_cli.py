"""Public private-overlay and statistical-audit workflow test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from junctionlens.cli.main import app


def _audit_policy_for_fixture(tmp_path: Path) -> Path:
    payload = yaml.safe_load(
        Path("configs/data/openlane-v2-v2.1.audit-v1.yaml").read_text(encoding="utf-8")
    )
    payload["frozen_frames"] = [
        {"split_id": "train", "segment_id": "segment-1", "timestamp": "100"}
    ]
    config_root = tmp_path / "configs"
    slice_path = config_root / "slices/v1.yaml"
    slice_path.parent.mkdir(parents=True)
    slice_path.write_bytes(Path("configs/slices/v1.yaml").read_bytes())
    path = config_root / "data/audit-policy.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_visual_audit_public_cli_writes_frozen_private_bundle(
    openlane_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public command writes inspectable overlays plus aggregate-only JSON evidence."""
    monkeypatch.setattr(
        "junctionlens.cli.data._registered_root",
        lambda *_args, **_kwargs: openlane_root,
    )
    (openlane_root / "data_dict_subset_A.json").write_bytes(
        (openlane_root / "data_dict_example.json").read_bytes()
    )
    output = tmp_path / "audit-bundle"
    result = CliRunner().invoke(
        app,
        [
            "data",
            "visual-audit",
            "--root",
            str(openlane_root),
            "--profile",
            "full",
            "--policy",
            str(_audit_policy_for_fixture(tmp_path)),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert receipt["state"] == "PENDING_HUMAN_INSPECTION"
    assert receipt["range_gate_accepted"] is True
    assert receipt["selected_frame_count"] == 1
    assert (output / "index.html").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "train/segment-1/100/bev.svg").is_file()
    assert (output / "train/segment-1/100/camera-front_center.png").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert "source_domain" in summary["slice_support_preview"]

    monkeypatch.chdir(tmp_path)
    signoff = CliRunner().invoke(
        app,
        [
            "data",
            "signoff-visual-audit",
            "--bundle",
            str(output),
            "--accept-camera-projection-alignment",
            "--accept-bev-geometry-alignment",
            "--accept-label-identity-and-topology",
            "--confirm-private-data-handling",
        ],
    )
    assert signoff.exit_code == 0, signoff.output
    signoff_receipt = json.loads(signoff.stdout)
    assert signoff_receipt["state"] == "ACCEPTED"
    assert signoff_receipt["bundle_manifest_sha256"] == receipt["bundle_manifest_sha256"]
    private_receipt = tmp_path / ".junctionlens/data-audit-signoffs/openlane-v2-v2.1.json"
    assert private_receipt.stat().st_mode & 0o777 == 0o600


def test_visual_signoff_rejects_tampered_bundle(
    openlane_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "junctionlens.cli.data._registered_root",
        lambda *_args, **_kwargs: openlane_root,
    )
    (openlane_root / "data_dict_subset_A.json").write_bytes(
        (openlane_root / "data_dict_example.json").read_bytes()
    )
    output = tmp_path / "audit-bundle"
    generated = CliRunner().invoke(
        app,
        [
            "data",
            "visual-audit",
            "--root",
            str(openlane_root),
            "--profile",
            "full",
            "--policy",
            str(_audit_policy_for_fixture(tmp_path)),
            "--output",
            str(output),
        ],
    )
    assert generated.exit_code == 0, generated.output
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    signoff = CliRunner().invoke(
        app,
        [
            "data",
            "signoff-visual-audit",
            "--bundle",
            str(output),
            "--accept-camera-projection-alignment",
            "--accept-bev-geometry-alignment",
            "--accept-label-identity-and-topology",
            "--confirm-private-data-handling",
        ],
    )

    assert signoff.exit_code != 0
    assert "failed size or digest" in signoff.output

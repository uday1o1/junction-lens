"""Public manifest, split-freeze, and split-audit CLI tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from junctionlens.cli.main import app
from junctionlens.registry import ContentAddressedStore
from junctionlens.registry.store import canonical_json_bytes

_SCHEMA = Path("schemas/artifact-manifest-v1.schema.json")


def test_frame_manifest_public_cli_streams_registered_fixture(
    openlane_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real public command streams an adapter into the immutable registry."""
    registration = {
        "dataset_id": "openlane-v2-v2.1",
        "profile": "sample",
        "root": str(openlane_root),
        "archive_sha256": "1" * 64,
        "license_receipt_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
    }
    monkeypatch.setattr("junctionlens.cli.data.load_registration", lambda *_args: registration)
    artifact_root = tmp_path / "artifacts"
    result = CliRunner().invoke(
        app,
        [
            "data",
            "manifest",
            "--profile",
            "sample",
            "--root",
            str(openlane_root),
            "--artifact-root",
            str(artifact_root),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["state"] == "ACCEPTED"
    assert payload["frame_count"] == 2
    assert payload["split_segment_counts"] == {"train": 1}
    store = ContentAddressedStore(artifact_root, _SCHEMA)
    manifest = store.read_manifest(payload["artifact_manifest_sha256"])
    assert manifest["kind"] == "frame_manifest"


def test_split_freeze_and_independent_audit_public_cli(tmp_path: Path) -> None:
    """The public workflow freezes exact V1 counts and audits its immutable export."""
    artifact_root = tmp_path / "artifacts"
    store = ContentAddressedStore(artifact_root, _SCHEMA)
    catalog = [
        {
            "split_id": "train",
            "segment_id": f"segment-{index:04d}",
            "source_domain": "argoverse2" if index < 400 else "nuscenes",
            "source_segment_id_sha256": f"{index:064x}",
        }
        for index in range(700)
    ]
    frame_records = b"".join(
        canonical_json_bytes(
            {
                "split_id": "train",
                "segment_id": item["segment_id"],
                "timestamp_ns": index,
                "source_domain": item["source_domain"],
                "source_segment_id_sha256": item["source_segment_id_sha256"],
                "frame_metadata_sha256": f"{index + 1000:064x}",
                "calibration_sha256": f"{index + 2000:064x}",
                "valid_camera_slots": ["FRONT_CENTER"],
                "pose_valid": True,
                "annotations_valid": True,
            }
        )
        + b"\n"
        for index, item in enumerate(catalog)
    )
    frame_receipt = store.put_bytes(
        frame_records,
        kind="frame_manifest",
        media_type="application/x-ndjson",
        license_id="CC-BY-NC-SA-4.0",
        metadata={
            "schema_version": "junctionlens.frame-manifest.v1",
            "dataset_id": "openlane-v2-v2.1",
            "dataset_version": "2.1",
            "profile": "full",
            "ordering": "split_id,segment_id,numeric_timestamp",
            "record_media_type": "application/x-ndjson",
            "record_schema": "junctionlens.frame-content-record.v1",
            "frame_count": 700,
            "segment_count": 700,
            "split_segment_counts": {"train": 700},
            "segment_catalog": catalog,
            "segment_catalog_sha256": hashlib.sha256(canonical_json_bytes(catalog)).hexdigest(),
            "source_registration": {"manifest_sha256": "3" * 64},
        },
    )
    output = tmp_path / "openlane-v2-v2.1.split-v1.json"
    result = CliRunner().invoke(
        app,
        [
            "data",
            "freeze-splits",
            "--frame-manifest-sha256",
            frame_receipt.manifest_sha256,
            "--artifact-root",
            str(artifact_root),
            "--export",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["state"] == "ACCEPTED"
    assert payload["overlap_count"] == 0
    assert payload["partition_counts"] == {
        "calibration": 70,
        "internal_holdout": 200,
        "model_selection": 80,
        "model_training": 350,
    }
    audit = CliRunner().invoke(
        app,
        ["data", "audit-splits", "--manifest", str(output)],
    )
    assert audit.exit_code == 0, audit.output
    assert json.loads(audit.stdout)["state"] == "ACCEPTED"

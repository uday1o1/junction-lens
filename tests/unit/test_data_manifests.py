"""Streaming content-manifest and frozen split-policy tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from junctionlens.data.manifests import (
    ManifestError,
    SegmentRecord,
    audit_split_manifest,
    freeze_split_manifest,
    load_split_policy,
    write_frame_records,
    write_immutable_split_manifest,
)
from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.registry.store import canonical_json_bytes

_CONFIG = Path("configs/data/openlane-v2-v2.1.adapter.yaml")
_POLICY_PATH = Path("configs/data/openlane-v2-v2.1.split-v1.yaml")
_HASHES = {
    "source_frame_manifest_sha256": "a" * 64,
    "source_frame_records_sha256": "b" * 64,
    "source_dataset_manifest_sha256": "c" * 64,
}


def _records() -> list[SegmentRecord]:
    return [
        SegmentRecord(f"segment-{index:04d}", "argoverse2" if index < 400 else "nuscenes")
        for index in range(700)
    ]


def _json_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def test_frame_content_records_stream_only_bounded_provenance(
    openlane_root: Path, tmp_path: Path
) -> None:
    """Frame manifests retain hashes and source identity without annotations or image paths."""
    output = tmp_path / "frames.ndjson"
    metadata = write_frame_records(
        OpenLaneAdapter(openlane_root, _CONFIG),
        "sample",
        output,
        {
            "dataset_id": "openlane-v2-v2.1",
            "profile": "sample",
            "archive_sha256": "1" * 64,
            "license_receipt_sha256": "2" * 64,
            "manifest_sha256": "3" * 64,
        },
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["timestamp_ns"] for record in records] == [100, 200]
    assert metadata["frame_count"] == 2
    assert metadata["segment_count"] == 1
    assert metadata["split_segment_counts"] == {"train": 1}
    assert set(records[0]) == {
        "annotations_valid",
        "calibration_sha256",
        "frame_metadata_sha256",
        "pose_valid",
        "segment_id",
        "source_domain",
        "source_segment_id_sha256",
        "split_id",
        "timestamp_ns",
        "valid_camera_slots",
    }
    assert "image" not in output.read_text(encoding="utf-8")


def test_v1_split_is_exact_disjoint_and_byte_stable_across_input_order() -> None:
    """The frozen 350/80/70/200 allocation is independent of source iteration order."""
    policy = load_split_policy(_POLICY_PATH)
    records = _records()
    first = freeze_split_manifest(records, policy, **_HASHES)
    records.reverse()
    second = freeze_split_manifest(records, policy, **_HASHES)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    audit = audit_split_manifest(first, policy)
    assert audit.state == "ACCEPTED"
    assert audit.overlap_count == 0
    assert audit.partition_counts == {
        "model_training": 350,
        "model_selection": 80,
        "calibration": 70,
        "internal_holdout": 200,
    }
    partitions = first["partitions"]
    all_segments = [
        segment for partition in partitions.values() for segment in partition["segments"]
    ]
    assert len(all_segments) == len(set(all_segments)) == 700
    for partition in partitions.values():
        assert sum(partition["source_domain_counts"].values()) == partition["segment_count"]
    assert {
        partition: value["source_domain_counts"] for partition, value in partitions.items()
    } == {
        "model_training": {"argoverse2": 200, "nuscenes": 150},
        "model_selection": {"argoverse2": 46, "nuscenes": 34},
        "calibration": {"argoverse2": 40, "nuscenes": 30},
        "internal_holdout": {"argoverse2": 114, "nuscenes": 86},
    }


def test_seeded_cross_partition_leak_fails_for_leakage_with_nearby_control() -> None:
    """A hash-consistent seeded overlap fails for leakage while the untouched control passes."""
    policy = load_split_policy(_POLICY_PATH)
    original = freeze_split_manifest(_records(), policy, **_HASHES)
    assert audit_split_manifest(original, policy).overlap_count == 0
    corrupted = copy.deepcopy(original)
    training = corrupted["partitions"]["model_training"]
    selection = corrupted["partitions"]["model_selection"]
    selected_domain = next(
        item["source_domain"]
        for item in corrupted["segment_catalog"]
        if item["segment_id"] == selection["segments"][0]
    )
    leaked = next(
        segment
        for segment in training["segments"]
        if next(
            item["source_domain"]
            for item in corrupted["segment_catalog"]
            if item["segment_id"] == segment
        )
        == selected_domain
    )
    selection["segments"][0] = leaked
    selection["segments_sha256"] = _json_hash(selection["segments"])
    with pytest.raises(ManifestError, match="segment leakage detected"):
        audit_split_manifest(corrupted, policy)


def test_immutable_split_export_rejects_rewrite(tmp_path: Path) -> None:
    """A frozen split path permits identical reruns and rejects changed content."""
    policy = load_split_policy(_POLICY_PATH)
    manifest = freeze_split_manifest(_records(), policy, **_HASHES)
    output = tmp_path / "split-v1.json"
    first_hash = write_immutable_split_manifest(output, manifest)
    assert write_immutable_split_manifest(output, manifest) == first_hash
    changed = copy.deepcopy(manifest)
    changed["source_frame_manifest_sha256"] = "d" * 64
    with pytest.raises(ManifestError, match="different content"):
        write_immutable_split_manifest(output, changed)


def test_policy_rejects_result_derived_assignment_field(tmp_path: Path) -> None:
    """A seeded attempt to stratify from label statistics fails before allocation."""
    payload = _POLICY_PATH.read_text(encoding="utf-8")
    payload = payload.replace(
        "stratification_fields:\n  - source_domain",
        "stratification_fields:\n  - source_domain\n  - label_statistics",
    )
    path = tmp_path / "policy.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ManifestError, match="may use only"):
        load_split_policy(path)

"""Public exact-frame comparison and immutable report-data workflow tests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from junctionlens.cli.main import app
from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import canonical_json_bytes

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "schemas/artifact-manifest-v1.schema.json"
METRICS = ROOT / "configs/metrics/v1.yaml"
SLICES = ROOT / "configs/slices/v1.yaml"
HASH = "a" * 64
HARDWARE_HASH = "b" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_charter(path: Path) -> None:
    body: dict[str, Any] = {
        "schema_version": "junctionlens.acceptance-charter.v1",
        "charter_id": "junctionlens-acceptance-v1",
        "frozen": True,
        "frozen_at": "2026-08-13T00:00:00+00:00",
        "signer": "release-owner",
        "source_commit": "c" * 40,
        "draft_sha256": HASH,
        "baseline_run_manifest_sha256": HASH,
        "baseline_evidence_payload_sha256": HASH,
        "metric_registry_sha256": _sha256(METRICS),
        "slice_registry_sha256": _sha256(SLICES),
        "family_alpha": 0.05,
        "bootstrap": {
            "algorithm": "paired-segment-cluster-bootstrap-v1",
            "replicates": 10000,
            "seed": 20260813,
            "minimum_finite_replicates": 9900,
        },
        "runtime_bootstrap": {
            "algorithm": "paired-trial-block-bootstrap-v1",
            "replicates": 10000,
            "seed": 20260813,
            "minimum_valid_pairs": 8,
            "balanced_order_schedule": ["AB", "BA"] * 5,
        },
        "support": {
            "overall_paired_segments": 200,
            "gating_slice_paired_segments": 30,
            "lane_control_ground_truth_edges": 500,
            "temporal_adjacent_frame_transitions": 500,
            "temporal_segments": 30,
        },
        "absolute_runtime": {
            "throughput_per_second_minimum": 10.0,
            "p95_latency_ms_maximum": 100.0,
            "p99_latency_ms_maximum": 125.0,
            "peak_device_memory_bytes_maximum": 6442450944,
            "long_run_frames": 10000,
            "unexpected_cpu_provider_nodes_maximum": 0,
        },
        "cells": [
            {
                "id": "overall.score",
                "metric": "score",
                "slice": "overall",
                "direction": "higher_is_better",
                "margin": 0.005,
                "support": "overall",
                "estimator": "ratio",
                "stage": "accuracy",
            },
            {
                "id": "overall.error",
                "metric": "error",
                "slice": "overall",
                "direction": "lower_is_better",
                "margin": 0.005,
                "support": "overall",
                "estimator": "ratio",
                "stage": "accuracy",
            },
            {
                "id": "argoverse2.score",
                "metric": "score",
                "slice": "source_domain:argoverse2",
                "direction": "higher_is_better",
                "margin": 0.005,
                "support": "slice",
                "estimator": "ratio",
                "stage": "accuracy",
            },
        ],
        "primary_hypotheses": [],
        "freeze_evidence": {
            "baseline_seed_checkpoint_sha256": {"20260813": HASH},
            "m0_hardware_baseline_manifest_sha256": HARDWARE_HASH,
            "power_simulation_artifact_sha256": HASH,
            "product_priorities_sha256": HASH,
        },
    }
    body["charter_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def _arm(score: float, error: float) -> dict[str, Any]:
    frames = []
    for index in range(200):
        frames.append(
            {
                "frame_token": f"frame-{index:03d}",
                "segment_id": f"segment-{index:03d}",
                "timestamp_ns": index,
                "slice_values": {"source_domain": "argoverse2" if index % 2 == 0 else "nuscenes"},
                "metrics": {
                    "score": {"numerator": score, "denominator": 1.0},
                    "error": {"numerator": error, "denominator": 1.0},
                },
            }
        )
    return {
        "schema_version": "junctionlens.comparison-arm.v1",
        "arm_id": f"arm-{score}-{error}",
        "evaluator_image_digest": HASH,
        "data_manifest_sha256": HASH,
        "split_manifest_sha256": HASH,
        "preprocessing_sha256": HASH,
        "postprocessing_sha256": HASH,
        "metric_registry_sha256": _sha256(METRICS),
        "slice_registry_sha256": _sha256(SLICES),
        "integrity": {
            "artifact_integrity": True,
            "schema_major_compatible": True,
            "identifiers_valid": True,
            "coordinate_metadata_valid": True,
            "required_values_finite": True,
            "calibrator_valid": True,
            "evaluator_compatible": True,
            "provenance_complete": True,
            "leakage_free": True,
            "partial_inference_approved": True,
            "provider_fallback_free": True,
            "calibration_ranks_frozen": True,
            "training_holdout_access_count": 0,
        },
        "frames": frames,
        "runtime": {
            "hardware_baseline_manifest_sha256": HARDWARE_HASH,
            "gpu_provider_active": True,
            "throughput_per_second": 12.0,
            "p95_latency_ms": 90.0,
            "p99_latency_ms": 110.0,
            "peak_device_memory_bytes": 5 * 1024**3,
            "long_run_frames": 10000,
            "unbounded_memory_growth": False,
            "unexpected_cpu_provider_nodes": 0,
            "warmup_frames_per_block": 200,
            "measured_frames_per_block": 2000,
            "trial_block_order": ["AB", "BA"] * 5,
            "environment_valid": True,
            "metrics": {},
        },
    }


def _put_arm(registry: EvidenceRegistry, body: dict[str, Any], *, with_parent: bool = True) -> str:
    parents: tuple[str, ...] = ()
    if with_parent:
        parent = registry.put_bytes(
            b'{"schema_version":"junctionlens.synthetic-source.v1"}\n',
            kind="run_configuration",
            media_type="application/json",
            license_id="Apache-2.0",
            metadata={"fixture": "comparison-source"},
        )
        parents = (parent.manifest_sha256,)
    receipt = registry.put_bytes(
        canonical_json_bytes(body) + b"\n",
        kind="prediction_bundle",
        media_type="application/vnd.junctionlens.comparison-arm+json",
        license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
        metadata={"arm_id": body["arm_id"]},
        parents=parents,
    )
    return receipt.manifest_sha256


def _payload(registry: EvidenceRegistry, manifest_sha256: str) -> bytes:
    manifest = registry.store.read_manifest(manifest_sha256)
    return registry.store.object_path(manifest["payload"]["sha256"]).read_bytes()


def _arguments(
    artifact_root: Path, charter: Path, baseline_hash: str, candidate_hash: str
) -> list[str]:
    return [
        "compare",
        "--baseline",
        baseline_hash,
        "--candidate",
        candidate_hash,
        "--charter",
        str(charter),
        "--artifact-root",
        str(artifact_root),
        "--schema",
        str(SCHEMA),
        "--metrics",
        str(METRICS),
        "--slices",
        str(SLICES),
    ]


def test_compare_cli_persists_exact_frame_decision_and_report_data(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    baseline_hash = _put_arm(registry, _arm(0.5, 0.5))
    candidate_hash = _put_arm(registry, _arm(0.51, 0.49))
    charter = tmp_path / "acceptance-v1.yaml"
    _write_charter(charter)
    arguments = _arguments(artifact_root, charter, baseline_hash, candidate_hash)

    first = CliRunner().invoke(app, arguments)
    second = CliRunner().invoke(app, arguments)

    assert first.exit_code == second.exit_code == 0, first.output
    assert first.stdout == second.stdout
    receipt = json.loads(first.stdout)
    assert receipt["status"] == "PASS"
    decision = json.loads(_payload(registry, receipt["decision_manifest_sha256"]))
    report = json.loads(_payload(registry, receipt["report_data_manifest_sha256"]))
    assert {cell["status"] for cell in decision["cells"]} == {"PASS"}
    assert {cell["interval"]["adjusted_two_sided_alpha"] for cell in decision["cells"]} == {
        0.05 / 3
    }
    assert report["decision_manifest_sha256"] == receipt["decision_manifest_sha256"]
    assert report["filtering_changes_release_status"] is False
    slice_manifest = registry.store.read_manifest(receipt["slice_table_manifest_sha256"])
    slice_table = pq.read_table(registry.store.object_path(slice_manifest["payload"]["sha256"]))
    assert slice_table.num_rows == 200
    assert slice_table.column("frame_token").to_pylist() == sorted(
        slice_table.column("frame_token").to_pylist()
    )


def test_compare_cli_persists_frame_mismatch_as_integrity_failure(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    baseline_hash = _put_arm(registry, _arm(0.5, 0.5))
    candidate = deepcopy(_arm(0.51, 0.49))
    candidate["frames"].pop()
    candidate_hash = _put_arm(registry, candidate)
    charter = tmp_path / "acceptance-v1.yaml"
    _write_charter(charter)

    result = CliRunner().invoke(
        app, _arguments(artifact_root, charter, baseline_hash, candidate_hash)
    )

    assert result.exit_code == 3, result.output
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "FAIL_INTEGRITY"
    decision = json.loads(_payload(registry, receipt["decision_manifest_sha256"]))
    assert "GATE_INTEGRITY_FRAME_SET_MISMATCH" in decision["integrity_reason_codes"]


def test_compare_cli_accepts_exact_noninferiority_margin_in_both_directions(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    baseline_hash = _put_arm(registry, _arm(0.5, 0.5))
    candidate_hash = _put_arm(registry, _arm(0.495, 0.505))
    charter = tmp_path / "acceptance-v1.yaml"
    _write_charter(charter)

    result = CliRunner().invoke(
        app, _arguments(artifact_root, charter, baseline_hash, candidate_hash)
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "PASS"


def test_compare_cli_rejects_arm_bound_to_different_metric_registry(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    baseline_hash = _put_arm(registry, _arm(0.5, 0.5))
    candidate = _arm(0.51, 0.49)
    candidate["metric_registry_sha256"] = "f" * 64
    candidate_hash = _put_arm(registry, candidate)
    charter = tmp_path / "acceptance-v1.yaml"
    _write_charter(charter)

    result = CliRunner().invoke(
        app, _arguments(artifact_root, charter, baseline_hash, candidate_hash)
    )

    assert result.exit_code == 3, result.output
    receipt = json.loads(result.stdout)
    decision = json.loads(_payload(registry, receipt["decision_manifest_sha256"]))
    assert "GATE_INTEGRITY_ARM_REGISTRY_MISMATCH" in decision["integrity_reason_codes"]


def test_compare_cli_persists_incomplete_provenance_failure(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    baseline_hash = _put_arm(registry, _arm(0.5, 0.5), with_parent=False)
    candidate_hash = _put_arm(registry, _arm(0.51, 0.49))
    charter = tmp_path / "acceptance-v1.yaml"
    _write_charter(charter)

    result = CliRunner().invoke(
        app, _arguments(artifact_root, charter, baseline_hash, candidate_hash)
    )

    assert result.exit_code == 3, result.output
    receipt = json.loads(result.stdout)
    decision = json.loads(_payload(registry, receipt["decision_manifest_sha256"]))
    assert "GATE_INTEGRITY_PROVENANCE_INCOMPLETE" in decision["integrity_reason_codes"]

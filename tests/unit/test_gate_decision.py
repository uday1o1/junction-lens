"""Seeded acceptance cases for every V1 release status class."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from junctionlens.gate.decision import decide_release
from junctionlens.registry.store import canonical_json_bytes

HASH = "a" * 64
HARDWARE_HASH = "b" * 64


def _write_charter(path: Path) -> str:
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
        "metric_registry_sha256": "d" * 64,
        "slice_registry_sha256": "e" * 64,
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
                "id": "overall.TOP_lt",
                "metric": "TOP_lt",
                "slice": "overall",
                "direction": "higher_is_better",
                "margin": 0.005,
                "support": "overall",
                "estimator": "ratio",
                "stage": "accuracy",
            }
        ],
        "primary_hypotheses": [
            {
                "id": "top_lt_superiority_vs_e0",
                "cell_id": "overall.TOP_lt",
                "minimum_improvement": 0.01,
            }
        ],
        "freeze_evidence": {
            "baseline_seed_checkpoint_sha256": {
                "20260813": HASH,
                "20260814": HASH,
                "20260815": HASH,
            },
            "m0_hardware_baseline_manifest_sha256": HARDWARE_HASH,
            "power_simulation_artifact_sha256": HASH,
            "product_priorities_sha256": HASH,
        },
    }
    charter_hash = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    body["charter_sha256"] = charter_hash
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return charter_hash


def _ratio_segments(value: float) -> list[dict[str, object]]:
    return [
        {"segment_id": f"segment-{index:03d}", "numerator": value, "denominator": 1.0}
        for index in range(200)
    ]


def _evidence(charter_hash: str, candidate_value: float = 0.52) -> dict[str, Any]:
    return {
        "schema_version": "junctionlens.gate-evidence.v1",
        "charter_sha256": charter_hash,
        "integrity": {
            "artifact_integrity": True,
            "baseline_evaluator_image_digest": HASH,
            "candidate_evaluator_image_digest": HASH,
            "baseline_data_manifest_sha256": HASH,
            "candidate_data_manifest_sha256": HASH,
            "baseline_split_manifest_sha256": HASH,
            "candidate_split_manifest_sha256": HASH,
            "baseline_preprocessing_sha256": HASH,
            "candidate_preprocessing_sha256": HASH,
            "baseline_postprocessing_sha256": HASH,
            "candidate_postprocessing_sha256": HASH,
            "metric_registry_sha256": "d" * 64,
            "slice_registry_sha256": "e" * 64,
            "calibration_ranks_frozen": True,
            "candidate_training_holdout_access_count": 0,
            "frame_sets_match": True,
            "slice_values_match": True,
            "support_values_match": True,
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
            "registry_inputs_match": True,
        },
        "cells": {
            "overall.TOP_lt": {
                "estimator": "ratio",
                "baseline": {"segments": _ratio_segments(0.5)},
                "candidate": {"segments": _ratio_segments(candidate_value)},
                "support": {
                    "paired_segments": 200,
                    "eligible_ground_truth_edges": 0,
                    "adjacent_frame_transitions": 0,
                    "temporal_segments": 0,
                },
                "counterexample_query": "metric == 'TOP_lt'",
            }
        },
        "runtime": {
            "baseline_hardware_manifest_sha256": HARDWARE_HASH,
            "hardware_baseline_manifest_sha256": HARDWARE_HASH,
            "baseline_gpu_provider_active": True,
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
            "trial_blocks": 10,
            "trial_block_order": ["AB", "BA"] * 5,
            "baseline_warmup_frames_per_block": 200,
            "baseline_measured_frames_per_block": 2000,
            "baseline_trial_block_order": ["AB", "BA"] * 5,
            "baseline_environment_valid": True,
            "paired_trial_ids_match": True,
            "environment_valid": True,
        },
    }


def _decide(tmp_path: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    charter = tmp_path / "acceptance-v1.yaml"
    evidence_path = tmp_path / "evidence.json"
    charter_hash = _write_charter(charter)
    evidence["charter_sha256"] = charter_hash
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return dict(decide_release(charter, evidence_path))


def test_seeded_pass_case(tmp_path: Path) -> None:
    evidence = _evidence("pending")
    decision = _decide(tmp_path, evidence)

    assert decision["status"] == "PASS"
    assert decision["cells"][0]["reason_code"] == "GATE_CELL_ACCEPTED"
    assert decision["primary_hypotheses"][0]["status"] == "PASS"


def test_seeded_regression_case(tmp_path: Path) -> None:
    evidence = _evidence("pending", candidate_value=0.4)
    decision = _decide(tmp_path, evidence)

    assert decision["status"] == "FAIL_REGRESSION"
    assert decision["cells"][0]["reason_code"] == "GATE_REGRESSION_CI_BELOW_MARGIN"


def test_seeded_insufficient_case(tmp_path: Path) -> None:
    evidence = _evidence("pending")
    evidence["cells"]["overall.TOP_lt"]["baseline"]["segments"].pop()
    evidence["cells"]["overall.TOP_lt"]["candidate"]["segments"].pop()
    evidence["cells"]["overall.TOP_lt"]["support"]["paired_segments"] = 199
    decision = _decide(tmp_path, evidence)

    assert decision["status"] == "INSUFFICIENT_EVIDENCE"
    assert decision["cells"][0]["reason_code"] == "GATE_INSUFFICIENT_OVERALL_SEGMENTS"


def test_seeded_integrity_case(tmp_path: Path) -> None:
    evidence = _evidence("pending")
    evidence["integrity"]["artifact_integrity"] = False
    decision = _decide(tmp_path, evidence)

    assert decision["status"] == "FAIL_INTEGRITY"
    assert "GATE_INTEGRITY_ARTIFACT_HASH_MISMATCH" in decision["integrity_reason_codes"]


def test_seeded_performance_case(tmp_path: Path) -> None:
    evidence = _evidence("pending")
    evidence["runtime"]["p95_latency_ms"] = 101.0
    decision = _decide(tmp_path, evidence)

    assert decision["status"] == "FAIL_PERFORMANCE"
    assert "GATE_PERFORMANCE_P95_LATENCY_BUDGET" in decision["performance_reason_codes"]


def test_seeded_infrastructure_case(tmp_path: Path) -> None:
    evidence = deepcopy(_evidence("pending"))
    evidence["runtime"]["hardware_baseline_manifest_sha256"] = "f" * 64
    decision = _decide(tmp_path, evidence)

    assert decision["status"] == "BLOCKED_INFRASTRUCTURE"
    assert "GATE_INFRASTRUCTURE_HARDWARE_MISMATCH" in decision["infrastructure_reason_codes"]

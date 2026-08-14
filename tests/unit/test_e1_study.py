"""Frozen E1 training selection and promotion-study tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from junctionlens.model.e0_profile import load_e0_profile
from junctionlens.model.e1_profile import load_e1_profile
from junctionlens.model.e1_study import E1StudyError, finalize_e1_study
from junctionlens.model.e1_training import E1TrainingError, select_e1_checkpoint

BASE_PATH = Path("configs/model/e0-independent-v1.yaml")
E1_PATH = Path("configs/model/e1-joint-v1.yaml")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    root: Path,
    experiment_id: str,
    profile_hash: str,
    *,
    selection_split: str = "c" * 64,
) -> tuple[str, str]:
    base = load_e0_profile(BASE_PATH)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    checkpoint = checkpoint_root / "epoch-03.pt"
    checkpoint.write_bytes(experiment_id.encode())
    checkpoint_sha256 = _hash(checkpoint)
    manifest = {
        "schema_version": f"junctionlens.{experiment_id}.test-run.v1",
        "state": "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION",
        "experiment_id": experiment_id,
        "seed": 20260813,
        "training_split_manifest_sha256": "a" * 64,
        "source_dataset_manifest_sha256": "b" * 64,
        "source_frame_manifest_sha256": "d" * 64,
    }
    if experiment_id == "E0-independent":
        manifest["profile_sha256"] = profile_hash
    else:
        manifest["base_profile_sha256"] = base.canonical_sha256()
        manifest["e1_profile_sha256"] = profile_hash
    (root / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    selection = {
        "schema_version": f"junctionlens.{experiment_id}.test-selection.v1",
        "state": "SELECTED_ON_MODEL_SELECTION",
        "selection_split_manifest_sha256": selection_split,
        "checkpoint_sha256": checkpoint_sha256,
        "selected": {"epoch": 3},
    }
    selection_path = root / "selection-receipt.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    return checkpoint_sha256, _hash(selection_path)


def _evidence(
    path: Path,
    baseline: tuple[str, str],
    candidate: tuple[str, str],
    *,
    candidate_metrics: dict[str, float] | None = None,
    internal_holdout_access_count: int = 0,
) -> None:
    baseline_metrics = {
        "DET_l": 0.60,
        "DET_t": 0.70,
        "TOP_lt": 0.50,
        "wrong_control_assignment_rate": 0.20,
        "official_composite": 0.61,
        "negative_log_likelihood": 0.40,
    }
    candidate_metrics = candidate_metrics or {
        "DET_l": 0.59,
        "DET_t": 0.69,
        "TOP_lt": 0.52,
        "wrong_control_assignment_rate": 0.19,
        "official_composite": 0.62,
        "negative_log_likelihood": 0.38,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.e1-study-evidence.v1",
                "study_id": "E0-independent-vs-E1-joint",
                "source_partition": "model_selection",
                "internal_holdout_access_count": internal_holdout_access_count,
                "selection_split_manifest_sha256": "c" * 64,
                "source_frame_manifest_sha256": "d" * 64,
                "evaluator": {
                    "official_implementation": "OpenLane-V2-v2.1",
                    "official_version": "2.1.0",
                    "official_config_sha256": "e" * 64,
                    "custom_match_version": "CustomMatchV1",
                    "custom_match_config_sha256": "f" * 64,
                },
                "baseline": {
                    "experiment_id": "E0-independent",
                    "seed": 20260813,
                    "checkpoint_sha256": baseline[0],
                    "selection_receipt_sha256": baseline[1],
                    "prediction_manifest_sha256": "1" * 64,
                    "evaluation_artifact_sha256": "2" * 64,
                    "metrics": baseline_metrics,
                },
                "candidate": {
                    "experiment_id": "E1-joint",
                    "seed": 20260813,
                    "checkpoint_sha256": candidate[0],
                    "selection_receipt_sha256": candidate[1],
                    "prediction_manifest_sha256": "3" * 64,
                    "evaluation_artifact_sha256": "4" * 64,
                    "metrics": candidate_metrics,
                },
            }
        ),
        encoding="utf-8",
    )


def test_e1_checkpoint_selection_is_lexicographic_and_immutable(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "run-manifest.json").write_text(
        json.dumps(
            {
                "state": "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION",
                "base_profile_sha256": "a" * 64,
                "e1_profile_sha256": "b" * 64,
                "recipe": {
                    "minimum_early_stopping_epoch": 20,
                    "early_stopping_patience": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    for epoch in (1, 2, 3):
        (checkpoints / f"epoch-{epoch:02d}.pt").write_bytes(str(epoch).encode())
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.e1-selection-scores.v1",
                "scores": [
                    {
                        "epoch": 1,
                        "lane_control_topology": 0.8,
                        "official_composite": 0.9,
                        "negative_log_likelihood": 0.1,
                        "selection_split_manifest_sha256": "c" * 64,
                    },
                    {
                        "epoch": 2,
                        "lane_control_topology": 0.9,
                        "official_composite": 0.7,
                        "negative_log_likelihood": 0.2,
                        "selection_split_manifest_sha256": "c" * 64,
                    },
                    {
                        "epoch": 3,
                        "lane_control_topology": 0.9,
                        "official_composite": 0.8,
                        "negative_log_likelihood": 0.5,
                        "selection_split_manifest_sha256": "c" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    receipt = select_e1_checkpoint(run, scores, expected_selection_split_manifest_sha256="c" * 64)

    assert receipt["selected"]["epoch"] == 3
    assert receipt["selection_rule"].startswith("lane-control-topology-desc")
    with pytest.raises(E1TrainingError, match="different split"):
        select_e1_checkpoint(run, scores, expected_selection_split_manifest_sha256="d" * 64)


def test_e1_selection_excludes_improvements_after_frozen_patience(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "run-manifest.json").write_text(
        json.dumps(
            {
                "state": "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION",
                "base_profile_sha256": "a" * 64,
                "e1_profile_sha256": "b" * 64,
                "recipe": {
                    "minimum_early_stopping_epoch": 20,
                    "early_stopping_patience": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    raw_scores = []
    for epoch in range(1, 31):
        (checkpoints / f"epoch-{epoch:02d}.pt").write_bytes(str(epoch).encode())
        topology = 0.9 if epoch == 1 else 1.0 if epoch == 28 else 0.8
        raw_scores.append(
            {
                "epoch": epoch,
                "lane_control_topology": topology,
                "official_composite": 0.5,
                "negative_log_likelihood": 0.5,
                "selection_split_manifest_sha256": "c" * 64,
            }
        )
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.e1-selection-scores.v1",
                "scores": raw_scores,
            }
        ),
        encoding="utf-8",
    )

    receipt = select_e1_checkpoint(run, scores, expected_selection_split_manifest_sha256="c" * 64)

    assert receipt["early_stopping"] == {
        "eligible_count": 27,
        "stopping_epoch": 27,
        "patience_exhausted": True,
    }
    assert receipt["selected"]["epoch"] == 1


def test_e1_study_accepts_exact_keep_gate_boundaries_and_promotes(tmp_path: Path) -> None:
    base = load_e0_profile(BASE_PATH)
    profile = load_e1_profile(E1_PATH, base)
    baseline = _run(tmp_path / "e0", "E0-independent", base.canonical_sha256())
    candidate = _run(tmp_path / "e1", "E1-joint", profile.canonical_sha256())
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "report.json"
    _evidence(evidence, baseline, candidate)

    report = finalize_e1_study(base, profile, tmp_path / "e0", tmp_path / "e1", evidence, output)

    assert report["state"] == "ACCEPTED"
    assert report["outcome"] == "PROMOTED"
    assert report["selected_experiment_id"] == "E1-joint"
    assert all(gate["passed"] for gate in report["keep_gates"])
    with pytest.raises(E1StudyError, match="replace"):
        finalize_e1_study(base, profile, tmp_path / "e0", tmp_path / "e1", evidence, output)


def test_valid_negative_e1_study_retains_e0_without_failing_study(tmp_path: Path) -> None:
    base = load_e0_profile(BASE_PATH)
    profile = load_e1_profile(E1_PATH, base)
    baseline = _run(tmp_path / "e0", "E0-independent", base.canonical_sha256())
    candidate = _run(tmp_path / "e1", "E1-joint", profile.canonical_sha256())
    evidence = tmp_path / "evidence.json"
    _evidence(
        evidence,
        baseline,
        candidate,
        candidate_metrics={
            "DET_l": 0.60,
            "DET_t": 0.70,
            "TOP_lt": 0.51,
            "wrong_control_assignment_rate": 0.20,
            "official_composite": 0.61,
            "negative_log_likelihood": 0.40,
        },
    )

    report = finalize_e1_study(
        base,
        profile,
        tmp_path / "e0",
        tmp_path / "e1",
        evidence,
        tmp_path / "report.json",
    )

    assert report["study_validity"] == "ACCEPTED"
    assert report["outcome"] == "REJECTED_BY_KEEP_GATE"
    assert report["selected_experiment_id"] == "E0-independent"
    assert "E1_TOP_LT_IMPROVEMENT_FAILED" in report["reason_codes"]
    assert "E1_WRONG_CONTROL_REDUCTION_FAILED" in report["reason_codes"]


def test_e1_study_rejects_holdout_access_and_split_mismatch(tmp_path: Path) -> None:
    base = load_e0_profile(BASE_PATH)
    profile = load_e1_profile(E1_PATH, base)
    baseline = _run(tmp_path / "e0", "E0-independent", base.canonical_sha256())
    candidate = _run(
        tmp_path / "e1",
        "E1-joint",
        profile.canonical_sha256(),
        selection_split="9" * 64,
    )
    evidence = tmp_path / "evidence.json"
    _evidence(evidence, baseline, candidate, internal_holdout_access_count=1)

    with pytest.raises(E1StudyError, match="failed validation"):
        finalize_e1_study(
            base,
            profile,
            tmp_path / "e0",
            tmp_path / "e1",
            evidence,
            tmp_path / "report.json",
        )

    _evidence(evidence, baseline, candidate)
    with pytest.raises(E1StudyError, match="selection split"):
        finalize_e1_study(
            base,
            profile,
            tmp_path / "e0",
            tmp_path / "e1",
            evidence,
            tmp_path / "report.json",
        )

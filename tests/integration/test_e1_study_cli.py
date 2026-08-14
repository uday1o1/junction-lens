"""Public command-path coverage for the E1 experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from junctionlens.cli.main import app
from junctionlens.model.e0_profile import load_e0_profile
from junctionlens.model.e1_profile import load_e1_profile

ROOT = Path(__file__).parents[2]
BASE_PATH = ROOT / "configs/model/e0-independent-v1.yaml"
E1_PATH = ROOT / "configs/model/e1-joint-v1.yaml"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(root: Path, experiment: str, profile_hash: str) -> tuple[str, str]:
    base = load_e0_profile(BASE_PATH)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "epoch-02.pt"
    checkpoint.write_bytes(experiment.encode())
    manifest = {
        "state": "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION",
        "experiment_id": experiment,
        "seed": 20260813,
        "training_split_manifest_sha256": "a" * 64,
        "source_dataset_manifest_sha256": "b" * 64,
        "source_frame_manifest_sha256": "d" * 64,
    }
    if experiment == "E0-independent":
        manifest["profile_sha256"] = profile_hash
    else:
        manifest["base_profile_sha256"] = base.canonical_sha256()
        manifest["e1_profile_sha256"] = profile_hash
    (root / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    selection = root / "selection-receipt.json"
    selection.write_text(
        json.dumps(
            {
                "state": "SELECTED_ON_MODEL_SELECTION",
                "selection_split_manifest_sha256": "c" * 64,
                "selected": {"epoch": 2},
                "checkpoint_sha256": _hash(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    return _hash(checkpoint), _hash(selection)


def test_finalize_e1_study_cli_persists_truthful_negative_result(tmp_path: Path) -> None:
    base = load_e0_profile(BASE_PATH)
    e1 = load_e1_profile(E1_PATH, base)
    baseline = _run(tmp_path / "e0", "E0-independent", base.canonical_sha256())
    candidate = _run(tmp_path / "e1", "E1-joint", e1.canonical_sha256())
    evidence = tmp_path / "evidence.json"
    metrics = {
        "DET_l": 0.6,
        "DET_t": 0.7,
        "TOP_lt": 0.5,
        "wrong_control_assignment_rate": 0.2,
        "official_composite": 0.61,
        "negative_log_likelihood": 0.4,
    }
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.e1-study-evidence.v1",
                "study_id": "E0-independent-vs-E1-joint",
                "source_partition": "model_selection",
                "internal_holdout_access_count": 0,
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
                    "metrics": metrics,
                },
                "candidate": {
                    "experiment_id": "E1-joint",
                    "seed": 20260813,
                    "checkpoint_sha256": candidate[0],
                    "selection_receipt_sha256": candidate[1],
                    "prediction_manifest_sha256": "3" * 64,
                    "evaluation_artifact_sha256": "4" * 64,
                    "metrics": metrics,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        [
            "model",
            "finalize-e1-study",
            "--baseline-run-root",
            str(tmp_path / "e0"),
            "--candidate-run-root",
            str(tmp_path / "e1"),
            "--evidence",
            str(evidence),
            "--output",
            str(output),
            "--base-profile",
            str(BASE_PATH),
            "--profile",
            str(E1_PATH),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["state"] == "ACCEPTED"
    assert report["outcome"] == "REJECTED_BY_KEEP_GATE"
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_select_e1_cli_applies_frozen_lexicographic_rule(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    for epoch in (1, 2):
        (checkpoints / f"epoch-{epoch:02d}.pt").write_bytes(str(epoch).encode())
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
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.e1-selection-scores.v1",
                "scores": [
                    {
                        "epoch": 1,
                        "lane_control_topology": 0.5,
                        "official_composite": 0.9,
                        "negative_log_likelihood": 0.1,
                        "selection_split_manifest_sha256": "c" * 64,
                    },
                    {
                        "epoch": 2,
                        "lane_control_topology": 0.6,
                        "official_composite": 0.1,
                        "negative_log_likelihood": 1.0,
                        "selection_split_manifest_sha256": "c" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "model",
            "select-e1",
            "--run-root",
            str(run),
            "--scores",
            str(scores),
            "--selection-split-manifest-sha256",
            "c" * 64,
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["selected"]["epoch"] == 2


def test_train_e1_cli_fails_closed_without_registered_licensed_data(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    split = tmp_path / "split.json"
    split.write_text("{}", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text("{}", encoding="utf-8")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            app,
            [
                "model",
                "train-e1",
                "--dataset-root",
                str(dataset),
                "--split-manifest",
                str(split),
                "--topology-diagnostic",
                str(diagnostic),
                "--output-root",
                str(tmp_path / "run"),
                "--base-profile",
                str(BASE_PATH),
                "--profile",
                str(E1_PATH),
                "--adapter-config",
                str(ROOT / "configs/data/openlane-v2-v2.1.adapter.yaml"),
                "--split-policy",
                str(ROOT / "configs/data/openlane-v2-v2.1.split-v1.yaml"),
            ],
        )

    assert result.exit_code == 2
    assert "dataset profile is not registered" in result.stderr

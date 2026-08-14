"""Resumable remote model-study orchestration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from scripts.gpu import qualify_models


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_model_qualification_runs_frozen_seed_matrix_and_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    dataset = tmp_path / "dataset"
    data = tmp_path / "data-qualification"
    output = tmp_path / "models"
    project.mkdir()
    dataset.mkdir()
    data.mkdir()
    split = data / "split-manifest.json"
    split.write_text("{}\n", encoding="utf-8")
    split_sha256 = hashlib.sha256(split.read_bytes()).hexdigest()
    _write_json(
        data / "qualification.json",
        {"mechanical_state": "ACCEPTED", "segment_count": 700},
    )
    labels: list[str] = []

    def fake_cli(
        label: str,
        arguments: list[str],
        **_kwargs: object,
    ) -> dict[str, Any]:
        labels.append(label)
        if label in {"topology-diagnostic", "fit-e0-linker"}:
            path = Path(arguments[arguments.index("--output") + 1])
            _write_json(path, {"state": "ACCEPTED"})
        elif label.startswith("train-"):
            run = Path(arguments[arguments.index("--output-root") + 1])
            experiment = "E0-independent" if "e0" in label else "E1-joint"
            seed = int(label.rsplit("-", 1)[-1])
            checkpoint = run / "checkpoints/epoch-20.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(label, encoding="utf-8")
            _write_json(
                run / "run-manifest.json",
                {
                    "state": "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION",
                    "experiment_id": experiment,
                    "seed": seed,
                },
            )
        elif label.startswith("score-"):
            scores = Path(arguments[arguments.index("--output-root") + 1])
            _write_json(scores / "scores.json", {"schema_version": "test", "scores": []})
        elif label.startswith("select-"):
            run = Path(arguments[arguments.index("--run-root") + 1])
            checkpoint = run / "checkpoints/epoch-20.pt"
            _write_json(
                run / "selection-receipt.json",
                {
                    "state": "SELECTED_ON_MODEL_SELECTION",
                    "selection_split_manifest_sha256": split_sha256,
                    "selected": {"epoch": 20},
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
            )
        return {"state": "ACCEPTED"}

    monkeypatch.setattr(qualify_models, "_run_cli", fake_cli)

    first = qualify_models.qualify(project, dataset, data, output)
    second = qualify_models.qualify(project, dataset, data, output)

    assert first == second
    assert first["state"] == "TRAINING_AND_SELECTION_ACCEPTED"
    assert [item["seed"] for item in first["e0_runs"]] == list(qualify_models.SEEDS)
    assert labels.count("train-e0-20260813") == 1
    assert labels.count("train-e0-20260814") == 1
    assert labels.count("train-e0-20260815") == 1
    assert labels.count("train-e1-20260813") == 1
    assert labels.count("topology-diagnostic") == 1
    assert labels.count("fit-e0-linker") == 1

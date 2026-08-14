"""Public E0 baseline command-path tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from junctionlens.cli.main import app


def test_select_e0_cli_uses_frozen_lexicographic_rule(tmp_path: Path) -> None:
    checkpoints = tmp_path / "run" / "checkpoints"
    checkpoints.mkdir(parents=True)
    for epoch in (1, 2):
        (checkpoints / f"epoch-{epoch:02d}.pt").write_bytes(str(epoch).encode())
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.e0-selection-scores.v1",
                "scores": [
                    {
                        "epoch": 1,
                        "lane_control_topology": 0.5,
                        "official_composite": 0.9,
                        "negative_log_likelihood": 0.1,
                        "selection_split_manifest_sha256": "a" * 64,
                    },
                    {
                        "epoch": 2,
                        "lane_control_topology": 0.6,
                        "official_composite": 0.1,
                        "negative_log_likelihood": 1.0,
                        "selection_split_manifest_sha256": "a" * 64,
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
            "select-e0",
            "--run-root",
            str(tmp_path / "run"),
            "--scores",
            str(scores),
            "--selection-split-manifest-sha256",
            "a" * 64,
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["selected"]["epoch"] == 2


def test_train_e0_cli_fails_closed_without_registered_licensed_data(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    split = tmp_path / "split.json"
    split.write_text("{}", encoding="utf-8")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            app,
            [
                "model",
                "train-e0",
                "--dataset-root",
                str(dataset),
                "--split-manifest",
                str(split),
                "--seed",
                "20260813",
                "--output-root",
                str(tmp_path / "run"),
                "--profile",
                str(Path(__file__).parents[2] / "configs/model/e0-independent-v1.yaml"),
                "--adapter-config",
                str(Path(__file__).parents[2] / "configs/data/openlane-v2-v2.1.adapter.yaml"),
                "--split-policy",
                str(Path(__file__).parents[2] / "configs/data/openlane-v2-v2.1.split-v1.yaml"),
            ],
        )

    assert result.exit_code == 2
    assert "dataset profile is not registered" in result.stderr

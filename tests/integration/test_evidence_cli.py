"""Public CLI coverage for calibration and runtime evidence."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from junctionlens.cli.main import app


def test_calibration_and_runtime_user_paths(tmp_path: Path) -> None:
    calibration_input = tmp_path / "calibration.json"
    calibration_input.write_text(
        json.dumps(
            {
                "binary": [
                    {"id": "b0", "outcome": 0, "probability": 0.0},
                    {"id": "b1", "outcome": 1, "probability": 0.75},
                ],
                "geometry": [{"factor": 1.0, "id": "g0", "residual_m": 0.0, "scale_m": 1.0}],
                "multiclass": [{"id": "m0", "outcome_class": 1, "probabilities": [0.1, 0.9]}],
                "schema_version": "junctionlens.calibration-input.v1",
            }
        ),
        encoding="utf-8",
    )
    runtime_input = tmp_path / "runtime.json"
    runtime_input.write_text(
        json.dumps(
            {
                "clock_source": "time.perf_counter_ns",
                "samples": [
                    {
                        "duration_ms": 100.0,
                        "iteration": 0,
                        "phase": "inference",
                        "sample_kind": "warmup",
                    },
                    {
                        "duration_ms": 10.0,
                        "iteration": 0,
                        "phase": "inference",
                        "sample_kind": "measured",
                    },
                ],
                "schema_version": "junctionlens.runtime-input.v1",
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    calibration = runner.invoke(app, ["calibrate", "--input", str(calibration_input)])
    assert calibration.exit_code == 0, calibration.output
    calibration_result = json.loads(calibration.stdout)
    assert calibration_result["binary"]["nll_saturation_count"] == 1
    assert calibration_result["geometry"]["coverage_90"] == 1.0

    runtime = runner.invoke(app, ["evaluate", "--runtime-input", str(runtime_input)])
    assert runtime.exit_code == 0, runtime.output
    runtime_result = json.loads(runtime.stdout)
    assert runtime_result["clock_source"] == "time.perf_counter_ns"
    assert runtime_result["phases"]["inference"]["warmup"]["mean_ms"] == 100.0
    assert runtime_result["phases"]["inference"]["measured"]["mean_ms"] == 10.0


def test_calibration_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    input_path = tmp_path / "duplicate.json"
    input_path.write_text(
        '{"binary":[],"binary":[],"geometry":[],"multiclass":[],'
        '"schema_version":"junctionlens.calibration-input.v1"}',
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["calibrate", "--input", str(input_path)])

    assert result.exit_code == 2
    assert "duplicate JSON object key: binary" in result.stderr

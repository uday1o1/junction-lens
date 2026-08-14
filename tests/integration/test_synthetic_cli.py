"""Public CLI tests for synthetic corpus generation and verification."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from junctionlens.cli.main import app

runner = CliRunner()


def test_generate_then_verify_through_public_cli(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    generated = runner.invoke(app, ["synthetic", "generate", "--output", str(output)])
    assert generated.exit_code == 0, generated.output
    generated_result = json.loads(generated.stdout)
    assert generated_result["state"] == "ACCEPTED"
    assert generated_result["file_count"] > 20

    verified = runner.invoke(app, ["synthetic", "verify", "--root", str(output)])
    assert verified.exit_code == 0, verified.output
    verified_result = json.loads(verified.stdout)
    assert verified_result["manifest_sha256"] == generated_result["manifest_sha256"]


def test_verify_reports_changed_fixture_as_failure(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    assert runner.invoke(app, ["synthetic", "generate", "--output", str(output)]).exit_code == 0
    (output / "manifest.json").write_text("{}\n", encoding="utf-8")
    result = runner.invoke(app, ["synthetic", "verify", "--root", str(output)])
    assert result.exit_code == 2
    assert "synthetic corpus byte mismatch" in result.output

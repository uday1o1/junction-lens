"""Public CLI coverage for the V1 graph contract."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from junctionlens.cli.main import app


def test_contract_validate_and_convert_user_path(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "tests/fixtures/contract/v1/golden.pb"
    runner = CliRunner()
    validated = runner.invoke(app, ["contract", "validate", "--input", str(source)])
    assert validated.exit_code == 0, validated.output
    result = json.loads(validated.stdout)
    assert result["nodes"] == 3
    assert result["edges"] == 1

    converted = tmp_path / "golden.json"
    conversion = runner.invoke(
        app,
        [
            "contract",
            "convert",
            "--input",
            str(source),
            "--output",
            str(converted),
            "--from",
            "binary",
            "--to",
            "json",
        ],
    )
    assert conversion.exit_code == 0, conversion.output
    assert '"node_id": "72057594037927937"' in converted.read_text(encoding="utf-8")

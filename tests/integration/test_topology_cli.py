"""Public CLI coverage for the E1 learned-topology diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from junctionlens.cli.main import app


def test_verify_topology_cli_runs_both_declared_modes(tmp_path: Path) -> None:
    output = tmp_path / "topology.json"
    arguments = [
        "model",
        "verify-topology",
        "--output",
        str(output),
        "--base-profile",
        str(Path("configs/model/e0-independent-v1.yaml").resolve()),
        "--profile",
        str(Path("configs/model/e1-joint-v1.yaml").resolve()),
    ]

    result = CliRunner().invoke(app, arguments)
    repeated = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["state"] == "ACCEPTED"
    assert [item["mode"] for item in report["results"]] == [
        "oracle-nodes",
        "predicted-nodes",
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert repeated.exit_code == 2
    assert "already exists" in repeated.stderr

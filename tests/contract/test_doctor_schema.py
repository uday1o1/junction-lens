"""Contract tests for the machine-readable doctor response."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema

from junctionlens.doctor.models import DoctorReport


def test_committed_schema_matches_generated_contract() -> None:
    """The committed schema cannot drift from the typed model."""
    committed = json.loads(Path("schemas/doctor-report-v1.schema.json").read_text(encoding="utf-8"))
    assert committed == DoctorReport.model_json_schema()


def test_real_doctor_json_validates_against_schema() -> None:
    """The public CLI returns schema-valid JSON and no stdout decoration."""
    result = subprocess.run(
        [sys.executable, "-m", "junctionlens.cli.main", "doctor", "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    schema = json.loads(Path("schemas/doctor-report-v1.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    assert result.stdout.count("\n") == 1

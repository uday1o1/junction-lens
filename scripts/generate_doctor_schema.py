#!/usr/bin/env python3
"""Generate the committed doctor-report JSON schema."""

from __future__ import annotations

import json
from pathlib import Path

from junctionlens.doctor.models import DoctorReport


def main() -> None:
    """Write a stable schema generated from the Pydantic contract."""
    root = Path(__file__).resolve().parents[1]
    target = root / "schemas/doctor-report-v1.schema.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = DoctorReport.model_json_schema()
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

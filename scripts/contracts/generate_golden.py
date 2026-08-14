#!/usr/bin/env python3
"""Regenerate the committed V1 logical and wire compatibility fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from junctionlens.contract import to_binary, to_json  # noqa: E402
from junctionlens.contract.golden import make_golden_envelope  # noqa: E402


def main() -> None:
    destination = ROOT / "tests/fixtures/contract/v1"
    destination.mkdir(parents=True, exist_ok=True)
    envelope = make_golden_envelope()
    (destination / "golden.pb").write_bytes(to_binary(envelope))
    (destination / "golden.json").write_text(to_json(envelope), encoding="utf-8")


if __name__ == "__main__":
    main()

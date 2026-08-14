#!/usr/bin/env python3
"""Generate reproducible language bindings from the canonical V1 schema."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def generate_python(root: Path) -> None:
    """Generate Python bindings with the exact repository-local compiler."""
    protoc = root / ".tools/bin/protoc"
    if not protoc.is_file():
        raise RuntimeError("locked protoc is absent; run ./tools/jl bootstrap-cpu")
    output = root / "python"
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    subprocess.run(
        [
            str(protoc),
            f"--proto_path={root / 'proto'}",
            f"--python_out={output}",
            f"--pyi_out={output}",
            str(root / "proto/junctionlens/v1/scene_control_graph.proto"),
        ],
        check=True,
        env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", action="store_true", help="generate Python bindings")
    arguments = parser.parse_args()
    if not arguments.python:
        parser.error("select at least one generated language")
    generate_python(project_root())


if __name__ == "__main__":
    main()

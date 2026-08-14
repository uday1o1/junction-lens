#!/usr/bin/env python3
"""Exercise every mandatory V1 fault through the public immutable CLI workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from junctionlens.faults.models import FAULT_REASON_CODES, FaultKind  # noqa: E402
from junctionlens.faults.service import put_prediction_bundle  # noqa: E402
from junctionlens.faults.synthetic import build_synthetic_fault_bundle  # noqa: E402
from junctionlens.registry.service import EvidenceRegistry  # noqa: E402
from junctionlens.registry.store import canonical_json_bytes  # noqa: E402
from junctionlens.security.parsing import (  # noqa: E402
    ParseBoundaryError,
    ParseLimits,
    load_json_object,
)

SEEDS = (20260813, 20260814, 20260815)


class FaultQualificationError(RuntimeError):
    """Raised when one public fault derivation or nearby control is inconclusive."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualify(project_root: Path, output_root: Path) -> dict[str, object]:
    """Run the exact fault-by-seed matrix and persist all immutable receipts."""
    project_root = project_root.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
        raise FaultQualificationError("fault qualification output must be a real directory")
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = output_root / "artifacts"
    registry = EvidenceRegistry(
        artifacts,
        project_root / "schemas/artifact-manifest-v1.schema.json",
    )
    parent = put_prediction_bundle(registry, build_synthetic_fault_bundle())
    records = []
    for kind in FaultKind:
        for seed in SEEDS:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "junctionlens.cli.main",
                    "fault",
                    "--input",
                    parent,
                    "--kind",
                    kind.value,
                    "--seed",
                    str(seed),
                    "--fraction",
                    "0.5",
                    "--artifact-root",
                    str(artifacts),
                    "--schema",
                    str(project_root / "schemas/artifact-manifest-v1.schema.json"),
                ],
                cwd=project_root,
                check=False,
                capture_output=True,
                timeout=300,
            )
            if completed.returncode != 0:
                raise FaultQualificationError(
                    f"public fault workflow failed for {kind.value} seed {seed}"
                )
            try:
                receipt = load_json_object(
                    completed.stdout,
                    "public fault receipt",
                    ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
                )
            except ParseBoundaryError as error:
                raise FaultQualificationError("public fault receipt is invalid") from error
            expected_state = (
                "CONTROL_PASSED" if kind is FaultKind.PERMUTE_NODES_CORRECTLY else "DETECTED"
            )
            if (
                receipt.get("state") != expected_state
                or receipt.get("fault_kind") != kind.value
                or receipt.get("primary_reason_code") != FAULT_REASON_CODES[kind]
                or receipt.get("parent_manifest_sha256") != parent
            ):
                raise FaultQualificationError(
                    f"fault detector evidence differs for {kind.value} seed {seed}"
                )
            records.append({"seed": seed, **receipt})
    expected_count = len(FaultKind) * len(SEEDS)
    if len(records) != expected_count:
        raise FaultQualificationError("fault qualification matrix is incomplete")
    result: dict[str, object] = {
        "schema_version": "junctionlens.remote-fault-qualification.v1",
        "state": "ACCEPTED",
        "input_kind": "repository-owned-synthetic",
        "v1_release_acceptance_run": False,
        "parent_manifest_sha256": parent,
        "fault_kind_count": len(FaultKind),
        "seed_count": len(SEEDS),
        "case_count": expected_count,
        "detected_or_control_passed_count": len(records),
        "detection_rate": 1.0,
        "records": records,
    }
    qualification = output_root / "qualification.json"
    payload = canonical_json_bytes(result) + b"\n"
    if qualification.exists() and qualification.read_bytes() != payload:
        raise FaultQualificationError("existing fault qualification receipt differs")
    qualification.write_bytes(payload)
    result["qualification_sha256"] = _sha256(qualification)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = qualify(arguments.project_root, arguments.output_root)
    except (FaultQualificationError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"fault qualification error: {error}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

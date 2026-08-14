#!/usr/bin/env python3
"""Run the complete licensed-data core qualification through the public CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from junctionlens.registry.store import canonical_json_bytes  # noqa: E402
from junctionlens.security.parsing import (  # noqa: E402
    ParseBoundaryError,
    ParseLimits,
    load_json_object,
    load_json_object_path,
)
from junctionlens.security.redaction import redact_sensitive_text  # noqa: E402


class DataQualificationError(RuntimeError):
    """Raised when the licensed-data workflow cannot satisfy its frozen contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact(text: str, paths: Sequence[Path]) -> str:
    result = text
    for path in sorted((str(item) for item in paths), key=len, reverse=True):
        result = result.replace(path, "<REDACTED_PATH>")
    return redact_sensitive_text(result)


def _run_cli(
    label: str,
    arguments: Sequence[str],
    *,
    project_root: Path,
    staging_root: Path,
    sensitive_paths: Sequence[Path],
    timeout: int = 14_400,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "junctionlens.cli.main", *arguments]
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        timeout=timeout,
    )
    logs = staging_root / "logs"
    receipts = staging_root / "receipts"
    logs.mkdir(exist_ok=True)
    receipts.mkdir(exist_ok=True)
    (logs / f"{label}.stdout.log").write_text(
        _redact(completed.stdout.decode("utf-8", "replace"), sensitive_paths),
        encoding="utf-8",
    )
    (logs / f"{label}.stderr.log").write_text(
        _redact(completed.stderr.decode("utf-8", "replace"), sensitive_paths),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise DataQualificationError(
            f"public data workflow step {label} failed with exit code {completed.returncode}"
        )
    try:
        receipt = load_json_object(
            completed.stdout,
            f"{label} public CLI receipt",
            ParseLimits(max_bytes=16 * 1024 * 1024, max_depth=32, max_nodes=500_000),
        )
    except ParseBoundaryError as error:
        raise DataQualificationError(
            f"public data workflow step {label} returned invalid JSON"
        ) from error
    (receipts / f"{label}.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DataQualificationError(f"{label} did not return a lowercase SHA-256")
    return value


def _validate_receipts(receipts: Mapping[str, Mapping[str, Any]]) -> None:
    registration = receipts["register"]
    if (
        registration.get("dataset_id") != "openlane-v2-v2.1"
        or registration.get("profile") != "full"
        or registration.get("root") != "<registered-dataset-root>"
    ):
        raise DataQualificationError("full dataset registration receipt is invalid")
    audit = receipts["audit"]
    if audit.get("capacity_gate_accepted") is not True or not isinstance(
        audit.get("frame_count"), int
    ):
        raise DataQualificationError("full dataset capacity audit did not pass")
    parity = receipts["verify-adapter"]
    if (
        parity.get("state") != "ACCEPTED"
        or parity.get("frame_count") != 3
        or not isinstance(parity.get("maximum_absolute_numeric_error"), int | float)
        or not isinstance(parity.get("tolerance"), int | float)
        or float(parity["maximum_absolute_numeric_error"]) > float(parity["tolerance"])
    ):
        raise DataQualificationError("official adapter parity did not pass")
    frame_manifest = receipts["manifest"]
    if frame_manifest.get("state") != "ACCEPTED":
        raise DataQualificationError("frame manifest did not reach ACCEPTED")
    _sha256(frame_manifest.get("artifact_manifest_sha256"), "frame manifest")
    split = receipts["split"]
    if (
        split.get("state") != "ACCEPTED"
        or split.get("segment_count") != 700
        or split.get("overlap_count") != 0
    ):
        raise DataQualificationError("frozen 700-segment split did not pass")
    _sha256(split.get("split_manifest_sha256"), "split manifest")
    split_audit = receipts["audit-splits"]
    if (
        split_audit.get("state") != "ACCEPTED"
        or split_audit.get("segment_count") != 700
        or split_audit.get("overlap_count") != 0
    ):
        raise DataQualificationError("independent split audit did not pass")
    visual = receipts["visual-audit"]
    if (
        visual.get("state") != "PENDING_HUMAN_INSPECTION"
        or visual.get("selected_frame_count") != 12
        or visual.get("range_gate_accepted") is not True
    ):
        raise DataQualificationError("private visual audit bundle is incomplete")
    _sha256(visual.get("bundle_manifest_sha256"), "visual audit bundle")


def qualify(project_root: Path, dataset_root: Path, output_root: Path) -> Mapping[str, Any]:
    """Create one no-clobber mechanical data qualification and pending visual gate."""
    project_root = project_root.resolve(strict=True)
    dataset_root = dataset_root.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise DataQualificationError("licensed dataset root must be a real directory")
    if output_root.exists() or output_root.is_symlink():
        raise DataQualificationError("data qualification output already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    sensitive_paths = (project_root, dataset_root, staging, output_root, Path.home())
    try:
        receipts: dict[str, Mapping[str, Any]] = {}
        receipts["register"] = _run_cli(
            "register",
            ["data", "register", "--root", str(dataset_root), "--profile", "full"],
            project_root=project_root,
            staging_root=staging,
            sensitive_paths=sensitive_paths,
        )
        receipts["audit"] = _run_cli(
            "audit",
            ["data", "audit", "--root", str(dataset_root), "--profile", "full"],
            project_root=project_root,
            staging_root=staging,
            sensitive_paths=sensitive_paths,
        )
        receipts["verify-adapter"] = _run_cli(
            "verify-adapter",
            ["data", "verify-adapter", "--root", str(dataset_root), "--profile", "full"],
            project_root=project_root,
            staging_root=staging,
            sensitive_paths=sensitive_paths,
        )
        registry = staging / "registry"
        receipts["manifest"] = _run_cli(
            "manifest",
            [
                "data",
                "manifest",
                "--root",
                str(dataset_root),
                "--profile",
                "full",
                "--artifact-root",
                str(registry),
            ],
            project_root=project_root,
            staging_root=staging,
            sensitive_paths=sensitive_paths,
        )
        frame_manifest_sha256 = _sha256(
            receipts["manifest"].get("artifact_manifest_sha256"), "frame manifest"
        )
        split_path = staging / "split-manifest.json"
        receipts["split"] = _run_cli(
            "split",
            [
                "data",
                "split",
                "--frame-manifest-sha256",
                frame_manifest_sha256,
                "--artifact-root",
                str(registry),
                "--export",
                str(split_path),
            ],
            project_root=project_root,
            staging_root=staging,
            sensitive_paths=sensitive_paths,
        )
        receipts["audit-splits"] = _run_cli(
            "audit-splits",
            ["data", "audit-splits", "--manifest", str(split_path)],
            project_root=project_root,
            staging_root=staging,
            sensitive_paths=sensitive_paths,
        )
        receipts["visual-audit"] = _run_cli(
            "visual-audit",
            [
                "data",
                "visual-audit",
                "--root",
                str(dataset_root),
                "--profile",
                "full",
                "--output",
                str(staging / "private-visual-audit"),
            ],
            project_root=project_root,
            staging_root=staging,
            sensitive_paths=sensitive_paths,
        )
        _validate_receipts(receipts)
        visual_manifest_path = staging / "private-visual-audit/manifest.json"
        try:
            visual_manifest = load_json_object_path(
                visual_manifest_path,
                "private visual audit manifest",
                ParseLimits(max_bytes=4 * 1024 * 1024, max_depth=24, max_nodes=100_000),
            )
        except ParseBoundaryError as error:
            raise DataQualificationError("private visual audit manifest is invalid") from error
        if (
            visual_manifest.get("policy_id") != "openlane-v2-v2.1-audit-v1"
            or _sha256_file(visual_manifest_path)
            != receipts["visual-audit"]["bundle_manifest_sha256"]
        ):
            raise DataQualificationError("private visual audit identity does not match")
        receipt_hashes = {
            name: _sha256_file(staging / "receipts" / f"{name}.json") for name in sorted(receipts)
        }
        result: Mapping[str, Any] = {
            "schema_version": "junctionlens.remote-data-qualification.v1",
            "state": "PENDING_HUMAN_INSPECTION",
            "mechanical_state": "ACCEPTED",
            "dataset_id": "openlane-v2-v2.1",
            "profile": "full",
            "segment_count": 700,
            "official_parity_frame_count": 3,
            "visual_audit_frame_count": 12,
            "visual_audit_manifest_sha256": receipts["visual-audit"]["bundle_manifest_sha256"],
            "visual_audit_policy_id": visual_manifest["policy_id"],
            "split_manifest_sha256": receipts["split"]["split_manifest_sha256"],
            "receipt_sha256": receipt_hashes,
        }
        (staging / "qualification.json").write_bytes(canonical_json_bytes(result) + b"\n")
        staging.replace(output_root)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = qualify(arguments.project_root, arguments.dataset_root, arguments.output_root)
    except (DataQualificationError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"data qualification error: {error}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

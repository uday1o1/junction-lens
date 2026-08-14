#!/usr/bin/env python3
"""Run resumable E0 and E1 training and frozen model-selection through the CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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

SEEDS = (20260813, 20260814, 20260815)


class ModelQualificationError(RuntimeError):
    """Raised when the resumable model study cannot satisfy a frozen gate."""


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
    output_root: Path,
    sensitive_paths: Sequence[Path],
    timeout: int = 172_800,
) -> Mapping[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "junctionlens.cli.main", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        timeout=timeout,
    )
    logs = output_root / "logs"
    receipts = output_root / "receipts"
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
        raise ModelQualificationError(
            f"public model workflow step {label} failed with exit code {completed.returncode}"
        )
    try:
        receipt = load_json_object(
            completed.stdout,
            f"{label} public CLI receipt",
            ParseLimits(max_bytes=64 * 1024 * 1024, max_depth=32, max_nodes=1_000_000),
        )
    except ParseBoundaryError as error:
        raise ModelQualificationError(
            f"public model workflow step {label} returned invalid JSON"
        ) from error
    (receipts / f"{label}.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def _load(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return load_json_object_path(
            path,
            label,
            ParseLimits(max_bytes=64 * 1024 * 1024, max_depth=32, max_nodes=1_000_000),
        )
    except ParseBoundaryError as error:
        raise ModelQualificationError(str(error)) from error


def _training_complete(run_root: Path, experiment: str, seed: int) -> bool:
    manifest_path = run_root / "run-manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _load(manifest_path, f"{experiment} training manifest")
    if manifest.get("experiment_id") != experiment or manifest.get("seed") != seed:
        raise ModelQualificationError(f"existing {experiment} run identity differs")
    return manifest.get("state") == "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION"


def _selected(run_root: Path, split_sha256: str) -> Mapping[str, Any]:
    receipt = _load(run_root / "selection-receipt.json", "model selection receipt")
    selected = receipt.get("selected")
    if (
        receipt.get("state") != "SELECTED_ON_MODEL_SELECTION"
        or receipt.get("selection_split_manifest_sha256") != split_sha256
        or not isinstance(selected, dict)
        or not isinstance(selected.get("epoch"), int)
        or not isinstance(receipt.get("checkpoint_sha256"), str)
    ):
        raise ModelQualificationError("model selection receipt is incomplete")
    checkpoint = run_root / "checkpoints" / f"epoch-{int(selected['epoch']):02d}.pt"
    if _sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
        raise ModelQualificationError("selected checkpoint failed hash verification")
    return receipt


def qualify(
    project_root: Path,
    dataset_root: Path,
    data_qualification_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    """Train, score, and select all predeclared core model runs with resume."""
    project_root = project_root.resolve(strict=True)
    dataset_root = dataset_root.resolve(strict=True)
    data_qualification_root = data_qualification_root.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
        raise ModelQualificationError("model qualification output must be a real directory")
    output_root.mkdir(parents=True, exist_ok=True)
    split_manifest = data_qualification_root / "split-manifest.json"
    data_receipt = _load(
        data_qualification_root / "qualification.json",
        "licensed-data qualification",
    )
    if (
        data_receipt.get("mechanical_state") != "ACCEPTED"
        or data_receipt.get("segment_count") != 700
        or not split_manifest.is_file()
    ):
        raise ModelQualificationError("licensed-data qualification is incomplete")
    split_sha256 = _sha256_file(split_manifest)
    sensitive_paths = (
        project_root,
        dataset_root,
        data_qualification_root,
        output_root,
        Path.home(),
    )
    topology = output_root / "topology-diagnostic.json"
    if not topology.exists():
        _run_cli(
            "topology-diagnostic",
            ["model", "verify-topology", "--output", str(topology)],
            project_root=project_root,
            output_root=output_root,
            sensitive_paths=sensitive_paths,
        )
    if _load(topology, "topology diagnostic").get("state") != "ACCEPTED":
        raise ModelQualificationError("topology diagnostic did not pass")
    linker = output_root / "e0-independent-linker.json"
    if not linker.exists():
        _run_cli(
            "fit-e0-linker",
            [
                "model",
                "fit-e0-linker",
                "--dataset-root",
                str(dataset_root),
                "--split-manifest",
                str(split_manifest),
                "--output",
                str(linker),
            ],
            project_root=project_root,
            output_root=output_root,
            sensitive_paths=sensitive_paths,
        )
    e0_runs = []
    for seed in SEEDS:
        run_root = output_root / f"e0-seed-{seed}"
        if not _training_complete(run_root, "E0-independent", seed):
            arguments = [
                "model",
                "train-e0",
                "--dataset-root",
                str(dataset_root),
                "--split-manifest",
                str(split_manifest),
                "--seed",
                str(seed),
                "--output-root",
                str(run_root),
                "--device",
                "cuda",
            ]
            if run_root.exists():
                arguments.append("--resume")
            _run_cli(
                f"train-e0-{seed}",
                arguments,
                project_root=project_root,
                output_root=output_root,
                sensitive_paths=sensitive_paths,
            )
        scores = output_root / f"e0-seed-{seed}-selection"
        _run_cli(
            f"score-e0-{seed}",
            [
                "model",
                "score-checkpoints",
                "--experiment",
                "E0-independent",
                "--run-root",
                str(run_root),
                "--dataset-root",
                str(dataset_root),
                "--split-manifest",
                str(split_manifest),
                "--linker",
                str(linker),
                "--output-root",
                str(scores),
                "--device",
                "cuda",
            ],
            project_root=project_root,
            output_root=output_root,
            sensitive_paths=sensitive_paths,
        )
        _run_cli(
            f"select-e0-{seed}",
            [
                "model",
                "select-e0",
                "--run-root",
                str(run_root),
                "--scores",
                str(scores / "scores.json"),
                "--selection-split-manifest-sha256",
                split_sha256,
            ],
            project_root=project_root,
            output_root=output_root,
            sensitive_paths=sensitive_paths,
        )
        e0_runs.append(
            {
                "seed": seed,
                "run_manifest_sha256": _sha256_file(run_root / "run-manifest.json"),
                "selection_receipt_sha256": _sha256_file(run_root / "selection-receipt.json"),
                "checkpoint_sha256": _selected(run_root, split_sha256)["checkpoint_sha256"],
            }
        )
    e1_run = output_root / "e1-seed-20260813"
    if not _training_complete(e1_run, "E1-joint", 20260813):
        arguments = [
            "model",
            "train-e1",
            "--dataset-root",
            str(dataset_root),
            "--split-manifest",
            str(split_manifest),
            "--topology-diagnostic",
            str(topology),
            "--output-root",
            str(e1_run),
            "--device",
            "cuda",
        ]
        if e1_run.exists():
            arguments.append("--resume")
        _run_cli(
            "train-e1-20260813",
            arguments,
            project_root=project_root,
            output_root=output_root,
            sensitive_paths=sensitive_paths,
        )
    e1_scores = output_root / "e1-seed-20260813-selection"
    _run_cli(
        "score-e1-20260813",
        [
            "model",
            "score-checkpoints",
            "--experiment",
            "E1-joint",
            "--run-root",
            str(e1_run),
            "--dataset-root",
            str(dataset_root),
            "--split-manifest",
            str(split_manifest),
            "--output-root",
            str(e1_scores),
            "--device",
            "cuda",
        ],
        project_root=project_root,
        output_root=output_root,
        sensitive_paths=sensitive_paths,
    )
    _run_cli(
        "select-e1-20260813",
        [
            "model",
            "select-e1",
            "--run-root",
            str(e1_run),
            "--scores",
            str(e1_scores / "scores.json"),
            "--selection-split-manifest-sha256",
            split_sha256,
        ],
        project_root=project_root,
        output_root=output_root,
        sensitive_paths=sensitive_paths,
    )
    result: Mapping[str, Any] = {
        "schema_version": "junctionlens.remote-model-qualification.v1",
        "state": "TRAINING_AND_SELECTION_ACCEPTED",
        "source_partition": ["model_training", "model_selection"],
        "internal_holdout_access_count": 0,
        "split_manifest_sha256": split_sha256,
        "topology_diagnostic_sha256": _sha256_file(topology),
        "independent_linker_sha256": _sha256_file(linker),
        "e0_runs": e0_runs,
        "e1_run": {
            "seed": 20260813,
            "run_manifest_sha256": _sha256_file(e1_run / "run-manifest.json"),
            "selection_receipt_sha256": _sha256_file(e1_run / "selection-receipt.json"),
            "checkpoint_sha256": _selected(e1_run, split_sha256)["checkpoint_sha256"],
        },
    }
    qualification = output_root / "qualification.json"
    payload = canonical_json_bytes(result) + b"\n"
    if qualification.exists() and qualification.read_bytes() != payload:
        raise ModelQualificationError("existing model qualification receipt differs")
    qualification.write_bytes(payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-qualification-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = qualify(
            arguments.project_root,
            arguments.dataset_root,
            arguments.data_qualification_root,
            arguments.output_root,
        )
    except (ModelQualificationError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"model qualification error: {error}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

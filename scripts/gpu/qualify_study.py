#!/usr/bin/env python3
"""Finalize selected model evidence and freeze the pre-holdout acceptance charter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from junctionlens.gate.charter import load_charter_draft, load_frozen_charter  # noqa: E402
from junctionlens.registry import ContentAddressedStore  # noqa: E402
from junctionlens.registry.store import canonical_json_bytes  # noqa: E402
from junctionlens.security.parsing import (  # noqa: E402
    ParseBoundaryError,
    ParseLimits,
    load_json_object,
    load_json_object_path,
)
from junctionlens.security.redaction import redact_sensitive_text  # noqa: E402

SEEDS = (20260813, 20260814, 20260815)
POWER_REPLICATES = 10_000
POWER_SEED = 20260813
OFFICIAL_EVALUATOR_CONFIG_SHA256 = (
    "f8c88d0f5ecc6823d5ed60d22ed04066345bc1fd5237aca72f3990054d08d0c7"
)


class StudyQualificationError(RuntimeError):
    """Raised when selected-model or charter-freeze evidence is incomplete."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise StudyQualificationError(f"existing {path.name} differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _load(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return load_json_object_path(
            path,
            label,
            ParseLimits(max_bytes=64 * 1024 * 1024, max_depth=48, max_nodes=2_000_000),
        )
    except ParseBoundaryError as error:
        raise StudyQualificationError(str(error)) from error


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
        raise StudyQualificationError(
            f"public study workflow step {label} failed with exit code {completed.returncode}"
        )
    try:
        receipt = load_json_object(
            completed.stdout,
            f"{label} public CLI receipt",
            ParseLimits(max_bytes=64 * 1024 * 1024, max_depth=48, max_nodes=2_000_000),
        )
    except ParseBoundaryError as error:
        raise StudyQualificationError(f"public study step {label} returned invalid JSON") from error
    (receipts / f"{label}.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def _selected_result(
    root: Path, *, experiment: str, seed: int, split_sha256: str
) -> Mapping[str, Any]:
    result = _load(root / "selected-evaluation.json", "selected model evaluation")
    metrics = root / "metrics.json"
    predictions = root / "prediction-manifest.json"
    if (
        result.get("schema_version") != "junctionlens.selected-model-evaluation.v1"
        or result.get("state") != "ACCEPTED"
        or result.get("experiment_id") != experiment
        or result.get("seed") != seed
        or result.get("source_partition") != "model_selection"
        or result.get("internal_holdout_access_count") != 0
        or result.get("selection_split_manifest_sha256") != split_sha256
        or not metrics.is_file()
        or not predictions.is_file()
        or result.get("evaluation_artifact_sha256") != _sha256_file(metrics)
        or result.get("prediction_manifest_sha256") != _sha256_file(predictions)
    ):
        raise StudyQualificationError("selected model evaluation identity is incomplete")
    freeze_metrics = result.get("freeze_metrics")
    if not isinstance(freeze_metrics, dict) or len(freeze_metrics) != 12:
        raise StudyQualificationError("selected model evaluation lacks frozen metric cells")
    return result


def _evaluate_selected(
    *,
    project_root: Path,
    dataset_root: Path,
    models_root: Path,
    split_manifest: Path,
    split_sha256: str,
    artifact_root: Path,
    output_root: Path,
    experiment: str,
    seed: int,
    sensitive_paths: Sequence[Path],
) -> Mapping[str, Any]:
    prefix = "e0" if experiment == "E0-independent" else "e1"
    run_root = models_root / f"{prefix}-seed-{seed}"
    evaluation_root = output_root / f"{prefix}-seed-{seed}-evaluation"
    if not evaluation_root.exists():
        arguments = [
            "model",
            "evaluate-selected",
            "--experiment",
            experiment,
            "--run-root",
            str(run_root),
            "--dataset-root",
            str(dataset_root),
            "--split-manifest",
            str(split_manifest),
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(evaluation_root),
            "--device",
            "cuda",
        ]
        if experiment == "E0-independent":
            arguments.extend(("--linker", str(models_root / "e0-independent-linker.json")))
        _run_cli(
            f"evaluate-{prefix}-{seed}",
            arguments,
            project_root=project_root,
            output_root=output_root,
            sensitive_paths=sensitive_paths,
        )
    return _selected_result(
        evaluation_root,
        experiment=experiment,
        seed=seed,
        split_sha256=split_sha256,
    )


def _study_arm(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = cast(Mapping[str, Any], result["metrics"])
    return {
        "experiment_id": result["experiment_id"],
        "seed": result["seed"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "selection_receipt_sha256": result["selection_receipt_sha256"],
        "prediction_manifest_sha256": result["prediction_manifest_sha256"],
        "evaluation_artifact_sha256": result["evaluation_artifact_sha256"],
        "metrics": {
            name: metrics[name]
            for name in (
                "DET_l",
                "DET_t",
                "TOP_lt",
                "wrong_control_assignment_rate",
                "official_composite",
                "negative_log_likelihood",
            )
        },
    }


def _power_simulation(
    baseline_variability: Mapping[str, list[float]], margins: Mapping[str, float]
) -> Mapping[str, Any]:
    alpha = 0.05 / len(margins)
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    cells: dict[str, Any] = {}
    for index, cell_id in enumerate(sorted(baseline_variability)):
        values = baseline_variability[cell_id]
        observed_stddev = statistics.stdev(values)
        standard_error = max(observed_stddev / math.sqrt(len(values)), 1.0e-6)
        target_effect = max(float(margins[cell_id]), 0.005)
        generator = random.Random(POWER_SEED + index)  # noqa: S311 - deterministic simulation
        detections = sum(
            generator.gauss(target_effect, standard_error) - critical * standard_error > 0.0
            for _ in range(POWER_REPLICATES)
        )
        cells[cell_id] = {
            "baseline_values": values,
            "observed_seed_standard_deviation": observed_stddev,
            "assumed_standard_error": standard_error,
            "target_absolute_effect": target_effect,
            "estimated_detection_probability": detections / POWER_REPLICATES,
        }
    return {
        "schema_version": "junctionlens.preholdout-power-simulation.v1",
        "method": "three-seed-normal-approximation-v1",
        "replicates": POWER_REPLICATES,
        "seed": POWER_SEED,
        "bonferroni_two_sided_alpha": alpha,
        "candidate_results_used": False,
        "internal_holdout_used": False,
        "source_partitions": ["model_training", "model_selection"],
        "cells": cells,
        "limitations": [
            "This pre-holdout approximation uses between-seed variability from three E0 runs.",
            "It does not replace the paired segment bootstrap used for the release decision.",
        ],
    }


def _store_hardware_baseline(
    store: ContentAddressedStore,
    *,
    environment_path: Path,
    performance_path: Path,
    provider_path: Path,
    source_commit: str,
) -> str:
    environment = _load(environment_path, "remote qualification environment")
    performance = _load(performance_path, "M0 hardware performance baseline")
    provider = _load(provider_path, "CUDA provider assignment")
    if (
        environment.get("failures") != []
        or performance.get("status") != "PASSED"
        or provider.get("status") != "PASSED"
        or provider.get("provider_profile") != "cuda"
    ):
        raise StudyQualificationError("M0 hardware baseline inputs are not accepted")
    payload = {
        "schema_version": "junctionlens.m0-hardware-baseline.v1",
        "source_commit": source_commit,
        "selected_gpu": environment.get("selected_gpu"),
        "driver_version": environment.get("driver_version"),
        "cuda_toolkit_version": environment.get("cuda_toolkit_version"),
        "cudnn_package_version": environment.get("cudnn_package_version"),
        "environment_sha256": _sha256_file(environment_path),
        "performance_sha256": _sha256_file(performance_path),
        "provider_assignment_sha256": _sha256_file(provider_path),
        "performance": performance,
    }
    receipt = store.put_bytes(
        canonical_json_bytes(payload),
        kind="benchmark",
        media_type="application/json",
        license_id="Apache-2.0",
        metadata={"evidence_type": "m0_hardware_baseline", "source_commit": source_commit},
    )
    return receipt.manifest_sha256


def qualify(
    *,
    project_root: Path,
    dataset_root: Path,
    data_qualification_root: Path,
    models_root: Path,
    environment_path: Path,
    performance_path: Path,
    provider_path: Path,
    output_root: Path,
    source_commit: str,
) -> Mapping[str, Any]:
    """Produce selected-model evidence and freeze policy before any holdout inference."""
    project_root = project_root.resolve(strict=True)
    dataset_root = dataset_root.resolve(strict=True)
    data_qualification_root = data_qualification_root.resolve(strict=True)
    models_root = models_root.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
        raise StudyQualificationError("study qualification output must be a real directory")
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = output_root / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    split_manifest = data_qualification_root / "split-manifest.json"
    split_sha256 = _sha256_file(split_manifest)
    sensitive_paths = (
        project_root,
        dataset_root,
        data_qualification_root,
        models_root,
        output_root,
        Path.home(),
    )
    e0_results = [
        _evaluate_selected(
            project_root=project_root,
            dataset_root=dataset_root,
            models_root=models_root,
            split_manifest=split_manifest,
            split_sha256=split_sha256,
            artifact_root=artifact_root,
            output_root=output_root,
            experiment="E0-independent",
            seed=seed,
            sensitive_paths=sensitive_paths,
        )
        for seed in SEEDS
    ]
    e1_result = _evaluate_selected(
        project_root=project_root,
        dataset_root=dataset_root,
        models_root=models_root,
        split_manifest=split_manifest,
        split_sha256=split_sha256,
        artifact_root=artifact_root,
        output_root=output_root,
        experiment="E1-joint",
        seed=20260813,
        sensitive_paths=sensitive_paths,
    )
    measured = {
        "schema_version": "junctionlens.e0-measured-evidence.v1",
        "seed_metrics": [
            {
                "seed": result["seed"],
                "DET_l": result["metrics"]["DET_l"],
                "DET_t": result["metrics"]["DET_t"],
                "TOP_ll": result["freeze_metrics"]["overall.TOP_ll"],
                "TOP_lt": result["metrics"]["TOP_lt"],
                "negative_log_likelihood": result["metrics"]["negative_log_likelihood"],
            }
            for result in e0_results
        ],
        "limitations": [
            "These measurements use model_selection only and make no internal-holdout claim.",
            "E0 topology is produced by frozen independent geometric rules, not a joint learner.",
            "Dataset-derived weights remain restricted and are not redistribution artifacts.",
        ],
        "failed_examples": [
            {
                "frame_key": str(result["measured_example"]["frame_token"]),
                "reason_code": str(result["measured_example"]["classification"]),
                "summary": (
                    f"Worst measured seed {result['seed']} frame in "
                    f"{result['measured_example']['source_domain']} with severity "
                    f"{float(result['measured_example']['severity_score']):.6f}."
                ),
            }
            for result in e0_results
        ],
    }
    measured_path = output_root / "e0-measured-evidence.json"
    _atomic_json(measured_path, measured)
    e0_final_root = output_root / "e0-final"
    if not e0_final_root.exists():
        _run_cli(
            "finalize-e0",
            [
                "model",
                "finalize-e0",
                *(
                    item
                    for seed in SEEDS
                    for item in ("--run-root", str(models_root / f"e0-seed-{seed}"))
                ),
                "--measured-evidence",
                str(measured_path),
                "--output-root",
                str(e0_final_root),
            ],
            project_root=project_root,
            output_root=output_root,
            sensitive_paths=sensitive_paths,
        )
    model_manifest = _load(e0_final_root / "model-manifest.json", "final E0 model manifest")
    finalization_receipt = _load(
        output_root / "receipts/finalize-e0.json", "E0 finalization receipt"
    )
    if (
        model_manifest.get("state") != "THREE_SEED_BASELINE_COMPLETE"
        or model_manifest.get("measured_evidence_sha256") != _sha256_file(measured_path)
        or finalization_receipt.get("state") != "ACCEPTED"
        or finalization_receipt.get("model_manifest_sha256")
        != _sha256_file(e0_final_root / "model-manifest.json")
        or finalization_receipt.get("model_card_sha256")
        != _sha256_file(e0_final_root / "MODEL_CARD.md")
    ):
        raise StudyQualificationError("E0 finalization is incomplete")
    e1_evidence = {
        "schema_version": "junctionlens.e1-study-evidence.v1",
        "study_id": "E0-independent-vs-E1-joint",
        "source_partition": "model_selection",
        "internal_holdout_access_count": 0,
        "selection_split_manifest_sha256": split_sha256,
        "source_frame_manifest_sha256": e0_results[0]["source_frame_manifest_sha256"],
        "evaluator": {
            "official_implementation": "OpenLane-V2-v2.1",
            "official_version": "2.1.0",
            "official_config_sha256": OFFICIAL_EVALUATOR_CONFIG_SHA256,
            "custom_match_version": "CustomMatchV1",
            "custom_match_config_sha256": _sha256_file(project_root / "configs/metrics/v1.yaml"),
        },
        "baseline": _study_arm(e0_results[0]),
        "candidate": _study_arm(e1_result),
    }
    e1_evidence_path = output_root / "e1-study-evidence.json"
    _atomic_json(e1_evidence_path, e1_evidence)
    e1_report_path = output_root / "e1-study-report.json"
    if not e1_report_path.exists():
        _run_cli(
            "finalize-e1-study",
            [
                "model",
                "finalize-e1-study",
                "--baseline-run-root",
                str(models_root / "e0-seed-20260813"),
                "--candidate-run-root",
                str(models_root / "e1-seed-20260813"),
                "--evidence",
                str(e1_evidence_path),
                "--output",
                str(e1_report_path),
            ],
            project_root=project_root,
            output_root=output_root,
            sensitive_paths=sensitive_paths,
        )
    e1_report = _load(e1_report_path, "E1 study report")
    if (
        e1_report.get("state") != "ACCEPTED"
        or e1_report.get("study_validity") != "ACCEPTED"
        or e1_report.get("source_partition") != "model_selection"
        or e1_report.get("internal_holdout_access_count") != 0
        or e1_report.get("evidence_sha256") != _sha256_file(e1_evidence_path)
    ):
        raise StudyQualificationError("E1 keep-gate study is not accepted")
    draft = load_charter_draft(project_root / "configs/gates/acceptance-v1.draft.yaml")
    margins = {cell.id: cell.margin for cell in draft.cells}
    nonperformance_ids = [cell.id for cell in draft.cells if cell.stage != "performance"]
    baseline_variability = {
        cell_id: [float(result["freeze_metrics"][cell_id]) for result in e0_results]
        for cell_id in nonperformance_ids
    }
    store = ContentAddressedStore(
        artifact_root,
        project_root / "schemas/artifact-manifest-v1.schema.json",
    )
    power = _power_simulation(baseline_variability, margins)
    power_receipt = store.put_bytes(
        canonical_json_bytes(power),
        kind="comparison",
        media_type="application/json",
        license_id="Apache-2.0",
        metadata={
            "evidence_type": "preholdout_power_simulation",
            "candidate_results_used": False,
            "internal_holdout_used": False,
        },
    )
    hardware_manifest_sha256 = _store_hardware_baseline(
        store,
        environment_path=environment_path.resolve(strict=True),
        performance_path=performance_path.resolve(strict=True),
        provider_path=provider_path.resolve(strict=True),
        source_commit=source_commit,
    )
    priorities = {
        "draft_sha256": _sha256_file(project_root / "configs/gates/acceptance-v1.draft.yaml"),
        "budgets_sha256": _sha256_file(project_root / "configs/budgets/v1.yaml"),
        "metrics_sha256": _sha256_file(project_root / "configs/metrics/v1.yaml"),
        "slices_sha256": _sha256_file(project_root / "configs/slices/v1.yaml"),
    }
    freeze_evidence = {
        "schema_version": "junctionlens.baseline-freeze-evidence.v1",
        "experiment_id": "E0-independent",
        "source_partitions": ["model_training", "model_selection"],
        "internal_holdout_access_count": 0,
        "baseline_seed_checkpoint_sha256": {
            str(result["seed"]): result["checkpoint_sha256"] for result in e0_results
        },
        "baseline_variability": baseline_variability,
        "m0_hardware_baseline_manifest_sha256": hardware_manifest_sha256,
        "power_simulation": {
            "artifact_sha256": power_receipt.manifest_sha256,
            "candidate_results_used": False,
            "internal_holdout_used": False,
            "source_partitions": ["model_training", "model_selection"],
        },
        "product_priorities_sha256": hashlib.sha256(canonical_json_bytes(priorities)).hexdigest(),
        "proposed_margins": margins,
    }
    freeze_receipt = store.put_bytes(
        canonical_json_bytes(freeze_evidence),
        kind="baseline_freeze_evidence",
        media_type="application/json",
        license_id="Apache-2.0",
        metadata={"source_commit": source_commit},
    )
    charter_path = output_root / "acceptance-v1.yaml"
    if not charter_path.exists():
        _run_cli(
            "freeze-charter",
            [
                "gate",
                "freeze",
                "--draft",
                str(project_root / "configs/gates/acceptance-v1.draft.yaml"),
                "--baseline-run",
                f"artifacts://runs/{freeze_receipt.manifest_sha256}",
                "--output",
                str(charter_path),
                "--signer",
                f"remote-qualification:{source_commit[:12]}",
                "--artifact-root",
                str(artifact_root),
                "--metrics",
                str(project_root / "configs/metrics/v1.yaml"),
                "--slices",
                str(project_root / "configs/slices/v1.yaml"),
                "--project-root",
                str(project_root),
            ],
            project_root=project_root,
            output_root=output_root,
            sensitive_paths=sensitive_paths,
        )
    charter = load_frozen_charter(charter_path)
    if (
        charter.source_commit != source_commit
        or charter.baseline_run_manifest_sha256 != freeze_receipt.manifest_sha256
        or charter.freeze_evidence.m0_hardware_baseline_manifest_sha256 != hardware_manifest_sha256
    ):
        raise StudyQualificationError("frozen acceptance charter identity differs")
    result: Mapping[str, Any] = {
        "schema_version": "junctionlens.remote-study-qualification.v1",
        "state": "ACCEPTED",
        "source_commit": source_commit,
        "source_partitions": ["model_training", "model_selection"],
        "internal_holdout_access_count": 0,
        "e0_model_manifest_sha256": _sha256_file(e0_final_root / "model-manifest.json"),
        "e1_study_report_sha256": _sha256_file(e1_report_path),
        "e1_outcome": e1_report["outcome"],
        "selected_experiment_id": e1_report["selected_experiment_id"],
        "power_simulation_manifest_sha256": power_receipt.manifest_sha256,
        "hardware_baseline_manifest_sha256": hardware_manifest_sha256,
        "freeze_evidence_manifest_sha256": freeze_receipt.manifest_sha256,
        "charter_sha256": _sha256_file(charter_path),
    }
    _atomic_json(output_root / "qualification.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-qualification-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--provider-assignment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    try:
        result = qualify(
            project_root=arguments.project_root,
            dataset_root=arguments.dataset_root,
            data_qualification_root=arguments.data_qualification_root,
            models_root=arguments.models_root,
            environment_path=arguments.environment,
            performance_path=arguments.performance,
            provider_path=arguments.provider_assignment,
            output_root=arguments.output_root,
            source_commit=arguments.source_commit,
        )
    except (OSError, StudyQualificationError, subprocess.SubprocessError, ValueError) as error:
        parser.exit(2, f"study qualification error: {error}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit portfolio-core evidence and render a narrowly measured demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
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
    load_json_object_path,
)


class CoreEvidenceError(RuntimeError):
    """Raised when a core claim lacks complete measured evidence."""


def _load(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return load_json_object_path(
            path,
            label,
            ParseLimits(max_bytes=64 * 1024 * 1024, max_depth=48, max_nodes=2_000_000),
        )
    except ParseBoundaryError as error:
        raise CoreEvidenceError(str(error)) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _accepted(value: Mapping[str, Any], *, field: str = "state") -> bool:
    return value.get(field) in {"ACCEPTED", "PASSED", "TRAINING_AND_SELECTION_ACCEPTED"}


def qualify(arguments: argparse.Namespace) -> Mapping[str, Any]:
    """Verify every portfolio-core result before rendering its measured summary."""
    data = _load(arguments.data / "qualification.json", "licensed-data qualification")
    micro = _load(arguments.micro, "micro-overfit qualification")
    models = _load(arguments.models / "qualification.json", "model qualification")
    study = _load(arguments.study / "qualification.json", "study qualification")
    parity = _load(arguments.parity, "CUDA parity qualification")
    performance = _load(arguments.performance, "accelerated performance qualification")
    profiler = _load(arguments.profiler, "profiler qualification")
    provider = _load(arguments.provider, "provider assignment")
    faults = _load(arguments.faults / "qualification.json", "fault qualification")
    visual_signoff_sha256 = arguments.visual_signoff_sha256
    if (
        data.get("mechanical_state") != "ACCEPTED"
        or data.get("segment_count") != 700
        or data.get("visual_audit_frame_count") != 12
        or len(visual_signoff_sha256) != 64
        or any(character not in "0123456789abcdef" for character in visual_signoff_sha256)
    ):
        raise CoreEvidenceError("licensed-data and visual-review evidence is incomplete")
    if (
        micro.get("status") != "PASSED"
        or micro.get("frames") != 32
        or not _accepted(models)
        or models.get("internal_holdout_access_count") != 0
        or not _accepted(study)
        or study.get("internal_holdout_access_count") != 0
    ):
        raise CoreEvidenceError("model training or pre-holdout study evidence is incomplete")
    if (
        parity.get("status") != "PASSED"
        or performance.get("status") != "PASSED"
        or profiler.get("status") != "PASSED"
        or provider.get("status") != "PASSED"
        or provider.get("provider_profile") != "cuda"
    ):
        raise CoreEvidenceError("accelerated runtime evidence is incomplete")
    if (
        faults.get("state") != "ACCEPTED"
        or faults.get("detection_rate") != 1.0
        or faults.get("case_count") != faults.get("detected_or_control_passed_count")
    ):
        raise CoreEvidenceError("fault-lab evidence is incomplete")
    e0_manifest = _load(arguments.study / "e0-final/model-manifest.json", "E0 model manifest")
    seed_metrics = e0_manifest.get("seed_metrics")
    if not isinstance(seed_metrics, list) or len(seed_metrics) != 3:
        raise CoreEvidenceError("E0 model manifest lacks the three-seed measurement table")
    inputs = {
        "licensed_data": arguments.data / "qualification.json",
        "micro_overfit": arguments.micro,
        "models": arguments.models / "qualification.json",
        "study": arguments.study / "qualification.json",
        "cuda_parity": arguments.parity,
        "performance": arguments.performance,
        "profiler": arguments.profiler,
        "provider": arguments.provider,
        "faults": arguments.faults / "qualification.json",
        "e0_model_manifest": arguments.study / "e0-final/model-manifest.json",
        "acceptance_charter": arguments.study / "acceptance-v1.yaml",
    }
    evidence = {
        "schema_version": "junctionlens.portfolio-core-evidence.v1",
        "state": "ACCEPTED",
        "source_commit": arguments.source_commit,
        "licensed_dataset": {
            "dataset_id": "openlane-v2-v2.1",
            "segment_count": 700,
            "visual_audit_frame_count": 12,
            "visual_signoff_sha256": visual_signoff_sha256,
        },
        "micro_overfit": {
            "frames": micro["frames"],
            "final_total_loss": micro["final_total_loss"],
            "checkpoint_sha256": micro["checkpoint_sha256"],
        },
        "e0_seed_metrics": seed_metrics,
        "e1_screening": {
            "outcome": study["e1_outcome"],
            "selected_experiment_id": study["selected_experiment_id"],
        },
        "runtime": {
            "provider_profile": provider["provider_profile"],
            "cuda_raw_maximum_absolute_error": parity["raw_maximum_absolute_error"],
            "cuda_graph_maximum_absolute_error": parity["graph_maximum_absolute_error"],
            "performance": performance,
            "profiled_benchmark_publishable": profiler["profiled_benchmark_publishable"],
        },
        "fault_lab": {
            "case_count": faults["case_count"],
            "detection_rate": faults["detection_rate"],
            "v1_release_acceptance_run": faults["v1_release_acceptance_run"],
        },
        "internal_holdout_access_count": 0,
        "input_sha256": {name: _sha256(path) for name, path in inputs.items()},
        "limitations": [
            "The model metrics are private model-selection measurements, not holdout results.",
            "The runtime benchmark uses the frozen synthetic qualification workload.",
            "The fault lab uses repository-owned synthetic inputs and is not a release run.",
            "Dataset-derived weights and licensed visual material are not publication artifacts.",
        ],
    }
    output_root = arguments.output.resolve(strict=False)
    if output_root.exists() or output_root.is_symlink():
        raise CoreEvidenceError("portfolio-core evidence output already exists")
    output_root.mkdir(parents=True)
    evidence_path = output_root / "core-evidence.json"
    evidence_path.write_bytes(canonical_json_bytes(evidence) + b"\n")
    lines = [
        "# Portfolio-core measured demonstration",
        "",
        "This evidence bundle passed the repository's private portfolio-core qualification.",
        (
            "It binds licensed-data audit, model-selection, CUDA parity, runtime, "
            "profiler, and fault-lab evidence to one source commit."
        ),
        "",
        "## Measured scope",
        "",
        (
            f"The licensed qualification covered {data['segment_count']} segments and "
            f"{data['visual_audit_frame_count']} signed visual-audit frames."
        ),
        (
            f"The M0 feasibility path overfit {micro['frames']} frames and reached final "
            f"total loss {float(micro['final_total_loss']):.6f}."
        ),
        (
            f"The E1 keep-gate outcome was `{study['e1_outcome']}`, selecting "
            f"`{study['selected_experiment_id']}` for the next stage."
        ),
        (
            "CUDA raw-output maximum absolute error was "
            f"{float(parity['raw_maximum_absolute_error']):.6g}."
        ),
        (
            "CUDA graph maximum absolute error was "
            f"{float(parity['graph_maximum_absolute_error']):.6g}."
        ),
        (
            f"All {faults['case_count']} seeded fault or invariant-control cases produced "
            "the intended result."
        ),
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in evidence["limitations"])
    lines.extend(
        (
            "",
            "The public synthetic demo remains the distributable UI path.",
            "Licensed inputs, predictions, and model weights remain private.",
            "",
        )
    )
    (output_root / "CORE_DEMO.md").write_text("\n".join(lines), encoding="utf-8")
    receipt = {
        "schema_version": "junctionlens.portfolio-core-receipt.v1",
        "state": "ACCEPTED",
        "core_evidence_sha256": _sha256(evidence_path),
        "core_demo_sha256": _sha256(output_root / "CORE_DEMO.md"),
        "source_commit": arguments.source_commit,
    }
    (output_root / "qualification.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--micro", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--profiler", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--faults", type=Path, required=True)
    parser.add_argument("--visual-signoff-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        result = qualify(_parse_args())
    except (CoreEvidenceError, OSError, ValueError) as error:
        print(f"core evidence error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

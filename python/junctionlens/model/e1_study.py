"""Frozen E0 versus E1 model-selection study and keep-gate decision."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from junctionlens.model.e0_profile import E0Profile
from junctionlens.model.e0_training import _hash_file
from junctionlens.model.e1_profile import E1Profile
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_json_object_path


class E1StudyError(RuntimeError):
    """Raised when E1 study evidence is invalid, contaminated, or inconsistent."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StudyMetrics(_Frozen):
    DET_l: float = Field(ge=0.0, le=1.0)
    DET_t: float = Field(ge=0.0, le=1.0)
    TOP_lt: float = Field(ge=0.0, le=1.0)
    wrong_control_assignment_rate: float = Field(ge=0.0, le=1.0)
    official_composite: float = Field(ge=0.0, le=1.0)
    negative_log_likelihood: float = Field(ge=0.0)


class StudyArm(_Frozen):
    experiment_id: Literal["E0-independent", "E1-joint"]
    seed: Literal[20260813]
    checkpoint_sha256: str
    selection_receipt_sha256: str
    prediction_manifest_sha256: str
    evaluation_artifact_sha256: str
    metrics: StudyMetrics

    @model_validator(mode="after")
    def validate_hashes(self) -> StudyArm:
        for label, value in (
            ("checkpoint", self.checkpoint_sha256),
            ("selection receipt", self.selection_receipt_sha256),
            ("prediction manifest", self.prediction_manifest_sha256),
            ("evaluation artifact", self.evaluation_artifact_sha256),
        ):
            _validate_sha256(value, f"E1 study {label}")
        return self


class EvaluatorIdentity(_Frozen):
    official_implementation: Literal["OpenLane-V2-v2.1"]
    official_version: str = Field(min_length=1)
    official_config_sha256: str
    custom_match_version: Literal["CustomMatchV1"]
    custom_match_config_sha256: str

    @model_validator(mode="after")
    def validate_hashes(self) -> EvaluatorIdentity:
        _validate_sha256(self.official_config_sha256, "official evaluator config")
        _validate_sha256(self.custom_match_config_sha256, "custom match config")
        return self


class E1StudyEvidence(_Frozen):
    schema_version: Literal["junctionlens.e1-study-evidence.v1"]
    study_id: Literal["E0-independent-vs-E1-joint"]
    source_partition: Literal["model_selection"]
    internal_holdout_access_count: Literal[0]
    selection_split_manifest_sha256: str
    source_frame_manifest_sha256: str
    evaluator: EvaluatorIdentity
    baseline: StudyArm
    candidate: StudyArm

    @model_validator(mode="after")
    def validate_contract(self) -> E1StudyEvidence:
        _validate_sha256(self.selection_split_manifest_sha256, "selection split manifest")
        _validate_sha256(self.source_frame_manifest_sha256, "source frame manifest")
        if self.baseline.experiment_id != "E0-independent":
            raise ValueError("E1 study baseline must be E0-independent")
        if self.candidate.experiment_id != "E1-joint":
            raise ValueError("E1 study candidate must be E1-joint")
        return self


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return load_json_object_path(
            path,
            label,
            ParseLimits(
                max_bytes=16 * 1024 * 1024,
                max_depth=24,
                max_nodes=500_000,
                max_container_items=100_000,
            ),
        )
    except ParseBoundaryError as error:
        raise E1StudyError(str(error)) from error


def load_e1_study_evidence(path: Path) -> E1StudyEvidence:
    try:
        return E1StudyEvidence.model_validate(_load_json(path, "E1 study evidence"))
    except ValueError as error:
        raise E1StudyError(f"E1 study evidence failed validation: {error}") from error


def _validate_run(
    root: Path,
    *,
    experiment_id: str,
    profile_hash: str,
    arm: StudyArm,
    evidence: E1StudyEvidence,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest_path = root / "run-manifest.json"
    selection_path = root / "selection-receipt.json"
    manifest = _load_json(manifest_path, f"{experiment_id} run manifest")
    selection = _load_json(selection_path, f"{experiment_id} selection receipt")
    if (
        manifest.get("state") != "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION"
        or manifest.get("experiment_id") != experiment_id
        or manifest.get("seed") != 20260813
    ):
        raise E1StudyError(f"{experiment_id} run identity or state is invalid")
    profile_key = "profile_sha256" if experiment_id == "E0-independent" else "e1_profile_sha256"
    if manifest.get(profile_key) != profile_hash:
        raise E1StudyError(f"{experiment_id} profile identity differs from the frozen profile")
    if manifest.get("source_frame_manifest_sha256") != evidence.source_frame_manifest_sha256:
        raise E1StudyError(f"{experiment_id} used a different source frame manifest")
    if (
        selection.get("state") != "SELECTED_ON_MODEL_SELECTION"
        or selection.get("selection_split_manifest_sha256")
        != evidence.selection_split_manifest_sha256
    ):
        raise E1StudyError(f"{experiment_id} selection split or state is invalid")
    selected = selection.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("epoch"), int):
        raise E1StudyError(f"{experiment_id} selected checkpoint metadata is invalid")
    checkpoint_path = root / "checkpoints" / f"epoch-{int(selected['epoch']):02d}.pt"
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise E1StudyError(f"{experiment_id} selected checkpoint is missing")
    checkpoint_sha256 = _hash_file(checkpoint_path)
    if (
        checkpoint_sha256 != selection.get("checkpoint_sha256")
        or checkpoint_sha256 != arm.checkpoint_sha256
    ):
        raise E1StudyError(f"{experiment_id} selected checkpoint failed identity verification")
    if _hash_file(selection_path) != arm.selection_receipt_sha256:
        raise E1StudyError(f"{experiment_id} selection receipt hash differs from evidence")
    return manifest, selection


def _gate(
    name: str,
    baseline: float,
    candidate: float,
    threshold: Decimal,
    comparison: Literal["delta-gte", "candidate-lt"],
) -> Mapping[str, object]:
    base_decimal = Decimal(str(baseline))
    candidate_decimal = Decimal(str(candidate))
    delta = candidate_decimal - base_decimal
    passed = delta >= threshold if comparison == "delta-gte" else candidate_decimal < base_decimal
    return {
        "name": name,
        "baseline": baseline,
        "candidate": candidate,
        "delta": float(delta),
        "comparison": comparison,
        "threshold": float(threshold),
        "passed": passed,
        "reason_code": f"E1_{name}_{'PASSED' if passed else 'FAILED'}",
    }


def finalize_e1_study(
    base: E0Profile,
    profile: E1Profile,
    baseline_run_root: Path,
    candidate_run_root: Path,
    evidence_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Validate isolated evidence and persist the immutable E1 promotion outcome."""
    profile.validate_base(base)
    evidence = load_e1_study_evidence(evidence_path)
    baseline_manifest, _ = _validate_run(
        baseline_run_root,
        experiment_id="E0-independent",
        profile_hash=base.canonical_sha256(),
        arm=evidence.baseline,
        evidence=evidence,
    )
    candidate_manifest, _ = _validate_run(
        candidate_run_root,
        experiment_id="E1-joint",
        profile_hash=profile.canonical_sha256(),
        arm=evidence.candidate,
        evidence=evidence,
    )
    if candidate_manifest.get("base_profile_sha256") != base.canonical_sha256():
        raise E1StudyError("E1 run was trained against a different E0 base profile")
    if baseline_manifest.get("training_split_manifest_sha256") != candidate_manifest.get(
        "training_split_manifest_sha256"
    ) or baseline_manifest.get("source_dataset_manifest_sha256") != candidate_manifest.get(
        "source_dataset_manifest_sha256"
    ):
        raise E1StudyError("E0 and E1 were not trained from the same frozen corpus split")
    baseline_metrics = evidence.baseline.metrics
    candidate_metrics = evidence.candidate.metrics
    gates = [
        _gate(
            "TOP_LT_IMPROVEMENT",
            baseline_metrics.TOP_lt,
            candidate_metrics.TOP_lt,
            Decimal("0.02"),
            "delta-gte",
        ),
        _gate(
            "DET_L_FLOOR",
            baseline_metrics.DET_l,
            candidate_metrics.DET_l,
            Decimal("-0.01"),
            "delta-gte",
        ),
        _gate(
            "DET_T_FLOOR",
            baseline_metrics.DET_t,
            candidate_metrics.DET_t,
            Decimal("-0.01"),
            "delta-gte",
        ),
        _gate(
            "WRONG_CONTROL_REDUCTION",
            baseline_metrics.wrong_control_assignment_rate,
            candidate_metrics.wrong_control_assignment_rate,
            Decimal("0"),
            "candidate-lt",
        ),
    ]
    promoted = all(bool(gate["passed"]) for gate in gates)
    outcome = "PROMOTED" if promoted else "REJECTED_BY_KEEP_GATE"
    selected_experiment = "E1-joint" if promoted else "E0-independent"
    report: dict[str, Any] = {
        "schema_version": "junctionlens.e1-study-report.v1",
        "state": "ACCEPTED",
        "study_validity": "ACCEPTED",
        "outcome": outcome,
        "selected_experiment_id": selected_experiment,
        "study_id": evidence.study_id,
        "source_partition": evidence.source_partition,
        "internal_holdout_access_count": evidence.internal_holdout_access_count,
        "selection_split_manifest_sha256": evidence.selection_split_manifest_sha256,
        "source_frame_manifest_sha256": evidence.source_frame_manifest_sha256,
        "base_profile_sha256": base.canonical_sha256(),
        "e1_profile_sha256": profile.canonical_sha256(),
        "evaluator": evidence.evaluator.model_dump(mode="json"),
        "baseline": evidence.baseline.model_dump(mode="json"),
        "candidate": evidence.candidate.model_dump(mode="json"),
        "keep_gates": gates,
        "reason_codes": [str(gate["reason_code"]) for gate in gates],
        "evidence_sha256": _hash_file(evidence_path),
    }
    serialized = canonical_json_bytes(report) + b"\n"
    if output_path.exists() or output_path.is_symlink():
        raise E1StudyError("refusing to replace an existing E1 study report")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=".e1-study-", delete=False
    ) as destination:
        destination.write(serialized)
        destination.flush()
        os.fsync(destination.fileno())
        temporary = Path(destination.name)
    temporary.replace(output_path)
    return report


__all__ = [
    "E1StudyError",
    "E1StudyEvidence",
    "StudyArm",
    "StudyMetrics",
    "finalize_e1_study",
    "load_e1_study_evidence",
]

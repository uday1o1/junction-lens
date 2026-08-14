"""Persisted deterministic release decisions from a frozen charter."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from junctionlens.gate.bootstrap import (
    BootstrapError,
    BootstrapResult,
    paired_segment_bootstrap,
    paired_trial_bootstrap,
)
from junctionlens.gate.charter import CharterCell, CharterError, FrozenCharter, load_frozen_charter
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object_path,
)

ReleaseStatus = Literal[
    "PASS",
    "FAIL_INTEGRITY",
    "FAIL_REGRESSION",
    "FAIL_PERFORMANCE",
    "INSUFFICIENT_EVIDENCE",
    "BLOCKED_INFRASTRUCTURE",
]


class DecisionError(RuntimeError):
    """Raised when release evidence is malformed or does not match its charter."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntegrityEvidence(_Strict):
    artifact_integrity: bool
    baseline_evaluator_image_digest: str
    candidate_evaluator_image_digest: str
    baseline_data_manifest_sha256: str
    candidate_data_manifest_sha256: str
    baseline_split_manifest_sha256: str
    candidate_split_manifest_sha256: str
    baseline_preprocessing_sha256: str
    candidate_preprocessing_sha256: str
    baseline_postprocessing_sha256: str
    candidate_postprocessing_sha256: str
    metric_registry_sha256: str
    slice_registry_sha256: str
    calibration_ranks_frozen: bool
    candidate_training_holdout_access_count: int = Field(ge=0)
    frame_sets_match: bool
    slice_values_match: bool
    support_values_match: bool
    schema_major_compatible: bool
    identifiers_valid: bool
    coordinate_metadata_valid: bool
    required_values_finite: bool
    calibrator_valid: bool
    evaluator_compatible: bool
    provenance_complete: bool
    leakage_free: bool
    partial_inference_approved: bool
    provider_fallback_free: bool
    registry_inputs_match: bool


class SupportEvidence(_Strict):
    paired_segments: int = Field(ge=0)
    eligible_ground_truth_edges: int = Field(ge=0)
    adjacent_frame_transitions: int = Field(ge=0)
    temporal_segments: int = Field(ge=0)


class CellEvidence(_Strict):
    estimator: Literal["ratio", "average_precision", "adaptive_ece", "paired_runtime"]
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    support: SupportEvidence
    counterexample_query: str

    @model_validator(mode="after")
    def validate_query(self) -> CellEvidence:
        if not self.counterexample_query or len(self.counterexample_query) > 500:
            raise ValueError("counterexample query is invalid")
        return self


class RuntimeEvidence(_Strict):
    baseline_hardware_manifest_sha256: str
    hardware_baseline_manifest_sha256: str
    baseline_gpu_provider_active: bool
    gpu_provider_active: bool
    throughput_per_second: float
    p95_latency_ms: float
    p99_latency_ms: float
    peak_device_memory_bytes: int = Field(ge=0)
    long_run_frames: int = Field(ge=0)
    unbounded_memory_growth: bool
    unexpected_cpu_provider_nodes: int = Field(ge=0)
    warmup_frames_per_block: int = Field(ge=0)
    measured_frames_per_block: int = Field(ge=0)
    trial_blocks: int = Field(ge=0)
    trial_block_order: tuple[Literal["AB", "BA"], ...]
    baseline_warmup_frames_per_block: int = Field(ge=0)
    baseline_measured_frames_per_block: int = Field(ge=0)
    baseline_trial_block_order: tuple[Literal["AB", "BA"], ...]
    baseline_environment_valid: bool
    paired_trial_ids_match: bool
    environment_valid: bool

    @model_validator(mode="after")
    def validate_numbers(self) -> RuntimeEvidence:
        for value in (self.throughput_per_second, self.p95_latency_ms, self.p99_latency_ms):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("runtime evidence contains an invalid measurement")
        return self


class GateEvidence(_Strict):
    schema_version: Literal["junctionlens.gate-evidence.v1"]
    charter_sha256: str
    integrity: IntegrityEvidence
    cells: dict[str, CellEvidence]
    runtime: RuntimeEvidence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        return load_json_object_path(
            path,
            "gate evidence",
            ParseLimits(
                max_bytes=128 * 1024 * 1024,
                max_depth=32,
                max_nodes=2_000_000,
                max_container_items=1_000_000,
                max_string_bytes=4 * 1024 * 1024,
            ),
        )
    except ParseBoundaryError as error:
        raise DecisionError(str(error)) from error


def load_gate_evidence(path: Path) -> GateEvidence:
    try:
        return GateEvidence.model_validate(_load_json(path))
    except ValueError as error:
        raise DecisionError(f"gate evidence schema is invalid: {error}") from error


def _cell_record(
    cell: CharterCell,
    *,
    status: ReleaseStatus,
    reason_code: str,
    support: SupportEvidence,
    result: BootstrapResult | None,
    query: str,
) -> dict[str, Any]:
    return {
        "cell_id": cell.id,
        "metric": cell.metric,
        "slice": cell.slice,
        "status": status,
        "reason_code": reason_code,
        "support": support.model_dump(mode="json"),
        "point_estimate": None if result is None else result.point_estimate,
        "interval": {
            "lower": None if result is None else result.lower,
            "upper": None if result is None else result.upper,
            "adjusted_two_sided_alpha": (None if result is None else result.interval_alpha),
        },
        "margin": cell.margin,
        "finite_replicates": None if result is None else result.finite_replicates,
        "invalid_replicates": None if result is None else result.invalid_replicates,
        "counterexample_query": query,
    }


def _support_failure(
    cell: CharterCell, support: SupportEvidence, charter: FrozenCharter
) -> str | None:
    if (
        cell.support == "overall"
        and support.paired_segments < charter.support.overall_paired_segments
    ):
        return "GATE_INSUFFICIENT_OVERALL_SEGMENTS"
    if (
        cell.support == "slice"
        and support.paired_segments < charter.support.gating_slice_paired_segments
    ):
        return "GATE_INSUFFICIENT_SLICE_SEGMENTS"
    if cell.support == "lane_control":
        if support.paired_segments < charter.support.overall_paired_segments:
            return "GATE_INSUFFICIENT_OVERALL_SEGMENTS"
        if support.eligible_ground_truth_edges < charter.support.lane_control_ground_truth_edges:
            return "GATE_INSUFFICIENT_LANE_CONTROL_EDGES"
    if cell.support == "temporal":
        if support.adjacent_frame_transitions < charter.support.temporal_adjacent_frame_transitions:
            return "GATE_INSUFFICIENT_TEMPORAL_TRANSITIONS"
        if support.temporal_segments < charter.support.temporal_segments:
            return "GATE_INSUFFICIENT_TEMPORAL_SEGMENTS"
    return None


def _validate_paired_segment_count(cell: CharterCell, evidence: CellEvidence) -> None:
    if cell.estimator == "paired_runtime":
        return
    baseline_segments = evidence.baseline.get("segments")
    candidate_segments = evidence.candidate.get("segments")
    if not isinstance(baseline_segments, list) or not isinstance(candidate_segments, list):
        raise DecisionError(f"segment evidence is missing for {cell.id}")
    if evidence.support.paired_segments != len(baseline_segments):
        raise DecisionError(f"declared paired segment support is inconsistent for {cell.id}")
    if len(candidate_segments) != len(baseline_segments):
        raise DecisionError(f"baseline and candidate segment counts differ for {cell.id}")


def _integrity_failures(evidence: GateEvidence, charter: FrozenCharter) -> list[str]:
    integrity = evidence.integrity
    failures = []
    if not integrity.artifact_integrity:
        failures.append("GATE_INTEGRITY_ARTIFACT_HASH_MISMATCH")
    if integrity.baseline_evaluator_image_digest != integrity.candidate_evaluator_image_digest:
        failures.append("GATE_INTEGRITY_EVALUATOR_MISMATCH")
    if integrity.baseline_data_manifest_sha256 != integrity.candidate_data_manifest_sha256:
        failures.append("GATE_INTEGRITY_DATA_MANIFEST_MISMATCH")
    if integrity.baseline_split_manifest_sha256 != integrity.candidate_split_manifest_sha256:
        failures.append("GATE_INTEGRITY_SPLIT_MANIFEST_MISMATCH")
    if integrity.baseline_preprocessing_sha256 != integrity.candidate_preprocessing_sha256:
        failures.append("GATE_INTEGRITY_PREPROCESSING_MISMATCH")
    if integrity.baseline_postprocessing_sha256 != integrity.candidate_postprocessing_sha256:
        failures.append("GATE_INTEGRITY_POSTPROCESSING_MISMATCH")
    if integrity.metric_registry_sha256 != charter.metric_registry_sha256:
        failures.append("GATE_INTEGRITY_METRIC_REGISTRY_MISMATCH")
    if integrity.slice_registry_sha256 != charter.slice_registry_sha256:
        failures.append("GATE_INTEGRITY_SLICE_REGISTRY_MISMATCH")
    if not integrity.calibration_ranks_frozen:
        failures.append("GATE_INTEGRITY_CALIBRATION_RANK_CHANGED")
    if integrity.candidate_training_holdout_access_count != 0:
        failures.append("GATE_INTEGRITY_HOLDOUT_USED_FOR_TRAINING")
    for valid, reason in (
        (integrity.frame_sets_match, "GATE_INTEGRITY_FRAME_SET_MISMATCH"),
        (integrity.slice_values_match, "GATE_INTEGRITY_SLICE_VALUES_MISMATCH"),
        (integrity.support_values_match, "GATE_INTEGRITY_SUPPORT_MISMATCH"),
        (integrity.schema_major_compatible, "GATE_INTEGRITY_SCHEMA_MAJOR_MISMATCH"),
        (integrity.identifiers_valid, "GATE_INTEGRITY_IDENTIFIERS_INVALID"),
        (integrity.coordinate_metadata_valid, "GATE_INTEGRITY_COORDINATE_METADATA_INVALID"),
        (integrity.required_values_finite, "GATE_INTEGRITY_NONFINITE_REQUIRED_VALUE"),
        (integrity.calibrator_valid, "GATE_INTEGRITY_CALIBRATOR_INVALID"),
        (integrity.evaluator_compatible, "GATE_INTEGRITY_EVALUATOR_COMPATIBILITY_FAILED"),
        (integrity.provenance_complete, "GATE_INTEGRITY_PROVENANCE_INCOMPLETE"),
        (integrity.leakage_free, "GATE_INTEGRITY_DATA_LEAKAGE"),
        (integrity.partial_inference_approved, "GATE_INTEGRITY_PARTIAL_INFERENCE"),
        (integrity.provider_fallback_free, "GATE_INTEGRITY_PROVIDER_FALLBACK"),
        (integrity.registry_inputs_match, "GATE_INTEGRITY_ARM_REGISTRY_MISMATCH"),
    ):
        if not valid:
            failures.append(reason)
    return failures


def _infrastructure_failures(evidence: GateEvidence, charter: FrozenCharter) -> list[str]:
    runtime = evidence.runtime
    failures = []
    if (
        runtime.hardware_baseline_manifest_sha256
        != charter.freeze_evidence.m0_hardware_baseline_manifest_sha256
        or runtime.baseline_hardware_manifest_sha256
        != charter.freeze_evidence.m0_hardware_baseline_manifest_sha256
        or runtime.baseline_hardware_manifest_sha256 != runtime.hardware_baseline_manifest_sha256
    ):
        failures.append("GATE_INFRASTRUCTURE_HARDWARE_MISMATCH")
    if not runtime.environment_valid or not runtime.baseline_environment_valid:
        failures.append("GATE_INFRASTRUCTURE_CONTAMINATED_RUN")
    if not runtime.gpu_provider_active or not runtime.baseline_gpu_provider_active:
        failures.append("GATE_INFRASTRUCTURE_GPU_PROVIDER_UNAVAILABLE")
    if (
        runtime.warmup_frames_per_block != 200
        or runtime.measured_frames_per_block != 2000
        or runtime.baseline_warmup_frames_per_block != 200
        or runtime.baseline_measured_frames_per_block != 2000
    ):
        failures.append("GATE_INFRASTRUCTURE_RUNTIME_PROTOCOL_MISMATCH")
    if runtime.trial_blocks != 10:
        failures.append("GATE_INFRASTRUCTURE_TRIAL_BLOCK_COUNT_MISMATCH")
    if (
        runtime.trial_block_order != charter.runtime_bootstrap.balanced_order_schedule
        or runtime.baseline_trial_block_order != charter.runtime_bootstrap.balanced_order_schedule
    ):
        failures.append("GATE_INFRASTRUCTURE_TRIAL_ORDER_MISMATCH")
    if not runtime.paired_trial_ids_match:
        failures.append("GATE_INFRASTRUCTURE_TRIAL_PAIR_MISMATCH")
    return failures


def _absolute_performance_failures(evidence: GateEvidence, charter: FrozenCharter) -> list[str]:
    runtime = evidence.runtime
    policy = charter.absolute_runtime
    failures = []
    if runtime.throughput_per_second < policy.throughput_per_second_minimum:
        failures.append("GATE_PERFORMANCE_THROUGHPUT_BUDGET")
    if runtime.p95_latency_ms > policy.p95_latency_ms_maximum:
        failures.append("GATE_PERFORMANCE_P95_LATENCY_BUDGET")
    if runtime.p99_latency_ms > policy.p99_latency_ms_maximum:
        failures.append("GATE_PERFORMANCE_P99_LATENCY_BUDGET")
    if runtime.peak_device_memory_bytes > policy.peak_device_memory_bytes_maximum:
        failures.append("GATE_PERFORMANCE_DEVICE_MEMORY_BUDGET")
    if runtime.long_run_frames < policy.long_run_frames:
        failures.append("GATE_PERFORMANCE_LONG_RUN_INCOMPLETE")
    if runtime.unbounded_memory_growth:
        failures.append("GATE_PERFORMANCE_UNBOUNDED_MEMORY_GROWTH")
    if runtime.unexpected_cpu_provider_nodes > policy.unexpected_cpu_provider_nodes_maximum:
        failures.append("GATE_PERFORMANCE_UNEXPECTED_CPU_PROVIDER")
    return failures


def _decision_from_interval(result: BootstrapResult, margin: float) -> tuple[ReleaseStatus, str]:
    if result.status != "VALID" or result.lower is None or result.upper is None:
        return "INSUFFICIENT_EVIDENCE", "GATE_INSUFFICIENT_FINITE_REPLICATES"
    boundary = -margin
    scale = max(1.0, abs(boundary), abs(result.lower), abs(result.upper))
    tolerance = math.ulp(scale) * 64.0
    if result.lower >= boundary or math.isclose(
        result.lower, boundary, rel_tol=0.0, abs_tol=tolerance
    ):
        return "PASS", "GATE_CELL_ACCEPTED"
    if result.upper < boundary and not math.isclose(
        result.upper, boundary, rel_tol=0.0, abs_tol=tolerance
    ):
        return "FAIL_REGRESSION", "GATE_REGRESSION_CI_BELOW_MARGIN"
    return "INSUFFICIENT_EVIDENCE", "GATE_INSUFFICIENT_INTERVAL_CROSSES_MARGIN"


def _overall_status(statuses: list[ReleaseStatus]) -> ReleaseStatus:
    precedence: tuple[ReleaseStatus, ...] = (
        "FAIL_INTEGRITY",
        "BLOCKED_INFRASTRUCTURE",
        "FAIL_PERFORMANCE",
        "FAIL_REGRESSION",
        "INSUFFICIENT_EVIDENCE",
        "PASS",
    )
    return next(status for status in precedence if status in statuses)


def decide_release(charter_path: Path, evidence_path: Path) -> Mapping[str, Any]:
    """Evaluate every frozen cell and return the persisted decision body."""
    try:
        charter = load_frozen_charter(charter_path)
    except CharterError as error:
        raise DecisionError(str(error)) from error
    evidence = load_gate_evidence(evidence_path)
    if evidence.charter_sha256 != charter.charter_sha256:
        raise DecisionError("gate evidence references a different frozen charter")
    expected_cells = {cell.id for cell in charter.cells}
    if set(evidence.cells) != expected_cells:
        raise DecisionError("gate evidence does not cover the exact frozen cell set")
    for cell in charter.cells:
        if evidence.cells[cell.id].estimator != cell.estimator:
            raise DecisionError(f"gate evidence estimator mismatch for {cell.id}")
        _validate_paired_segment_count(cell, evidence.cells[cell.id])

    integrity_failures = _integrity_failures(evidence, charter)
    infrastructure_failures = _infrastructure_failures(evidence, charter)
    gating_cells = len(charter.cells)
    records: list[dict[str, Any]] = []
    statuses: list[ReleaseStatus] = []
    results: dict[str, BootstrapResult] = {}
    for cell in charter.cells:
        item = evidence.cells[cell.id]
        if integrity_failures:
            status: ReleaseStatus = "FAIL_INTEGRITY"
            records.append(
                _cell_record(
                    cell,
                    status=status,
                    reason_code="GATE_CELL_NOT_EVALUATED_INTEGRITY",
                    support=item.support,
                    result=None,
                    query=item.counterexample_query,
                )
            )
            statuses.append(status)
            continue
        if cell.support == "runtime" and infrastructure_failures:
            status = "BLOCKED_INFRASTRUCTURE"
            records.append(
                _cell_record(
                    cell,
                    status=status,
                    reason_code=infrastructure_failures[0],
                    support=item.support,
                    result=None,
                    query=item.counterexample_query,
                )
            )
            statuses.append(status)
            continue
        support_reason = _support_failure(cell, item.support, charter)
        if support_reason is not None:
            status = "INSUFFICIENT_EVIDENCE"
            records.append(
                _cell_record(
                    cell,
                    status=status,
                    reason_code=support_reason,
                    support=item.support,
                    result=None,
                    query=item.counterexample_query,
                )
            )
            statuses.append(status)
            continue
        try:
            if cell.estimator == "paired_runtime":
                result = paired_trial_bootstrap(
                    item.baseline,
                    item.candidate,
                    direction=cell.direction,
                    family_alpha=charter.family_alpha,
                    gating_cells=gating_cells,
                    replicates=charter.runtime_bootstrap.replicates,
                    seed=charter.runtime_bootstrap.seed,
                    minimum_valid_pairs=cast(int, charter.runtime_bootstrap.minimum_valid_pairs),
                )
                if result.status != "VALID":
                    status = "BLOCKED_INFRASTRUCTURE"
                    reason = "GATE_INFRASTRUCTURE_INSUFFICIENT_VALID_TRIAL_PAIRS"
                else:
                    status, reason = _decision_from_interval(result, cell.margin)
                    if status == "FAIL_REGRESSION":
                        status = "FAIL_PERFORMANCE"
                        reason = "GATE_PERFORMANCE_CI_BELOW_MARGIN"
            else:
                result = paired_segment_bootstrap(
                    item.baseline,
                    item.candidate,
                    estimator=cell.estimator,
                    direction=cell.direction,
                    family_alpha=charter.family_alpha,
                    gating_cells=gating_cells,
                    replicates=charter.bootstrap.replicates,
                    seed=charter.bootstrap.seed,
                    minimum_finite_replicates=cast(
                        int, charter.bootstrap.minimum_finite_replicates
                    ),
                )
                status, reason = _decision_from_interval(result, cell.margin)
        except BootstrapError as error:
            raise DecisionError(f"invalid primitive evidence for {cell.id}: {error}") from error
        results[cell.id] = result
        records.append(
            _cell_record(
                cell,
                status=status,
                reason_code=reason,
                support=item.support,
                result=result,
                query=item.counterexample_query,
            )
        )
        statuses.append(status)

    absolute_failures = (
        []
        if integrity_failures or infrastructure_failures
        else _absolute_performance_failures(evidence, charter)
    )
    statuses.extend("FAIL_INTEGRITY" for _ in integrity_failures)
    statuses.extend("BLOCKED_INFRASTRUCTURE" for _ in infrastructure_failures)
    statuses.extend("FAIL_PERFORMANCE" for _ in absolute_failures)

    hypothesis_records = []
    for hypothesis in charter.primary_hypotheses:
        hypothesis_result = results.get(hypothesis.cell_id)
        if (
            hypothesis_result is None
            or hypothesis_result.lower is None
            or hypothesis_result.upper is None
        ):
            hypothesis_status: ReleaseStatus = "INSUFFICIENT_EVIDENCE"
            hypothesis_reason = "GATE_HYPOTHESIS_NOT_EVALUABLE"
        elif hypothesis_result.lower >= hypothesis.minimum_improvement:
            hypothesis_status = "PASS"
            hypothesis_reason = "GATE_HYPOTHESIS_ACCEPTED"
        elif hypothesis_result.upper < hypothesis.minimum_improvement:
            hypothesis_status = "FAIL_REGRESSION"
            hypothesis_reason = "GATE_HYPOTHESIS_THRESHOLD_NOT_MET"
        else:
            hypothesis_status = "INSUFFICIENT_EVIDENCE"
            hypothesis_reason = "GATE_HYPOTHESIS_INTERVAL_CROSSES_THRESHOLD"
        hypothesis_records.append(
            {
                "hypothesis_id": hypothesis.id,
                "cell_id": hypothesis.cell_id,
                "status": hypothesis_status,
                "reason_code": hypothesis_reason,
                "minimum_improvement": hypothesis.minimum_improvement,
                "interval": {
                    "lower": None if hypothesis_result is None else hypothesis_result.lower,
                    "upper": None if hypothesis_result is None else hypothesis_result.upper,
                },
            }
        )
        statuses.append(hypothesis_status)

    body: dict[str, Any] = {
        "schema_version": "junctionlens.gate-decision.v1",
        "status": _overall_status(statuses),
        "charter_sha256": charter.charter_sha256,
        "evidence_sha256": _sha256_file(evidence_path),
        "bootstrap": {
            "algorithm": charter.bootstrap.algorithm,
            "replicates": charter.bootstrap.replicates,
            "seed": charter.bootstrap.seed,
            "interval_method": "type7-percentile-bonferroni-two-sided",
            "gating_cells": gating_cells,
        },
        "integrity_reason_codes": integrity_failures,
        "infrastructure_reason_codes": infrastructure_failures,
        "performance_reason_codes": absolute_failures,
        "cells": records,
        "primary_hypotheses": hypothesis_records,
    }
    body["decision_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def persist_decision(
    charter_path: Path, evidence_path: Path, output_path: Path
) -> Mapping[str, Any]:
    """Write one immutable decision and refuse replacement or client recalculation."""
    if output_path.exists() or output_path.is_symlink():
        raise DecisionError("gate decision output already exists")
    body = decide_release(charter_path, evidence_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=".decision-", delete=False
    ) as temporary:
        temporary.write(canonical_json_bytes(body) + b"\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o444)
    temporary_path.replace(output_path)
    return body


__all__ = ["DecisionError", "decide_release", "load_gate_evidence", "persist_decision"]

"""Exact-frame comparison materialization for immutable release evidence."""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from junctionlens.gate.charter import CharterCell, FrozenCharter, load_frozen_charter
from junctionlens.gate.decision import GateEvidence, decide_release
from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import RegistryError, canonical_json_bytes
from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object,
    load_yaml_object_path,
    read_bounded_file,
)

_MAX_ARM_BYTES = 512 * 1024 * 1024
_SHA256_LENGTH = 64


class ComparisonError(RuntimeError):
    """Raised when comparison inputs cannot produce trustworthy evidence."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArmIntegrity(_Strict):
    artifact_integrity: bool
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
    calibration_ranks_frozen: bool
    training_holdout_access_count: int = Field(ge=0)


class RuntimeBlock(_Strict):
    block_id: str = Field(min_length=1, max_length=128)
    value: float
    valid: bool

    @model_validator(mode="after")
    def validate_value(self) -> RuntimeBlock:
        if not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError("runtime block value must be finite and nonnegative")
        return self


class RuntimeArm(_Strict):
    hardware_baseline_manifest_sha256: str
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
    trial_block_order: tuple[Literal["AB", "BA"], ...]
    environment_valid: bool
    metrics: dict[str, tuple[RuntimeBlock, ...]]

    @model_validator(mode="after")
    def validate_runtime(self) -> RuntimeArm:
        _sha256(self.hardware_baseline_manifest_sha256, "hardware baseline manifest")
        for label, value in (
            ("throughput", self.throughput_per_second),
            ("P95 latency", self.p95_latency_ms),
            ("P99 latency", self.p99_latency_ms),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and nonnegative")
        for metric, blocks in self.metrics.items():
            if not metric or len(metric) > 128 or not blocks or len(blocks) > 10_000:
                raise ValueError("runtime metric identity and blocks must be nonempty")
            identifiers = [block.block_id for block in blocks]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"runtime metric {metric} has duplicate block IDs")
        return self


class FrameEvidence(_Strict):
    frame_token: str = Field(min_length=1, max_length=256)
    segment_id: str = Field(min_length=1, max_length=256)
    timestamp_ns: int = Field(ge=0)
    slice_values: dict[str, str]
    metrics: dict[str, dict[str, Any]]

    @model_validator(mode="after")
    def validate_fields(self) -> FrameEvidence:
        if any(not key or len(key) > 128 for key in self.slice_values):
            raise ValueError("slice dimensions must be bounded and nonempty")
        if any(not value or len(value) > 256 for value in self.slice_values.values()):
            raise ValueError("slice values must be bounded and nonempty")
        if any(not key or len(key) > 128 for key in self.metrics):
            raise ValueError("metric identities must be bounded and nonempty")
        _finite_tree(self.metrics, f"frame {self.frame_token} metric primitives")
        return self


class ComparisonArm(_Strict):
    schema_version: Literal["junctionlens.comparison-arm.v1"]
    arm_id: str = Field(min_length=1, max_length=128)
    evaluator_image_digest: str = Field(min_length=1, max_length=256)
    data_manifest_sha256: str
    split_manifest_sha256: str
    preprocessing_sha256: str
    postprocessing_sha256: str
    metric_registry_sha256: str
    slice_registry_sha256: str
    integrity: ArmIntegrity
    frames: tuple[FrameEvidence, ...]
    runtime: RuntimeArm

    @model_validator(mode="after")
    def validate_identity(self) -> ComparisonArm:
        for label, value in (
            ("data manifest", self.data_manifest_sha256),
            ("split manifest", self.split_manifest_sha256),
            ("preprocessing", self.preprocessing_sha256),
            ("postprocessing", self.postprocessing_sha256),
            ("metric registry", self.metric_registry_sha256),
            ("slice registry", self.slice_registry_sha256),
        ):
            _sha256(value, label)
        tokens = [frame.frame_token for frame in self.frames]
        if len(tokens) != len(set(tokens)):
            raise ValueError("comparison arm contains duplicate frame tokens")
        if not self.frames or len(self.frames) > 100_000:
            raise ValueError("comparison arm must contain at least one frame")
        return self


@dataclass(frozen=True, slots=True)
class ComparisonReceipt:
    """Immutable artifact identities produced by one comparison."""

    status: str
    slice_table_manifest_sha256: str
    evidence_manifest_sha256: str
    decision_manifest_sha256: str
    metrics_table_manifest_sha256: str
    report_data_manifest_sha256: str


def _sha256(value: str, label: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite value")
        return
    if isinstance(value, list):
        for item in value:
            _finite_tree(item, label)
        return
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item, label)
        return
    raise ValueError(f"{label} contains an unsupported value")


def _strict_object(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        return load_json_object(
            payload,
            label,
            ParseLimits(
                max_bytes=_MAX_ARM_BYTES,
                max_depth=32,
                max_nodes=2_000_000,
                max_container_items=1_000_000,
                max_string_bytes=8 * 1024 * 1024,
            ),
        )
    except ParseBoundaryError as error:
        raise ComparisonError(str(error)) from error


def _load_arm(registry: EvidenceRegistry, manifest_sha256: str, label: str) -> ComparisonArm:
    try:
        manifest = registry.store.read_manifest(manifest_sha256)
    except (OSError, RegistryError) as error:
        raise ComparisonError(f"{label} artifact failed immutable verification: {error}") from error
    if manifest["kind"] != "prediction_bundle":
        raise ComparisonError(f"{label} must be a prediction_bundle artifact")
    payload = cast(Mapping[str, Any], manifest["payload"])
    if payload["media_type"] != "application/vnd.junctionlens.comparison-arm+json":
        raise ComparisonError(f"{label} has an unsupported media type")
    if cast(int, payload["byte_size"]) > _MAX_ARM_BYTES:
        raise ComparisonError(f"{label} exceeds the comparison input byte limit")
    try:
        raw = read_bounded_file(
            registry.store.object_path(cast(str, payload["sha256"])),
            label,
            _MAX_ARM_BYTES,
        )
    except ParseBoundaryError as error:
        raise ComparisonError(str(error)) from error
    try:
        arm = ComparisonArm.model_validate(_strict_object(raw, label))
    except ValueError as error:
        raise ComparisonError(f"{label} schema is invalid: {error}") from error
    try:
        provenance = registry.provenance(manifest_sha256)
    except (OSError, RuntimeError, ValueError) as error:
        raise ComparisonError(f"{label} provenance cannot be verified: {error}") from error
    artifacts = provenance.get("artifacts")
    provenance_complete = isinstance(artifacts, list) and len(artifacts) > 1
    integrity = arm.integrity.model_copy(
        update={"provenance_complete": arm.integrity.provenance_complete and provenance_complete}
    )
    return arm.model_copy(update={"integrity": integrity})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slice_ids(path: Path) -> set[str]:
    try:
        value = load_yaml_object_path(
            path,
            "slice registry",
            ParseLimits(
                max_bytes=4 * 1024 * 1024,
                max_depth=16,
                max_nodes=100_000,
                max_container_items=10_000,
                max_string_bytes=64 * 1024,
            ),
        )
    except ParseBoundaryError as error:
        raise ComparisonError(str(error)) from error
    if not isinstance(value.get("slices"), list):
        raise ComparisonError("slice registry must contain a slices array")
    result: set[str] = set()
    for item in value["slices"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ComparisonError("slice registry contains an invalid definition")
        identifier = cast(str, item["id"])
        if identifier in result:
            raise ComparisonError(f"slice registry contains duplicate ID {identifier}")
        result.add(identifier)
    return result


def _frame_map(arm: ComparisonArm) -> dict[str, FrameEvidence]:
    return {frame.frame_token: frame for frame in arm.frames}


def _frame_metadata(frame: FrameEvidence) -> tuple[str, int, tuple[tuple[str, str], ...]]:
    return frame.segment_id, frame.timestamp_ns, tuple(sorted(frame.slice_values.items()))


def _slice_match(frame: FrameEvidence, expression: str, known_slices: set[str]) -> bool:
    if expression == "overall":
        return True
    dimension, separator, value = expression.partition(":")
    if separator != ":" or dimension not in known_slices or not value:
        raise ComparisonError(f"charter contains unsupported slice expression {expression}")
    return frame.slice_values.get(dimension) == value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ComparisonError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComparisonError(f"{label} must be a nonnegative integer")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise ComparisonError(f"{label} has unsupported fields: {sorted(unexpected)}")


def _support_values(primitive: Mapping[str, Any], label: str) -> tuple[int, int]:
    return (
        _integer(primitive.get("eligible_ground_truth_edges", 0), f"{label} edge support"),
        _integer(primitive.get("adjacent_frame_transitions", 0), f"{label} transition support"),
    )


def _aggregate_segments(
    frames: Sequence[FrameEvidence], cell: CharterCell
) -> tuple[list[dict[str, Any]], int, int, int]:
    grouped: dict[str, list[FrameEvidence]] = defaultdict(list)
    edge_support = 0
    transition_support = 0
    temporal_segments: set[str] = set()
    for frame in frames:
        grouped[frame.segment_id].append(frame)
        primitive = frame.metrics[cell.metric]
        edges, transitions = _support_values(primitive, f"{cell.id}:{frame.frame_token}")
        edge_support += edges
        transition_support += transitions
        if transitions:
            temporal_segments.add(frame.segment_id)

    segments: list[dict[str, Any]] = []
    for segment_id in sorted(grouped):
        items = sorted(grouped[segment_id], key=lambda item: (item.timestamp_ns, item.frame_token))
        if cell.estimator == "ratio":
            numerator = 0.0
            denominator = 0.0
            for frame in items:
                primitive = frame.metrics[cell.metric]
                _keys(
                    primitive,
                    {
                        "numerator",
                        "denominator",
                        "eligible_ground_truth_edges",
                        "adjacent_frame_transitions",
                    },
                    f"ratio primitive {cell.id}:{frame.frame_token}",
                )
                numerator += _number(primitive.get("numerator"), "ratio numerator")
                denominator += _number(primitive.get("denominator"), "ratio denominator")
            segments.append(
                {"segment_id": segment_id, "numerator": numerator, "denominator": denominator}
            )
        elif cell.estimator == "average_precision":
            ground_truth_count = 0
            predictions: list[dict[str, object]] = []
            for frame in items:
                primitive = frame.metrics[cell.metric]
                _keys(
                    primitive,
                    {
                        "ground_truth_count",
                        "predictions",
                        "eligible_ground_truth_edges",
                        "adjacent_frame_transitions",
                    },
                    f"AP primitive {cell.id}:{frame.frame_token}",
                )
                ground_truth_count += _integer(
                    primitive.get("ground_truth_count"), "AP ground truth count"
                )
                raw_predictions = primitive.get("predictions")
                if not isinstance(raw_predictions, list):
                    raise ComparisonError("AP predictions must be an array")
                seen: set[str] = set()
                for raw in raw_predictions:
                    if not isinstance(raw, dict):
                        raise ComparisonError("AP prediction must be an object")
                    _keys(raw, {"id", "confidence", "true_positive"}, "AP prediction")
                    identifier = raw.get("id")
                    confidence = _number(raw.get("confidence"), "AP confidence")
                    true_positive = raw.get("true_positive")
                    if (
                        not isinstance(identifier, str)
                        or not identifier
                        or identifier in seen
                        or not 0.0 <= confidence <= 1.0
                        or not isinstance(true_positive, bool)
                    ):
                        raise ComparisonError("AP prediction fields are invalid")
                    seen.add(identifier)
                    predictions.append(
                        {
                            "id": f"{frame.frame_token}:{identifier}",
                            "confidence": confidence,
                            "true_positive": true_positive,
                        }
                    )
            segments.append(
                {
                    "segment_id": segment_id,
                    "ground_truth_count": ground_truth_count,
                    "predictions": predictions,
                }
            )
        elif cell.estimator == "adaptive_ece":
            observations: list[dict[str, object]] = []
            for frame in items:
                primitive = frame.metrics[cell.metric]
                _keys(
                    primitive,
                    {
                        "observations",
                        "eligible_ground_truth_edges",
                        "adjacent_frame_transitions",
                    },
                    f"ECE primitive {cell.id}:{frame.frame_token}",
                )
                raw_observations = primitive.get("observations")
                if not isinstance(raw_observations, list):
                    raise ComparisonError("ECE observations must be an array")
                seen = set()
                for raw in raw_observations:
                    if not isinstance(raw, dict):
                        raise ComparisonError("ECE observation must be an object")
                    _keys(raw, {"id", "confidence", "correct"}, "ECE observation")
                    identifier = raw.get("id")
                    confidence = _number(raw.get("confidence"), "ECE confidence")
                    correct = raw.get("correct")
                    if (
                        not isinstance(identifier, str)
                        or not identifier
                        or identifier in seen
                        or not 0.0 <= confidence <= 1.0
                        or not isinstance(correct, bool)
                    ):
                        raise ComparisonError("ECE observation fields are invalid")
                    seen.add(identifier)
                    observations.append(
                        {
                            "id": f"{frame.frame_token}:{identifier}",
                            "confidence": confidence,
                            "correct": correct,
                        }
                    )
            segments.append({"segment_id": segment_id, "observations": observations})
        else:
            raise ComparisonError(f"runtime estimator cannot use frame primitives for {cell.id}")
    return segments, edge_support, transition_support, len(temporal_segments)


def _runtime_blocks(arm: ComparisonArm, cell: CharterCell) -> list[dict[str, object]]:
    blocks = arm.runtime.metrics.get(cell.metric)
    if blocks is None:
        return []
    return [block.model_dump(mode="json") for block in blocks]


def _empty_runtime_blocks() -> list[dict[str, object]]:
    return [
        {"block_id": f"missing-{index:02d}", "value": 0.0, "valid": False} for index in range(10)
    ]


def _integrity_payload(
    baseline: ComparisonArm,
    candidate: ComparisonArm,
    *,
    frame_sets_match: bool,
    slice_values_match: bool,
    support_values_match: bool,
    registry_inputs_match: bool,
    metric_registry_sha256: str,
    slice_registry_sha256: str,
) -> dict[str, object]:
    claims = (baseline.integrity, candidate.integrity)
    return {
        "artifact_integrity": all(item.artifact_integrity for item in claims),
        "baseline_evaluator_image_digest": baseline.evaluator_image_digest,
        "candidate_evaluator_image_digest": candidate.evaluator_image_digest,
        "baseline_data_manifest_sha256": baseline.data_manifest_sha256,
        "candidate_data_manifest_sha256": candidate.data_manifest_sha256,
        "baseline_split_manifest_sha256": baseline.split_manifest_sha256,
        "candidate_split_manifest_sha256": candidate.split_manifest_sha256,
        "baseline_preprocessing_sha256": baseline.preprocessing_sha256,
        "candidate_preprocessing_sha256": candidate.preprocessing_sha256,
        "baseline_postprocessing_sha256": baseline.postprocessing_sha256,
        "candidate_postprocessing_sha256": candidate.postprocessing_sha256,
        "metric_registry_sha256": metric_registry_sha256,
        "slice_registry_sha256": slice_registry_sha256,
        "calibration_ranks_frozen": all(item.calibration_ranks_frozen for item in claims),
        "candidate_training_holdout_access_count": (
            candidate.integrity.training_holdout_access_count
        ),
        "frame_sets_match": frame_sets_match,
        "slice_values_match": slice_values_match,
        "support_values_match": support_values_match,
        "schema_major_compatible": all(item.schema_major_compatible for item in claims),
        "identifiers_valid": all(item.identifiers_valid for item in claims),
        "coordinate_metadata_valid": all(item.coordinate_metadata_valid for item in claims),
        "required_values_finite": all(item.required_values_finite for item in claims),
        "calibrator_valid": all(item.calibrator_valid for item in claims),
        "evaluator_compatible": all(item.evaluator_compatible for item in claims),
        "provenance_complete": all(item.provenance_complete for item in claims),
        "leakage_free": all(item.leakage_free for item in claims),
        "partial_inference_approved": all(item.partial_inference_approved for item in claims),
        "provider_fallback_free": all(item.provider_fallback_free for item in claims),
        "registry_inputs_match": registry_inputs_match,
    }


def _runtime_payload(
    baseline: ComparisonArm, candidate: ComparisonArm, *, paired_trial_ids_match: bool
) -> dict[str, object]:
    runtime = candidate.runtime
    baseline_runtime = baseline.runtime
    return {
        "baseline_hardware_manifest_sha256": (baseline_runtime.hardware_baseline_manifest_sha256),
        "hardware_baseline_manifest_sha256": runtime.hardware_baseline_manifest_sha256,
        "baseline_gpu_provider_active": baseline_runtime.gpu_provider_active,
        "gpu_provider_active": runtime.gpu_provider_active,
        "throughput_per_second": runtime.throughput_per_second,
        "p95_latency_ms": runtime.p95_latency_ms,
        "p99_latency_ms": runtime.p99_latency_ms,
        "peak_device_memory_bytes": runtime.peak_device_memory_bytes,
        "long_run_frames": runtime.long_run_frames,
        "unbounded_memory_growth": runtime.unbounded_memory_growth,
        "unexpected_cpu_provider_nodes": runtime.unexpected_cpu_provider_nodes,
        "warmup_frames_per_block": runtime.warmup_frames_per_block,
        "measured_frames_per_block": runtime.measured_frames_per_block,
        "trial_blocks": len(runtime.trial_block_order),
        "trial_block_order": list(runtime.trial_block_order),
        "baseline_warmup_frames_per_block": baseline_runtime.warmup_frames_per_block,
        "baseline_measured_frames_per_block": baseline_runtime.measured_frames_per_block,
        "baseline_trial_block_order": list(baseline_runtime.trial_block_order),
        "baseline_environment_valid": baseline_runtime.environment_valid,
        "paired_trial_ids_match": paired_trial_ids_match,
        "environment_valid": runtime.environment_valid,
    }


def _build_evidence(
    charter: FrozenCharter,
    baseline: ComparisonArm,
    candidate: ComparisonArm,
    known_slices: set[str],
    *,
    metric_registry_sha256: str,
    slice_registry_sha256: str,
) -> tuple[GateEvidence, list[dict[str, object]]]:
    baseline_frames = _frame_map(baseline)
    candidate_frames = _frame_map(candidate)
    frame_sets_match = set(baseline_frames) == set(candidate_frames)
    common_tokens = sorted(set(baseline_frames) & set(candidate_frames))
    metadata_match = all(
        _frame_metadata(baseline_frames[token]) == _frame_metadata(candidate_frames[token])
        for token in common_tokens
    )
    slice_values_match = all(
        baseline_frames[token].slice_values == candidate_frames[token].slice_values
        for token in common_tokens
    )
    frame_sets_match = frame_sets_match and metadata_match
    support_values_match = True
    paired_trial_ids_match = True
    cells: dict[str, dict[str, object]] = {}
    for cell in charter.cells:
        if cell.estimator == "paired_runtime":
            baseline_blocks = _runtime_blocks(baseline, cell)
            candidate_blocks = _runtime_blocks(candidate, cell)
            baseline_ids = {cast(str, item["block_id"]) for item in baseline_blocks}
            candidate_ids = {cast(str, item["block_id"]) for item in candidate_blocks}
            if baseline_ids != candidate_ids:
                paired_trial_ids_match = False
            common_block_ids = sorted(baseline_ids & candidate_ids)
            if not baseline_blocks or not candidate_blocks or not common_block_ids:
                common_block_ids = [f"missing-{index:02d}" for index in range(10)]
                baseline_blocks = _empty_runtime_blocks()
                candidate_blocks = _empty_runtime_blocks()
            cells[cell.id] = {
                "estimator": cell.estimator,
                "baseline": {"blocks": baseline_blocks},
                "candidate": {"blocks": candidate_blocks},
                "support": {
                    "paired_segments": 0,
                    "eligible_ground_truth_edges": 0,
                    "adjacent_frame_transitions": 0,
                    "temporal_segments": 0,
                },
                "counterexample_query": f"metric == '{cell.metric}' and slice == '{cell.slice}'",
            }
            continue
        baseline_tokens = {
            token
            for token, frame in baseline_frames.items()
            if cell.metric in frame.metrics and _slice_match(frame, cell.slice, known_slices)
        }
        candidate_tokens = {
            token
            for token, frame in candidate_frames.items()
            if cell.metric in frame.metrics and _slice_match(frame, cell.slice, known_slices)
        }
        if baseline_tokens != candidate_tokens:
            frame_sets_match = False
        eligible = sorted(baseline_tokens & candidate_tokens)
        baseline_selected = [baseline_frames[token] for token in eligible]
        candidate_selected = [candidate_frames[token] for token in eligible]
        baseline_segments, baseline_edges, baseline_transitions, baseline_temporal = (
            _aggregate_segments(baseline_selected, cell)
        )
        candidate_segments, candidate_edges, candidate_transitions, candidate_temporal = (
            _aggregate_segments(candidate_selected, cell)
        )
        if (
            baseline_edges != candidate_edges
            or baseline_transitions != candidate_transitions
            or baseline_temporal != candidate_temporal
        ):
            support_values_match = False
        if cell.estimator == "average_precision" and any(
            baseline_item["ground_truth_count"] != candidate_item["ground_truth_count"]
            for baseline_item, candidate_item in zip(
                baseline_segments, candidate_segments, strict=True
            )
        ):
            support_values_match = False
        cells[cell.id] = {
            "estimator": cell.estimator,
            "baseline": {"segments": baseline_segments},
            "candidate": {"segments": candidate_segments},
            "support": {
                "paired_segments": len({frame.segment_id for frame in baseline_selected}),
                "eligible_ground_truth_edges": min(baseline_edges, candidate_edges),
                "adjacent_frame_transitions": min(baseline_transitions, candidate_transitions),
                "temporal_segments": min(baseline_temporal, candidate_temporal),
            },
            "counterexample_query": f"metric == '{cell.metric}' and slice == '{cell.slice}'",
        }

    slice_rows: list[dict[str, object]] = []
    for token in common_tokens:
        frame = baseline_frames[token]
        for dimension, value in sorted(frame.slice_values.items()):
            if dimension not in known_slices:
                raise ComparisonError(f"frame {token} references unknown slice {dimension}")
            slice_rows.append(
                {
                    "schema_version": "junctionlens.slice-row.v1",
                    "frame_token": token,
                    "segment_id": frame.segment_id,
                    "timestamp_ns": frame.timestamp_ns,
                    "slice_id": dimension,
                    "slice_value": value,
                }
            )

    payload = {
        "schema_version": "junctionlens.gate-evidence.v1",
        "charter_sha256": charter.charter_sha256,
        "integrity": _integrity_payload(
            baseline,
            candidate,
            frame_sets_match=frame_sets_match,
            slice_values_match=slice_values_match,
            support_values_match=support_values_match,
            registry_inputs_match=(
                baseline.metric_registry_sha256
                == candidate.metric_registry_sha256
                == metric_registry_sha256
                and baseline.slice_registry_sha256
                == candidate.slice_registry_sha256
                == slice_registry_sha256
            ),
            metric_registry_sha256=metric_registry_sha256,
            slice_registry_sha256=slice_registry_sha256,
        ),
        "cells": cells,
        "runtime": _runtime_payload(
            baseline, candidate, paired_trial_ids_match=paired_trial_ids_match
        ),
    }
    try:
        return GateEvidence.model_validate(payload), slice_rows
    except ValueError as error:
        raise ComparisonError(f"derived gate evidence is invalid: {error}") from error


def _write_slice_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ComparisonError("comparison requires the pinned analytics extra") from error
    schema = pa.schema(
        [
            ("schema_version", pa.string()),
            ("frame_token", pa.string()),
            ("segment_id", pa.string()),
            ("timestamp_ns", pa.int64()),
            ("slice_id", pa.string()),
            ("slice_value", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(list(rows), schema=schema),
        path,
        compression="zstd",
        data_page_version="2.0",
        version="2.6",
        write_statistics=True,
    )


def _write_metrics_table(path: Path, decision: Mapping[str, Any]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ComparisonError("comparison requires the pinned analytics extra") from error
    rows = []
    for item in cast(list[Mapping[str, Any]], decision["cells"]):
        interval = cast(Mapping[str, Any], item["interval"])
        support = cast(Mapping[str, Any], item["support"])
        rows.append(
            {
                "schema_version": "junctionlens.comparison-metric-row.v1",
                "cell_id": item["cell_id"],
                "metric": item["metric"],
                "slice": item["slice"],
                "status": item["status"],
                "reason_code": item["reason_code"],
                "point_estimate": item["point_estimate"],
                "lower": interval["lower"],
                "upper": interval["upper"],
                "adjusted_two_sided_alpha": interval["adjusted_two_sided_alpha"],
                "margin": item["margin"],
                "paired_segments": support["paired_segments"],
                "eligible_ground_truth_edges": support["eligible_ground_truth_edges"],
                "adjacent_frame_transitions": support["adjacent_frame_transitions"],
                "temporal_segments": support["temporal_segments"],
                "finite_replicates": item["finite_replicates"],
                "invalid_replicates": item["invalid_replicates"],
                "counterexample_query": item["counterexample_query"],
            }
        )
    schema = pa.schema(
        [
            ("schema_version", pa.string()),
            ("cell_id", pa.string()),
            ("metric", pa.string()),
            ("slice", pa.string()),
            ("status", pa.string()),
            ("reason_code", pa.string()),
            ("point_estimate", pa.float64()),
            ("lower", pa.float64()),
            ("upper", pa.float64()),
            ("adjusted_two_sided_alpha", pa.float64()),
            ("margin", pa.float64()),
            ("paired_segments", pa.int64()),
            ("eligible_ground_truth_edges", pa.int64()),
            ("adjacent_frame_transitions", pa.int64()),
            ("temporal_segments", pa.int64()),
            ("finite_replicates", pa.int64()),
            ("invalid_replicates", pa.int64()),
            ("counterexample_query", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        data_page_version="2.0",
        version="2.6",
        write_statistics=True,
    )


def run_comparison(
    *,
    artifact_root: Path,
    schema_path: Path,
    charter_path: Path,
    metric_registry_path: Path,
    slice_registry_path: Path,
    baseline_manifest_sha256: str,
    candidate_manifest_sha256: str,
) -> ComparisonReceipt:
    """Compare two immutable arms and persist evidence, decision, and report tables."""
    registry = EvidenceRegistry(artifact_root, schema_path)
    charter = load_frozen_charter(charter_path)
    baseline = _load_arm(registry, baseline_manifest_sha256, "baseline")
    candidate = _load_arm(registry, candidate_manifest_sha256, "candidate")
    known_slices = _slice_ids(slice_registry_path)
    evidence, slice_rows = _build_evidence(
        charter,
        baseline,
        candidate,
        known_slices,
        metric_registry_sha256=_file_sha256(metric_registry_path),
        slice_registry_sha256=_file_sha256(slice_registry_path),
    )
    parents = tuple(sorted((baseline_manifest_sha256, candidate_manifest_sha256)))
    evidence_bytes = canonical_json_bytes(evidence.model_dump(mode="json")) + b"\n"
    with tempfile.TemporaryDirectory(prefix="comparison-", dir=registry.store.staging_root) as temp:
        temporary = Path(temp)
        slice_path = temporary / "slices.parquet"
        metrics_path = temporary / "metrics.parquet"
        evidence_path = temporary / "gate-evidence.json"
        _write_slice_table(slice_path, slice_rows)
        slice_receipt = registry.put_file(
            slice_path,
            kind="slice_table",
            media_type="application/vnd.apache.parquet",
            license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
            metadata={
                "row_count": len(slice_rows),
                "slice_registry_sha256": _file_sha256(slice_registry_path),
            },
            parents=parents,
        )
        evidence_path.write_bytes(evidence_bytes)
        evidence_receipt = registry.put_bytes(
            evidence_bytes,
            kind="comparison",
            media_type="application/vnd.junctionlens.gate-evidence+json",
            license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
            metadata={"charter_sha256": charter.charter_sha256},
            parents=(*parents, slice_receipt.manifest_sha256),
        )
        decision = decide_release(charter_path, evidence_path)
        decision_receipt = registry.put_bytes(
            canonical_json_bytes(decision) + b"\n",
            kind="release_decision",
            media_type="application/vnd.junctionlens.gate-decision+json",
            license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
            metadata={"charter_sha256": charter.charter_sha256, "status": decision["status"]},
            parents=(evidence_receipt.manifest_sha256,),
        )
        _write_metrics_table(metrics_path, decision)
        metrics_receipt = registry.put_file(
            metrics_path,
            kind="comparison",
            media_type="application/vnd.apache.parquet",
            license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
            metadata={"row_count": len(charter.cells)},
            parents=(decision_receipt.manifest_sha256,),
        )
        report_data = {
            "schema_version": "junctionlens.comparison-report-data.v1",
            "status": decision["status"],
            "baseline_manifest_sha256": baseline_manifest_sha256,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "charter_sha256": charter.charter_sha256,
            "decision_manifest_sha256": decision_receipt.manifest_sha256,
            "metrics_table_manifest_sha256": metrics_receipt.manifest_sha256,
            "slice_table_manifest_sha256": slice_receipt.manifest_sha256,
            "reason_codes": sorted(
                {
                    *cast(list[str], decision["integrity_reason_codes"]),
                    *cast(list[str], decision["infrastructure_reason_codes"]),
                    *cast(list[str], decision["performance_reason_codes"]),
                    *[
                        cast(str, item["reason_code"])
                        for item in cast(list[Mapping[str, Any]], decision["cells"])
                        if item["status"] != "PASS"
                    ],
                }
            ),
            "cells": decision["cells"],
            "primary_hypotheses": decision["primary_hypotheses"],
            "filtering_changes_release_status": False,
        }
        report_receipt = registry.put_bytes(
            canonical_json_bytes(report_data) + b"\n",
            kind="comparison",
            media_type="application/vnd.junctionlens.comparison-report-data+json",
            license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
            metadata={"status": decision["status"]},
            parents=(
                decision_receipt.manifest_sha256,
                metrics_receipt.manifest_sha256,
                slice_receipt.manifest_sha256,
            ),
        )
    return ComparisonReceipt(
        status=cast(str, decision["status"]),
        slice_table_manifest_sha256=slice_receipt.manifest_sha256,
        evidence_manifest_sha256=evidence_receipt.manifest_sha256,
        decision_manifest_sha256=decision_receipt.manifest_sha256,
        metrics_table_manifest_sha256=metrics_receipt.manifest_sha256,
        report_data_manifest_sha256=report_receipt.manifest_sha256,
    )


def receipt_dict(receipt: ComparisonReceipt) -> dict[str, str]:
    return {key: cast(str, value) for key, value in asdict(receipt).items()}


__all__ = [
    "ComparisonArm",
    "ComparisonError",
    "ComparisonReceipt",
    "receipt_dict",
    "run_comparison",
]

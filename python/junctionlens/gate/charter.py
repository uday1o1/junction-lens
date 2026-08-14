"""Strict acceptance-charter draft validation and one-way freeze."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from junctionlens.registry import ContentAddressedStore, RegistryError
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object,
    load_yaml_object_path,
    read_bounded_file,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class CharterError(RuntimeError):
    """Raised when a charter is mutable, contaminated, or unsupported."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _load_yaml(path: Path, label: str) -> object:
    try:
        return load_yaml_object_path(
            path,
            label,
            ParseLimits(
                max_bytes=4 * 1024 * 1024,
                max_depth=24,
                max_nodes=100_000,
                max_container_items=10_000,
                max_string_bytes=256 * 1024,
            ),
        )
    except ParseBoundaryError as error:
        raise CharterError(str(error)) from error


class BootstrapPolicy(_Frozen):
    algorithm: Literal["paired-segment-cluster-bootstrap-v1", "paired-trial-block-bootstrap-v1"]
    replicates: Literal[10000]
    seed: Literal[20260813]
    minimum_finite_replicates: Literal[9900] | None = None
    minimum_valid_pairs: Literal[8] | None = None
    balanced_order_schedule: tuple[Literal["AB", "BA"], ...] | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> BootstrapPolicy:
        if self.algorithm == "paired-segment-cluster-bootstrap-v1":
            if (
                self.minimum_finite_replicates != 9900
                or self.minimum_valid_pairs is not None
                or self.balanced_order_schedule is not None
            ):
                raise ValueError("segment bootstrap finite-replicate policy is invalid")
        elif (
            self.minimum_valid_pairs != 8
            or self.minimum_finite_replicates is not None
            or self.balanced_order_schedule is None
            or len(self.balanced_order_schedule) != 10
            or self.balanced_order_schedule.count("AB") != 5
            or self.balanced_order_schedule.count("BA") != 5
        ):
            raise ValueError("runtime bootstrap schedule or valid-pair policy is invalid")
        return self


class SupportPolicy(_Frozen):
    overall_paired_segments: Literal[200]
    gating_slice_paired_segments: Literal[30]
    lane_control_ground_truth_edges: Literal[500]
    temporal_adjacent_frame_transitions: Literal[500]
    temporal_segments: Literal[30]


class RuntimePolicy(_Frozen):
    throughput_per_second_minimum: float
    p95_latency_ms_maximum: float
    p99_latency_ms_maximum: float
    peak_device_memory_bytes_maximum: Literal[6442450944]
    long_run_frames: Literal[10000]
    unexpected_cpu_provider_nodes_maximum: Literal[0]

    @model_validator(mode="after")
    def validate_frozen_values(self) -> RuntimePolicy:
        expected = (10.0, 100.0, 125.0)
        observed = (
            self.throughput_per_second_minimum,
            self.p95_latency_ms_maximum,
            self.p99_latency_ms_maximum,
        )
        if observed != expected:
            raise ValueError("absolute runtime policy differs from the V1 contract")
        return self


class CharterCell(_Frozen):
    id: str
    metric: str
    slice: str
    direction: Literal["higher_is_better", "lower_is_better"]
    margin: float = Field(ge=0.0)
    support: Literal["overall", "slice", "lane_control", "temporal", "runtime"]
    estimator: Literal["ratio", "average_precision", "adaptive_ece", "paired_runtime"]
    stage: Literal["accuracy", "calibration", "performance"]

    @model_validator(mode="after")
    def validate_identity(self) -> CharterCell:
        if not self.id or not self.metric or not self.slice or len(self.id) > 160:
            raise ValueError("charter cell identity is invalid")
        if self.estimator == "paired_runtime" and self.stage != "performance":
            raise ValueError("paired runtime cell must be a performance cell")
        if self.stage == "performance" and self.support != "runtime":
            raise ValueError("performance cell must use runtime support")
        return self


class PrimaryHypothesis(_Frozen):
    id: str
    cell_id: str
    minimum_improvement: float


class FreezeEvidence(_Frozen):
    baseline_seed_checkpoint_sha256: dict[str, str]
    m0_hardware_baseline_manifest_sha256: str
    power_simulation_artifact_sha256: str
    product_priorities_sha256: str


class FrozenCharter(_Frozen):
    schema_version: Literal["junctionlens.acceptance-charter.v1"]
    charter_id: Literal["junctionlens-acceptance-v1"]
    frozen: Literal[True]
    frozen_at: str
    signer: str
    source_commit: str
    draft_sha256: str
    baseline_run_manifest_sha256: str
    baseline_evidence_payload_sha256: str
    metric_registry_sha256: str
    slice_registry_sha256: str
    family_alpha: float
    bootstrap: BootstrapPolicy
    runtime_bootstrap: BootstrapPolicy
    support: SupportPolicy
    absolute_runtime: RuntimePolicy
    cells: tuple[CharterCell, ...]
    primary_hypotheses: tuple[PrimaryHypothesis, ...]
    freeze_evidence: FreezeEvidence
    charter_sha256: str

    @model_validator(mode="after")
    def validate_hashes(self) -> FrozenCharter:
        hashes = (
            self.draft_sha256,
            self.baseline_run_manifest_sha256,
            self.baseline_evidence_payload_sha256,
            self.metric_registry_sha256,
            self.slice_registry_sha256,
            self.charter_sha256,
            self.freeze_evidence.m0_hardware_baseline_manifest_sha256,
            self.freeze_evidence.power_simulation_artifact_sha256,
            self.freeze_evidence.product_priorities_sha256,
            *self.freeze_evidence.baseline_seed_checkpoint_sha256.values(),
        )
        if any(_SHA256_PATTERN.fullmatch(value) is None for value in hashes):
            raise ValueError("frozen charter contains an invalid SHA-256")
        if _GIT_PATTERN.fullmatch(self.source_commit) is None:
            raise ValueError("frozen charter contains an invalid source commit")
        if self.family_alpha != 0.05:
            raise ValueError("frozen charter family alpha differs from V1")
        return self


class CharterDraft(_Frozen):
    schema_version: Literal["junctionlens.acceptance-charter-draft.v1"]
    charter_id: Literal["junctionlens-acceptance-v1"]
    family_alpha: float
    bootstrap: BootstrapPolicy
    runtime_bootstrap: BootstrapPolicy
    support: SupportPolicy
    absolute_runtime: RuntimePolicy
    cells: tuple[CharterCell, ...]
    primary_hypotheses: tuple[PrimaryHypothesis, ...]

    @model_validator(mode="after")
    def validate_cells(self) -> CharterDraft:
        if self.family_alpha != 0.05:
            raise ValueError("V1 family alpha must be exactly 0.05")
        identifiers = [cell.id for cell in self.cells]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("charter cell IDs must be nonempty and unique")
        hypotheses = [item.id for item in self.primary_hypotheses]
        if len(hypotheses) != len(set(hypotheses)):
            raise ValueError("primary hypothesis IDs are duplicated")
        known = set(identifiers)
        if any(item.cell_id not in known for item in self.primary_hypotheses):
            raise ValueError("primary hypothesis references an unknown cell")
        return self


def load_charter_draft(path: Path) -> CharterDraft:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise CharterError("charter draft must be a bounded regular file")
    value = _load_yaml(path, "charter draft")
    return CharterDraft.model_validate(value)


def load_frozen_charter(path: Path) -> FrozenCharter:
    """Load a read-only charter and verify its self-authenticating hash."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise CharterError("frozen charter must be a bounded regular file")
    value = _load_yaml(path, "frozen charter")
    if not isinstance(value, dict):
        raise CharterError("frozen charter must be an object")
    supplied_hash = value.get("charter_sha256")
    unsigned = dict(value)
    unsigned.pop("charter_sha256", None)
    observed_hash = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if supplied_hash != observed_hash:
        raise CharterError("frozen charter self-hash does not match its content")
    try:
        return FrozenCharter.model_validate(value)
    except ValueError as error:
        raise CharterError(f"frozen charter schema is invalid: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_bytes(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        return load_json_object(
            raw,
            label,
            ParseLimits(
                max_bytes=64 * 1024 * 1024,
                max_depth=24,
                max_nodes=1_000_000,
                max_container_items=100_000,
                max_string_bytes=4 * 1024 * 1024,
            ),
        )
    except ParseBoundaryError as error:
        raise CharterError(str(error)) from error


def _source_commit(project_root: Path) -> str:
    override = os.environ.get("JUNCTIONLENS_SOURCE_COMMIT")
    if override is not None and not (project_root / ".git").exists():
        if _GIT_PATTERN.fullmatch(override) is None:
            raise CharterError("verified source commit override is invalid")
        return override
    git = shutil.which("git")
    if git is None:
        raise CharterError("git is required to freeze a charter")
    status = subprocess.run(
        [git, "status", "--porcelain", "--untracked-files=normal"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if status.returncode != 0 or status.stdout:
        raise CharterError("charter freeze requires a clean source checkout")
    result = subprocess.run(
        [git, "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or _GIT_PATTERN.fullmatch(value) is None:
        raise CharterError("charter freeze cannot resolve a full source commit")
    return value


def _artifact_manifest_sha256(uri: str) -> str:
    prefix = "artifacts://runs/"
    if not uri.startswith(prefix):
        raise CharterError("baseline run must use artifacts://runs/<manifest-sha256>")
    value = uri.removeprefix(prefix)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise CharterError("baseline run URI does not contain a manifest SHA-256")
    return value


def _validate_freeze_evidence(
    evidence: Mapping[str, Any], draft: CharterDraft
) -> Mapping[str, float]:
    required = {
        "baseline_seed_checkpoint_sha256",
        "baseline_variability",
        "experiment_id",
        "internal_holdout_access_count",
        "m0_hardware_baseline_manifest_sha256",
        "power_simulation",
        "product_priorities_sha256",
        "proposed_margins",
        "schema_version",
        "source_partitions",
    }
    if set(evidence) != required:
        raise CharterError("baseline freeze evidence schema is invalid")
    if (
        evidence["schema_version"] != "junctionlens.baseline-freeze-evidence.v1"
        or evidence["experiment_id"] != "E0-independent"
        or evidence["source_partitions"] != ["model_training", "model_selection"]
        or evidence["internal_holdout_access_count"] != 0
    ):
        raise CharterError("baseline freeze evidence used a forbidden partition or experiment")
    seed_hashes = evidence["baseline_seed_checkpoint_sha256"]
    if (
        not isinstance(seed_hashes, dict)
        or set(seed_hashes)
        != {
            "20260813",
            "20260814",
            "20260815",
        }
        or any(
            not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
            for value in seed_hashes.values()
        )
    ):
        raise CharterError("baseline freeze evidence lacks the predeclared three seeds")
    for field in (
        "m0_hardware_baseline_manifest_sha256",
        "product_priorities_sha256",
    ):
        value = evidence[field]
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise CharterError(f"baseline freeze evidence has invalid {field}")
    variability = evidence["baseline_variability"]
    cell_ids = {cell.id for cell in draft.cells if cell.stage != "performance"}
    if not isinstance(variability, dict) or set(variability) != cell_ids:
        raise CharterError("baseline variability must cover every non-performance cell")
    for cell_id, values in variability.items():
        if not isinstance(values, list) or len(values) != 3:
            raise CharterError(f"baseline variability lacks three seed values for {cell_id}")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not float("-inf") < float(value) < float("inf")
            for value in values
        ):
            raise CharterError(f"baseline variability is nonfinite for {cell_id}")
    power = evidence["power_simulation"]
    if not isinstance(power, dict) or set(power) != {
        "artifact_sha256",
        "candidate_results_used",
        "internal_holdout_used",
        "source_partitions",
    }:
        raise CharterError("power simulation evidence schema is invalid")
    if (
        power["source_partitions"] != ["model_training", "model_selection"]
        or power["candidate_results_used"] is not False
        or power["internal_holdout_used"] is not False
        or not isinstance(power["artifact_sha256"], str)
        or _SHA256_PATTERN.fullmatch(power["artifact_sha256"]) is None
    ):
        raise CharterError("power simulation used forbidden evidence")
    proposed = evidence["proposed_margins"]
    draft_margins = {cell.id: cell.margin for cell in draft.cells}
    if not isinstance(proposed, dict) or set(proposed) != set(draft_margins):
        raise CharterError("proposed margins do not cover the exact charter cells")
    result = {}
    for cell_id, value in proposed.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not 0.0 <= float(value) <= draft_margins[cell_id]
        ):
            raise CharterError(f"proposed margin loosens or invalidates {cell_id}")
        result[cell_id] = float(value)
    return result


def freeze_charter(
    draft_path: Path,
    baseline_run_uri: str,
    output_path: Path,
    *,
    artifact_root: Path,
    project_root: Path,
    signer: str,
    metrics_path: Path,
    slices_path: Path,
) -> Mapping[str, Any]:
    """Freeze V1 only from immutable E0, hardware, and pre-holdout evidence."""
    if not signer.strip() or len(signer) > 200 or any(ord(character) < 32 for character in signer):
        raise CharterError("charter signer must be a short explicit local identity")
    if output_path.exists() or output_path.is_symlink():
        raise CharterError("frozen charter output already exists")
    draft = load_charter_draft(draft_path)
    manifest_sha256 = _artifact_manifest_sha256(baseline_run_uri)
    try:
        store = ContentAddressedStore(
            artifact_root, project_root / "schemas/artifact-manifest-v1.schema.json"
        )
        manifest = store.read_manifest(manifest_sha256)
    except (OSError, RegistryError, ValueError) as error:
        raise CharterError(f"cannot verify baseline run artifact: {error}") from error
    if manifest.get("kind") != "baseline_freeze_evidence":
        raise CharterError("baseline run artifact has the wrong kind")
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise CharterError("baseline run artifact payload is invalid")
    payload_path = store.object_path(str(payload.get("sha256")))
    try:
        evidence_bytes = read_bounded_file(
            payload_path,
            "baseline freeze evidence",
            64 * 1024 * 1024,
        )
    except ParseBoundaryError as error:
        raise CharterError(str(error)) from error
    evidence = _load_json_bytes(evidence_bytes, "baseline freeze evidence")
    frozen_margins = _validate_freeze_evidence(evidence, draft)
    source_commit = _source_commit(project_root)
    if not metrics_path.is_file() or metrics_path.is_symlink():
        raise CharterError("metric registry must be a regular file")
    if not slices_path.is_file() or slices_path.is_symlink():
        raise CharterError("slice registry must be a regular file")
    cells = []
    for cell in draft.cells:
        value = cell.model_dump(mode="json")
        value["margin"] = frozen_margins[cell.id]
        cells.append(value)
    body: dict[str, Any] = {
        "schema_version": "junctionlens.acceptance-charter.v1",
        "charter_id": draft.charter_id,
        "frozen": True,
        "frozen_at": datetime.now(UTC).isoformat(),
        "signer": signer,
        "source_commit": source_commit,
        "draft_sha256": _sha256_file(draft_path),
        "baseline_run_manifest_sha256": manifest_sha256,
        "baseline_evidence_payload_sha256": str(payload["sha256"]),
        "metric_registry_sha256": _sha256_file(metrics_path),
        "slice_registry_sha256": _sha256_file(slices_path),
        "family_alpha": draft.family_alpha,
        "bootstrap": draft.bootstrap.model_dump(mode="json"),
        "runtime_bootstrap": draft.runtime_bootstrap.model_dump(mode="json"),
        "support": draft.support.model_dump(mode="json"),
        "absolute_runtime": draft.absolute_runtime.model_dump(mode="json"),
        "cells": cells,
        "primary_hypotheses": [value.model_dump(mode="json") for value in draft.primary_hypotheses],
        "freeze_evidence": {
            "baseline_seed_checkpoint_sha256": evidence["baseline_seed_checkpoint_sha256"],
            "m0_hardware_baseline_manifest_sha256": evidence[
                "m0_hardware_baseline_manifest_sha256"
            ],
            "power_simulation_artifact_sha256": cast(
                Mapping[str, Any], evidence["power_simulation"]
            )["artifact_sha256"],
            "product_priorities_sha256": evidence["product_priorities_sha256"],
        },
    }
    body["charter_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(body, sort_keys=False, allow_unicode=False)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as temporary:
        temporary.write(serialized)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o444)
    temporary_path.replace(output_path)
    return {
        "schema_version": "junctionlens.charter-freeze-receipt.v1",
        "state": "FROZEN",
        "charter_sha256": body["charter_sha256"],
        "output_sha256": _sha256_file(output_path),
        "baseline_run_manifest_sha256": manifest_sha256,
        "source_commit": source_commit,
    }


__all__ = [
    "CharterCell",
    "CharterDraft",
    "CharterError",
    "FrozenCharter",
    "freeze_charter",
    "load_charter_draft",
    "load_frozen_charter",
]

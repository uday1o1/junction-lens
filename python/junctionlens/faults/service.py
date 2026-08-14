"""Immutable registry workflow for fault injection and detection evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from junctionlens.faults.analysis import analyze_fault, verify_clean_bundle
from junctionlens.faults.models import FaultKind, PredictionBundle
from junctionlens.faults.transforms import FaultTransformError, apply_fault
from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import RegistryError, canonical_json_bytes

_MAX_BUNDLE_BYTES = 512 * 1024 * 1024
_MEDIA_TYPE = "application/vnd.junctionlens.prediction-bundle+json"


class FaultError(RuntimeError):
    """Raised when fault evidence cannot be derived or independently detected."""


@dataclass(frozen=True, slots=True)
class FaultReceipt:
    state: str
    fault_kind: str
    primary_reason_code: str
    parent_manifest_sha256: str
    derived_manifest_sha256: str
    counterexample_manifest_sha256: str


def _strict_object(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise FaultError(f"{label} contains duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                FaultError(f"{label} contains nonfinite constant {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise FaultError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise FaultError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def load_prediction_bundle(registry: EvidenceRegistry, manifest_sha256: str) -> PredictionBundle:
    """Load one verified prediction bundle from its immutable manifest identity."""
    try:
        manifest = registry.store.read_manifest(manifest_sha256)
    except (OSError, RegistryError) as error:
        raise FaultError(f"prediction bundle failed immutable verification: {error}") from error
    if manifest["kind"] != "prediction_bundle":
        raise FaultError("fault input must be a prediction_bundle artifact")
    payload = cast(Mapping[str, Any], manifest["payload"])
    if payload["media_type"] != _MEDIA_TYPE:
        raise FaultError("fault input has an unsupported prediction-bundle media type")
    if cast(int, payload["byte_size"]) > _MAX_BUNDLE_BYTES:
        raise FaultError("fault input exceeds the prediction-bundle byte limit")
    raw = registry.store.object_path(cast(str, payload["sha256"])).read_bytes()
    try:
        return PredictionBundle.model_validate(_strict_object(raw, "prediction bundle"))
    except ValueError as error:
        raise FaultError(f"prediction bundle schema is invalid: {error}") from error


def put_prediction_bundle(
    registry: EvidenceRegistry,
    bundle: PredictionBundle,
    *,
    parents: tuple[str, ...] = (),
) -> str:
    """Store one strict bundle through the production artifact path."""
    receipt = registry.put_bytes(
        canonical_json_bytes(bundle.model_dump(mode="json")) + b"\n",
        kind="prediction_bundle",
        media_type=_MEDIA_TYPE,
        license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
        metadata={
            "bundle_id": bundle.bundle_id,
            "frame_count": len(bundle.frames),
            "fault_kind": (
                None if not bundle.fault_history else bundle.fault_history[0].kind.value
            ),
        },
        parents=parents,
    )
    return receipt.manifest_sha256


def inject_fault(
    *,
    artifact_root: Path,
    schema_path: Path,
    input_manifest_sha256: str,
    kind: FaultKind,
    seed: int = 20260813,
    fraction: float = 0.5,
) -> FaultReceipt:
    """Create a derived bundle, prove detection, and persist its counterexample report."""
    registry = EvidenceRegistry(artifact_root, schema_path)
    parent = load_prediction_bundle(registry, input_manifest_sha256)
    clean = verify_clean_bundle(parent)
    if clean["status"] != "PASS":
        raise FaultError("fault input failed the nearby clean-control gate")
    try:
        child = apply_fault(parent, kind, seed=seed, fraction=fraction)
    except FaultTransformError as error:
        raise FaultError(str(error)) from error
    report = dict(analyze_fault(parent, child))
    if report["status"] == "FAILED_TO_DETECT":
        raise FaultError(f"fault analyzer did not detect {kind.value}: {report['checks']}")
    derived_manifest = put_prediction_bundle(
        registry,
        child,
        parents=(input_manifest_sha256,),
    )
    report.update(
        {
            "parent_manifest_sha256": input_manifest_sha256,
            "derived_manifest_sha256": derived_manifest,
            "seed": seed,
            "fraction": fraction,
            "clean_control": clean,
        }
    )
    counterexample = registry.put_bytes(
        canonical_json_bytes(report) + b"\n",
        kind="counterexample_bundle",
        media_type="application/vnd.junctionlens.fault-detection+json",
        license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
        metadata={
            "fault_kind": kind.value,
            "primary_reason_code": report["primary_reason_code"],
            "status": report["status"],
        },
        parents=(input_manifest_sha256, derived_manifest),
    )
    return FaultReceipt(
        state=cast(str, report["status"]),
        fault_kind=kind.value,
        primary_reason_code=cast(str, report["primary_reason_code"]),
        parent_manifest_sha256=input_manifest_sha256,
        derived_manifest_sha256=derived_manifest,
        counterexample_manifest_sha256=counterexample.manifest_sha256,
    )


def receipt_dict(receipt: FaultReceipt) -> dict[str, str]:
    return {key: cast(str, value) for key, value in asdict(receipt).items()}


__all__ = [
    "FaultError",
    "FaultReceipt",
    "inject_fault",
    "load_prediction_bundle",
    "put_prediction_bundle",
    "receipt_dict",
]

"""Streaming frame manifests and frozen segment-isolated split contracts."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from junctionlens.data.contracts import AdaptedFrame
from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object,
    load_json_object_path,
    load_yaml_object_path,
)

_SHA256_LENGTH = 64
_MAX_SPLIT_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_FRAME_RECORD_BYTES = 64 * 1024
_FRAME_RECORD_KEYS = {
    "annotations_valid",
    "calibration_sha256",
    "frame_metadata_sha256",
    "pose_valid",
    "segment_id",
    "source_domain",
    "source_segment_id_sha256",
    "split_id",
    "timestamp_ns",
    "valid_camera_slots",
}
_FRAME_METADATA_KEYS = {
    "dataset_id",
    "dataset_version",
    "frame_count",
    "ordering",
    "profile",
    "record_media_type",
    "record_schema",
    "schema_version",
    "segment_catalog",
    "segment_catalog_sha256",
    "segment_count",
    "source_registration",
    "split_segment_counts",
}
_POLICY_KEYS = {
    "algorithm",
    "allocations",
    "dataset_id",
    "dataset_version",
    "expected_segment_count",
    "forbidden_assignment_fields",
    "hash_domain",
    "legal_assignment_fields",
    "policy_id",
    "schema_version",
    "source_profile",
    "source_split",
    "stratification_fields",
}
_REQUIRED_PARTITIONS = (
    "model_training",
    "model_selection",
    "calibration",
    "internal_holdout",
)
_REQUIRED_LEGAL_FIELDS = ("segment_id", "source_domain")
_REQUIRED_FORBIDDEN_FIELDS = frozenset(
    {"annotations", "images", "label_statistics", "model_outputs", "partition_statistics"}
)


class ManifestError(RuntimeError):
    """Raised when a frame or split manifest violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class PartitionPolicy:
    """One named split allocation and its exclusive purpose."""

    identifier: str
    count: int
    purpose: str


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """The complete immutable V1 split-generation policy."""

    policy_id: str
    dataset_id: str
    dataset_version: str
    source_profile: str
    source_split: str
    expected_segment_count: int
    algorithm: str
    hash_domain: str
    legal_assignment_fields: tuple[str, ...]
    stratification_fields: tuple[str, ...]
    forbidden_assignment_fields: tuple[str, ...]
    allocations: tuple[PartitionPolicy, ...]


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """Only pre-assignment fields authorized to influence a V1 split."""

    segment_id: str
    source_domain: str

    def __post_init__(self) -> None:
        if not self.segment_id or Path(self.segment_id).name != self.segment_id:
            raise ManifestError("segment ID must be one safe path component")
        if not self.source_domain or len(self.source_domain) > 128:
            raise ManifestError("source domain must be a short nonempty value")


@dataclass(frozen=True, slots=True)
class FrameSegmentRecord:
    """Stable source identity retained once per split and segment in a frame manifest."""

    split_id: str
    segment_id: str
    source_domain: str
    source_segment_id_sha256: str


@dataclass(frozen=True, slots=True)
class SplitAudit:
    """Deterministic integrity evidence for one split manifest."""

    state: str
    segment_count: int
    partition_counts: Mapping[str, int]
    overlap_count: int
    segment_catalog_sha256: str


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ManifestError(f"{label} must be an array of strings")
    result = tuple(cast(list[str], value))
    if not result or len(result) != len(set(result)):
        raise ManifestError(f"{label} must contain unique values")
    return result


def load_split_policy(path: Path) -> SplitPolicy:
    """Load a strict policy that cannot admit result-derived assignment fields."""
    try:
        value = load_yaml_object_path(
            path,
            "split policy",
            ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
        )
    except ParseBoundaryError as error:
        raise ManifestError(str(error)) from error
    if set(value) != _POLICY_KEYS:
        raise ManifestError("split policy has invalid top-level keys")
    if value["schema_version"] != "junctionlens.openlane-split-policy.v1":
        raise ManifestError("split policy schema is unsupported")
    if value["algorithm"] != "sequential-hamilton-sha256-v1":
        raise ManifestError("split policy algorithm is unsupported")
    legal = _strings(value["legal_assignment_fields"], "legal assignment fields")
    stratification = _strings(value["stratification_fields"], "stratification fields")
    forbidden = _strings(value["forbidden_assignment_fields"], "forbidden assignment fields")
    if legal != _REQUIRED_LEGAL_FIELDS or stratification != ("source_domain",):
        raise ManifestError("V1 assignments may use only segment_id and source_domain")
    if not _REQUIRED_FORBIDDEN_FIELDS.issubset(forbidden):
        raise ManifestError("split policy does not forbid every result-derived field")
    raw_allocations = value["allocations"]
    if not isinstance(raw_allocations, list):
        raise ManifestError("split policy allocations must be an array")
    allocations: list[PartitionPolicy] = []
    for index, raw in enumerate(raw_allocations):
        if not isinstance(raw, dict) or set(raw) != {"count", "id", "purpose"}:
            raise ManifestError(f"split policy allocation {index} has invalid keys")
        identifier = str(raw["id"])
        purpose = str(raw["purpose"])
        count = raw["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ManifestError(f"split policy allocation {identifier} has invalid count")
        if not identifier or not purpose:
            raise ManifestError("split allocation ID and purpose must be nonempty")
        allocations.append(PartitionPolicy(identifier, count, purpose))
    if tuple(item.identifier for item in allocations) != _REQUIRED_PARTITIONS:
        raise ManifestError("split policy partition order differs from V1")
    expected = value["expected_segment_count"]
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise ManifestError("expected segment count must be a positive integer")
    if sum(item.count for item in allocations) != expected:
        raise ManifestError("split allocation counts do not sum to the expected segment count")
    if (
        str(value["policy_id"]) != "openlane-v2-v2.1-split-v1"
        or str(value["dataset_id"]) != "openlane-v2-v2.1"
        or str(value["dataset_version"]) != "2.1"
        or str(value["source_profile"]) != "full"
        or str(value["source_split"]) != "train"
        or expected != 700
        or tuple(item.count for item in allocations) != (350, 80, 70, 200)
    ):
        raise ManifestError("split policy identity or V1 allocation counts changed")
    hash_domain = str(value["hash_domain"])
    if not hash_domain or len(hash_domain) > 255:
        raise ManifestError("split hash domain must be a short nonempty value")
    return SplitPolicy(
        policy_id=str(value["policy_id"]),
        dataset_id=str(value["dataset_id"]),
        dataset_version=str(value["dataset_version"]),
        source_profile=str(value["source_profile"]),
        source_split=str(value["source_split"]),
        expected_segment_count=expected,
        algorithm=str(value["algorithm"]),
        hash_domain=hash_domain,
        legal_assignment_fields=legal,
        stratification_fields=stratification,
        forbidden_assignment_fields=forbidden,
        allocations=tuple(allocations),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _frame_record(frame: AdaptedFrame) -> Mapping[str, Any]:
    valid_slots = [camera.slot.value for camera in frame.cameras if camera.valid]
    source_segment_hash = hashlib.sha256(
        frame.source_metadata.source_segment_id.encode("utf-8")
    ).hexdigest()
    return {
        "split_id": frame.key.split_id,
        "segment_id": frame.key.segment_id,
        "timestamp_ns": frame.key.timestamp_ns,
        "source_domain": frame.key.source_domain,
        "source_segment_id_sha256": source_segment_hash,
        "frame_metadata_sha256": frame.key.frame_manifest_sha256,
        "calibration_sha256": frame.key.calibration_sha256,
        "valid_camera_slots": valid_slots,
        "pose_valid": frame.pose_valid,
        "annotations_valid": frame.annotations_valid,
    }


def write_frame_records(
    adapter: OpenLaneAdapter,
    profile: str,
    output_path: Path,
    registration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Stream deterministic frame records without retaining frames or annotations."""
    if registration.get("profile") != profile:
        raise ManifestError("registration profile differs from frame manifest profile")
    required_registration = {
        key: registration.get(key)
        for key in (
            "archive_sha256",
            "dataset_id",
            "license_receipt_sha256",
            "manifest_sha256",
        )
    }
    if required_registration["dataset_id"] != "openlane-v2-v2.1" or any(
        not _is_sha256(required_registration[key])
        for key in ("archive_sha256", "license_receipt_sha256", "manifest_sha256")
    ):
        raise ManifestError("registration lacks stable checksum evidence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise ManifestError("frame-record output already exists")
    segment_state: dict[tuple[str, str], tuple[str, str]] = {}
    frame_count = 0
    with output_path.open("xb") as output:
        try:
            for frame in adapter.iter_frames(profile):
                source_hash = hashlib.sha256(
                    frame.source_metadata.source_segment_id.encode("utf-8")
                ).hexdigest()
                identity = (frame.key.source_domain, source_hash)
                segment_key = (frame.key.split_id, frame.key.segment_id)
                previous = segment_state.setdefault(segment_key, identity)
                if previous != identity:
                    raise ManifestError("source identity changes within one segment")
                output.write(canonical_json_bytes(_frame_record(frame)) + b"\n")
                frame_count += 1
            output.flush()
            os.fsync(output.fileno())
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
    if frame_count == 0 or not segment_state:
        output_path.unlink(missing_ok=True)
        raise ManifestError("frame manifest requires at least one frame and segment")
    segment_catalog = [
        asdict(FrameSegmentRecord(split_id, segment_id, state[0], state[1]))
        for (split_id, segment_id), state in sorted(segment_state.items())
    ]
    split_segment_counts: dict[str, int] = defaultdict(int)
    for split_id, _ in segment_state:
        split_segment_counts[split_id] += 1
    return {
        "schema_version": "junctionlens.frame-manifest.v1",
        "dataset_id": "openlane-v2-v2.1",
        "dataset_version": "2.1",
        "profile": profile,
        "ordering": "split_id,segment_id,numeric_timestamp",
        "record_media_type": "application/x-ndjson",
        "record_schema": "junctionlens.frame-content-record.v1",
        "frame_count": frame_count,
        "segment_count": len(segment_catalog),
        "split_segment_counts": dict(sorted(split_segment_counts.items())),
        "segment_catalog": segment_catalog,
        "segment_catalog_sha256": _sha256_json(segment_catalog),
        "source_registration": required_registration,
    }


def _hamilton_counts(
    domain_counts: Mapping[str, int], allocations: Sequence[PartitionPolicy]
) -> Mapping[str, Mapping[str, int]]:
    remaining = {allocation.identifier: allocation.count for allocation in allocations}
    result: dict[str, dict[str, int]] = {}
    ordered_domains = sorted(domain_counts)
    for index, domain in enumerate(ordered_domains):
        count = domain_counts[domain]
        if index == len(ordered_domains) - 1:
            assigned = dict(remaining)
        else:
            remaining_total = sum(remaining.values())
            ideals = {
                partition: count * capacity / remaining_total
                for partition, capacity in remaining.items()
            }
            assigned = {
                partition: min(remaining[partition], math.floor(ideal))
                for partition, ideal in ideals.items()
            }
            unassigned = count - sum(assigned.values())
            order = sorted(
                remaining,
                key=lambda partition: (
                    -(ideals[partition] - math.floor(ideals[partition])),
                    partition,
                ),
            )
            for partition in order:
                if unassigned == 0:
                    break
                if assigned[partition] < remaining[partition]:
                    assigned[partition] += 1
                    unassigned -= 1
            if unassigned:
                raise ManifestError("stratified apportionment could not satisfy exact counts")
        if sum(assigned.values()) != count:
            raise ManifestError("stratified apportionment produced an invalid domain row")
        result[domain] = assigned
        for partition, assigned_count in assigned.items():
            remaining[partition] -= assigned_count
            if remaining[partition] < 0:
                raise ManifestError("stratified apportionment exceeded a partition capacity")
    if any(remaining.values()):
        raise ManifestError("stratified apportionment did not fill every partition")
    return result


def split_records_from_frame_metadata(
    metadata: Mapping[str, Any], policy: SplitPolicy
) -> tuple[SegmentRecord, ...]:
    """Select the policy source split from a validated frame-manifest artifact."""
    if set(metadata) != _FRAME_METADATA_KEYS:
        raise ManifestError("frame manifest metadata has invalid keys")
    if (
        metadata.get("schema_version") != "junctionlens.frame-manifest.v1"
        or metadata.get("dataset_id") != policy.dataset_id
        or metadata.get("dataset_version") != policy.dataset_version
        or metadata.get("profile") != policy.source_profile
    ):
        raise ManifestError("frame manifest metadata differs from the split policy")
    raw_catalog = metadata.get("segment_catalog")
    if not isinstance(raw_catalog, list):
        raise ManifestError("frame manifest has no segment catalog")
    if metadata.get("segment_catalog_sha256") != _sha256_json(raw_catalog):
        raise ManifestError("frame manifest segment catalog hash mismatch")
    result: list[SegmentRecord] = []
    observed_keys: set[tuple[str, str]] = set()
    observed_split_counts: dict[str, int] = defaultdict(int)
    for item in raw_catalog:
        if not isinstance(item, dict) or set(item) != {
            "segment_id",
            "source_domain",
            "source_segment_id_sha256",
            "split_id",
        }:
            raise ManifestError("frame manifest segment catalog contains an invalid record")
        split_id = item["split_id"]
        source_hash = item["source_segment_id_sha256"]
        if not isinstance(split_id, str) or not split_id or Path(split_id).name != split_id:
            raise ManifestError("frame manifest segment catalog has an invalid split ID")
        if not _is_sha256(source_hash):
            raise ManifestError("frame manifest segment catalog has an invalid source hash")
        record = SegmentRecord(str(item["segment_id"]), str(item["source_domain"]))
        key = (split_id, record.segment_id)
        if key in observed_keys:
            raise ManifestError("frame manifest segment catalog repeats a source segment")
        observed_keys.add(key)
        observed_split_counts[split_id] += 1
        if split_id == policy.source_split:
            result.append(record)
    if metadata.get("segment_count") != len(raw_catalog):
        raise ManifestError("frame manifest segment count differs from its catalog")
    if metadata.get("split_segment_counts") != dict(sorted(observed_split_counts.items())):
        raise ManifestError("frame manifest split counts differ from its catalog")
    return tuple(result)


def verify_frame_records(path: Path, metadata: Mapping[str, Any]) -> None:
    """Stream and cross-check frame-record payload bytes against artifact metadata."""
    if path.is_symlink() or not path.is_file():
        raise ManifestError("frame-record payload must be a regular file")
    segment_state: dict[tuple[str, str], tuple[str, str]] = {}
    frame_count = 0
    previous_key: tuple[str, str, int] | None = None
    with path.open("rb") as source:
        while True:
            line = source.readline(_MAX_FRAME_RECORD_BYTES + 1)
            if not line:
                break
            if len(line) > _MAX_FRAME_RECORD_BYTES or not line.endswith(b"\n"):
                raise ManifestError("frame-record payload has an oversized or incomplete line")
            try:
                value = load_json_object(
                    line,
                    "frame record",
                    ParseLimits(
                        max_bytes=_MAX_FRAME_RECORD_BYTES,
                        max_depth=8,
                        max_nodes=128,
                        max_container_items=32,
                        max_string_bytes=4096,
                    ),
                )
            except ParseBoundaryError as error:
                raise ManifestError("frame-record payload contains invalid JSON") from error
            if set(value) != _FRAME_RECORD_KEYS:
                raise ManifestError("frame-record payload has invalid fields")
            split_id = value["split_id"]
            segment_id = value["segment_id"]
            source_domain = value["source_domain"]
            timestamp = value["timestamp_ns"]
            if (
                not isinstance(split_id, str)
                or not split_id
                or Path(split_id).name != split_id
                or not isinstance(segment_id, str)
                or not isinstance(source_domain, str)
            ):
                raise ManifestError("frame-record payload has invalid source identity")
            SegmentRecord(segment_id, source_domain)
            if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
                raise ManifestError("frame-record payload has an invalid timestamp")
            frame_key = (split_id, segment_id, timestamp)
            if previous_key is not None and frame_key <= previous_key:
                raise ManifestError("frame-record payload is not in strict source order")
            previous_key = frame_key
            for key in (
                "calibration_sha256",
                "frame_metadata_sha256",
                "source_segment_id_sha256",
            ):
                if not _is_sha256(value[key]):
                    raise ManifestError(f"frame-record payload has an invalid {key}")
            if not isinstance(value["annotations_valid"], bool) or not isinstance(
                value["pose_valid"], bool
            ):
                raise ManifestError("frame-record payload has an invalid validity flag")
            camera_slots = value["valid_camera_slots"]
            if (
                not isinstance(camera_slots, list)
                or not all(isinstance(slot, str) for slot in camera_slots)
                or len(camera_slots) != len(set(camera_slots))
            ):
                raise ManifestError("frame-record payload has invalid camera slots")
            identity = (source_domain, cast(str, value["source_segment_id_sha256"]))
            segment_key = (split_id, segment_id)
            previous_identity = segment_state.setdefault(segment_key, identity)
            if previous_identity != identity:
                raise ManifestError("frame-record source identity changes within one segment")
            frame_count += 1
    if metadata.get("frame_count") != frame_count:
        raise ManifestError("frame-record count differs from artifact metadata")
    catalog = [
        asdict(FrameSegmentRecord(split_id, segment_id, state[0], state[1]))
        for (split_id, segment_id), state in sorted(segment_state.items())
    ]
    if metadata.get("segment_catalog") != catalog:
        raise ManifestError("frame-record segment catalog differs from artifact metadata")
    if metadata.get("segment_catalog_sha256") != _sha256_json(catalog):
        raise ManifestError("frame-record segment catalog hash mismatch")


def _segment_score(policy: SplitPolicy, record: SegmentRecord) -> bytes:
    payload = (f"{policy.hash_domain}\0{record.source_domain}\0{record.segment_id}").encode()
    return hashlib.sha256(payload).digest()


def freeze_split_manifest(
    records: Iterable[SegmentRecord],
    policy: SplitPolicy,
    *,
    source_frame_manifest_sha256: str,
    source_frame_records_sha256: str,
    source_dataset_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Freeze exact partition lists from only the policy-authorized input fields."""
    for label, value in {
        "source frame manifest": source_frame_manifest_sha256,
        "source frame records": source_frame_records_sha256,
        "source dataset manifest": source_dataset_manifest_sha256,
    }.items():
        if len(value) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ManifestError(f"{label} hash must be lowercase SHA-256")
    catalog = sorted(records, key=lambda record: (record.segment_id, record.source_domain))
    if len(catalog) != policy.expected_segment_count:
        raise ManifestError(
            f"expected {policy.expected_segment_count} segments, observed {len(catalog)}"
        )
    identifiers = [record.segment_id for record in catalog]
    if len(identifiers) != len(set(identifiers)):
        raise ManifestError("segment catalog contains duplicate segment IDs")
    by_domain: dict[str, list[SegmentRecord]] = defaultdict(list)
    for record in catalog:
        by_domain[record.source_domain].append(record)
    counts = _hamilton_counts(
        {domain: len(domain_records) for domain, domain_records in by_domain.items()},
        policy.allocations,
    )
    partition_records: dict[str, list[SegmentRecord]] = {
        allocation.identifier: [] for allocation in policy.allocations
    }
    for domain in sorted(by_domain):
        ordered = sorted(
            by_domain[domain],
            key=lambda record: (_segment_score(policy, record), record.segment_id),
        )
        offset = 0
        for allocation in policy.allocations:
            next_offset = offset + counts[domain][allocation.identifier]
            partition_records[allocation.identifier].extend(ordered[offset:next_offset])
            offset = next_offset
        if offset != len(ordered):
            raise ManifestError("stratified apportionment did not consume a domain")
    partitions: dict[str, Any] = {}
    for allocation in policy.allocations:
        ordered = sorted(
            partition_records[allocation.identifier],
            key=lambda record: (_segment_score(policy, record), record.segment_id),
        )
        segment_ids = [record.segment_id for record in ordered]
        domain_counts: dict[str, int] = defaultdict(int)
        for record in ordered:
            domain_counts[record.source_domain] += 1
        partitions[allocation.identifier] = {
            "purpose": allocation.purpose,
            "segment_count": len(segment_ids),
            "segments": segment_ids,
            "segments_sha256": _sha256_json(segment_ids),
            "source_domain_counts": dict(sorted(domain_counts.items())),
        }
    segment_catalog = [asdict(record) for record in catalog]
    manifest: Mapping[str, Any] = {
        "schema_version": "junctionlens.openlane-split-manifest.v1",
        "policy_id": policy.policy_id,
        "dataset_id": policy.dataset_id,
        "dataset_version": policy.dataset_version,
        "source_profile": policy.source_profile,
        "source_split": policy.source_split,
        "algorithm": policy.algorithm,
        "hash_domain": policy.hash_domain,
        "legal_assignment_fields": list(policy.legal_assignment_fields),
        "stratification_fields": list(policy.stratification_fields),
        "source_frame_manifest_sha256": source_frame_manifest_sha256,
        "source_frame_records_sha256": source_frame_records_sha256,
        "source_dataset_manifest_sha256": source_dataset_manifest_sha256,
        "segment_catalog": segment_catalog,
        "segment_catalog_sha256": _sha256_json(segment_catalog),
        "partitions": partitions,
        "frozen": True,
    }
    audit_split_manifest(manifest, policy)
    return manifest


def audit_split_manifest(
    manifest: Mapping[str, Any],
    policy: SplitPolicy,
) -> SplitAudit:
    """Reject count, hash, catalog, or cross-partition leakage defects."""
    expected_keys = {
        "algorithm",
        "dataset_id",
        "dataset_version",
        "frozen",
        "hash_domain",
        "legal_assignment_fields",
        "partitions",
        "policy_id",
        "schema_version",
        "segment_catalog",
        "segment_catalog_sha256",
        "source_dataset_manifest_sha256",
        "source_frame_manifest_sha256",
        "source_frame_records_sha256",
        "source_profile",
        "source_split",
        "stratification_fields",
    }
    if set(manifest) != expected_keys:
        raise ManifestError("split manifest has invalid top-level keys")
    identity = {
        "schema_version": "junctionlens.openlane-split-manifest.v1",
        "policy_id": policy.policy_id,
        "dataset_id": policy.dataset_id,
        "dataset_version": policy.dataset_version,
        "source_profile": policy.source_profile,
        "source_split": policy.source_split,
        "algorithm": policy.algorithm,
        "hash_domain": policy.hash_domain,
        "frozen": True,
        "legal_assignment_fields": list(policy.legal_assignment_fields),
        "stratification_fields": list(policy.stratification_fields),
    }
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise ManifestError(f"split manifest {key} differs from its frozen policy")
    for key in (
        "source_dataset_manifest_sha256",
        "source_frame_manifest_sha256",
        "source_frame_records_sha256",
    ):
        value = manifest[key]
        if (
            not isinstance(value, str)
            or len(value) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ManifestError(f"split manifest {key} must be lowercase SHA-256")
    raw_catalog = manifest["segment_catalog"]
    if not isinstance(raw_catalog, list):
        raise ManifestError("split segment catalog must be an array")
    try:
        catalog = [
            SegmentRecord(str(item["segment_id"]), str(item["source_domain"]))
            for item in raw_catalog
            if isinstance(item, dict) and set(item) == {"segment_id", "source_domain"}
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError("split segment catalog contains an invalid record") from error
    if len(catalog) != len(raw_catalog) or len(catalog) != policy.expected_segment_count:
        raise ManifestError("split segment catalog has an invalid size or record")
    if raw_catalog != [
        asdict(record)
        for record in sorted(catalog, key=lambda item: (item.segment_id, item.source_domain))
    ]:
        raise ManifestError("split segment catalog is not in canonical order")
    catalog_hash = _sha256_json(raw_catalog)
    if manifest["segment_catalog_sha256"] != catalog_hash:
        raise ManifestError("split segment catalog hash mismatch")
    catalog_domains = {record.segment_id: record.source_domain for record in catalog}
    if len(catalog_domains) != len(catalog):
        raise ManifestError("split segment catalog contains duplicate segment IDs")
    raw_partitions = manifest["partitions"]
    if not isinstance(raw_partitions, dict) or set(raw_partitions) != set(_REQUIRED_PARTITIONS):
        raise ManifestError("split manifest partitions differ from V1")
    seen: dict[str, str] = {}
    partition_counts: dict[str, int] = {}
    overlap_count = 0
    for allocation in policy.allocations:
        raw_partition = raw_partitions.get(allocation.identifier)
        if not isinstance(raw_partition, dict) or set(raw_partition) != {
            "purpose",
            "segment_count",
            "segments",
            "segments_sha256",
            "source_domain_counts",
        }:
            raise ManifestError(f"split partition {allocation.identifier} has invalid keys")
        if raw_partition["purpose"] != allocation.purpose:
            raise ManifestError(f"split partition {allocation.identifier} purpose mismatch")
        segments = raw_partition["segments"]
        if not isinstance(segments, list) or not all(isinstance(item, str) for item in segments):
            raise ManifestError(f"split partition {allocation.identifier} has invalid segments")
        segment_ids = cast(list[str], segments)
        if len(segment_ids) != len(set(segment_ids)):
            raise ManifestError(f"split partition {allocation.identifier} repeats a segment")
        if len(segment_ids) != allocation.count or raw_partition["segment_count"] != len(
            segment_ids
        ):
            raise ManifestError(f"split partition {allocation.identifier} count mismatch")
        if raw_partition["segments_sha256"] != _sha256_json(segment_ids):
            raise ManifestError(f"split partition {allocation.identifier} hash mismatch")
        observed_domains: dict[str, int] = defaultdict(int)
        for segment_id in segment_ids:
            if segment_id not in catalog_domains:
                raise ManifestError(f"split partition {allocation.identifier} has unknown segment")
            observed_domains[catalog_domains[segment_id]] += 1
            if segment_id in seen:
                overlap_count += 1
            else:
                seen[segment_id] = allocation.identifier
        if raw_partition["source_domain_counts"] != dict(sorted(observed_domains.items())):
            raise ManifestError(f"split partition {allocation.identifier} stratum counts mismatch")
        partition_counts[allocation.identifier] = len(segment_ids)
    if overlap_count:
        raise ManifestError(
            f"segment leakage detected across partitions: {overlap_count} overlap(s)"
        )
    if set(seen) != set(catalog_domains):
        raise ManifestError("split partitions do not cover the exact segment catalog")
    return SplitAudit(
        state="ACCEPTED",
        segment_count=len(seen),
        partition_counts=partition_counts,
        overlap_count=0,
        segment_catalog_sha256=catalog_hash,
    )


def load_split_manifest(path: Path) -> Mapping[str, Any]:
    """Load one bounded JSON split artifact for independent audit."""
    try:
        return load_json_object_path(
            path,
            "split manifest",
            ParseLimits(
                max_bytes=_MAX_SPLIT_MANIFEST_BYTES,
                max_depth=24,
                max_nodes=2_000_000,
                max_container_items=1_000_000,
            ),
        )
    except ParseBoundaryError as error:
        raise ManifestError(str(error)) from error


def write_immutable_split_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    """Create a frozen export once, or verify an identical existing export."""
    payload = canonical_json_bytes(manifest) + b"\n"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ManifestError("split export cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ManifestError("immutable split export already exists with different content")
        return expected_sha256
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, path)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ManifestError("immutable split export raced with different content") from None
    finally:
        temporary_path.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return expected_sha256


__all__ = [
    "FrameSegmentRecord",
    "ManifestError",
    "PartitionPolicy",
    "SegmentRecord",
    "SplitAudit",
    "SplitPolicy",
    "audit_split_manifest",
    "freeze_split_manifest",
    "load_split_manifest",
    "load_split_policy",
    "split_records_from_frame_metadata",
    "verify_frame_records",
    "write_frame_records",
    "write_immutable_split_manifest",
]

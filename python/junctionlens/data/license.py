"""Explicit local license acknowledgment and checksum-verified registration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object_path,
    load_yaml_object_path,
)


class DatasetRegistrationError(RuntimeError):
    """Raised when license or local dataset evidence is incomplete."""


def _load_lock(lock_path: Path) -> Mapping[str, Any]:
    try:
        return load_yaml_object_path(
            lock_path,
            "dataset lock",
            ParseLimits(max_bytes=4 * 1024 * 1024, max_depth=24, max_nodes=100_000),
        )
    except ParseBoundaryError as error:
        raise DatasetRegistrationError(str(error)) from error


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _license_contract(lock: Mapping[str, Any]) -> Mapping[str, Any]:
    devkit = lock.get("devkit")
    if not isinstance(devkit, dict):
        raise DatasetRegistrationError("dataset lock has no devkit contract")
    licenses = lock.get("dataset_licenses")
    if not isinstance(licenses, list) or not all(isinstance(item, str) for item in licenses):
        raise DatasetRegistrationError("dataset lock has invalid license identifiers")
    return {
        "dataset_id": lock.get("dataset_id"),
        "dataset_version": lock.get("dataset_version"),
        "devkit_commit": devkit.get("commit"),
        "dataset_licenses": sorted(licenses),
        "redistribution_allowed": lock.get("redistribution_allowed"),
    }


def _private_path(repository_root: Path, relative: Path, *, create_parents: bool) -> Path:
    root = repository_root.resolve(strict=True)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise DatasetRegistrationError("receipt path must remain below the repository")
    current = root
    missing_parent = False
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise DatasetRegistrationError("receipt path cannot traverse a symlink")
        if create_parents:
            current.mkdir(mode=0o700, exist_ok=True)
            current.chmod(0o700)
        elif not current.exists():
            missing_parent = True
            continue
        if current.exists() and not current.is_dir():
            raise DatasetRegistrationError("receipt parent must be a directory")
    if not missing_parent:
        try:
            current.resolve(strict=True).relative_to(root)
        except ValueError as error:
            raise DatasetRegistrationError("receipt path escapes the repository") from error
    path = current / relative.name
    if path.is_symlink():
        raise DatasetRegistrationError("receipt path cannot be a symlink")
    return path


def _write_private_json(repository_root: Path, relative: Path, payload: Mapping[str, Any]) -> None:
    path = _private_path(repository_root, relative, create_parents=True)
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(serialized)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o600)
    temporary_path.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def acknowledge_licenses(
    lock_path: Path,
    repository_root: Path,
    accepted_terms: Iterable[str],
    *,
    confirmed_restricted_noncommercial_use: bool,
) -> Mapping[str, Any]:
    """Write a machine-local receipt only after exact explicit acknowledgments."""
    lock = _load_lock(lock_path)
    contract = _license_contract(lock)
    required = frozenset(cast(list[str], contract["dataset_licenses"]))
    accepted = frozenset(accepted_terms)
    if accepted != required:
        missing = sorted(required - accepted)
        unexpected = sorted(accepted - required)
        raise DatasetRegistrationError(
            f"license terms do not match lock; missing={missing}, unexpected={unexpected}"
        )
    if not confirmed_restricted_noncommercial_use:
        raise DatasetRegistrationError(
            "explicit confirmation of restricted noncommercial use is required"
        )
    receipt_relative = lock.get("acknowledgment_receipt")
    if not isinstance(receipt_relative, str):
        raise DatasetRegistrationError("dataset lock has no receipt path")
    receipt_path = Path(receipt_relative)
    payload: Mapping[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_id": contract["dataset_id"],
        "license_contract_sha256": _canonical_hash(contract),
        "accepted_terms": sorted(accepted),
        "confirmed_restricted_noncommercial_use": True,
        "redistribution_allowed": False,
        "acknowledged_at": datetime.now(UTC).isoformat(),
    }
    _write_private_json(repository_root, receipt_path, payload)
    return payload


def load_valid_acknowledgment(
    lock_path: Path,
    repository_root: Path,
) -> Mapping[str, Any]:
    """Validate the machine-local receipt against the current license contract."""
    lock = _load_lock(lock_path)
    receipt_relative = lock.get("acknowledgment_receipt")
    if not isinstance(receipt_relative, str):
        raise DatasetRegistrationError("dataset lock has no receipt path")
    receipt_path = _private_path(
        repository_root,
        Path(receipt_relative),
        create_parents=False,
    )
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise DatasetRegistrationError(
            "license acknowledgment is missing; run `junctionlens data acknowledge`"
        )
    try:
        receipt = load_json_object_path(
            receipt_path,
            "license receipt",
            ParseLimits(max_bytes=64 * 1024, max_depth=12, max_nodes=10_000),
        )
    except ParseBoundaryError as error:
        raise DatasetRegistrationError(str(error)) from error
    expected_contract_hash = _canonical_hash(_license_contract(lock))
    if receipt.get("license_contract_sha256") != expected_contract_hash:
        raise DatasetRegistrationError("license receipt does not match the current lock contract")
    if receipt.get("confirmed_restricted_noncommercial_use") is not True:
        raise DatasetRegistrationError("license receipt lacks explicit use confirmation")
    return receipt


def _hash_file(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest(), size


def register_dataset(
    lock_path: Path,
    repository_root: Path,
    dataset_root: Path,
    *,
    profile: str,
    archive_path: Path | None,
) -> Mapping[str, Any]:
    """Verify one extracted dataset profile and its original archive checksum."""
    lock = _load_lock(lock_path)
    acknowledgment = load_valid_acknowledgment(lock_path, repository_root)
    root = dataset_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DatasetRegistrationError("dataset root is not a directory")
    manifest_name = {"sample": "data_dict_example.json", "full": "data_dict_subset_A.json"}.get(
        profile
    )
    archive_name = {"sample": "openlane-v2-sample", "full": "subset-a-metadata"}.get(profile)
    expected_filename = {
        "sample": "OpenLane-V2_sample.tar",
        "full": "OpenLane-V2.tar",
    }.get(profile)
    if manifest_name is None or archive_name is None or expected_filename is None:
        raise DatasetRegistrationError(f"unsupported dataset profile: {profile}")
    manifest_path = root / manifest_name
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise DatasetRegistrationError(f"dataset root is missing {manifest_name}")
    if archive_path is None:
        candidates = [root / expected_filename, root.parent / expected_filename]
        archive = next((candidate for candidate in candidates if candidate.is_file()), None)
        if archive is None:
            raise DatasetRegistrationError(
                f"cannot verify the official checksum; pass --archive or retain {expected_filename}"
            )
    else:
        if archive_path.is_symlink():
            raise DatasetRegistrationError("dataset archive cannot be a symlink")
        archive = archive_path.expanduser().resolve(strict=True)
    if not archive.is_file() or archive.is_symlink():
        raise DatasetRegistrationError("dataset archive must be an ordinary file")

    raw_archives = lock.get("archives")
    if not isinstance(raw_archives, list):
        raise DatasetRegistrationError("dataset lock has no archive entries")
    archive_entries = [
        cast(Mapping[str, Any], entry)
        for entry in raw_archives
        if isinstance(entry, dict) and entry.get("name") == archive_name
    ]
    if len(archive_entries) != 1:
        raise DatasetRegistrationError(f"dataset lock has no unique {archive_name} entry")
    md5, sha256, byte_size = _hash_file(archive)
    expected_md5 = archive_entries[0].get("published_md5")
    if md5 != expected_md5:
        raise DatasetRegistrationError(
            f"archive checksum mismatch for {archive_name}: expected {expected_md5}, observed {md5}"
        )
    manifest_sha256 = _hash_file(manifest_path)[1]
    registration: Mapping[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_id": lock.get("dataset_id"),
        "profile": profile,
        "root": str(root),
        "archive_name": archive_name,
        "archive_md5": md5,
        "archive_sha256": sha256,
        "archive_byte_size": byte_size,
        "manifest_sha256": manifest_sha256,
        "license_receipt_sha256": _canonical_hash(acknowledgment),
        "registered_at": datetime.now(UTC).isoformat(),
    }
    receipt_path = (
        Path(".junctionlens") / "registrations" / f"{lock.get('dataset_id')}-{profile}.json"
    )
    _write_private_json(repository_root, receipt_path, registration)
    return registration


def load_registration(
    repository_root: Path,
    dataset_id: str,
    profile: str,
) -> Mapping[str, Any]:
    """Load a bounded machine-local registration receipt."""
    receipt_path = _private_path(
        repository_root,
        Path(".junctionlens") / "registrations" / f"{dataset_id}-{profile}.json",
        create_parents=False,
    )
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise DatasetRegistrationError(
            "dataset profile is not registered; run "
            f"`junctionlens data register --profile {profile}`"
        )
    try:
        value = load_json_object_path(
            receipt_path,
            "dataset registration receipt",
            ParseLimits(max_bytes=64 * 1024, max_depth=12, max_nodes=10_000),
        )
    except ParseBoundaryError as error:
        raise DatasetRegistrationError(str(error)) from error
    if value.get("dataset_id") != dataset_id:
        raise DatasetRegistrationError("dataset registration receipt identity mismatch")
    return value

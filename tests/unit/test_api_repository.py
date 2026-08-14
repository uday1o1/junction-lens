"""Containment and persisted-decision tests for the read-only repository."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from junctionlens.api.models import ServiceConfig
from junctionlens.api.repository import EvidenceReadError, EvidenceRepository, VerifiedPayload
from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import canonical_json_bytes

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "schemas/artifact-manifest-v1.schema.json"


def _decision(*, correct_identity: bool = True) -> bytes:
    body: dict[str, object] = {
        "schema_version": "junctionlens.gate-decision.v1",
        "status": "FAIL_REGRESSION",
        "cells": [],
        "integrity_reason_codes": [],
        "infrastructure_reason_codes": [],
        "performance_reason_codes": [],
    }
    identity = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    body["decision_sha256"] = identity if correct_identity else "0" * 64
    return canonical_json_bytes(body) + b"\n"


def _registry(tmp_path: Path) -> tuple[Path, EvidenceRegistry]:
    root = tmp_path / "artifacts"
    return root, EvidenceRegistry(root, SCHEMA)


def test_repository_rejects_symlink_root_and_payload(tmp_path: Path) -> None:
    root, registry = _registry(tmp_path)
    receipt = registry.put_bytes(
        _decision(),
        kind="release_decision",
        media_type="application/vnd.junctionlens.gate-decision+json",
        license_id="Apache-2.0",
        metadata={"status": "FAIL_REGRESSION"},
    )
    repository = EvidenceRepository(ServiceConfig(artifact_root=root, schema_path=SCHEMA))
    payload_path = registry.store.object_path(receipt.payload_sha256)
    outside = tmp_path / "outside.json"
    outside.write_bytes(_decision())
    payload_path.unlink()
    payload_path.symlink_to(outside)

    with pytest.raises(EvidenceReadError, match="integrity verification"):
        repository.artifact(receipt.manifest_sha256)

    alias = tmp_path / "artifact-alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(EvidenceReadError, match="cannot be a symlink"):
        EvidenceRepository(ServiceConfig(artifact_root=alias, schema_path=SCHEMA))


def test_repository_serves_only_decision_with_matching_persisted_identity(tmp_path: Path) -> None:
    root, registry = _registry(tmp_path)
    valid = registry.put_bytes(
        _decision(),
        kind="release_decision",
        media_type="application/vnd.junctionlens.gate-decision+json",
        license_id="Apache-2.0",
        metadata={"status": "FAIL_REGRESSION"},
    )
    invalid = registry.put_bytes(
        _decision(correct_identity=False),
        kind="release_decision",
        media_type="application/vnd.junctionlens.gate-decision+json",
        license_id="Apache-2.0",
        metadata={"status": "FAIL_REGRESSION"},
    )
    repository = EvidenceRepository(ServiceConfig(artifact_root=root, schema_path=SCHEMA))

    assert repository.decision(valid.manifest_sha256)["status"] == "FAIL_REGRESSION"
    with pytest.raises(EvidenceReadError, match="identity does not match"):
        repository.decision(invalid.manifest_sha256)


def test_repository_enforces_payload_byte_limit(tmp_path: Path) -> None:
    root, registry = _registry(tmp_path)
    receipt = registry.put_bytes(
        b"bounded evidence",
        kind="evidence_report",
        media_type="text/plain",
        license_id="Apache-2.0",
        metadata={},
    )
    repository = EvidenceRepository(
        ServiceConfig(
            artifact_root=root,
            schema_path=SCHEMA,
            max_artifact_bytes=4,
        )
    )

    with pytest.raises(EvidenceReadError, match="response byte limit"):
        repository.open_payload(receipt.manifest_sha256)


def test_verified_payload_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    root = tmp_path / "registered"
    contained = root / "objects" / "sha256" / "aa"
    contained.mkdir(parents=True)
    payload = b"registered evidence"
    digest = hashlib.sha256(payload).hexdigest()
    path = contained / digest
    path.write_bytes(payload)

    assert VerifiedPayload(root, path, digest, len(payload), 1024).read() == payload

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / digest).write_bytes(payload)
    path.unlink()
    contained.rmdir()
    contained.symlink_to(outside, target_is_directory=True)

    with pytest.raises(EvidenceReadError, match="opened safely"):
        VerifiedPayload(root, path, digest, len(payload), 1024)

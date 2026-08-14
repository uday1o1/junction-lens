"""High-level immutable evidence registry, resumable runs, and GC audit."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from junctionlens.registry.index import RegistryIndex
from junctionlens.registry.store import ArtifactReceipt, ContentAddressedStore, canonical_json_bytes

_OBJECT_DIRECTORY = re.compile(r"^[0-9a-f]{2}$")
_OBJECT_NAME = re.compile(r"^[0-9a-f]{62}$")
_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")


class EvidenceRegistryError(RuntimeError):
    """Raised when a registry workflow cannot preserve provenance or resume safety."""


class RunIdentity(BaseModel):
    """Complete immutable input identity for one resumable command run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["junctionlens.run-identity.v1"]
    run_kind: str = Field(min_length=1, max_length=128)
    parent_artifact_hashes: tuple[str, ...]
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    model_profile_sha256: str
    configuration_sha256: str
    source_git_commit: str
    source_dirty: bool
    dependency_lock_hashes: dict[str, str]
    container_image_digests: dict[str, str]
    seed: int = Field(ge=0)
    command_schema_version: str = Field(min_length=1, max_length=128)
    execution_provider_profile: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_hashes(self) -> RunIdentity:
        if tuple(sorted(set(self.parent_artifact_hashes))) != self.parent_artifact_hashes:
            raise ValueError("run parent artifact hashes must be unique and sorted")
        for label, value in (
            ("dataset manifest", self.dataset_manifest_sha256),
            ("split manifest", self.split_manifest_sha256),
            ("model profile", self.model_profile_sha256),
            ("configuration", self.configuration_sha256),
            *[("parent artifact", item) for item in self.parent_artifact_hashes],
            *[
                (f"dependency lock {key}", value)
                for key, value in self.dependency_lock_hashes.items()
            ],
            *[
                (f"container image {key}", value)
                for key, value in self.container_image_digests.items()
            ],
        ):
            _validate_sha256(value, label)
        if len(self.source_git_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_git_commit
        ):
            raise ValueError("source Git commit must be a full lowercase object ID")
        if any(not key or len(key) > 128 for key in self.dependency_lock_hashes):
            raise ValueError("dependency lock names must be bounded and nonempty")
        if any(not key or len(key) > 128 for key in self.container_image_digests):
            raise ValueError("container image names must be bounded and nonempty")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def run_id(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


class EvidenceRegistry:
    """Immutable artifacts plus a rebuildable, single-writer DuckDB index."""

    def __init__(
        self,
        root: Path,
        schema_path: Path,
        *,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.store = ContentAddressedStore(root, schema_path)
        self.fault_injector = fault_injector
        self.index = RegistryIndex(self.store, fault_injector=fault_injector)

    def _inject(self, stage: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(stage)

    def put_bytes(
        self,
        payload: bytes,
        *,
        kind: str,
        media_type: str,
        license_id: str,
        metadata: Mapping[str, Any],
        parents: Sequence[str] = (),
    ) -> ArtifactReceipt:
        receipt = self.store.put_bytes(
            payload,
            kind=kind,
            media_type=media_type,
            license_id=license_id,
            metadata=metadata,
            parents=parents,
        )
        self._inject("after_manifest_publish")
        self.index.index_manifest(receipt.manifest_sha256)
        return receipt

    def put_file(
        self,
        path: Path,
        *,
        kind: str,
        media_type: str,
        license_id: str,
        metadata: Mapping[str, Any],
        parents: Sequence[str] = (),
    ) -> ArtifactReceipt:
        receipt = self.store.put_file(
            path,
            kind=kind,
            media_type=media_type,
            license_id=license_id,
            metadata=metadata,
            parents=parents,
        )
        self._inject("after_manifest_publish")
        self.index.index_manifest(receipt.manifest_sha256)
        return receipt

    def begin_or_resume_run(
        self, identity: RunIdentity, environment_fingerprint: str
    ) -> Mapping[str, str]:
        """Persist immutable run provenance and create or resume only an exact environment."""
        _validate_sha256(environment_fingerprint, "environment compatibility fingerprint")
        run_id = identity.run_id()
        identity_json = identity.canonical_bytes().decode("utf-8")
        existing = self.index.run_row(run_id)
        if existing is not None and (
            existing[0] != identity_json or existing[1] != environment_fingerprint
        ):
            raise EvidenceRegistryError("run resume identity or environment fingerprint differs")
        payload = (
            canonical_json_bytes(
                {
                    "schema_version": "junctionlens.run-configuration.v1",
                    "run_id": run_id,
                    "identity": identity.model_dump(mode="json"),
                    "environment_fingerprint": environment_fingerprint,
                }
            )
            + b"\n"
        )
        receipt = self.put_bytes(
            payload,
            kind="run_configuration",
            media_type="application/json",
            license_id="Apache-2.0",
            metadata={"run_id": run_id, "run_kind": identity.run_kind},
            parents=identity.parent_artifact_hashes,
        )
        state = self.index.begin_or_resume_run(
            run_id,
            identity_json,
            environment_fingerprint,
            receipt.manifest_sha256,
        )
        return {
            "schema_version": "junctionlens.run-resume-receipt.v1",
            "run_id": run_id,
            "run_manifest_sha256": receipt.manifest_sha256,
            "state": state,
        }

    def provenance(self, manifest_sha256: str) -> Mapping[str, object]:
        _validate_sha256(manifest_sha256, "artifact manifest")
        artifacts = self.index.provenance(manifest_sha256)
        return {
            "schema_version": "junctionlens.provenance-view.v1",
            "root_manifest_sha256": manifest_sha256,
            "artifacts": artifacts,
        }

    @staticmethod
    def _validate_alias(alias: str) -> None:
        if not _ALIAS.fullmatch(alias) or any(part in {"", ".", ".."} for part in alias.split("/")):
            raise EvidenceRegistryError("registry alias is invalid")

    def set_alias(self, alias: str, manifest_sha256: str) -> Mapping[str, object]:
        """Set a mutable convenience pointer that is never an evidence identity."""
        self._validate_alias(alias)
        _validate_sha256(manifest_sha256, "alias artifact manifest")
        previous = self.index.set_alias(alias, manifest_sha256)
        return {
            "schema_version": "junctionlens.registry-alias-receipt.v1",
            "state": "CREATED" if previous is None else "UPDATED",
            "alias": alias,
            "manifest_sha256": manifest_sha256,
            "previous_manifest_sha256": previous,
            "evidence_identifier": False,
        }

    def resolve_alias(self, alias: str) -> Mapping[str, object]:
        self._validate_alias(alias)
        return {
            "schema_version": "junctionlens.registry-alias-resolution.v1",
            "alias": alias,
            "manifest_sha256": self.index.resolve_alias(alias),
            "evidence_identifier": False,
        }

    def rebuild_missing_index_rows(self) -> Mapping[str, object]:
        """Recover index rows from manifests without trusting DuckDB as provenance."""
        candidates: set[str] = set()
        for path in self._object_paths():
            object_hash = f"{path.parent.name}{path.name}"
            try:
                self.store.read_manifest(object_hash)
            except (OSError, RuntimeError):
                continue
            candidates.add(object_hash)
        indexed = self.index.indexed_hashes()
        pending = candidates - indexed
        recovered: list[str] = []
        while pending:
            progressed = False
            for manifest_hash in sorted(pending):
                manifest = self.store.read_manifest(manifest_hash)
                parents = set(manifest["parents"])
                if parents - (indexed | set(recovered)):
                    continue
                self.index.index_manifest(manifest_hash)
                recovered.append(manifest_hash)
                pending.remove(manifest_hash)
                progressed = True
                break
            if not progressed:
                raise EvidenceRegistryError(
                    "cannot rebuild index because artifact parents are missing or cyclic"
                )
        return {
            "schema_version": "junctionlens.registry-rebuild.v1",
            "recovered_manifest_sha256": recovered,
            "state": "ACCEPTED",
        }

    def _object_paths(self) -> list[Path]:
        root = self.store.root / "objects" / "sha256"
        if not root.exists():
            return []
        if root.is_symlink() or not root.is_dir():
            raise EvidenceRegistryError("registry object root is invalid")
        result = []
        for directory in sorted(root.iterdir()):
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or not _OBJECT_DIRECTORY.fullmatch(directory.name)
            ):
                raise EvidenceRegistryError("registry object fanout contains an invalid path")
            for path in sorted(directory.iterdir()):
                if path.is_symlink() or not path.is_file() or not _OBJECT_NAME.fullmatch(path.name):
                    raise EvidenceRegistryError("registry object fanout contains an invalid object")
                result.append(path)
        return result

    def garbage_collection_dry_run(self) -> Mapping[str, object]:
        """Report unindexed immutable objects and staging residue without deleting either."""
        reachable = self.index.indexed_hashes()
        objects = {
            f"{path.parent.name}{path.name}": path.stat().st_size for path in self._object_paths()
        }
        orphaned = sorted(set(objects) - reachable)
        staging = []
        staging_bytes = 0
        for path in sorted(self.store.staging_root.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise EvidenceRegistryError("registry staging area contains an invalid path")
            byte_size = path.stat().st_size
            staging_bytes += byte_size
            staging.append({"name": path.name, "byte_size": byte_size})
        return {
            "schema_version": "junctionlens.registry-gc-dry-run.v1",
            "state": "DRY_RUN_COMPLETE",
            "deleted": False,
            "indexed_object_count": len(reachable),
            "object_count": len(objects),
            "orphaned_objects": [
                {"sha256": value, "byte_size": objects[value]} for value in orphaned
            ],
            "staging_files": staging,
            "reclaimable_bytes": sum(objects[value] for value in orphaned) + staging_bytes,
        }


__all__ = ["EvidenceRegistry", "EvidenceRegistryError", "RunIdentity"]

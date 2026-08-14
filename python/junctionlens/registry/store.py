"""Immutable content-addressed artifact objects and validated manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

_HASH_BYTES = 32
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class RegistryError(RuntimeError):
    """Raised when an artifact cannot be stored or verified immutably."""


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Stable identities for one payload and its artifact manifest."""

    kind: str
    payload_sha256: str
    payload_byte_size: int
    manifest_sha256: str


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically without platform-specific whitespace."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _validate_sha256(value: str, label: str) -> None:
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise RegistryError(f"{label} must be lowercase SHA-256") from error
    if len(decoded) != _HASH_BYTES or value != value.lower():
        raise RegistryError(f"{label} must be lowercase SHA-256")


class ContentAddressedStore:
    """Store verified immutable blobs under their SHA-256 identities."""

    def __init__(self, root: Path, schema_path: Path) -> None:
        expanded_root = root.expanduser()
        if expanded_root.is_symlink():
            raise RegistryError("artifact root cannot be a symlink")
        self.root = expanded_root.resolve(strict=False)
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise RegistryError("artifact root must be a real directory")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise RegistryError("artifact root cannot be a symlink")
        try:
            schema_bytes = schema_path.resolve(strict=True).read_bytes()
            schema = json.loads(schema_bytes)
            Draft202012Validator.check_schema(schema)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise RegistryError(f"invalid artifact manifest schema: {error}") from error
        self._validator = Draft202012Validator(schema)
        self.staging_root = self._safe_directory((".staging",), create=True)

    def _safe_directory(self, parts: Sequence[str], *, create: bool) -> Path:
        current = self.root
        missing = False
        for part in parts:
            current = current / part
            if missing:
                if create:
                    current.mkdir(mode=0o755)
                else:
                    continue
            elif current.is_symlink():
                raise RegistryError("artifact store path cannot traverse a symlink")
            elif current.exists():
                if not current.is_dir():
                    raise RegistryError("artifact store directory path is not a directory")
            elif create:
                current.mkdir(mode=0o755)
            else:
                missing = True
        return current

    def object_path(self, sha256: str) -> Path:
        """Resolve one hash to its contained object path."""
        _validate_sha256(sha256, "object hash")
        directory = self._safe_directory(
            ("objects", "sha256", sha256[:2]),
            create=False,
        )
        return directory / sha256[2:]

    def _verify_existing(self, path: Path, expected_hash: str, expected_size: int) -> None:
        if path.is_symlink() or not path.is_file():
            raise RegistryError("content-addressed object target is not a regular file")
        observed_hash, observed_size = _sha256_file(path)
        if observed_hash != expected_hash or observed_size != expected_size:
            raise RegistryError("existing content-addressed object failed integrity verification")

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _install_file(self, source_path: Path, sha256: str, byte_size: int) -> Path:
        target = self.object_path(sha256)
        target_directory = self._safe_directory(
            ("objects", "sha256", sha256[:2]),
            create=True,
        )
        target = target_directory / sha256[2:]
        if target.exists() or target.is_symlink():
            self._verify_existing(target, sha256, byte_size)
            return target
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=".object-",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with source_path.open("rb") as source:
                    shutil.copyfileobj(source, temporary, length=1024 * 1024)
                temporary.flush()
                os.fsync(temporary.fileno())
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        copied_hash, copied_size = _sha256_file(temporary_path)
        if copied_hash != sha256 or copied_size != byte_size:
            temporary_path.unlink(missing_ok=True)
            raise RegistryError("artifact source changed while it was being stored")
        temporary_path.chmod(0o444)
        try:
            os.link(temporary_path, target)
        except FileExistsError:
            self._verify_existing(target, sha256, byte_size)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._sync_directory(target.parent)
        return target

    def _install_bytes(self, payload: bytes) -> tuple[str, int, Path]:
        sha256 = hashlib.sha256(payload).hexdigest()
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.staging_root, delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            target = self._install_file(temporary_path, sha256, len(payload))
        finally:
            temporary_path.unlink(missing_ok=True)
        return sha256, len(payload), target

    def put_file(
        self,
        source_path: Path,
        *,
        kind: str,
        media_type: str,
        license_id: str,
        metadata: Mapping[str, Any],
        parents: Sequence[str] = (),
    ) -> ArtifactReceipt:
        """Store a payload and its deterministic schema-validated manifest."""
        if source_path.is_symlink():
            raise RegistryError("artifact source must be a regular file")
        source = source_path.resolve(strict=True)
        if not source.is_file():
            raise RegistryError("artifact source must be a regular file")
        payload_sha256, byte_size = _sha256_file(source)
        target = self._install_file(source, payload_sha256, byte_size)
        for parent in parents:
            _validate_sha256(parent, "parent hash")
        manifest: Mapping[str, Any] = {
            "schema_version": "junctionlens.artifact-manifest.v1",
            "kind": kind,
            "payload": {
                "sha256": payload_sha256,
                "byte_size": byte_size,
                "media_type": media_type,
                "relative_uri": target.relative_to(self.root).as_posix(),
            },
            "parents": sorted(set(parents)),
            "license_id": license_id,
            "metadata": dict(metadata),
        }
        errors = sorted(self._validator.iter_errors(manifest), key=lambda error: list(error.path))
        if errors:
            raise RegistryError(f"artifact manifest failed schema validation: {errors[0].message}")
        manifest_sha256, _, _ = self._install_bytes(canonical_json_bytes(manifest))
        return ArtifactReceipt(
            kind=kind,
            payload_sha256=payload_sha256,
            payload_byte_size=byte_size,
            manifest_sha256=manifest_sha256,
        )

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
        """Store an in-memory payload through the same immutable file path."""
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.staging_root, delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            return self.put_file(
                temporary_path,
                kind=kind,
                media_type=media_type,
                license_id=license_id,
                metadata=metadata,
                parents=parents,
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    def read_manifest(self, manifest_sha256: str) -> Mapping[str, Any]:
        """Load and revalidate one content-addressed artifact manifest."""
        path = self.object_path(manifest_sha256)
        self._verify_existing(path, manifest_sha256, path.stat().st_size)
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise RegistryError("artifact manifest exceeds the byte limit")
        try:
            value = json.loads(path.read_bytes())
        except json.JSONDecodeError as error:
            raise RegistryError("artifact manifest contains invalid JSON") from error
        if not isinstance(value, dict):
            raise RegistryError("artifact manifest must be an object")
        manifest = cast(Mapping[str, Any], value)
        errors = sorted(self._validator.iter_errors(manifest), key=lambda error: list(error.path))
        if errors:
            raise RegistryError(f"artifact manifest failed schema validation: {errors[0].message}")
        payload = cast(Mapping[str, Any], manifest["payload"])
        payload_hash = cast(str, payload["sha256"])
        payload_size = cast(int, payload["byte_size"])
        self._verify_existing(self.object_path(payload_hash), payload_hash, payload_size)
        return manifest


__all__ = [
    "ArtifactReceipt",
    "ContentAddressedStore",
    "RegistryError",
    "canonical_json_bytes",
]

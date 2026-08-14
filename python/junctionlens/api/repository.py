"""Integrity-checked read queries over the immutable evidence registry."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import stat
from collections.abc import Iterator, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO, cast

import duckdb
from pydantic import JsonValue

from junctionlens.api.models import (
    ArtifactDetail,
    ArtifactSummary,
    PageInfo,
    RunSummary,
    SceneBundle,
    ServiceConfig,
)
from junctionlens.registry.store import ContentAddressedStore, RegistryError, canonical_json_bytes

_SHA256_LENGTH = 64
_DECISION_MEDIA_TYPE = "application/vnd.junctionlens.gate-decision+json"
_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_SCENE_MEDIA_TYPE = "application/vnd.junctionlens.scene-bundle+json"


class EvidenceReadError(RuntimeError):
    """Raised when registered evidence cannot be served safely."""


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise EvidenceReadError(f"{label} contains duplicate object key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                EvidenceReadError(f"{label} contains nonfinite constant {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise EvidenceReadError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise EvidenceReadError(f"{label} must be a JSON object")
    return value


def _validate_sha256(value: str, label: str) -> None:
    if (
        len(value) != _SHA256_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceReadError(f"{label} must be a lowercase SHA-256")


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceReadError("metric table contains a nonfinite value")
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dt.date | dt.datetime | dt.time):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise EvidenceReadError(f"metric table contains unsupported value type {type(value).__name__}")


class VerifiedPayload:
    """An already hashed file descriptor retained until response consumption."""

    def __init__(self, path: Path, expected_sha256: str, expected_size: int, limit: int) -> None:
        if expected_size > limit:
            raise EvidenceReadError("artifact payload exceeds the response byte limit")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise EvidenceReadError("artifact payload cannot be opened safely") from error
        self._source: BinaryIO | None = os.fdopen(descriptor, "rb", closefd=True)
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode) or details.st_size != expected_size:
                raise EvidenceReadError("artifact payload is not the registered regular file")
            digest = hashlib.sha256()
            source = self._source
            if source is None:
                raise EvidenceReadError("artifact payload was closed during verification")
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise EvidenceReadError("artifact payload failed its registered SHA-256")
            self._source.seek(0)
        except Exception:
            self.close()
            raise

    def read(self) -> bytes:
        if self._source is None:
            raise EvidenceReadError("artifact payload is closed")
        try:
            return self._source.read()
        finally:
            self.close()

    def chunks(self) -> Iterator[bytes]:
        if self._source is None:
            raise EvidenceReadError("artifact payload is closed")
        try:
            while chunk := self._source.read(1024 * 1024):
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None


class EvidenceRepository:
    """Read-only, bounded access to one canonical artifact registry root."""

    def __init__(self, config: ServiceConfig) -> None:
        root = config.artifact_root.expanduser()
        if root.is_symlink():
            raise EvidenceReadError("artifact root cannot be a symlink")
        try:
            self.root = root.resolve(strict=True)
        except OSError as error:
            raise EvidenceReadError("artifact root does not exist") from error
        if not self.root.is_dir() or self.root.is_symlink():
            raise EvidenceReadError("artifact root must be a real directory")
        schema = config.schema_path.expanduser()
        if schema.is_symlink():
            raise EvidenceReadError("artifact schema cannot be a symlink")
        try:
            schema = schema.resolve(strict=True)
        except OSError as error:
            raise EvidenceReadError("artifact schema does not exist") from error
        if not schema.is_file():
            raise EvidenceReadError("artifact schema must be a regular file")
        database = self.root / ".index" / "registry.duckdb"
        if database.is_symlink() or not database.is_file():
            raise EvidenceReadError("artifact registry index is missing or unsafe")
        self.database_path = database
        try:
            self.store = ContentAddressedStore(self.root, schema)
        except (OSError, RegistryError, ValueError) as error:
            raise EvidenceReadError(f"artifact registry is invalid: {error}") from error
        self.config = config

    def _connect(self) -> duckdb.DuckDBPyConnection:
        if self.database_path.is_symlink() or not self.database_path.is_file():
            raise EvidenceReadError("artifact registry index is missing or unsafe")
        return duckdb.connect(str(self.database_path), read_only=True)

    def counts(self) -> tuple[int, int]:
        connection = self._connect()
        try:
            artifact_row = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()
            run_row = connection.execute("SELECT COUNT(*) FROM runs").fetchone()
            if artifact_row is None or run_row is None:
                raise EvidenceReadError("artifact registry count query returned no row")
            artifact_count = int(artifact_row[0])
            run_count = int(run_row[0])
        except duckdb.Error as error:
            raise EvidenceReadError("artifact registry index cannot be read") from error
        finally:
            connection.close()
        return artifact_count, run_count

    def list_runs(self, offset: int, limit: int) -> tuple[tuple[RunSummary, ...], PageInfo]:
        connection = self._connect()
        try:
            count_row = connection.execute("SELECT COUNT(*) FROM runs").fetchone()
            if count_row is None:
                raise EvidenceReadError("run count query returned no row")
            total = int(count_row[0])
            rows = connection.execute(
                """
                SELECT run_id, identity_json, environment_fingerprint,
                       run_manifest_sha256, state
                FROM runs ORDER BY run_id LIMIT ? OFFSET ?
                """,
                [limit, offset],
            ).fetchall()
        except duckdb.Error as error:
            raise EvidenceReadError("run index cannot be read") from error
        finally:
            connection.close()
        items = []
        for run_id, identity_json, environment, run_manifest, state in rows:
            identity = _strict_json_object(str(identity_json).encode(), "run identity")
            items.append(
                RunSummary(
                    run_id=str(run_id),
                    run_kind=str(identity.get("run_kind", "")),
                    state=str(state),
                    environment_fingerprint=str(environment),
                    run_manifest_sha256=str(run_manifest),
                    execution_provider_profile=str(identity.get("execution_provider_profile", "")),
                    source_git_commit=str(identity.get("source_git_commit", "")),
                    source_dirty=bool(identity.get("source_dirty", False)),
                )
            )
        page = PageInfo(offset=offset, limit=limit, returned=len(items), total=total)
        return tuple(items), page

    def list_artifacts(
        self, offset: int, limit: int, *, kind: str | None = None
    ) -> tuple[tuple[ArtifactSummary, ...], PageInfo]:
        connection = self._connect()
        try:
            if kind is None:
                count_row = connection.execute("SELECT COUNT(*) FROM artifact_summary").fetchone()
                if count_row is None:
                    raise EvidenceReadError("artifact count query returned no row")
                total = int(count_row[0])
                rows = connection.execute(
                    """
                    SELECT manifest_sha256, kind, payload_sha256, payload_byte_size,
                           media_type, license_id, metadata_json
                    FROM artifact_summary
                    ORDER BY manifest_sha256 LIMIT ? OFFSET ?
                    """,
                    [limit, offset],
                ).fetchall()
            else:
                count_row = connection.execute(
                    "SELECT COUNT(*) FROM artifact_summary WHERE kind = ?", [kind]
                ).fetchone()
                if count_row is None:
                    raise EvidenceReadError("artifact count query returned no row")
                total = int(count_row[0])
                rows = connection.execute(
                    """
                    SELECT manifest_sha256, kind, payload_sha256, payload_byte_size,
                           media_type, license_id, metadata_json
                    FROM artifact_summary WHERE kind = ?
                    ORDER BY manifest_sha256 LIMIT ? OFFSET ?
                    """,
                    [kind, limit, offset],
                ).fetchall()
        except duckdb.Error as error:
            raise EvidenceReadError("artifact index cannot be read") from error
        finally:
            connection.close()
        items = tuple(self._artifact_summary(row) for row in rows)
        return items, PageInfo(offset=offset, limit=limit, returned=len(items), total=total)

    def list_artifacts_by_kind_and_media(
        self,
        offset: int,
        limit: int,
        *,
        kind: str,
        media_type: str,
    ) -> tuple[tuple[ArtifactSummary, ...], PageInfo]:
        connection = self._connect()
        try:
            parameters = [kind, media_type]
            count_row = connection.execute(
                """
                SELECT COUNT(*) FROM artifact_summary
                WHERE kind = ? AND media_type = ?
                """,
                parameters,
            ).fetchone()
            if count_row is None:
                raise EvidenceReadError("artifact count query returned no row")
            total = int(count_row[0])
            rows = connection.execute(
                """
                SELECT manifest_sha256, kind, payload_sha256, payload_byte_size,
                       media_type, license_id, metadata_json
                FROM artifact_summary WHERE kind = ? AND media_type = ?
                ORDER BY manifest_sha256 LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchall()
        except duckdb.Error as error:
            raise EvidenceReadError("artifact index cannot be read") from error
        finally:
            connection.close()
        items = tuple(self._artifact_summary(row) for row in rows)
        return items, PageInfo(offset=offset, limit=limit, returned=len(items), total=total)

    @staticmethod
    def _artifact_summary(row: tuple[object, ...]) -> ArtifactSummary:
        metadata = _strict_json_object(str(row[6]).encode(), "artifact metadata")
        return ArtifactSummary(
            manifest_sha256=str(row[0]),
            kind=str(row[1]),
            payload_sha256=str(row[2]),
            payload_byte_size=int(cast(int, row[3])),
            media_type=str(row[4]),
            license_id=str(row[5]),
            metadata=cast(dict[str, JsonValue], metadata),
        )

    def artifact(self, manifest_sha256: str) -> ArtifactDetail:
        _validate_sha256(manifest_sha256, "artifact manifest")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT manifest_sha256, kind, payload_sha256, payload_byte_size,
                       media_type, license_id, metadata_json
                FROM artifact_summary WHERE manifest_sha256 = ?
                """,
                [manifest_sha256],
            ).fetchone()
        except duckdb.Error as error:
            raise EvidenceReadError("artifact index cannot be read") from error
        finally:
            connection.close()
        if row is None:
            raise KeyError(manifest_sha256)
        summary = self._artifact_summary(tuple(row))
        try:
            manifest = self.store.read_manifest(manifest_sha256)
        except (OSError, RegistryError) as error:
            raise EvidenceReadError(f"artifact failed integrity verification: {error}") from error
        payload = cast(Mapping[str, Any], manifest["payload"])
        if payload["sha256"] != summary.payload_sha256:
            raise EvidenceReadError("artifact index differs from its immutable manifest")
        return ArtifactDetail(
            schema_version="junctionlens.api-artifact.v1",
            **summary.model_dump(),
            parents=tuple(str(value) for value in cast(list[str], manifest["parents"])),
            relative_uri=str(payload["relative_uri"]),
        )

    def open_payload(self, manifest_sha256: str, *, limit: int | None = None) -> VerifiedPayload:
        artifact = self.artifact(manifest_sha256)
        payload_limit = self.config.max_artifact_bytes if limit is None else limit
        path = self.store.object_path(artifact.payload_sha256)
        return VerifiedPayload(
            path,
            artifact.payload_sha256,
            artifact.payload_byte_size,
            payload_limit,
        )

    def decision(self, manifest_sha256: str) -> dict[str, JsonValue]:
        artifact = self.artifact(manifest_sha256)
        if artifact.kind != "release_decision" or artifact.media_type != _DECISION_MEDIA_TYPE:
            raise EvidenceReadError("artifact is not a persisted release decision")
        payload = self.open_payload(manifest_sha256, limit=16 * 1024 * 1024).read()
        decision = _strict_json_object(payload, "release decision")
        if decision.get("schema_version") != "junctionlens.gate-decision.v1":
            raise EvidenceReadError("release decision schema version is unsupported")
        observed_hash = decision.get("decision_sha256")
        if not isinstance(observed_hash, str):
            raise EvidenceReadError("release decision identity is missing")
        body = dict(decision)
        del body["decision_sha256"]
        if hashlib.sha256(canonical_json_bytes(body)).hexdigest() != observed_hash:
            raise EvidenceReadError("release decision identity does not match its content")
        return cast(dict[str, JsonValue], decision)

    def metric_rows(
        self, manifest_sha256: str, offset: int, limit: int
    ) -> tuple[tuple[str, ...], tuple[dict[str, JsonValue], ...], PageInfo]:
        artifact = self.artifact(manifest_sha256)
        if artifact.media_type != _PARQUET_MEDIA_TYPE or artifact.kind not in {
            "comparison",
            "frame_kpi_table",
            "segment_kpi_table",
            "slice_table",
            "benchmark",
        }:
            raise EvidenceReadError("artifact is not a registered metric table")
        verified = self.open_payload(manifest_sha256, limit=self.config.max_metric_table_bytes)
        verified.close()
        path = self.store.object_path(artifact.payload_sha256)
        connection = self._connect()
        try:
            count_row = connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [str(path)]
            ).fetchone()
            if count_row is None:
                raise EvidenceReadError("metric table count query returned no row")
            total = int(count_row[0])
            cursor = connection.execute(
                "SELECT * FROM read_parquet(?) LIMIT ? OFFSET ?",
                [str(path), limit, offset],
            )
            if cursor.description is None:
                raise EvidenceReadError("metric table query returned no schema")
            columns = tuple(str(item[0]) for item in cursor.description)
            raw_rows = cursor.fetchall()
        except duckdb.Error as error:
            raise EvidenceReadError("registered metric table cannot be read") from error
        finally:
            connection.close()
        second_verification = self.open_payload(
            manifest_sha256, limit=self.config.max_metric_table_bytes
        )
        second_verification.close()
        rows = tuple(
            {column: _json_value(value) for column, value in zip(columns, row, strict=True)}
            for row in raw_rows
        )
        return (
            columns,
            rows,
            PageInfo(
                offset=offset,
                limit=limit,
                returned=len(rows),
                total=total,
            ),
        )

    def scene_bundle(self, manifest_sha256: str) -> tuple[SceneBundle, dict[str, JsonValue]]:
        artifact = self.artifact(manifest_sha256)
        if artifact.kind != "counterexample_bundle" or artifact.media_type != _SCENE_MEDIA_TYPE:
            raise EvidenceReadError("artifact is not a registered scene bundle")
        payload = self.open_payload(manifest_sha256, limit=64 * 1024 * 1024).read()
        value = _strict_json_object(payload, "scene bundle")
        try:
            bundle = SceneBundle.model_validate(value)
        except ValueError as error:
            raise EvidenceReadError(f"scene bundle schema is invalid: {error}") from error
        if bundle.decision_manifest_sha256 not in artifact.parents:
            raise EvidenceReadError("scene bundle decision is not an immutable parent")
        decision = self.decision(bundle.decision_manifest_sha256)
        for frame in bundle.frames:
            for camera in frame.cameras:
                if camera.artifact_manifest_sha256 is None:
                    continue
                if camera.artifact_manifest_sha256 not in artifact.parents:
                    raise EvidenceReadError("scene camera artifact is not an immutable parent")
                image_artifact = self.artifact(camera.artifact_manifest_sha256)
                if image_artifact.media_type not in _IMAGE_MEDIA_TYPES:
                    raise EvidenceReadError("scene camera references an unsupported image artifact")
        return bundle, decision

    @staticmethod
    def image_media_types() -> frozenset[str]:
        return _IMAGE_MEDIA_TYPES


__all__ = ["EvidenceReadError", "EvidenceRepository", "VerifiedPayload"]

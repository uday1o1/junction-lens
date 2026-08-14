"""DuckDB metadata index whose rows are rebuildable from immutable manifests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any, cast

import duckdb

from junctionlens.registry.lock import RegistryWriterLock
from junctionlens.registry.store import ContentAddressedStore, RegistryError, canonical_json_bytes


class RegistryIndexError(RuntimeError):
    """Raised when the rebuildable DuckDB registry index is inconsistent."""


FaultInjector = Callable[[str], None]


class RegistryIndex:
    """Single-writer metadata index with deterministic provenance queries."""

    def __init__(
        self,
        store: ContentAddressedStore,
        *,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.store = store
        self.root = store.root / ".index"
        if self.root.is_symlink():
            raise RegistryIndexError("registry index root cannot be a symlink")
        self.root.mkdir(mode=0o755, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise RegistryIndexError("registry index root must be a real directory")
        self.database_path = self.root / "registry.duckdb"
        if self.database_path.is_symlink():
            raise RegistryIndexError("registry database cannot be a symlink")
        self.lock_root = self.root / "locks"
        self.fault_injector = fault_injector
        self.initialize()

    def _inject(self, stage: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(stage)

    def writer_lock(self) -> RegistryWriterLock:
        return RegistryWriterLock(self.lock_root)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        if self.database_path.is_symlink():
            raise RegistryIndexError("registry database cannot be a symlink")
        return duckdb.connect(str(self.database_path))

    def initialize(self) -> None:
        """Create the deterministic index schema under exclusive ownership."""
        with self.writer_lock():
            connection = self._connect()
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS artifacts (
                        manifest_sha256 VARCHAR PRIMARY KEY,
                        kind VARCHAR NOT NULL,
                        payload_sha256 VARCHAR NOT NULL,
                        payload_byte_size UBIGINT NOT NULL,
                        media_type VARCHAR NOT NULL,
                        license_id VARCHAR NOT NULL,
                        metadata_json VARCHAR NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS artifact_parents (
                        child_manifest_sha256 VARCHAR NOT NULL,
                        parent_manifest_sha256 VARCHAR NOT NULL,
                        PRIMARY KEY (child_manifest_sha256, parent_manifest_sha256)
                    );
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id VARCHAR PRIMARY KEY,
                        identity_json VARCHAR NOT NULL,
                        environment_fingerprint VARCHAR NOT NULL,
                        run_manifest_sha256 VARCHAR NOT NULL,
                        state VARCHAR NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS aliases (
                        alias VARCHAR PRIMARY KEY,
                        manifest_sha256 VARCHAR NOT NULL
                    );
                    CREATE OR REPLACE VIEW artifact_summary AS
                        SELECT manifest_sha256, kind, payload_sha256,
                               payload_byte_size, media_type, license_id, metadata_json
                        FROM artifacts;
                    CREATE OR REPLACE VIEW provenance_edges AS
                        SELECT child_manifest_sha256, parent_manifest_sha256
                        FROM artifact_parents;
                    """
                )
            finally:
                connection.close()

    @staticmethod
    def _record(manifest_sha256: str, manifest: Mapping[str, Any]) -> tuple[object, ...]:
        payload = cast(Mapping[str, Any], manifest["payload"])
        return (
            manifest_sha256,
            str(manifest["kind"]),
            str(payload["sha256"]),
            int(payload["byte_size"]),
            str(payload["media_type"]),
            str(manifest["license_id"]),
            canonical_json_bytes(manifest["metadata"]).decode("utf-8"),
        )

    def index_manifest(self, manifest_sha256: str) -> None:
        """Index one verified manifest transactionally without becoming provenance authority."""
        try:
            manifest = self.store.read_manifest(manifest_sha256)
        except (OSError, RegistryError) as error:
            raise RegistryIndexError(f"cannot verify artifact manifest: {error}") from error
        parents = cast(Sequence[str], manifest["parents"])
        record = self._record(manifest_sha256, manifest)
        with self.writer_lock() as lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN TRANSACTION")
                existing = connection.execute(
                    """
                    SELECT manifest_sha256, kind, payload_sha256, payload_byte_size,
                           media_type, license_id, metadata_json
                    FROM artifacts WHERE manifest_sha256 = ?
                    """,
                    [manifest_sha256],
                ).fetchone()
                if existing is not None and tuple(existing) != record:
                    raise RegistryIndexError("indexed artifact differs from its immutable manifest")
                if existing is None:
                    for parent in parents:
                        parent_row = connection.execute(
                            "SELECT 1 FROM artifacts WHERE manifest_sha256 = ?", [parent]
                        ).fetchone()
                        if parent_row is None:
                            raise RegistryIndexError(
                                "artifact parent must be verified and indexed before its child"
                            )
                    connection.execute(
                        "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)", list(record)
                    )
                    for parent in sorted(parents):
                        connection.execute(
                            "INSERT INTO artifact_parents VALUES (?, ?)",
                            [manifest_sha256, parent],
                        )
                lock.heartbeat()
                self._inject("before_index_commit")
                connection.execute("COMMIT")
            except Exception:
                with suppress(duckdb.Error):
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def read_artifact(self, manifest_sha256: str) -> Mapping[str, Any]:
        """Read through DuckDB and cross-check the immutable manifest on disk."""
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT kind, payload_sha256, payload_byte_size, media_type,
                       license_id, metadata_json
                FROM artifact_summary WHERE manifest_sha256 = ?
                """,
                [manifest_sha256],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise RegistryIndexError("artifact manifest is not indexed")
        manifest = self.store.read_manifest(manifest_sha256)
        if tuple(row) != self._record(manifest_sha256, manifest)[1:]:
            raise RegistryIndexError("registry index row differs from immutable provenance")
        return manifest

    def provenance(self, manifest_sha256: str) -> list[Mapping[str, Any]]:
        """Return deterministic transitive provenance from the DuckDB edge view."""
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                WITH RECURSIVE ancestors(manifest_sha256, depth) AS (
                    SELECT CAST(? AS VARCHAR), 0
                    UNION
                    SELECT edge.parent_manifest_sha256, ancestors.depth + 1
                    FROM ancestors
                    JOIN provenance_edges AS edge
                      ON edge.child_manifest_sha256 = ancestors.manifest_sha256
                )
                SELECT artifact.manifest_sha256, MIN(ancestors.depth) AS depth,
                       artifact.kind, artifact.payload_sha256,
                       artifact.payload_byte_size, artifact.media_type,
                       artifact.license_id, artifact.metadata_json
                FROM ancestors
                JOIN artifact_summary AS artifact USING (manifest_sha256)
                GROUP BY ALL
                ORDER BY depth, artifact.manifest_sha256
                """,
                [manifest_sha256],
            ).fetchall()
            if len(rows) > 100_000:
                raise RegistryIndexError("artifact provenance exceeds the object limit")
            hashes = [str(row[0]) for row in rows]
            edge_rows = (
                []
                if not hashes
                else connection.execute(
                    """
                    SELECT child_manifest_sha256, parent_manifest_sha256
                    FROM provenance_edges
                    WHERE child_manifest_sha256 IN (SELECT * FROM UNNEST(?))
                    """,
                    [hashes],
                ).fetchall()
            )
        finally:
            connection.close()
        if not rows or rows[0][0] != manifest_sha256:
            raise RegistryIndexError("artifact manifest is not indexed")
        indexed_parents: dict[str, list[str]] = {str(row[0]): [] for row in rows}
        for child, parent in edge_rows:
            indexed_parents[str(child)].append(str(parent))
        result: list[Mapping[str, Any]] = []
        for row in rows:
            artifact_hash = str(row[0])
            manifest = self.store.read_manifest(artifact_hash)
            indexed_record = (
                artifact_hash,
                str(row[2]),
                str(row[3]),
                int(row[4]),
                str(row[5]),
                str(row[6]),
                str(row[7]),
            )
            if indexed_record != self._record(artifact_hash, manifest):
                raise RegistryIndexError("provenance row differs from immutable manifest")
            parents = sorted(cast(Sequence[str], manifest["parents"]))
            if sorted(indexed_parents[artifact_hash]) != parents:
                raise RegistryIndexError("provenance edges differ from immutable manifest")
            result.append(
                {
                    "manifest_sha256": artifact_hash,
                    "depth": int(row[1]),
                    "kind": str(row[2]),
                    "payload_sha256": str(row[3]),
                    "payload_byte_size": int(row[4]),
                    "media_type": str(row[5]),
                    "license_id": str(row[6]),
                    "metadata": json.loads(str(row[7])),
                    "parents": parents,
                }
            )
        return result

    def indexed_hashes(self) -> set[str]:
        """Return every object hash reachable from indexed immutable manifests."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT manifest_sha256, payload_sha256 FROM artifact_summary"
            ).fetchall()
        finally:
            connection.close()
        return {str(value) for row in rows for value in row}

    def run_row(self, run_id: str) -> tuple[str, str, str, str] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT identity_json, environment_fingerprint, run_manifest_sha256, state
                FROM runs WHERE run_id = ?
                """,
                [run_id],
            ).fetchone()
        finally:
            connection.close()
        return None if row is None else cast(tuple[str, str, str, str], tuple(row))

    def begin_or_resume_run(
        self,
        run_id: str,
        identity_json: str,
        environment_fingerprint: str,
        run_manifest_sha256: str,
    ) -> str:
        """Create or validate one operational run row under the single writer lock."""
        with self.writer_lock():
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT identity_json, environment_fingerprint,
                           run_manifest_sha256, state
                    FROM runs WHERE run_id = ?
                    """,
                    [run_id],
                ).fetchone()
                expected = (identity_json, environment_fingerprint, run_manifest_sha256)
                if row is None:
                    connection.execute(
                        "INSERT INTO runs VALUES (?, ?, ?, ?, 'RUNNING')",
                        [run_id, *expected],
                    )
                    return "CREATED"
                if tuple(row[:3]) != expected:
                    raise RegistryIndexError(
                        "run resume identity or environment fingerprint differs"
                    )
                return "ALREADY_COMPLETE" if row[3] == "COMPLETE" else "RESUMED"
            finally:
                connection.close()

    def set_run_state(self, run_id: str, state: str) -> None:
        if state not in {"COMPLETE", "FAILED", "INTERRUPTED"}:
            raise RegistryIndexError("run terminal state is invalid")
        with self.writer_lock():
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT state FROM runs WHERE run_id = ?", [run_id]
                ).fetchone()
                if row is None:
                    raise RegistryIndexError("run identity is not indexed")
                if row[0] == "COMPLETE" and state != "COMPLETE":
                    raise RegistryIndexError("completed registry run state is immutable")
                connection.execute("UPDATE runs SET state = ? WHERE run_id = ?", [state, run_id])
            finally:
                connection.close()

    def set_alias(self, alias: str, manifest_sha256: str) -> str | None:
        """Move one explicitly non-evidence alias under the single writer lock."""
        self.read_artifact(manifest_sha256)
        with self.writer_lock():
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT manifest_sha256 FROM aliases WHERE alias = ?", [alias]
                ).fetchone()
                connection.execute(
                    "INSERT OR REPLACE INTO aliases VALUES (?, ?)",
                    [alias, manifest_sha256],
                )
                return None if row is None else str(row[0])
            finally:
                connection.close()

    def resolve_alias(self, alias: str) -> str:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT manifest_sha256 FROM aliases WHERE alias = ?", [alias]
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise RegistryIndexError("registry alias does not exist")
        manifest_sha256 = str(row[0])
        self.read_artifact(manifest_sha256)
        return manifest_sha256


__all__ = ["FaultInjector", "RegistryIndex", "RegistryIndexError"]

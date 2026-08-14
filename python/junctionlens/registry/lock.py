"""Fail-closed advisory writer lock with validated stale-owner recovery."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import shutil
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

from junctionlens.registry.store import canonical_json_bytes


class RegistryLockError(RuntimeError):
    """Raised when exclusive registry writer ownership cannot be proven."""


@dataclass(frozen=True, slots=True)
class LockOwner:
    schema_version: str
    lock_id: str
    pid: int
    host_fingerprint: str
    created_ns: int
    heartbeat_ns: int


def host_fingerprint() -> str:
    """Return a stable redacted identity for stale-lock host validation."""
    identity = f"{socket.gethostname()}|{platform.system()}|{platform.machine()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno == errno.EPERM:
            return True
        raise RegistryLockError(f"cannot inspect lock owner process: {error}") from error
    return True


class RegistryWriterLock:
    """One atomic directory lock for every DuckDB writer transaction."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "registry-writer.lock"
        self.events = root / "recovery-events.jsonl"
        self.owner: LockOwner | None = None

    @property
    def owner_path(self) -> Path:
        return self.path / "owner.json"

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_owner(path: Path) -> LockOwner:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            raise RegistryLockError("registry writer lock owner record is invalid")

        def reject_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in items:
                if key in result:
                    raise RegistryLockError(f"duplicate registry lock owner key: {key}")
                result[key] = item
            return result

        try:
            value = json.loads(
                path.read_bytes(),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    RegistryLockError(f"nonfinite registry lock owner value: {item}")
                ),
            )
        except json.JSONDecodeError as error:
            raise RegistryLockError("registry writer lock owner record is invalid JSON") from error
        if not isinstance(value, dict) or set(value) != {
            "created_ns",
            "heartbeat_ns",
            "host_fingerprint",
            "lock_id",
            "pid",
            "schema_version",
        }:
            raise RegistryLockError("registry writer lock owner schema is invalid")
        try:
            owner = LockOwner(**value)
        except TypeError as error:
            raise RegistryLockError("registry writer lock owner types are invalid") from error
        if (
            not isinstance(owner.schema_version, str)
            or not isinstance(owner.lock_id, str)
            or isinstance(owner.pid, bool)
            or not isinstance(owner.pid, int)
            or not isinstance(owner.host_fingerprint, str)
            or isinstance(owner.created_ns, bool)
            or not isinstance(owner.created_ns, int)
            or isinstance(owner.heartbeat_ns, bool)
            or not isinstance(owner.heartbeat_ns, int)
            or owner.schema_version != "junctionlens.registry-writer-lock.v1"
            or len(owner.lock_id) != 32
            or any(character not in "0123456789abcdef" for character in owner.lock_id)
            or owner.pid <= 0
            or len(owner.host_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in owner.host_fingerprint)
            or owner.created_ns <= 0
            or owner.heartbeat_ns < owner.created_ns
        ):
            raise RegistryLockError("registry writer lock owner values are invalid")
        return owner

    def _write_owner(self, owner: LockOwner) -> None:
        temporary = self.path / f".owner-{owner.lock_id}.tmp"
        with temporary.open("xb") as destination:
            destination.write(canonical_json_bytes(asdict(owner)) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(self.owner_path)
        self._sync_directory(self.path)

    def _record_recovery(self, owner: LockOwner) -> None:
        event = {
            "schema_version": "junctionlens.registry-lock-recovery.v1",
            "event": "STALE_LOCK_RECLAIMED",
            "reason_code": "OWNER_PROCESS_ABSENT",
            "lock_id": owner.lock_id,
            "owner_pid": owner.pid,
            "host_fingerprint": owner.host_fingerprint,
        }
        descriptor = os.open(self.events, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, canonical_json_bytes(event) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._sync_directory(self.root)

    def _reclaim_if_safe(self) -> bool:
        if self.path.is_symlink() or not self.path.is_dir():
            raise RegistryLockError("registry writer lock path is not a real directory")
        owner = self._read_owner(self.owner_path)
        if owner.host_fingerprint != host_fingerprint():
            raise RegistryLockError("registry writer lock belongs to an unverifiable host")
        if _process_alive(owner.pid):
            raise RegistryLockError("registry writer lock owner process is still alive")
        tombstone = self.root / f".reclaimed-{owner.lock_id}"
        try:
            self.path.replace(tombstone)
        except FileNotFoundError:
            return False
        self._record_recovery(owner)
        shutil.rmtree(tombstone)
        self._sync_directory(self.root)
        return True

    def acquire(self) -> RegistryWriterLock:
        """Acquire the lock or reclaim only a proven dead same-host owner."""
        if self.root.is_symlink():
            raise RegistryLockError("registry lock root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise RegistryLockError("registry lock root must be a real directory")
        while True:
            try:
                self.path.mkdir(mode=0o700)
            except FileExistsError:
                if self._reclaim_if_safe():
                    continue
                continue
            now = time.time_ns()
            owner = LockOwner(
                schema_version="junctionlens.registry-writer-lock.v1",
                lock_id=uuid.uuid4().hex,
                pid=os.getpid(),
                host_fingerprint=host_fingerprint(),
                created_ns=now,
                heartbeat_ns=now,
            )
            try:
                self._write_owner(owner)
            except Exception:
                shutil.rmtree(self.path)
                raise
            self.owner = owner
            self._sync_directory(self.root)
            return self

    def heartbeat(self) -> None:
        """Refresh the durable heartbeat after verifying current ownership."""
        if self.owner is None:
            raise RegistryLockError("registry writer lock is not held")
        observed = self._read_owner(self.owner_path)
        if observed.lock_id != self.owner.lock_id or observed.pid != self.owner.pid:
            raise RegistryLockError("registry writer lock ownership changed unexpectedly")
        refreshed = LockOwner(
            schema_version=self.owner.schema_version,
            lock_id=self.owner.lock_id,
            pid=self.owner.pid,
            host_fingerprint=self.owner.host_fingerprint,
            created_ns=self.owner.created_ns,
            heartbeat_ns=time.time_ns(),
        )
        self._write_owner(refreshed)
        self.owner = refreshed

    def release(self) -> None:
        """Release by atomically removing the owned lock directory from its live name."""
        if self.owner is None:
            return
        observed = self._read_owner(self.owner_path)
        if observed.lock_id != self.owner.lock_id or observed.pid != self.owner.pid:
            raise RegistryLockError("refusing to release a registry lock owned by another process")
        tombstone = self.root / f".released-{self.owner.lock_id}"
        self.path.replace(tombstone)
        shutil.rmtree(tombstone)
        self._sync_directory(self.root)
        self.owner = None

    def __enter__(self) -> RegistryWriterLock:
        return self.acquire()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


__all__ = [
    "LockOwner",
    "RegistryLockError",
    "RegistryWriterLock",
    "host_fingerprint",
]

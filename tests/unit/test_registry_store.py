"""Content-addressed registry atomicity and reproducibility tests."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from junctionlens.registry import ContentAddressedStore, RegistryError
from junctionlens.registry.lock import (
    LockOwner,
    RegistryLockError,
    RegistryWriterLock,
    host_fingerprint,
)
from junctionlens.registry.service import EvidenceRegistry, RunIdentity
from junctionlens.registry.store import canonical_json_bytes

_SCHEMA = Path("schemas/artifact-manifest-v1.schema.json")


def test_registry_reuses_identical_objects_and_manifests(tmp_path: Path) -> None:
    """Identical payload and provenance produce identical immutable identities."""
    store = ContentAddressedStore(tmp_path / "artifacts", _SCHEMA)
    arguments = {
        "kind": "split_manifest",
        "media_type": "application/json",
        "license_id": "CC-BY-NC-SA-4.0",
        "metadata": {"policy_id": "test"},
    }
    first = store.put_bytes(b'{"value":1}\n', **arguments)
    second = store.put_bytes(b'{"value":1}\n', **arguments)
    assert first == second
    manifest = store.read_manifest(first.manifest_sha256)
    assert manifest["payload"]["sha256"] == first.payload_sha256
    assert store.object_path(first.payload_sha256).read_bytes() == b'{"value":1}\n'


def test_registry_detects_corrupted_existing_object(tmp_path: Path) -> None:
    """An existing hash path is verified and never overwritten to hide corruption."""
    store = ContentAddressedStore(tmp_path / "artifacts", _SCHEMA)
    arguments = {
        "kind": "frame_manifest",
        "media_type": "application/x-ndjson",
        "license_id": "CC-BY-NC-SA-4.0",
        "metadata": {"frame_count": 1},
    }
    receipt = store.put_bytes(b"one\n", **arguments)
    target = store.object_path(receipt.payload_sha256)
    target.chmod(0o644)
    target.write_bytes(b"corrupt\n")
    with pytest.raises(RegistryError, match="integrity verification"):
        store.put_bytes(b"one\n", **arguments)


def test_registry_rejects_symlink_root(tmp_path: Path) -> None:
    """A local artifact root cannot redirect writes through a symlink."""
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "artifacts"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RegistryError, match="cannot be a symlink"):
        ContentAddressedStore(link, _SCHEMA)


def test_registry_rejects_nested_object_symlink(tmp_path: Path) -> None:
    """A nested object directory cannot redirect immutable writes outside the root."""
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "objects").symlink_to(outside, target_is_directory=True)
    store = ContentAddressedStore(root, _SCHEMA)
    with pytest.raises(RegistryError, match="cannot traverse a symlink"):
        store.put_bytes(
            b"bounded\n",
            kind="frame_manifest",
            media_type="application/x-ndjson",
            license_id="CC-BY-NC-SA-4.0",
            metadata={},
        )
    assert list(outside.iterdir()) == []


def _evidence_registry(tmp_path: Path, **kwargs: object) -> EvidenceRegistry:
    return EvidenceRegistry(tmp_path / "artifacts", _SCHEMA, **kwargs)


def _run_identity(parents: tuple[str, ...] = ()) -> RunIdentity:
    return RunIdentity(
        schema_version="junctionlens.run-identity.v1",
        run_kind="synthetic-evaluation",
        parent_artifact_hashes=parents,
        dataset_manifest_sha256="1" * 64,
        split_manifest_sha256="2" * 64,
        model_profile_sha256="3" * 64,
        configuration_sha256="4" * 64,
        source_git_commit="5" * 40,
        source_dirty=False,
        dependency_lock_hashes={"uv.lock": "6" * 64},
        container_image_digests={"evaluator": "7" * 64},
        seed=20260813,
        command_schema_version="junctionlens.evaluate.v1",
        execution_provider_profile="cpu-reference",
    )


def test_indexed_parent_provenance_and_reruns_are_byte_identical(tmp_path: Path) -> None:
    registry = _evidence_registry(tmp_path)
    parent = registry.put_bytes(
        b"parent\n",
        kind="prediction_bundle",
        media_type="application/octet-stream",
        license_id="Apache-2.0",
        metadata={"name": "parent"},
    )
    arguments = {
        "kind": "comparison",
        "media_type": "application/json",
        "license_id": "Apache-2.0",
        "metadata": {"candidate": "E1", "baseline": "E0"},
        "parents": (parent.manifest_sha256,),
    }

    first = registry.put_bytes(b'{"result":"same"}\n', **arguments)
    first_view = canonical_json_bytes(registry.provenance(first.manifest_sha256))
    reopened = _evidence_registry(tmp_path)
    second = reopened.put_bytes(b'{"result":"same"}\n', **arguments)
    second_view = canonical_json_bytes(reopened.provenance(second.manifest_sha256))

    assert first == second
    assert first_view == second_view
    view = json.loads(first_view)
    assert [item["depth"] for item in view["artifacts"]] == [0, 1]
    assert [item["kind"] for item in view["artifacts"]] == [
        "comparison",
        "prediction_bundle",
    ]


@pytest.mark.parametrize("stage", ["after_manifest_publish", "before_index_commit"])
def test_crash_injection_leaves_auditable_objects_and_rerun_recovers(
    tmp_path: Path, stage: str
) -> None:
    fired = False

    def inject(observed: str) -> None:
        nonlocal fired
        if observed == stage and not fired:
            fired = True
            raise RuntimeError(f"injected crash at {stage}")

    registry = _evidence_registry(tmp_path, fault_injector=inject)
    arguments = {
        "kind": "evidence_report",
        "media_type": "application/json",
        "license_id": "Apache-2.0",
        "metadata": {"case": stage},
    }

    with pytest.raises(RuntimeError, match="injected crash"):
        registry.put_bytes(b'{"durable":true}\n', **arguments)

    reopened = _evidence_registry(tmp_path)
    before = reopened.garbage_collection_dry_run()
    receipt = reopened.put_bytes(b'{"durable":true}\n', **arguments)
    after = reopened.garbage_collection_dry_run()

    assert before["orphaned_objects"]
    assert after["orphaned_objects"] == []
    assert reopened.index.read_artifact(receipt.manifest_sha256)["kind"] == "evidence_report"


def test_concurrent_readers_observe_complete_provenance_while_writer_lock_is_held(
    tmp_path: Path,
) -> None:
    registry = _evidence_registry(tmp_path)
    receipt = registry.put_bytes(
        b"complete\n",
        kind="evidence_report",
        media_type="application/octet-stream",
        license_id="Apache-2.0",
        metadata={},
    )
    expected = canonical_json_bytes(registry.provenance(receipt.manifest_sha256))
    results: list[bytes] = []
    errors: list[BaseException] = []

    def read() -> None:
        try:
            results.append(canonical_json_bytes(registry.provenance(receipt.manifest_sha256)))
        except BaseException as error:
            errors.append(error)

    with registry.index.writer_lock():
        threads = [threading.Thread(target=read) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert not errors
    assert len(results) == 12
    assert all(value == expected for value in results)


def test_reader_observes_committed_snapshot_during_active_writer_transaction(
    tmp_path: Path,
) -> None:
    registry = _evidence_registry(tmp_path)
    existing = registry.put_bytes(
        b"existing\n",
        kind="evidence_report",
        media_type="application/octet-stream",
        license_id="Apache-2.0",
        metadata={"order": 1},
    )
    writer_active = threading.Event()
    allow_commit = threading.Event()
    errors: list[BaseException] = []

    def inject(stage: str) -> None:
        if stage == "before_index_commit":
            writer_active.set()
            if not allow_commit.wait(timeout=10):
                raise RuntimeError("writer test timed out")

    registry.index.fault_injector = inject

    def write() -> None:
        try:
            registry.put_bytes(
                b"new\n",
                kind="evidence_report",
                media_type="application/octet-stream",
                license_id="Apache-2.0",
                metadata={"order": 2},
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=write)
    thread.start()
    assert writer_active.wait(timeout=10)

    observed = registry.provenance(existing.manifest_sha256)

    allow_commit.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors
    assert observed["root_manifest_sha256"] == existing.manifest_sha256


def _write_lock_owner(root: Path, owner: LockOwner) -> None:
    path = root / "registry-writer.lock"
    path.mkdir(parents=True)
    (path / "owner.json").write_bytes(canonical_json_bytes(asdict(owner)) + b"\n")


def test_writer_lock_refuses_live_and_foreign_owners_and_reclaims_dead_owner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "locks"
    now = time.time_ns()
    live = LockOwner(
        "junctionlens.registry-writer-lock.v1",
        "a" * 32,
        os.getpid(),
        host_fingerprint(),
        now,
        now,
    )
    _write_lock_owner(root, live)
    with pytest.raises(RegistryLockError, match="still alive"):
        RegistryWriterLock(root).acquire()

    lock_path = root / "registry-writer.lock"
    (lock_path / "owner.json").write_bytes(
        canonical_json_bytes(
            asdict(
                LockOwner(
                    "junctionlens.registry-writer-lock.v1",
                    "b" * 32,
                    2_000_000_000,
                    "f" * 64,
                    now,
                    now,
                )
            )
        )
        + b"\n"
    )
    with pytest.raises(RegistryLockError, match="unverifiable host"):
        RegistryWriterLock(root).acquire()

    (lock_path / "owner.json").write_bytes(
        canonical_json_bytes(
            asdict(
                LockOwner(
                    "junctionlens.registry-writer-lock.v1",
                    "c" * 32,
                    2_000_000_000,
                    host_fingerprint(),
                    now,
                    now,
                )
            )
        )
        + b"\n"
    )
    reclaimed = RegistryWriterLock(root).acquire()
    reclaimed.release()

    event = json.loads((root / "recovery-events.jsonl").read_text(encoding="utf-8"))
    assert event["event"] == "STALE_LOCK_RECLAIMED"
    assert event["reason_code"] == "OWNER_PROCESS_ABSENT"
    assert not lock_path.exists()


def test_only_one_concurrent_writer_can_hold_registry_lock(tmp_path: Path) -> None:
    root = tmp_path / "locks"
    acquired = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with RegistryWriterLock(root):
            acquired.set()
            release.wait(timeout=10)

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(timeout=10)
    with pytest.raises(RegistryLockError, match="still alive"):
        RegistryWriterLock(root).acquire()
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_run_resume_requires_identical_inputs_and_environment(tmp_path: Path) -> None:
    registry = _evidence_registry(tmp_path)
    parent = registry.put_bytes(
        b"parent\n",
        kind="prediction_bundle",
        media_type="application/octet-stream",
        license_id="Apache-2.0",
        metadata={},
    )
    identity = _run_identity((parent.manifest_sha256,))

    created = registry.begin_or_resume_run(identity, "8" * 64)
    resumed = registry.begin_or_resume_run(identity, "8" * 64)

    assert created["state"] == "CREATED"
    assert resumed["state"] == "RESUMED"
    assert created["run_id"] == resumed["run_id"] == identity.run_id()
    with pytest.raises(RuntimeError, match="environment fingerprint differs"):
        registry.begin_or_resume_run(identity, "9" * 64)
    registry.index.set_run_state(identity.run_id(), "COMPLETE")
    assert registry.begin_or_resume_run(identity, "8" * 64)["state"] == "ALREADY_COMPLETE"


def test_mutable_alias_is_never_reported_as_evidence_identity(tmp_path: Path) -> None:
    registry = _evidence_registry(tmp_path)
    first = registry.put_bytes(
        b"first\n",
        kind="evidence_report",
        media_type="application/octet-stream",
        license_id="Apache-2.0",
        metadata={},
    )
    second = registry.put_bytes(
        b"second\n",
        kind="evidence_report",
        media_type="application/octet-stream",
        license_id="Apache-2.0",
        metadata={},
    )

    created = registry.set_alias("models/current", first.manifest_sha256)
    updated = registry.set_alias("models/current", second.manifest_sha256)
    resolved = registry.resolve_alias("models/current")

    assert created["state"] == "CREATED"
    assert updated["state"] == "UPDATED"
    assert updated["previous_manifest_sha256"] == first.manifest_sha256
    assert resolved["manifest_sha256"] == second.manifest_sha256
    assert resolved["evidence_identifier"] is False
    with pytest.raises(RuntimeError, match="alias is invalid"):
        registry.set_alias("../escape", first.manifest_sha256)


def test_gc_dry_run_and_manifest_rebuild_never_delete_objects(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ContentAddressedStore(root, _SCHEMA)
    receipt = store.put_bytes(
        b"unindexed\n",
        kind="evidence_report",
        media_type="application/octet-stream",
        license_id="Apache-2.0",
        metadata={},
    )
    registry = EvidenceRegistry(root, _SCHEMA)
    staging = registry.store.staging_root / "interrupted.tmp"
    staging.write_bytes(b"partial")

    dry_run = registry.garbage_collection_dry_run()
    rebuilt = registry.rebuild_missing_index_rows()
    after = registry.garbage_collection_dry_run()

    assert len(dry_run["orphaned_objects"]) == 2
    assert dry_run["deleted"] is False
    assert dry_run["staging_files"] == [{"name": "interrupted.tmp", "byte_size": 7}]
    assert receipt.manifest_sha256 in rebuilt["recovered_manifest_sha256"]
    assert after["orphaned_objects"] == []
    assert staging.read_bytes() == b"partial"
    assert registry.store.object_path(receipt.payload_sha256).read_bytes() == b"unindexed\n"

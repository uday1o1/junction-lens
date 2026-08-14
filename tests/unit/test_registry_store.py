"""Content-addressed registry atomicity and reproducibility tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from junctionlens.registry import ContentAddressedStore, RegistryError

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

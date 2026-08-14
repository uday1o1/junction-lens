"""Security and reproducibility tests for remote source synchronization."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import tarfile
from pathlib import Path

import pytest
from scripts.gpu.source_bundle import (
    SourceBundleError,
    _strict_json_object,
    create_bundle,
    verify_and_extract,
)


def test_tracked_source_bundle_round_trips_and_is_reproducible(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    first_archive = tmp_path / "first.tar"
    first_manifest = tmp_path / "first.json"
    second_archive = tmp_path / "second.tar"
    second_manifest = tmp_path / "second.json"
    first = create_bundle(root, first_archive, first_manifest, require_clean=False)
    second = create_bundle(root, second_archive, second_manifest, require_clean=False)
    assert first["content_sha256"] == second["content_sha256"]
    assert first["archive_sha256"] == second["archive_sha256"]
    target = tmp_path / "checkout"
    verified = verify_and_extract(first_archive, first_manifest, target)
    assert verified["git_commit"] == first["git_commit"]
    assert (target / "BUILD_PLAN.md").is_file()
    assert (target / "configs/runtime/qualification-v1.yaml").is_file()
    for relative in (
        "scripts/gpu/benchmark_runtime.py",
        "scripts/gpu/profile_runtime.py",
    ):
        extracted = target / relative
        assert extracted.is_file()
        assert extracted.stat().st_mode & stat.S_IXUSR
    assert not (target / ".git").exists()


def test_source_bundle_rejects_traversal_even_with_matching_transport_hash(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "malicious.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("../escape")
        payload = b"forbidden"
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))
    manifest = tmp_path / "manifest.json"
    base = {
        "schema_version": "junctionlens.source-bundle.v1",
        "git_commit": "0" * 40,
        "entries": [
            {
                "path": "declared",
                "mode": "100644",
                "type": "file",
                "byte_size": 9,
                "sha256": "0" * 64,
            }
        ],
        "submodules": [],
        "dependency_lock_sha256": {},
    }
    canonical = json.dumps(
        base, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    base["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    base["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(SourceBundleError, match="undeclared|unsafe"):
        verify_and_extract(archive, manifest, tmp_path / "target")


def test_source_bundle_refuses_clobber(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "manifest.json"
    create_bundle(root, archive, manifest, require_clean=False)
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(SourceBundleError, match="already exists"):
        verify_and_extract(archive, manifest, target)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"one","schema_version":"two"}',
        ('{"schema_version":' + "[" * 20 + "0" + "]" * 20 + "}").encode(),
    ],
)
def test_source_manifest_rejects_adversarial_json_shape(payload: bytes) -> None:
    with pytest.raises(SourceBundleError):
        _strict_json_object(payload)

"""Security tests for verified bootstrap archive extraction."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from junctionlens.bootstrap import (
    ArchiveLimits,
    BootstrapError,
    extract_tar_safely,
    extract_zip_safely,
)


def test_tar_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    """A tar member cannot write above the selected destination."""
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        info = tarfile.TarInfo("../escaped")
        payload = b"unsafe"
        info.size = len(payload)
        writer.addfile(info, io.BytesIO(payload))
    with pytest.raises(BootstrapError, match="unsafe archive path"):
        extract_tar_safely(archive, tmp_path / "output", 0)
    assert not (tmp_path / "escaped").exists()


def test_tar_extraction_rejects_escaping_symlink(tmp_path: Path) -> None:
    """An archive symlink cannot escape the staging root."""
    archive = tmp_path / "bad-link.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        info = tarfile.TarInfo("nested/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escaped"
        writer.addfile(info)
    with pytest.raises(BootstrapError, match="escaping archive symlink"):
        extract_tar_safely(archive, tmp_path / "output", 0)


def test_zip_extraction_rejects_absolute_path(tmp_path: Path) -> None:
    """A ZIP member cannot use an absolute path."""
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("/absolute", "unsafe")
    with pytest.raises(BootstrapError, match="unsafe archive path"):
        extract_zip_safely(archive, tmp_path / "output", 0)


def test_safe_zip_extracts_regular_file(tmp_path: Path) -> None:
    """A valid ordinary file is preserved exactly."""
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("root/bin/tool", "content")
    destination = tmp_path / "output"
    extract_zip_safely(archive, destination, 1)
    assert (destination / "bin/tool").read_text(encoding="utf-8") == "content"


def test_archive_preflight_rejects_member_and_expansion_budgets(tmp_path: Path) -> None:
    archive = tmp_path / "bounded.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("first", "1234")
        writer.writestr("second", "5678")
    destination = tmp_path / "output"
    with pytest.raises(BootstrapError, match="member-count"):
        extract_zip_safely(
            archive,
            destination,
            0,
            limits=ArchiveLimits(max_members=1),
        )
    assert not destination.exists()

    with pytest.raises(BootstrapError, match="total expanded-byte"):
        extract_zip_safely(
            archive,
            destination,
            0,
            limits=ArchiveLimits(max_total_bytes=7),
        )
    assert not destination.exists()


def test_archive_rejects_symlink_destination(tmp_path: Path) -> None:
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("file", "content")
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "alias"
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BootstrapError, match="destination cannot be a symlink"):
        extract_zip_safely(archive, destination, 0)
    assert not (outside / "file").exists()


def test_archive_preflight_rejects_special_member_without_partial_output(tmp_path: Path) -> None:
    archive = tmp_path / "special.tar"
    with tarfile.open(archive, "w") as writer:
        ordinary = tarfile.TarInfo("ordinary")
        ordinary.size = 2
        writer.addfile(ordinary, io.BytesIO(b"ok"))
        special = tarfile.TarInfo("device")
        special.type = tarfile.CHRTYPE
        writer.addfile(special)
    destination = tmp_path / "output"

    with pytest.raises(BootstrapError, match="unsupported archive member type"):
        extract_tar_safely(archive, destination, 0)
    assert not destination.exists()


def test_archive_preflight_rejects_compression_bomb_ratio(tmp_path: Path) -> None:
    archive = tmp_path / "compressed.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        payload = b"0" * (256 * 1024)
        info = tarfile.TarInfo("payload")
        info.size = len(payload)
        writer.addfile(info, io.BytesIO(payload))
    destination = tmp_path / "output"

    with pytest.raises(BootstrapError, match="compression-ratio"):
        extract_tar_safely(
            archive,
            destination,
            0,
            limits=ArchiveLimits(max_compression_ratio=2),
        )
    assert not destination.exists()

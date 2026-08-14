"""Security tests for verified bootstrap archive extraction."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from junctionlens.bootstrap import (
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

"""Checksum verification and safe extraction for bootstrap artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


class BootstrapError(RuntimeError):
    """Raised when bootstrap integrity validation fails."""


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(name: str, strip_components: int) -> PurePosixPath | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise BootstrapError(f"unsafe archive path: {name}")
    parts = path.parts[strip_components:]
    if not parts:
        return None
    relative = PurePosixPath(*parts)
    if relative.is_absolute() or ".." in relative.parts:
        raise BootstrapError(f"unsafe stripped archive path: {name}")
    return relative


def _safe_symlink_target(destination: Path, link_target: str, root: Path) -> None:
    target = Path(link_target)
    if target.is_absolute():
        raise BootstrapError(f"absolute archive symlink target: {link_target}")
    resolved = (destination.parent / target).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise BootstrapError(f"escaping archive symlink target: {link_target}") from error


def extract_tar_safely(archive_path: Path, destination: Path, strip_components: int) -> None:
    """Extract regular files, directories, and contained symlinks only."""
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            relative = _safe_relative(member.name, strip_components)
            if relative is None:
                continue
            output = destination.joinpath(*relative.parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym():
                output.parent.mkdir(parents=True, exist_ok=True)
                _safe_symlink_target(output, member.linkname, destination)
                output.symlink_to(member.linkname)
                continue
            if not member.isfile():
                raise BootstrapError(f"unsupported archive member type: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise BootstrapError(f"archive member has no data: {member.name}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with source, output.open("wb") as output_stream:
                shutil.copyfileobj(source, output_stream)
            output.chmod(member.mode & 0o777)


def _zip_entry_kind(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0o170000


def extract_zip_safely(archive_path: Path, destination: Path, strip_components: int) -> None:
    """Extract bounded ordinary ZIP contents with traversal protection."""
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = _safe_relative(info.filename, strip_components)
            if relative is None:
                continue
            output = destination.joinpath(*relative.parts)
            kind = _zip_entry_kind(info)
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if kind == stat.S_IFLNK:
                symlink_target = archive.read(info).decode("utf-8")
                output.parent.mkdir(parents=True, exist_ok=True)
                _safe_symlink_target(output, symlink_target, destination)
                output.symlink_to(symlink_target)
                continue
            if kind not in {0, stat.S_IFREG}:
                raise BootstrapError(f"unsupported ZIP member type: {info.filename}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output.open("wb") as output_stream:
                shutil.copyfileobj(source, output_stream)
            mode = (info.external_attr >> 16) & 0o777
            output.chmod(mode or 0o644)


def download_verified(url: str, expected_sha256: str, cache_path: Path) -> Path:
    """Download one asset atomically and fail closed on hash mismatch."""
    if cache_path.is_file() and sha256_file(cache_path) == expected_sha256:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(  # noqa: S310
        url,
        headers={"User-Agent": "junctionlens-bootstrap/1"},
    )
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                shutil.copyfileobj(response, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    observed = sha256_file(temporary_path)
    if observed != expected_sha256:
        temporary_path.unlink(missing_ok=True)
        raise BootstrapError(
            f"checksum mismatch for {url}: expected {expected_sha256}, observed {observed}"
        )
    temporary_path.replace(cache_path)
    return cache_path

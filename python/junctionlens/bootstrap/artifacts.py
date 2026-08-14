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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class BootstrapError(RuntimeError):
    """Raised when bootstrap integrity validation fails."""


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    """Expanded archive resource limits enforced before extraction begins."""

    max_members: int = 100_000
    max_member_bytes: int = 2 * 1024 * 1024 * 1024
    max_total_bytes: int = 8 * 1024 * 1024 * 1024
    max_path_bytes: int = 4096
    max_compression_ratio: int = 1000

    def __post_init__(self) -> None:
        if (
            min(
                self.max_members,
                self.max_member_bytes,
                self.max_total_bytes,
                self.max_path_bytes,
                self.max_compression_ratio,
            )
            < 1
        ):
            raise ValueError("archive limits must be positive")


DEFAULT_ARCHIVE_LIMITS = ArchiveLimits()


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(name: str, strip_components: int) -> PurePosixPath | None:
    if strip_components < 0:
        raise BootstrapError("archive strip component count cannot be negative")
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


def _validate_member_budget(
    *,
    count: int,
    name: str,
    member_bytes: int,
    total_bytes: int,
    limits: ArchiveLimits,
) -> int:
    if count > limits.max_members:
        raise BootstrapError("archive exceeds the member-count limit")
    if len(name.encode("utf-8", errors="surrogateescape")) > limits.max_path_bytes:
        raise BootstrapError(f"archive member path exceeds the byte limit: {name}")
    if member_bytes < 0 or member_bytes > limits.max_member_bytes:
        raise BootstrapError(f"archive member exceeds the byte limit: {name}")
    expanded = total_bytes + member_bytes
    if expanded > limits.max_total_bytes:
        raise BootstrapError("archive exceeds the total expanded-byte limit")
    return expanded


def _prepare_destination(destination: Path) -> Path:
    if destination.is_symlink():
        raise BootstrapError("archive destination cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve(strict=True)
    if not root.is_dir():
        raise BootstrapError("archive destination must be a directory")
    return root


def _contained_output(root: Path, relative: PurePosixPath) -> Path:
    output = root.joinpath(*relative.parts)
    if output.is_symlink():
        raise BootstrapError(f"archive output cannot replace a symlink: {relative}")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        parent.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise BootstrapError(f"archive output escapes the destination: {relative}") from error
    return output


def _safe_symlink_target(destination: Path, link_target: str, root: Path) -> None:
    target = Path(link_target)
    if target.is_absolute():
        raise BootstrapError(f"absolute archive symlink target: {link_target}")
    resolved = (destination.parent / target).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise BootstrapError(f"escaping archive symlink target: {link_target}") from error


def _preflight_symlink_target(relative: PurePosixPath, link_target: str) -> None:
    virtual_root = Path("/__junctionlens_archive_root__")
    destination = virtual_root.joinpath(*relative.parts)
    _safe_symlink_target(destination, link_target, virtual_root)


def _decode_symlink_target(payload: bytes, name: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BootstrapError(f"archive symlink target is not UTF-8: {name}") from error


def extract_tar_safely(
    archive_path: Path,
    destination: Path,
    strip_components: int,
    *,
    limits: ArchiveLimits = DEFAULT_ARCHIVE_LIMITS,
) -> None:
    """Extract regular files, directories, and contained symlinks only."""
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        total_bytes = 0
        for count, member in enumerate(members, start=1):
            total_bytes = _validate_member_budget(
                count=count,
                name=member.name,
                member_bytes=member.size if member.isfile() else 0,
                total_bytes=total_bytes,
                limits=limits,
            )
            relative = _safe_relative(member.name, strip_components)
            if not (member.isdir() or member.issym() or member.isfile()):
                raise BootstrapError(f"unsupported archive member type: {member.name}")
            if relative is not None and member.issym():
                _preflight_symlink_target(relative, member.linkname)
        compressed_bytes = archive_path.stat().st_size
        if total_bytes > 0 and (
            compressed_bytes == 0 or total_bytes > compressed_bytes * limits.max_compression_ratio
        ):
            raise BootstrapError("archive exceeds the compression-ratio limit")
        root = _prepare_destination(destination)
        for member in members:
            relative = _safe_relative(member.name, strip_components)
            if relative is None:
                continue
            output = _contained_output(root, relative)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym():
                _safe_symlink_target(output, member.linkname, root)
                output.symlink_to(member.linkname)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise BootstrapError(f"archive member has no data: {member.name}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with source, output.open("wb") as output_stream:
                shutil.copyfileobj(source, output_stream)
            output.chmod(member.mode & 0o777)


def _zip_entry_kind(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0o170000


def extract_zip_safely(
    archive_path: Path,
    destination: Path,
    strip_components: int,
    *,
    limits: ArchiveLimits = DEFAULT_ARCHIVE_LIMITS,
) -> None:
    """Extract bounded ordinary ZIP contents with traversal protection."""
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        total_bytes = 0
        for count, info in enumerate(members, start=1):
            total_bytes = _validate_member_budget(
                count=count,
                name=info.filename,
                member_bytes=0 if info.is_dir() else info.file_size,
                total_bytes=total_bytes,
                limits=limits,
            )
            if (info.file_size > 0 and info.compress_size == 0) or (
                info.compress_size > 0
                and info.file_size > info.compress_size * limits.max_compression_ratio
            ):
                raise BootstrapError(
                    f"ZIP member exceeds the compression-ratio limit: {info.filename}"
                )
            relative = _safe_relative(info.filename, strip_components)
            kind = _zip_entry_kind(info)
            if not info.is_dir() and kind not in {0, stat.S_IFREG, stat.S_IFLNK}:
                raise BootstrapError(f"unsupported ZIP member type: {info.filename}")
            if relative is not None and kind == stat.S_IFLNK:
                link_target = _decode_symlink_target(archive.read(info), info.filename)
                _preflight_symlink_target(relative, link_target)
        root = _prepare_destination(destination)
        for info in members:
            relative = _safe_relative(info.filename, strip_components)
            if relative is None:
                continue
            output = _contained_output(root, relative)
            kind = _zip_entry_kind(info)
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if kind == stat.S_IFLNK:
                symlink_target = _decode_symlink_target(archive.read(info), info.filename)
                _safe_symlink_target(output, symlink_target, root)
                output.symlink_to(symlink_target)
                continue
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

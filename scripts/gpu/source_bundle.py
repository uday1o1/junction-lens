#!/usr/bin/env python3
"""Create and verify deterministic Git-tracked remote source bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "junctionlens.source-bundle.v1"
ALLOWED_MODES = {"100644", "100755", "120000", "160000"}
LOCK_PATHS = (
    "configs/toolchains/v1.lock.json",
    "configs/toolchains/gpu-v1.lock.json",
    "containers/images.lock",
    "uv.lock",
    "pnpm-lock.yaml",
)
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_DEPTH = 16
MAX_MANIFEST_NODES = 500_000


class SourceBundleError(RuntimeError):
    """Raised when source synchronization cannot be proven safe."""


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_MANIFEST_BYTES:
        raise SourceBundleError("source manifest exceeds the byte limit")

    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SourceBundleError(f"source manifest repeats JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                SourceBundleError(f"source manifest contains nonfinite value {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise SourceBundleError("source manifest is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SourceBundleError("source manifest must be an object")
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_MANIFEST_NODES:
            raise SourceBundleError("source manifest exceeds the node limit")
        if depth > MAX_MANIFEST_DEPTH:
            raise SourceBundleError("source manifest exceeds the depth limit")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return value


def _run_git(root: Path, arguments: Sequence[str]) -> bytes:
    git = shutil.which("git")
    if git is None:
        raise SourceBundleError("git is required for source synchronization")
    completed = subprocess.run(
        [git, *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise SourceBundleError(completed.stderr.decode("utf-8", "replace").strip())
    return completed.stdout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _validate_relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value or any(ord(character) < 0x20 for character in value):
        raise SourceBundleError("tracked path contains nonportable characters")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SourceBundleError(f"tracked path is unsafe: {value!r}")
    return path


def _validate_symlink(root: Path, relative: PurePosixPath, target: str) -> None:
    if not target or "\\" in target or any(ord(character) < 0x20 for character in target):
        raise SourceBundleError(f"symbolic link target is nonportable: {relative}")
    target_path = PurePosixPath(target)
    if target_path.is_absolute():
        raise SourceBundleError(f"symbolic link target is absolute: {relative}")
    lexical = relative.parent.joinpath(target_path)
    depth = 0
    for part in lexical.parts:
        if part == "..":
            depth -= 1
        elif part not in {"", "."}:
            depth += 1
        if depth < 0:
            raise SourceBundleError(f"symbolic link escapes source root: {relative}")
    resolved_parent = (root / Path(*relative.parent.parts)).resolve()
    resolved_target = (resolved_parent / target).resolve(strict=False)
    try:
        resolved_target.relative_to(root.resolve())
    except ValueError as error:
        raise SourceBundleError(f"symbolic link escapes source root: {relative}") from error


def build_manifest(root: Path, *, require_clean: bool = True) -> dict[str, Any]:
    """Inspect Git identity and return a canonical source manifest."""
    root = root.resolve()
    if require_clean:
        status = _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
        if status:
            raise SourceBundleError("source synchronization requires a clean worktree")
    commit = _run_git(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise SourceBundleError("Git did not return a lowercase commit SHA")
    records = _run_git(root, ["ls-files", "-s", "-z"]).split(b"\0")
    entries: list[dict[str, Any]] = []
    submodules: list[dict[str, str]] = []
    for record in records:
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise SourceBundleError("git ls-files returned a malformed entry")
        pieces = metadata.decode("ascii").split()
        if len(pieces) != 3:
            raise SourceBundleError("git ls-files returned malformed metadata")
        mode, object_sha, stage = pieces
        if stage != "0" or mode not in ALLOWED_MODES:
            raise SourceBundleError("source bundle contains a staged conflict or special file")
        path_text = raw_path.decode("utf-8", "strict")
        relative = _validate_relative_path(path_text)
        local = root / Path(*relative.parts)
        if mode == "160000":
            submodules.append({"path": path_text, "commit": object_sha})
            entries.append(
                {"path": path_text, "mode": mode, "type": "gitlink", "commit": object_sha}
            )
            continue
        if mode == "120000":
            if not local.is_symlink():
                raise SourceBundleError(f"tracked symbolic link is missing: {path_text}")
            target = str(local.readlink())
            _validate_symlink(root, relative, target)
            entries.append({"path": path_text, "mode": mode, "type": "symlink", "target": target})
            continue
        file_status = local.stat(follow_symlinks=False)
        if not stat.S_ISREG(file_status.st_mode):
            raise SourceBundleError(f"tracked path is not a regular file: {path_text}")
        entries.append(
            {
                "path": path_text,
                "mode": mode,
                "type": "file",
                "byte_size": file_status.st_size,
                "sha256": _sha256_file(local),
            }
        )
    entries.sort(key=lambda entry: str(entry["path"]))
    submodules.sort(key=lambda entry: entry["path"])
    locks = {path: _sha256_file(root / path) for path in LOCK_PATHS if (root / path).is_file()}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": commit,
        "entries": entries,
        "submodules": submodules,
        "dependency_lock_sha256": locks,
    }
    manifest["content_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    return manifest


def create_bundle(
    root: Path,
    archive_path: Path,
    manifest_path: Path,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Create a deterministic uncompressed tar from declared tracked files."""
    root = root.resolve()
    manifest = build_manifest(root, require_clean=require_clean)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() or manifest_path.exists():
        raise SourceBundleError("source bundle outputs already exist")
    temporary_archive = archive_path.with_name(f".{archive_path.name}.tmp")
    try:
        with tarfile.open(temporary_archive, "w", format=tarfile.PAX_FORMAT) as archive:
            for entry in manifest["entries"]:
                if entry["type"] == "gitlink":
                    continue
                relative = _validate_relative_path(str(entry["path"]))
                local = root / Path(*relative.parts)
                info = tarfile.TarInfo(str(relative))
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.mode = 0o755 if entry["mode"] == "100755" else 0o644
                if entry["type"] == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = str(entry["target"])
                    info.size = 0
                    archive.addfile(info)
                else:
                    info.type = tarfile.REGTYPE
                    info.size = int(entry["byte_size"])
                    with local.open("rb") as source:
                        archive.addfile(info, source)
        archive_sha256 = _sha256_file(temporary_archive)
        final_manifest = dict(manifest)
        final_manifest["archive_sha256"] = archive_sha256
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.write_bytes(_canonical_json(final_manifest) + b"\n")
        temporary_archive.replace(archive_path)
        temporary_manifest.replace(manifest_path)
        return final_manifest
    finally:
        temporary_archive.unlink(missing_ok=True)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise SourceBundleError("source manifest is not a regular file")
        value = _strict_json_object(path.read_bytes())
    except OSError as error:
        raise SourceBundleError("source manifest is unreadable") from error
    if value.get("schema_version") != SCHEMA_VERSION:
        raise SourceBundleError("source manifest schema is invalid")
    content_sha256 = value.get("content_sha256")
    archive_sha256 = value.get("archive_sha256")
    without_transport = {
        key: item for key, item in value.items() if key not in {"content_sha256", "archive_sha256"}
    }
    if content_sha256 != hashlib.sha256(_canonical_json(without_transport)).hexdigest():
        raise SourceBundleError("source manifest content digest is invalid")
    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
        raise SourceBundleError("source manifest archive digest is invalid")
    return value


def verify_and_extract(archive_path: Path, manifest_path: Path, target: Path) -> dict[str, Any]:
    """Verify an archive before and after safe no-clobber extraction."""
    manifest = _load_manifest(manifest_path)
    if _sha256_file(archive_path) != manifest["archive_sha256"]:
        raise SourceBundleError("source archive digest mismatch")
    expected = {
        str(entry["path"]): entry for entry in manifest["entries"] if entry["type"] != "gitlink"
    }
    if target.exists():
        raise SourceBundleError("source extraction target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            observed_names: set[str] = set()
            for member in members:
                relative = _validate_relative_path(member.name)
                name = str(relative)
                if name in observed_names or name not in expected:
                    raise SourceBundleError("source archive contains an undeclared path")
                observed_names.add(name)
                entry = expected[name]
                destination = staging / Path(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if entry["type"] == "symlink":
                    if not member.issym() or member.linkname != entry["target"]:
                        raise SourceBundleError("source archive symbolic link differs")
                    _validate_symlink(staging, relative, member.linkname)
                    destination.symlink_to(member.linkname)
                    continue
                if not member.isreg() or member.size != entry["byte_size"]:
                    raise SourceBundleError("source archive file type or size differs")
                source = archive.extractfile(member)
                if source is None:
                    raise SourceBundleError("source archive file cannot be read")
                with destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                destination.chmod(0o755 if entry["mode"] == "100755" else 0o644)
            if observed_names != set(expected):
                raise SourceBundleError("source archive is missing declared paths")
        for name, entry in expected.items():
            destination = staging / Path(*PurePosixPath(name).parts)
            if entry["type"] == "file" and (
                destination.stat().st_size != entry["byte_size"]
                or _sha256_file(destination) != entry["sha256"]
            ):
                raise SourceBundleError("extracted source file digest differs")
            if entry["type"] == "symlink" and str(destination.readlink()) != entry["target"]:
                raise SourceBundleError("extracted source symbolic link differs")
        staging.replace(target)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def create_remote_config(
    manifest_path: Path,
    output_path: Path,
    *,
    profile: str,
    remote_data_root: str | None,
    gpu_uuid: str | None,
) -> dict[str, Any]:
    """Create the non-secret configuration consumed by the remote runner."""
    if profile not in {"m0.3", "runtime-cuda", "runtime-performance", "core", "full-v1"}:
        raise SourceBundleError("remote qualification profile is invalid")
    manifest = _load_manifest(manifest_path)
    value = {
        "schema_version": "junctionlens.remote-qualification-config.v1",
        "profile": profile,
        "source_commit": manifest["git_commit"],
        "source_content_sha256": manifest["content_sha256"],
        "remote_data_root": remote_data_root or None,
        "gpu_uuid": gpu_uuid or None,
    }
    if output_path.exists():
        raise SourceBundleError("remote configuration output already exists")
    output_path.write_bytes(_canonical_json(value) + b"\n")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--archive", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    extract = subparsers.add_parser("verify-extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--target", type=Path, required=True)
    config = subparsers.add_parser("make-config")
    config.add_argument("--manifest", type=Path, required=True)
    config.add_argument("--output", type=Path, required=True)
    config.add_argument("--profile", required=True)
    config.add_argument("--remote-data-root")
    config.add_argument("--gpu-uuid")
    return parser.parse_args()


def main() -> int:
    """Run the source-bundle command line interface."""
    arguments = _parse_args()
    try:
        if arguments.command == "create":
            result = create_bundle(arguments.root, arguments.archive, arguments.manifest)
        elif arguments.command == "verify-extract":
            result = verify_and_extract(arguments.archive, arguments.manifest, arguments.target)
        else:
            result = create_remote_config(
                arguments.manifest,
                arguments.output,
                profile=arguments.profile,
                remote_data_root=arguments.remote_data_root,
                gpu_uuid=arguments.gpu_uuid,
            )
    except SourceBundleError as error:
        print(f"source bundle error: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

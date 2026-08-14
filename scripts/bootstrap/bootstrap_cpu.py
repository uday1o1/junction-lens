#!/usr/bin/env python3
"""Install verified repository-local CPU development tools."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PYTHON_SOURCE = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from junctionlens.bootstrap import (  # noqa: E402
    BootstrapError,
    download_verified,
    extract_tar_safely,
    extract_zip_safely,
)


def project_root() -> Path:
    """Return the repository root from this script location."""
    return Path(__file__).resolve().parents[2]


def platform_key() -> str:
    """Map supported hosts to lock-file platform keys."""
    import platform

    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        return "darwin-arm64"
    if system == "Linux" and machine in {"x86_64", "AMD64"}:
        return "linux-x86_64"
    raise BootstrapError(f"unsupported bootstrap platform: {system} {machine}")


def _load_lock(root: Path) -> Mapping[str, Any]:
    with (root / "configs/toolchains/v1.lock.json").open(encoding="utf-8") as source:
        payload: Mapping[str, Any] = json.load(source)
    return payload


def _archive_suffix(asset: Mapping[str, Any]) -> str:
    archive_type = str(asset["archive_type"])
    return {"tar.gz": ".tar.gz", "tar.xz": ".tar.xz", "zip": ".zip"}[archive_type]


def install_tool(root: Path, name: str, spec: Mapping[str, Any], key: str) -> Path:
    """Install one locked tool below `.tools` with atomic replacement."""
    version = str(spec["version"])
    target = root / ".tools" / name / version
    marker = target / ".junctionlens-tool.json"
    if marker.is_file():
        return target
    assets = spec["assets"]
    asset: Mapping[str, Any] = assets[key]
    cache_path = root / ".cache" / "bootstrap" / f"{name}-{version}-{key}{_archive_suffix(asset)}"
    archive = download_verified(str(asset["url"]), str(asset["sha256"]), cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-{version}-", dir=target.parent))
    try:
        archive_type = str(asset["archive_type"])
        strip_components = int(asset["strip_components"])
        if archive_type in {"tar.gz", "tar.xz"}:
            extract_tar_safely(archive, staging, strip_components)
        elif archive_type == "zip":
            extract_zip_safely(archive, staging, strip_components)
        else:
            raise BootstrapError(f"unsupported archive type: {archive_type}")
        marker_payload = {
            "name": name,
            "version": version,
            "platform": key,
            "archive_sha256": str(asset["sha256"]),
        }
        (staging / ".junctionlens-tool.json").write_text(
            json.dumps(marker_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def _link_binary(root: Path, target: Path, name: str, relative: str) -> None:
    bin_root = root / ".tools" / "bin"
    bin_root.mkdir(parents=True, exist_ok=True)
    link = bin_root / name
    source = target / relative
    if not source.is_file():
        raise BootstrapError(f"installed tool is missing {source}")
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(source)


def _run_checked(command: Iterable[str], env: Mapping[str, str]) -> None:
    result = subprocess.run(list(command), check=False, env=dict(env))
    if result.returncode != 0:
        raise BootstrapError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def bootstrap(sync: bool) -> None:
    """Install all locked tools and optionally sync application dependencies."""
    root = project_root()
    key = platform_key()
    lock = _load_lock(root)
    tools: Mapping[str, Mapping[str, Any]] = lock["tools"]
    installed = {name: install_tool(root, name, spec, key) for name, spec in tools.items()}
    _link_binary(root, installed["uv"], "uv", "uv")
    _link_binary(root, installed["uv"], "uvx", "uvx")
    _link_binary(root, installed["cmake"], "cmake", "bin/cmake")
    _link_binary(root, installed["cmake"], "ctest", "bin/ctest")
    _link_binary(root, installed["ninja"], "ninja", "ninja")
    _link_binary(root, installed["protoc"], "protoc", "bin/protoc")
    _link_binary(root, installed["node"], "node", "bin/node")
    _link_binary(root, installed["node"], "npm", "bin/npm")
    _link_binary(root, installed["node"], "npx", "bin/npx")
    env = os.environ.copy()
    env["PATH"] = f"{root / '.tools/bin'}:{installed['node'] / 'bin'}:{env.get('PATH', '')}"
    env["COREPACK_HOME"] = str(root / ".tools" / "corepack")
    env["CI"] = "true"
    env["UV_CACHE_DIR"] = str(root / ".cache" / "uv")
    corepack = installed["node"] / "bin" / "corepack"
    if not corepack.is_file():
        raise BootstrapError("locked Node distribution does not contain corepack")
    _run_checked([str(corepack), "prepare", "pnpm@10.17.1", "--activate"], env)
    _run_checked(
        [
            str(corepack),
            "enable",
            "pnpm",
            "--install-directory",
            str(installed["node"] / "bin"),
        ],
        env,
    )
    _link_binary(root, installed["node"], "corepack", "bin/corepack")
    pnpm = installed["node"] / "bin" / "pnpm"
    if not pnpm.exists():
        raise BootstrapError("corepack did not activate pnpm 10.17.1")
    _link_binary(root, installed["node"], "pnpm", "bin/pnpm")
    if sync:
        _run_checked(
            [
                str(root / ".tools/bin/uv"),
                "sync",
                "--locked",
                "--python",
                "3.12.13",
                "--extra",
                "cpu",
                "--extra",
                "analytics",
                "--extra",
                "service",
            ],
            env,
        )
        _run_checked([str(root / ".tools/bin/pnpm"), "install", "--frozen-lockfile"], env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sync", action="store_true", help="install tools without dependencies")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        arguments = parse_args()
        bootstrap(sync=not arguments.no_sync)
    except BootstrapError as error:
        print(f"bootstrap error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

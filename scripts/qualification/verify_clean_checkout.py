#!/usr/bin/env python3
"""Export and verify the exact staged source tree with fresh local caches."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


class CleanCheckoutError(RuntimeError):
    """Raised when an isolated qualification cannot prove a clean workflow."""


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: int,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"clean-checkout: {subprocess.list2cmdline(list(command))}", flush=True)
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        check=False,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )
    if result.returncode != 0:
        details = ""
        if capture:
            details = f"\nstdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        raise CleanCheckoutError(
            f"command failed with exit code {result.returncode}: {command}{details}"
        )
    return result


def _git(root: Path) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise CleanCheckoutError("Git is required for clean-checkout qualification")
    probe = subprocess.run(
        [executable, "rev-parse", "--show-toplevel"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode != 0 or Path(probe.stdout.strip()).resolve() != root.resolve():
        raise CleanCheckoutError("qualification must run from the repository root")
    return executable


def _candidate_is_fully_staged(root: Path, git: str) -> None:
    unstaged = subprocess.run([git, "diff", "--quiet"], cwd=root, check=False, timeout=10)
    if unstaged.returncode not in {0, 1}:
        raise CleanCheckoutError("Git could not inspect unstaged changes")
    if unstaged.returncode == 1:
        raise CleanCheckoutError(
            "clean-checkout qualification requires every candidate source change to be staged"
        )
    status = subprocess.run(
        [git, "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if status.returncode != 0:
        raise CleanCheckoutError("Git could not inspect candidate source status")
    untracked = [line for line in status.stdout.splitlines() if line.startswith("??")]
    if untracked:
        raise CleanCheckoutError(f"candidate contains untracked source: {untracked}")


def _index_listing(root: Path, git: str) -> str:
    result = subprocess.run(
        [git, "ls-files", "--stage"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def _export_index(source: Path, checkout: Path, git: str) -> None:
    checkout.mkdir(mode=0o755)
    commit_environment = os.environ.copy()
    commit_environment.update(
        {
            "GIT_AUTHOR_DATE": "2026-08-14T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-08-14T00:00:00+00:00",
        }
    )
    _run(
        [git, "checkout-index", "--all", f"--prefix={checkout}{os.sep}"],
        cwd=source,
        env=os.environ,
        timeout=120,
    )
    _run([git, "init", "--quiet"], cwd=checkout, env=os.environ, timeout=30)
    _run([git, "add", "--all"], cwd=checkout, env=os.environ, timeout=120)
    if _index_listing(source, git) != _index_listing(checkout, git):
        raise CleanCheckoutError("exported checkout differs from the candidate Git index")
    _run(
        [
            git,
            "-c",
            "user.name=JunctionLens Qualification",
            "-c",
            "user.email=qualification@invalid.local",
            "commit",
            "--quiet",
            "-m",
            "test: materialize qualification candidate",
        ],
        cwd=checkout,
        env=commit_environment,
        timeout=120,
    )


def _isolated_environment(checkout: Path) -> dict[str, str]:
    environment = os.environ.copy()
    isolated_cache = checkout / ".cache" / "clean-checkout"
    environment.update(
        {
            "CI": "true",
            "COREPACK_HOME": str(isolated_cache / "corepack"),
            "NPM_CONFIG_CACHE": str(isolated_cache / "npm"),
            "npm_config_store_dir": str(isolated_cache / "pnpm-store"),
            "PLAYWRIGHT_BROWSERS_PATH": str(isolated_cache / "playwright"),
            "UV_CACHE_DIR": str(isolated_cache / "uv"),
            "XDG_CACHE_HOME": str(isolated_cache / "xdg"),
        }
    )
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("PYTHONPATH", None)
    return environment


def _assert_clean(checkout: Path, git: str) -> None:
    result = subprocess.run(
        [git, "status", "--porcelain", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.stdout:
        raise CleanCheckoutError(
            f"qualified workflow left unexplained source changes:\n{result.stdout}"
        )


def _temporary_root(source: Path) -> Path | None:
    configured = os.environ.get("JL_QUALIFICATION_TMPDIR")
    if configured is not None:
        candidate = Path(configured).expanduser()
        if candidate.is_symlink():
            raise CleanCheckoutError("JL_QUALIFICATION_TMPDIR must be a real directory")
        root = candidate.resolve(strict=True)
        if not root.is_dir():
            raise CleanCheckoutError("JL_QUALIFICATION_TMPDIR must be a real directory")
        return root
    if platform.system() == "Darwin":
        # Docker Desktop and Colima both share the user's source tree by default,
        # while their treatment of macOS /private/tmp differs.
        # A fresh sibling remains isolated from the candidate and visible to both.
        return source.parent
    return None


def qualify(source: Path, *, keep: bool) -> Path | None:
    source = source.resolve(strict=True)
    git = _git(source)
    _candidate_is_fully_staged(source, git)
    temporary = Path(
        tempfile.mkdtemp(prefix="junctionlens-clean-checkout-", dir=_temporary_root(source))
    )
    checkout = temporary / "junction-lens"
    try:
        _export_index(source, checkout, git)
        environment = _isolated_environment(checkout)
        _run(["./tools/jl", "bootstrap-cpu"], cwd=checkout, env=environment, timeout=1800)
        _run(
            ["./tools/jl", "install-browser"],
            cwd=checkout,
            env=environment,
            timeout=900,
        )
        _run(["./tools/jl", "verify-m10-1"], cwd=checkout, env=environment, timeout=2400)
        _run(
            ["./tools/jl", "demo-synthetic", "--output", "artifacts/demo"],
            cwd=checkout,
            env=environment,
            timeout=600,
        )
        _run(
            [
                "./tools/jl",
                "inspect-demo",
                "--artifact-root",
                "artifacts/demo",
            ],
            cwd=checkout,
            env=environment,
            timeout=180,
        )
        _assert_clean(checkout, git)
        receipt = {
            "schema_version": "junctionlens.clean-checkout-qualification.v1",
            "state": "ACCEPTED",
            "source": "exact-candidate-git-index",
            "dependency_cache": "fresh-checkout-local",
            "workflow": [
                "bootstrap-cpu",
                "verify-m10-1",
                "demo-synthetic",
                "inspect-demo",
            ],
            "worktree_clean": True,
        }
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
        if keep:
            return checkout
        return None
    finally:
        if not keep:
            shutil.rmtree(temporary, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).parents[2], type=Path)
    parser.add_argument("--keep", action="store_true")
    arguments = parser.parse_args()
    try:
        kept = qualify(arguments.project_root, keep=arguments.keep)
    except (CleanCheckoutError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"clean-checkout qualification error: {error}\n")
    if kept is not None:
        print(f"clean-checkout: retained {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

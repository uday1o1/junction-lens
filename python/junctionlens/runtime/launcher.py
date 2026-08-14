"""Safe public launcher for the native JunctionLens batch runtime."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

from junctionlens.model.profile import load_m0_profile


class RuntimeLaunchError(RuntimeError):
    """Raised when the native runtime cannot produce a valid batch receipt."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(project_root: Path) -> tuple[str, bool]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeLaunchError("git is required to record runtime provenance")
    revision = subprocess.run(
        [git, "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if revision.returncode != 0:
        raise RuntimeLaunchError(f"cannot resolve Git revision: {revision.stderr.strip()}")
    commit = revision.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeLaunchError("Git revision is not a lowercase 40-character SHA")
    status = subprocess.run(
        [git, "status", "--porcelain", "--untracked-files=normal"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if status.returncode != 0:
        raise RuntimeLaunchError(f"cannot inspect Git worktree: {status.stderr.strip()}")
    return commit, bool(status.stdout.strip())


def run_cpu_batch(
    *,
    model_path: Path,
    profile_path: Path,
    input_list_path: Path,
    asset_root: Path,
    output_directory: Path,
    runtime_binary: Path,
    project_root: Path,
    repeat_loads: int = 1,
    buffer_slots: int = 2,
) -> dict[str, Any]:
    """Execute the real C++ CPU path and validate its versioned receipt."""
    if not 1 <= repeat_loads <= 100:
        raise RuntimeLaunchError("repeat_loads must be within [1, 100]")
    if not 1 <= buffer_slots <= 1024:
        raise RuntimeLaunchError("buffer_slots must be within [1, 1024]")
    for path, label in (
        (model_path, "model"),
        (profile_path, "profile"),
        (input_list_path, "input list"),
        (runtime_binary, "runtime binary"),
    ):
        if not path.is_file():
            raise RuntimeLaunchError(f"{label} is not a file: {path}")
    if not asset_root.is_dir() or not project_root.is_dir():
        raise RuntimeLaunchError("asset root and project root must be directories")
    profile = load_m0_profile(profile_path)
    commit, dirty = _git_state(project_root)
    command = [
        str(runtime_binary.resolve()),
        "infer",
        "--model",
        str(model_path.resolve()),
        "--expected-profile-sha256",
        profile.canonical_sha256(),
        "--input-list",
        str(input_list_path.resolve()),
        "--asset-root",
        str(asset_root.resolve()),
        "--output-dir",
        str(output_directory.resolve()),
        "--git-commit",
        commit,
        "--configuration-sha256",
        profile.canonical_sha256(),
        "--runtime-build-sha256",
        _sha256_file(runtime_binary),
        "--repeat-loads",
        str(repeat_loads),
        "--buffer-slots",
        str(buffer_slots),
    ]
    if dirty:
        command.append("--git-dirty")
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(120, 120 * repeat_loads),
    )
    if completed.returncode != 0:
        raise RuntimeLaunchError(f"native runtime failed: {completed.stderr.strip()}")
    try:
        receipt = cast(dict[str, Any], json.loads(completed.stdout))
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeLaunchError("native runtime returned malformed JSON") from error
    expected = {
        "schema_version": "junctionlens.runtime-batch.v1",
        "status": "PASSED",
        "provider": "CPUExecutionProvider",
        "processed_frames": sum(
            1
            for line in input_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ),
        "repeat_loads": repeat_loads,
        "buffer_capacity": buffer_slots,
        "all_slots_free": True,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeLaunchError(f"native runtime receipt differs at {key}")
    high_water = receipt.get("buffer_high_water_mark")
    if not isinstance(high_water, int) or not 1 <= high_water <= buffer_slots:
        raise RuntimeLaunchError("native runtime reported an invalid buffer high-water mark")
    return receipt

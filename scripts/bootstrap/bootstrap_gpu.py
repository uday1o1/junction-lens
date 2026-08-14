#!/usr/bin/env python3
"""Install the exact GPU qualification runtime into repository-local tools."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

PYTHON_SOURCE = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from junctionlens.bootstrap import (  # noqa: E402
    BootstrapError,
    download_verified,
    extract_tar_safely,
)


def bootstrap_gpu() -> Path:
    """Install the locked Linux x86-64 ONNX Runtime GPU release artifact."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise BootstrapError("GPU bootstrap requires Linux x86-64")
    root = Path(__file__).resolve().parents[2]
    lock = cast(
        dict[str, Any],
        json.loads((root / "configs/toolchains/gpu-v1.lock.json").read_text(encoding="utf-8")),
    )
    spec = cast(dict[str, Any], lock["onnxruntime_gpu"])
    target = root / ".tools" / "onnxruntime-gpu-cpp" / str(spec["version"])
    marker = target / ".junctionlens-tool.json"
    if marker.is_file():
        return target
    archive = download_verified(
        str(spec["url"]),
        str(spec["sha256"]),
        root / ".cache" / "bootstrap" / f"onnxruntime-gpu-{spec['version']}.tgz",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        extract_tar_safely(archive, staging, int(spec["strip_components"]))
        required = (
            staging / "include" / "onnxruntime_cxx_api.h",
            staging / "lib" / "libonnxruntime.so",
            staging / "lib" / "libonnxruntime_providers_cuda.so",
        )
        if not all(path.is_file() for path in required):
            raise BootstrapError("locked GPU runtime archive lacks required CUDA provider files")
        marker.write_text(
            json.dumps(
                {
                    "name": "onnxruntime-gpu-cpp",
                    "version": spec["version"],
                    "platform": lock["platform"],
                    "archive_sha256": spec["sha256"],
                    "role": "qualification-release-artifact",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


if __name__ == "__main__":
    try:
        print(bootstrap_gpu())
    except BootstrapError as error:
        print(f"GPU bootstrap error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

#!/usr/bin/env python3
"""Inspect generated demonstration evidence in a real browser."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class InspectionError(RuntimeError):
    """Raised when the local product cannot be inspected safely."""


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_health(url: str, server: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if server.poll() is not None:
            stdout, stderr = server.communicate(timeout=5)
            raise InspectionError(
                f"demo server exited early: stdout={stdout[-1000:]!r}, stderr={stderr[-1000:]!r}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:  # noqa: S310
                if response.status == 200:
                    payload = json.loads(response.read())
                    if payload.get("state") == "READY":
                        return
        except (OSError, TimeoutError, urllib.error.URLError, ValueError):
            pass
        time.sleep(0.1)
    raise InspectionError("demo server did not become ready within 30 seconds")


def inspect(project_root: Path, artifact_root: Path) -> None:
    project_root = project_root.resolve(strict=True)
    artifact_root = artifact_root.resolve(strict=True)
    port = _port()
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [
            str(project_root / ".venv/bin/junctionlens"),
            "serve",
            "--artifact-root",
            str(artifact_root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--schema",
            str(project_root / "schemas/artifact-manifest-v1.schema.json"),
            "--web-root",
            str(project_root / "web/dist"),
        ],
        cwd=project_root,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_health(f"{url}/api/v1/health", server)
        subprocess.run(
            [
                str(project_root / ".tools/bin/node"),
                str(project_root / "scripts/qualification/inspect_demo.mjs"),
                "--url",
                url,
                "--screenshot",
                str(artifact_root / "browser-inspection.png"),
            ],
            cwd=project_root,
            check=True,
            timeout=60,
        )
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--project-root", default=Path(__file__).parents[2], type=Path)
    arguments = parser.parse_args()
    try:
        inspect(arguments.project_root, arguments.artifact_root)
    except (InspectionError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"demo browser inspection error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

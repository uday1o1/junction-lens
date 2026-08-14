"""Security tests for the host-to-container evaluator boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from junctionlens.evaluator.official import evaluate_official


def test_official_wrapper_enforces_restricted_docker_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    (root / "containers").mkdir(parents=True)
    request = root / "request.json"
    source = Path("tests/fixtures/evaluator/perfect.json").read_bytes()
    request.write_bytes(source)
    image = {
        "config_sha256": "a" * 64,
        "local_reference": "junctionlens/official-evaluator:test",
        "platform_manifest_sha256": "b" * 64,
        "state": "ACCEPTED_LOCAL",
    }
    (root / "containers/images.lock").write_text(
        yaml.safe_dump({"application_images": {"official_evaluator": image}}),
        encoding="utf-8",
    )
    output = {
        "environment": {
            "numpy": "1.23.5",
            "openlane_v2": "2.1.0",
            "ortools": "9.3.10497",
            "python": "3.8.20",
            "scipy": "1.8.0",
            "shapely": "2.0.0",
        },
        "input_sha256": hashlib.sha256(source).hexdigest(),
        "matching": {},
        "metrics": {
            "DET_a": 1.0,
            "DET_l": 1.0,
            "DET_t": 1.0,
            "OLUS": 1.0,
            "TOP_ll": 1.0,
            "TOP_lt": 1.0,
        },
        "schema_version": "junctionlens.official-evaluator-output.v1",
    }
    commands: list[list[str]] = []
    monkeypatch.setenv("JUNCTIONLENS_DOCKER_STAGING_ROOT", str(root / "staging"))

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=f"sha256:{'a' * 64}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/docker")
    assert evaluate_official(request, root)["metrics"]["OLUS"] == 1.0
    run_command = commands[1]
    assert run_command[1:4] == ["run", "--rm", "--platform"]
    assert run_command[run_command.index("--platform") + 1] == "linux/amd64"
    assert "none" in run_command
    assert "--read-only" in run_command
    assert run_command[run_command.index("--cap-drop") + 1] == "ALL"
    assert run_command[run_command.index("--security-opt") + 1] == "no-new-privileges"
    assert run_command[run_command.index("--user") + 1] == "65532:65532"
    mount = run_command[run_command.index("--mount") + 1]
    assert mount.endswith(",readonly")

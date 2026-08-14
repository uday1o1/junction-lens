"""Security tests for the official adapter parity container boundary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from junctionlens.data.parity import run_official_projection


def test_parity_wrapper_mounts_dataset_read_only_and_disables_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Licensed frames are exposed only to the locked, read-only compatibility image."""
    repository = tmp_path / "repository"
    runner = repository / "containers/adapter_parity_runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# test runner\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    commands: list[list[str]] = []
    output = {
        "schema_version": "junctionlens.openlane-official-projection.v1",
        "devkit_version": "2.1.0",
        "frames": [{}],
    }

    monkeypatch.setattr(
        "junctionlens.data.parity.load_evaluator_image_contract",
        lambda _: {
            "local_reference": "junctionlens/official-evaluator:test",
            "config_sha256": "a" * 64,
            "platform_manifest_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr("junctionlens.data.parity.inspect_evaluator_image", lambda *_: None)
    monkeypatch.setattr("junctionlens.data.parity.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setenv("JUNCTIONLENS_DOCKER_STAGING_ROOT", str(tmp_path / "staging"))

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr("junctionlens.data.parity.subprocess.run", fake_run)
    assert (
        run_official_projection(
            dataset,
            [("train", "segment", "100")],
            repository,
        )["devkit_version"]
        == "2.1.0"
    )
    command = commands[0]
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert len(mounts) == 3
    assert all(mount.endswith(",readonly") for mount in mounts)
    assert any("dst=/dataset" in mount for mount in mounts)
    assert command[command.index("--entrypoint") + 1] == "python"

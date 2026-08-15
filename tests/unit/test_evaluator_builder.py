"""Tests for the digest-pinned evaluator image builder."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.containers import build_evaluator

from junctionlens.locks import verify

_DIGEST = "0168606be2315b7c807a03b3d8aa79beefdb31c98740cebdffdfeebf31190c9f"
_IMAGE = f"docker.io/moby/buildkit@sha256:{_DIGEST}"
_SPEC = {
    "driver": "docker-container",
    "image": _IMAGE,
    "image_index_sha256": _DIGEST,
    "version": "0.30.0",
}


def test_pinned_builder_uses_exact_image_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The public build path creates and removes repository-scoped builder state."""
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(
        command: list[str],
        root: Path,
        timeout: int = 3_600,
        *,
        env: dict[str, str] | None = None,
    ) -> str:
        del root, timeout
        calls.append((command, dict(env or {})))
        if command[1] == "inspect":
            return "Driver:        docker-container\nBuildKit version:      v0.30.0\n"
        if command[1:3] == ["container", "inspect"]:
            return f"{_IMAGE}\n"
        return ""

    monkeypatch.setattr(build_evaluator, "_run", fake_run)
    monkeypatch.setattr(build_evaluator, "_docker", lambda: "docker")
    monkeypatch.setattr(
        build_evaluator.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="0123456789abcdef"),
    )
    with build_evaluator._pinned_builder(tmp_path, _SPEC) as (name, environment):
        assert name == "junctionlens-evaluator-0123456789ab"
        config = Path(environment["BUILDX_CONFIG"])
        assert config.is_dir()
        assert config.is_relative_to(tmp_path / ".cache/evaluator-build")

    assert not config.exists()
    assert calls[0][0] == [
        str(tmp_path / ".tools/bin/docker-buildx"),
        "create",
        "--name",
        "junctionlens-evaluator-0123456789ab",
        "--driver",
        "docker-container",
        "--driver-opt",
        f"image={_IMAGE}",
    ]
    assert calls[-1][0][1:] == [
        "rm",
        "--force",
        "junctionlens-evaluator-0123456789ab",
    ]
    assert all(call_env["BUILDX_CONFIG"] == str(config) for _, call_env in calls)


def test_pinned_builder_rejects_image_drift_and_still_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mutable or substituted builder cannot produce accepted evaluator evidence."""
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        root: Path,
        timeout: int = 3_600,
        *,
        env: dict[str, str] | None = None,
    ) -> str:
        del root, timeout, env
        commands.append(command)
        if command[1] == "inspect":
            return "Driver:        docker-container\nBuildKit version:      v0.30.0\n"
        if command[1:3] == ["container", "inspect"]:
            return "docker.io/moby/buildkit:latest\n"
        return ""

    monkeypatch.setattr(build_evaluator, "_run", fake_run)
    monkeypatch.setattr(build_evaluator, "_docker", lambda: "docker")
    with (
        pytest.raises(
            build_evaluator.EvaluatorBuildError,
            match="running BuildKit image differs",
        ),
        build_evaluator._pinned_builder(tmp_path, _SPEC),
    ):
        pytest.fail("builder with a substituted image must not be yielded")

    assert commands[-1][1:3] == ["rm", "--force"]
    assert not list((tmp_path / ".cache/evaluator-build").glob("buildx-config-*"))


def test_image_lock_validator_rejects_mutable_builder_reference() -> None:
    """The structural lock gate rejects a tag even when a digest is recorded beside it."""
    payload = verify._load_yaml(Path("containers/images.lock"))
    corrupted = copy.deepcopy(payload)
    corrupted["builders"]["evaluator"]["image"] = "docker.io/moby/buildkit:v0.30.0"
    with pytest.raises(verify.LockVerificationError, match="not digest-pinned"):
        verify._validate_images(corrupted)

"""Unit tests for clean-checkout qualification isolation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path("scripts/qualification/verify_clean_checkout.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("junctionlens_clean_checkout", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_darwin_uses_docker_shared_source_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    source = tmp_path / "repository"
    source.mkdir()
    monkeypatch.delenv("JL_QUALIFICATION_TMPDIR", raising=False)
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")

    assert module._temporary_root(source) == tmp_path


def test_non_darwin_uses_platform_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    source = tmp_path / "repository"
    source.mkdir()
    monkeypatch.delenv("JL_QUALIFICATION_TMPDIR", raising=False)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    assert module._temporary_root(source) is None


def test_explicit_temporary_root_overrides_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    source = tmp_path / "repository"
    override = tmp_path / "override"
    source.mkdir()
    override.mkdir()
    monkeypatch.setenv("JL_QUALIFICATION_TMPDIR", str(override))
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")

    assert module._temporary_root(source) == override


def test_explicit_temporary_root_rejects_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    source = tmp_path / "repository"
    target = tmp_path / "target"
    override = tmp_path / "override"
    source.mkdir()
    target.mkdir()
    override.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("JL_QUALIFICATION_TMPDIR", str(override))

    with pytest.raises(module.CleanCheckoutError, match="must be a real directory"):
        module._temporary_root(source)

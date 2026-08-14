"""Tests for dependency-lock validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from junctionlens.locks import verify


def test_repository_lock_set_is_structurally_complete() -> None:
    """The committed lock set covers every required identity."""
    assert Path("configs/data/openlane-v2-v2.1.lock.yaml").is_file()
    verify.validate_lock_set(Path.cwd())


def test_toolchain_validator_rejects_a_seeded_bad_hash() -> None:
    """A checksum corruption fails instead of becoming a warning."""
    source = json.loads(Path("configs/toolchains/v1.lock.json").read_text(encoding="utf-8"))
    corrupted = copy.deepcopy(source)
    corrupted["tools"]["uv"]["assets"]["darwin-arm64"]["sha256"] = "0" * 63
    with pytest.raises(verify.LockVerificationError, match="not a valid lowercase hash"):
        verify._validate_toolchains(corrupted)


def test_dataset_validator_rejects_checksum_drift() -> None:
    """Official dataset MD5 values are immutable evidence."""
    source = verify._load_yaml(Path("configs/data/openlane-v2-v2.1.lock.yaml"))
    corrupted = copy.deepcopy(source)
    corrupted["archives"][0]["published_md5"] = "0" * 32
    with pytest.raises(verify.LockVerificationError, match="differs from the pinned official"):
        verify._validate_dataset(corrupted)


def test_adapter_lock_rejects_seeded_config_hash_drift(tmp_path: Path) -> None:
    """A changed preprocessing contract cannot retain an old reproducibility identity."""
    source_root = Path.cwd()
    config_relative = Path("configs/data/openlane-v2-v2.1.adapter.yaml")
    copied_config = tmp_path / config_relative
    copied_config.parent.mkdir(parents=True)
    copied_config.write_bytes((source_root / config_relative).read_bytes() + b"# drift\n")
    dataset = verify._load_yaml(source_root / "configs/data/openlane-v2-v2.1.lock.yaml")
    with pytest.raises(verify.LockVerificationError, match="config differs"):
        verify._validate_adapter_lock(tmp_path, dataset)


def test_split_policy_lock_rejects_seeded_config_hash_drift(tmp_path: Path) -> None:
    """A changed split algorithm cannot retain the V1 reproducibility identity."""
    source_root = Path.cwd()
    policy_relative = Path("configs/data/openlane-v2-v2.1.split-v1.yaml")
    copied_policy = tmp_path / policy_relative
    copied_policy.parent.mkdir(parents=True)
    copied_policy.write_bytes((source_root / policy_relative).read_bytes() + b"# drift\n")
    dataset = verify._load_yaml(source_root / "configs/data/openlane-v2-v2.1.lock.yaml")
    with pytest.raises(verify.LockVerificationError, match="split policy differs"):
        verify._validate_split_policy_lock(tmp_path, dataset)


def test_audit_policy_lock_rejects_seeded_slice_registry_drift(tmp_path: Path) -> None:
    """Changed slice semantics cannot retain the V1 audit reproducibility identity."""
    source_root = Path.cwd()
    audit_relative = Path("configs/data/openlane-v2-v2.1.audit-v1.yaml")
    slices_relative = Path("configs/slices/v1.yaml")
    for relative in (audit_relative, slices_relative):
        copied = tmp_path / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes((source_root / relative).read_bytes())
    (tmp_path / slices_relative).write_bytes(
        (source_root / slices_relative).read_bytes() + b"# drift\n"
    )
    dataset = verify._load_yaml(source_root / "configs/data/openlane-v2-v2.1.lock.yaml")
    with pytest.raises(verify.LockVerificationError, match="slice registry differs"):
        verify._validate_audit_policy_lock(tmp_path, dataset)

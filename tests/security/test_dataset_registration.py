"""License and dataset checksum registration security tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from junctionlens.data.license import (
    DatasetRegistrationError,
    acknowledge_licenses,
    load_valid_acknowledgment,
    register_dataset,
)


def _test_lock(tmp_path: Path, archive_md5: str) -> Path:
    source = yaml.safe_load(
        Path("configs/data/openlane-v2-v2.1.lock.yaml").read_text(encoding="utf-8")
    )
    source["archives"][0]["published_md5"] = archive_md5
    path = tmp_path / "dataset.lock.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path


def _acknowledge(lock_path: Path, repository_root: Path) -> None:
    acknowledge_licenses(
        lock_path,
        repository_root,
        ["CC-BY-NC-SA-4.0", "nuScenes-terms", "Argoverse-2-terms"],
        confirmed_restricted_noncommercial_use=True,
    )


def test_acknowledgment_requires_every_exact_term(tmp_path: Path) -> None:
    """An incomplete click-through cannot become a valid license receipt."""
    lock_path = _test_lock(tmp_path, "0" * 32)
    with pytest.raises(DatasetRegistrationError, match="missing"):
        acknowledge_licenses(
            lock_path,
            tmp_path,
            ["CC-BY-NC-SA-4.0"],
            confirmed_restricted_noncommercial_use=True,
        )


def test_receipt_is_private_and_contract_bound(tmp_path: Path) -> None:
    """The ignored local receipt is owner-only and invalidates on contract drift."""
    lock_path = _test_lock(tmp_path, "0" * 32)
    _acknowledge(lock_path, tmp_path)
    receipt = tmp_path / ".junctionlens/license-acknowledgments/openlane-v2-v2.1.json"
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert load_valid_acknowledgment(lock_path, tmp_path)["redistribution_allowed"] is False
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    lock["dataset_licenses"].append("new-term")
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    with pytest.raises(DatasetRegistrationError, match="does not match"):
        load_valid_acknowledgment(lock_path, tmp_path)


def test_registration_verifies_archive_and_manifest(tmp_path: Path) -> None:
    """Registration records SHA-256 only after the published MD5 contract passes."""
    archive = tmp_path / "OpenLane-V2_sample.tar"
    archive.write_bytes(b"repository-owned-test-archive")
    expected_md5 = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    lock_path = _test_lock(tmp_path, expected_md5)
    _acknowledge(lock_path, tmp_path)
    dataset_root = tmp_path / "OpenLane-V2"
    dataset_root.mkdir()
    (dataset_root / "data_dict_example.json").write_text("{}\n", encoding="utf-8")
    result = register_dataset(
        lock_path,
        tmp_path,
        dataset_root,
        profile="sample",
        archive_path=archive,
    )
    assert result["archive_md5"] == expected_md5
    assert result["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_registration_rejects_seeded_checksum_corruption(tmp_path: Path) -> None:
    """A plausible archive with the wrong bytes is a hard failure."""
    archive = tmp_path / "OpenLane-V2_sample.tar"
    archive.write_bytes(b"wrong")
    lock_path = _test_lock(tmp_path, "0" * 32)
    _acknowledge(lock_path, tmp_path)
    dataset_root = tmp_path / "OpenLane-V2"
    dataset_root.mkdir()
    (dataset_root / "data_dict_example.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(DatasetRegistrationError, match="checksum mismatch"):
        register_dataset(
            lock_path,
            tmp_path,
            dataset_root,
            profile="sample",
            archive_path=archive,
        )


def test_acknowledgment_rejects_an_escaping_state_symlink(tmp_path: Path) -> None:
    """A repository-local receipt path cannot traverse an attacker-controlled symlink."""
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / ".junctionlens").symlink_to(outside, target_is_directory=True)
    lock_path = _test_lock(tmp_path, "0" * 32)
    with pytest.raises(DatasetRegistrationError, match="cannot traverse a symlink"):
        _acknowledge(lock_path, repository)

"""Security and reproducibility tests for remote source synchronization."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import tarfile
from pathlib import Path

import pytest
from scripts.gpu.source_bundle import (
    SourceBundleError,
    _strict_json_object,
    create_bundle,
    create_remote_config,
    verify_and_extract,
)


def test_tracked_source_bundle_round_trips_and_is_reproducible(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    first_archive = tmp_path / "first.tar"
    first_manifest = tmp_path / "first.json"
    second_archive = tmp_path / "second.tar"
    second_manifest = tmp_path / "second.json"
    first = create_bundle(root, first_archive, first_manifest, require_clean=False)
    second = create_bundle(root, second_archive, second_manifest, require_clean=False)
    assert first["content_sha256"] == second["content_sha256"]
    assert first["archive_sha256"] == second["archive_sha256"]
    target = tmp_path / "checkout"
    verified = verify_and_extract(first_archive, first_manifest, target)
    assert verified["git_commit"] == first["git_commit"]
    assert (target / "BUILD_PLAN.md").is_file()
    assert (target / "configs/runtime/qualification-v1.yaml").is_file()
    for relative in (
        "scripts/gpu/benchmark_runtime.py",
        "scripts/gpu/profile_runtime.py",
    ):
        extracted = target / relative
        assert extracted.is_file()
        assert extracted.stat().st_mode & stat.S_IXUSR
    assert not (target / ".git").exists()


def test_source_bundle_rejects_traversal_even_with_matching_transport_hash(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "malicious.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("../escape")
        payload = b"forbidden"
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))
    manifest = tmp_path / "manifest.json"
    base = {
        "schema_version": "junctionlens.source-bundle.v1",
        "git_commit": "0" * 40,
        "entries": [
            {
                "path": "declared",
                "mode": "100644",
                "type": "file",
                "byte_size": 9,
                "sha256": "0" * 64,
            }
        ],
        "submodules": [],
        "dependency_lock_sha256": {},
    }
    canonical = json.dumps(
        base, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    base["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    base["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(SourceBundleError, match="undeclared|unsafe"):
        verify_and_extract(archive, manifest, tmp_path / "target")


def test_source_bundle_refuses_clobber(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "manifest.json"
    create_bundle(root, archive, manifest, require_clean=False)
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(SourceBundleError, match="already exists"):
        verify_and_extract(archive, manifest, target)


def _license_acknowledgment(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_id": "openlane-v2-v2.1",
                "license_contract_sha256": "a" * 64,
                "accepted_terms": [
                    "Argoverse-2-terms",
                    "CC-BY-NC-SA-4.0",
                    "nuScenes-terms",
                ],
                "confirmed_restricted_noncommercial_use": True,
                "redistribution_allowed": False,
                "acknowledged_at": "2026-08-14T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


def _visual_audit_signoff(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.visual-audit-signoff.v1",
                "dataset_id": "openlane-v2-v2.1",
                "policy_id": "openlane-v2-v2.1-audit-v1",
                "bundle_manifest_sha256": "b" * 64,
                "reviewed_file_count": 24,
                "assertions": {
                    "camera_projection_alignment_accepted": True,
                    "bev_geometry_alignment_accepted": True,
                    "label_identity_and_topology_accepted": True,
                    "private_data_handling_confirmed": True,
                },
                "reviewed_at": "2026-08-14T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_core_config_carries_only_validated_license_acknowledgment(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "source.json"
    create_bundle(root, archive, manifest, require_clean=False)
    receipt = _license_acknowledgment(tmp_path / "receipt.json")

    config = create_remote_config(
        manifest,
        tmp_path / "config.json",
        profile="core",
        remote_data_root="/licensed/data",
        gpu_uuid=None,
        license_acknowledgment_path=receipt,
    )

    assert config["license_acknowledgment"]["accepted_terms"] == [
        "Argoverse-2-terms",
        "CC-BY-NC-SA-4.0",
        "nuScenes-terms",
    ]
    assert "root" not in config["license_acknowledgment"]
    assert len(config["qualification_sha256"]) == 64


def test_core_config_carries_hash_bound_visual_audit_signoff(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "source.json"
    create_bundle(root, archive, manifest, require_clean=False)

    config = create_remote_config(
        manifest,
        tmp_path / "config.json",
        profile="core",
        remote_data_root="/licensed/data",
        gpu_uuid=None,
        license_acknowledgment_path=_license_acknowledgment(tmp_path / "license.json"),
        visual_audit_signoff_path=_visual_audit_signoff(tmp_path / "visual.json"),
    )

    assert config["visual_audit_signoff"]["bundle_manifest_sha256"] == "b" * 64
    assert all(config["visual_audit_signoff"]["assertions"].values())


def test_core_config_rejects_false_visual_audit_assertion(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "source.json"
    create_bundle(root, archive, manifest, require_clean=False)
    signoff = _visual_audit_signoff(tmp_path / "visual.json")
    value = json.loads(signoff.read_text(encoding="utf-8"))
    value["assertions"]["camera_projection_alignment_accepted"] = False
    signoff.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(SourceBundleError, match="signoff contract is invalid"):
        create_remote_config(
            manifest,
            tmp_path / "config.json",
            profile="core",
            remote_data_root="/licensed/data",
            gpu_uuid=None,
            license_acknowledgment_path=_license_acknowledgment(tmp_path / "license.json"),
            visual_audit_signoff_path=signoff,
        )


def test_remote_run_identity_changes_with_profile_and_machine_inputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "source.json"
    create_bundle(root, archive, manifest, require_clean=False)
    cuda = create_remote_config(
        manifest,
        tmp_path / "cuda.json",
        profile="runtime-cuda",
        remote_data_root=None,
        gpu_uuid=None,
    )
    performance = create_remote_config(
        manifest,
        tmp_path / "performance.json",
        profile="runtime-performance",
        remote_data_root=None,
        gpu_uuid="GPU-1234",
    )

    assert cuda["source_content_sha256"] == performance["source_content_sha256"]
    assert cuda["qualification_sha256"] != performance["qualification_sha256"]


def test_core_config_rejects_missing_or_tampered_acknowledgment(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "source.json"
    create_bundle(root, archive, manifest, require_clean=False)
    with pytest.raises(SourceBundleError, match="requires explicit"):
        create_remote_config(
            manifest,
            tmp_path / "missing.json",
            profile="core",
            remote_data_root="/licensed/data",
            gpu_uuid=None,
        )
    receipt = _license_acknowledgment(tmp_path / "receipt.json")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["private_path"] = "/do/not/transfer"
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SourceBundleError, match="contract is invalid"):
        create_remote_config(
            manifest,
            tmp_path / "tampered.json",
            profile="core",
            remote_data_root="/licensed/data",
            gpu_uuid=None,
            license_acknowledgment_path=receipt,
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"one","schema_version":"two"}',
        ('{"schema_version":' + "[" * 20 + "0" + "]" * 20 + "}").encode(),
    ],
)
def test_source_manifest_rejects_adversarial_json_shape(payload: bytes) -> None:
    with pytest.raises(SourceBundleError):
        _strict_json_object(payload)

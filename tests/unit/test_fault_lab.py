"""Seeded detection matrix for every mandatory V1 fault and control."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from junctionlens.faults.analysis import analyze_fault, verify_clean_bundle
from junctionlens.faults.models import FAULT_REASON_CODES, FaultKind
from junctionlens.faults.runtime import run_allocator_fixture
from junctionlens.faults.service import inject_fault, put_prediction_bundle
from junctionlens.faults.synthetic import build_synthetic_fault_bundle
from junctionlens.faults.transforms import apply_fault
from junctionlens.registry.service import EvidenceRegistry

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "schemas/artifact-manifest-v1.schema.json"


def _payload(registry: EvidenceRegistry, manifest_sha256: str) -> dict[str, object]:
    manifest = registry.store.read_manifest(manifest_sha256)
    path = registry.store.object_path(manifest["payload"]["sha256"])
    return json.loads(path.read_bytes())


def test_nearby_clean_synthetic_control_passes() -> None:
    result = verify_clean_bundle(build_synthetic_fault_bundle())

    assert result["status"] == "PASS"
    assert all(result["checks"].values())


@pytest.mark.parametrize("kind", list(FaultKind))
@pytest.mark.parametrize("seed", [20260813, 20260814, 20260815])
def test_every_mandatory_fault_is_detected_for_every_seed(kind: FaultKind, seed: int) -> None:
    parent = build_synthetic_fault_bundle()
    child = apply_fault(parent, kind, seed=seed, fraction=0.5)

    result = analyze_fault(parent, child)

    expected_status = "CONTROL_PASSED" if kind == FaultKind.PERMUTE_NODES_CORRECTLY else "DETECTED"
    assert result["status"] == expected_status
    assert result["primary_reason_code"] == FAULT_REASON_CODES[kind]
    assert result["detection_rate"] == 1.0
    assert all(result["checks"].values())


def test_fault_service_preserves_parent_links_and_flagship_control_evidence(
    tmp_path: Path,
) -> None:
    registry = EvidenceRegistry(tmp_path / "artifacts", SCHEMA)
    parent_hash = put_prediction_bundle(registry, build_synthetic_fault_bundle())

    receipt = inject_fault(
        artifact_root=tmp_path / "artifacts",
        schema_path=SCHEMA,
        input_manifest_sha256=parent_hash,
        kind=FaultKind.SWAP_CONTROL_EDGES,
    )

    derived = registry.store.read_manifest(receipt.derived_manifest_sha256)
    report_manifest = registry.store.read_manifest(receipt.counterexample_manifest_sha256)
    report = _payload(registry, receipt.counterexample_manifest_sha256)
    assert derived["parents"] == [parent_hash]
    assert report_manifest["parents"] == sorted([parent_hash, receipt.derived_manifest_sha256])
    assert report["details"]["DET_l_delta"] == 0.0
    assert report["details"]["DET_t_delta"] == 0.0
    assert report["details"]["node_geometry_unchanged"] is True
    assert report["details"]["control_edge_recall"] == 0.0
    assert report["details"]["wrong_control_assignment_rate"] == 1.0
    assert report["details"]["lane_control_fault_cell"] == {
        "cell_id": "overall.control_edge_recall",
        "status": "FAIL_REGRESSION",
        "reason_code": "FAULT_CONTROL_ASSIGNMENT_CHANGED",
        "v1_release_acceptance_run": False,
    }


def test_fault_derivation_is_byte_identical_for_same_seed(tmp_path: Path) -> None:
    registry = EvidenceRegistry(tmp_path / "artifacts", SCHEMA)
    parent_hash = put_prediction_bundle(registry, build_synthetic_fault_bundle())
    arguments = {
        "artifact_root": tmp_path / "artifacts",
        "schema_path": SCHEMA,
        "input_manifest_sha256": parent_hash,
        "kind": FaultKind.JITTER_LANES,
        "seed": 20260813,
    }

    first = inject_fault(**arguments)
    second = inject_fault(**arguments)

    assert first == second


def test_control_edge_drop_severity_is_monotonic() -> None:
    parent = build_synthetic_fault_bundle()
    recalls = []
    for fraction in (0.25, 0.5, 1.0):
        child = apply_fault(
            parent,
            FaultKind.DROP_CONTROL_EDGES,
            seed=20260813,
            fraction=fraction,
        )
        report = analyze_fault(parent, child)
        assert report["checks"]["control_edges_strictly_dropped"] is True
        recalls.append(report["details"]["control_edge_recall"])

    assert recalls == sorted(recalls, reverse=True)
    assert recalls[0] > recalls[-1]


def test_allocator_fixture_distinguishes_release_from_seeded_leak() -> None:
    clean = run_allocator_fixture(frame_count=100, buffer_bytes=64 * 1024, leak=False)
    leaked = run_allocator_fixture(frame_count=100, buffer_bytes=64 * 1024, leak=True)

    assert clean.outstanding_bytes == 0
    assert clean.retained_buffers == 0
    assert leaked.outstanding_bytes == 100 * 64 * 1024
    assert leaked.retained_buffers == 100
    assert leaked.high_water_bytes > clean.high_water_bytes

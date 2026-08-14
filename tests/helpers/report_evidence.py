"""Small immutable comparison registry used by report tests."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import canonical_json_bytes

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "schemas/artifact-manifest-v1.schema.json"


@dataclass(frozen=True, slots=True)
class ReportEvidence:
    artifact_root: Path
    comparison_manifest_sha256: str
    decision_manifest_sha256: str
    metrics_manifest_sha256: str
    slices_manifest_sha256: str
    scene_manifest_sha256: str
    image_manifest_sha256: str
    image_bytes: bytes
    decision_bytes: bytes
    metrics_bytes: bytes
    slices_bytes: bytes


def _decision() -> dict[str, Any]:
    cell = {
        "cell_id": 'control-edge-recall/<script>alert("report")</script>|overall',
        "metric": "control_edge_recall",
        "slice": "source_domain:synthetic|night",
        "status": "FAIL_REGRESSION",
        "reason_code": "GATE_REGRESSION_CI_BELOW_MARGIN",
        "support": {
            "paired_segments": 200,
            "eligible_ground_truth_edges": 600,
            "adjacent_frame_transitions": 0,
            "temporal_segments": 0,
        },
        "point_estimate": -0.02,
        "interval": {
            "lower": -0.03,
            "upper": -0.01,
            "adjusted_two_sided_alpha": 0.05,
        },
        "margin": 0.005,
        "finite_replicates": 10000,
        "invalid_replicates": 0,
        "counterexample_query": "delta < -0.01 | order by severity `descending`",
    }
    body: dict[str, Any] = {
        "schema_version": "junctionlens.gate-decision.v1",
        "status": "FAIL_REGRESSION",
        "charter_sha256": "c" * 64,
        "evidence_sha256": "e" * 64,
        "bootstrap": {
            "algorithm": "paired-segment-cluster-bootstrap-v1",
            "replicates": 10000,
            "seed": 20260813,
            "interval_method": "type7-percentile-bonferroni-two-sided",
            "gating_cells": 1,
        },
        "integrity_reason_codes": [],
        "infrastructure_reason_codes": [],
        "performance_reason_codes": [],
        "cells": [cell],
        "primary_hypotheses": [],
    }
    body["decision_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def _parquet_bytes(table: pa.Table) -> bytes:
    output = io.BytesIO()
    pq.write_table(
        table,
        output,
        compression="zstd",
        data_page_version="2.0",
        version="2.6",
        use_dictionary=False,
        write_statistics=True,
    )
    return output.getvalue()


def _png() -> bytes:
    image = Image.new("RGB", (48, 32), (13, 70, 76))
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def create_report_evidence(root: Path) -> ReportEvidence:
    artifact_root = root / "artifacts"
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    decision = _decision()
    decision_bytes = canonical_json_bytes(decision) + b"\n"
    decision_receipt = registry.put_bytes(
        decision_bytes,
        kind="release_decision",
        media_type="application/vnd.junctionlens.gate-decision+json",
        license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
        metadata={"status": decision["status"]},
    )
    metrics_bytes = _parquet_bytes(
        pa.Table.from_pylist(
            [{"cell_id": decision["cells"][0]["cell_id"], "point_estimate": -0.02}],
            schema=pa.schema([("cell_id", pa.string()), ("point_estimate", pa.float64())]),
        )
    )
    metrics_receipt = registry.put_bytes(
        metrics_bytes,
        kind="comparison",
        media_type="application/vnd.apache.parquet",
        license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
        metadata={"row_count": 1},
        parents=(decision_receipt.manifest_sha256,),
    )
    slices_bytes = _parquet_bytes(
        pa.Table.from_pylist(
            [{"frame_token": "synthetic-frame", "source_domain": "synthetic"}],
            schema=pa.schema([("frame_token", pa.string()), ("source_domain", pa.string())]),
        )
    )
    slices_receipt = registry.put_bytes(
        slices_bytes,
        kind="slice_table",
        media_type="application/vnd.apache.parquet",
        license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
        metadata={"row_count": 1},
    )
    report_data = {
        "schema_version": "junctionlens.comparison-report-data.v1",
        "status": decision["status"],
        "baseline_manifest_sha256": "a" * 64,
        "candidate_manifest_sha256": "b" * 64,
        "charter_sha256": decision["charter_sha256"],
        "decision_manifest_sha256": decision_receipt.manifest_sha256,
        "metrics_table_manifest_sha256": metrics_receipt.manifest_sha256,
        "slice_table_manifest_sha256": slices_receipt.manifest_sha256,
        "reason_codes": ["GATE_REGRESSION_CI_BELOW_MARGIN"],
        "cells": decision["cells"],
        "primary_hypotheses": decision["primary_hypotheses"],
        "filtering_changes_release_status": False,
    }
    comparison_receipt = registry.put_bytes(
        canonical_json_bytes(report_data) + b"\n",
        kind="comparison",
        media_type="application/vnd.junctionlens.comparison-report-data+json",
        license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
        metadata={"status": decision["status"]},
        parents=(
            decision_receipt.manifest_sha256,
            metrics_receipt.manifest_sha256,
            slices_receipt.manifest_sha256,
        ),
    )
    image_bytes = _png()
    image_receipt = registry.put_bytes(
        image_bytes,
        kind="evidence_report",
        media_type="image/png",
        license_id="CC-BY-NC-SA-4.0",
        metadata={"source": "licensed-test-thumbnail"},
    )
    graph = {
        "lanes": [
            {
                "node_id": "lane-a",
                "points": [{"x": 0.0, "y": 0.0}, {"x": 4.0, "y": 1.0}],
                "confidence": 0.9,
            }
        ],
        "controls": [],
        "edges": [],
    }
    scene = {
        "schema_version": "junctionlens.scene-bundle.v1",
        "title": "Licensed regression scene",
        "decision_manifest_sha256": decision_receipt.manifest_sha256,
        "license_notice": "Test fixture representing a restricted source image.",
        "frames": [
            {
                "frame_id": "frame-00",
                "segment_id": "segment-00",
                "timestamp_ns": "1725000000000000000",
                "cameras": [
                    {
                        "slot": "front-center",
                        "label": "Front <center>",
                        "artifact_manifest_sha256": image_receipt.manifest_sha256,
                        "restriction_reason": None,
                    }
                ],
                "ground_truth": graph,
                "baseline": graph,
                "candidate": graph,
            }
        ],
    }
    scene_receipt = registry.put_bytes(
        canonical_json_bytes(scene) + b"\n",
        kind="counterexample_bundle",
        media_type="application/vnd.junctionlens.scene-bundle+json",
        license_id="CC-BY-NC-SA-4.0",
        metadata={"frame_count": 1},
        parents=(decision_receipt.manifest_sha256, image_receipt.manifest_sha256),
    )
    return ReportEvidence(
        artifact_root=artifact_root,
        comparison_manifest_sha256=comparison_receipt.manifest_sha256,
        decision_manifest_sha256=decision_receipt.manifest_sha256,
        metrics_manifest_sha256=metrics_receipt.manifest_sha256,
        slices_manifest_sha256=slices_receipt.manifest_sha256,
        scene_manifest_sha256=scene_receipt.manifest_sha256,
        image_manifest_sha256=image_receipt.manifest_sha256,
        image_bytes=image_bytes,
        decision_bytes=decision_bytes,
        metrics_bytes=metrics_bytes,
        slices_bytes=slices_bytes,
    )


__all__ = ["ReportEvidence", "create_report_evidence"]

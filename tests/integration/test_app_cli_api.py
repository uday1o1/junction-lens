"""End-to-end CLI and API coverage for the local evidence product."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from junctionlens.api import ServiceConfig, create_app
from junctionlens.cli.main import app
from junctionlens.registry.service import EvidenceRegistry, RunIdentity
from junctionlens.registry.store import ArtifactReceipt, canonical_json_bytes

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "schemas/artifact-manifest-v1.schema.json"


def _decision() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "junctionlens.gate-decision.v1",
        "status": "FAIL_REGRESSION",
        "cells": [
            {
                "cell_id": "control-edge-recall/overall",
                "status": "FAIL_REGRESSION",
                "reason_code": "GATE_REGRESSION_CI_BELOW_MARGIN",
            }
        ],
        "integrity_reason_codes": [],
        "infrastructure_reason_codes": [],
        "performance_reason_codes": [],
    }
    body["decision_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def _registered_evidence(
    tmp_path: Path,
) -> tuple[Path, ArtifactReceipt, ArtifactReceipt, ArtifactReceipt, dict[str, Any]]:
    artifact_root = tmp_path / "artifacts"
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    decision = _decision()
    decision_receipt = registry.put_bytes(
        canonical_json_bytes(decision) + b"\n",
        kind="release_decision",
        media_type="application/vnd.junctionlens.gate-decision+json",
        license_id="Apache-2.0",
        metadata={"status": decision["status"]},
    )
    metric_path = tmp_path / "metrics.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"segment_id": "segment-a", "delta": 0.25},
                {"segment_id": "segment-b", "delta": -0.5},
            ]
        ),
        metric_path,
    )
    metric_receipt = registry.put_file(
        metric_path,
        kind="comparison",
        media_type="application/vnd.apache.parquet",
        license_id="Apache-2.0",
        metadata={"row_count": 2},
        parents=(decision_receipt.manifest_sha256,),
    )
    image_buffer = BytesIO()
    Image.new("RGB", (32, 24), color=(20, 30, 40)).save(image_buffer, format="PNG")
    image_receipt = registry.put_bytes(
        image_buffer.getvalue(),
        kind="evidence_report",
        media_type="image/png",
        license_id="CC0-1.0",
        metadata={"redacted": True},
    )
    identity = RunIdentity(
        schema_version="junctionlens.run-identity.v1",
        run_kind="synthetic-evaluation",
        parent_artifact_hashes=(metric_receipt.manifest_sha256,),
        dataset_manifest_sha256="1" * 64,
        split_manifest_sha256="2" * 64,
        model_profile_sha256="3" * 64,
        configuration_sha256="4" * 64,
        source_git_commit="5" * 40,
        source_dirty=False,
        dependency_lock_hashes={"uv.lock": "6" * 64},
        container_image_digests={"evaluator": "7" * 64},
        seed=20260813,
        command_schema_version="junctionlens.evaluate.v1",
        execution_provider_profile="cpu-reference",
    )
    registry.begin_or_resume_run(identity, "8" * 64)
    return artifact_root, decision_receipt, metric_receipt, image_receipt, decision


def test_read_only_api_serves_paginated_registered_evidence(tmp_path: Path) -> None:
    artifact_root, decision_receipt, metric_receipt, image_receipt, decision = _registered_evidence(
        tmp_path
    )
    client = TestClient(
        create_app(ServiceConfig(artifact_root=artifact_root, schema_path=SCHEMA)),
        base_url="http://127.0.0.1",
    )

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["state"] == "READY"
    assert health.headers["x-content-type-options"] == "nosniff"
    policy = health.headers["content-security-policy"]
    assert "default-src 'self'" in policy
    assert "script-src 'self'" in policy
    assert "object-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert health.headers["permissions-policy"].startswith("camera=()")
    assert health.headers["cross-origin-resource-policy"] == "same-origin"

    first_page = client.get("/api/v1/artifacts", params={"limit": 2})
    second_page = client.get("/api/v1/artifacts", params={"limit": 2, "offset": 2})
    assert first_page.status_code == second_page.status_code == 200
    assert first_page.json()["page"]["returned"] == 2
    assert first_page.json()["page"]["total"] >= 4
    assert {item["manifest_sha256"] for item in first_page.json()["items"]}.isdisjoint(
        item["manifest_sha256"] for item in second_page.json()["items"]
    )

    runs = client.get("/api/v1/runs")
    assert runs.status_code == 200
    assert runs.json()["items"][0]["run_kind"] == "synthetic-evaluation"
    assert runs.json()["items"][0]["execution_provider_profile"] == "cpu-reference"

    persisted = client.get(f"/api/v1/decisions/{decision_receipt.manifest_sha256}")
    assert persisted.status_code == 200
    assert persisted.json()["decision"] == decision

    metrics = client.get(
        f"/api/v1/metrics/{metric_receipt.manifest_sha256}",
        params={"offset": 1, "limit": 1},
    )
    assert metrics.status_code == 200
    assert metrics.json()["columns"] == ["segment_id", "delta"]
    assert metrics.json()["rows"] == [{"segment_id": "segment-b", "delta": -0.5}]
    assert metrics.json()["page"]["total"] == 2

    image = client.get(f"/api/v1/images/{image_receipt.manifest_sha256}")
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG")
    assert image.headers["x-junctionlens-license"] == "CC0-1.0"


def test_image_route_rejects_declared_png_with_malformed_bytes(tmp_path: Path) -> None:
    artifact_root, _, _, _, _ = _registered_evidence(tmp_path)
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    malformed = registry.put_bytes(
        b"\x89PNG\r\n\x1a\nnot-a-raster",
        kind="evidence_report",
        media_type="image/png",
        license_id="CC0-1.0",
        metadata={"seeded_defect": "malformed-image"},
    )
    client = TestClient(
        create_app(ServiceConfig(artifact_root=artifact_root, schema_path=SCHEMA)),
        base_url="http://127.0.0.1",
    )

    response = client.get(f"/api/v1/images/{malformed.manifest_sha256}")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "API_REGISTRY_INVALID"
    assert "malformed" in response.json()["error"]["message"]


def test_metric_route_rejects_malformed_parquet_and_excess_rows(tmp_path: Path) -> None:
    artifact_root, _, metric_receipt, _, _ = _registered_evidence(tmp_path)
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    malformed = registry.put_bytes(
        b"not parquet",
        kind="comparison",
        media_type="application/vnd.apache.parquet",
        license_id="Apache-2.0",
        metadata={"seeded_defect": "malformed-parquet"},
    )
    client = TestClient(
        create_app(
            ServiceConfig(
                artifact_root=artifact_root,
                schema_path=SCHEMA,
                max_metric_rows=1,
            )
        ),
        base_url="http://127.0.0.1",
    )

    malformed_response = client.get(f"/api/v1/metrics/{malformed.manifest_sha256}")
    oversized_response = client.get(f"/api/v1/metrics/{metric_receipt.manifest_sha256}")

    assert malformed_response.status_code == 409
    assert malformed_response.json()["error"]["code"] == "API_REGISTRY_INVALID"
    assert oversized_response.status_code == 409
    assert "row limit" in oversized_response.json()["error"]["message"]


def test_api_errors_are_stable_and_all_routes_are_read_only(tmp_path: Path) -> None:
    artifact_root, _, _, _, _ = _registered_evidence(tmp_path)
    client = TestClient(
        create_app(ServiceConfig(artifact_root=artifact_root, schema_path=SCHEMA)),
        base_url="http://127.0.0.1",
    )

    invalid = client.get("/api/v1/artifacts/not-a-hash")
    missing = client.get(f"/api/v1/artifacts/{'f' * 64}")
    mutation = client.post("/api/v1/health")
    traversal = client.get("/api/v1/artifacts", params={"offset": -1})

    assert invalid.status_code == traversal.status_code == 422
    assert (
        invalid.json()
        == traversal.json()
        == {
            "schema_version": "junctionlens.api-error.v1",
            "error": {
                "code": "API_REQUEST_INVALID",
                "message": "request did not satisfy the API contract",
            },
        }
    )
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "API_ARTIFACT_NOT_FOUND"
    assert mutation.status_code == 405
    assert mutation.json()["error"]["code"] == "API_METHOD_NOT_ALLOWED"


def _scene_graph(prefix: str) -> dict[str, Any]:
    return {
        "lanes": [
            {
                "node_id": f"{prefix}-lane",
                "points": [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 1.0}],
                "confidence": 0.9,
            }
        ],
        "controls": [
            {
                "node_id": f"{prefix}-control",
                "x": 5.0,
                "y": -2.0,
                "control_type": "traffic_light",
                "state": "red",
                "confidence": 0.8,
            }
        ],
        "edges": [
            {
                "edge_id": f"{prefix}-edge",
                "edge_type": "control_applies_to_lane",
                "source_node_id": f"{prefix}-control",
                "target_node_id": f"{prefix}-lane",
                "confidence": 0.75,
            }
        ],
    }


def test_scene_api_requires_persisted_parent_and_serves_bundled_web_app(
    tmp_path: Path,
) -> None:
    artifact_root, decision_receipt, _, image_receipt, decision = _registered_evidence(tmp_path)
    registry = EvidenceRegistry(artifact_root, SCHEMA)
    bundle = {
        "schema_version": "junctionlens.scene-bundle.v1",
        "title": "Synthetic scene",
        "decision_manifest_sha256": decision_receipt.manifest_sha256,
        "license_notice": "Synthetic unrestricted cameras",
        "frames": [
            {
                "frame_id": "frame-00",
                "segment_id": "segment-a",
                "timestamp_ns": "1725000000000000000",
                "cameras": [
                    {
                        "slot": "front-center",
                        "label": "Front center",
                        "artifact_manifest_sha256": image_receipt.manifest_sha256,
                        "restriction_reason": None,
                    },
                    {
                        "slot": "front-left",
                        "label": "Front left",
                        "artifact_manifest_sha256": None,
                        "restriction_reason": "Licensed image excluded.",
                    },
                ],
                "ground_truth": _scene_graph("truth"),
                "baseline": _scene_graph("baseline"),
                "candidate": _scene_graph("candidate"),
            }
        ],
    }
    scene_receipt = registry.put_bytes(
        canonical_json_bytes(bundle) + b"\n",
        kind="counterexample_bundle",
        media_type="application/vnd.junctionlens.scene-bundle+json",
        license_id="CC0-1.0",
        metadata={"frame_count": 1},
        parents=(decision_receipt.manifest_sha256, image_receipt.manifest_sha256),
    )
    invalid_receipt = registry.put_bytes(
        canonical_json_bytes(bundle) + b" ",
        kind="counterexample_bundle",
        media_type="application/vnd.junctionlens.scene-bundle+json",
        license_id="CC0-1.0",
        metadata={"frame_count": 1, "invalid_parent_fixture": True},
        parents=(image_receipt.manifest_sha256,),
    )
    web_root = tmp_path / "web-dist"
    (web_root / "assets").mkdir(parents=True)
    (web_root / "index.html").write_text("<!doctype html><title>Viewer</title>", encoding="utf-8")
    client = TestClient(
        create_app(
            ServiceConfig(
                artifact_root=artifact_root,
                schema_path=SCHEMA,
                web_root=web_root,
            )
        ),
        base_url="http://127.0.0.1",
    )

    scene = client.get(f"/api/v1/scenes/{scene_receipt.manifest_sha256}")
    invalid = client.get(f"/api/v1/scenes/{invalid_receipt.manifest_sha256}")
    index = client.get("/")
    fallback = client.get("/scene/selected")

    assert scene.status_code == 200
    assert scene.json()["bundle"] == bundle
    assert scene.json()["decision"] == decision
    assert invalid.status_code == 409
    assert "immutable parent" in invalid.json()["error"]["message"]
    assert index.status_code == fallback.status_code == 200
    assert index.text == fallback.text
    assert index.headers["content-security-policy"].find("script-src 'self'") >= 0


def test_public_cli_report_and_service_paths_are_end_to_end(tmp_path: Path) -> None:
    artifact_root, decision_receipt, _, _, _ = _registered_evidence(tmp_path)
    runner = CliRunner()

    report = runner.invoke(
        app,
        [
            "report",
            "--decision",
            decision_receipt.manifest_sha256,
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
        ],
    )
    assert report.exit_code == 0, report.output
    report_receipt = json.loads(report.stdout)
    assert len(report_receipt["manifest_sha256"]) == 64
    assert report_receipt["immutable_path"].endswith(report_receipt["payload_sha256"][2:])
    stored = EvidenceRegistry(artifact_root, SCHEMA).index.read_artifact(
        report_receipt["manifest_sha256"]
    )
    assert stored["kind"] == "evidence_report"
    human_report = runner.invoke(
        app,
        [
            "--human",
            "report",
            "--decision",
            decision_receipt.manifest_sha256,
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
        ],
    )
    assert human_report.exit_code == 0, human_report.output
    assert human_report.stdout.startswith("report ")

    ready = runner.invoke(
        app,
        [
            "serve",
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
            "--api-only",
            "--check",
        ],
    )
    assert ready.exit_code == 0, ready.output
    assert json.loads(ready.stdout)["state"] == "READY"

    non_loopback = runner.invoke(
        app,
        [
            "serve",
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
            "--api-only",
            "--host",
            "0.0.0.0",  # noqa: S104 - seeded rejection case
            "--check",
        ],
    )
    assert non_loopback.exit_code == 2
    assert json.loads(non_loopback.stderr)["error"]["code"] == "SERVE_CONFIGURATION_INVALID"


def test_every_documented_public_command_has_help() -> None:
    runner = CliRunner()
    commands = (
        ("doctor",),
        ("data", "register"),
        ("data", "audit"),
        ("data", "split"),
        ("train",),
        ("export",),
        ("infer",),
        ("evaluate",),
        ("calibrate",),
        ("compare",),
        ("gate", "freeze"),
        ("gate", "decide"),
        ("fault",),
        ("report",),
        ("serve",),
    )

    for command in commands:
        result = runner.invoke(app, [*command, "--help"])
        assert result.exit_code == 0, f"{' '.join(command)}: {result.output}"


def test_service_cli_redacts_diagnostic_roots_and_credentials(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    artifact_root, _, _, _, _ = _registered_evidence(tmp_path)
    secret = "access_token=" + "seeded-private-value"

    def fail_check(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(f"root={artifact_root} {secret}")

    monkeypatch.setattr("junctionlens.api.server.check_service", fail_check)
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
            "--api-only",
            "--check",
        ],
    )

    assert result.exit_code == 2
    assert str(artifact_root) not in result.stderr
    assert "seeded-private-value" not in result.stderr
    assert "[LOCAL_ROOT]" in result.stderr
    assert "[REDACTED]" in result.stderr

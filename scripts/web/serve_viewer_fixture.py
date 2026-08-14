#!/usr/bin/env python3
"""Serve a repository-owned unrestricted scene through the production app."""

from __future__ import annotations

import argparse
import hashlib
import io
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
from PIL import Image, ImageDraw

from junctionlens.api import ServiceConfig, create_app
from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import canonical_json_bytes

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


def _image(label: str, frame_index: int, color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (640, 360), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((34, 34, 606, 326), outline=(220, 236, 238), width=3)
    draw.line((160, 326, 290, 170, 350, 170, 480, 326), fill=(246, 200, 95), width=7)
    draw.text((54, 55), f"{label} / frame {frame_index:02d}", fill=(245, 250, 251))
    draw.text((54, 86), "repository-owned synthetic view", fill=(184, 205, 209))
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _graph(offset: float, *, omit_control_edge: bool = False) -> dict[str, Any]:
    lanes = [
        {
            "node_id": f"lane-main-{offset}",
            "confidence": 0.93,
            "points": [
                {"x": 2.0, "y": offset},
                {"x": 12.0, "y": offset + 0.6},
                {"x": 24.0, "y": offset + 1.2},
            ],
        },
        {
            "node_id": f"lane-next-{offset}",
            "confidence": 0.88,
            "points": [
                {"x": 24.0, "y": offset + 1.2},
                {"x": 35.0, "y": offset + 4.0},
            ],
        },
    ]
    controls = [
        {
            "node_id": f"control-{offset}",
            "x": 18.0,
            "y": offset - 5.0,
            "control_type": "traffic_light",
            "state": "red",
            "confidence": 0.95,
        }
    ]
    edges = [
        {
            "edge_id": f"successor-{offset}",
            "edge_type": "lane_successor",
            "source_node_id": f"lane-main-{offset}",
            "target_node_id": f"lane-next-{offset}",
            "confidence": 0.9,
        }
    ]
    if not omit_control_edge:
        edges.append(
            {
                "edge_id": f"control-edge-{offset}",
                "edge_type": "control_applies_to_lane",
                "source_node_id": f"control-{offset}",
                "target_node_id": f"lane-main-{offset}",
                "confidence": 0.92,
            }
        )
    return {"lanes": lanes, "controls": controls, "edges": edges}


def _bundle(
    decision_manifest: str,
    image_manifests: list[tuple[str, str]],
) -> dict[str, Any]:
    frames = []
    for index, (front_center, front_right) in enumerate(image_manifests):
        frames.append(
            {
                "frame_id": f"frame-{index:02d}",
                "segment_id": "synthetic-intersection",
                "timestamp_ns": str(1_725_000_000_000_000_000 + index * 100_000_000),
                "cameras": [
                    {
                        "slot": "front-center",
                        "label": "Front center",
                        "artifact_manifest_sha256": front_center,
                        "restriction_reason": None,
                    },
                    {
                        "slot": "front-right",
                        "label": "Front right",
                        "artifact_manifest_sha256": front_right,
                        "restriction_reason": None,
                    },
                    {
                        "slot": "front-left",
                        "label": "Front left",
                        "artifact_manifest_sha256": None,
                        "restriction_reason": "Licensed image excluded from public export.",
                    },
                ],
                "ground_truth": _graph(float(index)),
                "baseline": _graph(float(index + 3)),
                "candidate": _graph(float(index + 1), omit_control_edge=True),
            }
        )
    return {
        "schema_version": "junctionlens.scene-bundle.v1",
        "title": "Synthetic intersection control regression",
        "decision_manifest_sha256": decision_manifest,
        "license_notice": "Synthetic unrestricted cameras",
        "frames": frames,
    }


def _populate(root: Path) -> None:
    registry = EvidenceRegistry(root, SCHEMA)
    decision = _decision()
    decision_receipt = registry.put_bytes(
        canonical_json_bytes(decision) + b"\n",
        kind="release_decision",
        media_type="application/vnd.junctionlens.gate-decision+json",
        license_id="Apache-2.0",
        metadata={"status": decision["status"]},
    )
    images: list[tuple[str, str]] = []
    image_parents: list[str] = []
    for index in range(2):
        frame_images = []
        for label, color in (
            ("front-center", (22 + index * 6, 57, 63)),
            ("front-right", (43, 49 + index * 6, 68)),
        ):
            receipt = registry.put_bytes(
                _image(label, index, color),
                kind="evidence_report",
                media_type="image/png",
                license_id="CC0-1.0",
                metadata={"frame_index": index, "slot": label, "synthetic": True},
            )
            frame_images.append(receipt.manifest_sha256)
            image_parents.append(receipt.manifest_sha256)
        images.append((frame_images[0], frame_images[1]))
    bundle = _bundle(decision_receipt.manifest_sha256, images)
    registry.put_bytes(
        canonical_json_bytes(bundle) + b"\n",
        kind="counterexample_bundle",
        media_type="application/vnd.junctionlens.scene-bundle+json",
        license_id="CC0-1.0",
        metadata={"frame_count": 2, "source": "repository-owned-synthetic"},
        parents=tuple(sorted([decision_receipt.manifest_sha256, *image_parents])),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--web-root", type=Path, default=ROOT / "web" / "dist")
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("port must be between 1 and 65535")
    with tempfile.TemporaryDirectory(prefix="junctionlens-viewer-") as temporary:
        artifact_root = Path(temporary) / "artifacts"
        _populate(artifact_root)
        app = create_app(
            ServiceConfig(
                artifact_root=artifact_root,
                schema_path=SCHEMA,
                web_root=arguments.web_root,
            )
        )
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=arguments.port,
            access_log=False,
            log_level="warning",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

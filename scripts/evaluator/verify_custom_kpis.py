#!/usr/bin/env python3
"""Exercise CustomMatchV1 and KPI persistence through the public CLI twice."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from junctionlens.contract import to_binary
from junctionlens.registry import ContentAddressedStore
from junctionlens.synthetic import generate_scene_frames


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_once(root: Path, truth: Path, predictions: Path, artifacts: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    staging = root / ".cache/docker-staging"
    staging.mkdir(parents=True, exist_ok=True)
    environment["JUNCTIONLENS_DOCKER_STAGING_ROOT"] = str(staging)
    result = subprocess.run(
        [
            str(root / ".venv/bin/junctionlens"),
            "evaluate",
            "--ground-truth",
            str(truth),
            "--predictions",
            str(predictions),
            "--artifact-root",
            str(artifacts),
            "--project-root",
            str(root),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"public custom evaluation failed ({result.returncode}): {result.stderr[:2048]}"
        )
    output = json.loads(result.stdout)
    if output.get("schema_version") != "junctionlens.custom-evaluation-receipt.v1":
        raise RuntimeError("public custom evaluation returned an unexpected receipt")
    return output


def _read_table(root: Path, manifest_sha256: str) -> Any:
    store = ContentAddressedStore(
        root,
        _root() / "schemas/artifact-manifest-v1.schema.json",
    )
    manifest = store.read_manifest(manifest_sha256)
    return pq.read_table(store.object_path(manifest["payload"]["sha256"]))


def _assert_tables(receipt: dict[str, Any], artifact_root: Path) -> None:
    frame_table = _read_table(
        artifact_root,
        receipt["artifacts"]["frame_kpi_table"]["manifest_sha256"],
    )
    segment_table = _read_table(
        artifact_root,
        receipt["artifacts"]["segment_kpi_table"]["manifest_sha256"],
    )
    if frame_table.num_rows != 80:
        raise RuntimeError(f"expected 80 frame KPI rows, observed {frame_table.num_rows}")
    if segment_table.num_rows != 72:
        raise RuntimeError(f"expected 72 segment KPI rows, observed {segment_table.num_rows}")
    frame_metrics = frame_table.to_pylist()
    if not any(
        row["metric"] == "control_edge_recall" and row["value"] == 1.0 for row in frame_metrics
    ):
        raise RuntimeError("perfect control-edge golden is absent from the frame table")
    if not any(
        row["metric"] == "reachability_recall_h3" and row["value"] == 1.0 for row in frame_metrics
    ):
        raise RuntimeError("perfect reachability golden is absent from the frame table")
    if not any(row["status"] == "EMPTY_DENOMINATOR" for row in frame_metrics):
        raise RuntimeError("empty-denominator behavior is absent from the frame table")


def _assert_duplicate_key_rejected(root: Path) -> None:
    fixture = root / "tests/fixtures/evaluator/custom_duplicate_key.json"
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker CLI is unavailable")
    result = subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "1g",
            "--cpus",
            "2",
            "--user",
            "65532:65532",
            "--mount",
            f"type=bind,src={fixture},dst=/input/request.json,readonly",
            "junctionlens/official-evaluator:v2.1.0",
            "--custom-match",
            "/input/request.json",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0 or "duplicate JSON object key" not in result.stderr:
        raise RuntimeError("CustomMatchV1 did not reject the seeded duplicate JSON key")


def main() -> int:
    root = _root()
    _assert_duplicate_key_rejected(root)
    with tempfile.TemporaryDirectory(prefix="junctionlens-custom-kpi-") as temporary:
        workspace = Path(temporary)
        truth_root = workspace / "truth"
        prediction_root = workspace / "predictions"
        truth_root.mkdir()
        prediction_root.mkdir()
        frames = generate_scene_frames()
        for index, frame in enumerate(frames):
            (truth_root / f"{index:02d}.pb").write_bytes(to_binary(frame.ground_truth))
            (prediction_root / f"{index:02d}.pb").write_bytes(to_binary(frame.perfect_prediction))
        first_root = workspace / "artifacts-first"
        second_root = workspace / "artifacts-second"
        first = _run_once(root, truth_root, prediction_root, first_root)
        second = _run_once(root, truth_root, prediction_root, second_root)
        first_hashes = {
            name: artifact["payload_sha256"] for name, artifact in first["artifacts"].items()
        }
        second_hashes = {
            name: artifact["payload_sha256"] for name, artifact in second["artifacts"].items()
        }
        if first_hashes != second_hashes:
            raise RuntimeError(
                "custom evaluation artifacts differ across reruns: "
                f"{first_hashes} != {second_hashes}"
            )
        _assert_tables(first, first_root)
        print(
            json.dumps(
                {
                    "frame_count": len(frames),
                    "payload_sha256": first_hashes,
                    "schema_version": "junctionlens.m3-2-verification.v1",
                    "state": "ACCEPTED_LOCAL",
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

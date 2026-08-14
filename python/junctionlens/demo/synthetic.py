"""End-to-end unrestricted synthetic evidence workflow."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw

from junctionlens.contract.ids import edge_id, predicted_node_id
from junctionlens.contract.validation import validate_envelope
from junctionlens.evaluator.custom import CustomEvaluationReceipt, evaluate_custom
from junctionlens.faults.models import FaultKind
from junctionlens.faults.service import inject_fault, put_prediction_bundle
from junctionlens.faults.synthetic import build_synthetic_fault_bundle
from junctionlens.gate.comparison import run_comparison
from junctionlens.gate.decision import persist_decision
from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.report import export_evidence_bundle
from junctionlens.synthetic import generate_scene_frames, write_corpus
from junctionlens.v1 import scene_control_graph_pb2 as scg

_DEMO_SEGMENTS = 200
_DEMO_CONTROLS_PER_SEGMENT = 3
_BASE_TIMESTAMP_NS = 1_725_000_000_000_000_000
_TRIAL_ORDER = ["AB", "BA", "BA", "AB", "AB", "BA", "BA", "AB", "AB", "BA"]


class DemoError(RuntimeError):
    """Raised when the synthetic product workflow cannot prove its declared outcome."""


@dataclass(frozen=True, slots=True)
class SyntheticDemoReceipt:
    """Stable identities and measured outcomes from one synthetic demonstration."""

    schema_version: str
    state: str
    release_status: str
    release_status_scope: str
    intended_fault: str
    intended_fault_state: str
    intended_fault_reason_code: str
    paired_segments: int
    eligible_lane_control_edges: int
    baseline_evaluation_manifest_sha256: str
    candidate_evaluation_manifest_sha256: str
    baseline_arm_manifest_sha256: str
    candidate_arm_manifest_sha256: str
    decision_manifest_sha256: str
    comparison_manifest_sha256: str
    scene_manifest_sha256: str
    report_manifest_sha256: str
    report_directory: str
    serve_command: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_u64(*parts: object) -> int:
    value = int.from_bytes(
        hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()[:8],
        "big",
    )
    return value or 1


def _set_frame_identity(envelope: scg.SceneControlGraphEnvelope, index: int) -> None:
    graph = envelope.graph
    graph.frame_key.segment_id = f"demo-segment-{index:03d}"
    graph.frame_key.timestamp_ns = _BASE_TIMESTAMP_NS + index * 1_000_000_000
    graph.frame_key.frame_manifest_sha256 = _sha256_bytes(
        f"junctionlens-synthetic-demo-v1|{index}".encode()
    )
    if graph.HasField("sensor_frame"):
        graph.sensor_frame.frame_key.CopyFrom(graph.frame_key)
    for track in graph.tracks:
        track.last_timestamp_ns = graph.frame_key.timestamp_ns


def _set_control_box(control: Any, ordinal: int) -> None:
    x_min = 0.1 + ordinal * 0.25
    x_max = x_min + 0.08
    control.normalized_half_open_box.x_min = x_min
    control.normalized_half_open_box.x_max = x_max
    if control.HasField("source_pixel_box"):
        width = control.source_pixel_box.image_width
        control.source_pixel_box.x0 = x_min * width
        control.source_pixel_box.x1 = x_max * width


def _expand_controls(envelope: scg.SceneControlGraphEnvelope, *, ground_truth: bool) -> None:
    graph = envelope.graph
    if len(graph.traffic_controls) != 1 or len(graph.lanes) < _DEMO_CONTROLS_PER_SEGMENT:
        raise DemoError("synthetic intersection lacks the demo control and lane preconditions")
    original = deepcopy(graph.traffic_controls[0])
    try:
        original_track = deepcopy(
            next(track for track in graph.tracks if track.current_node_id == original.node_id)
        )
    except StopIteration as error:
        raise DemoError("synthetic intersection control has no track") from error
    retained_tracks = [
        deepcopy(track) for track in graph.tracks if track.current_node_id != original.node_id
    ]
    retained_edges = [
        deepcopy(edge)
        for edge in graph.edges
        if edge.edge_type != scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE
    ]
    del graph.traffic_controls[:]
    del graph.tracks[:]
    graph.tracks.extend(retained_tracks)
    del graph.edges[:]
    graph.edges.extend(retained_edges)
    next_track_id = max((track.track_id for track in graph.tracks), default=0) + 1
    for ordinal in range(_DEMO_CONTROLS_PER_SEGMENT):
        control = graph.traffic_controls.add()
        control.CopyFrom(original)
        control.node_id = (
            _stable_u64("demo-ground-truth-control", ordinal)
            if ground_truth
            else predicted_node_id(scg.NODE_TYPE_TRAFFIC_CONTROL, ordinal)
        )
        control.track_id = next_track_id + ordinal
        control.decoder_query_index = ordinal
        _set_control_box(control, ordinal)
        if ground_truth:
            control.adapter_metadata.source_object_id = f"demo-control-{ordinal}"
            control.adapter_metadata.source_namespace = "synthetic/demo/control"
        track = graph.tracks.add()
        track.CopyFrom(original_track)
        track.track_id = control.track_id
        track.current_node_id = control.node_id
        edge = graph.edges.add(
            edge_type=scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE,
            source_node_id=control.node_id,
            target_node_id=graph.lanes[ordinal].node_id,
            raw_probability=1.0,
            calibrated_probability=1.0,
            binary_decision=True,
        )
        edge.uncertainty.standard_deviation = 0.01
        edge.uncertainty.method = "synthetic-demo-exact"
    _refresh_edge_ids(graph)


def _refresh_edge_ids(graph: scg.SceneControlGraph) -> None:
    for edge in graph.edges:
        edge.edge_id = edge_id(
            graph.frame_key,
            edge.edge_type,
            edge.source_node_id,
            edge.target_node_id,
        )


def _swap_control_edges(envelope: scg.SceneControlGraphEnvelope) -> None:
    graph = envelope.graph
    controls = {control.node_id: ordinal for ordinal, control in enumerate(graph.traffic_controls)}
    changed = 0
    for edge in graph.edges:
        if edge.edge_type != scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE:
            continue
        ordinal = controls[edge.source_node_id]
        edge.target_node_id = graph.lanes[(ordinal + 1) % _DEMO_CONTROLS_PER_SEGMENT].node_id
        edge.raw_probability = 0.99
        edge.calibrated_probability = 0.99
        changed += 1
    if changed != _DEMO_CONTROLS_PER_SEGMENT:
        raise DemoError("synthetic demo did not find every control edge to swap")
    _refresh_edge_ids(graph)


def _write_paired_graphs(root: Path) -> tuple[Path, Path, Path]:
    ground_truth_root = root / "paired" / "ground-truth"
    baseline_root = root / "paired" / "baseline"
    candidate_root = root / "paired" / "candidate"
    for directory in (ground_truth_root, baseline_root, candidate_root):
        directory.mkdir(parents=True, exist_ok=False)
    for index in range(_DEMO_SEGMENTS):
        frames = generate_scene_frames(seed=20_260_813 + index)
        source = next(
            frame
            for frame in frames
            if frame.scene_kind.value == "intersection-crosswalk" and frame.frame_index == 0
        )
        ground_truth = deepcopy(source.ground_truth)
        baseline = deepcopy(source.perfect_prediction)
        candidate = deepcopy(source.perfect_prediction)
        for envelope in (ground_truth, baseline, candidate):
            _set_frame_identity(envelope, index)
        _expand_controls(ground_truth, ground_truth=True)
        _expand_controls(baseline, ground_truth=False)
        _expand_controls(candidate, ground_truth=False)
        _swap_control_edges(candidate)
        for envelope in (ground_truth, baseline, candidate):
            validate_envelope(envelope)
        filename = f"demo-segment-{index:03d}.pb"
        ground_truth_root.joinpath(filename).write_bytes(
            ground_truth.SerializeToString(deterministic=True)
        )
        baseline_root.joinpath(filename).write_bytes(baseline.SerializeToString(deterministic=True))
        candidate_root.joinpath(filename).write_bytes(
            candidate.SerializeToString(deterministic=True)
        )
    return ground_truth_root, baseline_root, candidate_root


def _index_evaluation(registry: EvidenceRegistry, receipt: CustomEvaluationReceipt) -> None:
    for manifest in (
        receipt.match_manifest_sha256,
        receipt.frame_table_manifest_sha256,
        receipt.segment_table_manifest_sha256,
    ):
        registry.index.index_manifest(manifest)


def _read_frame_rows(registry: EvidenceRegistry, manifest_sha256: str) -> list[dict[str, object]]:
    manifest = registry.store.read_manifest(manifest_sha256)
    payload = cast(dict[str, object], manifest["payload"])
    table = pq.read_table(registry.store.object_path(cast(str, payload["sha256"])))
    return cast(list[dict[str, object]], table.to_pylist())


def _frame_evidence(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = {"control_edge_recall", "wrong_control_assignment_rate"}
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        metric = cast(str, row["metric"])
        if metric not in selected:
            continue
        token = cast(str, row["frame_token"])
        frame = grouped.setdefault(
            token,
            {
                "frame_token": token,
                "segment_id": cast(str, row["segment_id"]),
                "timestamp_ns": cast(int, row["timestamp_ns"]),
                "slice_values": {"source_domain": "synthetic"},
                "metrics": {},
            },
        )
        denominator = float(cast(float, row["denominator"]))
        cast(dict[str, object], frame["metrics"])[metric] = {
            "numerator": float(cast(float, row["numerator"])),
            "denominator": denominator,
            "eligible_ground_truth_edges": int(denominator),
        }
    frames = sorted(grouped.values(), key=lambda item: cast(str, item["frame_token"]))
    if len(frames) != _DEMO_SEGMENTS:
        raise DemoError(f"synthetic evaluation produced {len(frames)} rather than 200 frames")
    if any(set(cast(dict[str, object], frame["metrics"])) != selected for frame in frames):
        raise DemoError("synthetic evaluation omitted a mandatory lane-control metric")
    return frames


def _runtime(hardware_manifest_sha256: str) -> dict[str, object]:
    return {
        "hardware_baseline_manifest_sha256": hardware_manifest_sha256,
        "gpu_provider_active": False,
        "throughput_per_second": 0.0,
        "p95_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "peak_device_memory_bytes": 0,
        "long_run_frames": 0,
        "unbounded_memory_growth": False,
        "unexpected_cpu_provider_nodes": 0,
        "warmup_frames_per_block": 0,
        "measured_frames_per_block": 0,
        "trial_block_order": [],
        "environment_valid": True,
        "metrics": {},
    }


def _arm(
    *,
    arm_id: str,
    rows: list[dict[str, object]],
    source_manifest_sha256: str,
    split_manifest_sha256: str,
    hardware_manifest_sha256: str,
    metrics_sha256: str,
    slices_sha256: str,
    evaluator_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": "junctionlens.comparison-arm.v1",
        "arm_id": arm_id,
        "evaluator_image_digest": f"local-custom-match@sha256:{evaluator_digest}",
        "data_manifest_sha256": source_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "preprocessing_sha256": _sha256_bytes(b"junctionlens-synthetic-demo-preprocess-v1"),
        "postprocessing_sha256": _sha256_bytes(b"junctionlens-synthetic-demo-postprocess-v1"),
        "metric_registry_sha256": metrics_sha256,
        "slice_registry_sha256": slices_sha256,
        "integrity": {
            "artifact_integrity": True,
            "schema_major_compatible": True,
            "identifiers_valid": True,
            "coordinate_metadata_valid": True,
            "required_values_finite": True,
            "calibrator_valid": True,
            "evaluator_compatible": True,
            "provenance_complete": True,
            "leakage_free": True,
            "partial_inference_approved": True,
            "provider_fallback_free": True,
            "calibration_ranks_frozen": True,
            "training_holdout_access_count": 0,
        },
        "frames": _frame_evidence(rows),
        "runtime": _runtime(hardware_manifest_sha256),
    }


def _put_arm(
    registry: EvidenceRegistry,
    body: dict[str, object],
    parents: tuple[str, ...],
) -> str:
    receipt = registry.put_bytes(
        canonical_json_bytes(body) + b"\n",
        kind="prediction_bundle",
        media_type="application/vnd.junctionlens.comparison-arm+json",
        license_id="LicenseRef-JunctionLens-Synthetic",
        metadata={"arm_id": body["arm_id"], "demo_only": True},
        parents=parents,
    )
    return receipt.manifest_sha256


def _write_charter(
    path: Path,
    *,
    source_commit: str,
    hardware_manifest_sha256: str,
    metrics_sha256: str,
    slices_sha256: str,
) -> None:
    placeholder = _sha256_bytes(b"junctionlens-synthetic-demo-v1")
    body: dict[str, Any] = {
        "schema_version": "junctionlens.acceptance-charter.v1",
        "charter_id": "junctionlens-acceptance-v1",
        "frozen": True,
        "frozen_at": "2026-08-14T00:00:00+00:00",
        "signer": "repository-owned-synthetic-demo",
        "source_commit": source_commit,
        "draft_sha256": placeholder,
        "baseline_run_manifest_sha256": placeholder,
        "baseline_evidence_payload_sha256": placeholder,
        "metric_registry_sha256": metrics_sha256,
        "slice_registry_sha256": slices_sha256,
        "family_alpha": 0.05,
        "bootstrap": {
            "algorithm": "paired-segment-cluster-bootstrap-v1",
            "replicates": 10000,
            "seed": 20260813,
            "minimum_finite_replicates": 9900,
        },
        "runtime_bootstrap": {
            "algorithm": "paired-trial-block-bootstrap-v1",
            "replicates": 10000,
            "seed": 20260813,
            "minimum_valid_pairs": 8,
            "balanced_order_schedule": _TRIAL_ORDER,
        },
        "support": {
            "overall_paired_segments": 200,
            "gating_slice_paired_segments": 30,
            "lane_control_ground_truth_edges": 500,
            "temporal_adjacent_frame_transitions": 500,
            "temporal_segments": 30,
        },
        "absolute_runtime": {
            "throughput_per_second_minimum": 10.0,
            "p95_latency_ms_maximum": 100.0,
            "p99_latency_ms_maximum": 125.0,
            "peak_device_memory_bytes_maximum": 6442450944,
            "long_run_frames": 10000,
            "unexpected_cpu_provider_nodes_maximum": 0,
        },
        "cells": [
            {
                "id": "overall.control_edge_recall",
                "metric": "control_edge_recall",
                "slice": "overall",
                "direction": "higher_is_better",
                "margin": 0.005,
                "support": "lane_control",
                "estimator": "ratio",
                "stage": "accuracy",
            },
            {
                "id": "overall.wrong_control_assignment_rate",
                "metric": "wrong_control_assignment_rate",
                "slice": "overall",
                "direction": "lower_is_better",
                "margin": 0.005,
                "support": "lane_control",
                "estimator": "ratio",
                "stage": "accuracy",
            },
        ],
        "primary_hypotheses": [],
        "freeze_evidence": {
            "baseline_seed_checkpoint_sha256": {"20260813": placeholder},
            "m0_hardware_baseline_manifest_sha256": hardware_manifest_sha256,
            "power_simulation_artifact_sha256": placeholder,
            "product_priorities_sha256": placeholder,
        },
    }
    body["charter_sha256"] = _sha256_bytes(canonical_json_bytes(body))
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def _git_commit(project_root: Path) -> str:
    git = shutil.which("git")
    if git is None:
        raise DemoError("synthetic demo requires Git")
    result = subprocess.run(
        [git, "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit = result.stdout.strip()
    if (
        result.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise DemoError("synthetic demo requires a Git checkout with a full source commit")
    return commit


def _graph_view(envelope: scg.SceneControlGraphEnvelope) -> dict[str, object]:
    graph = envelope.graph
    lanes = [
        {
            "node_id": str(lane.node_id),
            "confidence": lane.existence_confidence,
            "points": [{"x": point.x, "y": point.y} for point in lane.centerline.points],
        }
        for lane in graph.lanes
    ]
    controls = [
        {
            "node_id": str(control.node_id),
            "x": 12.0 + index * 3.0,
            "y": -5.0,
            "control_type": "traffic_light",
            "state": "synthetic",
            "confidence": control.existence_confidence,
        }
        for index, control in enumerate(graph.traffic_controls)
    ]
    edge_types = {
        scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR: "lane_successor",
        scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE: "control_applies_to_lane",
    }
    edges = [
        {
            "edge_id": str(edge.edge_id),
            "edge_type": edge_types.get(edge.edge_type, "other"),
            "source_node_id": str(edge.source_node_id),
            "target_node_id": str(edge.target_node_id),
            "confidence": edge.calibrated_probability,
        }
        for edge in graph.edges
        if edge.edge_type in edge_types
    ]
    return {"lanes": lanes, "controls": controls, "edges": edges}


def _demo_png() -> bytes:
    image = Image.new("RGB", (640, 360), (16, 42, 47))
    draw = ImageDraw.Draw(image)
    draw.line((120, 350, 285, 140, 355, 140, 520, 350), fill=(246, 200, 95), width=8)
    draw.rectangle((120, 36, 520, 320), outline=(218, 235, 237), width=3)
    draw.text((145, 60), "JunctionLens synthetic control swap", fill=(245, 250, 251))
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _put_scene(
    registry: EvidenceRegistry,
    decision_manifest_sha256: str,
    ground_truth_path: Path,
    baseline_path: Path,
    candidate_path: Path,
) -> str:
    image = registry.put_bytes(
        _demo_png(),
        kind="evidence_report",
        media_type="image/png",
        license_id="CC0-1.0",
        metadata={"source": "repository-owned-synthetic-demo"},
    )

    def load(path: Path) -> scg.SceneControlGraphEnvelope:
        envelope = scg.SceneControlGraphEnvelope()
        envelope.ParseFromString(path.read_bytes())
        return envelope

    scene = {
        "schema_version": "junctionlens.scene-bundle.v1",
        "title": "Synthetic swapped lane-control associations",
        "decision_manifest_sha256": decision_manifest_sha256,
        "license_notice": "CC0-1.0 repository-owned synthetic scene",
        "frames": [
            {
                "frame_id": "demo-segment-000",
                "segment_id": "demo-segment-000",
                "timestamp_ns": str(_BASE_TIMESTAMP_NS),
                "cameras": [
                    {
                        "slot": "front-center",
                        "label": "Synthetic front center",
                        "artifact_manifest_sha256": image.manifest_sha256,
                        "restriction_reason": None,
                    }
                ],
                "ground_truth": _graph_view(load(ground_truth_path)),
                "baseline": _graph_view(load(baseline_path)),
                "candidate": _graph_view(load(candidate_path)),
            }
        ],
    }
    receipt = registry.put_bytes(
        canonical_json_bytes(scene) + b"\n",
        kind="counterexample_bundle",
        media_type="application/vnd.junctionlens.scene-bundle+json",
        license_id="CC0-1.0",
        metadata={"frame_count": 1, "source": "repository-owned-synthetic-demo"},
        parents=tuple(sorted((decision_manifest_sha256, image.manifest_sha256))),
    )
    return receipt.manifest_sha256


def _decision_payload(registry: EvidenceRegistry, manifest_sha256: str) -> dict[str, Any]:
    manifest = registry.store.read_manifest(manifest_sha256)
    payload = cast(dict[str, object], manifest["payload"])
    return cast(
        dict[str, Any],
        json.loads(registry.store.object_path(cast(str, payload["sha256"])).read_bytes()),
    )


def _validate_expected_decision(decision: dict[str, Any]) -> None:
    cells = {cell["cell_id"]: cell for cell in decision["cells"]}
    expected = {
        "overall.control_edge_recall",
        "overall.wrong_control_assignment_rate",
    }
    if set(cells) != expected:
        raise DemoError("synthetic comparison did not produce the exact declared gate cells")
    if any(cell["status"] != "FAIL_REGRESSION" for cell in cells.values()):
        raise DemoError("synthetic swapped-control candidate was not rejected by every gate cell")
    if any(cell["reason_code"] != "GATE_REGRESSION_CI_BELOW_MARGIN" for cell in cells.values()):
        raise DemoError("synthetic comparison failed for an unintended gate reason")
    if decision["status"] != "BLOCKED_INFRASTRUCTURE":
        raise DemoError(
            "CPU-only synthetic release status must remain blocked on GPU qualification"
        )
    if (
        "GATE_INFRASTRUCTURE_GPU_PROVIDER_UNAVAILABLE"
        not in decision["infrastructure_reason_codes"]
    ):
        raise DemoError("synthetic comparison did not preserve the missing-GPU reason")


def run_synthetic_demo(output_root: Path, project_root: Path) -> SyntheticDemoReceipt:
    """Build real synthetic evidence and refuse release claims without GPU qualification."""
    project_root = project_root.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if output_root.exists() or output_root.is_symlink():
        raise DemoError(f"demo output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(mode=0o755)
    try:
        schema = project_root / "schemas/artifact-manifest-v1.schema.json"
        metrics = project_root / "configs/metrics/v1.yaml"
        slices = project_root / "configs/slices/v1.yaml"
        registry = EvidenceRegistry(output_root, schema)
        corpus = write_corpus(output_root / "synthetic", seed=20_260_813)
        source = registry.put_bytes(
            corpus.files["manifest.json"],
            kind="dataset_lock",
            media_type="application/json",
            license_id="LicenseRef-JunctionLens-Synthetic",
            metadata={"profile": "synthetic-demo", "segments": _DEMO_SEGMENTS},
        )
        ground_truth_root, baseline_root, candidate_root = _write_paired_graphs(output_root)
        split_body = {
            "schema_version": "junctionlens.synthetic-demo-split.v1",
            "segments": [f"demo-segment-{index:03d}" for index in range(_DEMO_SEGMENTS)],
        }
        split = registry.put_bytes(
            canonical_json_bytes(split_body) + b"\n",
            kind="split_manifest",
            media_type="application/json",
            license_id="LicenseRef-JunctionLens-Synthetic",
            metadata={"segment_count": _DEMO_SEGMENTS},
            parents=(source.manifest_sha256,),
        )
        hardware_body = {
            "schema_version": "junctionlens.synthetic-demo-environment.v1",
            "profile": "cpu-local",
            "accelerated_qualification": False,
            "performance_measurements_included": False,
        }
        hardware = registry.put_bytes(
            canonical_json_bytes(hardware_body) + b"\n",
            kind="benchmark",
            media_type="application/json",
            license_id="Apache-2.0",
            metadata={"profile": "cpu-local", "accelerated_qualification": False},
        )
        baseline_evaluation = evaluate_custom(
            ground_truth_root, baseline_root, output_root, project_root
        )
        candidate_evaluation = evaluate_custom(
            ground_truth_root, candidate_root, output_root, project_root
        )
        _index_evaluation(registry, baseline_evaluation)
        _index_evaluation(registry, candidate_evaluation)
        metrics_sha256 = _sha256_file(metrics)
        slices_sha256 = _sha256_file(slices)
        evaluator_digest = _sha256_file(project_root / "containers/custom_match.py")
        baseline_arm = _arm(
            arm_id="synthetic-perfect-baseline",
            rows=_read_frame_rows(registry, baseline_evaluation.frame_table_manifest_sha256),
            source_manifest_sha256=source.manifest_sha256,
            split_manifest_sha256=split.manifest_sha256,
            hardware_manifest_sha256=hardware.manifest_sha256,
            metrics_sha256=metrics_sha256,
            slices_sha256=slices_sha256,
            evaluator_digest=evaluator_digest,
        )
        candidate_arm = _arm(
            arm_id="synthetic-swapped-control-candidate",
            rows=_read_frame_rows(registry, candidate_evaluation.frame_table_manifest_sha256),
            source_manifest_sha256=source.manifest_sha256,
            split_manifest_sha256=split.manifest_sha256,
            hardware_manifest_sha256=hardware.manifest_sha256,
            metrics_sha256=metrics_sha256,
            slices_sha256=slices_sha256,
            evaluator_digest=evaluator_digest,
        )
        shared_parents = tuple(sorted((source.manifest_sha256, split.manifest_sha256)))
        baseline_arm_manifest = _put_arm(
            registry,
            baseline_arm,
            tuple(sorted((*shared_parents, baseline_evaluation.frame_table_manifest_sha256))),
        )
        candidate_arm_manifest = _put_arm(
            registry,
            candidate_arm,
            tuple(sorted((*shared_parents, candidate_evaluation.frame_table_manifest_sha256))),
        )
        charter = output_root / "synthetic-demo-charter.yaml"
        _write_charter(
            charter,
            source_commit=_git_commit(project_root),
            hardware_manifest_sha256=hardware.manifest_sha256,
            metrics_sha256=metrics_sha256,
            slices_sha256=slices_sha256,
        )
        comparison = run_comparison(
            artifact_root=output_root,
            schema_path=schema,
            charter_path=charter,
            metric_registry_path=metrics,
            slice_registry_path=slices,
            baseline_manifest_sha256=baseline_arm_manifest,
            candidate_manifest_sha256=candidate_arm_manifest,
        )
        decision = _decision_payload(registry, comparison.decision_manifest_sha256)
        _validate_expected_decision(decision)
        evidence_manifest = registry.store.read_manifest(comparison.evidence_manifest_sha256)
        evidence_payload = cast(dict[str, object], evidence_manifest["payload"])
        evidence_path = registry.store.object_path(cast(str, evidence_payload["sha256"]))
        replay_path = output_root / "standalone-gate-decision.json"
        replayed = persist_decision(charter, evidence_path, replay_path)
        if canonical_json_bytes(replayed) + b"\n" != replay_path.read_bytes():
            raise DemoError("standalone gate replay was not byte-stable")
        if replayed["decision_sha256"] != decision["decision_sha256"]:
            raise DemoError("standalone gate replay differs from the comparison decision")
        fault_parent = put_prediction_bundle(registry, build_synthetic_fault_bundle())
        fault = inject_fault(
            artifact_root=output_root,
            schema_path=schema,
            input_manifest_sha256=fault_parent,
            kind=FaultKind.SWAP_CONTROL_EDGES,
        )
        if fault.state != "DETECTED" or fault.primary_reason_code != (
            "FAULT_CONTROL_ASSIGNMENT_CHANGED"
        ):
            raise DemoError("flagship swapped-control fault failed its intended detector")
        scene_manifest = _put_scene(
            registry,
            comparison.decision_manifest_sha256,
            ground_truth_root / "demo-segment-000.pb",
            baseline_root / "demo-segment-000.pb",
            candidate_root / "demo-segment-000.pb",
        )
        report_directory = output_root / "public-report"
        report = export_evidence_bundle(
            artifact_root=output_root,
            schema_path=schema,
            project_root=project_root,
            comparison_manifest_sha256=comparison.report_data_manifest_sha256,
            output_directory=report_directory,
            mode="public",
            scene_manifest_sha256=scene_manifest,
        )
        return SyntheticDemoReceipt(
            schema_version="junctionlens.synthetic-demo-receipt.v1",
            state="DEMONSTRATED",
            release_status=comparison.status,
            release_status_scope="BLOCKED until accelerated qualification is available",
            intended_fault=FaultKind.SWAP_CONTROL_EDGES.value,
            intended_fault_state=fault.state,
            intended_fault_reason_code=fault.primary_reason_code,
            paired_segments=_DEMO_SEGMENTS,
            eligible_lane_control_edges=_DEMO_SEGMENTS * _DEMO_CONTROLS_PER_SEGMENT,
            baseline_evaluation_manifest_sha256=(baseline_evaluation.frame_table_manifest_sha256),
            candidate_evaluation_manifest_sha256=(candidate_evaluation.frame_table_manifest_sha256),
            baseline_arm_manifest_sha256=baseline_arm_manifest,
            candidate_arm_manifest_sha256=candidate_arm_manifest,
            decision_manifest_sha256=comparison.decision_manifest_sha256,
            comparison_manifest_sha256=comparison.report_data_manifest_sha256,
            scene_manifest_sha256=scene_manifest,
            report_manifest_sha256=report.archive.manifest_sha256,
            report_directory=str(report_directory),
            serve_command=f"uv run junctionlens serve --artifact-root {output_root}",
        )
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


__all__ = ["DemoError", "SyntheticDemoReceipt", "run_synthetic_demo"]

"""Frozen checkpoint scoring on the isolated model-selection partition."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import functional

from junctionlens.contract import canonical_logical_sha256
from junctionlens.data.contracts import AdaptedFrame, CameraSlot
from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.evaluator.custom import (
    build_custom_match_request,
    compute_custom_metrics,
    evaluate_custom,
    load_graph_pairs,
    run_custom_match,
)
from junctionlens.evaluator.evidence import adaptive_ece
from junctionlens.evaluator.official import evaluate_official
from junctionlens.model.e0_data import (
    PartitionIsolation,
    frame_inputs,
    frame_rule_observations,
    frame_targets,
    iter_partition_frames,
)
from junctionlens.model.e0_losses import E0Matches, E0Targets, match_e0
from junctionlens.model.e0_profile import E0Profile
from junctionlens.model.e1_profile import E1Profile
from junctionlens.model.evaluation_graphs import (
    ground_truth_envelope,
    producer_info,
    sensor_frame,
)
from junctionlens.model.independent_linker import (
    IndependentLinkerArtifact,
    IndependentLinkerError,
    RuleObservation,
    control_rule_features,
    fit_independent_linker,
    successor_rule_features,
)
from junctionlens.model.reference import ReferenceNodeModel, e0_outputs_by_name
from junctionlens.model.topology import JointGraphModel, e1_outputs_by_name
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.runtime.reference import reference_postprocess
from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_json_object_path

Experiment = Literal["E0-independent", "E1-joint"]


class SelectionEvaluationError(RuntimeError):
    """Raised when checkpoint scoring differs from its frozen partition contract."""


def resolve_source_commit(project_root: Path) -> str:
    """Resolve a clean source identity or the verified remote-runner override."""
    value = os.environ.get("JUNCTIONLENS_SOURCE_COMMIT")
    if value is None:
        git = shutil.which("git")
        if git is None:
            raise SelectionEvaluationError("git is required to identify selected evaluation")
        status = subprocess.run(
            [git, "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.returncode != 0 or status.stdout:
            raise SelectionEvaluationError(
                "selected evaluation requires a clean, explainable source checkout"
            )
        commit = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if commit.returncode != 0:
            raise SelectionEvaluationError("cannot resolve selected evaluation source")
        value = commit.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise SelectionEvaluationError("source commit must be a full lowercase Git object ID")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SelectionEvaluationError("selection evidence output cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise SelectionEvaluationError("selection evidence output already differs")
        return
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
        temporary = Path(output.name)
    temporary.replace(path)


def fit_e0_linker(
    profile: E0Profile,
    adapter: OpenLaneAdapter,
    isolation: PartitionIsolation,
    output_path: Path,
) -> IndependentLinkerArtifact:
    """Fit and persist both independent rules from model-training labels only."""
    if isolation.partition != "model_training":
        raise SelectionEvaluationError("E0 linker fitting requires model_training")
    successor: list[RuleObservation] = []
    control: list[RuleObservation] = []
    for frame in iter_partition_frames(adapter, isolation):
        frame_successor, frame_control = frame_rule_observations(frame)
        successor.extend(frame_successor)
        control.extend(frame_control)
    try:
        artifact = fit_independent_linker(
            profile,
            tuple(successor),
            tuple(control),
            partition=isolation.partition,
            training_split_manifest_sha256=isolation.split_manifest_sha256,
        )
    except IndependentLinkerError as error:
        raise SelectionEvaluationError(str(error)) from error
    _atomic_bytes(output_path, artifact.canonical_bytes())
    return artifact


def load_e0_linker(path: Path, profile: E0Profile, split_sha256: str) -> IndependentLinkerArtifact:
    """Load an exact training-only linker artifact."""
    from junctionlens.model.independent_linker import FittedThreshold

    try:
        value = load_json_object_path(
            path,
            "E0 independent linker",
            ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
        )
        if set(value) != {
            "schema_version",
            "algorithm",
            "fit_partition",
            "training_split_manifest_sha256",
            "profile_sha256",
            "successor",
            "control",
            "observation_sha256",
        }:
            raise ValueError("linker keys differ")
        thresholds = []
        for name in ("successor", "control"):
            raw = value[name]
            if not isinstance(raw, dict):
                raise ValueError("linker threshold is not an object")
            thresholds.append(FittedThreshold(**raw))
        artifact = IndependentLinkerArtifact(
            schema_version=str(value["schema_version"]),
            algorithm=str(value["algorithm"]),
            fit_partition=str(value["fit_partition"]),
            training_split_manifest_sha256=str(value["training_split_manifest_sha256"]),
            profile_sha256=str(value["profile_sha256"]),
            successor=thresholds[0],
            control=thresholds[1],
            observation_sha256=str(value["observation_sha256"]),
        )
    except (KeyError, ParseBoundaryError, TypeError, ValueError) as error:
        raise SelectionEvaluationError("E0 independent linker artifact is invalid") from error
    if (
        artifact.schema_version != "junctionlens.independent-linker.v1"
        or artifact.fit_partition != "model_training"
        or artifact.training_split_manifest_sha256 != split_sha256
        or artifact.profile_sha256 != profile.canonical_sha256()
        or artifact.canonical_bytes() != path.read_bytes()
    ):
        raise SelectionEvaluationError("E0 independent linker identity differs")
    return artifact


def _sigmoid(value: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _ordered_box(values: npt.NDArray[np.float32]) -> tuple[float, float, float, float]:
    x_min, x_max = sorted((float(values[0]), float(values[2])))
    y_min, y_max = sorted((float(values[1]), float(values[3])))
    x_min, x_max = min(1.0, max(0.0, x_min)), min(1.0, max(0.0, x_max))
    y_min, y_max = min(1.0, max(0.0, y_min)), min(1.0, max(0.0, y_max))
    epsilon = 1.0e-7
    if x_min >= x_max:
        x_min, x_max = max(0.0, x_min - epsilon), min(1.0, x_max + epsilon)
    if y_min >= y_max:
        y_min, y_max = max(0.0, y_min - epsilon), min(1.0, y_max + epsilon)
    return x_min, y_min, x_max, y_max


def _truth_annotation(frame: AdaptedFrame) -> dict[str, Any]:
    return {
        "lane_segment": [
            {
                "id": index + 1,
                "centerline": [list(point) for point in item.centerline],
                "left_laneline": [list(point) for point in item.left_boundary],
                "right_laneline": [list(point) for point in item.right_boundary],
            }
            for index, item in enumerate(frame.lanes)
        ],
        "traffic_element": [
            {
                "id": index + 1,
                "attribute": item.attribute,
                "points": [list(point) for point in item.source_pixel_box.points],
            }
            for index, item in enumerate(frame.traffic_controls)
        ],
        "area": [
            {
                "id": index + 1,
                "category": item.category,
                "points": [list(point) for point in item.points],
            }
            for index, item in enumerate(frame.road_areas)
        ],
        "topology_lsls": [list(row) for row in frame.topology_lane_lane],
        "topology_lste": [list(row) for row in frame.topology_lane_traffic],
    }


def _safe_rule_probabilities(
    outputs: Mapping[str, npt.NDArray[np.float32]],
    frame: AdaptedFrame,
    linker: IndependentLinkerArtifact,
    lane_queries: list[int],
    traffic_queries: list[int],
) -> tuple[list[list[float]], list[list[float]]]:
    centers = {
        query: np.asarray(outputs["lane_centerline"][0, query], dtype=np.float64)
        for query in lane_queries
    }
    lane_matrix = [[0.0 for _ in lane_queries] for _ in lane_queries]
    for source_index, source_query in enumerate(lane_queries):
        for target_index, target_query in enumerate(lane_queries):
            if source_query == target_query:
                continue
            try:
                distance, heading = successor_rule_features(
                    centers[source_query], centers[target_query]
                )
            except IndependentLinkerError:
                continue
            lane_matrix[source_index][target_index] = float(
                distance <= linker.successor.distance
                and heading <= linker.successor.heading_difference_deg
            )
    camera = next(item for item in frame.cameras if item.slot == CameraSlot.FRONT_CENTER)
    intrinsic = np.asarray(camera.intrinsic, dtype=np.float64)
    transform = np.asarray(camera.t_vehicle_camera, dtype=np.float64)
    control_major = [[0.0 for _ in lane_queries] for _ in traffic_queries]
    for control_index, control_query in enumerate(traffic_queries):
        normalized = _ordered_box(outputs["traffic_boxes"][0, control_query])
        box = (
            normalized[0] * camera.original_width,
            normalized[1] * camera.original_height,
            normalized[2] * camera.original_width,
            normalized[3] * camera.original_height,
        )
        for lane_index, lane_query in enumerate(lane_queries):
            try:
                features = control_rule_features(
                    centers[lane_query],
                    box,
                    intrinsic,
                    transform,
                    camera.original_width,
                    camera.original_height,
                )
            except IndependentLinkerError:
                continue
            if features is not None:
                distance, heading = features
                control_major[control_index][lane_index] = float(
                    distance <= linker.control.distance
                    and heading <= linker.control.heading_difference_deg
                )
    return lane_matrix, control_major


def official_annotations(
    frame: AdaptedFrame,
    outputs: Mapping[str, npt.NDArray[np.float32]],
    *,
    experiment: Experiment,
    linker: IndependentLinkerArtifact | None,
    node_threshold: float = 0.5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one frame and raw outputs into the pinned evaluator schema."""
    lane_confidence = _sigmoid(outputs["lane_existence_logits"][0])
    traffic_confidence = _sigmoid(outputs["traffic_existence_logits"][0])
    area_confidence = _sigmoid(outputs["area_existence_logits"][0])
    lane_queries = [index for index, value in enumerate(lane_confidence) if value >= node_threshold]
    traffic_queries = [
        index for index, value in enumerate(traffic_confidence) if value >= node_threshold
    ]
    area_queries = [index for index, value in enumerate(area_confidence) if value >= node_threshold]
    camera = next(item for item in frame.cameras if item.slot == CameraSlot.FRONT_CENTER)
    if experiment == "E1-joint":
        lane_matrix = _sigmoid(outputs["lane_successor_logits"][0])
        control_major = _sigmoid(outputs["control_lane_logits"][0])
        topology_ll = [
            [float(lane_matrix[source, target]) for target in lane_queries]
            for source in lane_queries
        ]
        topology_lt = [
            [float(control_major[control, lane]) for control in traffic_queries]
            for lane in lane_queries
        ]
    else:
        if linker is None:
            raise SelectionEvaluationError("E0 scoring requires a frozen independent linker")
        topology_ll, control_major_rows = _safe_rule_probabilities(
            outputs, frame, linker, lane_queries, traffic_queries
        )
        topology_lt = [
            [control_major_rows[control][lane] for control in range(len(traffic_queries))]
            for lane in range(len(lane_queries))
        ]
    prediction: dict[str, Any] = {
        "lane_segment": [
            {
                "id": query + 1,
                "centerline": outputs["lane_centerline"][0, query].tolist(),
                "left_laneline": outputs["lane_left_boundary"][0, query].tolist(),
                "right_laneline": outputs["lane_right_boundary"][0, query].tolist(),
                "confidence": float(lane_confidence[query]),
            }
            for query in lane_queries
        ],
        "traffic_element": [],
        "area": [],
        "topology_lsls": topology_ll,
        "topology_lste": topology_lt,
    }
    for query in traffic_queries:
        box = _ordered_box(outputs["traffic_boxes"][0, query])
        prediction["traffic_element"].append(
            {
                "id": query + 1,
                "attribute": int(np.argmax(outputs["traffic_attribute_logits"][0, query])),
                "points": [
                    [box[0] * camera.original_width, box[1] * camera.original_height],
                    [box[2] * camera.original_width, box[3] * camera.original_height],
                ],
                "confidence": float(traffic_confidence[query]),
            }
        )
    for query in area_queries:
        valid_logits = outputs["area_valid_logits"][0, query]
        indices = [index for index, value in enumerate(_sigmoid(valid_logits)) if value >= 0.5]
        if len(indices) < 3:
            indices = sorted(np.argsort(-valid_logits)[:3].tolist())
        prediction["area"].append(
            {
                "id": query + 1,
                "category": int(np.argmax(outputs["area_category_logits"][0, query])) + 1,
                "points": outputs["area_points"][0, query, indices].tolist(),
                "confidence": float(area_confidence[query]),
            }
        )
    return _truth_annotation(frame), prediction


def _move_targets(targets: E0Targets, device: torch.device) -> E0Targets:
    return E0Targets(**{name: value.to(device) for name, value in asdict(targets).items()})


def _existence_nll(logits: Tensor, predictions: Tensor) -> Tensor:
    target = torch.zeros_like(logits)
    target[0, predictions] = 1.0
    return functional.binary_cross_entropy_with_logits(logits, target)


def selection_nll(outputs: Mapping[str, Tensor], targets: E0Targets, matches: E0Matches) -> float:
    """Compute the predeclared nonnegative node probability NLL tie breaker."""
    values = [
        _existence_nll(outputs["lane_existence_logits"], matches.lane.prediction),
        _existence_nll(outputs["traffic_existence_logits"], matches.traffic.prediction),
        _existence_nll(outputs["area_existence_logits"], matches.area.prediction),
    ]
    if matches.lane.prediction.numel():
        values.extend(
            (
                functional.cross_entropy(
                    outputs["lane_left_boundary_logits"][0, matches.lane.prediction],
                    targets.lane_left_type[matches.lane.target],
                ),
                functional.cross_entropy(
                    outputs["lane_right_boundary_logits"][0, matches.lane.prediction],
                    targets.lane_right_type[matches.lane.target],
                ),
            )
        )
    if matches.traffic.prediction.numel():
        values.extend(
            (
                functional.cross_entropy(
                    outputs["traffic_category_logits"][0, matches.traffic.prediction],
                    targets.traffic_category[matches.traffic.target],
                ),
                functional.cross_entropy(
                    outputs["traffic_attribute_logits"][0, matches.traffic.prediction],
                    targets.traffic_attribute[matches.traffic.target],
                ),
            )
        )
    if matches.area.prediction.numel():
        values.append(
            functional.cross_entropy(
                outputs["area_category_logits"][0, matches.area.prediction],
                targets.area_category[matches.area.target],
            )
        )
    result = float(torch.stack(values).mean().item())
    if not math.isfinite(result) or result < 0.0:
        raise SelectionEvaluationError("selection NLL is nonfinite or negative")
    return result


def _load_model(
    checkpoint_path: Path,
    experiment: Experiment,
    base: E0Profile,
    e1: E1Profile | None,
    device: torch.device,
) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise SelectionEvaluationError("checkpoint is not an object")
    if experiment == "E0-independent":
        if (
            state.get("schema_version") != "junctionlens.e0-checkpoint.v1"
            or state.get("profile_sha256") != base.canonical_sha256()
        ):
            raise SelectionEvaluationError("E0 checkpoint identity differs")
        model: torch.nn.Module = ReferenceNodeModel(base)
    else:
        if e1 is None:
            raise SelectionEvaluationError("E1 checkpoint requires the E1 profile")
        if (
            state.get("schema_version") != "junctionlens.e1-checkpoint.v1"
            or state.get("base_profile_sha256") != base.canonical_sha256()
            or state.get("e1_profile_sha256") != e1.canonical_sha256()
        ):
            raise SelectionEvaluationError("E1 checkpoint identity differs")
        model = JointGraphModel(base, e1)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def score_checkpoints(
    *,
    experiment: Experiment,
    run_root: Path,
    adapter: OpenLaneAdapter,
    isolation: PartitionIsolation,
    base_profile: E0Profile,
    e1_profile: E1Profile | None,
    linker: IndependentLinkerArtifact | None,
    project_root: Path,
    output_root: Path,
    device_name: str | None,
) -> Mapping[str, Any]:
    """Score every retained epoch with exact official metrics and isolated NLL."""
    if isolation.partition != "model_selection":
        raise SelectionEvaluationError("checkpoint scoring requires model_selection")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SelectionEvaluationError("CUDA scoring was requested but is unavailable")
    checkpoints = sorted((run_root / "checkpoints").glob("epoch-*.pt"))
    if not checkpoints:
        raise SelectionEvaluationError("training run has no retained epoch checkpoints")
    if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
        raise SelectionEvaluationError("selection output must be a real directory")
    output_root.mkdir(parents=True, exist_ok=True)
    scores = []
    schema = (
        "junctionlens.e0-selection-scores.v1"
        if experiment == "E0-independent"
        else "junctionlens.e1-selection-scores.v1"
    )
    for checkpoint_path in checkpoints:
        try:
            epoch = int(checkpoint_path.stem.split("-")[-1])
        except ValueError as error:
            raise SelectionEvaluationError("checkpoint filename has no integer epoch") from error
        result_path = output_root / f"epoch-{epoch:02d}.json"
        if result_path.is_file():
            cached = load_json_object_path(
                result_path,
                "checkpoint selection result",
                ParseLimits(max_bytes=4 * 1024 * 1024, max_depth=16, max_nodes=100_000),
            )
            if (
                cached.get("checkpoint_sha256") == _sha256_file(checkpoint_path)
                and cached.get("selection_split_manifest_sha256") == isolation.split_manifest_sha256
            ):
                scores.append(cast(dict[str, Any], cached["score"]))
                continue
        model = _load_model(checkpoint_path, experiment, base_profile, e1_profile, device)
        ground_truth: dict[str, Any] = {}
        predictions: dict[str, Any] = {}
        nll_values: list[float] = []
        with torch.inference_mode():
            for frame in iter_partition_frames(adapter, isolation):
                inputs = tuple(
                    value.to(device)
                    for value in frame_inputs(adapter, frame, base_profile).tensors()
                )
                raw = model(*inputs)
                named = (
                    e0_outputs_by_name(raw)
                    if experiment == "E0-independent"
                    else e1_outputs_by_name(raw)
                )
                targets = _move_targets(frame_targets(frame, base_profile), device)
                matches = match_e0(named, targets)
                nll_values.append(selection_nll(named, targets, matches))
                arrays = {
                    name: value.detach().to("cpu", dtype=torch.float32).numpy()
                    for name, value in named.items()
                }
                truth, prediction = official_annotations(
                    frame,
                    arrays,
                    experiment=experiment,
                    linker=linker,
                )
                token = f"{frame.key.segment_id}/{frame.key.timestamp_ns}"
                ground_truth[token] = {"annotation": truth}
                predictions[token] = {"predictions": prediction}
        if not nll_values:
            raise SelectionEvaluationError("model-selection partition contains no frames")
        request = {
            "schema_version": "junctionlens.official-evaluator-input.v1",
            "ground_truth": ground_truth,
            "predictions": {"results": predictions},
        }
        request_path = output_root / f"epoch-{epoch:02d}.official-input.json"
        request_bytes = canonical_json_bytes(request) + b"\n"
        _atomic_bytes(request_path, request_bytes)
        official = evaluate_official(request_path, project_root)
        metrics = official.get("metrics")
        if not isinstance(metrics, dict):
            raise SelectionEvaluationError("official evaluator returned no metrics")
        for name in ("TOP_lt", "OLUS"):
            if not isinstance(metrics.get(name), int | float):
                raise SelectionEvaluationError(f"official evaluator omitted {name}")
        score = {
            "epoch": epoch,
            "lane_control_topology": float(metrics["TOP_lt"]),
            "official_composite": float(metrics["OLUS"]),
            "negative_log_likelihood": sum(nll_values) / len(nll_values),
            "selection_split_manifest_sha256": isolation.split_manifest_sha256,
        }
        result = {
            "schema_version": "junctionlens.checkpoint-selection-result.v1",
            "experiment_id": experiment,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "official_input_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "official_evaluation": official,
            "selection_split_manifest_sha256": isolation.split_manifest_sha256,
            "frame_count": len(ground_truth),
            "score": score,
        }
        _atomic_bytes(result_path, canonical_json_bytes(result) + b"\n")
        scores.append(score)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "schema_version": schema,
        "scores": scores,
    }
    _atomic_bytes(output_root / "scores.json", canonical_json_bytes(output) + b"\n")
    return output


def _prediction_outputs(
    outputs: Mapping[str, npt.NDArray[np.float32]],
    frame: AdaptedFrame,
    experiment: Experiment,
    linker: IndependentLinkerArtifact | None,
) -> dict[str, npt.NDArray[np.float32]]:
    result = {name: value.copy() for name, value in outputs.items()}
    if experiment == "E1-joint":
        return result
    if linker is None:
        raise SelectionEvaluationError("E0 graph conversion requires its frozen linker")
    lane_count = int(outputs["lane_existence_logits"].shape[1])
    traffic_count = int(outputs["traffic_existence_logits"].shape[1])
    lane_probabilities, control_probabilities = _safe_rule_probabilities(
        outputs,
        frame,
        linker,
        list(range(lane_count)),
        list(range(traffic_count)),
    )
    result["lane_successor_logits"] = np.where(
        np.asarray(lane_probabilities, dtype=np.bool_), 20.0, -20.0
    ).astype(np.float32)[None]
    result["control_lane_logits"] = np.where(
        np.asarray(control_probabilities, dtype=np.bool_), 20.0, -20.0
    ).astype(np.float32)[None]
    return result


def _custom_ratio(frame_metrics: Mapping[str, tuple[Any, ...]], metric_name: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for metrics in frame_metrics.values():
        for metric in metrics:
            if metric.name != metric_name:
                continue
            if metric.numerator is not None and metric.denominator is not None:
                numerator += float(metric.numerator)
                denominator += float(metric.denominator)
    if denominator <= 0.0:
        raise SelectionEvaluationError(
            f"custom metric {metric_name} has no eligible model-selection population"
        )
    return numerator / denominator


def _run_seed(run_root: Path, experiment: Experiment) -> int:
    try:
        manifest = load_json_object_path(
            run_root / "run-manifest.json",
            "selected model run manifest",
            ParseLimits(max_bytes=4 * 1024 * 1024, max_depth=24, max_nodes=100_000),
        )
        seed = manifest.get("seed")
        if (
            manifest.get("experiment_id") != experiment
            or manifest.get("state") != "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION"
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError("run identity differs")
    except (ParseBoundaryError, ValueError) as error:
        raise SelectionEvaluationError("selected model run manifest is invalid") from error
    return seed


def _existence_ece(
    pairs: Sequence[Any], match_artifact: Mapping[str, Any]
) -> tuple[float, list[Mapping[str, Any]]]:
    confidences: list[float] = []
    correctness: list[int] = []
    stable_ids: list[str] = []
    frames = match_artifact.get("frames")
    if not isinstance(frames, dict):
        raise SelectionEvaluationError("custom match artifact has no frame population")
    for pair in pairs:
        frame = frames.get(pair.frame_token)
        if not isinstance(frame, dict):
            raise SelectionEvaluationError("custom match artifact frame identity differs")
        for object_type in ("area", "lane_segment", "traffic_element"):
            section = frame.get(object_type)
            records = section.get("predictions") if isinstance(section, dict) else None
            if not isinstance(records, list):
                raise SelectionEvaluationError("custom match existence population is incomplete")
            for record in records:
                if not isinstance(record, dict):
                    raise SelectionEvaluationError("custom match existence record is invalid")
                confidence = record.get("raw_confidence")
                prediction_id = record.get("prediction_id")
                if (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, int | float)
                    or not isinstance(prediction_id, str)
                ):
                    raise SelectionEvaluationError("custom match existence record is invalid")
                confidences.append(float(confidence))
                correctness.append(int(record.get("selected_ground_truth_source_id") is not None))
                stable_ids.append(f"{pair.frame_token}:{object_type}:{prediction_id}")
    if not confidences:
        raise SelectionEvaluationError("selected model emitted no calibration population")
    ece, bins = adaptive_ece(confidences, correctness, stable_ids, bins=15)
    return ece, [asdict(item) for item in bins]


def _measured_example(
    frame_metrics: Mapping[str, tuple[Any, ...]], source_domains: Mapping[str, str]
) -> Mapping[str, Any]:
    candidates: list[tuple[float, str, Mapping[str, float | None]]] = []
    adverse = {
        "wrong_control_assignment_rate",
        "path_blocking_rate_h3",
        "spurious_successor_rate",
    }
    protective = {"control_edge_recall", "reachability_recall_h3"}
    for token, values in frame_metrics.items():
        observed: dict[str, float | None] = {}
        severity = 0.0
        for metric in values:
            if metric.name not in adverse | protective:
                continue
            value = None if metric.value is None else float(metric.value)
            observed[metric.name] = value
            if value is not None:
                severity += value if metric.name in adverse else 1.0 - value
        candidates.append((severity, token, observed))
    if not candidates:
        raise SelectionEvaluationError("selected evaluation produced no per-frame custom evidence")
    severity, token, metrics = min(candidates, key=lambda item: (-item[0], item[1]))
    return {
        "frame_token": token,
        "source_domain": source_domains[token],
        "classification": "MEASURED_FAILURE" if severity > 0.0 else "WORST_OBSERVED_CONTROL",
        "severity_score": severity,
        "metrics": metrics,
    }


def _source_domain_evaluations(
    *,
    request: Mapping[str, Any],
    source_domains: Mapping[str, str],
    output_root: Path,
    project_root: Path,
) -> Mapping[str, Mapping[str, Any]]:
    ground_truth = cast(dict[str, Any], request["ground_truth"])
    predictions = cast(dict[str, Any], cast(dict[str, Any], request["predictions"])["results"])
    result: dict[str, Mapping[str, Any]] = {}
    for domain in sorted(set(source_domains.values())):
        tokens = sorted(token for token, value in source_domains.items() if value == domain)
        if not tokens:
            raise SelectionEvaluationError(f"source domain {domain} has no selected frames")
        subset = {
            "schema_version": request["schema_version"],
            "ground_truth": {token: ground_truth[token] for token in tokens},
            "predictions": {"results": {token: predictions[token] for token in tokens}},
        }
        path = output_root / f"official-input.{domain}.json"
        _atomic_bytes(path, canonical_json_bytes(subset) + b"\n")
        evaluation = evaluate_official(path, project_root)
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("TOP_lt"), int | float):
            raise SelectionEvaluationError(f"official {domain} evaluation is incomplete")
        result[domain] = {
            "frame_count": len(tokens),
            "TOP_lt": float(metrics["TOP_lt"]),
            "official_evaluation": evaluation,
        }
    return result


def _evaluate_selected_checkpoint_in_place(
    *,
    experiment: Experiment,
    run_root: Path,
    adapter: OpenLaneAdapter,
    isolation: PartitionIsolation,
    base_profile: E0Profile,
    e1_profile: E1Profile | None,
    linker: IndependentLinkerArtifact | None,
    project_root: Path,
    artifact_root: Path,
    output_root: Path,
    source_commit: str,
    device_name: str | None,
) -> Mapping[str, Any]:
    """Evaluate one irreversibly selected checkpoint with official and custom metrics."""
    if isolation.partition != "model_selection":
        raise SelectionEvaluationError("selected evaluation requires model_selection")
    selection_path = run_root / "selection-receipt.json"
    try:
        selection = load_json_object_path(
            selection_path,
            "checkpoint selection receipt",
            ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
        )
        selected = selection["selected"]
        if not isinstance(selected, dict) or not isinstance(selected.get("epoch"), int):
            raise ValueError("selected epoch is invalid")
        if (
            selection.get("state") != "SELECTED_ON_MODEL_SELECTION"
            or selection.get("selection_split_manifest_sha256") != isolation.split_manifest_sha256
        ):
            raise ValueError("selection identity differs")
        epoch = int(selected["epoch"])
    except (KeyError, ParseBoundaryError, TypeError, ValueError) as error:
        raise SelectionEvaluationError("checkpoint selection receipt is invalid") from error
    checkpoint_path = run_root / "checkpoints" / f"epoch-{epoch:02d}.pt"
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if checkpoint_sha256 != selection.get("checkpoint_sha256"):
        raise SelectionEvaluationError("selected checkpoint failed hash verification")
    if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
        raise SelectionEvaluationError("selected evaluation staging directory is invalid")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SelectionEvaluationError("CUDA selected evaluation was requested but unavailable")
    model = _load_model(checkpoint_path, experiment, base_profile, e1_profile, device)
    seed = _run_seed(run_root, experiment)
    ground_truth: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    nll_values: list[float] = []
    graph_truth_root = output_root / "paired/ground-truth"
    graph_prediction_root = output_root / "paired/predictions"
    graph_truth_root.mkdir(parents=True)
    graph_prediction_root.mkdir(parents=True)
    prediction_records = []
    source_domains: dict[str, str] = {}
    configuration_sha256 = (
        base_profile.canonical_sha256()
        if experiment == "E0-independent"
        else cast(E1Profile, e1_profile).canonical_sha256()
    )
    with torch.inference_mode():
        for frame_index, frame in enumerate(iter_partition_frames(adapter, isolation)):
            inputs = tuple(
                value.to(device) for value in frame_inputs(adapter, frame, base_profile).tensors()
            )
            raw = model(*inputs)
            named = (
                e0_outputs_by_name(raw)
                if experiment == "E0-independent"
                else e1_outputs_by_name(raw)
            )
            targets = _move_targets(frame_targets(frame, base_profile), device)
            matches = match_e0(named, targets)
            nll_values.append(selection_nll(named, targets, matches))
            arrays = {
                name: value.detach().to("cpu", dtype=torch.float32).numpy()
                for name, value in named.items()
            }
            truth, prediction = official_annotations(
                frame,
                arrays,
                experiment=experiment,
                linker=linker,
            )
            token = f"{frame.key.segment_id}/{frame.key.timestamp_ns}"
            ground_truth[token] = {"annotation": truth}
            predictions[token] = {"predictions": prediction}
            source_domains[token] = frame.key.source_domain
            truth_envelope = ground_truth_envelope(
                frame,
                adapter.root,
                source_commit=source_commit,
                configuration_sha256=configuration_sha256,
            )
            prediction_envelope = reference_postprocess(
                _prediction_outputs(arrays, frame, experiment, linker),
                sensor_frame(frame, adapter.root),
                producer_info(
                    source_commit=source_commit,
                    model_sha256=checkpoint_sha256,
                    configuration_sha256=configuration_sha256,
                    provider_profile=f"pytorch-{device.type}-model-selection",
                    seed=seed,
                ),
            )
            filename = f"{frame_index:06d}.pb"
            truth_path = graph_truth_root / filename
            prediction_path = graph_prediction_root / filename
            truth_path.write_bytes(truth_envelope.SerializeToString(deterministic=True))
            prediction_path.write_bytes(prediction_envelope.SerializeToString(deterministic=True))
            prediction_records.append(
                {
                    "frame_token": token,
                    "source_domain": frame.key.source_domain,
                    "frame_manifest_sha256": frame.key.frame_manifest_sha256,
                    "logical_graph_sha256": canonical_logical_sha256(prediction_envelope),
                    "payload_sha256": _sha256_file(prediction_path),
                }
            )
    if not nll_values:
        raise SelectionEvaluationError("model-selection partition contains no frames")
    request = {
        "schema_version": "junctionlens.official-evaluator-input.v1",
        "ground_truth": ground_truth,
        "predictions": {"results": predictions},
    }
    request_path = output_root / "official-input.json"
    _atomic_bytes(request_path, canonical_json_bytes(request) + b"\n")
    official = evaluate_official(request_path, project_root)
    source_domain_metrics = _source_domain_evaluations(
        request=request,
        source_domains=source_domains,
        output_root=output_root,
        project_root=project_root,
    )
    pairs = load_graph_pairs(graph_truth_root, graph_prediction_root)
    graph_source_domains = {
        pair.frame_token: source_domains[
            f"{pair.ground_truth.graph.frame_key.segment_id}/{pair.timestamp_ns}"
        ]
        for pair in pairs
    }
    match_artifact = run_custom_match(build_custom_match_request(pairs), project_root)
    frame_metrics, _segment_metrics = compute_custom_metrics(pairs, match_artifact)
    custom_ratios = {
        name: _custom_ratio(frame_metrics, name)
        for name in (
            "control_edge_recall",
            "wrong_control_assignment_rate",
            "reachability_recall_h3",
            "path_blocking_rate_h3",
        )
    }
    existence_ece, calibration_bins = _existence_ece(pairs, match_artifact)
    measured_example = _measured_example(frame_metrics, graph_source_domains)
    custom = evaluate_custom(
        graph_truth_root,
        graph_prediction_root,
        artifact_root,
        project_root,
    )
    official_metrics = official.get("metrics")
    if not isinstance(official_metrics, dict) or any(
        not isinstance(official_metrics.get(name), int | float)
        for name in ("DET_l", "DET_t", "DET_a", "TOP_ll", "TOP_lt", "OLUS")
    ):
        raise SelectionEvaluationError("official selected evaluation is incomplete")
    metrics = {
        "schema_version": "junctionlens.selected-model-metrics.v1",
        "experiment_id": experiment,
        "source_partition": "model_selection",
        "DET_l": float(official_metrics["DET_l"]),
        "DET_t": float(official_metrics["DET_t"]),
        "DET_a": float(official_metrics["DET_a"]),
        "TOP_ll": float(official_metrics["TOP_ll"]),
        "TOP_lt": float(official_metrics["TOP_lt"]),
        **custom_ratios,
        "adaptive_ece_15": existence_ece,
        "official_composite": float(official_metrics["OLUS"]),
        "negative_log_likelihood": sum(nll_values) / len(nll_values),
        "official_evaluation": official,
        "custom_evaluation": asdict(custom),
        "source_domain_metrics": source_domain_metrics,
        "calibration_bins": calibration_bins,
        "measured_example": measured_example,
    }
    metrics_path = output_root / "metrics.json"
    _atomic_bytes(metrics_path, canonical_json_bytes(metrics) + b"\n")
    prediction_manifest = {
        "schema_version": "junctionlens.private-prediction-manifest.v1",
        "experiment_id": experiment,
        "checkpoint_sha256": checkpoint_sha256,
        "selection_split_manifest_sha256": isolation.split_manifest_sha256,
        "records": prediction_records,
    }
    prediction_manifest_path = output_root / "prediction-manifest.json"
    _atomic_bytes(
        prediction_manifest_path,
        canonical_json_bytes(prediction_manifest) + b"\n",
    )
    result = {
        "schema_version": "junctionlens.selected-model-evaluation.v1",
        "state": "ACCEPTED",
        "experiment_id": experiment,
        "seed": seed,
        "source_partition": "model_selection",
        "internal_holdout_access_count": 0,
        "selection_split_manifest_sha256": isolation.split_manifest_sha256,
        "source_frame_manifest_sha256": isolation.source_frame_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "selection_receipt_sha256": _sha256_file(selection_path),
        "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
        "evaluation_artifact_sha256": _sha256_file(metrics_path),
        "metrics": {
            name: metrics[name]
            for name in (
                "DET_l",
                "DET_t",
                "TOP_lt",
                "wrong_control_assignment_rate",
                "official_composite",
                "negative_log_likelihood",
            )
        },
        "freeze_metrics": {
            "overall.DET_l": float(official_metrics["DET_l"]),
            "overall.DET_t": float(official_metrics["DET_t"]),
            "overall.DET_a": float(official_metrics["DET_a"]),
            "overall.TOP_ll": float(official_metrics["TOP_ll"]),
            "overall.TOP_lt": float(official_metrics["TOP_lt"]),
            "overall.control_edge_recall": custom_ratios["control_edge_recall"],
            "overall.wrong_control_assignment_rate": custom_ratios["wrong_control_assignment_rate"],
            "overall.reachability_recall_h3": custom_ratios["reachability_recall_h3"],
            "overall.path_blocking_rate_h3": custom_ratios["path_blocking_rate_h3"],
            "overall.adaptive_ece_15": existence_ece,
            "argoverse2.TOP_lt": float(source_domain_metrics["argoverse2"]["TOP_lt"]),
            "nuscenes.TOP_lt": float(source_domain_metrics["nuscenes"]["TOP_lt"]),
        },
        "measured_example": measured_example,
    }
    _atomic_bytes(
        output_root / "selected-evaluation.json",
        canonical_json_bytes(result) + b"\n",
    )
    return result


def evaluate_selected_checkpoint(
    *,
    experiment: Experiment,
    run_root: Path,
    adapter: OpenLaneAdapter,
    isolation: PartitionIsolation,
    base_profile: E0Profile,
    e1_profile: E1Profile | None,
    linker: IndependentLinkerArtifact | None,
    project_root: Path,
    artifact_root: Path,
    output_root: Path,
    source_commit: str,
    device_name: str | None,
) -> Mapping[str, Any]:
    """Atomically install one complete selected-checkpoint evidence directory."""
    if output_root.exists() or output_root.is_symlink():
        raise SelectionEvaluationError("selected evaluation output already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        result = _evaluate_selected_checkpoint_in_place(
            experiment=experiment,
            run_root=run_root,
            adapter=adapter,
            isolation=isolation,
            base_profile=base_profile,
            e1_profile=e1_profile,
            linker=linker,
            project_root=project_root,
            artifact_root=artifact_root,
            output_root=staging,
            source_commit=source_commit,
            device_name=device_name,
        )
        staging.replace(output_root)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "SelectionEvaluationError",
    "evaluate_selected_checkpoint",
    "fit_e0_linker",
    "load_e0_linker",
    "official_annotations",
    "resolve_source_commit",
    "score_checkpoints",
    "selection_nll",
]

"""CustomMatchV1 orchestration and exact graph and temporal KPI definitions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from platformdirs import user_cache_path

from junctionlens.contract import canonical_logical_sha256, parse_binary
from junctionlens.geometry import resample_polyline
from junctionlens.registry import ContentAddressedStore
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object,
    load_yaml_object,
    read_bounded_file,
)
from junctionlens.v1 import scene_control_graph_pb2 as scg

from .official import (
    EvaluationError,
    evaluator_container_command,
    inspect_evaluator_image,
    load_evaluator_image_contract,
)

EDGE_THRESHOLD = 0.5
CONFIDENT_WRONG_THRESHOLD = 0.9
ENDPOINT_GAP_THRESHOLD_M = 2.0
TEMPORAL_GEOMETRY_POINTS = 20
_OBJECT_TYPES = ("area", "lane_segment", "traffic_element")
_THRESHOLDS = {"area": 1.0, "lane_segment": 2.0, "traffic_element": 0.75}
_QUANTIZATION = {"area": 0.001, "lane_segment": 0.001, "traffic_element": 0.000001}


@dataclass(frozen=True, slots=True)
class GraphPair:
    """One schema-validated ground-truth and prediction frame pair."""

    frame_token: str
    segment_token: str
    timestamp_ns: int
    ground_truth: scg.SceneControlGraphEnvelope
    prediction: scg.SceneControlGraphEnvelope


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One metric value with enough population evidence for safe aggregation."""

    name: str
    value: float | None
    numerator: float | None
    denominator: float | None
    support: int
    status: str
    samples: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class CustomEvaluationReceipt:
    """Immutable identities produced by one complete custom evaluation."""

    match_manifest_sha256: str
    match_payload_sha256: str
    frame_table_manifest_sha256: str
    frame_table_payload_sha256: str
    segment_table_manifest_sha256: str
    segment_table_payload_sha256: str


def _frame_token(frame_key: scg.FrameKey) -> str:
    return json.dumps(
        {
            "dataset_id": frame_key.dataset_id,
            "dataset_version": frame_key.dataset_version,
            "segment_id": frame_key.segment_id,
            "split_id": frame_key.split_id,
            "timestamp_ns": frame_key.timestamp_ns,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _segment_token(frame_key: scg.FrameKey) -> str:
    return json.dumps(
        {
            "dataset_id": frame_key.dataset_id,
            "dataset_version": frame_key.dataset_version,
            "segment_id": frame_key.segment_id,
            "split_id": frame_key.split_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_role(root: Path, role: int) -> dict[str, scg.SceneControlGraphEnvelope]:
    resolved = root.resolve(strict=True)
    paths = [resolved] if resolved.is_file() else sorted(resolved.rglob("*.pb"))
    result: dict[str, scg.SceneControlGraphEnvelope] = {}
    for path in paths:
        envelope = parse_binary(path.read_bytes())
        if envelope.graph.role != role:
            continue
        token = _frame_token(envelope.graph.frame_key)
        if token in result:
            raise EvaluationError(f"duplicate graph role for frame {token}")
        result[token] = envelope
    if not result:
        raise EvaluationError(f"no role {role} graph envelopes found below {resolved}")
    return result


def load_graph_pairs(ground_truth_root: Path, prediction_root: Path) -> tuple[GraphPair, ...]:
    """Load exact frame pairs from validated Protobuf roots without ID equality matching."""
    ground_truth = _load_role(ground_truth_root, scg.GRAPH_ROLE_GROUND_TRUTH)
    predictions = _load_role(prediction_root, scg.GRAPH_ROLE_PREDICTION)
    if set(ground_truth) != set(predictions):
        missing = sorted(set(ground_truth) - set(predictions))
        extra = sorted(set(predictions) - set(ground_truth))
        raise EvaluationError(f"frame sets differ; missing={missing[:3]!r} extra={extra[:3]!r}")
    result = []
    for token in sorted(ground_truth):
        truth = ground_truth[token]
        prediction = predictions[token]
        if truth.graph.frame_key != prediction.graph.frame_key:
            raise EvaluationError(f"frame key content differs for {token}")
        result.append(
            GraphPair(
                frame_token=token,
                segment_token=_segment_token(truth.graph.frame_key),
                timestamp_ns=truth.graph.frame_key.timestamp_ns,
                ground_truth=truth,
                prediction=prediction,
            )
        )
    return tuple(sorted(result, key=lambda item: (item.segment_token, item.timestamp_ns)))


def _points(polyline: scg.Polyline3d) -> list[list[float]]:
    return [[point.x, point.y, point.z] for point in polyline.points]


def _source_id(node: Any, label: str) -> str:
    if not node.HasField("adapter_metadata") or not node.adapter_metadata.source_object_id:
        raise EvaluationError(f"{label} lacks a lossless source object ID")
    return cast(str, node.adapter_metadata.source_object_id)


def _node_geometry(node: Any, object_type: str) -> dict[str, object]:
    if object_type == "lane_segment":
        return {
            "centerline": _points(node.centerline),
            "left_laneline": _points(node.left_boundary),
            "right_laneline": _points(node.right_boundary),
        }
    if object_type == "traffic_element":
        box = node.normalized_half_open_box
        return {"points": [[box.x_min, box.y_min], [box.x_max, box.y_max]]}
    return {"points": _points(node.geometry)}


def _node_collections(graph: scg.SceneControlGraph) -> Mapping[str, Any]:
    return {
        "area": graph.road_areas,
        "lane_segment": graph.lanes,
        "traffic_element": graph.traffic_controls,
    }


def build_custom_match_request(pairs: Sequence[GraphPair]) -> dict[str, object]:
    """Project graph envelopes into the bounded official-primitive matching request."""
    frames: list[dict[str, object]] = []
    for pair in pairs:
        truth_collections = _node_collections(pair.ground_truth.graph)
        prediction_collections = _node_collections(pair.prediction.graph)
        truth: dict[str, object] = {}
        predictions: dict[str, object] = {}
        for object_type in _OBJECT_TYPES:
            truth[object_type] = [
                {
                    "geometry": _node_geometry(node, object_type),
                    "source_id": _source_id(node, f"{pair.frame_token} {object_type}"),
                }
                for node in truth_collections[object_type]
            ]
            predictions[object_type] = [
                {
                    "decoder_query_index": node.decoder_query_index,
                    "geometry": _node_geometry(node, object_type),
                    "prediction_id": node.node_id,
                    "raw_confidence": node.existence_confidence,
                }
                for node in prediction_collections[object_type]
            ]
        frames.append(
            {
                "frame_token": pair.frame_token,
                "ground_truth": truth,
                "predictions": predictions,
            }
        )
    return {"frames": frames, "schema_version": "junctionlens.custom-match-input.v1"}


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvaluationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationError(f"{label} must be finite")
    return number


def _quantized_request_geometry_key(
    prediction: Mapping[str, Any], object_type: str
) -> tuple[int, ...]:
    geometry = prediction["geometry"]
    keys = (
        ("centerline", "left_laneline", "right_laneline")
        if object_type == "lane_segment"
        else ("points",)
    )
    step = _QUANTIZATION[object_type]

    def quantize(value: object) -> int:
        scaled = _finite_number(value, "geometry coordinate") / step
        return math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)

    return tuple(
        quantize(component) for key in keys for point in geometry[key] for component in point
    )


def validate_custom_match_output(
    raw_output: str, request: Mapping[str, object], expected_input_sha256: str
) -> dict[str, Any]:
    """Validate all association decisions against the trusted mounted request."""
    try:
        value = load_json_object(
            raw_output.encode(),
            "CustomMatchV1 output",
            ParseLimits(
                max_bytes=64 * 1024 * 1024,
                max_depth=32,
                max_nodes=2_000_000,
                max_container_items=1_000_000,
            ),
        )
    except ParseBoundaryError as error:
        raise EvaluationError("CustomMatchV1 returned invalid JSON") from error
    if set(value) != {
        "frames",
        "input_sha256",
        "policy",
        "schema_version",
    }:
        raise EvaluationError("CustomMatchV1 output has an invalid top-level schema")
    if value["schema_version"] != "junctionlens.custom-match.v1":
        raise EvaluationError("CustomMatchV1 output schema is unsupported")
    if value["input_sha256"] != expected_input_sha256:
        raise EvaluationError("CustomMatchV1 did not echo the mounted input hash")
    expected_policy = {
        "cost_owner": "OpenLane-V2 v2.1.0 pinned official distance primitives",
        "geometry_quantization": _QUANTIZATION,
        "prediction_order": [
            "descending_raw_confidence",
            "quantized_geometry_key",
            "decoder_query_index",
        ],
        "threshold_comparison": "strictly_less_than",
        "thresholds": _THRESHOLDS,
    }
    if value["policy"] != expected_policy:
        raise EvaluationError("CustomMatchV1 policy differs from the frozen host contract")
    request_frames = {
        cast(str, frame["frame_token"]): frame
        for frame in cast(list[dict[str, Any]], request["frames"])
    }
    frames = value["frames"]
    if not isinstance(frames, dict) or set(frames) != set(request_frames):
        raise EvaluationError("CustomMatchV1 frame tokens differ from the request")
    for token, request_frame in request_frames.items():
        output_frame = frames[token]
        if not isinstance(output_frame, dict) or set(output_frame) != set(_OBJECT_TYPES):
            raise EvaluationError(f"CustomMatchV1 frame {token} is incomplete")
        for object_type in _OBJECT_TYPES:
            output_type = output_frame[object_type]
            if not isinstance(output_type, dict) or set(output_type) != {
                "ground_truth",
                "predictions",
                "threshold",
            }:
                raise EvaluationError(f"CustomMatchV1 {token} {object_type} schema is invalid")
            if output_type["threshold"] != _THRESHOLDS[object_type]:
                raise EvaluationError(f"CustomMatchV1 {token} {object_type} threshold differs")
            request_truth = request_frame["ground_truth"][object_type]
            request_predictions = request_frame["predictions"][object_type]
            truth_ids = sorted(item["source_id"] for item in request_truth)
            output_truth = output_type["ground_truth"]
            if (
                not isinstance(output_truth, list)
                or [item.get("source_id") for item in output_truth] != truth_ids
            ):
                raise EvaluationError(f"CustomMatchV1 {token} {object_type} truth IDs differ")
            records = output_type["predictions"]
            if not isinstance(records, list) or len(records) != len(request_predictions):
                raise EvaluationError(
                    f"CustomMatchV1 {token} {object_type} prediction count differs"
                )
            expected_by_id = {str(item["prediction_id"]): item for item in request_predictions}
            if {
                record.get("prediction_id") for record in records if isinstance(record, dict)
            } != set(expected_by_id):
                raise EvaluationError(f"CustomMatchV1 {token} {object_type} prediction IDs differ")
            selected: set[str] = set()
            ordering: list[tuple[float, tuple[int, ...], int]] = []
            for sorted_index, record in enumerate(records):
                if not isinstance(record, dict) or set(record) != {
                    "candidates",
                    "decoder_query_index",
                    "geometry_key",
                    "prediction_id",
                    "raw_confidence",
                    "selected_ground_truth_source_id",
                    "sorted_index",
                    "unmatched_reason",
                }:
                    raise EvaluationError(
                        f"CustomMatchV1 {token} {object_type} record schema is invalid"
                    )
                prediction = expected_by_id[record["prediction_id"]]
                if (
                    record["sorted_index"] != sorted_index
                    or record["decoder_query_index"] != prediction["decoder_query_index"]
                ):
                    raise EvaluationError(
                        f"CustomMatchV1 {token} {object_type} ordering metadata differs"
                    )
                score = _finite_number(record["raw_confidence"], "raw confidence")
                if score != prediction["raw_confidence"]:
                    raise EvaluationError(f"CustomMatchV1 {token} {object_type} confidence differs")
                key = record["geometry_key"]
                if not isinstance(key, list) or not all(
                    isinstance(item, int) and not isinstance(item, bool) for item in key
                ):
                    raise EvaluationError(
                        f"CustomMatchV1 {token} {object_type} geometry key is invalid"
                    )
                if tuple(key) != _quantized_request_geometry_key(prediction, object_type):
                    raise EvaluationError(
                        f"CustomMatchV1 {token} {object_type} geometry key differs "
                        "from the trusted request"
                    )
                ordering.append((-score, tuple(key), record["decoder_query_index"]))
                candidates = record["candidates"]
                if not isinstance(candidates, list) or {
                    item.get("ground_truth_source_id")
                    for item in candidates
                    if isinstance(item, dict)
                } != set(truth_ids):
                    raise EvaluationError(
                        f"CustomMatchV1 {token} {object_type} candidates are incomplete"
                    )
                previous: tuple[float, str] | None = None
                candidate_by_id: dict[str, dict[str, Any]] = {}
                for candidate in candidates:
                    if not isinstance(candidate, dict) or set(candidate) != {
                        "cost",
                        "ground_truth_source_id",
                        "passes_threshold",
                        "rejection_reason",
                    }:
                        raise EvaluationError(
                            f"CustomMatchV1 {token} {object_type} candidate schema is invalid"
                        )
                    source_id = candidate["ground_truth_source_id"]
                    cost = _finite_number(candidate["cost"], "matching cost")
                    if cost < 0.0 or candidate["passes_threshold"] is not (
                        cost < _THRESHOLDS[object_type]
                    ):
                        raise EvaluationError(
                            f"CustomMatchV1 {token} {object_type} threshold decision is invalid"
                        )
                    order_key = (cost, source_id)
                    if previous is not None and order_key < previous:
                        raise EvaluationError(
                            f"CustomMatchV1 {token} {object_type} candidates are not cost ordered"
                        )
                    previous = order_key
                    candidate_by_id[source_id] = candidate
                selected_id = record["selected_ground_truth_source_id"]
                expected_selected_id = next(
                    (
                        candidate["ground_truth_source_id"]
                        for candidate in candidates
                        if candidate["passes_threshold"]
                        and candidate["ground_truth_source_id"] not in selected
                    ),
                    None,
                )
                if selected_id != expected_selected_id:
                    raise EvaluationError(
                        f"CustomMatchV1 {token} {object_type} greedy selection is invalid"
                    )
                for candidate in candidates:
                    source_id = candidate["ground_truth_source_id"]
                    if source_id == expected_selected_id:
                        expected_reason = "SELECTED"
                    elif not candidate["passes_threshold"]:
                        expected_reason = "OUTSIDE_THRESHOLD"
                    elif source_id in selected:
                        expected_reason = "GROUND_TRUTH_ALREADY_MATCHED"
                    else:
                        expected_reason = "HIGHER_COST_THAN_SELECTED"
                    if candidate["rejection_reason"] != expected_reason:
                        raise EvaluationError(
                            f"CustomMatchV1 {token} {object_type} pair reason is invalid"
                        )
                if selected_id is not None:
                    if (
                        selected_id in selected
                        or candidate_by_id.get(selected_id, {}).get("rejection_reason")
                        != "SELECTED"
                    ):
                        raise EvaluationError(
                            f"CustomMatchV1 {token} {object_type} selection is not one-to-one"
                        )
                    selected.add(selected_id)
                    if record["unmatched_reason"] is not None:
                        raise EvaluationError(
                            f"CustomMatchV1 {token} {object_type} selected prediction "
                            "is marked unmatched"
                        )
                elif record["unmatched_reason"] not in {
                    "ALL_ELIGIBLE_GROUND_TRUTH_TAKEN",
                    "NO_CANDIDATE_INSIDE_THRESHOLD",
                }:
                    raise EvaluationError(
                        f"CustomMatchV1 {token} {object_type} unmatched reason is invalid"
                    )
                elif record["unmatched_reason"] != (
                    "NO_CANDIDATE_INSIDE_THRESHOLD"
                    if not any(candidate["passes_threshold"] for candidate in candidates)
                    else "ALL_ELIGIBLE_GROUND_TRUTH_TAKEN"
                ):
                    raise EvaluationError(
                        f"CustomMatchV1 {token} {object_type} unmatched reason differs"
                    )
            if ordering != sorted(ordering):
                raise EvaluationError(
                    f"CustomMatchV1 {token} {object_type} prediction order is invalid"
                )
            truth_selected = {
                item["source_id"] for item in output_truth if item.get("unmatched_reason") is None
            }
            if truth_selected != selected:
                raise EvaluationError(f"CustomMatchV1 {token} {object_type} truth status differs")
    return value


def run_custom_match(request: Mapping[str, object], root: Path) -> dict[str, Any]:
    """Run CustomMatchV1 in the digest-verified, networkless evaluator container."""
    root = root.resolve(strict=True)
    raw_bytes = canonical_json_bytes(request)
    contract = load_evaluator_image_contract(root)
    reference = str(contract["local_reference"])
    inspect_evaluator_image(
        root,
        reference,
        str(contract["config_sha256"]),
        str(contract["platform_manifest_sha256"]),
    )
    staging_override = os.environ.get("JUNCTIONLENS_DOCKER_STAGING_ROOT")
    cache_root = (
        Path(staging_override).expanduser()
        if staging_override is not None
        else user_cache_path("junctionlens") / "evaluator-inputs"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="custom-match-", dir=cache_root) as temporary:
        mounted = Path(temporary) / "request.json"
        mounted.write_bytes(raw_bytes)
        command = evaluator_container_command(
            shutil.which("docker") or "docker",
            reference,
            mounted,
            ("--custom-match", "/input/request.json"),
        )
        result = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        raise EvaluationError(
            f"CustomMatchV1 failed with exit code {result.returncode}: "
            f"{result.stderr[:2048].strip()}"
        )
    return validate_custom_match_output(
        result.stdout,
        request,
        hashlib.sha256(raw_bytes).hexdigest(),
    )


def _ratio(name: str, numerator: float, denominator: float) -> MetricValue:
    if denominator == 0.0:
        return MetricValue(name, None, numerator, denominator, 0, "EMPTY_DENOMINATOR")
    return MetricValue(
        name, numerator / denominator, numerator, denominator, int(denominator), "OK"
    )


def _quantile(samples: Sequence[float], probability: float) -> float:
    if not samples:
        raise EvaluationError("cannot calculate a quantile of an empty population")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(name: str, samples: Sequence[float], probability: float) -> MetricValue:
    values = tuple(float(item) for item in samples)
    if not values:
        return MetricValue(name, None, None, 0.0, 0, "EMPTY_DENOMINATOR")
    return MetricValue(
        name, _quantile(values, probability), None, float(len(values)), len(values), "OK", values
    )


def _associations(
    match_frame: Mapping[str, Any], object_type: str
) -> tuple[dict[int, str], dict[str, int]]:
    prediction_to_truth: dict[int, str] = {}
    truth_to_prediction: dict[str, int] = {}
    for record in match_frame[object_type]["predictions"]:
        source_id = record["selected_ground_truth_source_id"]
        if source_id is not None:
            prediction_id = int(record["prediction_id"])
            prediction_to_truth[prediction_id] = source_id
            truth_to_prediction[source_id] = prediction_id
    return prediction_to_truth, truth_to_prediction


def _truth_nodes(graph: scg.SceneControlGraph, object_type: str) -> dict[str, Any]:
    return {_source_id(node, object_type): node for node in _node_collections(graph)[object_type]}


def _prediction_nodes(graph: scg.SceneControlGraph, object_type: str) -> dict[int, Any]:
    return {node.node_id: node for node in _node_collections(graph)[object_type]}


def _edge_sets(
    graph: scg.SceneControlGraph, edge_type: int, *, predicted: bool
) -> dict[tuple[int, int], Any]:
    result: dict[tuple[int, int], Any] = {}
    for edge in graph.edges:
        if edge.edge_type != edge_type:
            continue
        if predicted and edge.raw_probability < EDGE_THRESHOLD:
            continue
        result[(edge.source_node_id, edge.target_node_id)] = edge
    return result


def _reachability(edges: Iterable[tuple[str, str]], maximum_hops: int = 3) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for source, target in edges:
        adjacency[source].add(target)
        nodes.update((source, target))
    result: dict[str, set[str]] = {}
    for source in nodes:
        reached: set[str] = set()
        frontier = {source}
        for _ in range(maximum_hops):
            frontier = {target for node in frontier for target in adjacency[node]} - reached
            reached.update(frontier)
        reached.discard(source)
        result[source] = reached
    return result


def _frame_graph_metrics(
    pair: GraphPair, match_frame: Mapping[str, Any]
) -> tuple[MetricValue, ...]:
    truth = pair.ground_truth.graph
    prediction = pair.prediction.graph
    pred_lane_to_truth, truth_lane_to_pred = _associations(match_frame, "lane_segment")
    pred_control_to_truth, truth_control_to_pred = _associations(match_frame, "traffic_element")
    truth_lanes = _truth_nodes(truth, "lane_segment")
    truth_controls = _truth_nodes(truth, "traffic_element")
    predicted_lanes = _prediction_nodes(prediction, "lane_segment")
    ground_successors_raw = _edge_sets(truth, scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR, predicted=False)
    predicted_successors_raw = _edge_sets(
        prediction, scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR, predicted=True
    )
    ground_controls_raw = _edge_sets(
        truth, scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE, predicted=False
    )
    predicted_controls_raw = _edge_sets(
        prediction, scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE, predicted=True
    )
    truth_lane_by_id = {node.node_id: source_id for source_id, node in truth_lanes.items()}
    truth_control_by_id = {node.node_id: source_id for source_id, node in truth_controls.items()}
    ground_control_edges = {
        (truth_control_by_id[source], truth_lane_by_id[target])
        for source, target in ground_controls_raw
    }
    eligible_controls = {
        control
        for control in truth_control_to_pred
        if any(
            source == control and lane in truth_lane_to_pred
            for source, lane in ground_control_edges
        )
    }
    mapped_predicted_control_edges = {
        (pred_control_to_truth[source], pred_lane_to_truth[target])
        for source, target in predicted_controls_raw
        if source in pred_control_to_truth
        and target in pred_lane_to_truth
        and pred_control_to_truth[source] in eligible_controls
    }
    recovered = mapped_predicted_control_edges & ground_control_edges
    recall_population = {
        edge
        for edge in ground_control_edges
        if edge[0] in truth_control_to_pred and edge[1] in truth_lane_to_pred
    }
    results = [
        _ratio(
            "control_edge_precision",
            float(len(recovered)),
            float(len(mapped_predicted_control_edges)),
        ),
        _ratio(
            "control_edge_recall",
            float(len(recovered & recall_population)),
            float(len(recall_population)),
        ),
    ]
    wrong = 0
    confident_wrong = 0
    wrong_denominator = 0
    for truth_control_id in sorted(eligible_controls):
        prediction_control_id = truth_control_to_pred[truth_control_id]
        outgoing = [
            edge
            for (source, _), edge in predicted_controls_raw.items()
            if source == prediction_control_id
        ]
        if not outgoing:
            continue
        wrong_denominator += 1
        top = min(
            outgoing,
            key=lambda edge: (-edge.raw_probability, edge.target_node_id),
        )
        mapped_lane = pred_lane_to_truth.get(top.target_node_id)
        is_wrong = (truth_control_id, mapped_lane) not in ground_control_edges
        if is_wrong:
            wrong += 1
            confident_wrong += int(top.calibrated_probability >= CONFIDENT_WRONG_THRESHOLD)
    results.extend(
        (
            _ratio("wrong_control_assignment_rate", float(wrong), float(wrong_denominator)),
            _ratio(
                "confident_wrong_control_rate",
                float(confident_wrong),
                float(wrong_denominator),
            ),
        )
    )
    ground_successors = {
        (truth_lane_by_id[source], truth_lane_by_id[target])
        for source, target in ground_successors_raw
    }
    mapped_predicted_successors = {
        (pred_lane_to_truth[source], pred_lane_to_truth[target])
        for source, target in predicted_successors_raw
        if source in pred_lane_to_truth and target in pred_lane_to_truth
    }
    ground_reachable = _reachability(ground_successors)
    predicted_reachable = _reachability(mapped_predicted_successors)
    reachable_population = {
        (source, target)
        for source, targets in ground_reachable.items()
        for target in targets
        if source in truth_lane_to_pred and target in truth_lane_to_pred
    }
    retained = sum(
        target in predicted_reachable.get(source, set()) for source, target in reachable_population
    )
    continuation_sources = {source for source, _ in reachable_population}
    blocked = sum(
        not bool(predicted_reachable.get(source, set()) & ground_reachable[source])
        for source in continuation_sources
    )
    results.extend(
        (
            _ratio("reachability_recall_h3", float(retained), float(len(reachable_population))),
            _ratio("path_blocking_rate_h3", float(blocked), float(len(continuation_sources))),
        )
    )
    endpoint_gaps = []
    for source, target in predicted_successors_raw:
        source_points = predicted_lanes[source].centerline.points
        target_points = predicted_lanes[target].centerline.points
        delta = (
            source_points[-1].x - target_points[0].x,
            source_points[-1].y - target_points[0].y,
            source_points[-1].z - target_points[0].z,
        )
        endpoint_gaps.append(math.sqrt(sum(item * item for item in delta)))
    results.extend(
        (
            _distribution("successor_endpoint_gap_m_median", endpoint_gaps, 0.5),
            _distribution("successor_endpoint_gap_m_p90", endpoint_gaps, 0.9),
            _distribution("successor_endpoint_gap_m_p95", endpoint_gaps, 0.95),
            _ratio(
                "successor_endpoint_gap_over_threshold_rate",
                float(sum(value > ENDPOINT_GAP_THRESHOLD_M for value in endpoint_gaps)),
                float(len(endpoint_gaps)),
            ),
            _ratio(
                "spurious_successor_rate",
                float(len(mapped_predicted_successors - ground_successors)),
                float(len(mapped_predicted_successors)),
            ),
        )
    )
    return tuple(results)


def _pose_matrix(graph: scg.SceneControlGraph) -> Any:
    import numpy as np

    if not graph.HasField("sensor_frame") or not graph.sensor_frame.pose_valid:
        raise EvaluationError("geometry jitter requires a valid ego pose")
    values = graph.sensor_frame.t_world_vehicle.values
    if len(values) != 16:
        raise EvaluationError("geometry jitter requires one 4x4 ego pose")
    return np.asarray(values, dtype=np.float64).reshape((4, 4))


def _aligned_residual(
    previous_truth: Any,
    current_truth: Any,
    previous_prediction: Any,
    current_prediction: Any,
    previous_truth_graph: scg.SceneControlGraph,
    current_truth_graph: scg.SceneControlGraph,
    object_type: str,
) -> float:
    import numpy as np

    previous_pose = _pose_matrix(previous_truth_graph)
    current_pose = _pose_matrix(current_truth_graph)
    previous_to_current = np.linalg.inv(current_pose) @ previous_pose

    def geometry(node: Any) -> Any:
        polyline = node.centerline if object_type == "lane_segment" else node.geometry
        return resample_polyline(
            np.asarray(_points(polyline), dtype=np.float64), TEMPORAL_GEOMETRY_POINTS
        )

    def align(points: Any) -> Any:
        homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
        return (previous_to_current @ homogeneous.T).T[:, :3]

    truth_change = geometry(current_truth) - align(geometry(previous_truth))
    prediction_change = geometry(current_prediction) - align(geometry(previous_prediction))
    residual = prediction_change - truth_change
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _consecutive_runs(frames: Sequence[GraphPair], object_type: str) -> dict[str, list[list[int]]]:
    present: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(frames):
        for source_id in _truth_nodes(pair.ground_truth.graph, object_type):
            present[source_id].append(index)
    result: dict[str, list[list[int]]] = {}
    for source_id, indices in present.items():
        runs: list[list[int]] = []
        for index in indices:
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        result[source_id] = runs
    return result


def _temporal_metrics(
    frames: Sequence[GraphPair], matches: Mapping[str, Any]
) -> tuple[dict[str, tuple[MetricValue, ...]], tuple[MetricValue, ...]]:
    frame_results: dict[str, list[MetricValue]] = {frame.frame_token: [] for frame in frames}
    presence_numerator = presence_denominator = 0
    for object_type in _OBJECT_TYPES:
        runs = _consecutive_runs(frames, object_type)
        for source_id, source_runs in runs.items():
            for run in source_runs:
                if len(run) < 3:
                    continue
                states = []
                for index in run:
                    _, truth_to_prediction = _associations(
                        matches[frames[index].frame_token], object_type
                    )
                    states.append(source_id in truth_to_prediction)
                for offset, current_index in enumerate(run[1:], start=1):
                    changed = int(states[offset] != states[offset - 1])
                    presence_numerator += changed
                    presence_denominator += 1
                    frame_results[frames[current_index].frame_token].append(
                        _ratio("presence_flicker_rate", float(changed), 1.0)
                    )
    edge_totals: dict[str, list[int]] = {
        "successor_edge_flip_rate": [0, 0],
        "control_edge_flip_rate": [0, 0],
    }
    edge_specs = (
        (
            "successor_edge_flip_rate",
            scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR,
            "lane_segment",
            "lane_segment",
        ),
        (
            "control_edge_flip_rate",
            scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE,
            "traffic_element",
            "lane_segment",
        ),
    )
    jitter_samples: list[float] = []
    id_switches = 0
    eligible_tracks: set[tuple[str, str]] = set()
    last_match: dict[tuple[str, str], int] = {}
    first = frames[0]
    first_match = matches[first.frame_token]
    for object_type in _OBJECT_TYPES:
        _, first_truth_to_pred = _associations(first_match, object_type)
        first_predictions = _prediction_nodes(first.prediction.graph, object_type)
        for source_id, prediction_id in first_truth_to_pred.items():
            node = first_predictions[prediction_id]
            if node.HasField("track_id"):
                last_match[(object_type, source_id)] = node.track_id
    for current_index in range(1, len(frames)):
        previous = frames[current_index - 1]
        current = frames[current_index]
        if previous.segment_token != current.segment_token:
            continue
        previous_match = matches[previous.frame_token]
        current_match = matches[current.frame_token]
        for name, edge_type, source_type, target_type in edge_specs:
            previous_source_pred_to_truth, _ = _associations(previous_match, source_type)
            previous_target_pred_to_truth, _ = _associations(previous_match, target_type)
            current_source_pred_to_truth, _ = _associations(current_match, source_type)
            current_target_pred_to_truth, _ = _associations(current_match, target_type)

            def mapped_edges(
                pair: GraphPair,
                source_map: Mapping[int, str],
                target_map: Mapping[int, str],
                selected_edge_type: int = edge_type,
            ) -> set[tuple[str, str]]:
                return {
                    (source_map[source], target_map[target])
                    for source, target in _edge_sets(
                        pair.prediction.graph, selected_edge_type, predicted=True
                    )
                    if source in source_map and target in target_map
                }

            previous_state = mapped_edges(
                previous, previous_source_pred_to_truth, previous_target_pred_to_truth
            )
            current_state = mapped_edges(
                current, current_source_pred_to_truth, current_target_pred_to_truth
            )
            previous_truth_sources = {
                node.node_id: source_id
                for source_id, node in _truth_nodes(
                    previous.ground_truth.graph, source_type
                ).items()
            }
            previous_truth_targets = {
                node.node_id: source_id
                for source_id, node in _truth_nodes(
                    previous.ground_truth.graph, target_type
                ).items()
            }
            current_truth_sources = set(_truth_nodes(current.ground_truth.graph, source_type))
            current_truth_targets = set(_truth_nodes(current.ground_truth.graph, target_type))
            persistent_edges = {
                (previous_truth_sources[source], previous_truth_targets[target])
                for source, target in _edge_sets(
                    previous.ground_truth.graph, edge_type, predicted=False
                )
                if previous_truth_sources[source] in current_truth_sources
                and previous_truth_targets[target] in current_truth_targets
            }
            changed = sum(
                (edge in previous_state) != (edge in current_state) for edge in persistent_edges
            )
            edge_totals[name][0] += changed
            edge_totals[name][1] += len(persistent_edges)
            frame_results[current.frame_token].append(
                _ratio(name, float(changed), float(len(persistent_edges)))
            )
        for object_type in ("lane_segment", "area"):
            _, previous_truth_to_pred = _associations(previous_match, object_type)
            _, current_truth_to_pred = _associations(current_match, object_type)
            previous_truth_nodes = _truth_nodes(previous.ground_truth.graph, object_type)
            current_truth_nodes = _truth_nodes(current.ground_truth.graph, object_type)
            previous_prediction_nodes = _prediction_nodes(previous.prediction.graph, object_type)
            current_prediction_nodes = _prediction_nodes(current.prediction.graph, object_type)
            transition_samples = []
            for source_id in sorted(set(previous_truth_to_pred) & set(current_truth_to_pred)):
                if source_id not in previous_truth_nodes or source_id not in current_truth_nodes:
                    continue
                transition_samples.append(
                    _aligned_residual(
                        previous_truth_nodes[source_id],
                        current_truth_nodes[source_id],
                        previous_prediction_nodes[previous_truth_to_pred[source_id]],
                        current_prediction_nodes[current_truth_to_pred[source_id]],
                        previous.ground_truth.graph,
                        current.ground_truth.graph,
                        object_type,
                    )
                )
            jitter_samples.extend(transition_samples)
            if transition_samples:
                frame_results[current.frame_token].append(
                    _distribution("geometry_jitter_m_median", transition_samples, 0.5)
                )
        live_track_ids = {
            track.track_id
            for track in current.prediction.graph.tracks
            if track.termination_reason == scg.TRACK_TERMINATION_REASON_ACTIVE
        }
        transition_switches = 0
        transition_tracks: set[tuple[str, str]] = set()
        for object_type in _OBJECT_TYPES:
            _, current_truth_to_pred = _associations(current_match, object_type)
            current_predictions = _prediction_nodes(current.prediction.graph, object_type)
            for source_id, prediction_id in current_truth_to_pred.items():
                node = current_predictions[prediction_id]
                if not node.HasField("track_id"):
                    continue
                key = (object_type, source_id)
                previous_track = last_match.get(key)
                if previous_track is not None:
                    eligible_tracks.add(key)
                    transition_tracks.add(key)
                    if node.track_id != previous_track and previous_track in live_track_ids:
                        id_switches += 1
                        transition_switches += 1
                last_match[key] = node.track_id
        frame_results[current.frame_token].append(
            _ratio(
                "id_switches_per_100_tracks",
                float(transition_switches * 100),
                float(len(transition_tracks)),
            )
        )
    segment = [
        _ratio("presence_flicker_rate", float(presence_numerator), float(presence_denominator)),
        *(_ratio(name, float(values[0]), float(values[1])) for name, values in edge_totals.items()),
        _distribution("geometry_jitter_m_median", jitter_samples, 0.5),
        _distribution("geometry_jitter_m_p90", jitter_samples, 0.9),
        _distribution("geometry_jitter_m_p95", jitter_samples, 0.95),
        _ratio(
            "id_switches_per_100_tracks",
            float(id_switches * 100),
            float(len(eligible_tracks)),
        ),
    ]
    normalized_frame_results: dict[str, tuple[MetricValue, ...]] = {}
    temporal_names = {
        "presence_flicker_rate",
        "successor_edge_flip_rate",
        "control_edge_flip_rate",
        "geometry_jitter_m_median",
        "id_switches_per_100_tracks",
    }
    for frame in frames:
        by_name: dict[str, list[MetricValue]] = defaultdict(list)
        for metric in frame_results[frame.frame_token]:
            by_name[metric.name].append(metric)
        values = []
        for name in sorted(temporal_names):
            pieces = by_name[name]
            if not pieces:
                values.append(MetricValue(name, None, 0.0, 0.0, 0, "EMPTY_DENOMINATOR"))
            elif any(piece.samples for piece in pieces):
                samples = tuple(item for piece in pieces for item in piece.samples)
                values.append(_distribution(name, samples, 0.5))
            else:
                values.append(
                    _ratio(
                        name,
                        sum(piece.numerator or 0.0 for piece in pieces),
                        sum(piece.denominator or 0.0 for piece in pieces),
                    )
                )
        normalized_frame_results[frame.frame_token] = tuple(values)
    return normalized_frame_results, tuple(segment)


def _aggregate_segment(frame_metrics: Sequence[Sequence[MetricValue]]) -> tuple[MetricValue, ...]:
    by_name: dict[str, list[MetricValue]] = defaultdict(list)
    for metrics in frame_metrics:
        for metric in metrics:
            by_name[metric.name].append(metric)
    result = []
    for name in sorted(by_name):
        pieces = by_name[name]
        samples = tuple(value for piece in pieces for value in piece.samples)
        if samples:
            probability = 0.5 if name.endswith("median") else 0.9 if name.endswith("p90") else 0.95
            result.append(_distribution(name, samples, probability))
        else:
            result.append(
                _ratio(
                    name,
                    sum(piece.numerator or 0.0 for piece in pieces),
                    sum(piece.denominator or 0.0 for piece in pieces),
                )
            )
    return tuple(result)


def compute_custom_metrics(
    pairs: Sequence[GraphPair], match_artifact: Mapping[str, Any]
) -> tuple[dict[str, tuple[MetricValue, ...]], dict[str, tuple[MetricValue, ...]]]:
    """Compute exact frame and segment graph/temporal KPI rows from a frozen match map."""
    by_segment: dict[str, list[GraphPair]] = defaultdict(list)
    frame_metrics: dict[str, tuple[MetricValue, ...]] = {}
    segment_metrics: dict[str, tuple[MetricValue, ...]] = {}
    matches = cast(Mapping[str, Any], match_artifact["frames"])
    for pair in pairs:
        by_segment[pair.segment_token].append(pair)
        frame_metrics[pair.frame_token] = _frame_graph_metrics(pair, matches[pair.frame_token])
    for segment, segment_pairs in by_segment.items():
        ordered = sorted(segment_pairs, key=lambda item: item.timestamp_ns)
        temporal_frames, temporal_segment = _temporal_metrics(ordered, matches)
        for pair in ordered:
            frame_metrics[pair.frame_token] = (
                *frame_metrics[pair.frame_token],
                *temporal_frames[pair.frame_token],
            )
        graph_aggregate = _aggregate_segment(
            [
                tuple(
                    metric
                    for metric in frame_metrics[pair.frame_token]
                    if metric.name
                    not in {
                        "presence_flicker_rate",
                        "successor_edge_flip_rate",
                        "control_edge_flip_rate",
                        "geometry_jitter_m_median",
                        "id_switches_per_100_tracks",
                    }
                )
                for pair in ordered
            ]
        )
        segment_metrics[segment] = (*graph_aggregate, *temporal_segment)
    return frame_metrics, segment_metrics


def _metric_contract(root: Path) -> tuple[Mapping[str, Any], str]:
    path = root / "configs/metrics/v1.yaml"
    try:
        raw = read_bounded_file(path, "metric registry", 4 * 1024 * 1024)
        payload = load_yaml_object(
            raw,
            "metric registry",
            ParseLimits(max_bytes=4 * 1024 * 1024, max_depth=24, max_nodes=100_000),
        )
    except ParseBoundaryError as error:
        raise EvaluationError(str(error)) from error
    if payload.get("schema_version") != "junctionlens.metrics.v1":
        raise EvaluationError("metric registry is invalid")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise EvaluationError("metric registry has no metrics")
    return cast(Mapping[str, Any], metrics), hashlib.sha256(raw).hexdigest()


def _rows(
    pairs: Sequence[GraphPair],
    values: Mapping[str, Sequence[MetricValue]],
    contract: Mapping[str, Any],
    *,
    level: str,
) -> list[dict[str, object]]:
    pair_by_frame = {pair.frame_token: pair for pair in pairs}
    result = []
    for token in sorted(values):
        pair = pair_by_frame.get(token)
        segment_token = pair.segment_token if pair is not None else token
        for metric in sorted(values[token], key=lambda item: item.name):
            definition = contract.get(metric.name)
            if not isinstance(definition, dict):
                raise EvaluationError(f"metric {metric.name} is absent from the frozen registry")
            result.append(
                {
                    "denominator": metric.denominator,
                    "direction": definition["direction"],
                    "frame_token": pair.frame_token if pair is not None else None,
                    "level": level,
                    "metric": metric.name,
                    "numerator": metric.numerator,
                    "schema_version": "junctionlens.kpi-row.v1",
                    "segment_id": segment_token,
                    "status": metric.status,
                    "support": metric.support,
                    "timestamp_ns": pair.timestamp_ns if pair is not None else None,
                    "unit": definition["unit"],
                    "value": metric.value,
                }
            )
    return result


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise EvaluationError("custom evaluation requires the pinned analytics extra") from error
    schema = pa.schema(
        [
            ("schema_version", pa.string()),
            ("level", pa.string()),
            ("segment_id", pa.string()),
            ("frame_token", pa.string()),
            ("timestamp_ns", pa.int64()),
            ("metric", pa.string()),
            ("direction", pa.string()),
            ("unit", pa.string()),
            ("status", pa.string()),
            ("value", pa.float64()),
            ("numerator", pa.float64()),
            ("denominator", pa.float64()),
            ("support", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        data_page_version="2.0",
        version="2.6",
        write_statistics=True,
    )


def evaluate_custom(
    ground_truth_root: Path,
    prediction_root: Path,
    artifact_root: Path,
    project_root: Path,
) -> CustomEvaluationReceipt:
    """Run the complete custom path and freeze association, frame, and segment artifacts."""
    project_root = project_root.resolve(strict=True)
    pairs = load_graph_pairs(ground_truth_root, prediction_root)
    request = build_custom_match_request(pairs)
    match_artifact = run_custom_match(request, project_root)
    frame_metrics, segment_metrics = compute_custom_metrics(pairs, match_artifact)
    contract, contract_sha256 = _metric_contract(project_root)
    frame_rows = _rows(pairs, frame_metrics, contract, level="frame")
    segment_rows = _rows(pairs, segment_metrics, contract, level="segment")
    store = ContentAddressedStore(
        artifact_root,
        project_root / "schemas/artifact-manifest-v1.schema.json",
    )
    source_hashes = sorted(
        {canonical_logical_sha256(pair.ground_truth) for pair in pairs}
        | {canonical_logical_sha256(pair.prediction) for pair in pairs}
    )
    match_receipt = store.put_bytes(
        canonical_json_bytes(match_artifact),
        kind="match_map",
        media_type="application/vnd.junctionlens.custom-match+json",
        license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
        metadata={
            "frame_count": len(pairs),
            "metric_registry_sha256": contract_sha256,
            "source_graph_sha256": source_hashes,
        },
    )
    with tempfile.TemporaryDirectory(prefix="custom-kpi-", dir=store.staging_root) as temporary:
        frame_path = Path(temporary) / "frame.parquet"
        segment_path = Path(temporary) / "segment.parquet"
        _write_parquet(frame_path, frame_rows)
        _write_parquet(segment_path, segment_rows)
        metadata = {
            "match_map_payload_sha256": match_receipt.payload_sha256,
            "metric_registry_sha256": contract_sha256,
        }
        frame_receipt = store.put_file(
            frame_path,
            kind="frame_kpi_table",
            media_type="application/vnd.apache.parquet",
            license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
            metadata={**metadata, "row_count": len(frame_rows)},
            parents=(match_receipt.manifest_sha256,),
        )
        segment_receipt = store.put_file(
            segment_path,
            kind="segment_kpi_table",
            media_type="application/vnd.apache.parquet",
            license_id="LicenseRef-DerivedEvaluation-SourceRestrictionsApply",
            metadata={**metadata, "row_count": len(segment_rows)},
            parents=(match_receipt.manifest_sha256,),
        )
    return CustomEvaluationReceipt(
        match_manifest_sha256=match_receipt.manifest_sha256,
        match_payload_sha256=match_receipt.payload_sha256,
        frame_table_manifest_sha256=frame_receipt.manifest_sha256,
        frame_table_payload_sha256=frame_receipt.payload_sha256,
        segment_table_manifest_sha256=segment_receipt.manifest_sha256,
        segment_table_payload_sha256=segment_receipt.payload_sha256,
    )


__all__ = [
    "CustomEvaluationReceipt",
    "GraphPair",
    "MetricValue",
    "build_custom_match_request",
    "compute_custom_metrics",
    "evaluate_custom",
    "load_graph_pairs",
    "run_custom_match",
    "validate_custom_match_output",
]

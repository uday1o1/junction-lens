from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from junctionlens.evaluator.custom import GraphPair, compute_custom_metrics
from junctionlens.synthetic import generate_scene_frames
from junctionlens.v1 import scene_control_graph_pb2 as scg


def _source_id(node: object) -> str:
    return node.adapter_metadata.source_object_id  # type: ignore[attr-defined, no-any-return]


def _pair(frame: object, token: str | None = None) -> GraphPair:
    return GraphPair(
        frame_token=token or f"{frame.scene_kind.value}-{frame.frame_index}",
        segment_token=frame.scene_kind.value,
        timestamp_ns=frame.ground_truth.graph.frame_key.timestamp_ns,
        ground_truth=deepcopy(frame.ground_truth),
        prediction=deepcopy(frame.perfect_prediction),
    )


def _matches(pairs: tuple[GraphPair, ...]) -> dict[str, object]:
    frames: dict[str, object] = {}
    for pair in pairs:
        frame: dict[str, object] = {}
        collections = (
            ("area", pair.ground_truth.graph.road_areas, pair.prediction.graph.road_areas),
            ("lane_segment", pair.ground_truth.graph.lanes, pair.prediction.graph.lanes),
            (
                "traffic_element",
                pair.ground_truth.graph.traffic_controls,
                pair.prediction.graph.traffic_controls,
            ),
        )
        for object_type, truth, prediction in collections:
            frame[object_type] = {
                "predictions": [
                    {
                        "prediction_id": str(predicted.node_id),
                        "selected_ground_truth_source_id": _source_id(actual),
                    }
                    for actual, predicted in zip(truth, prediction, strict=True)
                ]
            }
        frames[pair.frame_token] = frame
    return {"frames": frames}


def _metrics(values: tuple[object, ...]) -> dict[str, object]:
    return {value.name: value for value in values}  # type: ignore[attr-defined]


def _scene(name: str, frame_index: int = 0) -> object:
    return next(
        frame
        for frame in generate_scene_frames()
        if frame.scene_kind.value == name and frame.frame_index == frame_index
    )


def _remove_first_edge(graph: scg.SceneControlGraph, edge_type: int) -> None:
    removed = False
    retained = []
    for edge in graph.edges:
        if edge.edge_type == edge_type and not removed:
            removed = True
            continue
        retained.append(edge)
    del graph.edges[:]
    graph.edges.extend(retained)


def test_hand_calculated_control_and_topology_goldens() -> None:
    straight = _pair(_scene("straight-control"))
    merge = _pair(_scene("merge"))
    artifact = _matches((straight, merge))
    frame, _ = compute_custom_metrics((straight, merge), artifact)
    control = _metrics(frame[straight.frame_token])
    topology = _metrics(frame[merge.frame_token])

    assert control["control_edge_precision"].value == 1.0
    assert control["control_edge_recall"].value == 1.0
    assert control["wrong_control_assignment_rate"].value == 0.0
    assert control["confident_wrong_control_rate"].value == 0.0
    assert topology["reachability_recall_h3"].value == 1.0
    assert topology["path_blocking_rate_h3"].value == 0.0
    assert topology["spurious_successor_rate"].value == 0.0
    assert topology["successor_endpoint_gap_m_median"].value == 0.75
    assert topology["successor_endpoint_gap_m_p90"].value == 0.75
    assert topology["successor_endpoint_gap_m_p95"].value == 0.75


def test_empty_denominators_are_null_and_labeled() -> None:
    straight = _pair(_scene("straight-control"))
    artifact = _matches((straight,))
    frame, segment = compute_custom_metrics((straight,), artifact)
    values = (*frame[straight.frame_token], *segment[straight.segment_token])
    empty = [value for value in values if value.denominator == 0.0]

    assert empty
    assert all(value.value is None for value in empty)
    assert all(value.status == "EMPTY_DENOMINATOR" for value in empty)


def test_control_and_path_faults_degrade_only_from_clean_control() -> None:
    intersection = _pair(_scene("intersection-crosswalk"))
    merge = _pair(_scene("merge"))
    clean_artifact = _matches((intersection, merge))
    clean_frame, _ = compute_custom_metrics((intersection, merge), clean_artifact)

    wrong_control = deepcopy(intersection)
    edge = next(
        item
        for item in wrong_control.prediction.graph.edges
        if item.edge_type == scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE
    )
    wrong_target = wrong_control.prediction.graph.lanes[1].node_id
    edge.target_node_id = wrong_target
    edge.raw_probability = 0.99
    edge.calibrated_probability = 0.99
    broken_path = deepcopy(merge)
    _remove_first_edge(
        broken_path.prediction.graph,
        scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR,
    )
    fault_artifact = _matches((wrong_control, broken_path))
    fault_frame, _ = compute_custom_metrics((wrong_control, broken_path), fault_artifact)
    clean_control = _metrics(clean_frame[intersection.frame_token])
    fault_control = _metrics(fault_frame[wrong_control.frame_token])
    clean_topology = _metrics(clean_frame[merge.frame_token])
    fault_topology = _metrics(fault_frame[broken_path.frame_token])

    assert fault_control["control_edge_recall"].value < clean_control["control_edge_recall"].value
    assert fault_control["wrong_control_assignment_rate"].value == 1.0
    assert fault_control["confident_wrong_control_rate"].value == 1.0
    assert fault_topology["reachability_recall_h3"].value == 0.5
    assert fault_topology["path_blocking_rate_h3"].value == 0.5
    assert (
        fault_topology["reachability_recall_h3"].value
        < clean_topology["reachability_recall_h3"].value
    )


def test_temporal_goldens_and_alternating_faults() -> None:
    first = _pair(_scene("straight-control", 0), "frame-0")
    second = _pair(_scene("straight-control", 1), "frame-1")
    third = deepcopy(second)
    object.__setattr__(third, "frame_token", "frame-2")
    object.__setattr__(third, "timestamp_ns", second.timestamp_ns + 100_000_000)
    pairs = (first, second, third)
    clean_artifact = _matches(pairs)
    _, clean_segment = compute_custom_metrics(pairs, clean_artifact)
    clean = _metrics(clean_segment[first.segment_token])

    flicker_artifact = deepcopy(clean_artifact)
    flicker_artifact["frames"]["frame-1"]["lane_segment"]["predictions"][0][
        "selected_ground_truth_source_id"
    ] = None
    _, fault_segment = compute_custom_metrics(pairs, flicker_artifact)
    fault = _metrics(fault_segment[first.segment_token])

    assert clean["presence_flicker_rate"].value == 0.0
    assert clean["successor_edge_flip_rate"].value is None
    assert clean["control_edge_flip_rate"].value == 0.0
    assert clean["geometry_jitter_m_median"].value == pytest.approx(0.0, abs=1e-12)
    assert clean["id_switches_per_100_tracks"].value == 0.0
    assert fault["presence_flicker_rate"].value == 0.5
    assert fault["control_edge_flip_rate"].value == 1.0


def test_live_previous_track_change_counts_one_switch() -> None:
    first = _pair(_scene("straight-control", 0), "frame-0")
    second = _pair(_scene("straight-control", 1), "frame-1")
    second.prediction.graph.lanes[0].track_id += 10
    pairs = (first, second)
    artifact = _matches(pairs)

    _, segment = compute_custom_metrics(pairs, artifact)
    values = _metrics(segment[first.segment_token])

    assert values["id_switches_per_100_tracks"].numerator == 100.0
    assert values["id_switches_per_100_tracks"].denominator == 2.0
    assert values["id_switches_per_100_tracks"].value == 50.0


@settings(max_examples=20, deadline=None)
@given(retained=st.lists(st.booleans(), min_size=2, max_size=2))
def test_ratio_properties_remain_bounded(retained: list[bool]) -> None:
    merge = _pair(_scene("merge"))
    edges = [
        edge
        for edge in merge.prediction.graph.edges
        if edge.edge_type == scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR
    ]
    retained_ids = {edge.edge_id for keep, edge in zip(retained, edges, strict=True) if keep}
    copies = [
        edge
        for edge in merge.prediction.graph.edges
        if edge.edge_type != scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR or edge.edge_id in retained_ids
    ]
    del merge.prediction.graph.edges[:]
    merge.prediction.graph.edges.extend(copies)
    artifact = _matches((merge,))
    frame, _ = compute_custom_metrics((merge,), artifact)

    for value in frame[merge.frame_token]:
        if value.name.endswith(("rate", "recall", "precision", "h3")) and value.value is not None:
            assert 0.0 <= value.value <= 1.0

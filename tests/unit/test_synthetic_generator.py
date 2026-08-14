"""Deterministic synthetic graph truth, projection, and corruption tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from junctionlens.contract import parse_binary, parse_json
from junctionlens.data.geometry import align_points_between_vehicle_frames
from junctionlens.synthetic import (
    CorruptionKind,
    GeneratedSceneFrame,
    SceneKind,
    SyntheticCorpusError,
    generate_corpus,
    generate_corruptions,
    generate_scene_frames,
    verify_corpus,
    write_corpus,
)
from junctionlens.synthetic.calibration import camera_calibrations
from junctionlens.synthetic.render import project_polyline
from junctionlens.v1 import scene_control_graph_pb2 as scg


def _frame(scene_kind: SceneKind, frame_index: int = 0) -> GeneratedSceneFrame:
    return next(
        frame
        for frame in generate_scene_frames()
        if frame.scene_kind == scene_kind and frame.frame_index == frame_index
    )


def _matrix4(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def test_every_generated_graph_encoding_passes_the_public_contract() -> None:
    corpus = generate_corpus()
    for path, payload in corpus.files.items():
        if path.endswith(".pb"):
            assert parse_binary(payload).schema_major == 1
        elif path.endswith(".json") and path != "manifest.json":
            assert parse_json(payload).schema_major == 1


def test_corpus_is_byte_identical_for_one_seed_and_changes_for_another() -> None:
    first = generate_corpus(20_260_813)
    second = generate_corpus(20_260_813)
    different = generate_corpus(20_260_814)
    assert first.files == second.files
    assert first.files["manifest.json"] != different.files["manifest.json"]
    assert (
        first.files["graphs/straight-control/frame-00.ground-truth.pb"]
        != different.files["graphs/straight-control/frame-00.ground-truth.pb"]
    )


def test_mandatory_graph_shapes_are_present_in_frozen_scene_definitions() -> None:
    corpus = generate_corpus()
    manifest = json.loads(corpus.files["manifest.json"])
    assert set(manifest["mandatory_shapes"]) == {
        "control",
        "crosswalk",
        "intersection",
        "merge",
        "road",
        "split",
        "temporal-ego-motion",
    }

    merge = _frame(SceneKind.MERGE).ground_truth.graph
    merge_in_degree: dict[int, int] = {}
    for edge in merge.edges:
        if edge.edge_type == scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR:
            merge_in_degree[edge.target_node_id] = merge_in_degree.get(edge.target_node_id, 0) + 1
    assert max(merge_in_degree.values()) == 2

    split = _frame(SceneKind.SPLIT).ground_truth.graph
    split_out_degree: dict[int, int] = {}
    for edge in split.edges:
        if edge.edge_type == scg.GRAPH_EDGE_TYPE_LANE_SUCCESSOR:
            split_out_degree[edge.source_node_id] = split_out_degree.get(edge.source_node_id, 0) + 1
    assert max(split_out_degree.values()) == 2

    intersection = _frame(SceneKind.INTERSECTION_CROSSWALK).ground_truth.graph
    assert any(lane.intersection_or_connector_probability == 1.0 for lane in intersection.lanes)
    assert len(intersection.traffic_controls) == 1
    assert len(intersection.road_areas) == 1
    assert intersection.road_areas[0].category_distribution.probabilities[0] == 1.0


def test_front_render_projection_matches_independent_analytic_truth() -> None:
    frame = _frame(SceneKind.STRAIGHT_CONTROL)
    lane = frame.ground_truth.graph.lanes[0]
    calibration = camera_calibrations()[0]
    pixels, valid = project_polyline(lane.centerline, calibration)
    first = lane.centerline.points[0]
    expected_horizontal = 320.0 - 200.0 * first.y / first.x
    expected_vertical = 192.0 + 200.0 * 1.5 / first.x
    assert valid[0]
    np.testing.assert_allclose(
        pixels[0],
        [expected_horizontal, expected_vertical],
        atol=0.25,
        rtol=0.0,
    )
    image_path = frame.ground_truth.graph.sensor_frame.cameras[0].original_image.relative_uri
    rendered = frame.camera_images[image_path].decode("utf-8")
    assert f"{expected_horizontal:.3f},{expected_vertical:.3f}" in rendered
    for camera in frame.ground_truth.graph.sensor_frame.cameras:
        image = frame.camera_images[camera.original_image.relative_uri]
        assert hashlib.sha256(image).hexdigest() == camera.original_image.sha256
        assert len(image) == camera.original_image.byte_size


def test_temporal_ego_motion_aligns_persistent_world_geometry() -> None:
    previous = _frame(SceneKind.STRAIGHT_CONTROL, 0).ground_truth.graph
    current = _frame(SceneKind.STRAIGHT_CONTROL, 1).ground_truth.graph
    previous_points = np.asarray(
        [(point.x, point.y, point.z) for point in previous.lanes[0].centerline.points]
    )
    current_points = np.asarray(
        [(point.x, point.y, point.z) for point in current.lanes[0].centerline.points]
    )
    aligned = align_points_between_vehicle_frames(
        previous_points,
        _matrix4(previous.sensor_frame.t_world_vehicle.values),
        _matrix4(current.sensor_frame.t_world_vehicle.values),
    )
    np.testing.assert_allclose(aligned, current_points, atol=1e-9, rtol=0.0)
    assert current.frame_key.timestamp_ns > previous.frame_key.timestamp_ns
    assert current.lanes[0].track_id == previous.lanes[0].track_id


def test_perfect_predictions_preserve_semantics_without_reusing_truth_ids() -> None:
    for frame in generate_scene_frames():
        truth = frame.ground_truth.graph
        prediction = frame.perfect_prediction.graph
        assert truth.role == scg.GRAPH_ROLE_GROUND_TRUTH
        assert prediction.role == scg.GRAPH_ROLE_PREDICTION
        assert len(truth.lanes) == len(prediction.lanes)
        assert len(truth.traffic_controls) == len(prediction.traffic_controls)
        assert len(truth.road_areas) == len(prediction.road_areas)
        assert [lane.node_id for lane in truth.lanes] != [lane.node_id for lane in prediction.lanes]
        for expected, observed in zip(truth.lanes, prediction.lanes, strict=True):
            assert expected.centerline == observed.centerline
            assert expected.left_boundary == observed.left_boundary
            assert expected.right_boundary == observed.right_boundary
        for expected, observed in zip(
            truth.traffic_controls,
            prediction.traffic_controls,
            strict=True,
        ):
            assert expected.normalized_half_open_box == observed.normalized_half_open_box
            assert expected.category_distribution == observed.category_distribution
        for expected, observed in zip(truth.road_areas, prediction.road_areas, strict=True):
            assert expected.geometry == observed.geometry
            assert expected.category_distribution == observed.category_distribution


def test_controlled_corruptions_change_only_the_intended_family() -> None:
    frames = generate_scene_frames()
    corruptions = {corruption.corruption: corruption for corruption in generate_corruptions(frames)}

    dropped = corruptions[CorruptionKind.DROP_CONTROL].prediction.graph
    straight = _frame(SceneKind.STRAIGHT_CONTROL).perfect_prediction.graph
    assert len(dropped.traffic_controls) == len(straight.traffic_controls) - 1
    assert dropped.lanes == straight.lanes

    broken = corruptions[CorruptionKind.BREAK_TOPOLOGY].prediction.graph
    merge = _frame(SceneKind.MERGE).perfect_prediction.graph
    assert len(broken.edges) == len(merge.edges) - 1
    assert broken.lanes == merge.lanes

    shifted = corruptions[CorruptionKind.SHIFT_LANE].prediction.graph
    intersection = _frame(SceneKind.INTERSECTION_CROSSWALK).perfect_prediction.graph
    assert shifted.traffic_controls == intersection.traffic_controls
    for shifted_point, control_point in zip(
        shifted.lanes[0].centerline.points,
        intersection.lanes[0].centerline.points,
        strict=True,
    ):
        assert shifted_point.x == control_point.x
        assert shifted_point.y == pytest.approx(control_point.y + 2.5)


def test_corpus_writer_is_idempotent_and_fails_on_stale_or_changed_files(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    expected = write_corpus(root)
    assert verify_corpus(root).files == expected.files
    assert write_corpus(root).files == expected.files

    changed = root / "graphs/merge/frame-00.perfect.pb"
    changed.write_bytes(b"changed")
    with pytest.raises(SyntheticCorpusError, match="byte mismatch"):
        verify_corpus(root)

    changed.write_bytes(expected.files["graphs/merge/frame-00.perfect.pb"])
    (root / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(SyntheticCorpusError, match="stale files"):
        write_corpus(root)


def test_committed_corpus_matches_the_generator_byte_for_byte() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures/synthetic/v1"
    assert verify_corpus(root).seed == 20_260_813

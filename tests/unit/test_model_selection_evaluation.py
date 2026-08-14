"""Model-selection conversion and topology-orientation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from junctionlens.contract.validation import validate_envelope
from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.model.evaluation_graphs import ground_truth_envelope
from junctionlens.model.selection_evaluation import official_annotations

ROOT = Path(__file__).parents[2]


def _outputs() -> dict[str, np.ndarray]:
    high = np.asarray([[10.0, 9.0]], dtype=np.float32)
    return {
        "lane_existence_logits": high,
        "lane_centerline": np.asarray(
            [[[[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 5.0, 0.0]]]],
            dtype=np.float32,
        ),
        "lane_left_boundary": np.asarray(
            [[[[-1.0, 0.0, 0.0], [-1.0, 5.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]]],
            dtype=np.float32,
        ),
        "lane_right_boundary": np.asarray(
            [[[[1.0, 0.0, 0.0], [1.0, 5.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 5.0, 0.0]]]],
            dtype=np.float32,
        ),
        "traffic_existence_logits": high,
        "traffic_boxes": np.asarray(
            [[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]], dtype=np.float32
        ),
        "traffic_attribute_logits": np.zeros((1, 2, 13), dtype=np.float32),
        "area_existence_logits": np.asarray([[10.0]], dtype=np.float32),
        "area_category_logits": np.asarray([[[5.0, 0.0]]], dtype=np.float32),
        "area_points": np.asarray(
            [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]],
            dtype=np.float32,
        ),
        "area_valid_logits": np.asarray([[[10.0, 10.0, 10.0]]], dtype=np.float32),
        "lane_successor_logits": np.asarray([[[-10.0, 4.0], [2.0, -10.0]]], dtype=np.float32),
        "control_lane_logits": np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
    }


def test_ground_truth_conversion_preserves_all_normalized_nodes(
    openlane_root: Path,
) -> None:
    adapter = OpenLaneAdapter(openlane_root, ROOT / "configs/data/openlane-v2-v2.1.adapter.yaml")
    frame = next(adapter.iter_frames("sample"))

    envelope = ground_truth_envelope(
        frame,
        openlane_root,
        source_commit="1" * 40,
        configuration_sha256="2" * 64,
    )

    validate_envelope(envelope)
    assert len(envelope.graph.lanes) == len(frame.lanes)
    assert len(envelope.graph.traffic_controls) == len(frame.traffic_controls)
    assert len(envelope.graph.road_areas) == len(frame.road_areas)
    assert envelope.graph.sensor_frame.cameras[0].original_image.sha256


def test_official_e1_export_transposes_control_major_logits(
    openlane_root: Path,
) -> None:
    adapter = OpenLaneAdapter(openlane_root, ROOT / "configs/data/openlane-v2-v2.1.adapter.yaml")
    frame = next(adapter.iter_frames("sample"))

    _truth, prediction = official_annotations(
        frame,
        _outputs(),
        experiment="E1-joint",
        linker=None,
    )

    matrix = prediction["topology_lste"]
    expected = 1.0 / (1.0 + np.exp(-np.asarray([[1.0, 3.0], [2.0, 4.0]])))
    assert np.allclose(matrix, expected)
    assert len(prediction["lane_segment"]) == 2
    assert len(prediction["traffic_element"]) == 2

"""Primitive materialization tests for paired release comparisons."""

from __future__ import annotations

from junctionlens.gate.charter import CharterCell
from junctionlens.gate.comparison import FrameEvidence, _aggregate_segments


def _frame(token: str, timestamp: int, metric: str, primitive: dict[str, object]) -> FrameEvidence:
    return FrameEvidence.model_validate(
        {
            "frame_token": token,
            "segment_id": "segment",
            "timestamp_ns": timestamp,
            "slice_values": {"source_domain": "synthetic"},
            "metrics": {metric: primitive},
        }
    )


def _cell(metric: str, estimator: str) -> CharterCell:
    return CharterCell.model_validate(
        {
            "id": f"overall.{metric}",
            "metric": metric,
            "slice": "overall",
            "direction": "higher_is_better",
            "margin": 0.0,
            "support": "overall",
            "estimator": estimator,
            "stage": "calibration" if estimator == "adaptive_ece" else "accuracy",
        }
    )


def test_average_precision_aggregation_rebuilds_ranked_frame_population() -> None:
    frames = [
        _frame(
            "frame-b",
            2,
            "DET_l",
            {
                "ground_truth_count": 2,
                "predictions": [{"id": "prediction", "confidence": 0.7, "true_positive": True}],
            },
        ),
        _frame(
            "frame-a",
            1,
            "DET_l",
            {
                "ground_truth_count": 1,
                "predictions": [{"id": "prediction", "confidence": 0.8, "true_positive": False}],
            },
        ),
    ]

    segments, edges, transitions, temporal_segments = _aggregate_segments(
        frames, _cell("DET_l", "average_precision")
    )

    assert segments[0]["ground_truth_count"] == 3
    assert [item["id"] for item in segments[0]["predictions"]] == [
        "frame-a:prediction",
        "frame-b:prediction",
    ]
    assert edges == 0
    assert transitions == 0
    assert temporal_segments == 0


def test_ratio_aggregation_sums_primitives_and_support() -> None:
    frames = [
        _frame(
            "frame-a",
            1,
            "control_edge_recall",
            {
                "numerator": 2,
                "denominator": 3,
                "eligible_ground_truth_edges": 3,
                "adjacent_frame_transitions": 1,
            },
        ),
        _frame(
            "frame-b",
            2,
            "control_edge_recall",
            {
                "numerator": 1,
                "denominator": 2,
                "eligible_ground_truth_edges": 2,
                "adjacent_frame_transitions": 1,
            },
        ),
    ]

    segments, edges, transitions, temporal_segments = _aggregate_segments(
        frames, _cell("control_edge_recall", "ratio")
    )

    assert segments == [{"segment_id": "segment", "numerator": 3.0, "denominator": 5.0}]
    assert edges == 5
    assert transitions == 2
    assert temporal_segments == 1


def test_adaptive_ece_aggregation_preserves_observations_with_frame_identity() -> None:
    frames = [
        _frame(
            "frame-a",
            1,
            "adaptive_ece_15",
            {
                "observations": [
                    {"id": "edge", "confidence": 0.9, "correct": True},
                ]
            },
        ),
        _frame(
            "frame-b",
            2,
            "adaptive_ece_15",
            {
                "observations": [
                    {"id": "edge", "confidence": 0.6, "correct": False},
                ]
            },
        ),
    ]

    segments, _, _, _ = _aggregate_segments(frames, _cell("adaptive_ece_15", "adaptive_ece"))

    assert [item["id"] for item in segments[0]["observations"]] == [
        "frame-a:edge",
        "frame-b:edge",
    ]

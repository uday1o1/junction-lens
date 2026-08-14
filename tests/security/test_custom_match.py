from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from junctionlens.evaluator.custom import validate_custom_match_output
from junctionlens.evaluator.official import EvaluationError
from junctionlens.registry.store import canonical_json_bytes


def _request() -> dict[str, object]:
    empty = {"area": [], "traffic_element": []}
    geometry = {
        "centerline": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "left_laneline": [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        "right_laneline": [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0]],
    }
    return {
        "frames": [
            {
                "frame_token": "frame",
                "ground_truth": {
                    **empty,
                    "lane_segment": [{"geometry": geometry, "source_id": "lane-a"}],
                },
                "predictions": {
                    **empty,
                    "lane_segment": [
                        {
                            "decoder_query_index": 0,
                            "geometry": geometry,
                            "prediction_id": 1,
                            "raw_confidence": 1.0,
                        }
                    ],
                },
            }
        ],
        "schema_version": "junctionlens.custom-match-input.v1",
    }


def _output(request: dict[str, object]) -> dict[str, object]:
    empty = {"ground_truth": [], "predictions": []}
    return {
        "frames": {
            "frame": {
                "area": {**empty, "threshold": 1.0},
                "lane_segment": {
                    "ground_truth": [{"source_id": "lane-a", "unmatched_reason": None}],
                    "predictions": [
                        {
                            "candidates": [
                                {
                                    "cost": 0.0,
                                    "ground_truth_source_id": "lane-a",
                                    "passes_threshold": True,
                                    "rejection_reason": "SELECTED",
                                }
                            ],
                            "decoder_query_index": 0,
                            "geometry_key": [
                                0,
                                0,
                                0,
                                1000,
                                0,
                                0,
                                0,
                                1000,
                                0,
                                1000,
                                1000,
                                0,
                                0,
                                -1000,
                                0,
                                1000,
                                -1000,
                                0,
                            ],
                            "prediction_id": "1",
                            "raw_confidence": 1.0,
                            "selected_ground_truth_source_id": "lane-a",
                            "sorted_index": 0,
                            "unmatched_reason": None,
                        }
                    ],
                    "threshold": 2.0,
                },
                "traffic_element": {**empty, "threshold": 0.75},
            }
        },
        "input_sha256": hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
        "policy": {
            "cost_owner": "OpenLane-V2 v2.1.0 pinned official distance primitives",
            "geometry_quantization": {
                "area": 0.001,
                "lane_segment": 0.001,
                "traffic_element": 0.000001,
            },
            "prediction_order": [
                "descending_raw_confidence",
                "quantized_geometry_key",
                "decoder_query_index",
            ],
            "threshold_comparison": "strictly_less_than",
            "thresholds": {
                "area": 1.0,
                "lane_segment": 2.0,
                "traffic_element": 0.75,
            },
        },
        "schema_version": "junctionlens.custom-match.v1",
    }


def test_custom_match_output_accepts_complete_one_to_one_evidence() -> None:
    request = _request()
    output = _output(request)

    assert (
        validate_custom_match_output(
            canonical_json_bytes(output).decode(),
            request,
            output["input_sha256"],
        )
        == output
    )


@pytest.mark.parametrize(
    "defect",
    ["threshold", "passes", "truth_status", "hash", "geometry_key", "reason"],
)
def test_custom_match_output_rejects_seeded_integrity_defects(defect: str) -> None:
    request = _request()
    output = deepcopy(_output(request))
    if defect == "threshold":
        output["frames"]["frame"]["lane_segment"]["threshold"] = 2.1
    elif defect == "passes":
        output["frames"]["frame"]["lane_segment"]["predictions"][0]["candidates"][0][
            "passes_threshold"
        ] = False
    elif defect == "truth_status":
        output["frames"]["frame"]["lane_segment"]["ground_truth"][0]["unmatched_reason"] = (
            "NO_PREDICTION_SELECTED"
        )
    elif defect == "geometry_key":
        output["frames"]["frame"]["lane_segment"]["predictions"][0]["geometry_key"][0] = 1
    elif defect == "reason":
        output["frames"]["frame"]["lane_segment"]["predictions"][0]["candidates"][0][
            "rejection_reason"
        ] = "HIGHER_COST_THAN_SELECTED"
    else:
        output["input_sha256"] = "0" * 64

    with pytest.raises(EvaluationError):
        validate_custom_match_output(
            canonical_json_bytes(output).decode(),
            request,
            hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
        )

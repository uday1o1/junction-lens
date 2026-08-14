"""Unit tests for paired pooled bootstrap estimators."""

from __future__ import annotations

import pytest

from junctionlens.gate.bootstrap import (
    BootstrapError,
    paired_segment_bootstrap,
    paired_trial_bootstrap,
)


def test_ratio_bootstrap_recomputes_pooled_primitives() -> None:
    baseline = {
        "segments": [
            {"segment_id": "a", "numerator": 1, "denominator": 1},
            {"segment_id": "b", "numerator": 0, "denominator": 9},
        ]
    }
    candidate = {
        "segments": [
            {"segment_id": "a", "numerator": 2, "denominator": 2},
            {"segment_id": "b", "numerator": 1, "denominator": 8},
        ]
    }

    result = paired_segment_bootstrap(
        baseline,
        candidate,
        estimator="ratio",
        direction="higher_is_better",
        family_alpha=0.05,
        gating_cells=1,
        replicates=100,
        minimum_finite_replicates=100,
    )

    assert result.point_estimate == pytest.approx(0.2)
    assert result.status == "VALID"


def test_ap_bootstrap_retains_duplicate_segment_draws() -> None:
    baseline = {
        "segments": [
            {
                "segment_id": "only",
                "ground_truth_count": 1,
                "predictions": [{"id": "p", "confidence": 0.8, "true_positive": True}],
            }
        ]
    }
    candidate = {
        "segments": [
            {
                "segment_id": "only",
                "ground_truth_count": 1,
                "predictions": [{"id": "p", "confidence": 0.9, "true_positive": True}],
            }
        ]
    }

    result = paired_segment_bootstrap(
        baseline,
        candidate,
        estimator="average_precision",
        direction="higher_is_better",
        family_alpha=0.05,
        gating_cells=1,
        replicates=20,
        minimum_finite_replicates=20,
    )

    assert result.point_estimate == 0.0
    assert result.invalid_replicates == 0


def test_segment_bootstrap_rejects_unpaired_populations() -> None:
    with pytest.raises(BootstrapError, match="exact same segment IDs"):
        paired_segment_bootstrap(
            {"segments": [{"segment_id": "a", "numerator": 1, "denominator": 1}]},
            {"segments": [{"segment_id": "b", "numerator": 1, "denominator": 1}]},
            estimator="ratio",
            direction="higher_is_better",
            family_alpha=0.05,
            gating_cells=1,
            replicates=1,
            minimum_finite_replicates=1,
        )


def test_runtime_bootstrap_uses_valid_relative_paired_blocks() -> None:
    baseline_blocks = [
        {"block_id": str(index), "value": 100.0, "valid": index != 9} for index in range(10)
    ]
    candidate_blocks = [
        {"block_id": str(index), "value": 90.0, "valid": index != 8} for index in range(10)
    ]

    result = paired_trial_bootstrap(
        {"blocks": baseline_blocks},
        {"blocks": candidate_blocks},
        direction="lower_is_better",
        family_alpha=0.05,
        gating_cells=1,
        replicates=100,
    )

    assert result.status == "VALID"
    assert result.point_estimate == pytest.approx(0.1)


def test_runtime_bootstrap_blocks_with_fewer_than_eight_pairs() -> None:
    baseline = {
        "blocks": [
            {"block_id": str(index), "value": 100.0, "valid": index < 7} for index in range(10)
        ]
    }
    candidate = {
        "blocks": [{"block_id": str(index), "value": 90.0, "valid": True} for index in range(10)]
    }

    result = paired_trial_bootstrap(
        baseline,
        candidate,
        direction="lower_is_better",
        family_alpha=0.05,
        gating_cells=1,
        replicates=100,
    )

    assert result.status == "INSUFFICIENT_FINITE"

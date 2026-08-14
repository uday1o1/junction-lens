from __future__ import annotations

import math

import pytest

from junctionlens.evaluator.evidence import (
    adaptive_ece,
    aurc,
    binary_brier,
    binary_nll,
    geometry_uncertainty,
    multiclass_brier,
    multiclass_nll,
    runtime_distributions,
)


def test_binary_calibration_analytic_fixture_and_clipping_count() -> None:
    probabilities = [0.0, 0.25, 0.75, 1.0]
    outcomes = [0, 0, 1, 1]
    nll, clipped = binary_nll(probabilities, outcomes)

    assert binary_brier(probabilities, outcomes) == 0.03125
    assert clipped == 2
    expected = (-2.0 * math.log(1.0 - 1.0e-7) - 2.0 * math.log(0.75)) / 4.0
    assert nll == pytest.approx(expected, abs=1e-15)


def test_multiclass_calibration_analytic_fixture() -> None:
    probabilities = [[0.75, 0.25], [0.1, 0.9]]
    outcomes = [0, 1]
    nll, clipped = multiclass_nll(probabilities, outcomes)

    assert multiclass_brier(probabilities, outcomes) == pytest.approx(0.0725)
    assert nll == pytest.approx(-(math.log(0.75) + math.log(0.9)) / 2.0)
    assert clipped == 0


def test_adaptive_ece_bins_and_aurc_ties_are_deterministic() -> None:
    ece, bins = adaptive_ece([0.25, 0.75], [0, 1], ["b", "a"])

    assert ece == 0.25
    assert [item.count for item in bins] == [1, 1]
    assert aurc([0.9, 0.8], [0.0, 1.0], ["a", "b"]) == 0.125
    assert aurc([0.5, 0.5], [0.0, 1.0], ["b", "a"]) == 0.625


def test_geometry_coverage_and_width_are_analytic() -> None:
    half_width = math.log(10.0)
    result = geometry_uncertainty(
        [half_width, half_width + 0.001],
        [1.0, 1.0],
        [1.0, 1.0],
    )

    assert result["coverage_90"] == 0.5
    assert result["covered"] == 1
    assert result["interval_width_m_median"] == pytest.approx(2.0 * half_width)
    assert result["interval_width_m_p90"] == pytest.approx(2.0 * half_width)


def test_runtime_warmup_is_separate_from_measured_distribution() -> None:
    result = runtime_distributions(
        [
            {"duration_ms": 100.0, "iteration": 0, "phase": "end_to_end", "sample_kind": "warmup"},
            {"duration_ms": 80.0, "iteration": 1, "phase": "end_to_end", "sample_kind": "warmup"},
            {"duration_ms": 10.0, "iteration": 0, "phase": "end_to_end", "sample_kind": "measured"},
            {"duration_ms": 20.0, "iteration": 1, "phase": "end_to_end", "sample_kind": "measured"},
        ],
        clock_source="std::chrono::steady_clock",
    )
    phase = result["phases"]["end_to_end"]

    assert phase["warmup"]["mean_ms"] == 90.0
    assert phase["measured"]["mean_ms"] == 15.0
    assert phase["measured"]["median_ms"] == 15.0
    assert phase["throughput_per_second"] == pytest.approx(1000.0 / 15.0)
    assert result["clock_source"] == "std::chrono::steady_clock"


def test_runtime_rejects_duplicate_sample_identity() -> None:
    sample = {"duration_ms": 1.0, "iteration": 0, "phase": "inference", "sample_kind": "measured"}

    with pytest.raises(ValueError, match="duplicated"):
        runtime_distributions([sample, sample], clock_source="steady_clock")


def test_runtime_rejects_warmup_only_phase() -> None:
    sample = {
        "duration_ms": 1.0,
        "iteration": 0,
        "phase": "inference",
        "sample_kind": "warmup",
    }

    with pytest.raises(ValueError, match="no measured samples"):
        runtime_distributions([sample], clock_source="steady_clock")


def test_direct_metric_inputs_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="probability"):
        binary_brier([1.1], [1])
    with pytest.raises(ValueError, match="probabilities or outcome"):
        multiclass_nll([[0.2, 0.2]], [0])
    with pytest.raises(ValueError, match="positive"):
        geometry_uncertainty([0.0], [0.0], [1.0])


def test_aurc_tie_order_depends_on_stable_id_not_input_order() -> None:
    expected = aurc([0.5, 0.5], [0.0, 1.0], ["b", "a"])

    assert aurc([0.5, 0.5], [1.0, 0.0], ["a", "b"]) == expected

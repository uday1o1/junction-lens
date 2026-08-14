"""Deterministic paired bootstrap estimators for release gating."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from junctionlens.evaluator.evidence import adaptive_ece

Estimator = Literal["ratio", "average_precision", "adaptive_ece", "paired_runtime"]
Direction = Literal["higher_is_better", "lower_is_better"]


class BootstrapError(ValueError):
    """Raised when paired primitive evidence is malformed or unpaired."""


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """One point estimate and its deterministic adjusted interval."""

    point_estimate: float | None
    lower: float | None
    upper: float | None
    finite_replicates: int
    invalid_replicates: int
    replicates: int
    seed: int
    interval_alpha: float
    status: Literal["VALID", "INVALID_EMPTY_REPLICATE", "INSUFFICIENT_FINITE"]


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BootstrapError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BootstrapError(f"{label} must be finite")
    return result


def _segment_map(value: object, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise BootstrapError(f"{label}.segments must be a nonempty array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BootstrapError(f"{label}.segments[{index}] must be an object")
        segment_id = item.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id or len(segment_id) > 256:
            raise BootstrapError(f"{label}.segments[{index}] has an invalid segment_id")
        if segment_id in result:
            raise BootstrapError(f"{label} contains duplicate segment_id {segment_id}")
        result[segment_id] = cast(Mapping[str, Any], item)
    return result


def _paired_segments(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[list[str], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    baseline_map = _segment_map(baseline.get("segments"), "baseline")
    candidate_map = _segment_map(candidate.get("segments"), "candidate")
    if set(baseline_map) != set(candidate_map):
        raise BootstrapError("baseline and candidate must contain the exact same segment IDs")
    identifiers = sorted(baseline_map)
    return (
        identifiers,
        [baseline_map[identifier] for identifier in identifiers],
        [candidate_map[identifier] for identifier in identifiers],
    )


def _ratio_value(segments: Sequence[Mapping[str, Any]], draws: NDArray[np.int64]) -> float:
    numerator = 0.0
    denominator = 0.0
    for index in draws:
        segment = segments[int(index)]
        numerator += _number(segment.get("numerator"), "ratio numerator")
        denominator += _number(segment.get("denominator"), "ratio denominator")
    if denominator <= 0.0:
        return math.nan
    return numerator / denominator


def _prediction_rows(segment: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    predictions = segment.get("predictions")
    if not isinstance(predictions, list):
        raise BootstrapError("AP segment predictions must be an array")
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in predictions:
        if not isinstance(item, dict):
            raise BootstrapError("AP prediction must be an object")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise BootstrapError("AP prediction IDs must be nonempty and unique per segment")
        seen.add(identifier)
        confidence = _number(item.get("confidence"), "AP confidence")
        true_positive = item.get("true_positive")
        if not 0.0 <= confidence <= 1.0 or not isinstance(true_positive, bool):
            raise BootstrapError("AP prediction fields are invalid")
        rows.append(cast(Mapping[str, Any], item))
    return rows


def _average_precision_value(
    segments: Sequence[Mapping[str, Any]], draws: NDArray[np.int64]
) -> float:
    ground_truth = 0
    ranked: list[tuple[float, str, bool]] = []
    for occurrence, index in enumerate(draws):
        segment = segments[int(index)]
        count = segment.get("ground_truth_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise BootstrapError("AP ground_truth_count must be a nonnegative integer")
        ground_truth += count
        source_id = str(segment["segment_id"])
        for prediction in _prediction_rows(segment):
            qualified_id = f"draw-{occurrence:08d}:{source_id}:{prediction['id']}"
            ranked.append(
                (
                    _number(prediction["confidence"], "AP confidence"),
                    qualified_id,
                    cast(bool, prediction["true_positive"]),
                )
            )
    if ground_truth == 0:
        return math.nan
    ranked.sort(key=lambda item: (-item[0], item[1]))
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, _, is_true_positive) in enumerate(ranked, start=1):
        if is_true_positive:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / ground_truth


def _ece_value(segments: Sequence[Mapping[str, Any]], draws: NDArray[np.int64]) -> float:
    confidences: list[float] = []
    correctness: list[int] = []
    stable_ids: list[str] = []
    for occurrence, index in enumerate(draws):
        segment = segments[int(index)]
        raw = segment.get("observations")
        if not isinstance(raw, list):
            raise BootstrapError("ECE observations must be an array")
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                raise BootstrapError("ECE observation must be an object")
            identifier = item.get("id")
            confidence = _number(item.get("confidence"), "ECE confidence")
            correct = item.get("correct")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in seen
                or not 0.0 <= confidence <= 1.0
                or not isinstance(correct, bool)
            ):
                raise BootstrapError("ECE observation fields are invalid")
            seen.add(identifier)
            stable_ids.append(f"draw-{occurrence:08d}:{segment['segment_id']}:{identifier}")
            confidences.append(confidence)
            correctness.append(int(correct))
    if not confidences:
        return math.nan
    value, _ = adaptive_ece(confidences, correctness, stable_ids, bins=15)
    return value


def _pooled_value(
    estimator: Estimator,
    segments: Sequence[Mapping[str, Any]],
    draws: NDArray[np.int64],
) -> float:
    if estimator == "ratio":
        return _ratio_value(segments, draws)
    if estimator == "average_precision":
        return _average_precision_value(segments, draws)
    if estimator == "adaptive_ece":
        return _ece_value(segments, draws)
    raise BootstrapError("paired_runtime is not a segment estimator")


def _directional_delta(candidate: float, baseline: float, direction: Direction) -> float:
    delta = candidate - baseline
    return delta if direction == "higher_is_better" else -delta


def _type7_quantile(values: NDArray[np.float64], probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def paired_segment_bootstrap(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    estimator: Estimator,
    direction: Direction,
    family_alpha: float,
    gating_cells: int,
    replicates: int = 10000,
    seed: int = 20260813,
    minimum_finite_replicates: int = 9900,
) -> BootstrapResult:
    """Recompute a pooled paired metric over segment-cluster draws."""
    if estimator == "paired_runtime":
        raise BootstrapError("runtime evidence requires paired_trial_bootstrap")
    if not 0.0 < family_alpha < 1.0 or gating_cells <= 0 or replicates <= 0:
        raise BootstrapError("bootstrap interval configuration is invalid")
    identifiers, baseline_segments, candidate_segments = _paired_segments(baseline, candidate)
    all_indices = np.arange(len(identifiers), dtype=np.int64)
    baseline_point = _pooled_value(estimator, baseline_segments, all_indices)
    candidate_point = _pooled_value(estimator, candidate_segments, all_indices)
    point = (
        _directional_delta(candidate_point, baseline_point, direction)
        if math.isfinite(baseline_point) and math.isfinite(candidate_point)
        else None
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    deltas = np.empty(replicates, dtype=np.float64)
    if estimator == "ratio":
        baseline_numerators = np.asarray(
            [_number(item.get("numerator"), "ratio numerator") for item in baseline_segments]
        )
        baseline_denominators = np.asarray(
            [_number(item.get("denominator"), "ratio denominator") for item in baseline_segments]
        )
        candidate_numerators = np.asarray(
            [_number(item.get("numerator"), "ratio numerator") for item in candidate_segments]
        )
        candidate_denominators = np.asarray(
            [_number(item.get("denominator"), "ratio denominator") for item in candidate_segments]
        )
        for offset in range(0, replicates, 512):
            count = min(512, replicates - offset)
            draws = generator.integers(
                0, len(identifiers), (count, len(identifiers)), dtype=np.int64
            )
            baseline_denominator = baseline_denominators[draws].sum(axis=1)
            candidate_denominator = candidate_denominators[draws].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                baseline_values = baseline_numerators[draws].sum(axis=1) / baseline_denominator
                candidate_values = candidate_numerators[draws].sum(axis=1) / candidate_denominator
            chunk = candidate_values - baseline_values
            deltas[offset : offset + count] = chunk if direction == "higher_is_better" else -chunk
    else:
        for replicate in range(replicates):
            draws = generator.integers(0, len(identifiers), len(identifiers), dtype=np.int64)
            baseline_value = _pooled_value(estimator, baseline_segments, draws)
            candidate_value = _pooled_value(estimator, candidate_segments, draws)
            deltas[replicate] = _directional_delta(candidate_value, baseline_value, direction)
    finite = deltas[np.isfinite(deltas)]
    invalid = replicates - int(finite.size)
    adjusted_alpha = family_alpha / gating_cells
    if finite.size < minimum_finite_replicates:
        return BootstrapResult(
            point,
            None,
            None,
            int(finite.size),
            invalid,
            replicates,
            seed,
            adjusted_alpha,
            "INSUFFICIENT_FINITE",
        )
    lower = _type7_quantile(finite, adjusted_alpha / 2.0)
    upper = _type7_quantile(finite, 1.0 - adjusted_alpha / 2.0)
    return BootstrapResult(
        point,
        lower,
        upper,
        int(finite.size),
        invalid,
        replicates,
        seed,
        adjusted_alpha,
        "VALID" if point is not None else "INVALID_EMPTY_REPLICATE",
    )


def _block_map(value: object, label: str) -> dict[str, float | None]:
    if not isinstance(value, list) or not value:
        raise BootstrapError(f"{label}.blocks must be a nonempty array")
    result: dict[str, float | None] = {}
    for item in value:
        if not isinstance(item, dict):
            raise BootstrapError(f"{label} trial block must be an object")
        identifier = item.get("block_id")
        valid = item.get("valid")
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise BootstrapError(f"{label} trial block IDs are invalid")
        if not isinstance(valid, bool):
            raise BootstrapError(f"{label} trial block validity is missing")
        result[identifier] = _number(item.get("value"), f"{label} block value") if valid else None
    return result


def paired_trial_bootstrap(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    direction: Direction,
    family_alpha: float,
    gating_cells: int,
    replicates: int = 10000,
    seed: int = 20260813,
    minimum_valid_pairs: int = 8,
) -> BootstrapResult:
    """Bootstrap paired runtime block differences after invalid-block exclusion."""
    baseline_blocks = _block_map(baseline.get("blocks"), "baseline")
    candidate_blocks = _block_map(candidate.get("blocks"), "candidate")
    if set(baseline_blocks) != set(candidate_blocks):
        raise BootstrapError("runtime evidence must use the exact same trial block IDs")
    pairs = [
        (cast(float, baseline_blocks[key]), cast(float, candidate_blocks[key]))
        for key in sorted(baseline_blocks)
        if baseline_blocks[key] is not None and candidate_blocks[key] is not None
    ]
    adjusted_alpha = family_alpha / gating_cells
    if len(pairs) < minimum_valid_pairs:
        return BootstrapResult(
            None,
            None,
            None,
            0,
            replicates,
            replicates,
            seed,
            adjusted_alpha,
            "INSUFFICIENT_FINITE",
        )
    if any(baseline <= 0.0 for baseline, _ in pairs):
        raise BootstrapError("runtime baseline block values must be positive")
    differences = np.asarray(
        [
            _directional_delta(candidate, baseline, direction) / baseline
            for baseline, candidate in pairs
        ],
        dtype=np.float64,
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(0, len(pairs), (replicates, len(pairs)), dtype=np.int64)
    bootstrapped = differences[draws].mean(axis=1)
    return BootstrapResult(
        float(differences.mean()),
        _type7_quantile(bootstrapped, adjusted_alpha / 2.0),
        _type7_quantile(bootstrapped, 1.0 - adjusted_alpha / 2.0),
        replicates,
        0,
        replicates,
        seed,
        adjusted_alpha,
        "VALID",
    )


__all__ = [
    "BootstrapError",
    "BootstrapResult",
    "paired_segment_bootstrap",
    "paired_trial_bootstrap",
]

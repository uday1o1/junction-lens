"""Deterministic calibration, geometry uncertainty, and runtime evidence metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json_object_path,
)

NLL_EPSILON = 1.0e-7
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_OBSERVATIONS = 100_000


class EvidenceError(ValueError):
    """Raised when evidence input or a metric population is invalid."""


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    index: int
    count: int
    mean_confidence: float
    mean_correctness: float
    lower_rank: int
    upper_rank: int


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise EvidenceError(f"{label} must be finite and in [0, 1]")
    return result


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise EvidenceError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise EvidenceError("quantile population is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_binary_population(
    probabilities: list[float], outcomes: list[int], label: str
) -> None:
    if not probabilities or len(probabilities) != len(outcomes):
        raise EvidenceError(f"{label} populations must be nonempty and aligned")
    for probability in probabilities:
        _probability(probability, f"{label} probability")
    if any(
        isinstance(outcome, bool) or not isinstance(outcome, int) or outcome not in {0, 1}
        for outcome in outcomes
    ):
        raise EvidenceError(f"{label} outcomes must be binary integers")


def _validate_multiclass_population(
    probabilities: list[list[float]], outcomes: list[int], label: str
) -> None:
    if not probabilities or len(probabilities) != len(outcomes):
        raise EvidenceError(f"{label} populations must be nonempty and aligned")
    for row, outcome in zip(probabilities, outcomes, strict=True):
        if not isinstance(row, list) or not row:
            raise EvidenceError(f"{label} probability rows must be nonempty lists")
        checked = [_probability(value, f"{label} probability") for value in row]
        if (
            isinstance(outcome, bool)
            or not isinstance(outcome, int)
            or not 0 <= outcome < len(row)
            or not math.isclose(sum(checked), 1.0, rel_tol=0.0, abs_tol=1.0e-9)
        ):
            raise EvidenceError(f"{label} probabilities or outcome are invalid")


def binary_brier(probabilities: list[float], outcomes: list[int]) -> float:
    _validate_binary_population(probabilities, outcomes, "binary Brier")
    return sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    ) / len(probabilities)


def multiclass_brier(probabilities: list[list[float]], outcomes: list[int]) -> float:
    _validate_multiclass_population(probabilities, outcomes, "multiclass Brier")
    total = 0.0
    for row, outcome in zip(probabilities, outcomes, strict=True):
        total += sum((value - int(index == outcome)) ** 2 for index, value in enumerate(row))
    return total / len(probabilities)


def binary_nll(probabilities: list[float], outcomes: list[int]) -> tuple[float, int]:
    _validate_binary_population(probabilities, outcomes, "binary NLL")
    saturation_count = sum(
        probability < NLL_EPSILON or probability > 1.0 - NLL_EPSILON
        for probability in probabilities
    )
    losses = []
    for probability, outcome in zip(probabilities, outcomes, strict=True):
        clipped = min(max(probability, NLL_EPSILON), 1.0 - NLL_EPSILON)
        losses.append(-(outcome * math.log(clipped) + (1 - outcome) * math.log(1.0 - clipped)))
    return sum(losses) / len(losses), saturation_count


def multiclass_nll(probabilities: list[list[float]], outcomes: list[int]) -> tuple[float, int]:
    _validate_multiclass_population(probabilities, outcomes, "multiclass NLL")
    targets = []
    for row, outcome in zip(probabilities, outcomes, strict=True):
        targets.append(row[outcome])
    saturation_count = sum(value < NLL_EPSILON or value > 1.0 - NLL_EPSILON for value in targets)
    return (
        -sum(math.log(min(max(value, NLL_EPSILON), 1.0 - NLL_EPSILON)) for value in targets)
        / len(targets),
        saturation_count,
    )


def adaptive_ece(
    confidences: list[float], correctness: list[int], stable_ids: list[str], bins: int = 15
) -> tuple[float, tuple[CalibrationBin, ...]]:
    if (
        not confidences
        or len(confidences) != len(correctness)
        or len(confidences) != len(stable_ids)
    ):
        raise EvidenceError("ECE populations must be nonempty and aligned")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise EvidenceError("ECE bin count and stable IDs are invalid")
    checked_confidences = [_probability(value, "ECE confidence") for value in confidences]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}
        for value in correctness
    ) or any(not isinstance(value, str) or not value for value in stable_ids):
        raise EvidenceError("ECE observations are invalid")
    if len(stable_ids) != len(set(stable_ids)):
        raise EvidenceError("ECE bin count and stable IDs are invalid")
    ordered = sorted(
        zip(checked_confidences, correctness, stable_ids, strict=True),
        key=lambda item: (item[0], item[2]),
    )
    bin_count = min(bins, len(ordered))
    base, remainder = divmod(len(ordered), bin_count)
    offset = 0
    result = []
    weighted_error = 0.0
    for index in range(bin_count):
        count = base + int(index < remainder)
        items = ordered[offset : offset + count]
        mean_confidence = sum(item[0] for item in items) / count
        mean_correctness = sum(item[1] for item in items) / count
        weighted_error += count * abs(mean_confidence - mean_correctness)
        result.append(
            CalibrationBin(
                index, count, mean_confidence, mean_correctness, offset, offset + count - 1
            )
        )
        offset += count
    return weighted_error / len(ordered), tuple(result)


def aurc(confidences: list[float], errors: list[float], stable_ids: list[str]) -> float:
    if not confidences or len(confidences) != len(errors) or len(confidences) != len(stable_ids):
        raise EvidenceError("AURC populations must be nonempty and aligned")
    checked_confidences = [_probability(value, "AURC confidence") for value in confidences]
    checked_errors = [_finite(value, "AURC error") for value in errors]
    if any(value < 0.0 for value in checked_errors):
        raise EvidenceError("AURC errors must be nonnegative")
    if any(not isinstance(value, str) or not value for value in stable_ids):
        raise EvidenceError("AURC stable IDs are invalid")
    if len(stable_ids) != len(set(stable_ids)):
        raise EvidenceError("AURC stable IDs are duplicated")
    ordered = sorted(
        zip(checked_confidences, checked_errors, stable_ids, strict=True),
        key=lambda item: (-item[0], item[2]),
    )
    previous_coverage = previous_risk = area = cumulative_error = 0.0
    total = len(ordered)
    for index, (_, error, _) in enumerate(ordered, start=1):
        cumulative_error += error
        coverage = index / total
        risk = cumulative_error / index
        area += (coverage - previous_coverage) * (risk + previous_risk) / 2.0
        previous_coverage, previous_risk = coverage, risk
    return area


def geometry_uncertainty(
    residuals_m: list[float], scales_m: list[float], factors: list[float]
) -> dict[str, float | int]:
    if not residuals_m or not (len(residuals_m) == len(scales_m) == len(factors)):
        raise EvidenceError("geometry uncertainty populations must be nonempty and aligned")
    checked_residuals = [_finite(value, "geometry residual") for value in residuals_m]
    checked_scales = [_finite(value, "geometry scale", positive=True) for value in scales_m]
    checked_factors = [_finite(value, "geometry factor", positive=True) for value in factors]
    widths = [
        2.0 * scale * factor * math.log(10.0)
        for scale, factor in zip(checked_scales, checked_factors, strict=True)
    ]
    covered = sum(
        abs(residual) <= width / 2.0
        for residual, width in zip(checked_residuals, widths, strict=True)
    )
    return {
        "coverage_90": covered / len(residuals_m),
        "covered": covered,
        "count": len(residuals_m),
        "interval_width_m_median": _quantile(widths, 0.5),
        "interval_width_m_p90": _quantile(widths, 0.9),
    }


def runtime_distributions(samples: list[dict[str, Any]], *, clock_source: str) -> dict[str, Any]:
    if not samples:
        raise EvidenceError("runtime sample population is empty")
    if not isinstance(clock_source, str) or not clock_source.strip():
        raise EvidenceError("runtime clock source must be a nonempty string")
    populations: dict[tuple[str, str], list[float]] = defaultdict(list)
    identities: set[tuple[str, str, int]] = set()
    for sample in samples:
        if set(sample) != {"duration_ms", "iteration", "phase", "sample_kind"}:
            raise EvidenceError("runtime sample has unknown or missing keys")
        phase = sample["phase"]
        kind = sample["sample_kind"]
        iteration = sample["iteration"]
        if (
            not isinstance(phase, str)
            or not phase
            or kind not in {"warmup", "measured"}
            or isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 0
        ):
            raise EvidenceError("runtime sample identity is invalid")
        identity = (phase, kind, iteration)
        if identity in identities:
            raise EvidenceError("runtime sample identity is duplicated")
        identities.add(identity)
        populations[(phase, kind)].append(
            _finite(sample["duration_ms"], "duration_ms", positive=True)
        )
    phases: dict[str, Any] = {}
    for phase in sorted({key[0] for key in populations}):
        phase_result: dict[str, Any] = {}
        for kind in ("warmup", "measured"):
            values = populations.get((phase, kind), [])
            phase_result[kind] = (
                None
                if not values
                else {
                    "count": len(values),
                    "maximum_ms": max(values),
                    "mean_ms": sum(values) / len(values),
                    "median_ms": _quantile(values, 0.5),
                    "minimum_ms": min(values),
                    "p90_ms": _quantile(values, 0.9),
                    "p95_ms": _quantile(values, 0.95),
                    "p99_ms": _quantile(values, 0.99),
                }
            )
        measured = populations.get((phase, "measured"), [])
        if not measured:
            raise EvidenceError(f"runtime phase {phase} has no measured samples")
        phase_result["throughput_per_second"] = 1000.0 / (sum(measured) / len(measured))
        phases[phase] = phase_result
    return {
        "clock_source": clock_source,
        "phases": phases,
        "schema_version": "junctionlens.runtime-evidence.v1",
    }


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        return load_json_object_path(
            path,
            "evidence input",
            ParseLimits(
                max_bytes=MAX_INPUT_BYTES,
                max_depth=32,
                max_nodes=1_000_000,
                max_container_items=MAX_OBSERVATIONS,
                max_string_bytes=1024 * 1024,
            ),
        )
    except ParseBoundaryError as error:
        raise EvidenceError(str(error)) from error


def evaluate_calibration_file(path: Path) -> dict[str, Any]:
    payload = _strict_json(path)
    if (
        set(payload) != {"binary", "geometry", "multiclass", "schema_version"}
        or payload["schema_version"] != "junctionlens.calibration-input.v1"
    ):
        raise EvidenceError("calibration input schema is invalid")
    binary = payload["binary"]
    multiclass = payload["multiclass"]
    geometry = payload["geometry"]
    if not all(
        isinstance(items, list) and len(items) <= MAX_OBSERVATIONS
        for items in (binary, multiclass, geometry)
    ):
        raise EvidenceError("calibration populations exceed limits")
    if not binary or not multiclass or not geometry:
        raise EvidenceError("calibration populations must be nonempty")
    if any(
        not isinstance(item, dict) or set(item) != {"id", "outcome", "probability"}
        for item in binary
    ):
        raise EvidenceError("binary observation schema is invalid")
    if any(
        not isinstance(item, dict)
        or set(item) != {"id", "outcome_class", "probabilities"}
        or not isinstance(item["probabilities"], list)
        for item in multiclass
    ):
        raise EvidenceError("multiclass observation schema is invalid")
    if any(
        not isinstance(item, dict) or set(item) != {"factor", "id", "residual_m", "scale_m"}
        for item in geometry
    ):
        raise EvidenceError("geometry observation schema is invalid")
    binary_ids = [item["id"] for item in binary]
    binary_probabilities = [
        _probability(item["probability"], "binary probability") for item in binary
    ]
    binary_outcomes = [item["outcome"] for item in binary]
    if any(not isinstance(identifier, str) or not identifier for identifier in binary_ids) or any(
        isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}
        for value in binary_outcomes
    ):
        raise EvidenceError("binary observation schema is invalid")
    binary_nll_value, binary_clipped = binary_nll(binary_probabilities, binary_outcomes)
    binary_ece, binary_bins = adaptive_ece(binary_probabilities, binary_outcomes, binary_ids)
    multi_ids = [item["id"] for item in multiclass]
    multi_probabilities = [
        [_probability(value, "multiclass probability") for value in item["probabilities"]]
        for item in multiclass
    ]
    multi_outcomes = [item["outcome_class"] for item in multiclass]
    if any(not isinstance(identifier, str) or not identifier for identifier in multi_ids) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in multi_outcomes
    ):
        raise EvidenceError("multiclass observation schema is invalid")
    multi_nll_value, multi_clipped = multiclass_nll(multi_probabilities, multi_outcomes)
    multi_confidence = [max(row) for row in multi_probabilities]
    multi_correct = [
        int(max(range(len(row)), key=row.__getitem__) == outcome)
        for row, outcome in zip(multi_probabilities, multi_outcomes, strict=True)
    ]
    multi_ece, multi_bins = adaptive_ece(multi_confidence, multi_correct, multi_ids)
    residuals = [_finite(item["residual_m"], "geometry residual") for item in geometry]
    scales = [_finite(item["scale_m"], "geometry scale", positive=True) for item in geometry]
    factors = [_finite(item["factor"], "geometry factor", positive=True) for item in geometry]
    geometry_ids = [item["id"] for item in geometry]
    if any(not isinstance(identifier, str) or not identifier for identifier in geometry_ids) or len(
        geometry_ids
    ) != len(set(geometry_ids)):
        raise EvidenceError("geometry observation IDs are invalid")
    return {
        "binary": {
            "adaptive_ece_15": binary_ece,
            "aurc": aurc(
                binary_probabilities,
                [
                    float(probability != outcome)
                    for probability, outcome in zip(
                        [int(value >= 0.5) for value in binary_probabilities],
                        binary_outcomes,
                        strict=True,
                    )
                ],
                binary_ids,
            ),
            "brier_score": binary_brier(binary_probabilities, binary_outcomes),
            "ece_bins": [asdict(item) for item in binary_bins],
            "negative_log_likelihood": binary_nll_value,
            "nll_saturation_count": binary_clipped,
        },
        "geometry": geometry_uncertainty(residuals, scales, factors),
        "multiclass": {
            "adaptive_ece_15": multi_ece,
            "aurc": aurc(
                multi_confidence, [float(not value) for value in multi_correct], multi_ids
            ),
            "brier_score": multiclass_brier(multi_probabilities, multi_outcomes),
            "ece_bins": [asdict(item) for item in multi_bins],
            "negative_log_likelihood": multi_nll_value,
            "nll_saturation_count": multi_clipped,
        },
        "schema_version": "junctionlens.calibration-evidence.v1",
    }


def evaluate_runtime_file(path: Path) -> dict[str, Any]:
    payload = _strict_json(path)
    if (
        set(payload) != {"clock_source", "samples", "schema_version"}
        or payload["schema_version"] != "junctionlens.runtime-input.v1"
        or not isinstance(payload["clock_source"], str)
        or not payload["clock_source"].strip()
        or not isinstance(payload["samples"], list)
        or len(payload["samples"]) > MAX_OBSERVATIONS
    ):
        raise EvidenceError("runtime input schema is invalid")
    return runtime_distributions(payload["samples"], clock_source=payload["clock_source"])


__all__ = [
    "EvidenceError",
    "adaptive_ece",
    "aurc",
    "binary_brier",
    "binary_nll",
    "evaluate_calibration_file",
    "evaluate_runtime_file",
    "geometry_uncertainty",
    "multiclass_brier",
    "multiclass_nll",
    "runtime_distributions",
]

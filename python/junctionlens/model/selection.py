"""Shared frozen checkpoint ordering and early-stopping policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class SelectionScore(Protocol):
    @property
    def epoch(self) -> int: ...

    @property
    def lane_control_topology(self) -> float: ...

    @property
    def official_composite(self) -> float: ...

    @property
    def negative_log_likelihood(self) -> float: ...


@dataclass(frozen=True, slots=True)
class EarlyStoppingResult:
    eligible_count: int
    stopping_epoch: int
    patience_exhausted: bool


def score_order(score: SelectionScore) -> tuple[float, float, float, int]:
    """Return the exact topology, official, NLL, epoch lexicographic key."""
    return (
        -score.lane_control_topology,
        -score.official_composite,
        score.negative_log_likelihood,
        score.epoch,
    )


def apply_frozen_early_stopping[ScoreT: SelectionScore](
    scores: Sequence[ScoreT], *, minimum_epoch: int = 20, patience: int = 8
) -> tuple[list[ScoreT], EarlyStoppingResult]:
    """Truncate a complete ordered score series at the frozen patience boundary."""
    if minimum_epoch <= 0 or patience <= 0:
        raise ValueError("early-stopping bounds must be positive")
    ordered = sorted(scores, key=lambda item: item.epoch)
    if [item.epoch for item in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("checkpoint selection scores must be consecutive from epoch one")
    best_key: tuple[float, float, float, int] | None = None
    stale = 0
    eligible_count = len(ordered)
    exhausted = False
    for index, score in enumerate(ordered):
        key = score_order(score)
        improved = best_key is None or key < best_key
        if improved:
            best_key = key
        if score.epoch < minimum_epoch:
            continue
        stale = 0 if improved else stale + 1
        if stale >= patience:
            eligible_count = index + 1
            exhausted = True
            break
    eligible = ordered[:eligible_count]
    return eligible, EarlyStoppingResult(
        eligible_count=eligible_count,
        stopping_epoch=eligible[-1].epoch,
        patience_exhausted=exhausted,
    )


__all__ = [
    "EarlyStoppingResult",
    "SelectionScore",
    "apply_frozen_early_stopping",
    "score_order",
]

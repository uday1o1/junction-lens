"""Strict pretraining budget and seed-policy validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_yaml_object_path


class BudgetError(RuntimeError):
    """Raised when a proposed experiment matrix exceeds a frozen limit."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HardLimits(_FrozenModel):
    gpu_hours: float = Field(ge=120.0, le=120.0)
    generated_artifacts_gib: float = Field(ge=400.0, le=400.0)


class ExperimentAllocation(_FrozenModel):
    id: str = Field(min_length=1)
    architecture_scope: Literal["E0", "E1", "E2", "E3", "selected-final", "infrastructure"]
    purpose: Literal["baseline-final", "screen", "robustness", "qualification"]
    seeds: tuple[int, ...] = Field(min_length=1)
    gpu_hours: float = Field(gt=0)
    generated_artifacts_gib: float = Field(gt=0)


class CheckpointPolicy(_FrozenModel):
    latest_cadence_epochs: int = Field(ge=1)
    model_selection_cadence_epochs: int = Field(ge=1)
    screen_best_checkpoints_retained: int = Field(ge=1)
    final_best_checkpoints_retained: int = Field(ge=1)
    retain_last_checkpoint: Literal[True]
    retain_optimizer_state_for: tuple[Literal["active", "promoted"], ...]
    discard_optimizer_state_for: tuple[Literal["rejected", "completed-screen"], ...]
    resumable_safe_checkpoint_on_budget_abort: Literal[True]


class PromotionPolicy(_FrozenModel):
    screening_seed: Literal[20260813]
    robustness_seeds: tuple[Literal[20260814, 20260815], ...]
    screened_architectures: tuple[Literal["E1", "E2", "E3"], ...]
    robustness_eligible_architectures: tuple[Literal["E0", "selected-final"], ...]
    selection_partition_only: Literal[True]
    holdout_selection_prohibited: Literal[True]


class BudgetPlan(_FrozenModel):
    schema_version: Literal["1.0.0"]
    hard_limits: HardLimits
    minimum_contingency_fraction: float = Field(ge=0.2, le=0.2)
    experiments: tuple[ExperimentAllocation, ...] = Field(min_length=1)
    checkpoint_policy: CheckpointPolicy
    promotion_policy: PromotionPolicy

    @model_validator(mode="after")
    def validate_matrix(self) -> BudgetPlan:
        ids = [item.id for item in self.experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment allocation IDs must be unique")
        for architecture in ("E1", "E2", "E3"):
            matches = [item for item in self.experiments if item.architecture_scope == architecture]
            if len(matches) != 1 or matches[0].seeds != (20260813,):
                raise ValueError(f"{architecture} must screen exactly seed 20260813")
        for item in self.experiments:
            has_robustness_seed = any(seed in {20260814, 20260815} for seed in item.seeds)
            if has_robustness_seed and item.architecture_scope not in {"E0", "selected-final"}:
                raise ValueError(
                    "only E0 and the selected final architecture receive robustness seeds"
                )
        gpu_total = sum(item.gpu_hours for item in self.experiments)
        artifact_total = sum(item.generated_artifacts_gib for item in self.experiments)
        gpu_contingency = (self.hard_limits.gpu_hours - gpu_total) / self.hard_limits.gpu_hours
        artifact_contingency = (
            self.hard_limits.generated_artifacts_gib - artifact_total
        ) / self.hard_limits.generated_artifacts_gib
        if gpu_contingency < self.minimum_contingency_fraction:
            raise ValueError("GPU-hour allocation leaves less than the frozen contingency")
        if artifact_contingency < self.minimum_contingency_fraction:
            raise ValueError("artifact allocation leaves less than the frozen contingency")
        return self

    def summary(self) -> dict[str, float]:
        """Return planned totals and unallocated contingency."""
        gpu_total = sum(item.gpu_hours for item in self.experiments)
        artifact_total = sum(item.generated_artifacts_gib for item in self.experiments)
        return {
            "planned_gpu_hours": gpu_total,
            "gpu_hour_contingency": self.hard_limits.gpu_hours - gpu_total,
            "gpu_hour_contingency_fraction": (self.hard_limits.gpu_hours - gpu_total)
            / self.hard_limits.gpu_hours,
            "planned_generated_artifacts_gib": artifact_total,
            "generated_artifact_contingency_gib": (
                self.hard_limits.generated_artifacts_gib - artifact_total
            ),
            "generated_artifact_contingency_fraction": (
                self.hard_limits.generated_artifacts_gib - artifact_total
            )
            / self.hard_limits.generated_artifacts_gib,
        }


def load_budget_plan(path: Path) -> BudgetPlan:
    """Load and fail closed on any budget, seed, cadence, or retention drift."""
    try:
        value = load_yaml_object_path(
            path,
            "experiment budget plan",
            ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
        )
        return BudgetPlan.model_validate(value)
    except (ParseBoundaryError, ValueError) as error:
        raise BudgetError(f"invalid budget plan: {error}") from error

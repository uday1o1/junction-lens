from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from junctionlens.model.budget import BudgetError, load_budget_plan

BUDGET_PATH = Path("configs/budgets/v1.yaml")


def test_frozen_budget_keeps_required_contingency_and_seed_policy() -> None:
    plan = load_budget_plan(BUDGET_PATH)
    assert plan.hard_limits.gpu_hours == 120.0
    assert plan.hard_limits.generated_artifacts_gib == 400.0
    assert plan.summary() == {
        "planned_gpu_hours": 90.0,
        "gpu_hour_contingency": 30.0,
        "gpu_hour_contingency_fraction": 0.25,
        "planned_generated_artifacts_gib": 268.0,
        "generated_artifact_contingency_gib": 132.0,
        "generated_artifact_contingency_fraction": 0.33,
    }


def test_budget_rejects_robustness_seed_on_screened_architecture(tmp_path: Path) -> None:
    value = yaml.safe_load(BUDGET_PATH.read_text(encoding="utf-8"))
    value["experiments"][1]["seeds"] = [20260813, 20260814]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(BudgetError, match="must screen exactly seed 20260813"):
        load_budget_plan(path)


def test_budget_rejects_less_than_twenty_percent_contingency(tmp_path: Path) -> None:
    value = yaml.safe_load(BUDGET_PATH.read_text(encoding="utf-8"))
    value["experiments"][0]["gpu_hours"] = 40.0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(BudgetError, match="GPU-hour allocation"):
        load_budget_plan(path)

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path

from junctionlens.model.overfit import MicroOverfitError, run_micro_overfit
from junctionlens.model.profile import load_m0_profile


def test_micro_overfit_logs_every_unweighted_head_and_gradient(tmp_path: Path) -> None:
    profile = load_m0_profile(Path("configs/model/m0-spike.yaml"))
    with suppress(MicroOverfitError):
        run_micro_overfit(profile, tmp_path, steps=100)
    lines = (tmp_path / "training-metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    first = json.loads(lines[0])
    assert set(first["losses"]) == {
        "area_category",
        "area_existence",
        "area_geometry",
        "lane_centerline",
        "lane_connector",
        "lane_existence",
        "lane_left_boundary",
        "lane_left_type",
        "lane_right_boundary",
        "lane_right_type",
        "traffic_attribute",
        "traffic_box",
        "traffic_category",
        "traffic_existence",
    }
    assert first["gradient_norm_before_clip"] > 0
    report = json.loads((tmp_path / "micro-overfit-report.json").read_text(encoding="utf-8"))
    assert report["frames"] == 32
    assert report["nonfinite_count"] == 0
    assert report["topology_gate_state"] == "DEFERRED_TO_M5_PER_ADR_0001"

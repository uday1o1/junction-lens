"""Deterministic 32-frame micro-overfit gate for the M0 model surface."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional

from junctionlens.model.profile import M0ModelProfile
from junctionlens.model.spike import M0GraphModel, outputs_by_name
from junctionlens.model.synthetic import MicroTargets, make_micro_inputs, make_micro_targets


class MicroOverfitError(RuntimeError):
    """Raised when the fixed tiny-set learning gate does not pass."""


def _losses(outputs: dict[str, Tensor], targets: MicroTargets) -> dict[str, Tensor]:
    return {
        "lane_existence": functional.binary_cross_entropy_with_logits(
            outputs["lane_existence_logits"], targets.lane_existence
        ),
        "lane_centerline": functional.smooth_l1_loss(
            outputs["lane_centerline"][:, 0], targets.centerline
        ),
        "lane_left_boundary": functional.smooth_l1_loss(
            outputs["lane_left_boundary"][:, 0], targets.left_boundary
        ),
        "lane_right_boundary": functional.smooth_l1_loss(
            outputs["lane_right_boundary"][:, 0], targets.right_boundary
        ),
        "lane_left_type": functional.cross_entropy(
            outputs["lane_left_boundary_logits"][:, 0], targets.left_type
        ),
        "lane_right_type": functional.cross_entropy(
            outputs["lane_right_boundary_logits"][:, 0], targets.right_type
        ),
        "lane_connector": functional.binary_cross_entropy_with_logits(
            outputs["lane_connector_logits"][:, 0], targets.connector
        ),
        "traffic_existence": functional.binary_cross_entropy_with_logits(
            outputs["traffic_existence_logits"], targets.traffic_existence
        ),
        "traffic_box": functional.smooth_l1_loss(
            outputs["traffic_boxes"][:, 0], targets.traffic_box
        ),
        "traffic_category": functional.cross_entropy(
            outputs["traffic_category_logits"][:, 0], targets.traffic_category
        ),
        "traffic_attribute": functional.cross_entropy(
            outputs["traffic_attribute_logits"][:, 0], targets.traffic_attribute
        ),
        "area_existence": functional.binary_cross_entropy_with_logits(
            outputs["area_existence_logits"], targets.area_existence
        ),
        "area_category": functional.cross_entropy(
            outputs["area_category_logits"][:, 0], targets.area_category
        ),
        "area_geometry": functional.smooth_l1_loss(
            outputs["area_points"][:, 0], targets.area_points
        ),
    }


def _weighted_total(losses: dict[str, Tensor]) -> Tensor:
    weights = {
        "lane_existence": 2.0,
        "lane_centerline": 5.0,
        "lane_left_boundary": 2.0,
        "lane_right_boundary": 2.0,
        "lane_left_type": 1.0,
        "lane_right_type": 1.0,
        "lane_connector": 1.0,
        "traffic_existence": 2.0,
        "traffic_box": 5.0,
        "traffic_category": 2.0,
        "traffic_attribute": 2.0,
        "area_existence": 1.0,
        "area_category": 1.0,
        "area_geometry": 1.0,
    }
    return sum(loss * weights[name] for name, loss in losses.items())


def _gate_metrics(outputs: dict[str, Tensor], targets: MicroTargets) -> dict[str, float]:
    category_correct = torch.cat(
        (
            outputs["lane_left_boundary_logits"][:, 0].argmax(1) == targets.left_type,
            outputs["lane_right_boundary_logits"][:, 0].argmax(1) == targets.right_type,
            (outputs["lane_connector_logits"][:, 0] >= 0) == targets.connector.bool(),
            outputs["traffic_category_logits"][:, 0].argmax(1) == targets.traffic_category,
            outputs["traffic_attribute_logits"][:, 0].argmax(1) == targets.traffic_attribute,
            outputs["area_category_logits"][:, 0].argmax(1) == targets.area_category,
        )
    )
    point_errors = torch.linalg.vector_norm(
        outputs["lane_centerline"][:, 0] - targets.centerline, dim=2
    )
    return {
        "matched_node_category_accuracy": float(category_correct.float().mean().item()),
        "median_matched_centerline_point_error_m": float(point_errors.median().item()),
    }


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_micro_overfit(
    profile: M0ModelProfile,
    output_dir: Path,
    *,
    steps: int | None = None,
) -> dict[str, Any]:
    """Train and evaluate the exact fixed 32-frame M0 gate on CPU."""
    requested_steps = steps or profile.micro_overfit.default_steps
    if not 100 <= requested_steps <= profile.micro_overfit.maximum_steps:
        raise MicroOverfitError("steps must be between 100 and the frozen maximum")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(profile.seed)
    np.random.seed(profile.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    frame_indices = torch.arange(profile.micro_overfit.frames, dtype=torch.int64)
    inputs = make_micro_inputs(
        profile,
        frame_indices,
        spatial_size=profile.micro_overfit.cpu_spatial_size,
    )
    targets = make_micro_targets(profile, frame_indices)
    model = M0GraphModel(profile)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=profile.micro_overfit.learning_rate,
        weight_decay=profile.micro_overfit.weight_decay,
    )
    loss_history: list[float] = []
    nonfinite_count = 0
    maximum_gradient_norm = 0.0
    metrics_path = output_dir / "training-metrics.jsonl"
    start = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(requested_steps):
            optimizer.zero_grad(set_to_none=True)
            named_outputs = outputs_by_name(model(*inputs))
            losses = _losses(named_outputs, targets)
            total = _weighted_total(losses)
            if not torch.isfinite(total):
                nonfinite_count += 1
                break
            total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), profile.micro_overfit.gradient_clip_norm
                ).item()
            )
            if not math.isfinite(gradient_norm):
                nonfinite_count += 1
                break
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            optimizer.step()
            total_value = float(total.detach().item())
            loss_history.append(total_value)
            record = {
                "gradient_norm_before_clip": gradient_norm,
                "losses": {name: float(loss.detach().item()) for name, loss in losses.items()},
                "step": step + 1,
                "weighted_total_loss": total_value,
            }
            metrics_file.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
    duration = time.perf_counter() - start
    if len(loss_history) < 100:
        raise MicroOverfitError("training terminated before the first-100-step baseline existed")
    model.eval()
    with torch.inference_mode():
        final_outputs = outputs_by_name(model(*inputs))
        final_losses = _losses(final_outputs, targets)
        final_total = float(_weighted_total(final_losses).item())
        metrics = _gate_metrics(final_outputs, targets)
    first_100_median = float(np.median(np.asarray(loss_history[:100], dtype=np.float64)))
    reduction = 1.0 - final_total / first_100_median
    metrics["loss_reduction_from_first_100_median"] = reduction
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "profile": profile.model_dump(mode="json"),
            "profile_sha256": profile.canonical_sha256(),
            "seed": profile.seed,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    passed = (
        nonfinite_count == 0
        and reduction >= profile.micro_overfit.minimum_loss_reduction
        and metrics["matched_node_category_accuracy"]
        >= profile.micro_overfit.minimum_node_category_accuracy
        and metrics["median_matched_centerline_point_error_m"]
        <= profile.micro_overfit.maximum_centerline_point_error_m
    )
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PASSED" if passed else "FAILED",
        "profile_id": profile.profile_id,
        "profile_sha256": profile.canonical_sha256(),
        "seed": profile.seed,
        "frames": profile.micro_overfit.frames,
        "steps_completed": len(loss_history),
        "first_100_median_total_loss": first_100_median,
        "final_total_loss": final_total,
        "metrics": metrics,
        "nonfinite_count": nonfinite_count,
        "mixed_precision_overflow_count": 0,
        "maximum_gradient_norm_before_clip": maximum_gradient_norm,
        "training_seconds": duration,
        "training_sequences_per_second": len(loss_history)
        * profile.micro_overfit.frames
        / duration,
        "peak_training_memory_bytes": _maximum_rss_bytes(),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "topology_gate_state": profile.micro_overfit.topology_gate_state,
    }
    _write_json(output_dir / "micro-overfit-report.json", report)
    if not passed:
        raise MicroOverfitError(f"micro-overfit gate failed: {report!r}")
    return report

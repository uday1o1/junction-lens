"""Deterministic oracle-node and predicted-node topology learning gate."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from junctionlens.model.e0_losses import E0Matches, E0Targets, MatchIndices
from junctionlens.model.e0_profile import E0Profile
from junctionlens.model.e1_losses import (
    E1EdgeWeights,
    E1Targets,
    topology_assignments,
    topology_only_losses,
)
from junctionlens.model.e1_profile import E1Profile
from junctionlens.model.topology import LearnedTopologyHeads


class TopologyDiagnosticError(RuntimeError):
    """Raised when a learned topology head cannot pass its declared diagnostic."""


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    mode: Literal["oracle-nodes", "predicted-nodes"]
    steps: int
    initial_loss: float
    final_loss: float
    lane_successor_f1: float
    control_lane_f1: float
    maximum_topology_gradient: float
    maximum_query_gradient: float | None
    state: Literal["ACCEPTED"]


def _synthetic_nodes(base: E0Profile) -> tuple[E1Targets, Tensor, Tensor, Tensor, Tensor]:
    lane_count = 6
    control_count = 3
    steps = torch.linspace(0.0, 5.0, base.architecture.lane_points)
    centerlines = torch.stack(
        [
            torch.stack(
                (
                    steps + 5.0 + lane * 5.0,
                    torch.zeros_like(steps),
                    torch.zeros_like(steps),
                ),
                dim=1,
            )
            for lane in range(lane_count)
        ]
    )
    lane_successor = torch.zeros(lane_count, lane_count, dtype=torch.bool)
    lane_successor[torch.arange(lane_count - 1), torch.arange(1, lane_count)] = True
    control_lane = torch.zeros(control_count, lane_count, dtype=torch.bool)
    control_lane[0, :2] = True
    control_lane[1, 2:4] = True
    control_lane[2, 4:] = True
    boxes = torch.tensor(
        [[0.45, 0.25, 0.55, 0.38], [0.45, 0.35, 0.55, 0.48], [0.45, 0.45, 0.55, 0.58]],
        dtype=torch.float32,
    )
    node_targets = E0Targets(
        lane_centerline=centerlines,
        lane_left_boundary=centerlines + torch.tensor([0.0, 1.75, 0.0]),
        lane_right_boundary=centerlines - torch.tensor([0.0, 1.75, 0.0]),
        lane_left_type=torch.zeros(lane_count, dtype=torch.int64),
        lane_right_type=torch.ones(lane_count, dtype=torch.int64),
        lane_connector=torch.zeros(lane_count, dtype=torch.bool),
        traffic_boxes=boxes,
        traffic_category=torch.zeros(control_count, dtype=torch.int64),
        traffic_attribute=torch.zeros(control_count, dtype=torch.int64),
        area_points=torch.empty(0, base.architecture.area_points, 3),
        area_valid=torch.empty(0, base.architecture.area_points, dtype=torch.bool),
        area_category=torch.empty(0, dtype=torch.int64),
    )
    intrinsic = torch.tensor([[[320.0, 0.0, 320.0], [0.0, 320.0, 192.0], [0.0, 0.0, 1.0]]])
    transform = torch.eye(4).reshape(1, 4, 4)
    transform[0, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    transform[0, 2, 3] = 1.5
    return (
        E1Targets(node_targets, lane_successor, control_lane),
        centerlines,
        boxes,
        intrinsic,
        transform,
    )


def _matches(
    lane_permutation: Tensor, control_permutation: Tensor, device: torch.device
) -> E0Matches:
    return E0Matches(
        MatchIndices(torch.arange(len(lane_permutation), device=device), lane_permutation),
        MatchIndices(torch.arange(len(control_permutation), device=device), control_permutation),
        MatchIndices(
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
        ),
    )


def _f1(logits: Tensor, target: Tensor, mask: Tensor, threshold: float) -> float:
    predicted = torch.sigmoid(logits[mask]) >= threshold
    truth = target[mask]
    true_positive = int((predicted & truth).sum().item())
    false_positive = int((predicted & ~truth).sum().item())
    false_negative = int((~predicted & truth).sum().item())
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2 * true_positive / denominator


def _positive_weight(target: Tensor, mask: Tensor) -> float:
    positive = int((target & mask).sum().item())
    possible = int(mask.sum().item())
    if positive == 0 or positive > possible:
        raise TopologyDiagnosticError("diagnostic topology population is invalid")
    return max(1.0, (possible - positive) / positive)


def _run_mode(
    mode: Literal["oracle-nodes", "predicted-nodes"], base: E0Profile, profile: E1Profile
) -> DiagnosticResult:
    torch.manual_seed(profile.diagnostics.seed)
    targets, truth_centerlines, truth_boxes, intrinsic, transform = _synthetic_nodes(base)
    if mode == "oracle-nodes":
        lane_permutation = torch.arange(6)
        control_permutation = torch.arange(3)
    else:
        lane_permutation = torch.tensor([2, 0, 5, 1, 4, 3])
        control_permutation = torch.tensor([2, 0, 1])
    centerlines = truth_centerlines[lane_permutation][None]
    boxes = truth_boxes[control_permutation][None]
    matches = _matches(lane_permutation, control_permutation, centerlines.device)
    empty_logits = {
        "lane_successor_logits": torch.zeros(1, 6, 6),
        "control_lane_logits": torch.zeros(1, 3, 6),
    }
    assignments = topology_assignments(empty_logits, targets, matches)
    weights = E1EdgeWeights(
        _positive_weight(assignments.lane_successor_target, assignments.lane_successor_mask),
        _positive_weight(assignments.control_lane_target, assignments.control_lane_mask),
        "model_training",
        "a" * 64,
    )
    head = LearnedTopologyHeads(base, profile)
    generator = torch.Generator().manual_seed(profile.diagnostics.seed + 1)
    lane_initial = torch.randn(1, 6, base.architecture.hidden_dimension, generator=generator)
    control_initial = torch.randn(1, 3, base.architecture.hidden_dimension, generator=generator)
    if mode == "predicted-nodes":
        lane_features: Tensor = nn.Parameter(lane_initial)
        control_features: Tensor = nn.Parameter(control_initial)
        parameters = [*head.parameters(), lane_features, control_features]
    else:
        lane_features = lane_initial
        control_features = control_initial
        parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.02, weight_decay=0.0)
    first_losses = []
    topology_gradient = 0.0
    query_gradient = 0.0
    final_loss = math.inf
    lane_f1 = control_f1 = 0.0
    for step in range(1, profile.diagnostics.maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        lane_logits, control_logits = head(
            lane_features,
            control_features,
            centerlines,
            boxes,
            intrinsic,
            transform,
        )
        outputs = {
            "lane_successor_logits": lane_logits,
            "control_lane_logits": control_logits,
            "lane_centerline": centerlines,
        }
        losses = topology_only_losses(outputs, assignments, weights)
        total = 3.0 * losses["lane_successor"] + 5.0 * losses["control_lane"]
        total = total + losses["successor_endpoint_continuity"]
        if not torch.isfinite(total):
            raise TopologyDiagnosticError(f"{mode} produced a nonfinite loss")
        total.backward()
        topology_gradient = max(
            topology_gradient,
            max(
                float(parameter.grad.detach().abs().max().item())
                for parameter in head.parameters()
                if parameter.grad is not None
            ),
        )
        if mode == "predicted-nodes":
            query_gradient = max(
                query_gradient,
                float(lane_features.grad.detach().abs().max().item()),
                float(control_features.grad.detach().abs().max().item()),
            )
        optimizer.step()
        final_loss = float(total.detach().item())
        if step <= 20:
            first_losses.append(final_loss)
        lane_f1 = _f1(
            lane_logits,
            assignments.lane_successor_target,
            assignments.lane_successor_mask,
            profile.diagnostics.prediction_threshold,
        )
        control_f1 = _f1(
            control_logits,
            assignments.control_lane_target,
            assignments.control_lane_mask,
            profile.diagnostics.prediction_threshold,
        )
        if step >= 20 and min(lane_f1, control_f1) >= 0.999:
            break
    initial_loss = float(torch.tensor(first_losses).median().item())
    if (
        lane_f1 < profile.diagnostics.oracle_lane_successor_f1_minimum
        or control_f1 < profile.diagnostics.oracle_control_lane_f1_minimum
        or topology_gradient <= 0.0
        or (mode == "predicted-nodes" and query_gradient <= 0.0)
        or final_loss >= initial_loss
    ):
        raise TopologyDiagnosticError(f"{mode} did not pass the frozen topology learning gate")
    return DiagnosticResult(
        mode,
        step,
        initial_loss,
        final_loss,
        lane_f1,
        control_f1,
        topology_gradient,
        query_gradient if mode == "predicted-nodes" else None,
        "ACCEPTED",
    )


def run_topology_diagnostic(
    base: E0Profile, profile: E1Profile, output: Path | None = None
) -> dict[str, object]:
    """Run both declared topology modes and optionally persist their evidence."""
    profile.validate_base(base)
    results = [_run_mode(mode, base, profile) for mode in profile.diagnostics.modes]
    report: dict[str, object] = {
        "schema_version": "junctionlens.e1-topology-diagnostic.v1",
        "experiment_id": profile.experiment_id,
        "base_profile_sha256": base.canonical_sha256(),
        "e1_profile_sha256": profile.canonical_sha256(),
        "seed": profile.diagnostics.seed,
        "results": [asdict(item) for item in results],
        "state": "ACCEPTED",
    }
    if output is not None:
        if output.exists() or output.is_symlink():
            raise TopologyDiagnosticError("topology diagnostic output already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=".topology-",
            delete=False,
        ) as destination:
            destination.write(
                json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
            destination.flush()
            os.fsync(destination.fileno())
            temporary = Path(destination.name)
        temporary.replace(output)
    return report


__all__ = ["TopologyDiagnosticError", "run_topology_diagnostic"]

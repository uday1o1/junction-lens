"""Node-induced edge matching and learned-topology losses for E1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from junctionlens.model.e0_losses import (
    E0_LOSS_WEIGHTS,
    E0Matches,
    E0Targets,
    E0TrainingWeights,
    e0_losses,
)


class E1LossError(ValueError):
    """Raised when topology targets, matches, or losses violate the E1 contract."""


@dataclass(frozen=True, slots=True)
class E1Targets:
    nodes: E0Targets
    lane_successor: Tensor
    control_lane: Tensor

    def validate(self) -> None:
        self.nodes.validate()
        lanes = self.nodes.lane_centerline.shape[0]
        controls = self.nodes.traffic_boxes.shape[0]
        if self.lane_successor.shape != (lanes, lanes):
            raise E1LossError("lane-successor target shape differs from the lane population")
        if self.control_lane.shape != (controls, lanes):
            raise E1LossError("control-lane target must be control-major and lane-minor")
        if self.lane_successor.dtype != torch.bool or self.control_lane.dtype != torch.bool:
            raise E1LossError("topology targets must use Boolean edge labels")
        if bool(torch.diagonal(self.lane_successor).any()):
            raise E1LossError("lane-successor targets contain a prohibited self-edge")


@dataclass(frozen=True, slots=True)
class E1EdgeWeights:
    lane_successor_positive: float
    control_lane_positive: float
    source_partition: str
    source_split_manifest_sha256: str

    def validate(self) -> None:
        if self.source_partition != "model_training":
            raise E1LossError("edge positive weights must come only from model_training")
        if min(self.lane_successor_positive, self.control_lane_positive) < 1.0:
            raise E1LossError("edge positive weights must be at least one")
        if len(self.source_split_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_split_manifest_sha256
        ):
            raise E1LossError("edge positive-weight split identity is invalid")


@dataclass(frozen=True, slots=True)
class TopologyAssignments:
    lane_successor_target: Tensor
    lane_successor_mask: Tensor
    control_lane_target: Tensor
    control_lane_mask: Tensor


def topology_assignments(
    outputs: dict[str, Tensor], targets: E1Targets, matches: E0Matches
) -> TopologyAssignments:
    """Lift separate node matches into canonical query-indexed directed edge labels."""
    targets.validate()
    lane_logits = outputs["lane_successor_logits"]
    control_logits = outputs["control_lane_logits"]
    if lane_logits.ndim != 3 or lane_logits.shape[0] != 1:
        raise E1LossError("lane-successor logits must have batch size one")
    if control_logits.ndim != 3 or control_logits.shape[0] != 1:
        raise E1LossError("control-lane logits must have batch size one")
    lane_target = torch.zeros_like(lane_logits, dtype=torch.bool)
    lane_mask = torch.zeros_like(lane_logits, dtype=torch.bool)
    control_target = torch.zeros_like(control_logits, dtype=torch.bool)
    control_mask = torch.zeros_like(control_logits, dtype=torch.bool)
    lane_prediction = matches.lane.prediction
    lane_truth = matches.lane.target
    traffic_prediction = matches.traffic.prediction
    traffic_truth = matches.traffic.target
    if lane_prediction.numel():
        source_prediction = lane_prediction[:, None]
        target_prediction = lane_prediction[None, :]
        lane_mask[0, source_prediction, target_prediction] = source_prediction != target_prediction
        lane_target[0, source_prediction, target_prediction] = targets.lane_successor[
            lane_truth[:, None], lane_truth[None, :]
        ]
    if traffic_prediction.numel() and lane_prediction.numel():
        control_mask[0, traffic_prediction[:, None], lane_prediction[None, :]] = True
        control_target[0, traffic_prediction[:, None], lane_prediction[None, :]] = (
            targets.control_lane[traffic_truth[:, None], lane_truth[None, :]]
        )
    return TopologyAssignments(lane_target, lane_mask, control_target, control_mask)


def _masked_focal_loss(
    logits: Tensor, targets: Tensor, mask: Tensor, positive_weight: float
) -> Tensor:
    selected_logits = logits[mask]
    selected_targets = targets[mask].to(dtype=logits.dtype)
    if selected_logits.numel() == 0:
        return logits.sum() * 0.0
    probability = torch.sigmoid(selected_logits)
    cross_entropy = functional.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, reduction="none"
    )
    modulating = torch.where(selected_targets > 0.5, 1.0 - probability, probability).square()
    alpha = torch.where(selected_targets > 0.5, 0.25, 0.75)
    weights = torch.where(selected_targets > 0.5, positive_weight, 1.0)
    return (weights * alpha * modulating * cross_entropy).mean()


def _endpoint_continuity(centerline: Tensor, assignments: TopologyAssignments) -> Tensor:
    positive = assignments.lane_successor_target & assignments.lane_successor_mask
    indices = torch.nonzero(positive[0], as_tuple=False)
    if indices.numel() == 0:
        return centerline.sum() * 0.0
    source_end = centerline[0, indices[:, 0], -1]
    target_start = centerline[0, indices[:, 1], 0]
    return torch.linalg.vector_norm(source_end - target_start, dim=-1).mean()


def topology_only_losses(
    outputs: dict[str, Tensor],
    assignments: TopologyAssignments,
    edge_weights: E1EdgeWeights,
) -> dict[str, Tensor]:
    """Compute the production topology objective for oracle-node diagnostics."""
    edge_weights.validate()
    return {
        "lane_successor": _masked_focal_loss(
            outputs["lane_successor_logits"],
            assignments.lane_successor_target,
            assignments.lane_successor_mask,
            edge_weights.lane_successor_positive,
        ),
        "control_lane": _masked_focal_loss(
            outputs["control_lane_logits"],
            assignments.control_lane_target,
            assignments.control_lane_mask,
            edge_weights.control_lane_positive,
        ),
        "successor_endpoint_continuity": _endpoint_continuity(
            outputs["lane_centerline"], assignments
        ),
    }


def e1_losses(
    outputs: dict[str, Tensor],
    targets: E1Targets,
    matches: E0Matches,
    node_weights: E0TrainingWeights,
    edge_weights: E1EdgeWeights,
) -> dict[str, Tensor]:
    """Compute every E1 loss as a separately logged unweighted scalar."""
    assignments = topology_assignments(outputs, targets, matches)
    losses = e0_losses(outputs, targets.nodes, matches, node_weights)
    losses.update(topology_only_losses(outputs, assignments, edge_weights))
    if not all(value.ndim == 0 and torch.isfinite(value) for value in losses.values()):
        raise E1LossError("an E1 loss is nonfinite or nonscalar")
    return losses


E1_LOSS_WEIGHTS = {
    **E0_LOSS_WEIGHTS,
    "lane_successor": 3.0,
    "control_lane": 5.0,
    "successor_endpoint_continuity": 1.0,
}


def weighted_e1_total(losses: dict[str, Tensor]) -> Tensor:
    if set(losses) != set(E1_LOSS_WEIGHTS):
        raise E1LossError("E1 loss set differs from the frozen joint objective")
    return sum(losses[name] * E1_LOSS_WEIGHTS[name] for name in sorted(losses))


__all__ = [
    "E1_LOSS_WEIGHTS",
    "E1EdgeWeights",
    "E1LossError",
    "E1Targets",
    "TopologyAssignments",
    "e1_losses",
    "topology_assignments",
    "topology_only_losses",
    "weighted_e1_total",
]

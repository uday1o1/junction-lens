"""Hungarian matching and complete node losses for the E0 baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from torch import Tensor
from torch.nn import functional as functional


class E0LossError(ValueError):
    """Raised when a training target cannot satisfy the frozen E0 objective."""


@dataclass(frozen=True, slots=True)
class E0Targets:
    lane_centerline: Tensor
    lane_left_boundary: Tensor
    lane_right_boundary: Tensor
    lane_left_type: Tensor
    lane_right_type: Tensor
    lane_connector: Tensor
    traffic_boxes: Tensor
    traffic_category: Tensor
    traffic_attribute: Tensor
    area_points: Tensor
    area_valid: Tensor
    area_category: Tensor

    def validate(self) -> None:
        lanes = self.lane_centerline.shape[0]
        controls = self.traffic_boxes.shape[0]
        areas = self.area_points.shape[0]
        if self.lane_centerline.ndim != 3 or self.lane_centerline.shape[-1] != 3:
            raise E0LossError("lane centerline target shape is invalid")
        if self.lane_left_boundary.shape != self.lane_centerline.shape:
            raise E0LossError("left boundary target shape differs from centerline")
        if self.lane_right_boundary.shape != self.lane_centerline.shape:
            raise E0LossError("right boundary target shape differs from centerline")
        if any(value.shape != (lanes,) for value in (self.lane_left_type, self.lane_right_type)):
            raise E0LossError("lane boundary-type target shape is invalid")
        if self.lane_connector.shape != (lanes,):
            raise E0LossError("lane connector target shape is invalid")
        if self.traffic_boxes.shape != (controls, 4):
            raise E0LossError("traffic box target shape is invalid")
        if any(
            value.shape != (controls,) for value in (self.traffic_category, self.traffic_attribute)
        ):
            raise E0LossError("traffic class target shape is invalid")
        if self.area_points.ndim != 3 or self.area_points.shape[0] != areas:
            raise E0LossError("area point target shape is invalid")
        if self.area_valid.shape != self.area_points.shape[:2] or self.area_category.shape != (
            areas,
        ):
            raise E0LossError("area validity or category target shape is invalid")
        tensors = (
            self.lane_centerline,
            self.lane_left_boundary,
            self.lane_right_boundary,
            self.traffic_boxes,
            self.area_points,
        )
        if not all(torch.isfinite(value).all() for value in tensors):
            raise E0LossError("E0 targets contain nonfinite geometry")


@dataclass(frozen=True, slots=True)
class MatchIndices:
    prediction: Tensor
    target: Tensor


@dataclass(frozen=True, slots=True)
class E0Matches:
    lane: MatchIndices
    traffic: MatchIndices
    area: MatchIndices


@dataclass(frozen=True, slots=True)
class E0TrainingWeights:
    lane_existence_positive: float
    traffic_existence_positive: float
    area_existence_positive: float
    boundary_type: Tensor
    traffic_category: Tensor
    traffic_attribute: Tensor
    area_category: Tensor

    def to(self, device: torch.device) -> E0TrainingWeights:
        return E0TrainingWeights(
            self.lane_existence_positive,
            self.traffic_existence_positive,
            self.area_existence_positive,
            self.boundary_type.to(device),
            self.traffic_category.to(device),
            self.traffic_attribute.to(device),
            self.area_category.to(device),
        )


def unit_training_weights(device: torch.device) -> E0TrainingWeights:
    return E0TrainingWeights(
        1.0,
        1.0,
        1.0,
        torch.ones(3, device=device),
        torch.ones(2, device=device),
        torch.ones(13, device=device),
        torch.ones(2, device=device),
    )


def _focal_positive_cost(logits: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    return -0.25 * (1.0 - probability).square() * torch.log(probability.clamp_min(1.0e-8))


def _pairwise_l1(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.cdist(prediction.flatten(1), target.flatten(1), p=1) / prediction[0].numel()


def _pairwise_chamfer(prediction: Tensor, target: Tensor) -> Tensor:
    distances = torch.linalg.vector_norm(
        prediction[:, None, :, None, :] - target[None, :, None, :, :], dim=-1
    )
    return distances.amin(dim=3).mean(dim=2) + distances.amin(dim=2).mean(dim=2)


def _generalized_box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=2)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0.0).prod(dim=1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0.0).prod(dim=1)
    union = area_a[:, None] + area_b[None] - intersection
    iou = intersection / union.clamp_min(1.0e-8)
    enclosing_top_left = torch.minimum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    enclosing_bottom_right = torch.maximum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    enclosing = (enclosing_bottom_right - enclosing_top_left).clamp_min(0.0).prod(dim=2)
    return iou - (enclosing - union) / enclosing.clamp_min(1.0e-8)


def _hungarian(cost: Tensor) -> MatchIndices:
    if cost.ndim != 2:
        raise E0LossError("Hungarian cost must be a matrix")
    if cost.shape[1] == 0:
        empty = torch.empty(0, dtype=torch.int64, device=cost.device)
        return MatchIndices(empty, empty)
    if not torch.isfinite(cost).all():
        raise E0LossError("Hungarian cost contains a nonfinite value")
    prediction, target = linear_sum_assignment(cost.detach().cpu().numpy())
    return MatchIndices(
        torch.as_tensor(prediction, dtype=torch.int64, device=cost.device),
        torch.as_tensor(target, dtype=torch.int64, device=cost.device),
    )


def match_e0(outputs: dict[str, Tensor], targets: E0Targets) -> E0Matches:
    """Perform separate lane, control, and area assignments for batch size one."""
    targets.validate()
    if any(value.shape[0] != 1 for value in outputs.values()):
        raise E0LossError("E0 matching requires the declared batch size of one")
    lane_cost = 2.0 * _focal_positive_cost(outputs["lane_existence_logits"][0])[:, None]
    if len(targets.lane_centerline):
        lane_cost = lane_cost + 5.0 * _pairwise_l1(
            outputs["lane_centerline"][0], targets.lane_centerline
        )
        boundary = _pairwise_chamfer(
            outputs["lane_left_boundary"][0], targets.lane_left_boundary
        ) + _pairwise_chamfer(outputs["lane_right_boundary"][0], targets.lane_right_boundary)
        lane_cost = lane_cost + boundary
        connector_probability = torch.sigmoid(outputs["lane_connector_logits"][0])
        connector = targets.lane_connector.to(dtype=connector_probability.dtype)
        connector_cost = torch.abs(connector_probability[:, None] - connector[None])
        lane_cost = lane_cost + connector_cost
    else:
        lane_cost = lane_cost[:, :0]

    traffic_cost = 2.0 * _focal_positive_cost(outputs["traffic_existence_logits"][0])[:, None]
    if len(targets.traffic_boxes):
        traffic_cost = (
            traffic_cost
            + 5.0 * torch.cdist(outputs["traffic_boxes"][0], targets.traffic_boxes, p=1) / 4.0
        )
        traffic_cost = traffic_cost + 2.0 * (
            1.0 - _generalized_box_iou(outputs["traffic_boxes"][0], targets.traffic_boxes)
        )
        categories = functional.log_softmax(outputs["traffic_category_logits"][0], dim=1)
        attributes = functional.log_softmax(outputs["traffic_attribute_logits"][0], dim=1)
        traffic_cost = traffic_cost - categories[:, targets.traffic_category]
        traffic_cost = traffic_cost - attributes[:, targets.traffic_attribute]
    else:
        traffic_cost = traffic_cost[:, :0]

    area_cost = 2.0 * _focal_positive_cost(outputs["area_existence_logits"][0])[:, None]
    if len(targets.area_points):
        area_cost = area_cost + 4.0 * _pairwise_chamfer(
            outputs["area_points"][0], targets.area_points
        )
        categories = functional.log_softmax(outputs["area_category_logits"][0], dim=1)
        area_cost = area_cost - categories[:, targets.area_category]
    else:
        area_cost = area_cost[:, :0]
    return E0Matches(_hungarian(lane_cost), _hungarian(traffic_cost), _hungarian(area_cost))


def _sigmoid_focal_loss(logits: Tensor, targets: Tensor, positive_weight: float) -> Tensor:
    probability = torch.sigmoid(logits)
    cross_entropy = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    modulating = torch.where(targets > 0.5, 1.0 - probability, probability).square()
    alpha = torch.where(targets > 0.5, 0.25, 0.75)
    class_weight = torch.where(targets > 0.5, positive_weight, 1.0)
    return (class_weight * alpha * modulating * cross_entropy).mean()


def _existence_targets(logits: Tensor, matches: MatchIndices) -> Tensor:
    target = torch.zeros_like(logits)
    target[0, matches.prediction] = 1.0
    return target


def _laplace_nll(residual: Tensor, scale: Tensor) -> Tensor:
    if residual.numel() == 0:
        return scale.sum() * 0.0
    return (torch.abs(residual) / scale + torch.log(2.0 * scale)).mean()


def _zero(outputs: dict[str, Tensor]) -> Tensor:
    return outputs["lane_existence_logits"].sum() * 0.0


def e0_losses(
    outputs: dict[str, Tensor],
    targets: E0Targets,
    matches: E0Matches,
    weights: E0TrainingWeights | None = None,
) -> dict[str, Tensor]:
    """Compute every unweighted E0 node loss as an independently logged scalar."""
    lane_prediction, lane_target = matches.lane.prediction, matches.lane.target
    traffic_prediction, traffic_target = matches.traffic.prediction, matches.traffic.target
    area_prediction, area_target = matches.area.prediction, matches.area.target
    lane_geometry = torch.stack(
        (
            outputs["lane_centerline"][0, lane_prediction],
            outputs["lane_left_boundary"][0, lane_prediction],
            outputs["lane_right_boundary"][0, lane_prediction],
        ),
        dim=1,
    )
    target_lane_geometry = torch.stack(
        (
            targets.lane_centerline[lane_target],
            targets.lane_left_boundary[lane_target],
            targets.lane_right_boundary[lane_target],
        ),
        dim=1,
    )
    area_valid = targets.area_valid[area_target]
    predicted_area = outputs["area_points"][0, area_prediction]
    target_area = targets.area_points[area_target]
    valid_coordinates = area_valid[..., None].expand_as(target_area)
    lane_present = lane_prediction.numel() > 0
    traffic_present = traffic_prediction.numel() > 0
    area_present = area_prediction.numel() > 0
    zero = _zero(outputs)
    selected_weights = weights or unit_training_weights(outputs["lane_existence_logits"].device)
    losses = {
        "lane_existence": _sigmoid_focal_loss(
            outputs["lane_existence_logits"],
            _existence_targets(outputs["lane_existence_logits"], matches.lane),
            selected_weights.lane_existence_positive,
        ),
        "lane_centerline": (
            functional.smooth_l1_loss(
                outputs["lane_centerline"][0, lane_prediction],
                targets.lane_centerline[lane_target],
            )
            if lane_present
            else zero
        ),
        "lane_boundaries": (
            functional.smooth_l1_loss(lane_geometry[:, 1:], target_lane_geometry[:, 1:])
            if lane_present
            else zero
        ),
        "lane_left_boundary_type": (
            functional.cross_entropy(
                outputs["lane_left_boundary_logits"][0, lane_prediction],
                targets.lane_left_type[lane_target],
                weight=selected_weights.boundary_type,
            )
            if lane_present
            else zero
        ),
        "lane_right_boundary_type": (
            functional.cross_entropy(
                outputs["lane_right_boundary_logits"][0, lane_prediction],
                targets.lane_right_type[lane_target],
                weight=selected_weights.boundary_type,
            )
            if lane_present
            else zero
        ),
        "lane_connector": (
            functional.binary_cross_entropy_with_logits(
                outputs["lane_connector_logits"][0, lane_prediction],
                targets.lane_connector[lane_target].to(dtype=torch.float32),
            )
            if lane_present
            else zero
        ),
        "traffic_existence": _sigmoid_focal_loss(
            outputs["traffic_existence_logits"],
            _existence_targets(outputs["traffic_existence_logits"], matches.traffic),
            selected_weights.traffic_existence_positive,
        ),
        "traffic_box_l1": (
            functional.l1_loss(
                outputs["traffic_boxes"][0, traffic_prediction],
                targets.traffic_boxes[traffic_target],
            )
            if traffic_present
            else zero
        ),
        "traffic_box_giou": (
            (
                1.0
                - _generalized_box_iou(
                    outputs["traffic_boxes"][0, traffic_prediction],
                    targets.traffic_boxes[traffic_target],
                ).diag()
            ).mean()
            if traffic_present
            else zero
        ),
        "traffic_category": (
            functional.cross_entropy(
                outputs["traffic_category_logits"][0, traffic_prediction],
                targets.traffic_category[traffic_target],
                weight=selected_weights.traffic_category,
            )
            if traffic_present
            else zero
        ),
        "traffic_attribute": (
            functional.cross_entropy(
                outputs["traffic_attribute_logits"][0, traffic_prediction],
                targets.traffic_attribute[traffic_target],
                weight=selected_weights.traffic_attribute,
            )
            if traffic_present
            else zero
        ),
        "area_existence": _sigmoid_focal_loss(
            outputs["area_existence_logits"],
            _existence_targets(outputs["area_existence_logits"], matches.area),
            selected_weights.area_existence_positive,
        ),
        "area_category": (
            functional.cross_entropy(
                outputs["area_category_logits"][0, area_prediction],
                targets.area_category[area_target],
                weight=selected_weights.area_category,
            )
            if area_present
            else zero
        ),
        "area_geometry": (
            functional.smooth_l1_loss(
                predicted_area[valid_coordinates], target_area[valid_coordinates]
            )
            if area_present and bool(valid_coordinates.any())
            else zero
        ),
        "area_validity": (
            functional.binary_cross_entropy_with_logits(
                outputs["area_valid_logits"][0, area_prediction], area_valid.to(dtype=torch.float32)
            )
            if area_present
            else zero
        ),
        "geometry_laplace_nll": _laplace_nll(
            lane_geometry - target_lane_geometry,
            outputs["lane_geometry_scales"][0, lane_prediction],
        )
        + _laplace_nll(
            (
                outputs["traffic_boxes"][0, traffic_prediction]
                - targets.traffic_boxes[traffic_target]
            ),
            outputs["traffic_box_scales"][0, traffic_prediction],
        )
        + _laplace_nll(
            (predicted_area - target_area)[valid_coordinates],
            outputs["area_geometry_scales"][0, area_prediction][valid_coordinates],
        ),
    }
    if not all(value.ndim == 0 and torch.isfinite(value) for value in losses.values()):
        raise E0LossError("an E0 loss is nonfinite or nonscalar")
    return losses


E0_LOSS_WEIGHTS = {
    "lane_existence": 2.0,
    "lane_centerline": 5.0,
    "lane_boundaries": 2.0,
    "lane_left_boundary_type": 0.5,
    "lane_right_boundary_type": 0.5,
    "lane_connector": 1.0,
    "traffic_existence": 2.0,
    "traffic_box_l1": 5.0,
    "traffic_box_giou": 2.0,
    "traffic_category": 1.0,
    "traffic_attribute": 1.0,
    "area_existence": 1.0,
    "area_category": 1.0,
    "area_geometry": 1.0,
    "area_validity": 0.5,
    "geometry_laplace_nll": 1.0,
}


def weighted_e0_total(losses: dict[str, Tensor]) -> Tensor:
    if set(losses) != set(E0_LOSS_WEIGHTS):
        raise E0LossError("E0 loss set differs from the frozen objective")
    return sum(losses[name] * E0_LOSS_WEIGHTS[name] for name in sorted(losses))


__all__ = [
    "E0_LOSS_WEIGHTS",
    "E0LossError",
    "E0Matches",
    "E0Targets",
    "E0TrainingWeights",
    "e0_losses",
    "match_e0",
    "unit_training_weights",
    "weighted_e0_total",
]

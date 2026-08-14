"""Learned directed lane and control topology for the E1 joint model."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from junctionlens.model.e0_profile import E0Profile
from junctionlens.model.e1_profile import E1Profile
from junctionlens.model.reference import E0_OUTPUT_NAMES, ReferenceNodeModel, e0_outputs_by_name

E1_OUTPUT_NAMES = (*E0_OUTPUT_NAMES, "lane_successor_logits", "control_lane_logits")


class TopologyOrderingError(ValueError):
    """Raised when canonical and OpenLane topology axes are confused."""


def _normalized_heading(start: Tensor, end: Tensor) -> Tensor:
    return functional.normalize(end[..., :2] - start[..., :2], dim=-1, eps=1.0e-6)


def _successor_geometry(centerline: Tensor) -> Tensor:
    start = centerline[..., 0, :]
    end = centerline[..., -1, :]
    delta = start[:, None, :, :] - end[:, :, None, :]
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    heading = _normalized_heading(start, end)
    source_heading = heading[:, :, None, :].expand(-1, -1, centerline.shape[1], -1)
    target_heading = heading[:, None, :, :].expand(-1, centerline.shape[1], -1, -1)
    dot = (source_heading * target_heading).sum(dim=-1, keepdim=True)
    cross = (
        source_heading[..., 0] * target_heading[..., 1]
        - source_heading[..., 1] * target_heading[..., 0]
    )[..., None]
    return torch.cat((delta, distance, source_heading, target_heading, dot, cross), dim=-1)


def _project_normalized(
    points_vehicle: Tensor,
    intrinsic: Tensor,
    t_vehicle_camera: Tensor,
    *,
    image_height: int,
    image_width: int,
) -> tuple[Tensor, Tensor]:
    rotation_camera_to_vehicle = t_vehicle_camera[:, :3, :3]
    translation_camera_in_vehicle = t_vehicle_camera[:, :3, 3]
    centered = points_vehicle - translation_camera_in_vehicle[:, None, None, :]
    camera = torch.einsum("bij,blpj->blpi", rotation_camera_to_vehicle.transpose(1, 2), centered)
    projected = torch.einsum("bij,blpj->blpi", intrinsic, camera)
    depth = camera[..., 2]
    safe_depth = depth.clamp_min(1.0e-6)
    normalized = torch.stack(
        (
            projected[..., 0] / safe_depth / float(image_width),
            projected[..., 1] / safe_depth / float(image_height),
        ),
        dim=-1,
    )
    visible = (
        (depth > 1.0e-6)
        & (normalized[..., 0] >= 0.0)
        & (normalized[..., 0] <= 1.0)
        & (normalized[..., 1] >= 0.0)
        & (normalized[..., 1] <= 1.0)
    )
    return torch.where(visible[..., None], normalized, torch.zeros_like(normalized)), visible


def _control_geometry(
    centerline: Tensor,
    boxes: Tensor,
    intrinsic: Tensor,
    t_vehicle_camera: Tensor,
    *,
    image_height: int,
    image_width: int,
) -> Tensor:
    endpoints = centerline[..., (0, -1), :]
    projected, visible = _project_normalized(
        endpoints,
        intrinsic,
        t_vehicle_camera,
        image_height=image_height,
        image_width=image_width,
    )
    box_center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    relative = projected[:, None, :, :, :] - box_center[:, :, None, None, :]
    lane_direction = projected[..., 1, :] - projected[..., 0, :]
    lane_direction = lane_direction[:, None].expand(-1, boxes.shape[1], -1, -1)
    visibility = visible.to(dtype=centerline.dtype)
    visibility = visibility[:, None].expand(-1, boxes.shape[1], -1, -1)
    return torch.cat((relative.flatten(-2), lane_direction, visibility), dim=-1)


class LearnedTopologyHeads(nn.Module):  # type: ignore[misc]
    """Bilinear edge scores augmented by directed geometric evidence."""

    def __init__(self, base: E0Profile, profile: E1Profile) -> None:
        super().__init__()
        profile.validate_base(base)
        hidden = base.architecture.hidden_dimension
        edge = profile.topology.edge_dimension
        geometry_hidden = profile.topology.geometry_hidden_dimension
        self.lane_source = nn.Linear(hidden, edge, bias=False)
        self.lane_target = nn.Linear(hidden, edge, bias=False)
        self.control_source = nn.Linear(hidden, edge, bias=False)
        self.control_target = nn.Linear(hidden, edge, bias=False)
        self.successor_geometry = nn.Sequential(
            nn.Linear(10, geometry_hidden), nn.GELU(), nn.Linear(geometry_hidden, 1)
        )
        self.control_geometry = nn.Sequential(
            nn.Linear(8, geometry_hidden), nn.GELU(), nn.Linear(geometry_hidden, 1)
        )
        self.edge_scale = math.sqrt(edge)
        self.allow_lane_self_edges = profile.topology.allow_lane_self_edges
        self.image_height = base.input.height
        self.image_width = base.input.width

    def forward(
        self,
        lane_features: Tensor,
        control_features: Tensor,
        lane_centerline: Tensor,
        control_boxes: Tensor,
        front_center_intrinsic: Tensor,
        front_center_t_vehicle_camera: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lane_logits = (
            torch.matmul(
                self.lane_source(lane_features), self.lane_target(lane_features).transpose(1, 2)
            )
            / self.edge_scale
        )
        lane_logits = lane_logits + self.successor_geometry(
            _successor_geometry(lane_centerline)
        ).squeeze(-1)
        if not self.allow_lane_self_edges:
            diagonal = torch.eye(lane_logits.shape[1], dtype=torch.bool, device=lane_logits.device)[
                None
            ]
            lane_logits = lane_logits.masked_fill(diagonal, -30.0)

        control_logits = (
            torch.matmul(
                self.control_source(control_features),
                self.control_target(lane_features).transpose(1, 2),
            )
            / self.edge_scale
        )
        control_logits = control_logits + self.control_geometry(
            _control_geometry(
                lane_centerline,
                control_boxes,
                front_center_intrinsic,
                front_center_t_vehicle_camera,
                image_height=self.image_height,
                image_width=self.image_width,
            )
        ).squeeze(-1)
        return lane_logits, control_logits


class JointGraphModel(ReferenceNodeModel):
    """E1 model with shared node queries and learned topology heads."""

    def __init__(self, base: E0Profile, profile: E1Profile) -> None:
        super().__init__(base)
        self.e1_profile = profile
        self.topology = LearnedTopologyHeads(base, profile)

    def forward(
        self,
        images: Tensor,
        camera_valid: Tensor,
        intrinsics: Tensor,
        t_vehicle_camera: Tensor,
        ego_motion_previous_to_current: Tensor,
        temporal_valid: Tensor,
    ) -> tuple[Tensor, ...]:
        lane, traffic, area = self.decode_queries(
            images,
            camera_valid,
            intrinsics,
            t_vehicle_camera,
            ego_motion_previous_to_current,
            temporal_valid,
        )
        node_outputs = self.node_outputs(lane, traffic, area)
        named = e0_outputs_by_name(node_outputs)
        current = self.profile.input.current_timestamp_index
        front_center = self.e1_profile.topology.front_center_camera_index
        lane_logits, control_logits = self.topology(
            lane,
            traffic,
            named["lane_centerline"],
            named["traffic_boxes"],
            intrinsics[:, current, front_center],
            t_vehicle_camera[:, current, front_center],
        )
        return (*node_outputs, lane_logits, control_logits)


def e1_outputs_by_name(outputs: Sequence[Tensor]) -> dict[str, Tensor]:
    if len(outputs) != len(E1_OUTPUT_NAMES):
        raise ValueError("E1 output count differs from the frozen joint tensor contract")
    return dict(zip(E1_OUTPUT_NAMES, outputs, strict=True))


def canonical_control_lane_to_openlane(
    control_lane: Tensor, *, control_count: int, lane_count: int
) -> Tensor:
    """Convert canonical control-major scores to OpenLane lane-major ordering."""
    if control_lane.ndim < 2 or control_lane.shape[-2:] != (control_count, lane_count):
        raise TopologyOrderingError(
            "canonical control-lane topology must be control-major and lane-minor"
        )
    return control_lane.transpose(-1, -2)


def openlane_lane_traffic_to_canonical(
    lane_traffic: Tensor, *, lane_count: int, control_count: int
) -> Tensor:
    """Convert OpenLane lane-major topology to canonical control-major ordering."""
    if lane_traffic.ndim < 2 or lane_traffic.shape[-2:] != (lane_count, control_count):
        raise TopologyOrderingError(
            "OpenLane lane-traffic topology must be lane-major and traffic-minor"
        )
    return lane_traffic.transpose(-1, -2)


__all__ = [
    "E1_OUTPUT_NAMES",
    "JointGraphModel",
    "LearnedTopologyHeads",
    "TopologyOrderingError",
    "canonical_control_lane_to_openlane",
    "e1_outputs_by_name",
    "openlane_lane_traffic_to_canonical",
]

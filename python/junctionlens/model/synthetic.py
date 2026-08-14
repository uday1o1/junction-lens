"""Deterministic unrestricted inputs and targets for the M0 model gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from junctionlens.model.profile import M0ModelProfile


@dataclass(frozen=True, slots=True)
class MicroTargets:
    lane_existence: Tensor
    centerline: Tensor
    left_boundary: Tensor
    right_boundary: Tensor
    left_type: Tensor
    right_type: Tensor
    connector: Tensor
    traffic_existence: Tensor
    traffic_box: Tensor
    traffic_category: Tensor
    traffic_attribute: Tensor
    area_existence: Tensor
    area_category: Tensor
    area_points: Tensor


def make_micro_inputs(
    profile: M0ModelProfile,
    frame_indices: Tensor,
    *,
    spatial_size: int | tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Encode frame identity in camera colors while preserving the public input surface."""
    batch = frame_indices.shape[0]
    shape = profile.model
    camera_count = profile.input.cameras
    height, width = (spatial_size, spatial_size) if isinstance(spatial_size, int) else spatial_size
    images = torch.zeros(
        batch,
        profile.input.timestamps,
        camera_count,
        profile.input.channels,
        height,
        width,
        dtype=torch.float32,
    )
    for camera_index in range(camera_count):
        bit = ((frame_indices >> camera_index) & 1).to(torch.float32)
        images[:, 1, camera_index, 0] = bit[:, None, None]
        images[:, 1, camera_index, 1] = 1.0 - bit[:, None, None]
        images[:, 0, camera_index, 2] = bit[:, None, None] * 0.5
    camera_valid = torch.ones(batch, 2, camera_count, dtype=torch.bool)
    intrinsics = torch.zeros(batch, 2, camera_count, 3, 3, dtype=torch.float32)
    intrinsics[..., 0, 0] = float(profile.input.width)
    intrinsics[..., 1, 1] = float(profile.input.height)
    intrinsics[..., 0, 2] = float(profile.input.width) / 2.0
    intrinsics[..., 1, 2] = float(profile.input.height) / 2.0
    intrinsics[..., 2, 2] = 1.0
    transforms = (
        torch.eye(4, dtype=torch.float32)
        .reshape(1, 1, 1, 4, 4)
        .repeat(batch, 2, camera_count, 1, 1)
    )
    camera_offsets = torch.linspace(-1.75, 1.75, camera_count)
    transforms[..., 1, 3] = camera_offsets.reshape(1, 1, camera_count)
    ego_motion = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).repeat(batch, 1, 1)
    ego_motion[:, 0, 3] = frame_indices.to(torch.float32) / 100.0
    temporal_valid = torch.ones(batch, dtype=torch.bool)
    if shape.lane_queries < 1 or shape.traffic_queries < 1 or shape.area_queries < 1:
        raise ValueError("micro-overfit requires at least one query of every node type")
    return images, camera_valid, intrinsics, transforms, ego_motion, temporal_valid


def make_micro_targets(profile: M0ModelProfile, frame_indices: Tensor) -> MicroTargets:
    """Create one matched node of each type per fixed synthetic frame."""
    batch = frame_indices.shape[0]
    shape = profile.model
    lane_existence = torch.zeros(batch, shape.lane_queries)
    traffic_existence = torch.zeros(batch, shape.traffic_queries)
    area_existence = torch.zeros(batch, shape.area_queries)
    lane_existence[:, 0] = 1.0
    traffic_existence[:, 0] = 1.0
    area_existence[:, 0] = 1.0
    longitudinal = torch.linspace(0.0, 10.0, shape.lane_points).reshape(1, -1)
    lateral = (frame_indices.to(torch.float32) - 15.5).reshape(-1, 1) / 8.0
    height = torch.zeros(batch, shape.lane_points)
    centerline = torch.stack(
        (longitudinal.repeat(batch, 1), lateral.repeat(1, shape.lane_points), height), dim=2
    )
    left_boundary = centerline.clone()
    right_boundary = centerline.clone()
    left_boundary[:, :, 1] += 1.75
    right_boundary[:, :, 1] -= 1.75
    box_x = 0.1 + (frame_indices.to(torch.float32) % 5.0) * 0.02
    box_y = 0.2 + (frame_indices.to(torch.float32) % 3.0) * 0.02
    traffic_box = torch.stack((box_x, box_y, box_x + 0.1, box_y + 0.15), dim=1)
    area_points = torch.zeros(batch, shape.area_points, 3)
    area_points[:, :, 0] = torch.linspace(-2.0, 2.0, shape.area_points)
    area_points[:, :, 1] = lateral
    return MicroTargets(
        lane_existence=lane_existence,
        centerline=centerline,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        left_type=frame_indices % shape.lane_boundary_classes,
        right_type=(frame_indices + 1) % shape.lane_boundary_classes,
        connector=frame_indices.to(torch.float32) % 2.0,
        traffic_existence=traffic_existence,
        traffic_box=traffic_box,
        traffic_category=frame_indices % shape.traffic_categories,
        traffic_attribute=frame_indices % shape.traffic_attributes,
        area_existence=area_existence,
        area_category=frame_indices % shape.area_categories,
        area_points=area_points,
    )

"""Calibration-aware M0 graph model used to prove export and runtime interfaces."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from junctionlens.model.profile import M0ModelProfile

INPUT_NAMES = (
    "images",
    "camera_valid",
    "intrinsics",
    "t_vehicle_camera",
    "ego_motion_previous_to_current",
    "temporal_valid",
)

OUTPUT_NAMES = (
    "lane_existence_logits",
    "lane_centerline",
    "lane_left_boundary",
    "lane_right_boundary",
    "lane_left_boundary_logits",
    "lane_right_boundary_logits",
    "lane_connector_logits",
    "lane_geometry_scales",
    "lane_track_embeddings",
    "traffic_existence_logits",
    "traffic_boxes",
    "traffic_category_logits",
    "traffic_attribute_logits",
    "traffic_box_scales",
    "traffic_track_embeddings",
    "area_existence_logits",
    "area_category_logits",
    "area_points",
    "area_valid_logits",
    "area_geometry_scales",
    "area_track_embeddings",
    "lane_successor_logits",
    "control_lane_logits",
)


def _linear_last(linear: nn.Linear, features: Tensor) -> Tensor:
    """Apply a linear layer without freezing leading dimensions during ONNX export."""
    leading_shape = features.shape[:-1]
    flattened = features.reshape(-1, features.shape[-1])
    return linear(flattened).reshape(*leading_shape, linear.out_features)


def _canonicalize_output(value: Tensor, zero_deadband: float) -> Tensor:
    """Canonicalize negligible values while retaining a straight-through gradient."""
    canonical = torch.where(torch.abs(value) < zero_deadband, torch.zeros_like(value), value)
    return value + (canonical - value).detach()


def project_image_center_rays(
    intrinsics: Tensor,
    t_vehicle_camera: Tensor,
    image_height: int,
    image_width: int,
) -> Tensor:
    """Project each image-center ray through calibration into the vehicle frame."""
    focal_x = intrinsics[..., 0, 0].clamp_min(1e-6)
    focal_y = intrinsics[..., 1, 1].clamp_min(1e-6)
    ray_camera = torch.stack(
        (
            (float(image_width) / 2.0 - intrinsics[..., 0, 2]) / focal_x,
            (float(image_height) / 2.0 - intrinsics[..., 1, 2]) / focal_y,
            torch.ones_like(focal_x),
        ),
        dim=-1,
    )
    ray_vehicle = torch.matmul(t_vehicle_camera[..., :3, :3], ray_camera.unsqueeze(-1)).squeeze(-1)
    return functional.normalize(ray_vehicle, dim=-1)


class _QueryBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, count: int, hidden: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(count, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, context: Tensor) -> Tensor:
        features = context[:, None, :] + self.queries[None, :, :]
        residual = _linear_last(self.mlp[0], features)
        residual = self.mlp[1](residual)
        residual = _linear_last(self.mlp[2], residual)
        return self.norm(features + residual)


class M0GraphModel(nn.Module):  # type: ignore[misc]
    """Bounded feasibility model with the complete V1 node and edge tensor surface."""

    def __init__(self, profile: M0ModelProfile) -> None:
        super().__init__()
        self.profile = profile
        shape = profile.model
        hidden = shape.hidden_dimension
        encoder_inputs = 48 + 16 + 64 + 48 + 48 + 12 + 1
        self.encoder = nn.Sequential(
            nn.Linear(encoder_inputs, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.lane_queries = _QueryBlock(shape.lane_queries, hidden)
        self.traffic_queries = _QueryBlock(shape.traffic_queries, hidden)
        self.area_queries = _QueryBlock(shape.area_queries, hidden)

        self.lane_existence = nn.Linear(hidden, 1)
        self.lane_geometry = nn.Linear(hidden, shape.lane_points * 3 * 3)
        self.lane_left_type = nn.Linear(hidden, shape.lane_boundary_classes)
        self.lane_right_type = nn.Linear(hidden, shape.lane_boundary_classes)
        self.lane_connector = nn.Linear(hidden, 1)
        self.lane_scales = nn.Linear(hidden, shape.lane_points * 3 * 3)
        self.lane_embedding = nn.Linear(hidden, shape.track_embedding_dimension)

        self.traffic_existence = nn.Linear(hidden, 1)
        self.traffic_box = nn.Linear(hidden, 4)
        self.traffic_category = nn.Linear(hidden, shape.traffic_categories)
        self.traffic_attribute = nn.Linear(hidden, shape.traffic_attributes)
        self.traffic_scales = nn.Linear(hidden, 4)
        self.traffic_embedding = nn.Linear(hidden, shape.track_embedding_dimension)

        self.area_existence = nn.Linear(hidden, 1)
        self.area_category = nn.Linear(hidden, shape.area_categories)
        self.area_geometry = nn.Linear(hidden, shape.area_points * 3)
        self.area_valid = nn.Linear(hidden, shape.area_points)
        self.area_scales = nn.Linear(hidden, shape.area_points * 3)
        self.area_embedding = nn.Linear(hidden, shape.track_embedding_dimension)

        self.lane_edge_source = nn.Linear(hidden, hidden, bias=False)
        self.lane_edge_target = nn.Linear(hidden, hidden, bias=False)
        self.control_edge_source = nn.Linear(hidden, hidden, bias=False)
        self.control_edge_target = nn.Linear(hidden, hidden, bias=False)

    def _encode(
        self,
        images: Tensor,
        camera_valid: Tensor,
        intrinsics: Tensor,
        t_vehicle_camera: Tensor,
        ego_motion_previous_to_current: Tensor,
        temporal_valid: Tensor,
    ) -> Tensor:
        mask = camera_valid.to(dtype=images.dtype)
        pooled_pixels = images.mean(dim=(-1, -2)) * mask[..., None]
        intrinsic_features = torch.stack(
            (
                intrinsics[..., 0, 0] / float(self.profile.input.width),
                intrinsics[..., 1, 1] / float(self.profile.input.height),
                intrinsics[..., 0, 2] / float(self.profile.input.width),
                intrinsics[..., 1, 2] / float(self.profile.input.height),
            ),
            dim=-1,
        )
        intrinsic_features = intrinsic_features * mask[..., None]
        translations = t_vehicle_camera[..., :3, 3] * mask[..., None] / 10.0
        projected_rays = project_image_center_rays(
            intrinsics,
            t_vehicle_camera,
            self.profile.input.height,
            self.profile.input.width,
        )
        projected_rays = projected_rays * mask[..., None]
        batch = images.shape[0]
        encoded = torch.cat(
            (
                pooled_pixels.reshape(batch, -1),
                mask.reshape(batch, -1),
                intrinsic_features.reshape(batch, -1),
                translations.reshape(batch, -1),
                projected_rays.reshape(batch, -1),
                ego_motion_previous_to_current[..., :3, :].reshape(batch, -1) / 10.0,
                temporal_valid.to(dtype=images.dtype).reshape(batch, 1),
            ),
            dim=1,
        )
        return self.encoder(encoded)

    def forward(
        self,
        images: Tensor,
        camera_valid: Tensor,
        intrinsics: Tensor,
        t_vehicle_camera: Tensor,
        ego_motion_previous_to_current: Tensor,
        temporal_valid: Tensor,
    ) -> tuple[Tensor, ...]:
        context = self._encode(
            images,
            camera_valid,
            intrinsics,
            t_vehicle_camera,
            ego_motion_previous_to_current,
            temporal_valid,
        )
        lane = self.lane_queries(context)
        traffic = self.traffic_queries(context)
        area = self.area_queries(context)
        shape = self.profile.model
        lane_geometry = _linear_last(self.lane_geometry, lane).reshape(
            images.shape[0], shape.lane_queries, 3, shape.lane_points, 3
        )
        lane_source = _linear_last(self.lane_edge_source, lane)
        lane_target = _linear_last(self.lane_edge_target, lane)
        lane_edges = torch.matmul(lane_source, lane_target.transpose(1, 2)) / (
            shape.hidden_dimension**0.5
        )
        diagonal = torch.eye(
            shape.lane_queries, dtype=torch.bool, device=lane_edges.device
        ).unsqueeze(0)
        lane_edges = lane_edges.masked_fill(diagonal, -20.0)
        control_edges = torch.matmul(
            _linear_last(self.control_edge_source, traffic),
            _linear_last(self.control_edge_target, lane).transpose(1, 2),
        ) / (shape.hidden_dimension**0.5)
        raw_outputs = (
            _linear_last(self.lane_existence, lane).squeeze(-1),
            lane_geometry[:, :, 0],
            lane_geometry[:, :, 1],
            lane_geometry[:, :, 2],
            _linear_last(self.lane_left_type, lane),
            _linear_last(self.lane_right_type, lane),
            _linear_last(self.lane_connector, lane).squeeze(-1),
            functional.softplus(
                _linear_last(self.lane_scales, lane).reshape(
                    images.shape[0], shape.lane_queries, 3, shape.lane_points, 3
                )
            )
            + 1e-4,
            functional.normalize(_linear_last(self.lane_embedding, lane), dim=-1),
            _linear_last(self.traffic_existence, traffic).squeeze(-1),
            torch.sigmoid(_linear_last(self.traffic_box, traffic)),
            _linear_last(self.traffic_category, traffic),
            _linear_last(self.traffic_attribute, traffic),
            functional.softplus(_linear_last(self.traffic_scales, traffic)) + 1e-4,
            functional.normalize(_linear_last(self.traffic_embedding, traffic), dim=-1),
            _linear_last(self.area_existence, area).squeeze(-1),
            _linear_last(self.area_category, area),
            _linear_last(self.area_geometry, area).reshape(
                images.shape[0], shape.area_queries, shape.area_points, 3
            ),
            _linear_last(self.area_valid, area),
            functional.softplus(
                _linear_last(self.area_scales, area).reshape(
                    images.shape[0], shape.area_queries, shape.area_points, 3
                )
            )
            + 1e-4,
            functional.normalize(_linear_last(self.area_embedding, area), dim=-1),
            lane_edges,
            control_edges,
        )
        return tuple(
            _canonicalize_output(value, self.profile.export.output_zero_deadband)
            for value in raw_outputs
        )


def outputs_by_name(outputs: Sequence[Tensor]) -> dict[str, Tensor]:
    """Associate the frozen output tuple with its ONNX tensor names."""
    if len(outputs) != len(OUTPUT_NAMES):
        raise ValueError("model output count differs from the frozen tensor contract")
    return dict(zip(OUTPUT_NAMES, outputs, strict=True))

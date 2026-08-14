"""Shared node architecture for the E0, E1, E2, and E3 model family."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torchvision.models import efficientnet_b0  # type: ignore[import-untyped]

from junctionlens.model.e0_profile import E0Profile

E0_OUTPUT_NAMES = (
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
)


class EfficientNetB0Fpn(nn.Module):  # type: ignore[misc]
    """EfficientNet-B0 stride 8, 16, and 32 features fused at stride 8."""

    def __init__(self, output_channels: int = 128) -> None:
        super().__init__()
        self.features = efficientnet_b0(weights=None).features
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, output_channels, 1) for channels in (40, 112, 320)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(output_channels * 3, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, images: Tensor) -> Tensor:
        selected: list[Tensor] = []
        value = images
        for index, block in enumerate(self.features):
            value = block(value)
            if index in {3, 5, 7}:
                selected.append(value)
        if len(selected) != 3:
            raise RuntimeError("EfficientNet-B0 feature layout differs from the locked profile")
        target_size = selected[0].shape[-2:]
        projected = [
            projection(feature)
            if feature.shape[-2:] == target_size
            else functional.interpolate(
                projection(feature), size=target_size, mode="bilinear", align_corners=False
            )
            for projection, feature in zip(self.projections, selected, strict=True)
        ]
        return self.fuse(torch.cat(projected, dim=1))


def _rigid_vehicle_to_camera(t_vehicle_camera: Tensor, points_vehicle: Tensor) -> Tensor:
    rotation_camera_to_vehicle = t_vehicle_camera[..., :3, :3]
    translation_camera_in_vehicle = t_vehicle_camera[..., :3, 3]
    rotation_vehicle_to_camera = rotation_camera_to_vehicle.transpose(-1, -2)
    centered = points_vehicle - translation_camera_in_vehicle[..., None, :]
    return torch.matmul(rotation_vehicle_to_camera[..., None, :, :], centered[..., None]).squeeze(
        -1
    )


def _projection_grid(
    intrinsics: Tensor,
    t_vehicle_camera: Tensor,
    points_vehicle: Tensor,
    image_height: int,
    image_width: int,
) -> tuple[Tensor, Tensor]:
    camera_points = _rigid_vehicle_to_camera(t_vehicle_camera, points_vehicle)
    depth = camera_points[..., 2]
    pixels_homogeneous = torch.matmul(
        intrinsics[..., None, :, :], camera_points[..., None]
    ).squeeze(-1)
    safe_depth = depth.clamp_min(1.0e-6)
    u = pixels_homogeneous[..., 0] / safe_depth
    v = pixels_homogeneous[..., 1] / safe_depth
    visible = (depth > 1.0e-6) & (u >= 0.0) & (u <= image_width - 1.0)
    visible = visible & (v >= 0.0) & (v <= image_height - 1.0)
    grid = torch.stack(
        (
            u * (2.0 / float(image_width - 1)) - 1.0,
            v * (2.0 / float(image_height - 1)) - 1.0,
        ),
        dim=-1,
    )
    return grid, visible


class GroundPlaneProjector(nn.Module):  # type: ignore[misc]
    """Calibration-driven ground-plane projection with fail-closed camera masking."""

    def __init__(self, profile: E0Profile) -> None:
        super().__init__()
        architecture = profile.architecture
        x = torch.arange(architecture.bev_shape[0], dtype=torch.float32)
        y = torch.arange(architecture.bev_shape[1], dtype=torch.float32)
        x = architecture.bev_x_range_m[0] + (x + 0.5) * architecture.bev_cell_size_m
        y = architecture.bev_y_range_m[0] + (y + 0.5) * architecture.bev_cell_size_m
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack((grid_x, grid_y, torch.zeros_like(grid_x)), dim=-1)
        self.register_buffer("points_vehicle", points.reshape(-1, 3), persistent=True)
        self.bev_shape = architecture.bev_shape
        self.image_height = profile.input.height
        self.image_width = profile.input.width
        self.camera_weight = nn.Conv2d(architecture.fpn_channels, 1, 1)
        self.height_residual = nn.Parameter(
            torch.zeros(1, 1, architecture.bev_shape[0], architecture.bev_shape[1])
        )
        self.height_fusion = nn.Sequential(
            nn.Conv2d(architecture.fpn_channels + 1, architecture.fpn_channels, 1),
            nn.SiLU(inplace=True),
        )

    def _grid(self, intrinsics: Tensor, transforms: Tensor) -> tuple[Tensor, Tensor]:
        batch, cameras = intrinsics.shape[:2]
        points = self.points_vehicle.reshape(1, 1, -1, 3).expand(batch, cameras, -1, -1)
        grid, visible = _projection_grid(
            intrinsics,
            transforms,
            points,
            self.image_height,
            self.image_width,
        )
        return (
            grid.reshape(batch, cameras, self.bev_shape[0], self.bev_shape[1], 2),
            visible.reshape(batch, cameras, self.bev_shape[0], self.bev_shape[1]),
        )

    def vectorized(
        self,
        camera_features: Tensor,
        camera_valid: Tensor,
        intrinsics: Tensor,
        transforms: Tensor,
    ) -> Tensor:
        batch, cameras, channels, height, width = camera_features.shape
        grid, visible = self._grid(intrinsics, transforms)
        sampled = functional.grid_sample(
            camera_features.reshape(batch * cameras, channels, height, width),
            grid.reshape(batch * cameras, self.bev_shape[0], self.bev_shape[1], 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, cameras, channels, *self.bev_shape)
        eligible = visible & camera_valid[..., None, None]
        logits = self.camera_weight(sampled.reshape(batch * cameras, channels, *self.bev_shape))
        logits = logits.reshape(batch, cameras, *self.bev_shape)
        logits = logits.masked_fill(~eligible, -torch.inf)
        maximum = torch.amax(logits, dim=1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        weights = torch.exp(logits - maximum) * eligible.to(dtype=sampled.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        bev = (sampled * weights[:, :, None]).sum(dim=1)
        height = self.height_residual.expand(batch, -1, -1, -1)
        return self.height_fusion(torch.cat((bev, height), dim=1))

    def reference(
        self,
        camera_features: Tensor,
        camera_valid: Tensor,
        intrinsics: Tensor,
        transforms: Tensor,
    ) -> Tensor:
        """Slow camera-loop implementation used only for numerical qualification."""
        sampled_views = []
        eligibility = []
        grid, visible = self._grid(intrinsics, transforms)
        for camera in range(camera_features.shape[1]):
            sampled_views.append(
                functional.grid_sample(
                    camera_features[:, camera],
                    grid[:, camera],
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
            )
            eligibility.append(visible[:, camera] & camera_valid[:, camera, None, None])
        sampled = torch.stack(sampled_views, dim=1)
        eligible = torch.stack(eligibility, dim=1)
        logits = torch.stack(
            [
                self.camera_weight(sampled[:, camera]).squeeze(1)
                for camera in range(sampled.shape[1])
            ],
            dim=1,
        ).masked_fill(~eligible, -torch.inf)
        maximum = torch.amax(logits, dim=1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        weights = torch.exp(logits - maximum) * eligible.to(dtype=sampled.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        bev = (sampled * weights[:, :, None]).sum(dim=1)
        return self.height_fusion(
            torch.cat((bev, self.height_residual.expand(bev.shape[0], -1, -1, -1)), dim=1)
        )


class SharedNodeDecoder(nn.Module):  # type: ignore[misc]
    """Four-layer shared transformer decoder with type-specific query banks."""

    def __init__(self, profile: E0Profile) -> None:
        super().__init__()
        architecture = profile.architecture
        self.memory_shape = architecture.decoder_memory_shape
        self.memory_projection = nn.Conv2d(
            architecture.fpn_channels, architecture.hidden_dimension, 1
        )
        layer = nn.TransformerDecoderLayer(
            d_model=architecture.hidden_dimension,
            nhead=architecture.attention_heads,
            dim_feedforward=architecture.hidden_dimension * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, architecture.decoder_layers)
        total_queries = (
            architecture.lane_queries + architecture.traffic_queries + architecture.area_queries
        )
        self.queries = nn.Parameter(torch.empty(total_queries, architecture.hidden_dimension))
        self.query_type = nn.Parameter(torch.empty(3, architecture.hidden_dimension))
        self.memory_position = nn.Parameter(
            torch.empty(self.memory_shape[0] * self.memory_shape[1], architecture.hidden_dimension)
        )
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.query_type, std=0.02)
        nn.init.normal_(self.memory_position, std=0.02)
        self.counts = (
            architecture.lane_queries,
            architecture.traffic_queries,
            architecture.area_queries,
        )

    def forward(self, bev: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        memory = functional.adaptive_avg_pool2d(bev, self.memory_shape)
        memory = self.memory_projection(memory).flatten(2).transpose(1, 2)
        memory = memory + self.memory_position[None]
        lane_count, traffic_count, _ = self.counts
        type_index = torch.cat(
            (
                torch.zeros(lane_count, dtype=torch.int64, device=bev.device),
                torch.ones(traffic_count, dtype=torch.int64, device=bev.device),
                torch.full((self.counts[2],), 2, dtype=torch.int64, device=bev.device),
            )
        )
        queries = self.queries + self.query_type[type_index]
        decoded = self.decoder(queries[None].expand(bev.shape[0], -1, -1), memory)
        values = torch.split(decoded, self.counts, dim=1)
        return values[0], values[1], values[2]


class ReferenceNodeModel(nn.Module):  # type: ignore[misc]
    """Production shared node architecture used by the independent E0 baseline."""

    def __init__(self, profile: E0Profile) -> None:
        super().__init__()
        self.profile = profile
        architecture = profile.architecture
        hidden = architecture.hidden_dimension
        self.backbone = EfficientNetB0Fpn(architecture.fpn_channels)
        self.projector = GroundPlaneProjector(profile)
        self.decoder = SharedNodeDecoder(profile)

        self.lane_existence = nn.Linear(hidden, 1)
        self.lane_geometry = nn.Linear(hidden, architecture.lane_points * 3 * 3)
        self.lane_left_type = nn.Linear(hidden, architecture.boundary_classes)
        self.lane_right_type = nn.Linear(hidden, architecture.boundary_classes)
        self.lane_connector = nn.Linear(hidden, 1)
        self.lane_scales = nn.Linear(hidden, architecture.lane_points * 3 * 3)
        self.lane_embedding = nn.Linear(hidden, architecture.track_embedding_dimension)
        self.traffic_existence = nn.Linear(hidden, 1)
        self.traffic_box = nn.Linear(hidden, 4)
        self.traffic_category = nn.Linear(hidden, architecture.traffic_categories)
        self.traffic_attribute = nn.Linear(hidden, architecture.traffic_attributes)
        self.traffic_scales = nn.Linear(hidden, 4)
        self.traffic_embedding = nn.Linear(hidden, architecture.track_embedding_dimension)
        self.area_existence = nn.Linear(hidden, 1)
        self.area_category = nn.Linear(hidden, architecture.area_categories)
        self.area_geometry = nn.Linear(hidden, architecture.area_points * 3)
        self.area_valid = nn.Linear(hidden, architecture.area_points)
        self.area_scales = nn.Linear(hidden, architecture.area_points * 3)
        self.area_embedding = nn.Linear(hidden, architecture.track_embedding_dimension)

    def forward(
        self,
        images: Tensor,
        camera_valid: Tensor,
        intrinsics: Tensor,
        t_vehicle_camera: Tensor,
        ego_motion_previous_to_current: Tensor,
        temporal_valid: Tensor,
    ) -> tuple[Tensor, ...]:
        del ego_motion_previous_to_current, temporal_valid
        current = self.profile.input.current_timestamp_index
        batch, cameras = images.shape[0], images.shape[2]
        current_images = images[:, current]
        features = self.backbone(current_images.reshape(-1, *current_images.shape[2:]))
        features = features.reshape(batch, cameras, *features.shape[1:])
        bev = self.projector.vectorized(
            features,
            camera_valid[:, current],
            intrinsics[:, current],
            t_vehicle_camera[:, current],
        )
        lane, traffic, area = self.decoder(bev)
        architecture = self.profile.architecture
        lane_geometry = self.lane_geometry(lane).reshape(
            batch, architecture.lane_queries, 3, architecture.lane_points, 3
        )
        return (
            self.lane_existence(lane).squeeze(-1),
            lane_geometry[:, :, 0],
            lane_geometry[:, :, 1],
            lane_geometry[:, :, 2],
            self.lane_left_type(lane),
            self.lane_right_type(lane),
            self.lane_connector(lane).squeeze(-1),
            functional.softplus(
                self.lane_scales(lane).reshape(
                    batch, architecture.lane_queries, 3, architecture.lane_points, 3
                )
            )
            + 1.0e-4,
            functional.normalize(self.lane_embedding(lane), dim=-1),
            self.traffic_existence(traffic).squeeze(-1),
            torch.sigmoid(self.traffic_box(traffic)),
            self.traffic_category(traffic),
            self.traffic_attribute(traffic),
            functional.softplus(self.traffic_scales(traffic)) + 1.0e-4,
            functional.normalize(self.traffic_embedding(traffic), dim=-1),
            self.area_existence(area).squeeze(-1),
            self.area_category(area),
            self.area_geometry(area).reshape(
                batch, architecture.area_queries, architecture.area_points, 3
            ),
            self.area_valid(area),
            functional.softplus(
                self.area_scales(area).reshape(
                    batch, architecture.area_queries, architecture.area_points, 3
                )
            )
            + 1.0e-4,
            functional.normalize(self.area_embedding(area), dim=-1),
        )


def e0_outputs_by_name(outputs: Sequence[Tensor]) -> dict[str, Tensor]:
    if len(outputs) != len(E0_OUTPUT_NAMES):
        raise ValueError("E0 output count differs from the frozen node tensor contract")
    return dict(zip(E0_OUTPUT_NAMES, outputs, strict=True))


@dataclass(frozen=True, slots=True)
class ProjectionParity:
    maximum_absolute_error: float
    passed: bool


def projection_parity(
    projector: GroundPlaneProjector,
    camera_features: Tensor,
    camera_valid: Tensor,
    intrinsics: Tensor,
    transforms: Tensor,
    *,
    tolerance: float = 1.0e-6,
) -> ProjectionParity:
    with torch.inference_mode():
        reference = projector.reference(camera_features, camera_valid, intrinsics, transforms)
        vectorized = projector.vectorized(camera_features, camera_valid, intrinsics, transforms)
    error = float(torch.max(torch.abs(reference - vectorized)).item())
    return ProjectionParity(error, error <= tolerance)


__all__ = [
    "E0_OUTPUT_NAMES",
    "EfficientNetB0Fpn",
    "GroundPlaneProjector",
    "ProjectionParity",
    "ReferenceNodeModel",
    "e0_outputs_by_name",
    "projection_parity",
]

"""Strict immutable configuration for the M0 deployment spike."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputProfile(_FrozenModel):
    timestamps: Literal[2]
    cameras: Literal[8]
    channels: Literal[3]
    height: Literal[384]
    width: Literal[640]
    color_order: Literal["RGB"]
    resize_policy: Literal["letterbox-fit"]
    interpolation: Literal["bilinear"]
    pad_value: Literal[0]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class ModelShape(_FrozenModel):
    hidden_dimension: int = Field(ge=32, le=256)
    track_embedding_dimension: int = Field(ge=8, le=64)
    lane_queries: int = Field(ge=1)
    traffic_queries: int = Field(ge=1)
    area_queries: int = Field(ge=1)
    lane_points: Literal[11]
    area_points: int = Field(ge=3, le=64)
    lane_boundary_classes: Literal[3]
    traffic_categories: Literal[2]
    traffic_attributes: Literal[13]
    area_categories: Literal[2]
    spike_encoder: Literal["calibration-aware-masked-global-pool"]
    final_reference_encoder: Literal["efficientnet-b0-feature-pyramid"]
    final_reference_hidden_dimension: Literal[256]


class ExportProfile(_FrozenModel):
    opset: Literal[18]
    dynamic_axes: tuple[Literal["batch"], ...]
    precision: Literal["fp32"]
    output_zero_deadband: float = Field(ge=0.05, le=0.05)


class MicroOverfitProfile(_FrozenModel):
    frames: Literal[32]
    maximum_steps: Literal[5000]
    default_steps: int = Field(ge=100, le=5000)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    cpu_spatial_size: int = Field(ge=2, le=64)
    minimum_loss_reduction: float = Field(ge=0.9, le=0.9)
    minimum_node_category_accuracy: float = Field(ge=0.98, le=0.98)
    maximum_centerline_point_error_m: float = Field(ge=0.25, le=0.25)
    topology_gate_state: Literal["DEFERRED_TO_M5_PER_ADR_0001"]


class M0ModelProfile(_FrozenModel):
    schema_version: Literal["1.0.0"]
    profile_id: Literal["m0-feasibility-spike-v1"]
    seed: Literal[20260813]
    input: InputProfile
    model: ModelShape
    export: ExportProfile
    micro_overfit: MicroOverfitProfile

    def canonical_sha256(self) -> str:
        """Return the stable hash embedded in checkpoints and ONNX metadata."""
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_m0_profile(path: Path) -> M0ModelProfile:
    """Load the exact M0 profile and reject unknown or changed fields."""
    with path.open(encoding="utf-8") as source:
        value = yaml.safe_load(source)
    return M0ModelProfile.model_validate(value)

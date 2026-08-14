"""Strict production profile for the independent E0 baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class E0Input(_Frozen):
    timestamps: Literal[2]
    cameras: Literal[8]
    channels: Literal[3]
    height: Literal[384]
    width: Literal[640]
    current_timestamp_index: Literal[1]


class E0Architecture(_Frozen):
    backbone: Literal["efficientnet-b0"]
    backbone_weights: Literal["random-initialization"]
    feature_strides: tuple[Literal[8, 16, 32], Literal[8, 16, 32], Literal[8, 16, 32]]
    feature_channels: tuple[Literal[40], Literal[112], Literal[320]]
    fpn_channels: Literal[128]
    bev_x_range_m: tuple[float, float]
    bev_y_range_m: tuple[float, float]
    bev_cell_size_m: float = Field(gt=0)
    bev_shape: tuple[Literal[200], Literal[160]]
    decoder_memory_shape: tuple[Literal[25], Literal[20]]
    decoder_layers: Literal[4]
    hidden_dimension: Literal[256]
    attention_heads: Literal[8]
    lane_queries: Literal[96]
    traffic_queries: Literal[64]
    area_queries: Literal[32]
    lane_points: Literal[11]
    area_points: int = Field(ge=3, le=64)
    boundary_classes: Literal[3]
    traffic_categories: Literal[2]
    traffic_attributes: Literal[13]
    area_categories: Literal[2]
    track_embedding_dimension: int = Field(ge=8, le=64)

    @model_validator(mode="after")
    def validate_bev(self) -> E0Architecture:
        x_cells = (self.bev_x_range_m[1] - self.bev_x_range_m[0]) / self.bev_cell_size_m
        y_cells = (self.bev_y_range_m[1] - self.bev_y_range_m[0]) / self.bev_cell_size_m
        if self.bev_x_range_m != (-20.0, 80.0) or self.bev_y_range_m != (-40.0, 40.0):
            raise ValueError("E0 BEV ranges differ from the V1 coordinate contract")
        if self.bev_cell_size_m != 0.5:
            raise ValueError("E0 BEV cell size differs from the V1 coordinate contract")
        if (round(x_cells), round(y_cells)) != self.bev_shape:
            raise ValueError("E0 BEV ranges, cell size, and shape disagree")
        return self


class E0Training(_Frozen):
    optimizer: Literal["AdamW"]
    base_learning_rate: float = Field(gt=0)
    backbone_learning_rate_multiplier: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_steps: Literal[1000]
    minimum_learning_rate: float = Field(gt=0)
    maximum_epochs: Literal[50]
    batch_size_per_gpu: Literal[1]
    gradient_accumulation_steps: Literal[8]
    gradient_clip_norm: float = Field(gt=0)
    minimum_early_stopping_epoch: Literal[20]
    early_stopping_patience: Literal[8]
    mixed_precision_preference: tuple[Literal["bf16", "fp16", "fp32"], ...]
    training_partition: Literal["model_training"]
    selection_partition: Literal["model_selection"]
    forbidden_statistics_partitions: tuple[str, ...]

    @model_validator(mode="after")
    def validate_isolation(self) -> E0Training:
        if (
            self.base_learning_rate != 2.0e-4
            or self.backbone_learning_rate_multiplier != 0.1
            or self.weight_decay != 0.01
            or self.minimum_learning_rate != 2.0e-6
            or self.gradient_clip_norm != 1.0
        ):
            raise ValueError("E0 optimizer recipe differs from the frozen V1 recipe")
        required = {
            "model_selection",
            "calibration",
            "internal_holdout",
            "external_diagnostic",
        }
        if set(self.forbidden_statistics_partitions) != required:
            raise ValueError("E0 training-statistics isolation policy changed")
        return self


class IndependentLinkerSearch(_Frozen):
    successor_distance_candidates_m: tuple[float, ...]
    successor_heading_candidates_deg: tuple[float, ...]
    control_endpoint_distance_candidates_px: tuple[float, ...]
    control_heading_candidates_deg: tuple[float, ...]
    fit_partition: Literal["model_training"]

    @model_validator(mode="after")
    def validate_candidates(self) -> IndependentLinkerSearch:
        for field in (
            self.successor_distance_candidates_m,
            self.successor_heading_candidates_deg,
            self.control_endpoint_distance_candidates_px,
            self.control_heading_candidates_deg,
        ):
            if not field or tuple(sorted(set(field))) != field or min(field) <= 0.0:
                raise ValueError("linker threshold candidates must be positive, unique, and sorted")
        return self


class E0Export(_Frozen):
    opset: Literal[18]
    precision: Literal["fp32"]


class E0Profile(_Frozen):
    schema_version: Literal["junctionlens.e0-profile.v1"]
    experiment_id: Literal["E0-independent"]
    seeds: tuple[Literal[20260813, 20260814, 20260815], ...]
    input: E0Input
    architecture: E0Architecture
    training: E0Training
    independent_linker: IndependentLinkerSearch
    export: E0Export

    @model_validator(mode="after")
    def validate_seeds(self) -> E0Profile:
        if self.seeds != (20260813, 20260814, 20260815):
            raise ValueError("E0 requires the predeclared primary and two robustness seeds")
        return self

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_e0_profile(path: Path) -> E0Profile:
    with path.open(encoding="utf-8") as source:
        value = yaml.safe_load(source)
    return E0Profile.model_validate(value)


__all__ = ["E0Profile", "load_e0_profile"]

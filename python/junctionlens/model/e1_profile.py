"""Strict E1 learned-topology profile bound to the shared E0 node architecture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from junctionlens.model.e0_profile import E0Profile
from junctionlens.security.parsing import ParseLimits, load_yaml_object_path


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class E1Topology(_Frozen):
    edge_dimension: Literal[64]
    geometry_hidden_dimension: Literal[64]
    allow_lane_self_edges: Literal[False]
    front_center_camera_index: Literal[0]
    lane_successor_loss_weight: float = Field(gt=0.0)
    control_lane_loss_weight: float = Field(gt=0.0)
    endpoint_continuity_loss_weight: float = Field(gt=0.0)
    positive_weight_partition: Literal["model_training"]

    @model_validator(mode="after")
    def validate_weights(self) -> E1Topology:
        if (
            self.lane_successor_loss_weight,
            self.control_lane_loss_weight,
            self.endpoint_continuity_loss_weight,
        ) != (3.0, 5.0, 1.0):
            raise ValueError("E1 topology weights differ from the frozen objective")
        return self


class E1Diagnostics(_Frozen):
    modes: tuple[Literal["oracle-nodes", "predicted-nodes"], ...]
    seed: Literal[20260813]
    maximum_steps: Literal[5000]
    oracle_lane_successor_f1_minimum: float
    oracle_control_lane_f1_minimum: float
    prediction_threshold: float

    @model_validator(mode="after")
    def validate_gate(self) -> E1Diagnostics:
        if self.modes != ("oracle-nodes", "predicted-nodes"):
            raise ValueError("E1 diagnostic modes differ from the frozen contract")
        if (
            self.oracle_lane_successor_f1_minimum != 0.95
            or self.oracle_control_lane_f1_minimum != 0.95
            or self.prediction_threshold != 0.5
        ):
            raise ValueError("E1 diagnostic thresholds differ from the frozen gate")
        return self


class E1Profile(_Frozen):
    schema_version: Literal["junctionlens.e1-profile.v1"]
    experiment_id: Literal["E1-joint"]
    base_profile_sha256: str
    topology: E1Topology
    diagnostics: E1Diagnostics

    @model_validator(mode="after")
    def validate_identity(self) -> E1Profile:
        if len(self.base_profile_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.base_profile_sha256
        ):
            raise ValueError("E1 base profile identity must be a lowercase SHA-256")
        return self

    def validate_base(self, base: E0Profile) -> None:
        if self.base_profile_sha256 != base.canonical_sha256():
            raise ValueError("E1 profile is bound to a different E0 node architecture")

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def load_e1_profile(path: Path, base: E0Profile) -> E1Profile:
    value = load_yaml_object_path(
        path,
        "E1 model profile",
        ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
    )
    profile = E1Profile.model_validate(value)
    profile.validate_base(base)
    return profile


__all__ = ["E1Profile", "load_e1_profile"]

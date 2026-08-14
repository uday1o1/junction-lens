"""Strict schemas and stable reason codes for the V1 fault lab."""

from __future__ import annotations

import base64
import binascii
import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FaultKind(StrEnum):
    SWAP_CONTROL_EDGES = "swap-control-edges"
    DROP_CONTROL_EDGES = "drop-control-edges"
    DROP_SUCCESSOR_CHAIN = "drop-successor-chain"
    ADD_SPURIOUS_SUCCESSORS = "add-spurious-successors"
    PERMUTE_NODES_CORRECTLY = "permute-nodes-correctly"
    PERMUTE_NODES_WITHOUT_EDGES = "permute-nodes-without-edges"
    DUPLICATE_NODE_ID = "duplicate-node-id"
    DANGLING_EDGE = "dangling-edge"
    JITTER_LANES = "jitter-lanes"
    FLIP_BOUNDARIES = "flip-boundaries"
    CORRUPT_EXTRINSIC = "corrupt-extrinsic"
    ZERO_UNCERTAINTY = "zero-uncertainty"
    INFLATE_UNCERTAINTY = "inflate-uncertainty"
    TEMPERATURE_COLLAPSE = "temperature-collapse"
    INJECT_NAN = "inject-nan"
    ALTERNATE_EDGE_CONFIDENCE = "alternate-edge-confidence"
    ALTERNATE_NODE_PRESENCE = "alternate-node-presence"
    REUSE_TRACK_ID = "reuse-track-id"
    FORCE_PROVIDER_FALLBACK = "force-provider-fallback"
    DELAY_POSTPROCESS = "delay-postprocess"
    LEAK_BUFFER = "leak-buffer"


FAULT_REASON_CODES: dict[FaultKind, str] = {
    FaultKind.SWAP_CONTROL_EDGES: "FAULT_CONTROL_ASSIGNMENT_CHANGED",
    FaultKind.DROP_CONTROL_EDGES: "FAULT_CONTROL_EDGE_RECALL_DEGRADED",
    FaultKind.DROP_SUCCESSOR_CHAIN: "FAULT_REACHABILITY_DEGRADED",
    FaultKind.ADD_SPURIOUS_SUCCESSORS: "FAULT_SPURIOUS_SUCCESSOR_ADDED",
    FaultKind.PERMUTE_NODES_CORRECTLY: "CONTROL_GRAPH_PERMUTATION_INVARIANT",
    FaultKind.PERMUTE_NODES_WITHOUT_EDGES: "FAULT_TOPOLOGY_NODE_ORDER_MISMATCH",
    FaultKind.DUPLICATE_NODE_ID: "CONTRACT_NODE_ID_DUPLICATE",
    FaultKind.DANGLING_EDGE: "CONTRACT_EDGE_DANGLING",
    FaultKind.JITTER_LANES: "FAULT_LANE_GEOMETRY_JITTER",
    FaultKind.FLIP_BOUNDARIES: "FAULT_LANE_BOUNDARIES_FLIPPED",
    FaultKind.CORRUPT_EXTRINSIC: "FAULT_CAMERA_EXTRINSIC_CHANGED",
    FaultKind.ZERO_UNCERTAINTY: "CONTRACT_UNCERTAINTY_SCALE",
    FaultKind.INFLATE_UNCERTAINTY: "FAULT_UNCERTAINTY_INFLATED",
    FaultKind.TEMPERATURE_COLLAPSE: "FAULT_CALIBRATION_OVERCONFIDENT",
    FaultKind.INJECT_NAN: "CONTRACT_NONFINITE",
    FaultKind.ALTERNATE_EDGE_CONFIDENCE: "FAULT_TEMPORAL_EDGE_FLIP",
    FaultKind.ALTERNATE_NODE_PRESENCE: "FAULT_TEMPORAL_PRESENCE_FLICKER",
    FaultKind.REUSE_TRACK_ID: "FAULT_TRACK_ID_REUSED_AFTER_TERMINATION",
    FaultKind.FORCE_PROVIDER_FALLBACK: "GATE_INTEGRITY_PROVIDER_FALLBACK",
    FaultKind.DELAY_POSTPROCESS: "GATE_PERFORMANCE_P95_LATENCY_BUDGET",
    FaultKind.LEAK_BUFFER: "GATE_PERFORMANCE_UNBOUNDED_MEMORY_GROWTH",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PredictionFrame(_Strict):
    frame_token: str = Field(min_length=1, max_length=256)
    envelope_pb_base64: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_base64(self) -> PredictionFrame:
        try:
            decoded = base64.b64decode(self.envelope_pb_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("frame payload must be canonical base64") from error
        if not decoded or base64.b64encode(decoded).decode("ascii") != self.envelope_pb_base64:
            raise ValueError("frame payload must be nonempty canonical base64")
        return self


class RuntimeFixture(_Strict):
    provider_node_counts: dict[str, int]
    postprocess_latency_ms: tuple[float, ...]
    device_memory_bytes: tuple[int, ...]

    @model_validator(mode="after")
    def validate_runtime(self) -> RuntimeFixture:
        if (
            not self.provider_node_counts
            or any(not name or count < 0 for name, count in self.provider_node_counts.items())
            or len(self.provider_node_counts) > 256
        ):
            raise ValueError("provider node counts are invalid")
        if not 10 <= len(self.postprocess_latency_ms) <= 100_000 or any(
            not math.isfinite(value) or value < 0.0 for value in self.postprocess_latency_ms
        ):
            raise ValueError("postprocess latency samples are invalid")
        if not 10 <= len(self.device_memory_bytes) <= 100_000 or any(
            value < 0 for value in self.device_memory_bytes
        ):
            raise ValueError("device memory samples are invalid")
        return self


class FaultDeclaration(_Strict):
    kind: FaultKind
    seed: int = Field(ge=0, lt=1 << 64)
    fraction: float = Field(gt=0.0, le=1.0)


class PredictionBundle(_Strict):
    schema_version: Literal["junctionlens.prediction-bundle.v1"]
    bundle_id: str = Field(min_length=1, max_length=128)
    frames: tuple[PredictionFrame, ...]
    runtime: RuntimeFixture
    fault_history: tuple[FaultDeclaration, ...] = ()

    @model_validator(mode="after")
    def validate_frames(self) -> PredictionBundle:
        tokens = [frame.frame_token for frame in self.frames]
        if not tokens or len(tokens) > 100_000 or len(tokens) != len(set(tokens)):
            raise ValueError("prediction frame tokens must be nonempty and unique")
        if len(self.fault_history) > 1:
            raise ValueError("V1 fault bundles allow exactly one derived transformation")
        return self


__all__ = [
    "FAULT_REASON_CODES",
    "FaultDeclaration",
    "FaultKind",
    "PredictionBundle",
    "PredictionFrame",
    "RuntimeFixture",
]

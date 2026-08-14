"""Strict schemas for the read-only local evidence API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictModel(BaseModel):
    """Reject undocumented fields and keep validated API values immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ServiceConfig(StrictModel):
    """Local service roots and response limits."""

    artifact_root: Path
    schema_path: Path
    web_root: Path | None = None
    max_artifact_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    max_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    max_metric_table_bytes: int = Field(default=256 * 1024 * 1024, ge=1)


class PageInfo(StrictModel):
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    returned: int = Field(ge=0)
    total: int = Field(ge=0)


class HealthResponse(StrictModel):
    schema_version: Literal["junctionlens.api-health.v1"]
    state: Literal["READY"]
    artifact_count: int = Field(ge=0)
    run_count: int = Field(ge=0)


class RunSummary(StrictModel):
    run_id: str
    run_kind: str
    state: str
    environment_fingerprint: str
    run_manifest_sha256: str
    execution_provider_profile: str
    source_git_commit: str
    source_dirty: bool


class RunPage(StrictModel):
    schema_version: Literal["junctionlens.api-run-page.v1"]
    page: PageInfo
    items: tuple[RunSummary, ...]


class ArtifactSummary(StrictModel):
    manifest_sha256: str
    kind: str
    payload_sha256: str
    payload_byte_size: int = Field(ge=0)
    media_type: str
    license_id: str
    metadata: dict[str, JsonValue]


class ArtifactPage(StrictModel):
    schema_version: Literal["junctionlens.api-artifact-page.v1"]
    page: PageInfo
    items: tuple[ArtifactSummary, ...]


class ArtifactDetail(ArtifactSummary):
    schema_version: Literal["junctionlens.api-artifact.v1"]
    parents: tuple[str, ...]
    relative_uri: str


class MetricTablePage(StrictModel):
    schema_version: Literal["junctionlens.api-metric-table.v1"]
    manifest_sha256: str
    columns: tuple[str, ...]
    page: PageInfo
    rows: tuple[dict[str, JsonValue], ...]


class DecisionDetail(StrictModel):
    schema_version: Literal["junctionlens.api-decision.v1"]
    manifest_sha256: str
    decision: dict[str, JsonValue]


class ScenePoint(StrictModel):
    x: float
    y: float


class SceneLane(StrictModel):
    node_id: str = Field(min_length=1, max_length=128)
    points: tuple[ScenePoint, ...] = Field(min_length=2, max_length=256)
    confidence: float = Field(ge=0.0, le=1.0)


class SceneControl(StrictModel):
    node_id: str = Field(min_length=1, max_length=128)
    x: float
    y: float
    control_type: str = Field(min_length=1, max_length=128)
    state: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)


class SceneEdge(StrictModel):
    edge_id: str = Field(min_length=1, max_length=128)
    edge_type: Literal["lane_successor", "control_applies_to_lane"]
    source_node_id: str = Field(min_length=1, max_length=128)
    target_node_id: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)


class SceneGraphLayer(StrictModel):
    lanes: tuple[SceneLane, ...] = Field(max_length=512)
    controls: tuple[SceneControl, ...] = Field(max_length=256)
    edges: tuple[SceneEdge, ...] = Field(max_length=2048)

    @model_validator(mode="after")
    def validate_references(self) -> SceneGraphLayer:
        lane_ids = {lane.node_id for lane in self.lanes}
        control_ids = {control.node_id for control in self.controls}
        if len(lane_ids) != len(self.lanes) or len(control_ids) != len(self.controls):
            raise ValueError("scene graph node identities must be unique")
        edge_ids = {edge.edge_id for edge in self.edges}
        if len(edge_ids) != len(self.edges):
            raise ValueError("scene graph edge identities must be unique")
        for edge in self.edges:
            if edge.edge_type == "lane_successor":
                if edge.source_node_id not in lane_ids or edge.target_node_id not in lane_ids:
                    raise ValueError("lane-successor edge must reference two lane nodes")
            elif edge.source_node_id not in control_ids or edge.target_node_id not in lane_ids:
                raise ValueError("control edge must reference one control and one lane node")
        return self


class SceneCamera(StrictModel):
    slot: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=128)
    artifact_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    restriction_reason: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_availability(self) -> SceneCamera:
        if (self.artifact_manifest_sha256 is None) == (self.restriction_reason is None):
            raise ValueError("camera must declare exactly one image artifact or restriction reason")
        return self


class SceneFrame(StrictModel):
    frame_id: str = Field(min_length=1, max_length=128)
    segment_id: str = Field(min_length=1, max_length=128)
    timestamp_ns: str = Field(pattern=r"^(0|[1-9][0-9]*)$")
    cameras: tuple[SceneCamera, ...] = Field(min_length=1, max_length=16)
    ground_truth: SceneGraphLayer
    baseline: SceneGraphLayer
    candidate: SceneGraphLayer

    @model_validator(mode="after")
    def validate_camera_slots(self) -> SceneFrame:
        slots = {camera.slot for camera in self.cameras}
        if len(slots) != len(self.cameras):
            raise ValueError("scene camera slots must be unique within a frame")
        return self


class SceneBundle(StrictModel):
    schema_version: Literal["junctionlens.scene-bundle.v1"]
    title: str = Field(min_length=1, max_length=256)
    decision_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_notice: str = Field(min_length=1, max_length=512)
    frames: tuple[SceneFrame, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_frame_ids(self) -> SceneBundle:
        identities = {frame.frame_id for frame in self.frames}
        if len(identities) != len(self.frames):
            raise ValueError("scene frame identities must be unique")
        return self


class SceneBundleDetail(StrictModel):
    schema_version: Literal["junctionlens.api-scene-bundle.v1"]
    manifest_sha256: str
    bundle: SceneBundle
    decision: dict[str, JsonValue]


class ApiErrorDetail(StrictModel):
    code: str
    message: str


class ApiErrorResponse(StrictModel):
    schema_version: Literal["junctionlens.api-error.v1"]
    error: ApiErrorDetail


__all__ = [
    "ApiErrorDetail",
    "ApiErrorResponse",
    "ArtifactDetail",
    "ArtifactPage",
    "ArtifactSummary",
    "DecisionDetail",
    "HealthResponse",
    "MetricTablePage",
    "PageInfo",
    "RunPage",
    "RunSummary",
    "SceneBundle",
    "SceneBundleDetail",
    "SceneCamera",
    "SceneControl",
    "SceneEdge",
    "SceneFrame",
    "SceneGraphLayer",
    "SceneLane",
    "ScenePoint",
    "ServiceConfig",
]

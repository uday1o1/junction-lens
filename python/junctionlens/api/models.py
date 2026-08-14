"""Strict schemas for the read-only local evidence API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class StrictModel(BaseModel):
    """Reject undocumented fields and keep validated API values immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ServiceConfig(StrictModel):
    """Local service roots and response limits."""

    artifact_root: Path
    schema_path: Path
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
    "ServiceConfig",
]

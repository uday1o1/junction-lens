"""Typed doctor report contract."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CapabilityState(StrEnum):
    """Observed state of one capability."""

    AVAILABLE = "AVAILABLE"
    ABSENT = "ABSENT"
    INACCESSIBLE = "INACCESSIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    ERROR = "ERROR"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CapabilityRequirement(StrEnum):
    """Role a capability plays in the selected profile."""

    REQUIRED_LOCAL = "REQUIRED_LOCAL"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    OPTIONAL = "OPTIONAL"
    TARGET_ONLY = "TARGET_ONLY"


class CapabilityEvidence(BaseModel):
    """Bounded evidence from a real capability probe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    capability: str
    state: CapabilityState
    requirement: CapabilityRequirement
    reason_code: str
    summary: str
    observed_version: str | None = None
    expected_version: str | None = None
    command: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class HostEvidence(BaseModel):
    """Observed host identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str
    release: str
    machine: str
    python_implementation: str
    python_version: str


class Readiness(BaseModel):
    """Independent readiness status for each evidence boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    local_cpu: bool
    linux_x86_64_reference: bool
    accelerated_target: bool
    licensed_data: bool


class DoctorReport(BaseModel):
    """Versioned doctor JSON response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "1.0.0"
    generated_at: datetime
    profile: str
    host: HostEvidence
    readiness: Readiness
    capabilities: list[CapabilityEvidence]

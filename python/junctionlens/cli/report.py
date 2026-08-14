"""Public deterministic evidence snapshot command."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from junctionlens.cli.output import emit, human_requested
from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import canonical_json_bytes


def _reason_codes(decision: dict[str, Any]) -> list[str]:
    result: set[str] = set()
    for key in (
        "integrity_reason_codes",
        "infrastructure_reason_codes",
        "performance_reason_codes",
    ):
        values = decision.get(key, [])
        if isinstance(values, list):
            result.update(str(value) for value in values)
    cells = decision.get("cells", [])
    if isinstance(cells, list):
        for cell in cells:
            if isinstance(cell, dict) and cell.get("status") != "PASS":
                reason = cell.get("reason_code")
                if isinstance(reason, str):
                    result.add(reason)
    return sorted(result)


def report_command(
    decision: Annotated[str, typer.Option("--decision")],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
    human: Annotated[bool, typer.Option("--human")] = False,
) -> None:
    """Persist a canonical JSON snapshot of one authoritative release decision."""
    try:
        from junctionlens.api.models import ServiceConfig
        from junctionlens.api.repository import EvidenceReadError, EvidenceRepository

        repository = EvidenceRepository(
            ServiceConfig(artifact_root=artifact_root, schema_path=schema)
        )
        decision_body = cast(dict[str, Any], repository.decision(decision))
        decision_artifact = repository.artifact(decision)
        body = {
            "schema_version": "junctionlens.evidence-report-snapshot.v1",
            "decision_manifest_sha256": decision,
            "decision_sha256": decision_body["decision_sha256"],
            "status": decision_body["status"],
            "reason_codes": _reason_codes(decision_body),
            "authoritative_decision_persisted": True,
            "decision_recalculated": False,
        }
        receipt = EvidenceRegistry(artifact_root, schema).put_bytes(
            canonical_json_bytes(body) + b"\n",
            kind="evidence_report",
            media_type="application/json",
            license_id=decision_artifact.license_id,
            metadata={
                "decision_manifest_sha256": decision,
                "format": "json-snapshot",
                "status": decision_body["status"],
            },
            parents=(decision,),
        )
        result = {
            "schema_version": "junctionlens.report-receipt.v1",
            **asdict(receipt),
            "immutable_path": (
                f"objects/sha256/{receipt.payload_sha256[:2]}/{receipt.payload_sha256[2:]}"
            ),
        }
    except (EvidenceReadError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"report error: {error}", err=True)
        raise typer.Exit(code=2) from error
    if human_requested(human):
        typer.echo(
            f"report {result['manifest_sha256']} at {result['immutable_path']} "
            f"for status {body['status']}"
        )
    else:
        emit(result)


__all__ = ["report_command"]

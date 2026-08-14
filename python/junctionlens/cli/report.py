"""Public deterministic evidence report command."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from junctionlens.api.models import ServiceConfig
from junctionlens.api.repository import EvidenceReadError, EvidenceRepository
from junctionlens.cli.output import emit, human_requested
from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.report import EvidenceBundleError, export_evidence_bundle


class ReportMode(str, Enum):
    """Privacy boundary for materialized evidence."""

    public = "public"
    private = "private"


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
    comparison: Annotated[
        str | None,
        typer.Option(
            "--comparison",
            help="Immutable comparison report-data manifest for a complete evidence bundle.",
        ),
    ] = None,
    decision: Annotated[
        str | None,
        typer.Option(
            "--decision",
            help="Legacy decision-only JSON snapshot; use --comparison for complete bundles.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="New directory for materialized bundle files.",
        ),
    ] = None,
    mode: Annotated[
        ReportMode,
        typer.Option("--mode", help="Public redacted export or acknowledged private export."),
    ] = ReportMode.public,
    scene: Annotated[
        str | None,
        typer.Option("--scene", help="Optional registered counterexample scene manifest."),
    ] = None,
    acknowledge_private_license: Annotated[
        bool,
        typer.Option(
            "--acknowledge-private-license",
            help="Acknowledge that private output may contain licensed thumbnails.",
        ),
    ] = False,
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
    project_root: Annotated[
        Path,
        typer.Option(hidden=True, exists=True, file_okay=False, resolve_path=True),
    ] = Path(),
    human: Annotated[bool, typer.Option("--human")] = False,
) -> None:
    """Export immutable comparison evidence for offline review and reproduction."""
    try:
        if (comparison is None) == (decision is None):
            raise ValueError("provide exactly one of --comparison or --decision")
        if comparison is not None:
            destination = output_dir or artifact_root / f"report-{comparison[:12]}-{mode.value}"
            bundle_receipt = export_evidence_bundle(
                artifact_root=artifact_root,
                schema_path=schema,
                project_root=project_root,
                comparison_manifest_sha256=comparison,
                output_directory=destination,
                mode=mode.value,
                scene_manifest_sha256=scene,
                private_license_acknowledged=acknowledge_private_license,
            )
            result = bundle_receipt.to_dict()
            if human_requested(human):
                typer.echo(
                    f"report {result['manifest_sha256']} at {result['immutable_path']} "
                    f"materialized in {result['output_directory']}"
                )
            else:
                emit(result)
            return
        if output_dir is not None or scene is not None or mode != ReportMode.public:
            raise ValueError("bundle output, scene, and private mode require --comparison")
        if acknowledge_private_license:
            raise ValueError("private license acknowledgment requires a private comparison bundle")
        repository = EvidenceRepository(
            ServiceConfig(artifact_root=artifact_root, schema_path=schema)
        )
        decision_hash = cast(str, decision)
        decision_body = cast(dict[str, Any], repository.decision(decision_hash))
        decision_artifact = repository.artifact(decision_hash)
        body = {
            "schema_version": "junctionlens.evidence-report-snapshot.v1",
            "decision_manifest_sha256": decision_hash,
            "decision_sha256": decision_body["decision_sha256"],
            "status": decision_body["status"],
            "reason_codes": _reason_codes(decision_body),
            "authoritative_decision_persisted": True,
            "decision_recalculated": False,
        }
        snapshot_receipt = EvidenceRegistry(artifact_root, schema).put_bytes(
            canonical_json_bytes(body) + b"\n",
            kind="evidence_report",
            media_type="application/json",
            license_id=decision_artifact.license_id,
            metadata={
                "decision_manifest_sha256": decision_hash,
                "format": "json-snapshot",
                "status": decision_body["status"],
            },
            parents=(decision_hash,),
        )
        result = {
            "schema_version": "junctionlens.report-receipt.v1",
            **asdict(snapshot_receipt),
            "immutable_path": (
                f"objects/sha256/{snapshot_receipt.payload_sha256[:2]}/"
                f"{snapshot_receipt.payload_sha256[2:]}"
            ),
        }
    except (
        EvidenceBundleError,
        EvidenceReadError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
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

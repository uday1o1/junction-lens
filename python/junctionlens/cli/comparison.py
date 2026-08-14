"""Public paired model comparison command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from junctionlens.gate.comparison import ComparisonError, receipt_dict, run_comparison


def compare_command(
    baseline: Annotated[
        str, typer.Option("--baseline", help="Baseline artifact manifest SHA-256.")
    ],
    candidate: Annotated[
        str, typer.Option("--candidate", help="Candidate artifact manifest SHA-256.")
    ],
    charter: Annotated[
        Path,
        typer.Option("--charter", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/gates/acceptance-v1.yaml"),
    artifact_root: Annotated[
        Path, typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True)
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
    metrics: Annotated[
        Path,
        typer.Option("--metrics", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/metrics/v1.yaml"),
    slices: Annotated[
        Path,
        typer.Option("--slices", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/slices/v1.yaml"),
) -> None:
    """Compare exact paired frame evidence and persist the authoritative decision."""
    try:
        receipt = run_comparison(
            artifact_root=artifact_root,
            schema_path=schema,
            charter_path=charter,
            metric_registry_path=metrics,
            slice_registry_path=slices,
            baseline_manifest_sha256=baseline,
            candidate_manifest_sha256=candidate,
        )
    except (ComparisonError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"comparison error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(receipt_dict(receipt), sort_keys=True, separators=(",", ":")))
    if receipt.status != "PASS":
        raise typer.Exit(code=3)


__all__ = ["compare_command"]

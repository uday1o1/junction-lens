"""Public deterministic fault-injection command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from junctionlens.cli.output import emit
from junctionlens.faults.models import FaultKind
from junctionlens.faults.service import FaultError, inject_fault, receipt_dict


def fault_command(
    input_manifest: Annotated[str, typer.Option("--input", help="Parent bundle manifest SHA-256.")],
    kind: Annotated[FaultKind, typer.Option("--kind", case_sensitive=False)],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260813,
    fraction: Annotated[float, typer.Option("--fraction", min=0.0, max=1.0)] = 0.5,
    artifact_root: Annotated[
        Path, typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True)
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
) -> None:
    """Derive one immutable fault bundle and prove its intended detector fires."""
    try:
        receipt = inject_fault(
            artifact_root=artifact_root,
            schema_path=schema,
            input_manifest_sha256=input_manifest,
            kind=kind,
            seed=seed,
            fraction=fraction,
        )
    except (FaultError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"fault error: {error}", err=True)
        raise typer.Exit(code=2) from error
    emit(receipt_dict(receipt))


__all__ = ["fault_command"]

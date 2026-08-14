"""Public acceptance-charter freeze and release-decision commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from junctionlens.cli.output import emit
from junctionlens.gate.charter import CharterError, freeze_charter
from junctionlens.gate.decision import DecisionError, persist_decision

gate_app = typer.Typer(help="Freeze and apply immutable release policy.", no_args_is_help=True)


def _print(value: object) -> None:
    emit(value)


@gate_app.command("freeze")
def freeze_command(
    draft: Annotated[
        Path,
        typer.Option("--draft", exists=True, dir_okay=False, resolve_path=True),
    ],
    baseline_run: Annotated[str, typer.Option("--baseline-run")],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ],
    signer: Annotated[str, typer.Option("--signer")],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    metrics: Annotated[
        Path,
        typer.Option("--metrics", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/metrics/v1.yaml"),
    slices: Annotated[
        Path,
        typer.Option("--slices", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/slices/v1.yaml"),
    project_root: Annotated[
        Path,
        typer.Option(hidden=True, exists=True, file_okay=False, resolve_path=True),
    ] = Path(),
) -> None:
    """Freeze V1 from E0 variability, M0 hardware, and pre-holdout power evidence."""
    try:
        receipt = freeze_charter(
            draft,
            baseline_run,
            output,
            artifact_root=artifact_root,
            project_root=project_root,
            signer=signer,
            metrics_path=metrics,
            slices_path=slices,
        )
    except (CharterError, OSError, ValueError) as error:
        typer.echo(f"gate error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(receipt)


@gate_app.command("decide")
def decide_command(
    charter: Annotated[
        Path,
        typer.Option("--charter", exists=True, dir_okay=False, resolve_path=True),
    ],
    evidence: Annotated[
        Path,
        typer.Option("--evidence", exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ],
) -> None:
    """Persist the only authoritative decision for one evidence bundle."""
    try:
        decision = persist_decision(charter, evidence, output)
    except (DecisionError, OSError, ValueError) as error:
        typer.echo(f"gate error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(decision)
    if decision["status"] != "PASS":
        raise typer.Exit(code=3)


__all__ = ["gate_app"]

"""Public unrestricted demonstration command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from junctionlens.cli.output import emit


def demo_synthetic_command(
    output: Annotated[
        Path,
        typer.Option("--output", file_okay=False, resolve_path=False),
    ] = Path("artifacts/demo"),
    project_root: Annotated[
        Path,
        typer.Option(hidden=True, exists=True, file_okay=False, resolve_path=True),
    ] = Path(),
) -> None:
    """Build an unrestricted fault, comparison, gate, report, and viewer workflow."""
    from junctionlens.demo import DemoError, run_synthetic_demo

    try:
        receipt = run_synthetic_demo(output, project_root)
    except (DemoError, OSError, RuntimeError, ValueError) as error:
        typer.echo(f"synthetic demo error: {error}", err=True)
        raise typer.Exit(code=2) from error
    emit(receipt.to_dict())


__all__ = ["demo_synthetic_command"]

"""Public JunctionLens CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from junctionlens.doctor.service import run_doctor

app = typer.Typer(
    name="junctionlens",
    help="Control-aware road-scene graph development and release evidence.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run a JunctionLens workflow command."""


@app.command("doctor")
def doctor_command(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Write the versioned machine-readable report to stdout."),
    ] = False,
    project_root: Annotated[
        Path,
        typer.Option(hidden=True, exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path(),
) -> None:
    """Inspect real local, data, and accelerated capabilities."""
    report = run_doctor(project_root)
    payload = report.model_dump(mode="json")
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        typer.echo(f"profile: {report.profile}")
        typer.echo(f"local CPU ready: {str(report.readiness.local_cpu).lower()}")
        for capability in report.capabilities:
            typer.echo(
                f"{capability.capability}: {capability.state.value} "
                f"[{capability.reason_code}] {capability.summary}"
            )
    if not report.readiness.local_cpu:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()

"""Public JunctionLens CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from junctionlens.cli.data import data_app
from junctionlens.doctor.service import run_doctor
from junctionlens.evaluator import EvaluationError, evaluate_official

app = typer.Typer(
    name="junctionlens",
    help="Control-aware road-scene graph development and release evidence.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")


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


@app.command("evaluate")
def evaluate_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, resolve_path=True),
    ],
    project_root: Annotated[
        Path,
        typer.Option(hidden=True, exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path(),
) -> None:
    """Compute official metrics in the isolated compatibility image."""
    try:
        result = evaluate_official(input_path, project_root)
    except (EvaluationError, OSError, ValueError) as error:
        typer.echo(f"evaluation error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    app()

"""Public JunctionLens CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer

from junctionlens.cli.contract import contract_app
from junctionlens.cli.data import data_app
from junctionlens.cli.gate import gate_app
from junctionlens.cli.model import model_app
from junctionlens.cli.registry import registry_app
from junctionlens.cli.synthetic import synthetic_app
from junctionlens.doctor.service import run_doctor
from junctionlens.evaluator import EvaluationError, evaluate_custom, evaluate_official
from junctionlens.evaluator.evidence import (
    EvidenceError,
    evaluate_calibration_file,
    evaluate_runtime_file,
)

app = typer.Typer(
    name="junctionlens",
    help="Control-aware road-scene graph development and release evidence.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
app.add_typer(gate_app, name="gate")
app.add_typer(contract_app, name="contract")
app.add_typer(model_app, name="model")
app.add_typer(registry_app, name="registry")
app.add_typer(synthetic_app, name="synthetic")


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


@app.command("calibrate")
def calibrate_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, resolve_path=True),
    ],
) -> None:
    """Compute deterministic probability and geometry calibration evidence."""
    try:
        result = evaluate_calibration_file(input_path)
    except (EvidenceError, OSError, ValueError) as error:
        typer.echo(f"calibration error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


@app.command("evaluate")
def evaluate_command(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    ground_truth_root: Annotated[
        Path | None,
        typer.Option("--ground-truth", exists=True, resolve_path=True),
    ] = None,
    prediction_root: Annotated[
        Path | None,
        typer.Option("--predictions", exists=True, resolve_path=True),
    ] = None,
    artifact_root: Annotated[
        Path | None,
        typer.Option("--artifact-root", file_okay=False, resolve_path=True),
    ] = None,
    runtime_input: Annotated[
        Path | None,
        typer.Option("--runtime-input", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    project_root: Annotated[
        Path,
        typer.Option(hidden=True, exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path(),
) -> None:
    """Compute official metrics or immutable custom graph and temporal KPIs."""
    try:
        custom_values = (ground_truth_root, prediction_root, artifact_root)
        if runtime_input is not None:
            if input_path is not None or any(value is not None for value in custom_values):
                raise EvaluationError("runtime evaluation cannot be combined with other inputs")
            result = evaluate_runtime_file(runtime_input)
        elif any(value is not None for value in custom_values):
            if input_path is not None or any(value is None for value in custom_values):
                raise EvaluationError(
                    "custom evaluation requires --ground-truth, --predictions, and "
                    "--artifact-root, with no --input"
                )
            receipt = evaluate_custom(
                cast(Path, ground_truth_root),
                cast(Path, prediction_root),
                cast(Path, artifact_root),
                project_root,
            )
            result = {
                "artifacts": {
                    "frame_kpi_table": {
                        "manifest_sha256": receipt.frame_table_manifest_sha256,
                        "payload_sha256": receipt.frame_table_payload_sha256,
                    },
                    "match_map": {
                        "manifest_sha256": receipt.match_manifest_sha256,
                        "payload_sha256": receipt.match_payload_sha256,
                    },
                    "segment_kpi_table": {
                        "manifest_sha256": receipt.segment_table_manifest_sha256,
                        "payload_sha256": receipt.segment_table_payload_sha256,
                    },
                },
                "schema_version": "junctionlens.custom-evaluation-receipt.v1",
            }
        elif input_path is not None:
            result = evaluate_official(input_path, project_root)
        else:
            raise EvaluationError("official evaluation requires --input")
    except (EvaluationError, OSError, ValueError) as error:
        typer.echo(f"evaluation error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    app()

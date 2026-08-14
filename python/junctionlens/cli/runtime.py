"""Public production inference command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from junctionlens.runtime import RuntimeLaunchError, run_cpu_batch


def infer_command(
    model: Annotated[Path, typer.Option("--model", exists=True, dir_okay=False, resolve_path=True)],
    input_list: Annotated[
        Path, typer.Option("--input-list", exists=True, dir_okay=False, resolve_path=True)
    ],
    asset_root: Annotated[
        Path, typer.Option("--asset-root", exists=True, file_okay=False, resolve_path=True)
    ],
    output_directory: Annotated[
        Path, typer.Option("--output-dir", file_okay=False, resolve_path=True)
    ],
    profile: Annotated[
        Path, typer.Option("--profile", exists=True, dir_okay=False, resolve_path=True)
    ] = Path("configs/model/m0-spike.yaml"),
    runtime_binary: Annotated[
        Path, typer.Option("--runtime-binary", exists=True, dir_okay=False, resolve_path=True)
    ] = Path("build/cpu/bin/junctionlens-runtime"),
    repeat_loads: Annotated[int, typer.Option("--repeat-loads", min=1, max=100)] = 1,
    buffer_slots: Annotated[int, typer.Option("--buffer-slots", min=1, max=1024)] = 2,
    project_root: Annotated[
        Path,
        typer.Option(hidden=True, exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path(),
) -> None:
    """Run bounded native CPU inference for a newline-delimited protobuf batch."""
    try:
        receipt = run_cpu_batch(
            model_path=model,
            profile_path=profile,
            input_list_path=input_list,
            asset_root=asset_root,
            output_directory=output_directory,
            runtime_binary=runtime_binary,
            project_root=project_root,
            repeat_loads=repeat_loads,
            buffer_slots=buffer_slots,
        )
    except (OSError, RuntimeLaunchError, ValueError) as error:
        typer.echo(f"inference error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))

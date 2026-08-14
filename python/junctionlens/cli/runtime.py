"""Public production inference command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from junctionlens.runtime import RuntimeLaunchError, run_batch


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
    provider_profile: Annotated[
        str, typer.Option("--provider-profile", help="cpu-reference, cuda, or tensorrt")
    ] = "cpu-reference",
    provider_log_output: Annotated[
        Path | None, typer.Option("--provider-log-output", dir_okay=False, resolve_path=True)
    ] = None,
    device_id: Annotated[int, typer.Option("--device-id", min=0, max=1024)] = 0,
    provider_cache_root: Annotated[
        Path | None, typer.Option("--provider-cache-root", file_okay=False, resolve_path=True)
    ] = None,
    gpu_compute_capability: Annotated[str | None, typer.Option("--gpu-compute-capability")] = None,
    cuda_version: Annotated[str | None, typer.Option("--cuda-version")] = None,
    driver_compatibility_class: Annotated[
        str | None, typer.Option("--driver-compatibility-class")
    ] = None,
    tensorrt_version: Annotated[str | None, typer.Option("--tensorrt-version")] = None,
    project_root: Annotated[
        Path,
        typer.Option(hidden=True, exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path(),
) -> None:
    """Run bounded native inference for a newline-delimited protobuf batch."""
    try:
        if provider_profile not in {"cpu-reference", "cuda", "tensorrt"}:
            raise RuntimeLaunchError("provider profile must be cpu-reference, cuda, or tensorrt")
        receipt = run_batch(
            model_path=model,
            profile_path=profile,
            input_list_path=input_list,
            asset_root=asset_root,
            output_directory=output_directory,
            runtime_binary=runtime_binary,
            project_root=project_root,
            repeat_loads=repeat_loads,
            buffer_slots=buffer_slots,
            provider_profile=cast(Literal["cpu-reference", "cuda", "tensorrt"], provider_profile),
            provider_log_output=provider_log_output,
            device_id=device_id,
            provider_cache_root=provider_cache_root,
            gpu_compute_capability=gpu_compute_capability,
            cuda_version=cuda_version,
            driver_compatibility_class=driver_compatibility_class,
            tensorrt_version=tensorrt_version,
        )
    except (OSError, RuntimeLaunchError, ValueError) as error:
        typer.echo(f"inference error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))

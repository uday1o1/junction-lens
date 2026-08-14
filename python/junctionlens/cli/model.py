"""Public model feasibility, export, parity, and provider commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from junctionlens.data.license import DatasetRegistrationError, load_registration
from junctionlens.data.manifests import ManifestError
from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
from junctionlens.model.benchmark import BenchmarkError, run_m0_benchmark
from junctionlens.model.budget import BudgetError, load_budget_plan
from junctionlens.model.e0_artifacts import E0ArtifactError, finalize_e0_artifacts
from junctionlens.model.e0_data import E0DataError, load_partition_isolation
from junctionlens.model.e0_profile import load_e0_profile
from junctionlens.model.e0_training import (
    E0TrainingError,
    run_e0_training,
    select_e0_checkpoint,
)
from junctionlens.model.export import ModelExportError, export_model
from junctionlens.model.overfit import MicroOverfitError, run_micro_overfit
from junctionlens.model.parity import ParityError, run_parity
from junctionlens.model.profile import load_m0_profile
from junctionlens.model.providers import ProviderProbeError, run_provider_probe

model_app = typer.Typer(help="Prove model training and deployment contracts.", no_args_is_help=True)

ProfileOption = Annotated[
    Path,
    typer.Option(
        "--profile",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="Strict M0 model profile.",
    ),
]


def _print(value: object) -> None:
    typer.echo(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _fail(error: Exception) -> None:
    typer.echo(f"model error: {error}", err=True)
    raise typer.Exit(code=2) from error


@model_app.command("train-e0")
def train_e0_command(
    dataset_root: Annotated[
        Path,
        typer.Option("--dataset-root", exists=True, file_okay=False, resolve_path=True),
    ],
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", exists=True, dir_okay=False, resolve_path=True),
    ],
    seed: Annotated[int, typer.Option("--seed")],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", file_okay=False, resolve_path=True),
    ],
    profile_path: Annotated[
        Path,
        typer.Option("--profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e0-independent-v1.yaml"),
    adapter_config: Annotated[
        Path,
        typer.Option("--adapter-config", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.adapter.yaml"),
    split_policy: Annotated[
        Path,
        typer.Option("--split-policy", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.split-v1.yaml"),
    device: Annotated[str | None, typer.Option("--device")] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
) -> None:
    """Train one predeclared E0 seed from the registered model-training partition."""
    try:
        project_root = Path.cwd().resolve()
        registration = load_registration(project_root, "openlane-v2-v2.1", "full")
        registered_root = Path(str(registration["root"])).resolve(strict=True)
        if registered_root != dataset_root:
            raise DatasetRegistrationError(
                "E0 dataset root differs from the checksum-verified full registration"
            )
        profile = load_e0_profile(profile_path)
        isolation = load_partition_isolation(
            split_manifest,
            split_policy,
            partition=profile.training.training_partition,
            statistics=True,
        )
        result = run_e0_training(
            profile,
            OpenLaneAdapter(dataset_root, adapter_config),
            isolation,
            output_root,
            seed=seed,
            project_root=project_root,
            dataset_registration=registration,
            device_name=device,
            resume=resume,
        )
    except (
        DatasetRegistrationError,
        E0DataError,
        E0TrainingError,
        ManifestError,
        OpenLaneAdapterError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        _fail(error)
        return
    _print(result)


@model_app.command("select-e0")
def select_e0_command(
    run_root: Annotated[
        Path,
        typer.Option("--run-root", exists=True, file_okay=False, resolve_path=True),
    ],
    scores: Annotated[
        Path,
        typer.Option("--scores", exists=True, dir_okay=False, resolve_path=True),
    ],
    selection_split_manifest_sha256: Annotated[
        str,
        typer.Option("--selection-split-manifest-sha256"),
    ],
) -> None:
    """Select an E0 epoch with the frozen topology, official, then NLL ordering."""
    try:
        result = select_e0_checkpoint(
            run_root,
            scores,
            expected_selection_split_manifest_sha256=selection_split_manifest_sha256,
        )
    except (E0TrainingError, OSError, TypeError, ValueError) as error:
        _fail(error)
        return
    _print(result)


@model_app.command("finalize-e0")
def finalize_e0_command(
    run_roots: Annotated[
        list[Path],
        typer.Option(
            "--run-root",
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Repeat for the primary and two robustness seed run roots.",
        ),
    ],
    measured_evidence: Annotated[
        Path,
        typer.Option("--measured-evidence", exists=True, dir_okay=False, resolve_path=True),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", file_okay=False, resolve_path=True),
    ],
    profile_path: Annotated[
        Path,
        typer.Option("--profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e0-independent-v1.yaml"),
) -> None:
    """Finalize the three-seed E0 manifest and measured model card."""
    try:
        result = finalize_e0_artifacts(
            load_e0_profile(profile_path), run_roots, measured_evidence, output_root
        )
    except (E0ArtifactError, OSError, TypeError, ValueError) as error:
        _fail(error)
        return
    _print(result)


@model_app.command("micro-overfit")
def micro_overfit_command(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", file_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/overfit"),
    profile_path: ProfileOption = Path("configs/model/m0-spike.yaml"),
    steps: Annotated[int | None, typer.Option("--steps", min=100, max=5000)] = None,
) -> None:
    """Overfit the fixed 32-frame set and enforce every M0-owned threshold."""
    try:
        report = run_micro_overfit(load_m0_profile(profile_path), output_dir, steps=steps)
    except (MicroOverfitError, OSError, ValueError) as error:
        _fail(error)
    _print(report)


@model_app.command("export")
def export_command(
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/overfit/checkpoint.pt"),
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/model.onnx"),
    profile_path: ProfileOption = Path("configs/model/m0-spike.yaml"),
) -> None:
    """Export and structurally validate the frozen opset-18 graph."""
    try:
        report = export_model(load_m0_profile(profile_path), checkpoint, output)
    except (ModelExportError, OSError, RuntimeError, ValueError) as error:
        _fail(error)
    _print(report)


@model_app.command("parity")
def parity_command(
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/overfit/checkpoint.pt"),
    model: Annotated[
        Path,
        typer.Option("--model", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/model.onnx"),
    native_runner: Annotated[
        Path,
        typer.Option("--native-runner", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("build/cpu/junctionlens-onnx-probe"),
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/parity.json"),
    profile_path: ProfileOption = Path("configs/model/m0-spike.yaml"),
) -> None:
    """Compare all raw outputs from PyTorch, Python ORT, and native C++ ORT."""
    try:
        report = run_parity(load_m0_profile(profile_path), checkpoint, model, native_runner, output)
    except (ParityError, OSError, RuntimeError, ValueError) as error:
        _fail(error)
    _print(report)


@model_app.command("probe-providers")
def providers_command(
    model: Annotated[
        Path,
        typer.Option("--model", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/model.onnx"),
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/provider-assignment.json"),
    profile_path: ProfileOption = Path("configs/model/m0-spike.yaml"),
) -> None:
    """Trace CPU and every locally available accelerated execution provider."""
    try:
        report = run_provider_probe(model, load_m0_profile(profile_path), output)
    except (ProviderProbeError, OSError, RuntimeError, ValueError) as error:
        _fail(error)
    _print(report)


@model_app.command("benchmark-m0")
def benchmark_command(
    model: Annotated[
        Path,
        typer.Option("--model", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/model.onnx"),
    micro_report: Annotated[
        Path,
        typer.Option("--micro-report", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/overfit/micro-overfit-report.json"),
    evaluator_report: Annotated[
        Path,
        typer.Option("--evaluator-report", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/evaluator.json"),
    budget_path: Annotated[
        Path,
        typer.Option("--budget", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/budgets/v1.yaml"),
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike"),
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m0/model-spike/benchmark.json"),
    profile_path: ProfileOption = Path("configs/model/m0-spike.yaml"),
) -> None:
    """Record local portability numbers and freeze target acceptance margins."""
    try:
        report = run_m0_benchmark(
            load_m0_profile(profile_path),
            load_budget_plan(budget_path),
            model,
            micro_report,
            evaluator_report,
            artifact_root,
            output,
        )
    except (BenchmarkError, BudgetError, OSError, RuntimeError, ValueError) as error:
        _fail(error)
    _print(report)

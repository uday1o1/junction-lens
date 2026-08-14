"""Public model feasibility, export, parity, and provider commands."""

from __future__ import annotations

from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import typer

from junctionlens.cli.output import emit

model_app = typer.Typer(help="Prove model training and deployment contracts.", no_args_is_help=True)


class CheckpointExperiment(StrEnum):
    """Public experiment identifiers accepted by checkpoint scoring."""

    E0 = "E0-independent"
    E1 = "E1-joint"


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
    emit(value)


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
    from junctionlens.data.license import DatasetRegistrationError, load_registration
    from junctionlens.data.manifests import ManifestError
    from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
    from junctionlens.model.e0_data import E0DataError, load_partition_isolation
    from junctionlens.model.e0_profile import load_e0_profile
    from junctionlens.model.e0_training import E0TrainingError, run_e0_training

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
    from junctionlens.model.e0_training import E0TrainingError, select_e0_checkpoint

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
    from junctionlens.model.e0_artifacts import E0ArtifactError, finalize_e0_artifacts
    from junctionlens.model.e0_profile import load_e0_profile

    try:
        result = finalize_e0_artifacts(
            load_e0_profile(profile_path), run_roots, measured_evidence, output_root
        )
    except (E0ArtifactError, OSError, TypeError, ValueError) as error:
        _fail(error)
        return
    _print(result)


@model_app.command("verify-topology")
def verify_topology_command(
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = Path("artifacts/m5/topology-diagnostic.json"),
    base_profile_path: Annotated[
        Path,
        typer.Option("--base-profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e0-independent-v1.yaml"),
    profile_path: Annotated[
        Path,
        typer.Option("--profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e1-joint-v1.yaml"),
) -> None:
    """Train and verify the frozen oracle-node and predicted-node topology modes."""
    from junctionlens.model.e0_profile import load_e0_profile
    from junctionlens.model.e1_profile import load_e1_profile
    from junctionlens.model.topology_diagnostic import (
        TopologyDiagnosticError,
        run_topology_diagnostic,
    )

    try:
        base = load_e0_profile(base_profile_path)
        result = run_topology_diagnostic(base, load_e1_profile(profile_path, base), output)
    except (TopologyDiagnosticError, OSError, RuntimeError, ValueError) as error:
        _fail(error)
        return
    _print(result)


@model_app.command("train-e1")
def train_e1_command(
    dataset_root: Annotated[
        Path,
        typer.Option("--dataset-root", exists=True, file_okay=False, resolve_path=True),
    ],
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", exists=True, dir_okay=False, resolve_path=True),
    ],
    topology_diagnostic: Annotated[
        Path,
        typer.Option("--topology-diagnostic", exists=True, dir_okay=False, resolve_path=True),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", file_okay=False, resolve_path=True),
    ],
    base_profile_path: Annotated[
        Path,
        typer.Option("--base-profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e0-independent-v1.yaml"),
    profile_path: Annotated[
        Path,
        typer.Option("--profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e1-joint-v1.yaml"),
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
    """Train the sole E1 screening seed after the topology learning gate."""
    from junctionlens.data.license import DatasetRegistrationError, load_registration
    from junctionlens.data.manifests import ManifestError
    from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
    from junctionlens.model.e0_data import E0DataError, load_partition_isolation
    from junctionlens.model.e0_profile import load_e0_profile
    from junctionlens.model.e1_profile import load_e1_profile
    from junctionlens.model.e1_training import E1TrainingError, run_e1_training

    try:
        project_root = Path.cwd().resolve()
        registration = load_registration(project_root, "openlane-v2-v2.1", "full")
        registered_root = Path(str(registration["root"])).resolve(strict=True)
        if registered_root != dataset_root:
            raise DatasetRegistrationError(
                "E1 dataset root differs from the checksum-verified full registration"
            )
        base = load_e0_profile(base_profile_path)
        profile = load_e1_profile(profile_path, base)
        isolation = load_partition_isolation(
            split_manifest,
            split_policy,
            partition=base.training.training_partition,
            statistics=True,
        )
        result = run_e1_training(
            base,
            profile,
            OpenLaneAdapter(dataset_root, adapter_config),
            isolation,
            topology_diagnostic,
            output_root,
            project_root=project_root,
            dataset_registration=registration,
            device_name=device,
            resume=resume,
        )
    except (
        DatasetRegistrationError,
        E0DataError,
        E1TrainingError,
        ManifestError,
        OpenLaneAdapterError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        _fail(error)
        return
    _print(result)


@model_app.command("select-e1")
def select_e1_command(
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
    """Select an E1 epoch with the frozen topology, official, then NLL ordering."""
    from junctionlens.model.e0_training import E0TrainingError
    from junctionlens.model.e1_training import E1TrainingError, select_e1_checkpoint

    try:
        result = select_e1_checkpoint(
            run_root,
            scores,
            expected_selection_split_manifest_sha256=selection_split_manifest_sha256,
        )
    except (E0TrainingError, E1TrainingError, OSError, TypeError, ValueError) as error:
        _fail(error)
        return
    _print(result)


@model_app.command("finalize-e1-study")
def finalize_e1_study_command(
    baseline_run_root: Annotated[
        Path,
        typer.Option("--baseline-run-root", exists=True, file_okay=False, resolve_path=True),
    ],
    candidate_run_root: Annotated[
        Path,
        typer.Option("--candidate-run-root", exists=True, file_okay=False, resolve_path=True),
    ],
    evidence: Annotated[
        Path,
        typer.Option("--evidence", exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ],
    base_profile_path: Annotated[
        Path,
        typer.Option("--base-profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e0-independent-v1.yaml"),
    profile_path: Annotated[
        Path,
        typer.Option("--profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e1-joint-v1.yaml"),
) -> None:
    """Freeze the valid E1 study and either promote E1 or retain E0."""
    from junctionlens.model.e0_profile import load_e0_profile
    from junctionlens.model.e1_profile import load_e1_profile
    from junctionlens.model.e1_study import E1StudyError, finalize_e1_study

    try:
        base = load_e0_profile(base_profile_path)
        result = finalize_e1_study(
            base,
            load_e1_profile(profile_path, base),
            baseline_run_root,
            candidate_run_root,
            evidence,
            output,
        )
    except (E1StudyError, OSError, TypeError, ValueError) as error:
        _fail(error)
        return
    _print(result)


@model_app.command("fit-e0-linker")
def fit_e0_linker_command(
    dataset_root: Annotated[
        Path,
        typer.Option("--dataset-root", exists=True, file_okay=False, resolve_path=True),
    ],
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
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
) -> None:
    """Fit the independent baseline topology rules on model-training labels only."""
    from junctionlens.data.license import DatasetRegistrationError, load_registration
    from junctionlens.data.manifests import ManifestError
    from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
    from junctionlens.model.e0_data import E0DataError, load_partition_isolation
    from junctionlens.model.e0_profile import load_e0_profile
    from junctionlens.model.selection_evaluation import (
        SelectionEvaluationError,
        fit_e0_linker,
    )

    try:
        project_root = Path.cwd().resolve()
        registration = load_registration(project_root, "openlane-v2-v2.1", "full")
        if Path(str(registration["root"])).resolve(strict=True) != dataset_root:
            raise DatasetRegistrationError(
                "E0 linker dataset root differs from the checksum-verified registration"
            )
        profile = load_e0_profile(profile_path)
        artifact = fit_e0_linker(
            profile,
            OpenLaneAdapter(dataset_root, adapter_config),
            load_partition_isolation(
                split_manifest,
                split_policy,
                partition="model_training",
                statistics=False,
            ),
            output,
        )
    except (
        DatasetRegistrationError,
        E0DataError,
        ManifestError,
        OpenLaneAdapterError,
        OSError,
        SelectionEvaluationError,
        TypeError,
        ValueError,
    ) as error:
        _fail(error)
        return
    _print({"state": "ACCEPTED", **asdict(artifact)})


@model_app.command("score-checkpoints")
def score_checkpoints_command(
    experiment: Annotated[
        CheckpointExperiment,
        typer.Option("--experiment"),
    ],
    run_root: Annotated[
        Path,
        typer.Option("--run-root", exists=True, file_okay=False, resolve_path=True),
    ],
    dataset_root: Annotated[
        Path,
        typer.Option("--dataset-root", exists=True, file_okay=False, resolve_path=True),
    ],
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", exists=True, dir_okay=False, resolve_path=True),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", file_okay=False, resolve_path=True),
    ],
    linker_path: Annotated[
        Path | None,
        typer.Option("--linker", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    base_profile_path: Annotated[
        Path,
        typer.Option("--base-profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e0-independent-v1.yaml"),
    e1_profile_path: Annotated[
        Path,
        typer.Option("--e1-profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e1-joint-v1.yaml"),
    adapter_config: Annotated[
        Path,
        typer.Option("--adapter-config", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.adapter.yaml"),
    split_policy: Annotated[
        Path,
        typer.Option("--split-policy", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.split-v1.yaml"),
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Score all epochs on model-selection with official metrics and frozen NLL."""
    from junctionlens.data.license import DatasetRegistrationError, load_registration
    from junctionlens.data.manifests import ManifestError
    from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
    from junctionlens.model.e0_data import E0DataError, load_partition_isolation
    from junctionlens.model.e0_profile import load_e0_profile
    from junctionlens.model.e1_profile import load_e1_profile
    from junctionlens.model.selection_evaluation import (
        SelectionEvaluationError,
        load_e0_linker,
        score_checkpoints,
    )

    try:
        project_root = Path.cwd().resolve()
        registration = load_registration(project_root, "openlane-v2-v2.1", "full")
        if Path(str(registration["root"])).resolve(strict=True) != dataset_root:
            raise DatasetRegistrationError(
                "checkpoint scoring dataset differs from the checksum-verified registration"
            )
        base = load_e0_profile(base_profile_path)
        isolation = load_partition_isolation(
            split_manifest,
            split_policy,
            partition="model_selection",
            statistics=False,
        )
        experiment_id: Literal["E0-independent", "E1-joint"] = (
            "E0-independent" if experiment is CheckpointExperiment.E0 else "E1-joint"
        )
        if experiment_id == "E0-independent":
            if linker_path is None:
                raise SelectionEvaluationError("E0 checkpoint scoring requires --linker")
            linker = load_e0_linker(linker_path, base, isolation.split_manifest_sha256)
            e1 = None
        else:
            linker = None
            e1 = load_e1_profile(e1_profile_path, base)
        result = score_checkpoints(
            experiment=experiment_id,
            run_root=run_root,
            adapter=OpenLaneAdapter(dataset_root, adapter_config),
            isolation=isolation,
            base_profile=base,
            e1_profile=e1,
            linker=linker,
            project_root=project_root,
            output_root=output_root,
            device_name=device,
        )
    except (
        DatasetRegistrationError,
        E0DataError,
        ManifestError,
        OpenLaneAdapterError,
        OSError,
        RuntimeError,
        SelectionEvaluationError,
        TypeError,
        ValueError,
    ) as error:
        _fail(error)
        return
    _print(result)


@model_app.command("evaluate-selected")
def evaluate_selected_command(
    experiment: Annotated[
        CheckpointExperiment,
        typer.Option("--experiment"),
    ],
    run_root: Annotated[
        Path,
        typer.Option("--run-root", exists=True, file_okay=False, resolve_path=True),
    ],
    dataset_root: Annotated[
        Path,
        typer.Option("--dataset-root", exists=True, file_okay=False, resolve_path=True),
    ],
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", exists=True, dir_okay=False, resolve_path=True),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", file_okay=False, resolve_path=True),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", file_okay=False, resolve_path=True),
    ],
    linker_path: Annotated[
        Path | None,
        typer.Option("--linker", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    base_profile_path: Annotated[
        Path,
        typer.Option("--base-profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e0-independent-v1.yaml"),
    e1_profile_path: Annotated[
        Path,
        typer.Option("--e1-profile", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/model/e1-joint-v1.yaml"),
    adapter_config: Annotated[
        Path,
        typer.Option("--adapter-config", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.adapter.yaml"),
    split_policy: Annotated[
        Path,
        typer.Option("--split-policy", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.split-v1.yaml"),
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Evaluate one selected checkpoint with official and CustomMatchV1 evidence."""
    from junctionlens.data.license import DatasetRegistrationError, load_registration
    from junctionlens.data.manifests import ManifestError
    from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
    from junctionlens.model.e0_data import E0DataError, load_partition_isolation
    from junctionlens.model.e0_profile import load_e0_profile
    from junctionlens.model.e1_profile import load_e1_profile
    from junctionlens.model.selection_evaluation import (
        SelectionEvaluationError,
        evaluate_selected_checkpoint,
        load_e0_linker,
        resolve_source_commit,
    )

    try:
        project_root = Path.cwd().resolve()
        registration = load_registration(project_root, "openlane-v2-v2.1", "full")
        if Path(str(registration["root"])).resolve(strict=True) != dataset_root:
            raise DatasetRegistrationError(
                "selected evaluation dataset differs from the checksum-verified registration"
            )
        base = load_e0_profile(base_profile_path)
        isolation = load_partition_isolation(
            split_manifest,
            split_policy,
            partition="model_selection",
            statistics=False,
        )
        experiment_id: Literal["E0-independent", "E1-joint"] = (
            "E0-independent" if experiment is CheckpointExperiment.E0 else "E1-joint"
        )
        if experiment_id == "E0-independent":
            if linker_path is None:
                raise SelectionEvaluationError("E0 selected evaluation requires --linker")
            linker = load_e0_linker(linker_path, base, isolation.split_manifest_sha256)
            e1 = None
        else:
            linker = None
            e1 = load_e1_profile(e1_profile_path, base)
        result = evaluate_selected_checkpoint(
            experiment=experiment_id,
            run_root=run_root,
            adapter=OpenLaneAdapter(dataset_root, adapter_config),
            isolation=isolation,
            base_profile=base,
            e1_profile=e1,
            linker=linker,
            project_root=project_root,
            artifact_root=artifact_root,
            output_root=output_root,
            source_commit=resolve_source_commit(project_root),
            device_name=device,
        )
    except (
        DatasetRegistrationError,
        E0DataError,
        ManifestError,
        OpenLaneAdapterError,
        OSError,
        RuntimeError,
        SelectionEvaluationError,
        TypeError,
        ValueError,
    ) as error:
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
    from junctionlens.model.overfit import MicroOverfitError, run_micro_overfit
    from junctionlens.model.profile import load_m0_profile

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
    from junctionlens.model.export import ModelExportError, export_model
    from junctionlens.model.profile import load_m0_profile

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
    from junctionlens.model.parity import ParityError, run_parity
    from junctionlens.model.profile import load_m0_profile

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
    from junctionlens.model.profile import load_m0_profile
    from junctionlens.model.providers import ProviderProbeError, run_provider_probe

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
    from junctionlens.model.benchmark import BenchmarkError, run_m0_benchmark
    from junctionlens.model.budget import BudgetError, load_budget_plan
    from junctionlens.model.profile import load_m0_profile

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

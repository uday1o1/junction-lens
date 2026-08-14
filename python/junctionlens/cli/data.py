"""Licensed dataset CLI commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, cast

import typer
import yaml

from junctionlens.data.audit import audit_report
from junctionlens.data.license import (
    DatasetRegistrationError,
    acknowledge_licenses,
    load_registration,
    register_dataset,
)
from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
from junctionlens.data.parity import AdapterParityError, verify_official_parity
from junctionlens.evaluator.official import EvaluationError

data_app = typer.Typer(help="Acknowledge, register, audit, and verify licensed datasets.")


def _emit(payload: object) -> None:
    typer.echo(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _fail(error: Exception) -> None:
    typer.echo(f"data error: {error}", err=True)
    raise typer.Exit(code=2) from error


def _registered_root(dataset_id: str, profile: str, root: Path | None) -> Path:
    registration = load_registration(Path.cwd().resolve(), dataset_id, profile)
    selected_root = root
    if selected_root is None:
        environment_root = os.environ.get("OPENLANE_V2_ROOT")
        if environment_root is None:
            selected_root = Path(str(registration["root"]))
        else:
            selected_root = Path(environment_root)
    selected_root = selected_root.expanduser().resolve(strict=True)
    if selected_root != Path(str(registration["root"])).resolve(strict=True):
        raise DatasetRegistrationError(
            "selected root differs from the checksum-verified registration"
        )
    return selected_root


@data_app.command("acknowledge")
def acknowledge_command(
    accepted_terms: Annotated[
        list[str],
        typer.Option(
            "--accept-term",
            help="Repeat once for every exact license identifier in the dataset lock.",
        ),
    ],
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm-restricted-noncommercial-use",
            help="Confirm the dataset's restricted noncommercial use and no-redistribution terms.",
        ),
    ] = False,
    lock_path: Annotated[
        Path,
        typer.Option(
            "--lock",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = Path("configs/data/openlane-v2-v2.1.lock.yaml"),
) -> None:
    """Record explicit machine-local acceptance without downloading data."""
    try:
        payload = acknowledge_licenses(
            lock_path,
            Path.cwd().resolve(),
            accepted_terms,
            confirmed_restricted_noncommercial_use=confirm,
        )
    except (DatasetRegistrationError, OSError, ValueError) as error:
        _fail(error)
        return
    _emit(payload)


@data_app.command("register")
def register_command(
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ],
    lock_path: Annotated[
        Path,
        typer.Option("--lock", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.lock.yaml"),
    profile: Annotated[str, typer.Option("--profile")] = "sample",
    archive: Annotated[
        Path | None,
        typer.Option(
            "--archive",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Original official archive retained for checksum verification.",
        ),
    ] = None,
) -> None:
    """Register a checksum-verified local OpenLane profile."""
    try:
        payload = register_dataset(
            lock_path,
            Path.cwd().resolve(),
            root,
            profile=profile,
            archive_path=archive,
        )
    except (DatasetRegistrationError, OSError, ValueError) as error:
        _fail(error)
        return
    redacted = dict(payload)
    redacted["root"] = "<registered-dataset-root>"
    _emit(redacted)


@data_app.command("audit")
def audit_command(
    dataset_id: Annotated[str, typer.Option("--dataset")] = "openlane-v2-v2.1",
    profile: Annotated[str, typer.Option("--profile")] = "sample",
    root: Annotated[
        Path | None,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.adapter.yaml"),
) -> None:
    """Audit all declared labels and source identities in a registered profile."""
    try:
        selected_root = _registered_root(dataset_id, profile, root)
        with config_path.open(encoding="utf-8") as source:
            raw_config = yaml.safe_load(source)
        if not isinstance(raw_config, dict):
            raise DatasetRegistrationError("adapter config must be a mapping")
        config = cast(dict[str, Any], raw_config)
        capacities = cast(dict[str, int], config["query_capacities"])
        identity = cast(dict[str, float], config["identity_audit"])
        continuity = {
            "lane_segment": float(identity["lane_world_centroid_jump_m"]),
            "traffic_element": float(identity["traffic_normalized_center_jump"]),
            "area": float(identity["area_world_centroid_jump_m"]),
        }
        payload = audit_report(
            OpenLaneAdapter(selected_root, config_path).iter_frames(profile),
            capacities,
            continuity,
            required_coverage=float(config["capacity_required_coverage"]),
        )
    except (
        DatasetRegistrationError,
        OpenLaneAdapterError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        _fail(error)
        return
    _emit(payload)
    if not payload["capacity_gate_accepted"]:
        raise typer.Exit(code=2)


@data_app.command("verify-adapter")
def verify_adapter_command(
    dataset_id: Annotated[str, typer.Option("--dataset")] = "openlane-v2-v2.1",
    profile: Annotated[str, typer.Option("--profile")] = "sample",
    root: Annotated[
        Path | None,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.adapter.yaml"),
    selector_path: Annotated[
        Path,
        typer.Option("--selector", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.parity.yaml"),
) -> None:
    """Compare frozen licensed frames with the pinned official devkit API."""
    try:
        selected_root = _registered_root(dataset_id, profile, root)
        payload = verify_official_parity(
            OpenLaneAdapter(selected_root, config_path),
            selector_path,
            Path.cwd().resolve(),
        )
    except (
        AdapterParityError,
        DatasetRegistrationError,
        EvaluationError,
        OpenLaneAdapterError,
        OSError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        _fail(error)
        return
    _emit(payload)

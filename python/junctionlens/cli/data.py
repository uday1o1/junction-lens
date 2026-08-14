"""Licensed dataset CLI commands."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from junctionlens.cli.output import emit
from junctionlens.data.audit import audit_report
from junctionlens.data.license import (
    DatasetRegistrationError,
    acknowledge_licenses,
    load_registration,
    register_dataset,
)
from junctionlens.data.manifests import (
    ManifestError,
    audit_split_manifest,
    freeze_split_manifest,
    load_split_manifest,
    load_split_policy,
    split_records_from_frame_metadata,
    verify_frame_records,
    write_frame_records,
    write_immutable_split_manifest,
)
from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
from junctionlens.data.parity import AdapterParityError, verify_official_parity
from junctionlens.data.visual_audit import (
    VisualAuditError,
    load_audit_policy,
    write_audit_bundle,
    write_visual_audit_signoff,
)
from junctionlens.evaluator.official import EvaluationError
from junctionlens.registry import ContentAddressedStore, RegistryError
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_yaml_object_path

data_app = typer.Typer(help="Acknowledge, register, audit, and verify licensed datasets.")


def _emit(payload: object) -> None:
    emit(payload)


def _fail(error: Exception) -> None:
    typer.echo(f"data error: {error}", err=True)
    raise typer.Exit(code=2) from error


def _registered_dataset(
    dataset_id: str, profile: str, root: Path | None
) -> tuple[Path, Mapping[str, Any]]:
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
    return selected_root, registration


def _registered_root(dataset_id: str, profile: str, root: Path | None) -> Path:
    return _registered_dataset(dataset_id, profile, root)[0]


def _store(artifact_root: Path) -> ContentAddressedStore:
    return ContentAddressedStore(
        artifact_root,
        Path.cwd().resolve() / "schemas/artifact-manifest-v1.schema.json",
    )


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
        config = load_yaml_object_path(
            config_path,
            "adapter config",
            ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
        )
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
        ParseBoundaryError,
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
        ParseBoundaryError,
        TypeError,
        ValueError,
    ) as error:
        _fail(error)
        return
    _emit(payload)


@data_app.command("manifest")
def manifest_command(
    dataset_id: Annotated[str, typer.Option("--dataset")] = "openlane-v2-v2.1",
    profile: Annotated[str, typer.Option("--profile")] = "full",
    root: Annotated[
        Path | None,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.adapter.yaml"),
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
) -> None:
    """Stream registered frame identities into an immutable content artifact."""
    try:
        selected_root, registration = _registered_dataset(dataset_id, profile, root)
        store = _store(artifact_root)
        with tempfile.TemporaryDirectory(prefix="frame-manifest-", dir=store.staging_root) as temp:
            frame_records = Path(temp) / "frames.ndjson"
            metadata = write_frame_records(
                OpenLaneAdapter(selected_root, config_path),
                profile,
                frame_records,
                registration,
            )
            receipt = store.put_file(
                frame_records,
                kind="frame_manifest",
                media_type="application/x-ndjson",
                license_id="CC-BY-NC-SA-4.0",
                metadata=metadata,
            )
    except (
        DatasetRegistrationError,
        ManifestError,
        OpenLaneAdapterError,
        OSError,
        ParseBoundaryError,
        RegistryError,
        TypeError,
        ValueError,
    ) as error:
        _fail(error)
        return
    _emit(
        {
            "schema_version": "junctionlens.frame-manifest-receipt.v1",
            "state": "ACCEPTED",
            "artifact_manifest_sha256": receipt.manifest_sha256,
            "frame_records_sha256": receipt.payload_sha256,
            "frame_count": metadata["frame_count"],
            "segment_count": metadata["segment_count"],
            "split_segment_counts": metadata["split_segment_counts"],
        }
    )


@data_app.command("split")
@data_app.command("freeze-splits", hidden=True)
def freeze_splits_command(
    frame_manifest_sha256: Annotated[
        str,
        typer.Option("--frame-manifest-sha256"),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    policy_path: Annotated[
        Path,
        typer.Option("--policy", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.split-v1.yaml"),
    export_path: Annotated[
        Path | None,
        typer.Option("--export", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Freeze exact V1 partitions from a verified full-profile frame manifest."""
    try:
        store = _store(artifact_root)
        frame_artifact = store.read_manifest(frame_manifest_sha256)
        if frame_artifact.get("kind") != "frame_manifest":
            raise ManifestError("source artifact is not a frame manifest")
        raw_metadata = frame_artifact.get("metadata")
        raw_payload = frame_artifact.get("payload")
        if not isinstance(raw_metadata, dict) or not isinstance(raw_payload, dict):
            raise ManifestError("frame artifact has invalid manifest fields")
        metadata = cast(Mapping[str, Any], raw_metadata)
        payload = cast(Mapping[str, Any], raw_payload)
        policy = load_split_policy(policy_path)
        verify_frame_records(store.object_path(str(payload.get("sha256"))), metadata)
        records = split_records_from_frame_metadata(metadata, policy)
        source_registration = metadata.get("source_registration")
        if not isinstance(source_registration, dict):
            raise ManifestError("frame manifest lacks source registration evidence")
        split_manifest = freeze_split_manifest(
            records,
            policy,
            source_frame_manifest_sha256=frame_manifest_sha256,
            source_frame_records_sha256=str(payload.get("sha256")),
            source_dataset_manifest_sha256=str(source_registration.get("manifest_sha256")),
        )
        audit = audit_split_manifest(split_manifest, policy)
        split_payload = canonical_json_bytes(split_manifest) + b"\n"
        receipt = store.put_bytes(
            split_payload,
            kind="split_manifest",
            media_type="application/json",
            license_id="CC-BY-NC-SA-4.0",
            parents=(frame_manifest_sha256,),
            metadata={
                "schema_version": "junctionlens.split-manifest-artifact-metadata.v1",
                "policy_id": policy.policy_id,
                "segment_count": audit.segment_count,
                "partition_counts": dict(audit.partition_counts),
                "segment_catalog_sha256": audit.segment_catalog_sha256,
            },
        )
        export_sha256 = None
        if export_path is not None:
            export_sha256 = write_immutable_split_manifest(export_path, split_manifest)
            if export_sha256 != receipt.payload_sha256:
                raise ManifestError("registry and exported split payload hashes differ")
    except (ManifestError, OSError, RegistryError, TypeError, ValueError) as error:
        _fail(error)
        return
    response: dict[str, Any] = {
        "schema_version": "junctionlens.split-freeze-receipt.v1",
        "state": audit.state,
        "artifact_manifest_sha256": receipt.manifest_sha256,
        "split_manifest_sha256": receipt.payload_sha256,
        "segment_count": audit.segment_count,
        "partition_counts": dict(audit.partition_counts),
        "overlap_count": audit.overlap_count,
        "segment_catalog_sha256": audit.segment_catalog_sha256,
    }
    if export_sha256 is not None:
        response["export_sha256"] = export_sha256
    _emit(response)


@data_app.command("audit-splits")
def audit_splits_command(
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, resolve_path=True),
    ],
    policy_path: Annotated[
        Path,
        typer.Option("--policy", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.split-v1.yaml"),
) -> None:
    """Independently reject split overlap, count, provenance, and hash defects."""
    try:
        audit = audit_split_manifest(
            load_split_manifest(manifest_path),
            load_split_policy(policy_path),
        )
    except (ManifestError, OSError, TypeError, ValueError) as error:
        _fail(error)
        return
    _emit(
        {
            "schema_version": "junctionlens.split-audit-report.v1",
            **asdict(audit),
        }
    )


@data_app.command("visual-audit")
def visual_audit_command(
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
    policy_path: Annotated[
        Path,
        typer.Option("--policy", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("configs/data/openlane-v2-v2.1.audit-v1.yaml"),
    output_root: Annotated[
        Path,
        typer.Option("--output", file_okay=False, resolve_path=True),
    ] = Path("artifacts/data-audit/openlane-v2-v2.1-sample"),
) -> None:
    """Create private calibration overlays, BEV labels, and aggregate data evidence."""
    try:
        selected_root = _registered_root(dataset_id, profile, root)
        receipt = write_audit_bundle(
            OpenLaneAdapter(selected_root, config_path),
            profile,
            load_audit_policy(policy_path),
            output_root,
        )
    except (
        DatasetRegistrationError,
        OSError,
        OpenLaneAdapterError,
        ParseBoundaryError,
        TypeError,
        ValueError,
        VisualAuditError,
    ) as error:
        _fail(error)
        return
    _emit(
        {
            "schema_version": "junctionlens.visual-audit-receipt.v1",
            "state": "PENDING_HUMAN_INSPECTION",
            **asdict(receipt),
        }
    )
    if not receipt.range_gate_accepted:
        raise typer.Exit(code=2)


@data_app.command("signoff-visual-audit")
def signoff_visual_audit_command(
    bundle_root: Annotated[
        Path,
        typer.Option("--bundle", exists=True, file_okay=False, resolve_path=True),
    ],
    camera_projection_alignment_accepted: Annotated[
        bool,
        typer.Option("--accept-camera-projection-alignment"),
    ] = False,
    bev_geometry_alignment_accepted: Annotated[
        bool,
        typer.Option("--accept-bev-geometry-alignment"),
    ] = False,
    label_identity_and_topology_accepted: Annotated[
        bool,
        typer.Option("--accept-label-identity-and-topology"),
    ] = False,
    private_data_handling_confirmed: Annotated[
        bool,
        typer.Option("--confirm-private-data-handling"),
    ] = False,
) -> None:
    """Record explicit inspection of a digest-verified private audit bundle."""
    try:
        receipt = write_visual_audit_signoff(
            bundle_root,
            Path.cwd().resolve(),
            camera_projection_alignment_accepted=camera_projection_alignment_accepted,
            bev_geometry_alignment_accepted=bev_geometry_alignment_accepted,
            label_identity_and_topology_accepted=label_identity_and_topology_accepted,
            private_data_handling_confirmed=private_data_handling_confirmed,
        )
    except (OSError, ParseBoundaryError, VisualAuditError, ValueError) as error:
        _fail(error)
        return
    _emit({**receipt, "state": "ACCEPTED"})

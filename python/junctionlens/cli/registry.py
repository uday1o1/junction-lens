"""Public immutable artifact registry and provenance commands."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from junctionlens.registry.service import EvidenceRegistry, RunIdentity

registry_app = typer.Typer(
    help="Store, index, resume, and inspect immutable evidence.", no_args_is_help=True
)


def _print(value: object) -> None:
    typer.echo(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"{label} must be a bounded regular file")

    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _registry(root: Path, schema: Path) -> EvidenceRegistry:
    return EvidenceRegistry(root, schema)


@registry_app.command("put")
def put_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, resolve_path=True),
    ],
    kind: Annotated[str, typer.Option("--kind")],
    media_type: Annotated[str, typer.Option("--media-type")],
    license_id: Annotated[str, typer.Option("--license-id")],
    metadata_path: Annotated[
        Path,
        typer.Option("--metadata", exists=True, dir_okay=False, resolve_path=True),
    ],
    parents: Annotated[list[str] | None, typer.Option("--parent")] = None,
    artifact_root: Annotated[
        Path, typer.Option("--artifact-root", file_okay=False, resolve_path=True)
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
) -> None:
    """Store and index one immutable artifact with declared parent manifests."""
    try:
        receipt = _registry(artifact_root, schema).put_file(
            input_path,
            kind=kind,
            media_type=media_type,
            license_id=license_id,
            metadata=_load_object(metadata_path, "artifact metadata"),
            parents=parents or (),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"registry error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(asdict(receipt))


@registry_app.command("inspect")
def inspect_command(
    manifest_sha256: Annotated[str, typer.Option("--manifest")],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
) -> None:
    """Read one indexed artifact and cross-check its immutable manifest."""
    try:
        result = _registry(artifact_root, schema).index.read_artifact(manifest_sha256)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"registry error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(result)


@registry_app.command("provenance")
def provenance_command(
    manifest_sha256: Annotated[str, typer.Option("--manifest")],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
) -> None:
    """Render the deterministic transitive provenance view for one artifact."""
    try:
        result = _registry(artifact_root, schema).provenance(manifest_sha256)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"registry error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(result)


@registry_app.command("resume")
def resume_command(
    identity_path: Annotated[
        Path,
        typer.Option("--identity", exists=True, dir_okay=False, resolve_path=True),
    ],
    environment_fingerprint: Annotated[str, typer.Option("--environment-fingerprint")],
    artifact_root: Annotated[
        Path, typer.Option("--artifact-root", file_okay=False, resolve_path=True)
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
) -> None:
    """Create or resume one run only when every input and environment hash matches."""
    try:
        identity = RunIdentity.model_validate(_load_object(identity_path, "run identity"))
        result = _registry(artifact_root, schema).begin_or_resume_run(
            identity, environment_fingerprint
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"registry error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(result)


@registry_app.command("gc")
def gc_command(
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
) -> None:
    """Audit reclaimable unindexed objects without deleting registry data."""
    if not dry_run:
        typer.echo("registry error: V1 garbage collection requires --dry-run", err=True)
        raise typer.Exit(code=2)
    try:
        result = _registry(artifact_root, schema).garbage_collection_dry_run()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"registry error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(result)


@registry_app.command("rebuild-index")
def rebuild_index_command(
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
) -> None:
    """Recover missing DuckDB rows from verified immutable manifests."""
    try:
        result = _registry(artifact_root, schema).rebuild_missing_index_rows()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"registry error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(result)


@registry_app.command("set-alias")
def set_alias_command(
    alias: Annotated[str, typer.Option("--alias")],
    manifest_sha256: Annotated[str, typer.Option("--manifest")],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
) -> None:
    """Move a human-readable alias without changing evidence identity."""
    try:
        result = _registry(artifact_root, schema).set_alias(alias, manifest_sha256)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"registry error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(result)


@registry_app.command("resolve-alias")
def resolve_alias_command(
    alias: Annotated[str, typer.Option("--alias")],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts"),
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False, resolve_path=True),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
) -> None:
    """Resolve a convenience alias and label it as non-evidence."""
    try:
        result = _registry(artifact_root, schema).resolve_alias(alias)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"registry error: {error}", err=True)
        raise typer.Exit(code=2) from error
    _print(result)


__all__ = ["registry_app"]

"""Public graph-contract inspection and conversion commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from junctionlens.contract import (
    ContractViolation,
    canonical_logical_sha256,
    parse_binary,
    parse_json,
    to_binary,
    to_json,
)
from junctionlens.contract.limits import MAX_SERIALIZED_BYTES
from junctionlens.security.parsing import ParseBoundaryError, read_bounded_file

contract_app = typer.Typer(help="Validate and convert SceneControlGraph V1 artifacts.")


def _parse(path: Path, encoding: str):  # type: ignore[no-untyped-def]
    try:
        payload = read_bounded_file(path, "contract input", MAX_SERIALIZED_BYTES)
    except ParseBoundaryError as error:
        raise ContractViolation("CONTRACT_IO", encoding, error.detail) from error
    if encoding == "binary":
        return parse_binary(payload)
    return parse_json(payload)


@contract_app.command("validate")
def validate_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    encoding: Annotated[str, typer.Option(help="Input encoding: binary or json.")] = "binary",
) -> None:
    """Validate a bounded graph artifact and print its logical identity."""
    if encoding not in {"binary", "json"}:
        typer.echo("contract error: encoding must be binary or json", err=True)
        raise typer.Exit(code=2)
    try:
        envelope = _parse(input_path, encoding)
    except (ContractViolation, OSError) as error:
        typer.echo(f"contract error: {error}", err=True)
        raise typer.Exit(code=2) from error
    result = {
        "schema_major": envelope.schema_major,
        "schema_minor": envelope.schema_minor,
        "logical_sha256": canonical_logical_sha256(envelope),
        "nodes": len(envelope.graph.lanes)
        + len(envelope.graph.traffic_controls)
        + len(envelope.graph.road_areas),
        "edges": len(envelope.graph.edges),
    }
    typer.echo(json.dumps(result, sort_keys=True, separators=(",", ":")))


@contract_app.command("convert")
def convert_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Option("--output", dir_okay=False)],
    from_encoding: Annotated[str, typer.Option("--from", help="Input encoding.")],
    to_encoding: Annotated[str, typer.Option("--to", help="Output encoding.")],
) -> None:
    """Convert a validated artifact between binary and strict ProtoJSON."""
    if from_encoding not in {"binary", "json"} or to_encoding not in {"binary", "json"}:
        typer.echo("contract error: encodings must be binary or json", err=True)
        raise typer.Exit(code=2)
    try:
        envelope = _parse(input_path, from_encoding)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if to_encoding == "binary":
            output_path.write_bytes(to_binary(envelope))
        else:
            output_path.write_text(to_json(envelope), encoding="utf-8")
    except (ContractViolation, OSError) as error:
        typer.echo(f"contract error: {error}", err=True)
        raise typer.Exit(code=2) from error

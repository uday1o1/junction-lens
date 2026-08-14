"""Public synthetic truth generation and verification commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated

import typer

from junctionlens.synthetic import SyntheticCorpusError, verify_corpus, write_corpus

synthetic_app = typer.Typer(help="Generate and verify unrestricted calibrated graph truth.")


def _result(root: Path, seed: int, files: int, manifest: bytes) -> str:
    return json.dumps(
        {
            "file_count": files,
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "root": str(root),
            "seed": seed,
            "state": "ACCEPTED",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


@synthetic_app.command("generate")
def generate_command(
    output: Annotated[Path, typer.Option("--output", file_okay=False, resolve_path=False)],
    seed: Annotated[int, typer.Option(min=0, max=(1 << 64) - 1)] = 20_260_813,
) -> None:
    """Generate the complete byte-stable synthetic V1 corpus."""
    try:
        corpus = write_corpus(output, seed=seed)
    except (OSError, SyntheticCorpusError, ValueError) as error:
        typer.echo(f"synthetic error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(_result(output, seed, len(corpus.files), corpus.files["manifest.json"]))


@synthetic_app.command("verify")
def verify_command(
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=False),
    ],
    seed: Annotated[int, typer.Option(min=0, max=(1 << 64) - 1)] = 20_260_813,
) -> None:
    """Regenerate and byte-compare an existing synthetic V1 corpus."""
    try:
        corpus = verify_corpus(root, seed=seed)
    except (OSError, SyntheticCorpusError, ValueError) as error:
        typer.echo(f"synthetic error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(_result(root, seed, len(corpus.files), corpus.files["manifest.json"]))

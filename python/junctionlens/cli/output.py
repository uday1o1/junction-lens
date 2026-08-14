"""Consistent machine-readable and optional human-readable CLI output."""

from __future__ import annotations

import json
from collections.abc import Mapping

import click
import typer


def human_requested(explicit: bool = False) -> bool:
    """Return the command override or root-level presentation preference."""
    context = click.get_current_context(silent=True)
    if context is None:
        return explicit
    return explicit or bool(context.find_root().params.get("human", False))


def emit(payload: object, *, human: bool = False) -> None:
    """Emit canonical compact JSON unless a human view was requested."""
    if not human_requested(human):
        typer.echo(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return
    if not isinstance(payload, Mapping):
        typer.echo(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
        return
    for key in sorted(payload, key=str):
        value = payload[key]
        if value is None or isinstance(value, bool | int | float | str):
            rendered = json.dumps(value, sort_keys=True, allow_nan=False)
            typer.echo(f"{key}: {rendered}")
        else:
            typer.echo(f"{key}:")
            typer.echo(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


__all__ = ["emit", "human_requested"]

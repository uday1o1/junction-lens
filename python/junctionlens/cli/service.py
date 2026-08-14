"""Public local evidence service command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from junctionlens.cli.output import emit, human_requested


def _error(code: str, message: str) -> str:
    return json.dumps(
        {
            "schema_version": "junctionlens.cli-error.v1",
            "error": {"code": code, "message": message},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def serve_command(
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", exists=True, file_okay=False),
    ],
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    schema: Annotated[
        Path,
        typer.Option("--schema", exists=True, dir_okay=False),
    ] = Path("schemas/artifact-manifest-v1.schema.json"),
    web_root: Annotated[
        Path,
        typer.Option("--web-root", file_okay=False),
    ] = Path("web/dist"),
    api_only: Annotated[
        bool,
        typer.Option("--api-only", help="Serve only the read API without the bundled viewer."),
    ] = False,
    open_browser: Annotated[bool, typer.Option("--open-browser")] = False,
    check: Annotated[
        bool,
        typer.Option("--check", help="Validate the service configuration without binding a port."),
    ] = False,
    human: Annotated[bool, typer.Option("--human")] = False,
) -> None:
    """Serve registered evidence read-only on the exact V1 loopback address."""
    try:
        from junctionlens.api.models import ServiceConfig
        from junctionlens.api.repository import EvidenceReadError
        from junctionlens.api.server import check_service, run_service

        config = ServiceConfig(
            artifact_root=artifact_root,
            schema_path=schema,
            web_root=None if api_only else web_root,
        )
        status = check_service(config, host=host, port=port)
        if check:
            if human_requested(human):
                typer.echo(
                    f"service ready: {status['artifact_count']} artifacts, "
                    f"{status['run_count']} runs at http://{host}:{port}"
                )
            else:
                emit(status)
            return
        run_service(
            config,
            host=host,
            port=port,
            open_browser=open_browser,
        )
    except (EvidenceReadError, ImportError, OSError, RuntimeError, ValueError) as error:
        typer.echo(_error("SERVE_CONFIGURATION_INVALID", str(error)), err=True)
        raise typer.Exit(code=2) from error


__all__ = ["serve_command"]

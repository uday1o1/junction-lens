"""Loopback-only service launcher."""

from __future__ import annotations

import threading
import webbrowser
from collections.abc import Mapping
from typing import Any

from junctionlens.api.app import create_app
from junctionlens.api.models import ServiceConfig
from junctionlens.api.repository import EvidenceRepository


class ServeError(RuntimeError):
    """Raised when the local-only service contract cannot be honored."""


def validate_host(host: str) -> None:
    """Reject every bind address except the frozen V1 loopback address."""
    if host != "127.0.0.1":
        raise ServeError("V1 service host must be exactly 127.0.0.1")


def check_service(config: ServiceConfig, *, host: str, port: int) -> Mapping[str, object]:
    """Validate configuration and registry readability without opening a socket."""
    validate_host(host)
    if not 1 <= port <= 65535:
        raise ServeError("service port must be between 1 and 65535")
    artifact_count, run_count = EvidenceRepository(config).counts()
    if config.web_root is not None:
        create_app(config)
    return {
        "schema_version": "junctionlens.serve-status.v1",
        "state": "READY",
        "host": host,
        "port": port,
        "artifact_count": artifact_count,
        "run_count": run_count,
        "read_only": True,
        "viewer_available": config.web_root is not None,
    }


def run_service(
    config: ServiceConfig,
    *,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    """Run the API on the exact V1 loopback interface."""
    check_service(config, host=host, port=port)
    try:
        import uvicorn
    except ImportError as error:
        raise ServeError("serve requires the pinned service dependency set") from error
    if open_browser:
        suffix = "/" if config.web_root is not None else "/api/v1/health"
        url = f"http://{host}:{port}{suffix}"
        timer = threading.Timer(0.75, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    options: dict[str, Any] = {
        "host": host,
        "port": port,
        "access_log": False,
        "log_level": "warning",
    }
    uvicorn.run(create_app(config), **options)


__all__ = ["ServeError", "check_service", "run_service", "validate_host"]

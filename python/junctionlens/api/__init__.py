"""Read-only local evidence API."""

from junctionlens.api.app import create_app
from junctionlens.api.models import ServiceConfig

__all__ = ["ServiceConfig", "create_app"]

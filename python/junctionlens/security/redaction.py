"""Deterministic redaction for diagnostic text and command evidence."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_AUTHORIZATION = re.compile(r"(?i)\b(authorization\s*[:=]\s*)(bearer|basic)\s+\S+")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_URL = re.compile(r"https?://[^\s<>\"']+")
_SENSITIVE_QUERY_NAMES = frozenset(
    {
        "access_token",
        "authorization",
        "credential",
        "key",
        "password",
        "sig",
        "signature",
        "token",
        "x-amz-credential",
        "x-amz-security-token",
        "x-amz-signature",
        "x-goog-credential",
        "x-goog-signature",
    }
)


def _redact_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    parts = urlsplit(raw)
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    netloc = hostname + port
    if parts.username is not None or parts.password is not None:
        netloc = f"[REDACTED]@{netloc}"
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        secret = lowered in _SENSITIVE_QUERY_NAMES or lowered.startswith(("x-amz-", "x-goog-"))
        query.append((key, "[REDACTED]" if secret else value))
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), parts.fragment))


def redact_sensitive_text(text: str, roots: tuple[Path, ...] = ()) -> str:
    """Remove credentials, signed query values, and explicitly supplied local roots."""
    redacted = text
    normalized_roots = sorted(
        {str(root.expanduser().resolve(strict=False)) for root in roots},
        key=len,
        reverse=True,
    )
    for root in normalized_roots:
        if root not in {"", "/"}:
            redacted = redacted.replace(root, "[LOCAL_ROOT]")
    redacted = _URL.sub(_redact_url, redacted)
    redacted = _AUTHORIZATION.sub(r"\1\2 [REDACTED]", redacted)
    return _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", redacted)


__all__ = ["redact_sensitive_text"]

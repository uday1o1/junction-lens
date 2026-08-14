"""Bounded binary and ProtoJSON conversion for V1 envelopes."""

from __future__ import annotations

import hashlib
import json
from typing import cast

from google.protobuf import json_format
from google.protobuf.message import DecodeError

from junctionlens.contract.limits import MAX_SERIALIZED_BYTES
from junctionlens.contract.validation import ContractViolation, validate_envelope
from junctionlens.v1 import scene_control_graph_pb2 as scg


def _bounded(payload: bytes, encoding: str) -> None:
    if len(payload) > MAX_SERIALIZED_BYTES:
        raise ContractViolation(
            "CONTRACT_SIZE_LIMIT",
            encoding,
            f"payload exceeds {MAX_SERIALIZED_BYTES} bytes",
        )


def parse_binary(payload: bytes) -> scg.SceneControlGraphEnvelope:
    """Parse and validate a bounded Protobuf binary payload."""
    _bounded(payload, "binary")
    envelope = scg.SceneControlGraphEnvelope()
    try:
        envelope.ParseFromString(payload)
    except DecodeError as error:
        raise ContractViolation("CONTRACT_BINARY_MALFORMED", "binary", str(error)) from error
    validate_envelope(envelope)
    return envelope


def to_binary(envelope: scg.SceneControlGraphEnvelope) -> bytes:
    """Validate and serialize an envelope without claiming canonical wire bytes."""
    validate_envelope(envelope)
    payload = cast(bytes, envelope.SerializeToString(deterministic=True))
    _bounded(payload, "binary")
    return payload


def parse_json(payload: str | bytes) -> scg.SceneControlGraphEnvelope:
    """Parse strict ProtoJSON, rejecting fields this version cannot preserve."""
    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    _bounded(encoded, "json")
    envelope = scg.SceneControlGraphEnvelope()
    try:
        json_format.Parse(encoded.decode("utf-8"), envelope, ignore_unknown_fields=False)
    except (json_format.ParseError, UnicodeDecodeError) as error:
        raise ContractViolation("CONTRACT_JSON_MALFORMED", "json", str(error)) from error
    validate_envelope(envelope)
    return envelope


def _logical_dict(envelope: scg.SceneControlGraphEnvelope) -> dict[str, object]:
    validate_envelope(envelope)
    value = json_format.MessageToDict(
        envelope,
        preserving_proto_field_name=True,
        use_integers_for_enums=False,
        always_print_fields_with_no_presence=True,
    )
    return dict(value)


def canonical_logical_json(envelope: scg.SceneControlGraphEnvelope) -> str:
    """Return the V1 known-field logical projection in canonical JSON form."""
    return json.dumps(
        _logical_dict(envelope),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_logical_sha256(envelope: scg.SceneControlGraphEnvelope) -> str:
    """Hash logical content rather than implementation-dependent wire bytes."""
    return hashlib.sha256(canonical_logical_json(envelope).encode("utf-8")).hexdigest()


def to_json(envelope: scg.SceneControlGraphEnvelope, *, indent: int | None = 2) -> str:
    """Return deterministic, field-name-preserving ProtoJSON."""
    logical = _logical_dict(envelope)
    separators = (",", ":") if indent is None else None
    return json.dumps(
        logical,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        separators=separators,
        sort_keys=True,
    ) + ("\n" if indent is not None else "")

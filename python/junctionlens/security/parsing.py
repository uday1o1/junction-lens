"""Fail-closed JSON and YAML parsing with explicit resource budgets."""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode


class ParseBoundaryError(ValueError):
    """A stable, typed failure at an untrusted structured-data boundary."""

    def __init__(self, code: str, label: str, detail: str) -> None:
        super().__init__(f"{label}: {detail}")
        self.code = code
        self.label = label
        self.detail = detail


@dataclass(frozen=True)
class ParseLimits:
    """Resource limits applied before a parsed value reaches application code."""

    max_bytes: int = 16 * 1024 * 1024
    max_depth: int = 64
    max_nodes: int = 250_000
    max_container_items: int = 100_000
    max_string_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            min(
                self.max_bytes,
                self.max_depth,
                self.max_nodes,
                self.max_container_items,
                self.max_string_bytes,
            )
            < 1
        ):
            raise ValueError("parse limits must be positive")


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate mapping keys."""


class _DuplicateYamlKey(ValueError):
    """Internal signal that preserves the duplicate key without parser diagnostics."""

    def __init__(self, key: object) -> None:
        super().__init__(str(key))
        self.key = key


def _construct_mapping(
    loader: _StrictSafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
        try:
            duplicate = key in result
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise _DuplicateYamlKey(key)
        result[key] = loader.construct_object(  # type: ignore[no-untyped-call]
            value_node, deep=deep
        )
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def read_bounded_file(path: Path, label: str, max_bytes: int) -> bytes:
    """Read one ordinary file without following its final symlink or exceeding a byte cap."""
    if max_bytes < 1:
        raise ValueError("file byte limit must be positive")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ParseBoundaryError("PARSE_FILE_TYPE", label, "input is not a regular file")
        if details.st_size > max_bytes:
            raise ParseBoundaryError("PARSE_BYTE_LIMIT", label, "payload exceeds the byte limit")
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            descriptor = None
            payload = source.read(max_bytes + 1)
    except ParseBoundaryError:
        raise
    except OSError as error:
        raise ParseBoundaryError("PARSE_IO", label, "input cannot be read safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(payload) > max_bytes:
        raise ParseBoundaryError("PARSE_BYTE_LIMIT", label, "payload exceeds the byte limit")
    return payload


def _decode(payload: bytes, label: str, limits: ParseLimits) -> str:
    if len(payload) > limits.max_bytes:
        raise ParseBoundaryError("PARSE_BYTE_LIMIT", label, "payload exceeds the byte limit")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ParseBoundaryError("PARSE_ENCODING", label, "payload is not UTF-8") from error


def _validate_shape(value: object, label: str, limits: ParseLimits) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > limits.max_nodes:
            raise ParseBoundaryError("PARSE_NODE_LIMIT", label, "value exceeds the node limit")
        if depth > limits.max_depth:
            raise ParseBoundaryError("PARSE_DEPTH_LIMIT", label, "value exceeds the depth limit")
        if isinstance(item, str):
            if len(item.encode("utf-8")) > limits.max_string_bytes:
                raise ParseBoundaryError(
                    "PARSE_STRING_LIMIT", label, "string exceeds the byte limit"
                )
        elif isinstance(item, list):
            if len(item) > limits.max_container_items:
                raise ParseBoundaryError(
                    "PARSE_CONTAINER_LIMIT", label, "array exceeds the item limit"
                )
            stack.extend((child, depth + 1) for child in reversed(item))
        elif isinstance(item, dict):
            if len(item) > limits.max_container_items:
                raise ParseBoundaryError(
                    "PARSE_CONTAINER_LIMIT", label, "object exceeds the item limit"
                )
            for key, child in reversed(tuple(item.items())):
                stack.append((child, depth + 1))
                stack.append((key, depth + 1))
        elif isinstance(item, float) and not math.isfinite(item):
            raise ParseBoundaryError("PARSE_NONFINITE", label, "number must be finite")
        elif item is not None and not isinstance(item, (bool, int, float)):  # noqa: UP038
            raise ParseBoundaryError(
                "PARSE_TYPE_UNSUPPORTED",
                label,
                f"value contains unsupported type {type(item).__name__}",
            )


def load_json(payload: bytes, label: str, limits: ParseLimits | None = None) -> object:
    """Parse one UTF-8 JSON value without duplicates, nonfinite numbers, or excess shape."""
    active = limits or ParseLimits()
    text = _decode(payload, label, active)

    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ParseBoundaryError(
                    "PARSE_DUPLICATE_KEY", label, f"duplicate JSON object key: {key}"
                )
            result[key] = value
        return result

    def reject_nonfinite(item: str) -> None:
        raise ParseBoundaryError("PARSE_NONFINITE", label, f"JSON constant {item} is not permitted")

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except ParseBoundaryError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise ParseBoundaryError("PARSE_JSON_INVALID", label, "invalid JSON payload") from error
    _validate_shape(value, label, active)
    return value


def load_json_object(
    payload: bytes,
    label: str,
    limits: ParseLimits | None = None,
) -> dict[str, Any]:
    """Parse one bounded JSON object."""
    value = load_json(payload, label, limits)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ParseBoundaryError(
            "PARSE_OBJECT_REQUIRED", label, "top-level value must be an object"
        )
    return value


def load_json_path(
    path: Path,
    label: str,
    limits: ParseLimits | None = None,
) -> object:
    """Read and parse one bounded JSON file."""
    active = limits or ParseLimits()
    return load_json(read_bounded_file(path, label, active.max_bytes), label, active)


def load_json_object_path(
    path: Path,
    label: str,
    limits: ParseLimits | None = None,
) -> dict[str, Any]:
    """Read and parse one bounded JSON object file."""
    active = limits or ParseLimits()
    return load_json_object(read_bounded_file(path, label, active.max_bytes), label, active)


def load_yaml(payload: bytes, label: str, limits: ParseLimits | None = None) -> object:
    """Parse one bounded YAML document while rejecting aliases and duplicate keys."""
    active = limits or ParseLimits(max_bytes=4 * 1024 * 1024)
    text = _decode(payload, label, active)
    event_count = 0
    try:
        for event in yaml.parse(text, Loader=_StrictSafeLoader):
            event_count += 1
            if event_count > active.max_nodes * 2:
                raise ParseBoundaryError(
                    "PARSE_NODE_LIMIT", label, "YAML event stream exceeds the node limit"
                )
            if isinstance(event, AliasEvent):
                raise ParseBoundaryError("PARSE_YAML_ALIAS", label, "YAML aliases are not accepted")
        value = yaml.load(text, Loader=_StrictSafeLoader)  # noqa: S506
    except ParseBoundaryError:
        raise
    except _DuplicateYamlKey as error:
        raise ParseBoundaryError(
            "PARSE_DUPLICATE_KEY", label, f"duplicate YAML key: {error.key}"
        ) from error
    except (yaml.YAMLError, RecursionError) as error:
        raise ParseBoundaryError(
            "PARSE_YAML_INVALID", label, "payload is not valid YAML"
        ) from error
    _validate_shape(value, label, active)
    return value


def load_yaml_object(
    payload: bytes,
    label: str,
    limits: ParseLimits | None = None,
) -> dict[str, Any]:
    """Parse one bounded YAML mapping."""
    value = load_yaml(payload, label, limits)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ParseBoundaryError(
            "PARSE_OBJECT_REQUIRED", label, "top-level value must be a mapping"
        )
    return value


def load_yaml_path(
    path: Path,
    label: str,
    limits: ParseLimits | None = None,
) -> object:
    """Read and parse one bounded YAML file."""
    active = limits or ParseLimits(max_bytes=4 * 1024 * 1024)
    return load_yaml(read_bounded_file(path, label, active.max_bytes), label, active)


def load_yaml_object_path(
    path: Path,
    label: str,
    limits: ParseLimits | None = None,
) -> dict[str, Any]:
    """Read and parse one bounded YAML mapping file."""
    active = limits or ParseLimits(max_bytes=4 * 1024 * 1024)
    return load_yaml_object(read_bounded_file(path, label, active.max_bytes), label, active)


__all__ = [
    "ParseBoundaryError",
    "ParseLimits",
    "load_json",
    "load_json_object",
    "load_json_object_path",
    "load_json_path",
    "load_yaml",
    "load_yaml_object",
    "load_yaml_object_path",
    "load_yaml_path",
    "read_bounded_file",
]

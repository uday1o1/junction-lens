"""Adversarial coverage for structured-data and redaction boundaries."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from junctionlens.security.parsing import (
    ParseBoundaryError,
    ParseLimits,
    load_json,
    load_json_object,
    load_json_object_path,
    load_yaml,
)
from junctionlens.security.redaction import redact_sensitive_text


def test_json_rejects_duplicate_nonfinite_and_oversized_values() -> None:
    with pytest.raises(ParseBoundaryError) as duplicate:
        load_json_object(b'{"key":1,"key":2}', "seed")
    assert duplicate.value.code == "PARSE_DUPLICATE_KEY"

    with pytest.raises(ParseBoundaryError) as nonfinite:
        load_json(b'{"value":NaN}', "seed")
    assert nonfinite.value.code == "PARSE_NONFINITE"

    limits = ParseLimits(max_bytes=8, max_depth=2, max_nodes=4, max_container_items=2)
    with pytest.raises(ParseBoundaryError) as oversized:
        load_json(b'{"long":1}', "seed", limits)
    assert oversized.value.code == "PARSE_BYTE_LIMIT"


def test_json_and_yaml_depth_fail_with_typed_errors_instead_of_recursion() -> None:
    limits = ParseLimits(max_depth=8, max_nodes=1000)
    nested_json = ("[" * 20 + "0" + "]" * 20).encode()
    with pytest.raises(ParseBoundaryError) as json_error:
        load_json(nested_json, "nested JSON", limits)
    assert json_error.value.code == "PARSE_DEPTH_LIMIT"

    nested_yaml = (
        "".join(f"{'  ' * depth}value:\n" for depth in range(20)) + f"{'  ' * 20}leaf: true\n"
    ).encode()
    with pytest.raises(ParseBoundaryError) as yaml_error:
        load_yaml(nested_yaml, "nested YAML", limits)
    assert yaml_error.value.code in {"PARSE_DEPTH_LIMIT", "PARSE_YAML_INVALID"}


def test_yaml_rejects_aliases_and_duplicate_keys() -> None:
    with pytest.raises(ParseBoundaryError) as alias:
        load_yaml(b"base: &base [1, 2]\ncopy: *base\n", "alias")
    assert alias.value.code == "PARSE_YAML_ALIAS"

    with pytest.raises(ParseBoundaryError) as duplicate:
        load_yaml(b"key: 1\nkey: 2\n", "duplicate")
    assert duplicate.value.code == "PARSE_DUPLICATE_KEY"


@given(st.binary(max_size=2048))
@settings(max_examples=250, deadline=None)
def test_json_boundary_never_leaks_untyped_parser_failures(payload: bytes) -> None:
    with suppress(ParseBoundaryError):
        load_json(payload, "fuzz", ParseLimits(max_bytes=2048, max_nodes=2048))


def test_redaction_removes_roots_signed_urls_and_credentials(tmp_path: Path) -> None:
    credential_url = (
        "https://" + "user" + ":" + "password" + "@example.test/file?X-Amz-Signature="
        "deadbeef&part=1"
    )
    raw = (
        f"root={tmp_path.resolve()} "
        f"{credential_url} "
        "Authorization: Bearer token-value api_key=top-secret"
    )
    redacted = redact_sensitive_text(raw, (tmp_path,))

    assert str(tmp_path.resolve()) not in redacted
    assert "password" not in redacted
    assert "deadbeef" not in redacted
    assert "token-value" not in redacted
    assert "top-secret" not in redacted
    assert "part=1" in redacted


def test_bounded_file_parser_rejects_symlink_and_oversized_input(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"value":1}', encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)

    with pytest.raises(ParseBoundaryError) as symlink:
        load_json_object_path(alias, "symlink input")
    assert symlink.value.code == "PARSE_IO"

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b'{"value":"123456"}')
    with pytest.raises(ParseBoundaryError) as byte_limit:
        load_json_object_path(
            oversized,
            "oversized input",
            ParseLimits(max_bytes=8),
        )
    assert byte_limit.value.code == "PARSE_BYTE_LIMIT"

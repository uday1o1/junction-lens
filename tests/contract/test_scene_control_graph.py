"""V1 contract compatibility, validation, and conversion tests."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from google.protobuf import descriptor_pb2
from google.protobuf.internal.encoder import _VarintBytes

from junctionlens.contract import (
    ContractViolation,
    canonical_logical_json,
    canonical_logical_sha256,
    limits,
    parse_binary,
    parse_json,
    to_binary,
    to_json,
)
from junctionlens.contract.golden import make_golden_envelope
from junctionlens.contract.ids import predicted_node_id
from junctionlens.v1 import scene_control_graph_pb2 as scg


def test_committed_golden_binary_and_json_have_same_logical_content() -> None:
    root = Path(__file__).resolve().parents[2]
    binary = parse_binary((root / "tests/fixtures/contract/v1/golden.pb").read_bytes())
    protojson = parse_json((root / "tests/fixtures/contract/v1/golden.json").read_bytes())
    assert canonical_logical_sha256(binary) == canonical_logical_sha256(protojson)
    assert canonical_logical_json(binary) == canonical_logical_json(protojson)


def test_committed_goldens_match_the_reproducible_generator() -> None:
    root = Path(__file__).resolve().parents[2]
    envelope = make_golden_envelope()
    assert (root / "tests/fixtures/contract/v1/golden.pb").read_bytes() == to_binary(envelope)
    assert (root / "tests/fixtures/contract/v1/golden.json").read_text(encoding="utf-8") == to_json(
        envelope
    )


def test_exact_protoc_compiles_the_canonical_schema(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    descriptor_path = tmp_path / "contract.pb"
    result = subprocess.run(
        [
            str(root / ".tools/bin/protoc"),
            f"--proto_path={root / 'proto'}",
            f"--descriptor_set_out={descriptor_path}",
            str(root / "proto/junctionlens/v1/scene_control_graph.proto"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    descriptors = descriptor_pb2.FileDescriptorSet.FromString(descriptor_path.read_bytes())
    assert descriptors.file[0].package == "junctionlens.v1"


def test_binary_and_json_round_trips_preserve_known_logical_content() -> None:
    envelope = make_golden_envelope()
    expected = canonical_logical_json(envelope)
    assert canonical_logical_json(parse_binary(to_binary(envelope))) == expected
    assert canonical_logical_json(parse_json(to_json(envelope))) == expected


def test_protojson_keeps_unsigned_ids_as_decimal_strings() -> None:
    payload = to_json(make_golden_envelope(), indent=None)
    assert '"node_id":"72057594037927937"' in payload
    assert '"track_id":"101"' in payload


def test_unknown_future_minor_binary_field_is_preserved() -> None:
    envelope = make_golden_envelope()
    envelope.schema_minor = 1
    unknown_field = _VarintBytes((99 << 3) | 0) + _VarintBytes(123456)
    payload = envelope.SerializeToString(deterministic=True) + unknown_field
    parsed = parse_binary(payload)
    without_unknown = make_golden_envelope()
    without_unknown.schema_minor = 1
    reserialized = parsed.SerializeToString(deterministic=True)
    assert len(reserialized) > len(without_unknown.SerializeToString(deterministic=True))
    assert parse_binary(reserialized).schema_minor == 1


def test_unknown_json_field_fails_because_protojson_cannot_preserve_it() -> None:
    payload = to_json(make_golden_envelope()).replace("{", '{\n  "future_field": true,', 1)
    with pytest.raises(ContractViolation, match="CONTRACT_JSON_MALFORMED"):
        parse_json(payload)


def test_protojson_rejects_duplicate_keys_and_excessive_depth() -> None:
    with pytest.raises(ContractViolation) as duplicate:
        parse_json('{"schema_major":1,"schema_major":1}')
    assert duplicate.value.reason_code == "CONTRACT_JSON_MALFORMED"

    nested = '{"schema_major":1,"graph":' + "[" * 40 + "0" + "]" * 40 + "}"
    with pytest.raises(ContractViolation) as deep:
        parse_json(nested)
    assert deep.value.reason_code == "CONTRACT_JSON_MALFORMED"


def test_major_mismatch_has_stable_reason_code() -> None:
    envelope = make_golden_envelope()
    envelope.schema_major = 2
    with pytest.raises(ContractViolation) as failure:
        parse_binary(envelope.SerializeToString())
    assert failure.value.reason_code == "CONTRACT_SCHEMA_MAJOR"


def _invalid_node_id(envelope: scg.SceneControlGraphEnvelope) -> None:
    envelope.graph.lanes[0].node_id = predicted_node_id(scg.NODE_TYPE_TRAFFIC_CONTROL, 9)


def _dangling_edge(envelope: scg.SceneControlGraphEnvelope) -> None:
    envelope.graph.edges[0].target_node_id = 999


def _bad_transform(envelope: scg.SceneControlGraphEnvelope) -> None:
    envelope.graph.sensor_frame.t_world_vehicle.values[15] = 0.0


def _bad_box(envelope: scg.SceneControlGraphEnvelope) -> None:
    envelope.graph.traffic_controls[0].normalized_half_open_box.x_max = 1.1


def _nonfinite(envelope: scg.SceneControlGraphEnvelope) -> None:
    envelope.graph.lanes[0].centerline.points[0].x = math.nan


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (_invalid_node_id, "CONTRACT_NODE_ID_TYPE"),
        (_dangling_edge, "CONTRACT_EDGE_DANGLING"),
        (_bad_transform, "CONTRACT_TRANSFORM_AFFINE"),
        (_bad_box, "CONTRACT_NORMALIZED_BOX"),
        (_nonfinite, "CONTRACT_NONFINITE"),
    ],
)
def test_required_invalid_cases_have_stable_reasons(
    mutate: Callable[[scg.SceneControlGraphEnvelope], None], reason: str
) -> None:
    envelope = deepcopy(make_golden_envelope())
    mutate(envelope)
    with pytest.raises(ContractViolation) as failure:
        to_binary(envelope)
    assert failure.value.reason_code == reason


def test_nonfault_copy_control_remains_valid() -> None:
    assert parse_binary(to_binary(deepcopy(make_golden_envelope())))


def test_predicted_node_id_rejects_invalid_type_and_range() -> None:
    with pytest.raises(ValueError, match="concrete"):
        predicted_node_id(scg.NODE_TYPE_UNSPECIFIED, 0)
    with pytest.raises(ValueError, match="56-bit"):
        predicted_node_id(scg.NODE_TYPE_LANE_SEGMENT, (1 << 56) - 1)


def test_oversized_payload_fails_before_protobuf_parsing() -> None:
    with pytest.raises(ContractViolation) as failure:
        parse_binary(bytes(limits.MAX_SERIALIZED_BYTES + 1))
    assert failure.value.reason_code == "CONTRACT_SIZE_LIMIT"


def test_frozen_limit_configuration_matches_runtime_constants() -> None:
    root = Path(__file__).resolve().parents[2]
    configuration = yaml.safe_load((root / "configs/contracts/v1.yaml").read_text(encoding="utf-8"))
    assert configuration == {
        "schema": {"major": limits.SCHEMA_MAJOR, "minor": limits.SCHEMA_MINOR},
        "limits": {
            "serialized_bytes": limits.MAX_SERIALIZED_BYTES,
            "string_bytes": limits.MAX_STRING_BYTES,
            "cameras_per_frame": limits.CAMERAS_PER_FRAME,
            "points_per_polyline": limits.MAX_POINTS_PER_POLYLINE,
            "lanes_per_frame": limits.MAX_LANES_PER_FRAME,
            "traffic_controls_per_frame": limits.MAX_TRAFFIC_CONTROLS_PER_FRAME,
            "road_areas_per_frame": limits.MAX_ROAD_AREAS_PER_FRAME,
            "edges_per_frame": limits.MAX_EDGES_PER_FRAME,
            "tracks_per_frame": limits.MAX_TRACKS_PER_FRAME,
            "artifacts_per_frame": limits.MAX_ARTIFACTS_PER_FRAME,
            "warnings_per_frame": limits.MAX_WARNINGS_PER_FRAME,
        },
    }

"""Cross-language compatibility against the exact native Protobuf runtime."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from google.protobuf.internal.encoder import _VarintBytes

from junctionlens.contract import canonical_logical_sha256, parse_binary, parse_json
from junctionlens.contract.golden import make_golden_envelope
from junctionlens.contract.ids import predicted_node_id
from junctionlens.contract.limits import MAX_SERIALIZED_BYTES
from junctionlens.v1 import scene_control_graph_pb2 as scg

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "build/cpu/junctionlens-contract-probe"


def _run(path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PROBE), "--input", str(path), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cpp_and_python_parse_identical_golden_logical_content() -> None:
    golden = ROOT / "tests/fixtures/contract/v1/golden.pb"
    native = _run(golden, "--emit-json")
    assert native.returncode == 0, native.stderr
    cpp_graph = parse_json(native.stdout)
    python_graph = parse_binary(golden.read_bytes())
    assert canonical_logical_sha256(cpp_graph) == canonical_logical_sha256(python_graph)


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
def test_cpp_seeded_invalid_cases_have_same_stable_reason(
    tmp_path: Path,
    mutate: Callable[[scg.SceneControlGraphEnvelope], None],
    reason: str,
) -> None:
    envelope = make_golden_envelope()
    mutate(envelope)
    path = tmp_path / "invalid.pb"
    path.write_bytes(envelope.SerializeToString(deterministic=True))
    result = _run(path)
    assert result.returncode == 2
    assert json.loads(result.stderr)["reason_code"] == reason


def test_cpp_accepts_future_minor_unknown_binary_field(tmp_path: Path) -> None:
    envelope = make_golden_envelope()
    envelope.schema_minor = 1
    unknown_field = _VarintBytes((99 << 3) | 0) + _VarintBytes(123456)
    path = tmp_path / "future-minor.pb"
    path.write_bytes(envelope.SerializeToString(deterministic=True) + unknown_field)
    result = _run(path)
    assert result.returncode == 0, result.stderr


def test_cpp_rejects_major_mismatch(tmp_path: Path) -> None:
    envelope = make_golden_envelope()
    envelope.schema_major = 2
    path = tmp_path / "future-major.pb"
    path.write_bytes(envelope.SerializeToString(deterministic=True))
    result = _run(path)
    assert result.returncode == 2
    assert json.loads(result.stderr)["reason_code"] == "CONTRACT_SCHEMA_MAJOR"


def test_cpp_rejects_oversized_input_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "oversized.pb"
    with path.open("wb") as output:
        output.seek(MAX_SERIALIZED_BYTES)
        output.write(b"\0")
    result = _run(path)
    assert result.returncode == 2
    assert json.loads(result.stderr)["reason_code"] == "CONTRACT_SIZE_LIMIT"

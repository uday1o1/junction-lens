"""Unit and security checks for the evaluator JSON boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from junctionlens.evaluator.fixtures import FixtureError, _replace, load_cases
from junctionlens.evaluator.payload import EvaluatorPayloadError, load_payload, validate_payload

FIXTURES = Path("tests/fixtures/evaluator")


def _perfect() -> dict[str, object]:
    value = json.loads((FIXTURES / "perfect.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_all_frozen_cases_materialize_as_valid_bounded_inputs() -> None:
    cases = load_cases(FIXTURES)
    assert set(cases) == {
        "corrupt_area_geometry",
        "corrupt_control_topology",
        "corrupt_lane_geometry",
        "corrupt_lane_topology",
        "corrupt_traffic_box",
        "perfect",
    }
    assert cases["perfect"]["expected_changed"] == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("unknown", True), "keys must be exactly"),
        (
            lambda payload: payload["ground_truth"]["segment-001/frame-0001"]["annotation"][
                "lane_segment"
            ][1].__setitem__("id", 100),
            "duplicate IDs",
        ),
        (
            lambda payload: payload["predictions"]["results"]["segment-001/frame-0001"][
                "predictions"
            ].__setitem__("topology_lsls", [[0.0]]),
            "must have 2 rows",
        ),
        (
            lambda payload: payload["predictions"]["results"]["segment-001/frame-0001"][
                "predictions"
            ]["lane_segment"][0].__setitem__("confidence", float("nan")),
            "must be finite",
        ),
    ],
)
def test_payload_rejects_seeded_integrity_faults(mutation: object, message: str) -> None:
    payload = _perfect()
    assert callable(mutation)
    mutation(payload)
    with pytest.raises(EvaluatorPayloadError, match=message):
        validate_payload(payload)


def test_file_loader_rejects_nonstandard_json_nan(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(EvaluatorPayloadError, match="constant NaN"):
        load_payload(path)


def test_fixture_replacement_cannot_create_a_new_path() -> None:
    payload = copy.deepcopy(_perfect())
    with pytest.raises(FixtureError, match="does not exist"):
        _replace(payload, "/predictions/not-a-field", 1)

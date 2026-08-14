"""Security tests for the host-to-container evaluator boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from junctionlens.evaluator.official import (
    EvaluationError,
    evaluate_official,
    validate_evaluator_output,
)


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _matching(source: dict[str, Any]) -> dict[str, object]:
    ground_frames = source["ground_truth"]
    prediction_frames = source["predictions"]["results"]
    frames: dict[str, object] = {}
    for token, ground_frame in ground_frames.items():
        truth = ground_frame["annotation"]
        prediction = prediction_frames[token]["predictions"]
        frame: dict[str, object] = {}
        for object_type, thresholds in {
            "lane_segment": ("1.0", "2.0", "3.0"),
            "traffic_element": ("0.75",),
        }.items():
            ground_items = truth[object_type]
            prediction_items = prediction[object_type]
            confidence = [_float32(float(item["confidence"])) for item in prediction_items]
            artifact = {
                "confidence": confidence,
                "confidence_thresholds": [confidence[0]] * 10 if confidence else [],
                "ground_truth_ids": [item["id"] for item in ground_items],
                "idx_match_gt": list(range(len(prediction_items))),
                "prediction_ids": [item["id"] for item in prediction_items],
            }
            frame[object_type] = dict.fromkeys(thresholds, artifact)
        frames[token] = frame
    return {
        "frames": frames,
        "schema_version": "openlane-v2.v2.1.threshold-matching.v1",
        "thresholds": {"lane_segment": [1.0, 2.0, 3.0], "traffic_element": [0.75]},
    }


def _output(source: bytes, source_payload: dict[str, Any]) -> dict[str, object]:
    return {
        "environment": {
            "numpy": "1.23.5",
            "opencv": "5.0.0",
            "opencv_distribution": "opencv-python-headless==5.0.0.93",
            "openlane_v2": "2.1.0",
            "ortools": "9.3.10497",
            "python": "3.8.20",
            "scipy": "1.8.0",
            "shapely": "2.0.0",
        },
        "input_sha256": hashlib.sha256(source).hexdigest(),
        "matching": _matching(source_payload),
        "metrics": {
            "DET_a": 1.0,
            "DET_l": 1.0,
            "DET_t": 1.0,
            "OLUS": 1.0,
            "TOP_ll": 1.0,
            "TOP_lt": 1.0,
        },
        "schema_version": "junctionlens.official-evaluator-output.v1",
    }


def test_official_wrapper_enforces_restricted_docker_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    (root / "containers").mkdir(parents=True)
    request = root / "request.json"
    source = Path("tests/fixtures/evaluator/perfect.json").read_bytes()
    request.write_bytes(source)
    source_payload = json.loads(source)
    image = {
        "config_sha256": "a" * 64,
        "local_reference": "junctionlens/official-evaluator:test",
        "platform_manifest_sha256": "b" * 64,
        "state": "ACCEPTED_LOCAL",
    }
    (root / "containers/images.lock").write_text(
        yaml.safe_dump({"application_images": {"official_evaluator": image}}),
        encoding="utf-8",
    )
    output = _output(source, source_payload)
    commands: list[list[str]] = []
    monkeypatch.setenv("JUNCTIONLENS_DOCKER_STAGING_ROOT", str(root / "staging"))

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=f"sha256:{'a' * 64}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/docker")
    assert evaluate_official(request, root)["metrics"]["OLUS"] == 1.0
    run_command = commands[1]
    assert run_command[1:4] == ["run", "--rm", "--platform"]
    assert run_command[run_command.index("--platform") + 1] == "linux/amd64"
    assert "none" in run_command
    assert "--read-only" in run_command
    assert run_command[run_command.index("--cap-drop") + 1] == "ALL"
    assert run_command[run_command.index("--security-opt") + 1] == "no-new-privileges"
    assert run_command[run_command.index("--user") + 1] == "65532:65532"
    mount = run_command[run_command.index("--mount") + 1]
    assert mount.endswith(",readonly")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda output: output["matching"].__setitem__(
                "thresholds", {"lane_segment": [2.0], "traffic_element": [0.75]}
            ),
            "thresholds differ",
        ),
        (
            lambda output: output["matching"]["frames"]["segment-001/frame-0001"]["lane_segment"][
                "2.0"
            ].__setitem__("idx_match_gt", [0, 0]),
            "more than once",
        ),
    ],
)
def test_official_wrapper_rejects_seeded_matching_artifact_corruption(
    mutation: Any, message: str
) -> None:
    source = Path("tests/fixtures/evaluator/perfect.json").read_bytes()
    source_payload = json.loads(source)
    output = copy.deepcopy(_output(source, source_payload))
    mutation(output)
    with pytest.raises(EvaluationError, match=message):
        validate_evaluator_output(
            json.dumps(output), hashlib.sha256(source).hexdigest(), source_payload
        )


@pytest.mark.parametrize(
    "raw_output",
    [
        '{"schema_version":"one","schema_version":"two"}',
        '{"schema_version":' + "[" * 40 + "0" + "]" * 40 + "}",
    ],
)
def test_official_wrapper_rejects_adversarial_json_shape(raw_output: str) -> None:
    source = Path("tests/fixtures/evaluator/perfect.json").read_bytes()
    with pytest.raises(EvaluationError, match="invalid JSON"):
        validate_evaluator_output(
            raw_output,
            hashlib.sha256(source).hexdigest(),
            json.loads(source),
        )

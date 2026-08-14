"""Hardened host wrapper for the sole official metric owner container."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_cache_path

from junctionlens.evaluator.payload import EvaluatorPayloadError, parse_payload_bytes


class EvaluationError(RuntimeError):
    """Raised when official evaluation cannot produce trusted output."""


def _docker() -> str:
    executable = shutil.which("docker")
    if executable is None:
        raise EvaluationError("Docker CLI is unavailable")
    return executable


def load_evaluator_image_contract(root: Path) -> Mapping[str, Any]:
    with (root / "containers/images.lock").open(encoding="utf-8") as source:
        payload = yaml.safe_load(source)
    try:
        evaluator = payload["application_images"]["official_evaluator"]
    except (KeyError, TypeError) as error:
        raise EvaluationError("official evaluator image is absent from the image lock") from error
    if not isinstance(evaluator, dict) or evaluator.get("state") != "ACCEPTED_LOCAL":
        raise EvaluationError("official evaluator image has not passed local qualification")
    return evaluator


def inspect_evaluator_image(
    root: Path, reference: str, expected_config: str, expected_manifest: str
) -> None:
    result = subprocess.run(
        [_docker(), "image", "inspect", "--format", "{{.Id}}", reference],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise EvaluationError(f"official evaluator image is unavailable: {result.stderr.strip()}")
    accepted_ids = {f"sha256:{expected_config}", f"sha256:{expected_manifest}"}
    if result.stdout.strip() not in accepted_ids:
        raise EvaluationError("local evaluator image identity differs from the OCI lock")


_MATCHING_THRESHOLDS = {
    "lane_segment": ("1.0", "2.0", "3.0"),
    "traffic_element": ("0.75",),
}


def _float_list(value: object, label: str) -> list[float]:
    if not isinstance(value, list):
        raise EvaluationError(f"{label} must be an array")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise EvaluationError(f"{label} contains a nonnumeric value")
        number = float(item)
        if not math.isfinite(number):
            raise EvaluationError(f"{label} contains a nonfinite value")
        result.append(number)
    return result


def _float32(value: float) -> float:
    return float(struct.unpack("!f", struct.pack("!f", value))[0])


def _validate_threshold_artifact(
    artifact: object,
    ground_items: list[dict[str, Any]],
    prediction_items: list[dict[str, Any]],
    label: str,
) -> None:
    if not isinstance(artifact, dict) or set(artifact) != {
        "confidence",
        "confidence_thresholds",
        "ground_truth_ids",
        "idx_match_gt",
        "prediction_ids",
    }:
        raise EvaluationError(f"{label} has an invalid threshold artifact schema")
    ground_ids = [item["id"] for item in ground_items]
    prediction_ids = [item["id"] for item in prediction_items]
    if artifact["ground_truth_ids"] != ground_ids:
        raise EvaluationError(f"{label} ground-truth IDs differ from the trusted input")
    if artifact["prediction_ids"] != prediction_ids:
        raise EvaluationError(f"{label} prediction IDs differ from the trusted input")
    confidence = _float_list(artifact["confidence"], f"{label} confidence")
    expected_confidence = [_float32(float(item["confidence"])) for item in prediction_items]
    if confidence != expected_confidence:
        raise EvaluationError(f"{label} confidence differs from the trusted input")
    thresholds = _float_list(artifact["confidence_thresholds"], f"{label} confidence thresholds")
    if len(thresholds) != (10 if prediction_items else 0) or any(
        threshold not in confidence for threshold in thresholds
    ):
        raise EvaluationError(f"{label} confidence thresholds are invalid")
    indices = artifact["idx_match_gt"]
    if not isinstance(indices, list) or len(indices) != len(prediction_items):
        raise EvaluationError(f"{label} matched-index array has invalid length")
    matched: list[int] = []
    for value in indices:
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not float(value).is_integer()
            or not 0 <= int(value) < len(ground_items)
        ):
            raise EvaluationError(f"{label} contains an invalid matched index")
        matched.append(int(value))
    if len(matched) != len(set(matched)):
        raise EvaluationError(f"{label} matches one ground-truth object more than once")


def _validate_matching(value: object, payload: Mapping[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {
        "frames",
        "schema_version",
        "thresholds",
    }:
        raise EvaluationError("official evaluator matching artifact has an invalid schema")
    if value["schema_version"] != "openlane-v2.v2.1.threshold-matching.v1":
        raise EvaluationError("official evaluator matching artifact version is unsupported")
    if value["thresholds"] != {
        "lane_segment": [1.0, 2.0, 3.0],
        "traffic_element": [0.75],
    }:
        raise EvaluationError("official evaluator matching thresholds differ from v2.1")
    frames = value["frames"]
    ground_truth = payload["ground_truth"]
    predictions = payload["predictions"]["results"]
    if not isinstance(frames, dict) or set(frames) != set(ground_truth):
        raise EvaluationError("official evaluator matching frame keys differ from the input")
    for token in sorted(ground_truth):
        frame = frames[token]
        if not isinstance(frame, dict) or set(frame) != set(_MATCHING_THRESHOLDS):
            raise EvaluationError(f"official evaluator matching frame {token} is incomplete")
        truth = ground_truth[token]["annotation"]
        prediction = predictions[token]["predictions"]
        for object_type, threshold_names in _MATCHING_THRESHOLDS.items():
            artifacts = frame[object_type]
            if not isinstance(artifacts, dict) or set(artifacts) != set(threshold_names):
                raise EvaluationError(
                    f"official evaluator {object_type} thresholds differ from v2.1"
                )
            for threshold_name in threshold_names:
                _validate_threshold_artifact(
                    artifacts[threshold_name],
                    truth[object_type],
                    prediction[object_type],
                    f"{token} {object_type} {threshold_name}",
                )


def validate_evaluator_output(
    raw_output: str,
    expected_input_sha256: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise EvaluationError("official evaluator returned invalid JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "environment",
        "input_sha256",
        "matching",
        "metrics",
        "schema_version",
    }:
        raise EvaluationError("official evaluator output does not match its frozen schema")
    if value["schema_version"] != "junctionlens.official-evaluator-output.v1":
        raise EvaluationError("official evaluator output schema is unsupported")
    if value["input_sha256"] != expected_input_sha256:
        raise EvaluationError("official evaluator did not echo the mounted input hash")
    environment = value["environment"]
    if not isinstance(environment, dict) or environment != {
        "numpy": "1.23.5",
        "opencv": "5.0.0",
        "opencv_distribution": "opencv-python-headless==5.0.0.93",
        "openlane_v2": "2.1.0",
        "ortools": "9.3.10497",
        "python": "3.8.20",
        "scipy": "1.8.0",
        "shapely": "2.0.0",
    }:
        raise EvaluationError("official evaluator environment differs from the lock")
    metrics = value["metrics"]
    expected_metrics = {"DET_a", "DET_l", "DET_t", "OLUS", "TOP_ll", "TOP_lt"}
    if not isinstance(metrics, dict) or set(metrics) != expected_metrics:
        raise EvaluationError("official evaluator returned an incomplete metric set")
    for name, raw_metric in metrics.items():
        if raw_metric is not None and (
            isinstance(raw_metric, bool)
            or not isinstance(raw_metric, int | float)
            or not math.isfinite(float(raw_metric))
            or not 0.0 <= float(raw_metric) <= 1.0
        ):
            raise EvaluationError(f"official evaluator metric {name} is invalid")
    _validate_matching(value["matching"], payload)
    return value


def evaluator_container_command(
    docker: str,
    reference: str,
    mounted_input: Path,
    arguments: Sequence[str] | None = None,
) -> list[str]:
    """Build the one restricted invocation shared by production and parity checks."""
    return [
        docker,
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",  # noqa: S108 - isolated container tmpfs
        "--user",
        "65532:65532",
        "--mount",
        f"type=bind,src={mounted_input},dst=/input/request.json,readonly",
        reference,
        *(arguments if arguments is not None else ("/input/request.json",)),
    ]


def evaluate_official(input_path: Path, root: Path) -> dict[str, Any]:
    """Run official metrics in a digest-verified, networkless, read-only container."""
    root = root.resolve(strict=True)
    input_path = input_path.resolve(strict=True)
    raw_bytes = input_path.read_bytes()
    try:
        payload = parse_payload_bytes(raw_bytes)
    except (EvaluatorPayloadError, OSError) as error:
        raise EvaluationError(str(error)) from error
    contract = load_evaluator_image_contract(root)
    reference = str(contract["local_reference"])
    inspect_evaluator_image(
        root,
        reference,
        str(contract["config_sha256"]),
        str(contract["platform_manifest_sha256"]),
    )
    input_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    staging_override = os.environ.get("JUNCTIONLENS_DOCKER_STAGING_ROOT")
    cache_root = (
        Path(staging_override).expanduser()
        if staging_override is not None
        else user_cache_path("junctionlens") / "evaluator-inputs"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise EvaluationError("Docker staging root must be a real directory")
    with tempfile.TemporaryDirectory(prefix="request-", dir=cache_root) as temp:
        mounted_input = Path(temp) / "request.json"
        mounted_input.write_bytes(raw_bytes)
        command = evaluator_container_command(_docker(), reference, mounted_input)
        result = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        detail = result.stderr[:2048].strip()
        raise EvaluationError(
            f"official evaluator failed with exit code {result.returncode}: {detail}"
        )
    return validate_evaluator_output(result.stdout, input_sha256, payload)

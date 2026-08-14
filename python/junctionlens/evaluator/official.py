"""Hardened host wrapper for the sole official metric owner container."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
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


def _load_image_contract(root: Path) -> Mapping[str, Any]:
    with (root / "containers/images.lock").open(encoding="utf-8") as source:
        payload = yaml.safe_load(source)
    try:
        evaluator = payload["application_images"]["official_evaluator"]
    except (KeyError, TypeError) as error:
        raise EvaluationError("official evaluator image is absent from the image lock") from error
    if not isinstance(evaluator, dict) or evaluator.get("state") != "ACCEPTED_LOCAL":
        raise EvaluationError("official evaluator image has not passed local qualification")
    return evaluator


def _inspect_image(
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


def _validate_output(raw_output: str, expected_input_sha256: str) -> dict[str, Any]:
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
        if raw_metric is not None and not isinstance(raw_metric, int | float):
            raise EvaluationError(f"official evaluator metric {name} is invalid")
    if not isinstance(value["matching"], dict):
        raise EvaluationError("official evaluator matching artifact is invalid")
    return value


def evaluate_official(input_path: Path, root: Path) -> dict[str, Any]:
    """Run official metrics in a digest-verified, networkless, read-only container."""
    root = root.resolve(strict=True)
    input_path = input_path.resolve(strict=True)
    raw_bytes = input_path.read_bytes()
    try:
        parse_payload_bytes(raw_bytes)
    except (EvaluatorPayloadError, OSError) as error:
        raise EvaluationError(str(error)) from error
    contract = _load_image_contract(root)
    reference = str(contract["local_reference"])
    _inspect_image(
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
        command = [
            _docker(),
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
            "/input/request.json",
        ]
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
    return _validate_output(result.stdout, input_sha256)

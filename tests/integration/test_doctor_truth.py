"""Integration tests that compare doctor output with direct observations."""

from __future__ import annotations

import json
import platform
import subprocess
import sys

import onnxruntime


def test_doctor_reports_real_python_and_onnxruntime_state() -> None:
    """Configured pins cannot be substituted for observed runtime evidence."""
    result = subprocess.run(
        [sys.executable, "-m", "junctionlens.cli.main", "doctor", "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["host"]["python_version"] == platform.python_version()
    ort = next(item for item in payload["capabilities"] if item["capability"] == "onnxruntime")
    assert ort["observed_version"] == onnxruntime.__version__
    assert ort["details"]["providers"] == onnxruntime.get_available_providers()


def test_absent_acceleration_and_dataset_are_not_reported_as_ready() -> None:
    """The portability profile keeps unavailable target prerequisites distinct."""
    result = subprocess.run(
        [sys.executable, "-m", "junctionlens.cli.main", "doctor", "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["readiness"]["accelerated_target"] is False
    assert payload["readiness"]["licensed_data"] is False
    reason_codes = {item["reason_code"] for item in payload["capabilities"]}
    assert "NVIDIA_SMI_NOT_FOUND" in reason_codes
    assert "CUDA_TOOLKIT_NOT_FOUND" in reason_codes
    assert "TENSORRT_NOT_FOUND" in reason_codes
    assert "DATASET_ROOT_UNSET" in reason_codes

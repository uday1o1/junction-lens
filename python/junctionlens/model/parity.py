"""Cross-runtime raw-output parity for the M0 ONNX graph."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
import torch

from junctionlens.model.contract import output_contract
from junctionlens.model.profile import M0ModelProfile
from junctionlens.model.spike import INPUT_NAMES, OUTPUT_NAMES, M0GraphModel
from junctionlens.model.synthetic import make_micro_inputs


class ParityError(RuntimeError):
    """Raised when a runtime fails or its raw tensors differ."""


def _load_model(profile: M0ModelProfile, checkpoint_path: Path) -> M0GraphModel:
    checkpoint = cast(
        dict[str, Any],
        torch.load(checkpoint_path, map_location="cpu", weights_only=True),
    )
    if checkpoint.get("profile_sha256") != profile.canonical_sha256():
        raise ParityError("checkpoint profile hash differs from the parity profile")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ParityError("checkpoint does not contain model_state_dict")
    model = M0GraphModel(profile)
    model.load_state_dict(cast(dict[str, Any], state), strict=True)
    model.eval()
    return model


def _native_outputs(
    native_runner: Path,
    model_path: Path,
    profile: M0ModelProfile,
    frame_index: int,
) -> tuple[dict[str, np.ndarray[Any, np.dtype[np.float32]]], dict[str, Any]]:
    completed = subprocess.run(
        [
            str(native_runner.resolve()),
            "--model",
            str(model_path.resolve()),
            "--expected-profile-sha256",
            profile.canonical_sha256(),
            "--frame-index",
            str(frame_index),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise ParityError(f"native ONNX probe failed: {completed.stderr.strip()}")
    try:
        payload = cast(dict[str, Any], json.loads(completed.stdout))
    except (json.JSONDecodeError, TypeError) as error:
        raise ParityError("native ONNX probe did not return valid JSON") from error
    if payload.get("status") != "PASSED":
        raise ParityError("native ONNX probe did not report PASSED")
    output_values = payload.get("outputs")
    if not isinstance(output_values, list) or len(output_values) != len(OUTPUT_NAMES):
        raise ParityError("native ONNX probe output contract is incomplete")
    outputs: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    for item, contract in zip(output_values, output_contract(profile), strict=True):
        if not isinstance(item, dict) or item.get("name") != contract.name:
            raise ParityError("native ONNX output name differs from the frozen contract")
        shape_value = item.get("shape")
        values = item.get("values")
        if not isinstance(shape_value, list) or not isinstance(values, list):
            raise ParityError(f"native ONNX output {contract.name} is malformed")
        shape = tuple(int(value) for value in shape_value)
        array = np.asarray(values, dtype=np.float32)
        try:
            outputs[contract.name] = array.reshape(shape)
        except ValueError as error:
            raise ParityError(f"native ONNX output {contract.name} has a size mismatch") from error
    return outputs, {
        "runtime_version": payload.get("runtime_version"),
        "requested_provider": payload.get("requested_provider"),
        "available_providers": payload.get("available_providers"),
    }


def _error_metrics(
    reference: np.ndarray[Any, Any], actual: np.ndarray[Any, Any]
) -> dict[str, float]:
    if reference.shape != actual.shape:
        raise ParityError(f"shape mismatch: {reference.shape!r} != {actual.shape!r}")
    difference = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    reference_magnitude = np.abs(reference.astype(np.float64))
    relative_mask = reference_magnitude > 1e-3
    maximum_relative = (
        float((difference[relative_mask] / reference_magnitude[relative_mask]).max())
        if relative_mask.any()
        else 0.0
    )
    return {
        "maximum_absolute_error": float(difference.max(initial=0.0)),
        "maximum_relative_error_above_1e-3": maximum_relative,
    }


def run_parity(
    profile: M0ModelProfile,
    checkpoint_path: Path,
    model_path: Path,
    native_runner: Path,
    output_path: Path,
    *,
    frame_index: int = 7,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compare PyTorch, Python ORT CPU, and native C++ ORT CPU tensors."""
    if not 0 <= frame_index < profile.micro_overfit.frames:
        raise ParityError("frame index is outside the frozen micro-overfit set")
    torch.set_num_threads(1)
    model = _load_model(profile, checkpoint_path)
    inputs = make_micro_inputs(
        profile,
        torch.tensor([frame_index], dtype=torch.int64),
        spatial_size=(profile.input.height, profile.input.width),
    )
    with torch.inference_mode():
        torch_values = model(*inputs)
    pytorch_outputs = {
        name: value.detach().cpu().numpy()
        for name, value in zip(OUTPUT_NAMES, torch_values, strict=True)
    }

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feed = {
        name: value.detach().cpu().numpy() for name, value in zip(INPUT_NAMES, inputs, strict=True)
    }
    ort_values = session.run(list(OUTPUT_NAMES), feed)
    python_ort_outputs = dict(zip(OUTPUT_NAMES, ort_values, strict=True))
    native_outputs, native_environment = _native_outputs(
        native_runner, model_path, profile, frame_index
    )

    batch_two_inputs = make_micro_inputs(
        profile,
        torch.tensor([0, 1], dtype=torch.int64),
        spatial_size=(profile.input.height, profile.input.width),
    )
    batch_two_values = session.run(
        list(OUTPUT_NAMES),
        {
            name: value.detach().cpu().numpy()
            for name, value in zip(INPUT_NAMES, batch_two_inputs, strict=True)
        },
    )
    if any(value.shape[0] != 2 for value in batch_two_values):
        raise ParityError("exported graph does not preserve its dynamic batch contract")

    tensor_reports: list[dict[str, Any]] = []
    passed = True
    for name in OUTPUT_NAMES:
        pytorch_to_python = _error_metrics(pytorch_outputs[name], python_ort_outputs[name])
        pytorch_to_native = _error_metrics(pytorch_outputs[name], native_outputs[name])
        tensor_passed = bool(
            pytorch_to_python["maximum_absolute_error"] <= absolute_tolerance
            and pytorch_to_python["maximum_relative_error_above_1e-3"] <= relative_tolerance
            and pytorch_to_native["maximum_absolute_error"] <= absolute_tolerance
            and pytorch_to_native["maximum_relative_error_above_1e-3"] <= relative_tolerance
        )
        passed = passed and tensor_passed
        tensor_reports.append(
            {
                "name": name,
                "shape": list(pytorch_outputs[name].shape),
                "pytorch_to_python_ort": pytorch_to_python,
                "pytorch_to_native_cpp_ort": pytorch_to_native,
                "status": "PASSED" if tensor_passed else "FAILED",
            }
        )
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PASSED" if passed else "FAILED",
        "profile_id": profile.profile_id,
        "profile_sha256": profile.canonical_sha256(),
        "frame_index": frame_index,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "python_onnxruntime_version": ort.__version__,
        "python_requested_provider": "CPUExecutionProvider",
        "native": native_environment,
        "dynamic_batch_two": "PASSED",
        "tensors": tensor_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise ParityError("one or more raw output tensors exceeded the frozen parity tolerances")
    return report

#!/usr/bin/env python3
"""Qualify CUDA raw-output and public C++ graph parity on one frozen corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message
from PIL import Image

from junctionlens.contract.validation import validate_envelope
from junctionlens.model.profile import load_m0_profile
from junctionlens.model.spike import INPUT_NAMES, OUTPUT_NAMES
from junctionlens.model.synthetic import make_micro_inputs
from junctionlens.runtime import run_batch
from junctionlens.v1 import scene_control_graph_pb2 as scg

IDENTITY3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
IDENTITY4 = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


class RuntimeQualificationError(RuntimeError):
    """Raised when accelerated correctness evidence fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_envelope(
    image_path: Path, timestamp: int, pose_x: float
) -> scg.SceneControlGraphEnvelope:
    envelope = scg.SceneControlGraphEnvelope(schema_major=1, schema_minor=0)
    envelope.producer.git_commit = "1" * 40
    envelope.producer.configuration_sha256 = "2" * 64
    envelope.producer.runtime_build_sha256 = "3" * 64
    envelope.producer.execution_provider_profile = "qualification-fixture"
    envelope.producer.provider_assignment_digest = "4" * 64
    graph = envelope.graph
    graph.role = scg.GRAPH_ROLE_GROUND_TRUTH
    graph.frame_key.dataset_id = "runtime-qualification"
    graph.frame_key.dataset_version = "1"
    graph.frame_key.split_id = "synthetic"
    graph.frame_key.segment_id = "fixed-two-frame"
    graph.frame_key.timestamp_ns = timestamp
    graph.frame_key.source_domain = scg.SOURCE_DOMAIN_SYNTHETIC
    graph.frame_key.calibration_sha256 = "5" * 64
    graph.frame_key.frame_manifest_sha256 = hashlib.sha256(str(timestamp).encode()).hexdigest()
    sensor = graph.sensor_frame
    sensor.frame_key.CopyFrom(graph.frame_key)
    sensor.t_world_vehicle.values.extend(IDENTITY4)
    sensor.t_world_vehicle.values[3] = pose_x
    sensor.pose_valid = True
    sensor.adapter_version = "runtime-qualification-v1"
    for slot in range(scg.CAMERA_SLOT_FRONT_CENTER, scg.CAMERA_SLOT_REAR_RIGHT + 1):
        camera = sensor.cameras.add(slot=slot, valid=slot == scg.CAMERA_SLOT_FRONT_CENTER)
        camera.capture_timestamp_ns = timestamp
        camera.original_width = 100 if camera.valid else 0
        camera.original_height = 80 if camera.valid else 0
        camera.intrinsic.values.extend((100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0))
        camera.t_vehicle_camera.values.extend(IDENTITY4)
        camera.distortion_model = scg.DISTORTION_MODEL_NONE
        camera.image_transform.original_to_model.values.extend(
            (4.8, 0.0, 80.0, 0.0, 4.8, 0.0, 0.0, 0.0, 1.0) if camera.valid else IDENTITY3
        )
        camera.image_transform.resized_width = 480 if camera.valid else 0
        camera.image_transform.resized_height = 384 if camera.valid else 0
        camera.image_transform.pad_left = 80 if camera.valid else 0
        if camera.valid:
            camera.original_image.kind = scg.ARTIFACT_KIND_SOURCE_IMAGE
            camera.original_image.sha256 = _sha256(image_path)
            camera.original_image.byte_size = image_path.stat().st_size
            camera.original_image.media_type = "image/png"
            camera.original_image.relative_uri = "images/frame.png"
            camera.original_image.license_id = "CC0-1.0"
    validate_envelope(envelope)
    return envelope


def write_synthetic_runtime_fixture(root: Path) -> Path:
    """Write the repository-owned two-frame runtime qualification fixture."""
    image = root / "images" / "frame.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(20, 30, 40)).save(image)
    inputs: list[Path] = []
    for index, (timestamp, pose_x) in enumerate(((100, 0.0), (200, 1.0))):
        path = root / f"input-{index}.pb"
        path.write_bytes(
            _input_envelope(image, timestamp, pose_x).SerializeToString(deterministic=True)
        )
        inputs.append(path)
    input_list = root / "inputs.txt"
    input_list.write_text("\n".join(path.name for path in inputs) + "\n", encoding="utf-8")
    return input_list


def _compare_messages(
    first: Message,
    second: Message,
    *,
    path: str,
    maximum_error: list[float],
    tolerance: float,
) -> None:
    first_fields = {field.number: (field, value) for field, value in first.ListFields()}
    second_fields = {field.number: (field, value) for field, value in second.ListFields()}
    if first_fields.keys() != second_fields.keys():
        raise RuntimeQualificationError(f"protobuf field presence differs at {path}")
    for number, (field, first_value) in first_fields.items():
        second_value = second_fields[number][1]
        first_items = list(first_value) if field.is_repeated else [first_value]
        second_items = list(second_value) if field.is_repeated else [second_value]
        if len(first_items) != len(second_items):
            raise RuntimeQualificationError(
                f"protobuf repeated length differs at {path}.{field.name}"
            )
        for index, (first_item, second_item) in enumerate(
            zip(first_items, second_items, strict=True)
        ):
            item_path = f"{path}.{field.name}[{index}]"
            if field.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
                _compare_messages(
                    first_item,
                    second_item,
                    path=item_path,
                    maximum_error=maximum_error,
                    tolerance=tolerance,
                )
            elif field.cpp_type in {
                FieldDescriptor.CPPTYPE_DOUBLE,
                FieldDescriptor.CPPTYPE_FLOAT,
            }:
                difference = abs(float(first_item) - float(second_item))
                maximum_error[0] = max(maximum_error[0], difference)
                if difference > tolerance:
                    raise RuntimeQualificationError(
                        f"protobuf continuous value differs at {item_path}: {difference}"
                    )
            elif first_item != second_item:
                raise RuntimeQualificationError(f"protobuf discrete value differs at {item_path}")


def _compare_graph_outputs(cpu_root: Path, cuda_root: Path) -> float:
    cpu_paths = sorted(cpu_root.glob("*.prediction.pb"))
    cuda_paths = sorted(cuda_root.glob("*.prediction.pb"))
    if [path.name for path in cpu_paths] != [path.name for path in cuda_paths] or not cpu_paths:
        raise RuntimeQualificationError("CPU and CUDA output sets differ")
    maximum_error = [0.0]
    for cpu_path, cuda_path in zip(cpu_paths, cuda_paths, strict=True):
        cpu = scg.SceneControlGraphEnvelope.FromString(cpu_path.read_bytes())
        cuda = scg.SceneControlGraphEnvelope.FromString(cuda_path.read_bytes())
        cpu.ClearField("producer")
        cuda.ClearField("producer")
        _compare_messages(
            cpu,
            cuda,
            path=cpu_path.name,
            maximum_error=maximum_error,
            tolerance=5.0e-3,
        )
    return maximum_error[0]


def _raw_parity(model: Path, profile_path: Path) -> tuple[float, float]:
    profile = load_m0_profile(profile_path)
    tensors = make_micro_inputs(
        profile,
        torch.tensor([7], dtype=torch.int64),
        spatial_size=(profile.input.height, profile.input.width),
    )
    inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(INPUT_NAMES, tensors, strict=True)
    }
    cpu = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    cuda = ort.InferenceSession(
        str(model),
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": True,
                },
            ),
            "CPUExecutionProvider",
        ],
    )
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for cpu_output, cuda_output in zip(
        cpu.run(list(OUTPUT_NAMES), inputs),
        cuda.run(list(OUTPUT_NAMES), inputs),
        strict=True,
    ):
        absolute = np.abs(cpu_output.astype(np.float64) - cuda_output.astype(np.float64))
        maximum_absolute = max(maximum_absolute, float(np.max(absolute, initial=0.0)))
        denominator = np.abs(cpu_output.astype(np.float64))
        mask = denominator > 1.0e-3
        if np.any(mask):
            maximum_relative = max(
                maximum_relative, float(np.max(absolute[mask] / denominator[mask], initial=0.0))
            )
    if maximum_absolute > 5.0e-3:
        raise RuntimeQualificationError("CUDA raw-output absolute error exceeds 5e-3")
    return maximum_absolute, maximum_relative


def qualify(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run raw and public-path CUDA parity and return structured evidence."""
    work = arguments.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    fixture = work / "fixture"
    fixture.mkdir()
    input_list = write_synthetic_runtime_fixture(fixture)
    common = {
        "model_path": arguments.model.resolve(),
        "profile_path": arguments.profile.resolve(),
        "input_list_path": input_list,
        "asset_root": fixture,
        "project_root": arguments.project_root.resolve(),
        "repeat_loads": 1,
        "buffer_slots": 2,
        "source_commit": arguments.source_commit,
        "source_dirty": False,
    }
    cpu_receipt = run_batch(
        **common,
        output_directory=work / "cpu-output",
        runtime_binary=arguments.cpu_runtime.resolve(),
    )
    cuda_receipt = run_batch(
        **common,
        output_directory=work / "cuda-output",
        runtime_binary=arguments.gpu_runtime.resolve(),
        provider_profile="cuda",
        provider_log_output=work / "cuda-provider.raw.log",
        device_id=0,
    )
    graph_maximum_absolute = _compare_graph_outputs(work / "cpu-output", work / "cuda-output")
    raw_maximum_absolute, raw_maximum_relative = _raw_parity(
        arguments.model.resolve(), arguments.profile.resolve()
    )
    return {
        "schema_version": "junctionlens.cuda-parity.v1",
        "status": "PASSED",
        "source_commit": arguments.source_commit,
        "model_sha256": _sha256(arguments.model),
        "raw_maximum_absolute_error": raw_maximum_absolute,
        "raw_maximum_relative_error_above_1e-3": raw_maximum_relative,
        "graph_maximum_absolute_error": graph_maximum_absolute,
        "discrete_graph_parity": True,
        "cpu_receipt": cpu_receipt,
        "cuda_receipt": cuda_receipt,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--cpu-runtime", type=Path, required=True)
    parser.add_argument("--gpu-runtime", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the command line qualification workflow."""
    arguments = _parse_args()
    try:
        report = qualify(arguments)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"CUDA parity error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

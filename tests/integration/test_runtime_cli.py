from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message
from PIL import Image
from typer.testing import CliRunner

from junctionlens.cli.main import app
from junctionlens.contract.validation import validate_envelope
from junctionlens.model.spike import INPUT_NAMES, OUTPUT_NAMES
from junctionlens.runtime.reference import reference_postprocess
from junctionlens.v1 import scene_control_graph_pb2 as scg

_IDENTITY3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_IDENTITY4 = (
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_envelope(
    image_path: Path, timestamp: int, pose_x: float
) -> scg.SceneControlGraphEnvelope:
    envelope = scg.SceneControlGraphEnvelope(schema_major=1, schema_minor=0)
    envelope.producer.git_commit = "1" * 40
    envelope.producer.configuration_sha256 = "2" * 64
    envelope.producer.runtime_build_sha256 = "3" * 64
    envelope.producer.execution_provider_profile = "fixture"
    envelope.producer.provider_assignment_digest = "4" * 64
    graph = envelope.graph
    graph.role = scg.GRAPH_ROLE_GROUND_TRUTH
    graph.frame_key.dataset_id = "runtime-fixture"
    graph.frame_key.dataset_version = "1"
    graph.frame_key.split_id = "test"
    graph.frame_key.segment_id = "segment"
    graph.frame_key.timestamp_ns = timestamp
    graph.frame_key.source_domain = scg.SOURCE_DOMAIN_SYNTHETIC
    graph.frame_key.calibration_sha256 = "5" * 64
    graph.frame_key.frame_manifest_sha256 = hashlib.sha256(str(timestamp).encode()).hexdigest()
    sensor = graph.sensor_frame
    sensor.frame_key.CopyFrom(graph.frame_key)
    sensor.t_world_vehicle.values.extend(_IDENTITY4)
    sensor.t_world_vehicle.values[3] = pose_x
    sensor.pose_valid = True
    sensor.adapter_version = "runtime-fixture-v1"
    for slot in range(scg.CAMERA_SLOT_FRONT_CENTER, scg.CAMERA_SLOT_REAR_RIGHT + 1):
        camera = sensor.cameras.add(slot=slot, valid=slot == scg.CAMERA_SLOT_FRONT_CENTER)
        camera.capture_timestamp_ns = timestamp
        camera.original_width = 100 if camera.valid else 0
        camera.original_height = 80 if camera.valid else 0
        camera.intrinsic.values.extend((100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0))
        camera.t_vehicle_camera.values.extend(_IDENTITY4)
        camera.distortion_model = scg.DISTORTION_MODEL_NONE
        camera.image_transform.original_to_model.values.extend(
            (4.8, 0.0, 80.0, 0.0, 4.8, 0.0, 0.0, 0.0, 1.0) if camera.valid else _IDENTITY3
        )
        camera.image_transform.resized_width = 480 if camera.valid else 0
        camera.image_transform.resized_height = 384 if camera.valid else 0
        camera.image_transform.pad_left = 80 if camera.valid else 0
        if camera.valid:
            camera.original_image.kind = scg.ARTIFACT_KIND_SOURCE_IMAGE
            camera.original_image.sha256 = _sha256(image_path)
            camera.original_image.byte_size = image_path.stat().st_size
            camera.original_image.media_type = "image/png"
            camera.original_image.relative_uri = image_path.relative_to(
                image_path.parents[1]
            ).as_posix()
            camera.original_image.license_id = "CC0-1.0"
    validate_envelope(envelope)
    return envelope


def _write_batch(root: Path) -> tuple[Path, list[scg.SceneControlGraphEnvelope]]:
    image_path = root / "images/frame.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), color=(20, 30, 40)).save(image_path)
    envelopes = [_input_envelope(image_path, 100, 0.0), _input_envelope(image_path, 200, 1.0)]
    input_paths: list[Path] = []
    for index, envelope in enumerate(envelopes):
        path = root / f"input-{index}.pb"
        path.write_bytes(envelope.SerializeToString(deterministic=True))
        input_paths.append(path)
    input_list = root / "inputs.txt"
    input_list.write_text("\n".join(path.name for path in input_paths) + "\n", encoding="utf-8")
    return input_list, envelopes


def _python_inputs(
    envelopes: list[scg.SceneControlGraphEnvelope], frame_index: int
) -> tuple[dict[str, np.ndarray[Any, Any]], scg.SensorFrame]:
    image = np.asarray(Image.new("RGB", (480, 384), color=(20, 30, 40)), dtype=np.float32)
    canvas = np.zeros((384, 640, 3), dtype=np.float32)
    canvas[:, 80:560] = image / np.float32(255.0)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (canvas - mean) / std
    one = np.zeros((8, 3, 384, 640), dtype=np.float32)
    one[0] = np.transpose(normalized, (2, 0, 1))
    images = np.stack((one, one))[None]
    camera_valid = np.zeros((1, 2, 8), dtype=np.bool_)
    camera_valid[:, :, 0] = True
    intrinsics = np.zeros((1, 2, 8, 3, 3), dtype=np.float32)
    intrinsics[:, :, 0] = np.asarray(
        ((480.0, 0.0, 320.0), (0.0, 480.0, 192.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    transforms = np.zeros((1, 2, 8, 4, 4), dtype=np.float32)
    transforms[:, :, 0] = np.eye(4, dtype=np.float32)
    ego = np.eye(4, dtype=np.float32)[None]
    temporal_valid = np.asarray([frame_index > 0], dtype=np.bool_)
    if frame_index > 0:
        ego[0, 0, 3] = -1.0
    return (
        dict(
            zip(
                INPUT_NAMES,
                (images, camera_valid, intrinsics, transforms, ego, temporal_valid),
                strict=True,
            )
        ),
        envelopes[frame_index].graph.sensor_frame,
    )


def _invoke(
    model_path: Path, input_list: Path, root: Path, output: Path, *, repeat_loads: int
) -> object:
    return CliRunner().invoke(
        app,
        [
            "infer",
            "--model",
            str(model_path),
            "--input-list",
            str(input_list),
            "--asset-root",
            str(root),
            "--output-dir",
            str(output),
            "--runtime-binary",
            "build/cpu/bin/junctionlens-runtime",
            "--repeat-loads",
            str(repeat_loads),
        ],
    )


def _assert_logical_parity(actual: Message, expected: Message, path: str = "graph") -> None:
    actual_fields = {field.number: (field, value) for field, value in actual.ListFields()}
    expected_fields = {field.number: (field, value) for field, value in expected.ListFields()}
    assert actual_fields.keys() == expected_fields.keys(), path
    for number, (field, actual_value) in actual_fields.items():
        expected_value = expected_fields[number][1]
        field_path = f"{path}.{field.name}"
        actual_items = list(actual_value) if field.is_repeated else [actual_value]
        expected_items = list(expected_value) if field.is_repeated else [expected_value]
        assert len(actual_items) == len(expected_items), field_path
        for index, (actual_item, expected_item) in enumerate(
            zip(actual_items, expected_items, strict=True)
        ):
            item_path = f"{field_path}[{index}]" if field.is_repeated else field_path
            if field.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
                _assert_logical_parity(actual_item, expected_item, item_path)
            elif field.cpp_type in {
                FieldDescriptor.CPPTYPE_DOUBLE,
                FieldDescriptor.CPPTYPE_FLOAT,
            }:
                assert abs(float(actual_item) - float(expected_item)) <= 1e-4, item_path
            else:
                assert actual_item == expected_item, item_path


def test_public_cli_matches_independent_postprocessor_and_releases_buffers(
    exported_m0_model: tuple[Path, Path], tmp_path: Path
) -> None:
    _, model_path = exported_m0_model
    input_list, envelopes = _write_batch(tmp_path)
    output = tmp_path / "predictions"
    result = _invoke(model_path, input_list, tmp_path, output, repeat_loads=3)
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert {
        key: receipt[key]
        for key in (
            "all_slots_free",
            "buffer_capacity",
            "buffer_high_water_mark",
            "io_binding_enabled",
            "processed_frames",
            "provider",
            "repeat_loads",
            "schema_version",
            "status",
        )
    } == {
        "all_slots_free": True,
        "buffer_capacity": 2,
        "buffer_high_water_mark": 1,
        "io_binding_enabled": False,
        "processed_frames": 2,
        "provider": "CPUExecutionProvider",
        "repeat_loads": 3,
        "schema_version": "junctionlens.runtime-batch.v1",
        "status": "PASSED",
    }
    for key in ("provider_assignment_sha256", "provider_log_sha256"):
        assert len(receipt[key]) == 64
        int(receipt[key], 16)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual_paths = sorted(output.glob("*.prediction.pb"))
    assert len(actual_paths) == 2
    for frame_index, actual_path in enumerate(actual_paths):
        actual = scg.SceneControlGraphEnvelope.FromString(actual_path.read_bytes())
        validate_envelope(actual)
        feed, sensor = _python_inputs(envelopes, frame_index)
        values = session.run(list(OUTPUT_NAMES), feed)
        expected = reference_postprocess(
            dict(zip(OUTPUT_NAMES, values, strict=True)), sensor, actual.producer
        )
        _assert_logical_parity(actual.graph, expected.graph)


def test_malformed_protobuf_fails_without_partial_output(
    exported_m0_model: tuple[Path, Path], tmp_path: Path
) -> None:
    _, model_path = exported_m0_model
    malformed = tmp_path / "malformed.pb"
    malformed.write_bytes(b"not-a-protobuf")
    input_list = tmp_path / "inputs.txt"
    input_list.write_text(f"{malformed.name}\n", encoding="utf-8")
    output = tmp_path / "predictions"
    result = _invoke(model_path, input_list, tmp_path, output, repeat_loads=1)
    assert result.exit_code == 2
    assert "RUNTIME_INPUT_CONTRACT" in result.output
    assert not tuple(output.glob("*"))


def test_wrong_image_digest_fails_closed(
    exported_m0_model: tuple[Path, Path], tmp_path: Path
) -> None:
    _, model_path = exported_m0_model
    input_list, _ = _write_batch(tmp_path)
    image_path = tmp_path / "images/frame.png"
    image_path.write_bytes(image_path.read_bytes() + b"corruption")
    output = tmp_path / "predictions"
    result = _invoke(model_path, input_list, tmp_path, output, repeat_loads=1)
    assert result.exit_code == 2
    assert "RUNTIME_IMAGE_SIZE" in result.output
    assert not tuple(output.glob("*"))


def test_public_cli_rejects_existing_output_instead_of_overwriting(
    exported_m0_model: tuple[Path, Path], tmp_path: Path
) -> None:
    _, model_path = exported_m0_model
    input_list, _ = _write_batch(tmp_path)
    output = tmp_path / "predictions"
    first = _invoke(model_path, input_list, tmp_path, output, repeat_loads=1)
    assert first.exit_code == 0, first.output
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    second = _invoke(model_path, input_list, tmp_path, output, repeat_loads=1)
    assert second.exit_code == 2
    assert "RUNTIME_OUTPUT_EXISTS" in second.output
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before

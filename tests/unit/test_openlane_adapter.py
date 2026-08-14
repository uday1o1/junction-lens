"""OpenLane-V2 sample adapter and audit tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from junctionlens.data.audit import audit_report
from junctionlens.data.contracts import CAMERA_SLOTS, CameraSlot
from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
from junctionlens.data.parity import adapted_source_projection

_CONFIG = Path("configs/data/openlane-v2-v2.1.adapter.yaml")


def test_adapter_normalizes_camera_order_coordinates_and_source_box(openlane_root: Path) -> None:
    """The real raw-JSON path preserves source data while freezing canonical tensors."""
    frames = tuple(OpenLaneAdapter(openlane_root, _CONFIG).iter_frames("sample"))
    assert [frame.key.timestamp_ns for frame in frames] == [100, 200]
    frame = frames[0]
    assert frame.source_metadata.source_name == "argoverse2"
    assert frame.source_metadata.source_segment_id == "source-segment"
    assert frame.source_metadata.schema_mode == "lane-segment-v2.1"
    assert tuple(camera.slot for camera in frame.cameras) == CAMERA_SLOTS
    assert [camera.slot for camera in frame.cameras if camera.valid] == [CameraSlot.FRONT_CENTER]
    assert frame.lanes[0].centerline == ((1.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    control = frame.traffic_controls[0]
    assert control.source_camera == CameraSlot.FRONT_CENTER
    assert control.source_pixel_box.points == ((10.0, 20.0), (30.0, 40.0))
    assert control.normalized_half_open_box == (0.1, 0.25, 0.3, 0.5)
    np.testing.assert_allclose(frame.t_world_vehicle, np.eye(4), atol=1e-12)


def test_metadata_iteration_is_lazy_and_model_inputs_have_explicit_masks(
    openlane_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata parsing opens no image while explicit tensor materialization opens one."""
    opened: list[object] = []
    real_open = Image.open

    def observed_open(*args: object, **kwargs: object) -> Image.Image:
        opened.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr("junctionlens.data.openlane.Image.open", observed_open)
    adapter = OpenLaneAdapter(openlane_root, _CONFIG)
    frame = adapter.load_frame("train", "segment-1", "100")
    assert opened == []
    assert tuple(camera.valid for camera in frame.cameras) == (
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    inputs = adapter.model_camera_inputs(frame)
    assert len(opened) == 1
    assert inputs.images.shape == (8, 3, 384, 640)
    assert inputs.camera_valid.tolist() == [True, False, False, False, False, False, False, False]
    assert np.count_nonzero(inputs.images[1:]) == 0
    assert np.count_nonzero(inputs.intrinsics[1:]) == 0
    assert np.count_nonzero(inputs.t_vehicle_camera[1:]) == 0
    expected_pixel = np.asarray(
        [
            (20.0 / 255.0 - 0.485) / 0.229,
            (30.0 / 255.0 - 0.456) / 0.224,
            (40.0 / 255.0 - 0.406) / 0.225,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(inputs.images[0, :, 192, 320], expected_pixel, atol=1e-6)
    np.testing.assert_allclose(
        inputs.intrinsics[0],
        [[480.0, 0.0, 320.0], [0.0, 480.0, 192.0], [0.0, 0.0, 1.0]],
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="read-only"):
        inputs.images[0, 0, 0, 0] = 1.0


def test_frame_iterator_does_not_eagerly_load_later_metadata(openlane_root: Path) -> None:
    """The full-profile iterator keeps only the current raw frame in its read path."""
    frames = OpenLaneAdapter(openlane_root, _CONFIG).iter_frames("sample")
    assert next(frames).key.timestamp_ns == 100
    later = openlane_root / "train/segment-1/info/200-ls.json"
    later.write_text("not-json", encoding="utf-8")
    with pytest.raises(OpenLaneAdapterError, match="invalid JSON"):
        next(frames)


def test_missing_source_dimensions_use_camera_specific_pins_without_decoding(
    openlane_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Official metadata can omit sizes without turning frame loading into image I/O."""
    metadata_path = openlane_root / "train/segment-1/info/100-ls.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sensor = metadata["sensor"]["ring_front_center"]
    del sensor["image_width"]
    del sensor["image_height"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        "junctionlens.data.openlane.Image.open",
        lambda *_args, **_kwargs: pytest.fail("metadata loading decoded an image"),
    )
    adapter = OpenLaneAdapter(openlane_root, _CONFIG)
    frame = adapter.load_frame("train", "segment-1", "100")
    front = frame.cameras[0]
    assert (front.original_width, front.original_height) == (1550, 2048)


def test_explicit_decode_rejects_pinned_or_declared_size_mismatch(openlane_root: Path) -> None:
    """Lazy loading validates real image headers before pixels reach the model."""
    metadata_path = openlane_root / "train/segment-1/info/100-ls.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sensor"]["ring_front_center"]["image_width"] = 101
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    adapter = OpenLaneAdapter(openlane_root, _CONFIG)
    frame = adapter.load_frame("train", "segment-1", "100")
    with pytest.raises(OpenLaneAdapterError, match="dimensions disagree"):
        adapter.load_camera_rgb(frame.cameras[0])


def test_inverse_projection_recovers_official_frame_fields(openlane_root: Path) -> None:
    """Canonical conversion remains lossless for every official evaluator field."""
    adapter = OpenLaneAdapter(openlane_root, _CONFIG)
    frame = adapter.load_frame("train", "segment-1", "100")
    projection = adapted_source_projection(frame)
    source = json.loads(
        (openlane_root / "train/segment-1/info/100-ls.json").read_text(encoding="utf-8")
    )
    assert projection["metadata_version"] == source["version"]
    assert projection["source_name"] == source["meta_data"]["source"]
    assert projection["source_segment_id"] == source["meta_data"]["source_id"]
    source_annotation = source["annotation"]
    for object_type in ("lane_segment", "traffic_element", "area"):
        for item in source_annotation[object_type]:
            item["id"] = str(item["id"])
    assert projection["annotation"] == source_annotation
    camera = projection["cameras"]["ring_front_center"]
    assert camera["image_path"] == source["sensor"]["ring_front_center"]["image_path"]
    np.testing.assert_allclose(
        camera["extrinsic"]["rotation"],
        source["sensor"]["ring_front_center"]["extrinsic"]["rotation"],
        atol=1e-12,
    )


def test_capacity_and_per_type_identity_audit(openlane_root: Path) -> None:
    """Lane, traffic-control, and area identities are never conflated."""
    frames = tuple(OpenLaneAdapter(openlane_root, _CONFIG).iter_frames("sample"))
    report = audit_report(
        frames,
        {"lane_segment": 96, "traffic_element": 64, "area": 32},
        {"lane_segment": 50.0, "traffic_element": 0.5, "area": 50.0},
        required_coverage=0.999,
    )
    assert report["capacity_gate_accepted"] is True
    assert report["capacity"]["lane_segment"]["maximum_count"] == 1
    assert {
        report["identity"][name]["temporal_kpi_state"]
        for name in ("lane_segment", "traffic_element", "area")
    } == {"ENABLED_SOURCE_IDENTITY"}


def test_adapter_rejects_duplicate_timestamps(openlane_root: Path) -> None:
    """Duplicate manifest timestamps cannot silently create duplicate frames."""
    manifest_path = openlane_root / "data_dict_example.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train"]["segment-1"].append("100.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OpenLaneAdapterError, match="duplicate timestamps"):
        tuple(OpenLaneAdapter(openlane_root, _CONFIG).iter_identifiers("sample"))


def test_adapter_rejects_transposed_or_misshaped_topology(openlane_root: Path) -> None:
    """Topology matrices must remain aligned with source list order."""
    metadata_path = openlane_root / "train/segment-1/info/100-ls.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["annotation"]["topology_lste"] = []
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(OpenLaneAdapterError, match="topology_lste must contain 1 rows"):
        OpenLaneAdapter(openlane_root, _CONFIG).load_frame("train", "segment-1", "100")


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda metadata: metadata["annotation"]["lane_segment"][0].__setitem__(
                "left_laneline_type", 9
            ),
            "left_laneline_type",
        ),
        (
            lambda metadata: metadata["annotation"]["topology_lsls"][0].__setitem__(0, 0.5),
            "exactly zero or one",
        ),
        (
            lambda metadata: metadata["annotation"]["traffic_element"].append(
                dict(metadata["annotation"]["traffic_element"][0])
            ),
            "duplicate IDs",
        ),
    ],
)
def test_map_element_bucket_categories_topology_and_ids_fail_closed(
    openlane_root: Path,
    mutate: Callable[[dict[str, Any]], None],
    reason: str,
) -> None:
    """Malformed Map Element Bucket labels never enter normalized training data."""
    metadata_path = openlane_root / "train/segment-1/info/100-ls.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(metadata)
    if "duplicate" in reason:
        metadata["annotation"]["topology_lste"] = [[1, 0]]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(OpenLaneAdapterError, match=reason):
        OpenLaneAdapter(openlane_root, _CONFIG).load_frame("train", "segment-1", "100")

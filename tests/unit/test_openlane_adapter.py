"""OpenLane-V2 sample adapter and audit tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from junctionlens.data.audit import audit_report
from junctionlens.data.contracts import CAMERA_SLOTS, CameraSlot
from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError

_CONFIG = Path("configs/data/openlane-v2-v2.1.adapter.yaml")


def test_adapter_normalizes_camera_order_coordinates_and_source_box(openlane_root: Path) -> None:
    """The real raw-JSON path preserves source data while freezing canonical tensors."""
    frames = tuple(OpenLaneAdapter(openlane_root, _CONFIG).iter_frames("sample"))
    assert [frame.key.timestamp_ns for frame in frames] == [100, 200]
    frame = frames[0]
    assert tuple(camera.slot for camera in frame.cameras) == CAMERA_SLOTS
    assert [camera.slot for camera in frame.cameras if camera.valid] == [CameraSlot.FRONT_CENTER]
    assert frame.lanes[0].centerline == ((1.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    control = frame.traffic_controls[0]
    assert control.source_camera == CameraSlot.FRONT_CENTER
    assert control.source_pixel_box.points == ((10.0, 20.0), (30.0, 40.0))
    assert control.normalized_half_open_box == (0.1, 0.25, 0.3, 0.5)
    np.testing.assert_allclose(frame.t_world_vehicle, np.eye(4), atol=1e-12)


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

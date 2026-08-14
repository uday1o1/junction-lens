"""Visual projection, range, and statistical audit tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from junctionlens.data.contracts import CameraFrame, CameraSlot
from junctionlens.data.geometry import rigid_transform
from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.data.visual_audit import (
    audit_dataset,
    load_audit_policy,
    project_points,
    render_bev_svg,
    render_camera_overlay,
    slice_values,
)

_ADAPTER_CONFIG = Path("configs/data/openlane-v2-v2.1.adapter.yaml")
_AUDIT_POLICY = Path("configs/data/openlane-v2-v2.1.audit-v1.yaml")


def audit_policy_for_fixture(tmp_path: Path) -> Path:
    """Freeze the repository-owned fixture frame under the production audit thresholds."""
    payload = yaml.safe_load(_AUDIT_POLICY.read_text(encoding="utf-8"))
    payload["frozen_frames"] = [
        {"split_id": "train", "segment_id": "segment-1", "timestamp": "100"}
    ]
    config_root = tmp_path / "configs"
    slice_path = config_root / "slices/v1.yaml"
    slice_path.parent.mkdir(parents=True)
    slice_path.write_bytes(Path("configs/slices/v1.yaml").read_bytes())
    path = config_root / "data/audit-policy.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_statistical_audit_covers_distributions_ranges_topology_and_slices(
    openlane_root: Path, tmp_path: Path
) -> None:
    """One streaming pass emits every required aggregate without retaining images."""
    adapter = OpenLaneAdapter(openlane_root, _ADAPTER_CONFIG)
    report = audit_dataset(adapter, "sample", load_audit_policy(audit_policy_for_fixture(tmp_path)))
    assert report["frame_count"] == 2
    assert report["capacity_distributions"]["lane_segment"]["histogram"] == {"1": 2}
    assert report["missing_camera_patterns"] == {"10000000": 2}
    assert report["topology_support"]["lane_control_positive"] == 2
    assert report["topology_support"]["lane_lane_self_positive"] == 0
    assert report["slice_support_preview"]["source_domain"]["argoverse2"] == {
        "frame_count": 2,
        "segment_count": 1,
    }
    assert report["slice_support_preview"]["low_luminance_proxy"]["low"]["frame_count"] == 2
    assert report["range_gate_accepted"] is True
    assert report["outside_hard_range_point_count"] == 0


def test_calibration_projection_matches_analytic_pixels(openlane_root: Path) -> None:
    """The overlay projection preserves a declared vehicle-to-camera analytic golden."""
    frame = OpenLaneAdapter(openlane_root, _ADAPTER_CONFIG).load_frame("train", "segment-1", "100")
    rotation = np.asarray([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    camera = CameraFrame(
        slot=CameraSlot.FRONT_CENTER,
        valid=True,
        source_camera="analytic",
        image_relative_path="analytic.png",
        capture_timestamp_ns=100,
        original_width=640,
        original_height=384,
        intrinsic=((100.0, 0.0, 320.0), (0.0, 100.0, 192.0), (0.0, 0.0, 1.0)),
        t_vehicle_camera=tuple(
            tuple(float(value) for value in row)
            for row in rigid_transform(rotation, [0.0, 0.0, 1.0], label="analytic")
        ),
        distortion_model="NONE",
        distortion_coefficients=(),
        image_transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    projected, valid = project_points(frame, camera, ((5.0, 0.0, 0.0), (10.0, -2.0, 0.0)))
    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(projected, [[320.0, 212.0], [340.0, 202.0]], atol=0.25)


def test_renderers_emit_inspectable_overlay_and_bev(openlane_root: Path, tmp_path: Path) -> None:
    """The frozen renderers expose source boxes, principal point, lanes, and topology."""
    adapter = OpenLaneAdapter(openlane_root, _ADAPTER_CONFIG)
    frame = adapter.load_frame("train", "segment-1", "100")
    overlay = render_camera_overlay(adapter, frame, frame.cameras[0])
    path = tmp_path / "overlay.png"
    path.write_bytes(overlay.png)
    with Image.open(path) as image:
        assert image.size == (100, 80)
        assert image.convert("RGB").getpixel((10, 20)) == (255, 72, 72)
        assert image.convert("RGB").getpixel((50, 40)) == (0, 255, 255)
    assert overlay.projected_point_count > 0
    bev = render_bev_svg(frame, load_audit_policy(audit_policy_for_fixture(tmp_path)))
    assert b'data-id="lane-10-center"' in bev
    assert b'data-edge="lane-control-0"' in bev


def test_seeded_geometry_range_defect_fails_with_nearby_control(
    openlane_root: Path, tmp_path: Path
) -> None:
    """Plausible fixture geometry passes while an extreme seeded point fails the range gate."""
    policy = load_audit_policy(audit_policy_for_fixture(tmp_path))
    adapter = OpenLaneAdapter(openlane_root, _ADAPTER_CONFIG)
    assert (
        audit_dataset(adapter, "sample", policy, compute_luminance=False)["range_gate_accepted"]
        is True
    )
    metadata_path = openlane_root / "train/segment-1/info/100-ls.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["annotation"]["lane_segment"][0]["centerline"][0][1] = 1000.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report = audit_dataset(
        OpenLaneAdapter(openlane_root, _ADAPTER_CONFIG),
        "sample",
        policy,
        compute_luminance=False,
    )
    assert report["range_gate_accepted"] is False
    assert report["outside_hard_range_point_count"] == 1


def test_slice_preview_distinguishes_merge_and_split_topology(
    openlane_root: Path, tmp_path: Path
) -> None:
    """In-degree and out-degree produce distinct reproducible topology slices."""
    frame = OpenLaneAdapter(openlane_root, _ADAPTER_CONFIG).load_frame("train", "segment-1", "100")
    lanes = (
        frame.lanes[0],
        replace(frame.lanes[0], source_object_id="lane-11"),
        replace(frame.lanes[0], source_object_id="lane-12"),
    )
    policy = load_audit_policy(audit_policy_for_fixture(tmp_path))
    merge = replace(
        frame,
        lanes=lanes,
        topology_lane_lane=((0, 0, 1), (0, 0, 1), (0, 0, 0)),
        topology_lane_traffic=((1,), (0,), (0,)),
    )
    split = replace(
        frame,
        lanes=lanes,
        topology_lane_lane=((0, 1, 1), (0, 0, 0), (0, 0, 0)),
        topology_lane_traffic=((1,), (0,), (0,)),
    )
    assert slice_values(merge, policy, "not-computed")["merge_or_split_topology"] == "merge"
    assert slice_values(split, policy, "not-computed")["merge_or_split_topology"] == "split"

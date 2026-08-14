"""Shared repository-owned OpenLane-like fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


def _metadata(root: Path, timestamp: int, pose_forward: float) -> None:
    image_relative = Path("train/segment-1/image/ring_front_center/frame.jpg")
    annotation = {
        "lane_segment": [
            {
                "id": 10,
                "centerline": [[0.0, 1.0, 0.0], [0.0, 5.0, 0.0]],
                "left_laneline": [[-1.0, 1.0, 0.0], [-1.0, 5.0, 0.0]],
                "right_laneline": [[1.0, 1.0, 0.0], [1.0, 5.0, 0.0]],
                "left_laneline_type": 1,
                "right_laneline_type": 2,
                "is_intersection_or_connector": False,
            }
        ],
        "traffic_element": [
            {
                "id": 20,
                "category": 1,
                "attribute": 2,
                "points": [[10.0, 20.0], [30.0, 40.0]],
            }
        ],
        "area": [
            {
                "id": 30,
                "category": 1,
                "points": [[-2.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 4.0, 0.0]],
            }
        ],
        "topology_lsls": [[0]],
        "topology_lste": [[1]],
    }
    metadata = {
        "version": "2.1",
        "segment_id": "segment-1",
        "meta_data": {"source": "argoverse2", "source_id": "source-segment"},
        "timestamp": timestamp,
        "sensor": {
            "ring_front_center": {
                "image_path": image_relative.as_posix(),
                "image_width": 100,
                "image_height": 80,
                "intrinsic": {
                    "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
                    "distortion": [],
                },
                "extrinsic": {
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation": [0.0, 0.0, 0.0],
                },
            }
        },
        "pose": {
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation": [0.0, pose_forward, 0.0],
        },
        "annotation": annotation,
    }
    info = root / "train/segment-1/info"
    info.mkdir(parents=True, exist_ok=True)
    (info / f"{timestamp}-ls.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


@pytest.fixture
def openlane_root(tmp_path: Path) -> Path:
    """Build a small repository-owned fixture in the pinned raw source shape."""
    root = tmp_path / "OpenLane-V2"
    image_path = root / "train/segment-1/image/ring_front_center/frame.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), color=(20, 30, 40)).save(image_path)
    (root / "data_dict_example.json").write_text(
        json.dumps({"train": {"segment-1": ["200.json", "100.json"]}}),
        encoding="utf-8",
    )
    _metadata(root, 100, 0.0)
    _metadata(root, 200, 1.0)
    return root

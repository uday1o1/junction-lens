"""Deterministic multi-camera calibration for synthetic truth."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from junctionlens.data.geometry import rigid_transform
from junctionlens.v1 import scene_control_graph_pb2 as scg

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 384
FOCAL_LENGTH_PX = 200.0


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """One canonical camera slot and its exact persisted calibration."""

    slot: int
    slug: str
    intrinsic: npt.NDArray[np.float64]
    t_vehicle_camera: npt.NDArray[np.float64]


_SLOT_PARAMETERS = (
    (scg.CAMERA_SLOT_FRONT_CENTER, "front-center", 0.0, (0.0, 0.0, 1.5)),
    (scg.CAMERA_SLOT_FRONT_LEFT, "front-left", math.pi / 4.0, (0.0, 0.8, 1.5)),
    (scg.CAMERA_SLOT_FRONT_RIGHT, "front-right", -math.pi / 4.0, (0.0, -0.8, 1.5)),
    (scg.CAMERA_SLOT_SIDE_LEFT, "side-left", math.pi / 2.0, (0.0, 0.9, 1.5)),
    (scg.CAMERA_SLOT_SIDE_RIGHT, "side-right", -math.pi / 2.0, (0.0, -0.9, 1.5)),
    (scg.CAMERA_SLOT_REAR_LEFT, "rear-left", 3.0 * math.pi / 4.0, (-1.0, 0.8, 1.5)),
    (scg.CAMERA_SLOT_REAR_CENTER, "rear-center", math.pi, (-1.0, 0.0, 1.5)),
    (scg.CAMERA_SLOT_REAR_RIGHT, "rear-right", -3.0 * math.pi / 4.0, (-1.0, -0.8, 1.5)),
)


def camera_calibrations() -> tuple[CameraCalibration, ...]:
    """Return all eight camera slots in canonical contract order."""
    intrinsic = np.asarray(
        [
            [FOCAL_LENGTH_PX, 0.0, IMAGE_WIDTH / 2.0],
            [0.0, FOCAL_LENGTH_PX, IMAGE_HEIGHT / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_to_forward = np.asarray(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    result: list[CameraCalibration] = []
    for slot, slug, yaw, translation in _SLOT_PARAMETERS:
        yaw_rotation = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result.append(
            CameraCalibration(
                slot=slot,
                slug=slug,
                intrinsic=intrinsic.copy(),
                t_vehicle_camera=rigid_transform(
                    yaw_rotation @ camera_to_forward,
                    translation,
                    label=f"synthetic.{slug}",
                ),
            )
        )
    return tuple(result)


def calibration_sha256() -> str:
    """Hash the persisted numerical calibration rather than object identity."""
    logical = [
        {
            "intrinsic": calibration.intrinsic.reshape(-1).tolist(),
            "slot": calibration.slot,
            "t_vehicle_camera": calibration.t_vehicle_camera.reshape(-1).tolist(),
        }
        for calibration in camera_calibrations()
    ]
    payload = json.dumps(logical, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

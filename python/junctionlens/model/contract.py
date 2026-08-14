"""Frozen tensor contract shared by export, parity, and native validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

from junctionlens.model.profile import M0ModelProfile
from junctionlens.model.spike import INPUT_NAMES, OUTPUT_NAMES

Dimension = int | Literal["batch"]


@dataclass(frozen=True, slots=True)
class TensorContract:
    """One named tensor in the public model boundary."""

    name: str
    element_type: Literal["bool", "float32"]
    shape: tuple[Dimension, ...]


def input_contract(profile: M0ModelProfile) -> tuple[TensorContract, ...]:
    """Return the ordered model input contract."""
    inputs = profile.input
    contract = (
        TensorContract(
            "images",
            "float32",
            (
                "batch",
                inputs.timestamps,
                inputs.cameras,
                inputs.channels,
                inputs.height,
                inputs.width,
            ),
        ),
        TensorContract("camera_valid", "bool", ("batch", inputs.timestamps, inputs.cameras)),
        TensorContract("intrinsics", "float32", ("batch", inputs.timestamps, inputs.cameras, 3, 3)),
        TensorContract(
            "t_vehicle_camera",
            "float32",
            ("batch", inputs.timestamps, inputs.cameras, 4, 4),
        ),
        TensorContract("ego_motion_previous_to_current", "float32", ("batch", 4, 4)),
        TensorContract("temporal_valid", "bool", ("batch",)),
    )
    if tuple(item.name for item in contract) != INPUT_NAMES:
        raise AssertionError("input implementation and contract names diverged")
    return contract


def output_contract(profile: M0ModelProfile) -> tuple[TensorContract, ...]:
    """Return the ordered model output contract."""
    shape = profile.model
    contract = (
        TensorContract("lane_existence_logits", "float32", ("batch", shape.lane_queries)),
        TensorContract(
            "lane_centerline", "float32", ("batch", shape.lane_queries, shape.lane_points, 3)
        ),
        TensorContract(
            "lane_left_boundary",
            "float32",
            ("batch", shape.lane_queries, shape.lane_points, 3),
        ),
        TensorContract(
            "lane_right_boundary",
            "float32",
            ("batch", shape.lane_queries, shape.lane_points, 3),
        ),
        TensorContract(
            "lane_left_boundary_logits",
            "float32",
            ("batch", shape.lane_queries, shape.lane_boundary_classes),
        ),
        TensorContract(
            "lane_right_boundary_logits",
            "float32",
            ("batch", shape.lane_queries, shape.lane_boundary_classes),
        ),
        TensorContract("lane_connector_logits", "float32", ("batch", shape.lane_queries)),
        TensorContract(
            "lane_geometry_scales",
            "float32",
            ("batch", shape.lane_queries, 3, shape.lane_points, 3),
        ),
        TensorContract(
            "lane_track_embeddings",
            "float32",
            ("batch", shape.lane_queries, shape.track_embedding_dimension),
        ),
        TensorContract("traffic_existence_logits", "float32", ("batch", shape.traffic_queries)),
        TensorContract("traffic_boxes", "float32", ("batch", shape.traffic_queries, 4)),
        TensorContract(
            "traffic_category_logits",
            "float32",
            ("batch", shape.traffic_queries, shape.traffic_categories),
        ),
        TensorContract(
            "traffic_attribute_logits",
            "float32",
            ("batch", shape.traffic_queries, shape.traffic_attributes),
        ),
        TensorContract("traffic_box_scales", "float32", ("batch", shape.traffic_queries, 4)),
        TensorContract(
            "traffic_track_embeddings",
            "float32",
            ("batch", shape.traffic_queries, shape.track_embedding_dimension),
        ),
        TensorContract("area_existence_logits", "float32", ("batch", shape.area_queries)),
        TensorContract(
            "area_category_logits",
            "float32",
            ("batch", shape.area_queries, shape.area_categories),
        ),
        TensorContract(
            "area_points", "float32", ("batch", shape.area_queries, shape.area_points, 3)
        ),
        TensorContract(
            "area_valid_logits", "float32", ("batch", shape.area_queries, shape.area_points)
        ),
        TensorContract(
            "area_geometry_scales",
            "float32",
            ("batch", shape.area_queries, shape.area_points, 3),
        ),
        TensorContract(
            "area_track_embeddings",
            "float32",
            ("batch", shape.area_queries, shape.track_embedding_dimension),
        ),
        TensorContract(
            "lane_successor_logits",
            "float32",
            ("batch", shape.lane_queries, shape.lane_queries),
        ),
        TensorContract(
            "control_lane_logits",
            "float32",
            ("batch", shape.traffic_queries, shape.lane_queries),
        ),
    )
    if tuple(item.name for item in contract) != OUTPUT_NAMES:
        raise AssertionError("output implementation and contract names diverged")
    return contract


def contract_sha256(contract: tuple[TensorContract, ...]) -> str:
    """Hash an ordered tensor contract with canonical JSON."""
    payload = json.dumps(
        [asdict(item) for item in contract],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

"""Deterministic unsigned 64-bit graph identifiers."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable

from junctionlens.v1 import scene_control_graph_pb2 as scg

NODE_TYPE_SHIFT = 56
ORDINAL_MASK = (1 << NODE_TYPE_SHIFT) - 1


def predicted_node_id(node_type: int, ordinal: int) -> int:
    """Encode a frozen node-type code and zero-based type-local ordinal."""
    if node_type not in {
        scg.NODE_TYPE_LANE_SEGMENT,
        scg.NODE_TYPE_TRAFFIC_CONTROL,
        scg.NODE_TYPE_ROAD_AREA,
    }:
        raise ValueError("node type must be a concrete V1 node type")
    if ordinal < 0 or ordinal >= ORDINAL_MASK:
        raise ValueError("ordinal is outside the 56-bit V1 range")
    return (node_type << NODE_TYPE_SHIFT) | (ordinal + 1)


def predicted_node_type(node_id: int) -> int:
    """Decode the frozen node-type code from a predicted node ID."""
    return node_id >> NODE_TYPE_SHIFT


def _parts(frame_key: scg.FrameKey, edge_type: int, source: int, target: int) -> Iterable[bytes]:
    yield b"junctionlens-edge-id-v1"
    for value in (
        frame_key.dataset_id,
        frame_key.dataset_version,
        frame_key.split_id,
        frame_key.segment_id,
    ):
        yield value.encode("utf-8")
    yield struct.pack(">q", frame_key.timestamp_ns)
    yield struct.pack(">I", frame_key.source_domain)
    yield frame_key.calibration_sha256.encode("ascii")
    yield frame_key.frame_manifest_sha256.encode("ascii")
    yield struct.pack(">IQQ", edge_type, source, target)


def edge_id(frame_key: scg.FrameKey, edge_type: int, source: int, target: int) -> int:
    """Hash the schema, frame identity, type, source, and target into a stable ID."""
    digest = hashlib.sha256()
    digest.update(struct.pack(">I", 1))
    for part in _parts(frame_key, edge_type, source, target):
        digest.update(struct.pack(">I", len(part)))
        digest.update(part)
    value = int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)
    return value or 1

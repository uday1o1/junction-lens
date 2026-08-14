"""Training-only topology targets and imbalance statistics for E1."""

from __future__ import annotations

import torch

from junctionlens.data.contracts import AdaptedFrame
from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.model.e0_data import PartitionIsolation, frame_targets, iter_partition_frames
from junctionlens.model.e0_profile import E0Profile
from junctionlens.model.e1_losses import E1EdgeWeights, E1LossError, E1Targets


def frame_e1_targets(frame: AdaptedFrame, base: E0Profile) -> E1Targets:
    """Convert lane-major OpenLane labels into the canonical control-major target."""
    lane_successor = torch.tensor(frame.topology_lane_lane, dtype=torch.bool)
    lane_traffic = torch.tensor(frame.topology_lane_traffic, dtype=torch.bool)
    if lane_traffic.ndim != 2:
        lane_traffic = lane_traffic.reshape(len(frame.lanes), len(frame.traffic_controls))
    target = E1Targets(
        nodes=frame_targets(frame, base),
        lane_successor=lane_successor.reshape(len(frame.lanes), len(frame.lanes)),
        control_lane=lane_traffic.transpose(0, 1).contiguous(),
    )
    target.validate()
    return target


def scan_edge_weights(
    adapter: OpenLaneAdapter, isolation: PartitionIsolation, base: E0Profile
) -> E1EdgeWeights:
    """Compute edge positive weights once from the isolated training partition."""
    if isolation.partition != "model_training":
        raise E1LossError("edge positive-weight scan may read only model_training")
    successor_positive = successor_possible = 0
    control_positive = control_possible = 0
    frames = 0
    for frame in iter_partition_frames(adapter, isolation):
        target = frame_e1_targets(frame, base)
        lanes = target.lane_successor.shape[0]
        controls = target.control_lane.shape[0]
        frames += 1
        successor_positive += int(target.lane_successor.sum().item())
        successor_possible += lanes * max(0, lanes - 1)
        control_positive += int(target.control_lane.sum().item())
        control_possible += controls * lanes
    if frames == 0 or successor_positive == 0 or control_positive == 0:
        raise E1LossError("training partition lacks a required positive topology population")
    if successor_positive > successor_possible or control_positive > control_possible:
        raise E1LossError("topology positive counts exceed their eligible populations")
    result = E1EdgeWeights(
        lane_successor_positive=max(
            1.0, (successor_possible - successor_positive) / successor_positive
        ),
        control_lane_positive=max(1.0, (control_possible - control_positive) / control_positive),
        source_partition=isolation.partition,
        source_split_manifest_sha256=isolation.split_manifest_sha256,
    )
    result.validate()
    return result


__all__ = ["frame_e1_targets", "scan_edge_weights"]

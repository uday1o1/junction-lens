"""Production learned-topology heads, targets, losses, and diagnostics."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.model.e0_data import PartitionIsolation
from junctionlens.model.e0_losses import E0Matches, MatchIndices, unit_training_weights
from junctionlens.model.e0_profile import load_e0_profile
from junctionlens.model.e0_training import build_e0_optimizer
from junctionlens.model.e1_data import frame_e1_targets, scan_edge_weights
from junctionlens.model.e1_losses import (
    E1EdgeWeights,
    E1LossError,
    e1_losses,
    topology_assignments,
    topology_only_losses,
    weighted_e1_total,
)
from junctionlens.model.e1_profile import load_e1_profile
from junctionlens.model.reference import E0_OUTPUT_NAMES
from junctionlens.model.topology import (
    JointGraphModel,
    LearnedTopologyHeads,
    TopologyOrderingError,
    canonical_control_lane_to_openlane,
    e1_outputs_by_name,
    openlane_lane_traffic_to_canonical,
)
from junctionlens.model.topology_diagnostic import run_topology_diagnostic

BASE_PATH = Path("configs/model/e0-independent-v1.yaml")
E1_PATH = Path("configs/model/e1-joint-v1.yaml")
ADAPTER_PATH = Path("configs/data/openlane-v2-v2.1.adapter.yaml")


def _profiles() -> tuple[object, object]:
    base = load_e0_profile(BASE_PATH)
    return base, load_e1_profile(E1_PATH, base)


def _head_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260813)
    lane = torch.randn(1, 3, 256)
    control = torch.randn(1, 2, 256)
    centerline = torch.zeros(1, 3, 11, 3)
    for index in range(3):
        centerline[0, index, :, 0] = torch.linspace(5.0 + 5 * index, 10.0 + 5 * index, 11)
    boxes = torch.tensor([[[0.4, 0.3, 0.5, 0.4], [0.5, 0.3, 0.6, 0.4]]])
    intrinsic = torch.tensor([[[320.0, 0.0, 320.0], [0.0, 320.0, 192.0], [0.0, 0.0, 1.0]]])
    transform = torch.eye(4).reshape(1, 4, 4)
    transform[0, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    transform[0, 2, 3] = 1.5
    return lane, control, centerline, boxes, intrinsic, transform


def test_e1_profile_is_hash_bound_to_shared_node_architecture(tmp_path: Path) -> None:
    base = load_e0_profile(BASE_PATH)
    profile = load_e1_profile(E1_PATH, base)
    changed = tmp_path / "e1.yaml"
    changed.write_text(
        E1_PATH.read_text(encoding="utf-8").replace(profile.base_profile_sha256, "a" * 64),
        encoding="utf-8",
    )

    assert profile.topology.allow_lane_self_edges is False
    with pytest.raises(ValueError, match="different E0"):
        load_e1_profile(changed, base)


def test_topology_heads_are_directed_geometry_aware_and_mask_self_edges() -> None:
    base = load_e0_profile(BASE_PATH)
    profile = load_e1_profile(E1_PATH, base)
    head = LearnedTopologyHeads(base, profile)

    lane, control = head(*_head_inputs())

    assert lane.shape == (1, 3, 3)
    assert control.shape == (1, 2, 3)
    assert torch.all(lane.diagonal(dim1=1, dim2=2) == -30.0)
    assert not torch.equal(lane, lane.transpose(1, 2))
    assert all(parameter.requires_grad for parameter in head.parameters())


def test_control_lane_adapter_uses_exact_asymmetric_transpose() -> None:
    canonical = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)

    openlane = canonical_control_lane_to_openlane(canonical, control_count=2, lane_count=3)
    restored = openlane_lane_traffic_to_canonical(openlane, lane_count=3, control_count=2)

    assert openlane.tolist() == [[1, 0], [0, 1], [1, 0]]
    assert torch.equal(restored, canonical)
    assert openlane.shape == (3, 2)
    with pytest.raises(TopologyOrderingError, match="lane-major"):
        openlane_lane_traffic_to_canonical(canonical, lane_count=3, control_count=2)


def test_node_matches_induce_query_ordered_edge_targets_and_gradients() -> None:
    base = load_e0_profile(BASE_PATH)
    profile = load_e1_profile(E1_PATH, base)
    _, _, target_centerline, target_boxes, intrinsic, transform = _head_inputs()
    from junctionlens.model.topology_diagnostic import _synthetic_nodes

    targets, _, _, _, _ = _synthetic_nodes(base)
    lane_permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    control_permutation = torch.tensor([2, 0, 1])
    matches = E0Matches(
        MatchIndices(torch.arange(6), lane_permutation),
        MatchIndices(torch.arange(3), control_permutation),
        MatchIndices(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)),
    )
    head = LearnedTopologyHeads(base, profile)
    lane_features = torch.randn(1, 6, 256, requires_grad=True)
    control_features = torch.randn(1, 3, 256, requires_grad=True)
    centerline = targets.nodes.lane_centerline[lane_permutation][None]
    boxes = targets.nodes.traffic_boxes[control_permutation][None]
    lane_logits, control_logits = head(
        lane_features, control_features, centerline, boxes, intrinsic, transform
    )
    outputs = {
        "lane_successor_logits": lane_logits,
        "control_lane_logits": control_logits,
        "lane_centerline": centerline,
    }
    assignments = topology_assignments(outputs, targets, matches)
    weights = E1EdgeWeights(5.0, 2.0, "model_training", "a" * 64)
    losses = topology_only_losses(outputs, assignments, weights)
    total = sum(losses.values())
    total.backward()

    assert assignments.lane_successor_target[0, 1, 3]
    assert assignments.control_lane_target[0, 1, 1]
    assert lane_features.grad is not None
    assert control_features.grad is not None
    assert target_centerline.shape == (1, 3, 11, 3)
    assert target_boxes.shape == (1, 2, 4)


def test_full_predicted_node_objective_backpropagates_through_shared_queries() -> None:
    base = load_e0_profile(BASE_PATH)
    profile = load_e1_profile(E1_PATH, base)
    from junctionlens.model.topology_diagnostic import _synthetic_nodes

    targets, _, _, intrinsic, transform = _synthetic_nodes(base)
    model = JointGraphModel(base, profile)
    lane = torch.randn(1, base.architecture.lane_queries, 256, requires_grad=True)
    control = torch.randn(1, base.architecture.traffic_queries, 256, requires_grad=True)
    area = torch.randn(1, base.architecture.area_queries, 256, requires_grad=True)
    node_outputs = model.node_outputs(lane, control, area)
    named_nodes = dict(zip(E0_OUTPUT_NAMES, node_outputs, strict=True))
    lane_logits, control_logits = model.topology(
        lane,
        control,
        named_nodes["lane_centerline"],
        named_nodes["traffic_boxes"],
        intrinsic,
        transform,
    )
    outputs = e1_outputs_by_name((*node_outputs, lane_logits, control_logits))
    matches = E0Matches(
        MatchIndices(torch.arange(6), torch.arange(6)),
        MatchIndices(torch.arange(3), torch.arange(3)),
        MatchIndices(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)),
    )
    losses = e1_losses(
        outputs,
        targets,
        matches,
        unit_training_weights(torch.device("cpu")),
        E1EdgeWeights(5.0, 2.0, "model_training", "a" * 64),
    )
    total = weighted_e1_total(losses)
    total.backward()

    assert tuple(outputs)[: len(E0_OUTPUT_NAMES)] == E0_OUTPUT_NAMES
    assert lane.grad is not None
    assert torch.isfinite(lane.grad).all()
    assert control.grad is not None
    assert torch.isfinite(control.grad).all()
    assert model.topology.lane_source.weight.grad is not None


def test_joint_optimizer_includes_every_topology_parameter_once() -> None:
    base = load_e0_profile(BASE_PATH)
    profile = load_e1_profile(E1_PATH, base)
    model = JointGraphModel(base, profile)

    optimizer = build_e0_optimizer(model, base)
    parameter_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]

    assert len(parameter_ids) == len(set(parameter_ids)) == len(list(model.parameters()))
    assert id(model.topology.lane_source.weight) in parameter_ids


def test_training_only_edge_weight_scan_uses_canonical_counts(
    openlane_root: Path,
) -> None:
    metadata_path = openlane_root / "train/segment-1/info/100-ls.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    second_lane = dict(metadata["annotation"]["lane_segment"][0])
    second_lane["id"] = 11
    second_lane["centerline"] = [[0.0, 5.0, 0.0], [0.0, 9.0, 0.0]]
    second_lane["left_laneline"] = [[-1.0, 5.0, 0.0], [-1.0, 9.0, 0.0]]
    second_lane["right_laneline"] = [[1.0, 5.0, 0.0], [1.0, 9.0, 0.0]]
    metadata["annotation"]["lane_segment"].append(second_lane)
    metadata["annotation"]["topology_lsls"] = [[0, 1], [0, 0]]
    metadata["annotation"]["topology_lste"] = [[1], [0]]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    second_path = openlane_root / "train/segment-1/info/200-ls.json"
    second_metadata = deepcopy(metadata)
    second_metadata["timestamp"] = 200
    second_path.write_text(json.dumps(second_metadata), encoding="utf-8")
    (openlane_root / "data_dict_subset_A.json").write_bytes(
        (openlane_root / "data_dict_example.json").read_bytes()
    )
    isolation = PartitionIsolation(
        "a" * 64,
        "model_training",
        frozenset({"segment-1"}),
        frozenset(),
        "b" * 64,
        "c" * 64,
        "d" * 64,
    )
    adapter = OpenLaneAdapter(openlane_root, ADAPTER_PATH)
    base = load_e0_profile(BASE_PATH)

    weights = scan_edge_weights(adapter, isolation, base)
    frame = next(adapter.iter_frames("full"))
    target = frame_e1_targets(frame, base)

    assert weights.source_partition == "model_training"
    assert weights.lane_successor_positive == 1.0
    assert weights.control_lane_positive == 1.0
    assert target.control_lane.tolist() == [[True, False]]
    with pytest.raises(E1LossError, match="model_training"):
        E1EdgeWeights(1.0, 1.0, "model_selection", "a" * 64).validate()


def test_oracle_and_predicted_node_learning_gates_pass() -> None:
    base = load_e0_profile(BASE_PATH)
    profile = load_e1_profile(E1_PATH, base)

    result = run_topology_diagnostic(base, profile)

    assert result["state"] == "ACCEPTED"
    assert result["base_profile_sha256"] == base.canonical_sha256()
    assert result["e1_profile_sha256"] == profile.canonical_sha256()
    modes = {item["mode"]: item for item in result["results"]}
    assert modes["oracle-nodes"]["lane_successor_f1"] == 1.0
    assert modes["oracle-nodes"]["control_lane_f1"] == 1.0
    assert modes["predicted-nodes"]["maximum_query_gradient"] > 0.0

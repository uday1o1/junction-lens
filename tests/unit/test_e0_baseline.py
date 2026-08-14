"""Production E0 architecture, loss, isolation, and geometric-rule tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from junctionlens.data.manifests import SegmentRecord, freeze_split_manifest, load_split_policy
from junctionlens.model.e0_artifacts import E0ArtifactError, finalize_e0_artifacts
from junctionlens.model.e0_data import E0DataError, load_partition_isolation
from junctionlens.model.e0_losses import (
    E0_LOSS_WEIGHTS,
    E0Targets,
    e0_losses,
    match_e0,
    weighted_e0_total,
)
from junctionlens.model.e0_profile import load_e0_profile
from junctionlens.model.e0_training import (
    E0TrainingError,
    build_e0_optimizer,
    select_e0_checkpoint,
)
from junctionlens.model.independent_linker import (
    FittedThreshold,
    RuleObservation,
    fit_independent_linker,
    successor_edges,
)
from junctionlens.model.reference import (
    EfficientNetB0Fpn,
    GroundPlaneProjector,
    ReferenceNodeModel,
    projection_parity,
)
from junctionlens.registry.store import canonical_json_bytes

_PROFILE = Path("configs/model/e0-independent-v1.yaml")
_POLICY = Path("configs/data/openlane-v2-v2.1.split-v1.yaml")


def test_e0_profile_freezes_architecture_recipe_and_seed_matrix() -> None:
    profile = load_e0_profile(_PROFILE)

    assert profile.architecture.bev_shape == (200, 160)
    assert profile.architecture.hidden_dimension == 256
    assert profile.training.gradient_accumulation_steps == 8
    assert profile.seeds == (20260813, 20260814, 20260815)
    assert len(profile.canonical_sha256()) == 64


def test_efficientnet_feature_pyramid_uses_locked_stride_eight_surface() -> None:
    model = EfficientNetB0Fpn().eval()

    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))

    assert output.shape == (1, 128, 8, 8)


def test_reference_and_vectorized_ground_plane_projection_match() -> None:
    torch.manual_seed(20260813)
    profile = load_e0_profile(_PROFILE)
    projector = GroundPlaneProjector(profile).eval()
    features = torch.randn(1, 2, 128, 8, 8)
    valid = torch.tensor([[True, True]])
    intrinsics = torch.zeros(1, 2, 3, 3)
    intrinsics[..., 0, 0] = 200.0
    intrinsics[..., 1, 1] = 200.0
    intrinsics[..., 0, 2] = 320.0
    intrinsics[..., 1, 2] = 192.0
    intrinsics[..., 2, 2] = 1.0
    rotation = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    transforms = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    transforms[..., :3, :3] = rotation
    transforms[..., 2, 3] = 1.5

    result = projection_parity(projector, features, valid, intrinsics, transforms)

    assert result.passed
    assert result.maximum_absolute_error <= 1.0e-6


def test_linker_fit_is_training_only_deterministic_and_smallest_on_ties() -> None:
    profile = load_e0_profile(_PROFILE)
    successor = (
        RuleObservation("s-positive", 1.2, 20.0, 1),
        RuleObservation("s-negative", 2.2, 20.0, 0),
    )
    control = (
        RuleObservation("c-positive", 35.0, 25.0, 1),
        RuleObservation("c-negative", 50.0, 25.0, 0),
    )

    artifact = fit_independent_linker(
        profile,
        successor,
        control,
        partition="model_training",
        training_split_manifest_sha256="a" * 64,
    )

    assert artifact.successor.distance == 1.5
    assert artifact.successor.heading_difference_deg == 25.0
    assert artifact.control.distance == 40.0
    assert artifact.control.heading_difference_deg == 30.0
    assert artifact.successor.f1 == artifact.control.f1 == 1.0
    assert artifact.canonical_bytes() == artifact.canonical_bytes()
    with pytest.raises(ValueError, match="only on model_training"):
        fit_independent_linker(
            profile,
            successor,
            control,
            partition="model_selection",
            training_split_manifest_sha256="a" * 64,
        )


def test_successor_rules_are_directed_and_exclude_self_edges() -> None:
    lanes = (
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        np.asarray([[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float64),
    )
    threshold = FittedThreshold(1.0, 10.0, 1, 0, 0, 1.0)

    assert successor_edges(lanes, threshold) == ((0, 1),)


def _outputs() -> dict[str, torch.Tensor]:
    def value(*shape: int, positive: bool = False) -> torch.Tensor:
        result = torch.randn(*shape, requires_grad=True)
        return torch.nn.functional.softplus(result) + 1.0e-4 if positive else result

    return {
        "lane_existence_logits": value(1, 3),
        "lane_centerline": value(1, 3, 11, 3),
        "lane_left_boundary": value(1, 3, 11, 3),
        "lane_right_boundary": value(1, 3, 11, 3),
        "lane_left_boundary_logits": value(1, 3, 3),
        "lane_right_boundary_logits": value(1, 3, 3),
        "lane_connector_logits": value(1, 3),
        "lane_geometry_scales": value(1, 3, 3, 11, 3, positive=True),
        "lane_track_embeddings": value(1, 3, 32),
        "traffic_existence_logits": value(1, 2),
        "traffic_boxes": torch.sigmoid(value(1, 2, 4)),
        "traffic_category_logits": value(1, 2, 2),
        "traffic_attribute_logits": value(1, 2, 13),
        "traffic_box_scales": value(1, 2, 4, positive=True),
        "traffic_track_embeddings": value(1, 2, 32),
        "area_existence_logits": value(1, 2),
        "area_category_logits": value(1, 2, 2),
        "area_points": value(1, 2, 20, 3),
        "area_valid_logits": value(1, 2, 20),
        "area_geometry_scales": value(1, 2, 20, 3, positive=True),
        "area_track_embeddings": value(1, 2, 32),
    }


def test_matching_and_every_e0_head_loss_are_finite_and_differentiable() -> None:
    torch.manual_seed(20260813)
    outputs = _outputs()
    centerline = torch.zeros(1, 11, 3)
    targets = E0Targets(
        lane_centerline=centerline,
        lane_left_boundary=centerline + torch.tensor([0.0, 1.75, 0.0]),
        lane_right_boundary=centerline - torch.tensor([0.0, 1.75, 0.0]),
        lane_left_type=torch.tensor([0]),
        lane_right_type=torch.tensor([1]),
        lane_connector=torch.tensor([True]),
        traffic_boxes=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        traffic_category=torch.tensor([1]),
        traffic_attribute=torch.tensor([2]),
        area_points=torch.zeros(1, 20, 3),
        area_valid=torch.ones(1, 20, dtype=torch.bool),
        area_category=torch.tensor([0]),
    )

    matches = match_e0(outputs, targets)
    losses = e0_losses(outputs, targets, matches)
    total = weighted_e0_total(losses)
    total.backward()

    assert set(losses) == set(E0_LOSS_WEIGHTS)
    assert all(torch.isfinite(value) for value in losses.values())
    assert torch.isfinite(total)


def test_empty_node_populations_keep_existence_losses_finite() -> None:
    outputs = _outputs()
    targets = E0Targets(
        lane_centerline=torch.empty(0, 11, 3),
        lane_left_boundary=torch.empty(0, 11, 3),
        lane_right_boundary=torch.empty(0, 11, 3),
        lane_left_type=torch.empty(0, dtype=torch.int64),
        lane_right_type=torch.empty(0, dtype=torch.int64),
        lane_connector=torch.empty(0, dtype=torch.bool),
        traffic_boxes=torch.empty(0, 4),
        traffic_category=torch.empty(0, dtype=torch.int64),
        traffic_attribute=torch.empty(0, dtype=torch.int64),
        area_points=torch.empty(0, 20, 3),
        area_valid=torch.empty(0, 20, dtype=torch.bool),
        area_category=torch.empty(0, dtype=torch.int64),
    )

    losses = e0_losses(outputs, targets, match_e0(outputs, targets))

    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["lane_centerline"] == 0.0
    assert losses["traffic_box_l1"] == 0.0
    assert losses["area_geometry"] == 0.0


def test_training_statistics_reject_selection_partition(tmp_path: Path) -> None:
    policy = load_split_policy(_POLICY)
    records = [
        SegmentRecord(f"segment-{index:04d}", "argoverse2" if index < 400 else "nuscenes")
        for index in range(700)
    ]
    manifest = freeze_split_manifest(
        records,
        policy,
        source_frame_manifest_sha256="a" * 64,
        source_frame_records_sha256="b" * 64,
        source_dataset_manifest_sha256="c" * 64,
    )
    split_path = tmp_path / "split.json"
    split_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    training = load_partition_isolation(
        split_path, _POLICY, partition="model_training", statistics=True
    )

    assert len(training.segment_ids) == 350
    assert not training.segment_ids & training.forbidden_segment_ids
    with pytest.raises(E0DataError, match="statistics"):
        load_partition_isolation(split_path, _POLICY, partition="model_selection", statistics=True)


def test_profile_rejects_recipe_drift(tmp_path: Path) -> None:
    payload = _PROFILE.read_text(encoding="utf-8").replace(
        "base_learning_rate: 0.0002", "base_learning_rate: 0.0003"
    )
    changed = tmp_path / "profile.yaml"
    changed.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="optimizer recipe"):
        load_e0_profile(changed)


def test_optimizer_uses_frozen_backbone_multiplier_without_overlap() -> None:
    profile = load_e0_profile(_PROFILE)
    model = ReferenceNodeModel(profile)

    optimizer = build_e0_optimizer(model, profile)

    assert [group["group_name"] for group in optimizer.param_groups] == ["backbone", "model"]
    assert [group["lr"] for group in optimizer.param_groups] == [2.0e-5, 2.0e-4]
    parameter_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids)) == len(list(model.parameters()))


def test_checkpoint_selection_is_lexicographic_and_split_bound(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    for epoch in (1, 2, 3):
        (checkpoint_root / f"epoch-{epoch:02d}.pt").write_bytes(f"epoch-{epoch}".encode())
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.e0-selection-scores.v1",
                "scores": [
                    {
                        "epoch": 1,
                        "lane_control_topology": 0.8,
                        "official_composite": 0.9,
                        "negative_log_likelihood": 0.4,
                        "selection_split_manifest_sha256": "d" * 64,
                    },
                    {
                        "epoch": 2,
                        "lane_control_topology": 0.9,
                        "official_composite": 0.7,
                        "negative_log_likelihood": 0.2,
                        "selection_split_manifest_sha256": "d" * 64,
                    },
                    {
                        "epoch": 3,
                        "lane_control_topology": 0.9,
                        "official_composite": 0.8,
                        "negative_log_likelihood": 0.5,
                        "selection_split_manifest_sha256": "d" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    receipt = select_e0_checkpoint(
        tmp_path, scores, expected_selection_split_manifest_sha256="d" * 64
    )

    assert receipt["selected"]["epoch"] == 3
    with pytest.raises(E0TrainingError, match="different split"):
        select_e0_checkpoint(tmp_path, scores, expected_selection_split_manifest_sha256="e" * 64)


def test_three_seed_finalization_requires_measured_failures_and_writes_once(
    tmp_path: Path,
) -> None:
    profile = load_e0_profile(_PROFILE)
    run_roots = []
    for seed in profile.seeds:
        run_root = tmp_path / str(seed)
        checkpoint_root = run_root / "checkpoints"
        checkpoint_root.mkdir(parents=True)
        checkpoint = checkpoint_root / "epoch-03.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (run_root / "run-manifest.json").write_text(
            json.dumps(
                {
                    "state": "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION",
                    "seed": seed,
                    "source_commit": "a" * 40,
                    "profile_sha256": profile.canonical_sha256(),
                    "training_split_manifest_sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        (run_root / "selection-receipt.json").write_text(
            json.dumps(
                {
                    "state": "SELECTED_ON_MODEL_SELECTION",
                    "selection_split_manifest_sha256": "c" * 64,
                    "checkpoint_sha256": checkpoint_sha256,
                    "selected": {"epoch": 3},
                }
            ),
            encoding="utf-8",
        )
        run_roots.append(run_root)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.e0-measured-evidence.v1",
                "seed_metrics": [
                    {
                        "seed": seed,
                        "DET_l": 0.5,
                        "DET_t": 0.4,
                        "TOP_ll": 0.3,
                        "TOP_lt": 0.2,
                        "negative_log_likelihood": 1.0,
                    }
                    for seed in profile.seeds
                ],
                "limitations": ["Geometric linking cannot learn scene-specific topology."],
                "failed_examples": [
                    {
                        "frame_key": "train/segment/1",
                        "reason_code": "WRONG_CONTROL_ASSIGNMENT",
                        "summary": "The distance rule attached a control to the wrong lane.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "final"

    receipt = finalize_e0_artifacts(profile, run_roots, evidence, output)

    assert receipt["state"] == "ACCEPTED"
    assert "Measured failed examples" in (output / "MODEL_CARD.md").read_text(encoding="utf-8")
    with pytest.raises(E0ArtifactError, match="replace existing"):
        finalize_e0_artifacts(profile, run_roots, evidence, output)

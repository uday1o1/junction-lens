"""Deterministic full-corpus training and selection support for E0."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.model.e0_data import (
    E0Inputs,
    PartitionIsolation,
    frame_inputs,
    frame_targets,
    iter_partition_frames,
)
from junctionlens.model.e0_losses import (
    E0Targets,
    E0TrainingWeights,
    e0_losses,
    match_e0,
    weighted_e0_total,
)
from junctionlens.model.e0_profile import E0Profile
from junctionlens.model.reference import ReferenceNodeModel, e0_outputs_by_name
from junctionlens.model.selection import apply_frozen_early_stopping, score_order
from junctionlens.registry.store import canonical_json_bytes
from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_json_object_path


class E0TrainingError(RuntimeError):
    """Raised when E0 training or checkpoint selection cannot remain reproducible."""


@dataclass(frozen=True, slots=True)
class TrainingStatistics:
    frame_count: int
    lane_count: int
    traffic_count: int
    area_count: int
    left_boundary_type_count: tuple[int, ...]
    right_boundary_type_count: tuple[int, ...]
    traffic_category_count: tuple[int, ...]
    traffic_attribute_count: tuple[int, ...]
    area_category_count: tuple[int, ...]
    source_partition: str
    source_split_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointScore:
    epoch: int
    lane_control_topology: float
    official_composite: float
    negative_log_likelihood: float
    selection_split_manifest_sha256: str


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(project_root: Path) -> str:
    override = os.environ.get("JUNCTIONLENS_SOURCE_COMMIT")
    if override is not None:
        value = override
    else:
        git = shutil.which("git")
        if git is None:
            raise E0TrainingError("git is required to identify E0 training source")
        status = subprocess.run(
            [git, "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.returncode != 0 or status.stdout:
            raise E0TrainingError("E0 training requires a clean, explainable source checkout")
        result = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise E0TrainingError("cannot resolve the source commit")
        value = result.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise E0TrainingError("source commit must be a full lowercase Git object ID")
    return value


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def build_e0_optimizer(model: ReferenceNodeModel, profile: E0Profile) -> torch.optim.AdamW:
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    remaining = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    if (
        not backbone
        or not remaining
        or len(backbone_ids) + len(remaining) != len(list(model.parameters()))
    ):
        raise E0TrainingError("E0 optimizer parameter groups are incomplete or overlapping")
    recipe = profile.training
    return torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": recipe.base_learning_rate * recipe.backbone_learning_rate_multiplier,
                "group_name": "backbone",
                "target_lr": recipe.base_learning_rate * recipe.backbone_learning_rate_multiplier,
            },
            {
                "params": remaining,
                "lr": recipe.base_learning_rate,
                "group_name": "model",
                "target_lr": recipe.base_learning_rate,
            },
        ],
        weight_decay=recipe.weight_decay,
    )


def _class_weights(counts: tuple[int, ...]) -> Tensor:
    total = sum(counts)
    if total <= 0:
        return torch.ones(len(counts), dtype=torch.float32)
    inverse = torch.tensor([0.0 if count == 0 else total / count for count in counts])
    supported = inverse > 0
    inverse[supported] /= inverse[supported].mean()
    return inverse


def scan_training_statistics(
    adapter: OpenLaneAdapter, isolation: PartitionIsolation, profile: E0Profile
) -> tuple[TrainingStatistics, E0TrainingWeights]:
    if isolation.partition != profile.training.training_partition:
        raise E0TrainingError("statistics isolation is not model_training")
    lane_count = traffic_count = area_count = frame_count = 0
    left = [0] * profile.architecture.boundary_classes
    right = [0] * profile.architecture.boundary_classes
    traffic_category = [0] * profile.architecture.traffic_categories
    traffic_attribute = [0] * profile.architecture.traffic_attributes
    area_category = [0] * profile.architecture.area_categories
    for frame in iter_partition_frames(adapter, isolation):
        frame_count += 1
        lane_count += len(frame.lanes)
        traffic_count += len(frame.traffic_controls)
        area_count += len(frame.road_areas)
        for lane in frame.lanes:
            left[lane.left_boundary_type] += 1
            right[lane.right_boundary_type] += 1
        for control in frame.traffic_controls:
            traffic_category[control.category - 1] += 1
            traffic_attribute[control.attribute] += 1
        for area in frame.road_areas:
            area_category[area.category - 1] += 1
    if frame_count == 0 or lane_count == 0 or traffic_count == 0 or area_count == 0:
        raise E0TrainingError("E0 training partition lacks a required node population")
    statistics = TrainingStatistics(
        frame_count,
        lane_count,
        traffic_count,
        area_count,
        tuple(left),
        tuple(right),
        tuple(traffic_category),
        tuple(traffic_attribute),
        tuple(area_category),
        isolation.partition,
        isolation.split_manifest_sha256,
    )
    boundary_counts = tuple(a + b for a, b in zip(left, right, strict=True))
    weights = E0TrainingWeights(
        max(1.0, (frame_count * profile.architecture.lane_queries - lane_count) / lane_count),
        max(
            1.0,
            (frame_count * profile.architecture.traffic_queries - traffic_count) / traffic_count,
        ),
        max(1.0, (frame_count * profile.architecture.area_queries - area_count) / area_count),
        _class_weights(boundary_counts),
        _class_weights(tuple(traffic_category)),
        _class_weights(tuple(traffic_attribute)),
        _class_weights(tuple(area_category)),
    )
    return statistics, weights


def _to_device_inputs(inputs: E0Inputs, device: torch.device) -> tuple[Tensor, ...]:
    return tuple(value.to(device, non_blocking=device.type == "cuda") for value in inputs.tensors())


def _to_device_targets(targets: E0Targets, device: torch.device) -> E0Targets:
    return E0Targets(
        **{
            name: value.to(device, non_blocking=device.type == "cuda")
            for name, value in asdict(targets).items()
        }
    )


def _learning_rate_scale(step: int, total_steps: int, profile: E0Profile) -> float:
    recipe = profile.training
    if step <= recipe.warmup_steps:
        return step / recipe.warmup_steps
    progress = min(1.0, (step - recipe.warmup_steps) / max(1, total_steps - recipe.warmup_steps))
    minimum_scale = recipe.minimum_learning_rate / recipe.base_learning_rate
    return minimum_scale + (1.0 - minimum_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_learning_rate(
    optimizer: torch.optim.AdamW, step: int, total_steps: int, profile: E0Profile
) -> Mapping[str, float]:
    scale = _learning_rate_scale(step, total_steps, profile)
    result = {}
    for group in optimizer.param_groups:
        name = str(group["group_name"])
        group["lr"] = float(group["target_lr"]) * scale
        result[name] = float(group["lr"])
    return result


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    temporary.replace(path)


def _write_immutable_json(path: Path, payload: object) -> None:
    serialized = canonical_json_bytes(payload) + b"\n"
    if path.is_symlink():
        raise E0TrainingError("immutable E0 receipt cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != serialized:
            raise E0TrainingError("immutable E0 receipt already exists with different content")
        return
    path.write_bytes(serialized)


def _load_score_json(path: Path) -> Mapping[str, Any]:
    try:
        return load_json_object_path(
            path,
            "E0 selection scores",
            ParseLimits(
                max_bytes=16 * 1024 * 1024,
                max_depth=24,
                max_nodes=500_000,
                max_container_items=100_000,
            ),
        )
    except ParseBoundaryError as error:
        raise E0TrainingError(str(error)) from error


def _checkpoint(
    path: Path,
    model: ReferenceNodeModel,
    optimizer: torch.optim.AdamW,
    *,
    epoch: int,
    optimizer_step: int,
    seed: int,
    profile: E0Profile,
) -> str:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": "junctionlens.e0-checkpoint.v1",
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "seed": seed,
            "profile": profile.model_dump(mode="json"),
            "profile_sha256": profile.canonical_sha256(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary,
    )
    temporary.replace(path)
    return _hash_file(path)


def run_e0_training(
    profile: E0Profile,
    adapter: OpenLaneAdapter,
    training_isolation: PartitionIsolation,
    output_root: Path,
    *,
    seed: int,
    project_root: Path,
    dataset_registration: Mapping[str, Any],
    device_name: str | None = None,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Train one complete declared E0 seed and retain every epoch for frozen selection."""
    if seed not in profile.seeds:
        raise E0TrainingError("E0 seed is outside the predeclared matrix")
    required_registration = {
        "archive_sha256",
        "dataset_id",
        "license_receipt_sha256",
        "manifest_sha256",
        "profile",
    }
    if (
        not required_registration.issubset(dataset_registration)
        or dataset_registration.get("dataset_id") != "openlane-v2-v2.1"
        or dataset_registration.get("profile") != "full"
        or dataset_registration.get("manifest_sha256")
        != training_isolation.source_dataset_manifest_sha256
    ):
        raise E0TrainingError("E0 dataset registration differs from the frozen split source")
    for key in ("archive_sha256", "license_receipt_sha256", "manifest_sha256"):
        value = dataset_registration.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise E0TrainingError(f"E0 dataset registration has an invalid {key}")
    seed_training(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise E0TrainingError("CUDA training was requested but no CUDA device is available")
    statistics, training_weights = scan_training_statistics(adapter, training_isolation, profile)
    training_weights = training_weights.to(device)
    output_exists = output_root.exists() or output_root.is_symlink()
    if output_root.is_symlink() or (output_exists and not output_root.is_dir()):
        raise E0TrainingError("E0 run output must be a real directory")
    if output_exists and not resume:
        raise E0TrainingError("E0 run output already exists; pass --resume or choose a new root")
    if not output_exists:
        output_root.mkdir(parents=True)
    checkpoint_root = output_root / "checkpoints"
    if checkpoint_root.is_symlink():
        raise E0TrainingError("E0 checkpoint root cannot be a symlink")
    checkpoint_root.mkdir(exist_ok=resume)
    model = ReferenceNodeModel(profile).to(device)
    optimizer = build_e0_optimizer(model, profile)
    recipe = profile.training
    updates_per_epoch = math.ceil(statistics.frame_count / recipe.gradient_accumulation_steps)
    total_updates = updates_per_epoch * recipe.maximum_epochs
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    source_commit = _source_commit(project_root)
    initial_manifest: dict[str, Any] = {
        "schema_version": "junctionlens.e0-training-run.v1",
        "state": "RUNNING",
        "experiment_id": profile.experiment_id,
        "seed": seed,
        "source_commit": source_commit,
        "profile_sha256": profile.canonical_sha256(),
        "training_split_manifest_sha256": training_isolation.split_manifest_sha256,
        "source_dataset_manifest_sha256": training_isolation.source_dataset_manifest_sha256,
        "source_frame_manifest_sha256": training_isolation.source_frame_manifest_sha256,
        "source_frame_records_sha256": training_isolation.source_frame_records_sha256,
        "dataset_registration": {
            "dataset_id": "openlane-v2-v2.1",
            "profile": "full",
            "archive_sha256": dataset_registration["archive_sha256"],
            "manifest_sha256": dataset_registration["manifest_sha256"],
            "license_receipt_sha256": dataset_registration["license_receipt_sha256"],
        },
        "training_statistics": asdict(statistics),
        "training_weights": {
            "lane_existence_positive": training_weights.lane_existence_positive,
            "traffic_existence_positive": training_weights.traffic_existence_positive,
            "area_existence_positive": training_weights.area_existence_positive,
            "boundary_type": training_weights.boundary_type.cpu().tolist(),
            "traffic_category": training_weights.traffic_category.cpu().tolist(),
            "traffic_attribute": training_weights.traffic_attribute.cpu().tolist(),
            "area_category": training_weights.area_category.cpu().tolist(),
        },
        "device_type": device.type,
        "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
        "recipe": profile.training.model_dump(mode="json"),
    }
    manifest_path = output_root / "run-manifest.json"
    optimizer_step = 0
    start_epoch = 1
    metrics_mode = "x"
    if resume:
        try:
            existing = load_json_object_path(
                manifest_path,
                "existing E0 run manifest",
                ParseLimits(max_bytes=16 * 1024 * 1024, max_depth=24, max_nodes=500_000),
            )
        except (OSError, ParseBoundaryError) as error:
            raise E0TrainingError("cannot load the existing E0 run manifest") from error
        run_manifest = existing
        if run_manifest.get("state") == "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION":
            return run_manifest
        for key in (
            "experiment_id",
            "seed",
            "source_commit",
            "profile_sha256",
            "training_split_manifest_sha256",
            "source_dataset_manifest_sha256",
            "source_frame_manifest_sha256",
            "source_frame_records_sha256",
            "training_statistics",
            "dataset_registration",
            "recipe",
        ):
            if run_manifest.get(key) != initial_manifest[key]:
                raise E0TrainingError(f"resume manifest differs in {key}")
        receipts = sorted(checkpoint_root.glob("epoch-*.json"))
        if not receipts:
            raise E0TrainingError("resume requested but no durable checkpoint receipt exists")
        try:
            receipt = load_json_object_path(
                receipts[-1],
                "E0 checkpoint receipt",
                ParseLimits(max_bytes=1024 * 1024, max_depth=16, max_nodes=10_000),
            )
        except ParseBoundaryError as error:
            raise E0TrainingError(str(error)) from error
        if not isinstance(receipt.get("epoch"), int):
            raise E0TrainingError("latest E0 checkpoint receipt is invalid")
        last_epoch = int(receipt["epoch"])
        checkpoint_path = checkpoint_root / f"epoch-{last_epoch:02d}.pt"
        if _hash_file(checkpoint_path) != receipt.get("checkpoint_sha256"):
            raise E0TrainingError("latest E0 checkpoint failed hash verification")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if (
            not isinstance(state, dict)
            or state.get("profile_sha256") != profile.canonical_sha256()
            or state.get("seed") != seed
            or state.get("epoch") != last_epoch
        ):
            raise E0TrainingError("latest E0 checkpoint identity differs from the run")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        cuda_states = state.get("cuda_rng_states", [])
        if device.type == "cuda":
            if not isinstance(cuda_states, list) or not cuda_states:
                raise E0TrainingError("CUDA checkpoint lacks deterministic RNG state")
            torch.cuda.set_rng_state_all(cuda_states)
        optimizer_step = int(state["optimizer_step"])
        start_epoch = last_epoch + 1
        metrics_mode = "a"
    else:
        run_manifest = initial_manifest
        _write_json(manifest_path, run_manifest)
    metrics_path = output_root / "training-metrics.jsonl"
    started = time.monotonic()
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, recipe.maximum_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            frame_index = 0
            for frame_index, frame in enumerate(
                iter_partition_frames(adapter, training_isolation), start=1
            ):
                group_start = ((frame_index - 1) // recipe.gradient_accumulation_steps) * (
                    recipe.gradient_accumulation_steps
                )
                group_size = min(
                    recipe.gradient_accumulation_steps, statistics.frame_count - group_start
                )
                inputs = _to_device_inputs(frame_inputs(adapter, frame, profile), device)
                targets = _to_device_targets(frame_targets(frame, profile), device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=device.type == "cuda",
                ):
                    outputs = e0_outputs_by_name(model(*inputs))
                    matches = match_e0(outputs, targets)
                    losses = e0_losses(outputs, targets, matches, training_weights)
                    total = weighted_e0_total(losses)
                    scaled_total = total / group_size
                if not torch.isfinite(total):
                    raise E0TrainingError("E0 training produced a nonfinite loss")
                scaler.scale(scaled_total).backward()
                end_group = (
                    frame_index % recipe.gradient_accumulation_steps == 0
                    or frame_index == statistics.frame_count
                )
                gradient_norm: float | None = None
                learning_rates: Mapping[str, float] | None = None
                if end_group:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), recipe.gradient_clip_norm
                    )
                    gradient_norm = float(norm.item())
                    if not math.isfinite(gradient_norm):
                        raise E0TrainingError("E0 training produced a nonfinite gradient norm")
                    optimizer_step += 1
                    learning_rates = _set_learning_rate(
                        optimizer, optimizer_step, total_updates, profile
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                record = {
                    "epoch": epoch,
                    "frame_index": frame_index,
                    "frame_key": {
                        "segment_id": frame.key.segment_id,
                        "timestamp_ns": frame.key.timestamp_ns,
                    },
                    "optimizer_step": optimizer_step,
                    "unweighted_losses": {
                        name: float(value.detach().item()) for name, value in sorted(losses.items())
                    },
                    "weighted_total_loss": float(total.detach().item()),
                    "gradient_norm_before_clip": gradient_norm,
                    "learning_rates": learning_rates,
                }
                metrics_file.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
            metrics_file.flush()
            os.fsync(metrics_file.fileno())
            if frame_index != statistics.frame_count:
                raise E0TrainingError("training epoch frame count differs from the frozen scan")
            checkpoint_path = checkpoint_root / f"epoch-{epoch:02d}.pt"
            checkpoint_sha256 = _checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                optimizer_step=optimizer_step,
                seed=seed,
                profile=profile,
            )
            _write_json(
                checkpoint_root / f"epoch-{epoch:02d}.json",
                {
                    "schema_version": "junctionlens.e0-checkpoint-receipt.v1",
                    "epoch": epoch,
                    "checkpoint_sha256": checkpoint_sha256,
                    "profile_sha256": profile.canonical_sha256(),
                    "source_commit": source_commit,
                },
            )
    run_manifest["state"] = "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION"
    run_manifest["optimizer_steps"] = optimizer_step
    run_manifest["training_seconds"] = time.monotonic() - started
    run_manifest["metrics_sha256"] = _hash_file(metrics_path)
    _write_json(output_root / "run-manifest.json", run_manifest)
    return run_manifest


def select_e0_checkpoint(
    run_root: Path,
    score_path: Path,
    *,
    expected_selection_split_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Apply the frozen lexicographic selection rule to externally evaluated epochs."""
    if len(expected_selection_split_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_selection_split_manifest_sha256
    ):
        raise E0TrainingError("expected E0 selection split hash is invalid")
    value = _load_score_json(score_path)
    if set(value) != {"schema_version", "scores"}:
        raise E0TrainingError("E0 selection score schema is invalid")
    if value["schema_version"] != "junctionlens.e0-selection-scores.v1":
        raise E0TrainingError("E0 selection score version is unsupported")
    if not isinstance(value["scores"], list) or not value["scores"]:
        raise E0TrainingError("E0 selection scores are empty")
    scores = []
    for raw in value["scores"]:
        if not isinstance(raw, dict) or set(raw) != {
            "epoch",
            "lane_control_topology",
            "negative_log_likelihood",
            "official_composite",
            "selection_split_manifest_sha256",
        }:
            raise E0TrainingError("E0 selection score schema is invalid")
        score = CheckpointScore(**raw)
        if isinstance(score.epoch, bool) or not isinstance(score.epoch, int) or score.epoch <= 0:
            raise E0TrainingError("E0 selection epoch is invalid")
        if score.selection_split_manifest_sha256 != expected_selection_split_manifest_sha256:
            raise E0TrainingError("E0 selection score used a different split manifest")
        if not all(
            isinstance(item, int | float) and not isinstance(item, bool) and math.isfinite(item)
            for item in (
                score.lane_control_topology,
                score.official_composite,
                score.negative_log_likelihood,
            )
        ):
            raise E0TrainingError("E0 selection score contains a nonfinite value")
        checkpoint = run_root / "checkpoints" / f"epoch-{score.epoch:02d}.pt"
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise E0TrainingError(f"E0 selection checkpoint for epoch {score.epoch} is missing")
        scores.append(score)
    epochs = [item.epoch for item in scores]
    if len(epochs) != len(set(epochs)):
        raise E0TrainingError("E0 selection scores repeat an epoch")
    eligible, early_stopping = apply_frozen_early_stopping(
        scores,
        minimum_epoch=20,
        patience=8,
    )
    selected = min(eligible, key=score_order)
    checkpoint = run_root / "checkpoints" / f"epoch-{selected.epoch:02d}.pt"
    receipt = {
        "schema_version": "junctionlens.e0-selection-receipt.v1",
        "state": "SELECTED_ON_MODEL_SELECTION",
        "selection_rule": "lane-control-topology-desc,official-composite-desc,nll-asc,epoch-asc",
        "early_stopping": asdict(early_stopping),
        "selection_split_manifest_sha256": expected_selection_split_manifest_sha256,
        "selected": asdict(selected),
        "checkpoint_sha256": _hash_file(checkpoint),
        "score_artifact_sha256": _hash_file(score_path),
    }
    _write_immutable_json(run_root / "selection-receipt.json", receipt)
    return receipt


__all__ = [
    "CheckpointScore",
    "E0TrainingError",
    "TrainingStatistics",
    "build_e0_optimizer",
    "run_e0_training",
    "scan_training_statistics",
    "seed_training",
    "select_e0_checkpoint",
]

"""Reproducible full-corpus training and frozen checkpoint selection for E1."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.model.e0_data import (
    PartitionIsolation,
    frame_inputs,
    iter_partition_frames,
)
from junctionlens.model.e0_losses import E0Targets, match_e0
from junctionlens.model.e0_profile import E0Profile
from junctionlens.model.e0_training import (
    CheckpointScore,
    _hash_file,
    _load_score_json,
    _set_learning_rate,
    _source_commit,
    _to_device_inputs,
    _write_immutable_json,
    _write_json,
    build_e0_optimizer,
    scan_training_statistics,
    seed_training,
)
from junctionlens.model.e1_data import frame_e1_targets, scan_edge_weights
from junctionlens.model.e1_losses import E1Targets, e1_losses, weighted_e1_total
from junctionlens.model.e1_profile import E1Profile
from junctionlens.model.selection import apply_frozen_early_stopping, score_order
from junctionlens.model.topology import JointGraphModel, e1_outputs_by_name
from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_json_object_path


class E1TrainingError(RuntimeError):
    """Raised when the E1 experiment cannot preserve its frozen contract."""


def _validate_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise E1TrainingError(f"{label} must be a lowercase SHA-256")
    return value


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return load_json_object_path(
            path,
            label,
            ParseLimits(
                max_bytes=16 * 1024 * 1024,
                max_depth=24,
                max_nodes=500_000,
                max_container_items=100_000,
            ),
        )
    except ParseBoundaryError as error:
        raise E1TrainingError(str(error)) from error


def _validate_diagnostic(path: Path, base: E0Profile, profile: E1Profile) -> str:
    value = _load_json_object(path, "E1 topology diagnostic")
    if set(value) != {
        "base_profile_sha256",
        "e1_profile_sha256",
        "experiment_id",
        "results",
        "schema_version",
        "seed",
        "state",
    }:
        raise E1TrainingError("E1 topology diagnostic schema is invalid")
    if (
        value.get("schema_version") != "junctionlens.e1-topology-diagnostic.v1"
        or value.get("state") != "ACCEPTED"
        or value.get("experiment_id") != profile.experiment_id
        or value.get("seed") != profile.diagnostics.seed
        or value.get("base_profile_sha256") != base.canonical_sha256()
        or value.get("e1_profile_sha256") != profile.canonical_sha256()
    ):
        raise E1TrainingError("E1 topology diagnostic identity or state is invalid")
    results = value.get("results")
    if not isinstance(results, list) or {
        item.get("mode") for item in results if isinstance(item, dict)
    } != set(profile.diagnostics.modes):
        raise E1TrainingError("E1 topology diagnostic modes are incomplete")
    for result in results:
        if (
            not isinstance(result, dict)
            or result.get("state") != "ACCEPTED"
            or not isinstance(result.get("lane_successor_f1"), int | float)
            or not isinstance(result.get("control_lane_f1"), int | float)
            or float(result["lane_successor_f1"])
            < profile.diagnostics.oracle_lane_successor_f1_minimum
            or float(result["control_lane_f1"]) < profile.diagnostics.oracle_control_lane_f1_minimum
        ):
            raise E1TrainingError("E1 topology diagnostic did not pass every learning gate")
    return _hash_file(path)


def _to_device_e0_targets(targets: E0Targets, device: torch.device) -> E0Targets:
    return E0Targets(
        **{
            name: value.to(device, non_blocking=device.type == "cuda")
            for name, value in asdict(targets).items()
        }
    )


def _to_device_e1_targets(targets: E1Targets, device: torch.device) -> E1Targets:
    return E1Targets(
        nodes=_to_device_e0_targets(targets.nodes, device),
        lane_successor=targets.lane_successor.to(device, non_blocking=device.type == "cuda"),
        control_lane=targets.control_lane.to(device, non_blocking=device.type == "cuda"),
    )


def _checkpoint(
    path: Path,
    model: JointGraphModel,
    optimizer: torch.optim.AdamW,
    *,
    epoch: int,
    optimizer_step: int,
    seed: int,
    base: E0Profile,
    profile: E1Profile,
) -> str:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": "junctionlens.e1-checkpoint.v1",
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "seed": seed,
            "base_profile_sha256": base.canonical_sha256(),
            "e1_profile": profile.model_dump(mode="json"),
            "e1_profile_sha256": profile.canonical_sha256(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary,
    )
    temporary.replace(path)
    return _hash_file(path)


def _validate_registration(
    dataset_registration: Mapping[str, Any], training_isolation: PartitionIsolation
) -> None:
    required = {
        "archive_sha256",
        "dataset_id",
        "license_receipt_sha256",
        "manifest_sha256",
        "profile",
    }
    if (
        not required.issubset(dataset_registration)
        or dataset_registration.get("dataset_id") != "openlane-v2-v2.1"
        or dataset_registration.get("profile") != "full"
        or dataset_registration.get("manifest_sha256")
        != training_isolation.source_dataset_manifest_sha256
    ):
        raise E1TrainingError("E1 dataset registration differs from the frozen split source")
    for key in ("archive_sha256", "license_receipt_sha256", "manifest_sha256"):
        _validate_sha256(dataset_registration.get(key), f"E1 dataset registration {key}")


def run_e1_training(
    base: E0Profile,
    profile: E1Profile,
    adapter: OpenLaneAdapter,
    training_isolation: PartitionIsolation,
    topology_diagnostic_path: Path,
    output_root: Path,
    *,
    project_root: Path,
    dataset_registration: Mapping[str, Any],
    device_name: str | None = None,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Train the sole predeclared E1 screening seed and retain every epoch."""
    profile.validate_base(base)
    seed = profile.diagnostics.seed
    if seed != base.seeds[0]:
        raise E1TrainingError("E1 seed differs from the predeclared release-decision seed")
    _validate_registration(dataset_registration, training_isolation)
    diagnostic_sha256 = _validate_diagnostic(topology_diagnostic_path, base, profile)
    seed_training(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise E1TrainingError("CUDA training was requested but no CUDA device is available")
    statistics, node_weights = scan_training_statistics(adapter, training_isolation, base)
    edge_weights = scan_edge_weights(adapter, training_isolation, base)
    node_weights = node_weights.to(device)
    output_exists = output_root.exists() or output_root.is_symlink()
    if output_root.is_symlink() or (output_exists and not output_root.is_dir()):
        raise E1TrainingError("E1 run output must be a real directory")
    if output_exists and not resume:
        raise E1TrainingError("E1 run output already exists; pass --resume or choose a new root")
    if not output_exists:
        output_root.mkdir(parents=True)
    checkpoint_root = output_root / "checkpoints"
    if checkpoint_root.is_symlink():
        raise E1TrainingError("E1 checkpoint root cannot be a symlink")
    checkpoint_root.mkdir(exist_ok=resume)
    model = JointGraphModel(base, profile).to(device)
    optimizer = build_e0_optimizer(model, base)
    recipe = base.training
    updates_per_epoch = math.ceil(statistics.frame_count / recipe.gradient_accumulation_steps)
    total_updates = updates_per_epoch * recipe.maximum_epochs
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    source_commit = _source_commit(project_root)
    initial_manifest: dict[str, Any] = {
        "schema_version": "junctionlens.e1-training-run.v1",
        "state": "RUNNING",
        "experiment_id": profile.experiment_id,
        "seed": seed,
        "source_commit": source_commit,
        "base_profile_sha256": base.canonical_sha256(),
        "e1_profile_sha256": profile.canonical_sha256(),
        "topology_diagnostic_sha256": diagnostic_sha256,
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
        "node_training_weights": {
            "lane_existence_positive": node_weights.lane_existence_positive,
            "traffic_existence_positive": node_weights.traffic_existence_positive,
            "area_existence_positive": node_weights.area_existence_positive,
            "boundary_type": node_weights.boundary_type.cpu().tolist(),
            "traffic_category": node_weights.traffic_category.cpu().tolist(),
            "traffic_attribute": node_weights.traffic_attribute.cpu().tolist(),
            "area_category": node_weights.area_category.cpu().tolist(),
        },
        "edge_training_weights": asdict(edge_weights),
        "device_type": device.type,
        "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
        "recipe": recipe.model_dump(mode="json"),
    }
    manifest_path = output_root / "run-manifest.json"
    optimizer_step = 0
    start_epoch = 1
    metrics_mode = "x"
    if resume:
        existing = _load_json_object(manifest_path, "E1 run manifest")
        run_manifest = dict(existing)
        if run_manifest.get("state") == "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION":
            return run_manifest
        for key in (
            "experiment_id",
            "seed",
            "source_commit",
            "base_profile_sha256",
            "e1_profile_sha256",
            "topology_diagnostic_sha256",
            "training_split_manifest_sha256",
            "source_dataset_manifest_sha256",
            "source_frame_manifest_sha256",
            "source_frame_records_sha256",
            "training_statistics",
            "dataset_registration",
            "recipe",
        ):
            if run_manifest.get(key) != initial_manifest[key]:
                raise E1TrainingError(f"resume manifest differs in {key}")
        receipts = sorted(checkpoint_root.glob("epoch-*.json"))
        if not receipts:
            raise E1TrainingError("resume requested but no durable E1 checkpoint exists")
        receipt = _load_json_object(receipts[-1], "E1 checkpoint receipt")
        if not isinstance(receipt.get("epoch"), int):
            raise E1TrainingError("latest E1 checkpoint receipt is invalid")
        last_epoch = int(receipt["epoch"])
        checkpoint_path = checkpoint_root / f"epoch-{last_epoch:02d}.pt"
        if _hash_file(checkpoint_path) != receipt.get("checkpoint_sha256"):
            raise E1TrainingError("latest E1 checkpoint failed hash verification")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if (
            not isinstance(state, dict)
            or state.get("base_profile_sha256") != base.canonical_sha256()
            or state.get("e1_profile_sha256") != profile.canonical_sha256()
            or state.get("seed") != seed
            or state.get("epoch") != last_epoch
        ):
            raise E1TrainingError("latest E1 checkpoint identity differs from the run")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        cuda_states = state.get("cuda_rng_states", [])
        if device.type == "cuda":
            if not isinstance(cuda_states, list) or not cuda_states:
                raise E1TrainingError("CUDA checkpoint lacks deterministic RNG state")
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
                inputs: tuple[Tensor, ...] = _to_device_inputs(
                    frame_inputs(adapter, frame, base), device
                )
                targets = _to_device_e1_targets(frame_e1_targets(frame, base), device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=device.type == "cuda",
                ):
                    outputs = e1_outputs_by_name(model(*inputs))
                    matches = match_e0(outputs, targets.nodes)
                    losses = e1_losses(outputs, targets, matches, node_weights, edge_weights)
                    total = weighted_e1_total(losses)
                    scaled_total = total / group_size
                if not torch.isfinite(total):
                    raise E1TrainingError("E1 training produced a nonfinite loss")
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
                        raise E1TrainingError("E1 training produced a nonfinite gradient norm")
                    optimizer_step += 1
                    learning_rates = _set_learning_rate(
                        optimizer, optimizer_step, total_updates, base
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
                raise E1TrainingError("training epoch frame count differs from the frozen scan")
            checkpoint_path = checkpoint_root / f"epoch-{epoch:02d}.pt"
            checkpoint_sha256 = _checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                optimizer_step=optimizer_step,
                seed=seed,
                base=base,
                profile=profile,
            )
            _write_json(
                checkpoint_root / f"epoch-{epoch:02d}.json",
                {
                    "schema_version": "junctionlens.e1-checkpoint-receipt.v1",
                    "epoch": epoch,
                    "checkpoint_sha256": checkpoint_sha256,
                    "base_profile_sha256": base.canonical_sha256(),
                    "e1_profile_sha256": profile.canonical_sha256(),
                    "source_commit": source_commit,
                },
            )
    run_manifest["state"] = "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION"
    run_manifest["optimizer_steps"] = optimizer_step
    run_manifest["training_seconds"] = time.monotonic() - started
    run_manifest["metrics_sha256"] = _hash_file(metrics_path)
    _write_json(manifest_path, run_manifest)
    return run_manifest


def select_e1_checkpoint(
    run_root: Path,
    score_path: Path,
    *,
    expected_selection_split_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Apply the frozen E1 topology, official, NLL checkpoint ordering."""
    _validate_sha256(
        expected_selection_split_manifest_sha256, "expected E1 selection split identity"
    )
    manifest = _load_json_object(run_root / "run-manifest.json", "E1 run manifest")
    if manifest.get("state") != "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION":
        raise E1TrainingError("E1 run is not ready for frozen selection")
    value = _load_score_json(score_path)
    if set(value) != {"schema_version", "scores"}:
        raise E1TrainingError("E1 selection score schema is invalid")
    if value["schema_version"] != "junctionlens.e1-selection-scores.v1":
        raise E1TrainingError("E1 selection score version is unsupported")
    if not isinstance(value["scores"], list) or not value["scores"]:
        raise E1TrainingError("E1 selection scores are empty")
    scores: list[CheckpointScore] = []
    for raw in value["scores"]:
        if not isinstance(raw, dict) or set(raw) != {
            "epoch",
            "lane_control_topology",
            "negative_log_likelihood",
            "official_composite",
            "selection_split_manifest_sha256",
        }:
            raise E1TrainingError("E1 selection score schema is invalid")
        score = CheckpointScore(**raw)
        if isinstance(score.epoch, bool) or not isinstance(score.epoch, int) or score.epoch <= 0:
            raise E1TrainingError("E1 selection epoch is invalid")
        if score.selection_split_manifest_sha256 != expected_selection_split_manifest_sha256:
            raise E1TrainingError("E1 selection score used a different split manifest")
        if not all(
            isinstance(item, int | float)
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in (
                score.lane_control_topology,
                score.official_composite,
                score.negative_log_likelihood,
            )
        ):
            raise E1TrainingError("E1 selection score contains a nonfinite value")
        checkpoint = run_root / "checkpoints" / f"epoch-{score.epoch:02d}.pt"
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise E1TrainingError(f"E1 selection checkpoint for epoch {score.epoch} is missing")
        scores.append(score)
    epochs = [score.epoch for score in scores]
    if len(epochs) != len(set(epochs)):
        raise E1TrainingError("E1 selection scores repeat an epoch")
    recipe = manifest.get("recipe")
    if (
        not isinstance(recipe, dict)
        or recipe.get("minimum_early_stopping_epoch") != 20
        or recipe.get("early_stopping_patience") != 8
    ):
        raise E1TrainingError("E1 run manifest lacks the frozen early-stopping recipe")
    eligible, early_stopping = apply_frozen_early_stopping(scores, minimum_epoch=20, patience=8)
    selected = min(eligible, key=score_order)
    checkpoint = run_root / "checkpoints" / f"epoch-{selected.epoch:02d}.pt"
    receipt = {
        "schema_version": "junctionlens.e1-selection-receipt.v1",
        "state": "SELECTED_ON_MODEL_SELECTION",
        "experiment_id": "E1-joint",
        "selection_rule": "lane-control-topology-desc,official-composite-desc,nll-asc,epoch-asc",
        "early_stopping": asdict(early_stopping),
        "selection_split_manifest_sha256": expected_selection_split_manifest_sha256,
        "selected": asdict(selected),
        "checkpoint_sha256": _hash_file(checkpoint),
        "score_artifact_sha256": _hash_file(score_path),
        "base_profile_sha256": manifest.get("base_profile_sha256"),
        "e1_profile_sha256": manifest.get("e1_profile_sha256"),
    }
    _write_immutable_json(run_root / "selection-receipt.json", receipt)
    return receipt


__all__ = [
    "E1TrainingError",
    "run_e1_training",
    "select_e1_checkpoint",
]

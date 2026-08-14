"""Final E0 manifest and truthful model-card assembly from measured seed evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from junctionlens.model.e0_profile import E0Profile
from junctionlens.registry.store import canonical_json_bytes


class E0ArtifactError(RuntimeError):
    """Raised when E0 evidence is incomplete, inconsistent, or unsafe to publish."""


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise E0ArtifactError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise E0ArtifactError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise E0ArtifactError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise E0ArtifactError(f"refusing to replace existing E0 artifact: {path.name}")
    path.write_bytes(payload)


def finalize_e0_artifacts(
    profile: E0Profile,
    run_roots: Sequence[Path],
    measured_evidence_path: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    """Require three selected seeds and render a model card without broadening claims."""
    if len(run_roots) != len(profile.seeds):
        raise E0ArtifactError("E0 finalization requires exactly three run roots")
    runs = []
    for run_root in run_roots:
        manifest = _load_object(run_root / "run-manifest.json", "E0 run manifest")
        selection = _load_object(run_root / "selection-receipt.json", "E0 selection receipt")
        if manifest.get("state") != "TRAINING_COMPLETE_AWAITING_FROZEN_SELECTION":
            raise E0ArtifactError("E0 run is not training-complete")
        if selection.get("state") != "SELECTED_ON_MODEL_SELECTION":
            raise E0ArtifactError("E0 run has no frozen model-selection receipt")
        if manifest.get("profile_sha256") != profile.canonical_sha256():
            raise E0ArtifactError("E0 run profile hash differs from the final profile")
        seed = manifest.get("seed")
        selected = selection.get("selected")
        if not isinstance(seed, int) or not isinstance(selected, dict):
            raise E0ArtifactError("E0 run seed or selected checkpoint is invalid")
        checkpoint = run_root / "checkpoints" / f"epoch-{int(selected['epoch']):02d}.pt"
        if _hash(checkpoint) != selection.get("checkpoint_sha256"):
            raise E0ArtifactError("E0 selected checkpoint failed hash verification")
        runs.append(
            {
                "seed": seed,
                "source_commit": manifest["source_commit"],
                "training_split_manifest_sha256": manifest["training_split_manifest_sha256"],
                "selection_split_manifest_sha256": selection["selection_split_manifest_sha256"],
                "checkpoint_sha256": selection["checkpoint_sha256"],
                "selected": selected,
                "run_manifest_sha256": _hash(run_root / "run-manifest.json"),
                "selection_receipt_sha256": _hash(run_root / "selection-receipt.json"),
            }
        )
    runs.sort(key=lambda item: int(item["seed"]))
    if tuple(int(item["seed"]) for item in runs) != tuple(sorted(profile.seeds)):
        raise E0ArtifactError("E0 run roots do not contain the three predeclared seeds")
    for field in (
        "source_commit",
        "training_split_manifest_sha256",
        "selection_split_manifest_sha256",
    ):
        if len({str(item[field]) for item in runs}) != 1:
            raise E0ArtifactError(f"E0 seed runs differ in {field}")
    evidence = _load_object(measured_evidence_path, "E0 measured evidence")
    if (
        set(evidence)
        != {
            "failed_examples",
            "limitations",
            "schema_version",
            "seed_metrics",
        }
        or evidence.get("schema_version") != "junctionlens.e0-measured-evidence.v1"
    ):
        raise E0ArtifactError("E0 measured evidence schema is invalid")
    seed_metrics = evidence["seed_metrics"]
    limitations = evidence["limitations"]
    failed_examples = evidence["failed_examples"]
    if not isinstance(seed_metrics, list) or len(seed_metrics) != 3:
        raise E0ArtifactError("E0 measured evidence must contain three seed metric records")
    if (
        not isinstance(limitations, list)
        or not limitations
        or not all(isinstance(item, str) and item.strip() for item in limitations)
    ):
        raise E0ArtifactError("E0 measured evidence requires nonempty limitations")
    if not isinstance(failed_examples, list) or not failed_examples:
        raise E0ArtifactError("E0 measured evidence requires at least one failed example")
    observed_seeds = []
    for raw in seed_metrics:
        if not isinstance(raw, dict) or set(raw) != {
            "DET_l",
            "DET_t",
            "TOP_ll",
            "TOP_lt",
            "negative_log_likelihood",
            "seed",
        }:
            raise E0ArtifactError("E0 seed metric record is invalid")
        observed_seeds.append(raw["seed"])
        if not all(
            isinstance(raw[key], int | float)
            and not isinstance(raw[key], bool)
            and math.isfinite(float(raw[key]))
            for key in ("DET_l", "DET_t", "TOP_ll", "TOP_lt", "negative_log_likelihood")
        ):
            raise E0ArtifactError("E0 seed metric record contains a nonfinite value")
    if tuple(sorted(observed_seeds)) != tuple(sorted(profile.seeds)):
        raise E0ArtifactError("E0 measured metrics differ from the predeclared seed matrix")
    for failed in failed_examples:
        if not isinstance(failed, dict) or set(failed) != {"frame_key", "reason_code", "summary"}:
            raise E0ArtifactError("E0 failed-example record is invalid")
        if not all(isinstance(failed[key], str) and failed[key].strip() for key in failed):
            raise E0ArtifactError("E0 failed-example fields must be nonempty strings")
    model_manifest = {
        "schema_version": "junctionlens.e0-model-manifest.v1",
        "state": "THREE_SEED_BASELINE_COMPLETE",
        "experiment_id": profile.experiment_id,
        "profile_sha256": profile.canonical_sha256(),
        "primary_seed": 20260813,
        "robustness_seeds": [20260814, 20260815],
        "dataset_license": "CC-BY-NC-SA-4.0",
        "weights_redistribution_allowed": False,
        "runs": runs,
        "measured_evidence_sha256": _hash(measured_evidence_path),
        "seed_metrics": seed_metrics,
    }
    model_manifest_bytes = canonical_json_bytes(model_manifest) + b"\n"
    lines = [
        "# E0 independent baseline model card",
        "",
        "E0 uses the shared multi-camera node architecture and frozen geometric linking rules.",
        "The model does not learn lane-successor or control-to-lane topology.",
        "Weights are dataset-derived under CC-BY-NC-SA-4.0 terms and are not approved "
        "for redistribution.",
        "",
        "## Measured three-seed results",
        "",
        "| Seed | DET_l | DET_t | TOP_ll | TOP_lt | NLL |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in sorted(seed_metrics, key=lambda item: int(cast(dict[str, Any], item)["seed"])):
        row = cast(dict[str, Any], metric)
        lines.append(
            f"| {row['seed']} | {row['DET_l']:.6f} | {row['DET_t']:.6f} | "
            f"{row['TOP_ll']:.6f} | {row['TOP_lt']:.6f} | "
            f"{row['negative_log_likelihood']:.6f} |"
        )
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(("", "## Measured failed examples", ""))
    lines.extend(
        f"- `{item['frame_key']}` - `{item['reason_code']}` - {item['summary']}"
        for item in failed_examples
    )
    lines.extend(
        (
            "",
            "These failures are measured examples, not a claim that the list covers every "
            "failure mode.",
            "",
        )
    )
    _write_once(output_root / "model-manifest.json", model_manifest_bytes)
    _write_once(output_root / "MODEL_CARD.md", "\n".join(lines).encode("utf-8"))
    return {
        "schema_version": "junctionlens.e0-finalization-receipt.v1",
        "state": "ACCEPTED",
        "model_manifest_sha256": hashlib.sha256(model_manifest_bytes).hexdigest(),
        "model_card_sha256": _hash(output_root / "MODEL_CARD.md"),
    }


__all__ = ["E0ArtifactError", "finalize_e0_artifacts"]

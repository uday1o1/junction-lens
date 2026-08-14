"""Fail-closed OpenLane adapter parity against the pinned official devkit."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from platformdirs import user_cache_path

from junctionlens.data.contracts import AdaptedFrame
from junctionlens.data.geometry import (
    OPENLANE_TO_JUNCTIONLENS,
    invert_transform,
    junctionlens_points_to_openlane,
)
from junctionlens.data.openlane import OpenLaneAdapter, OpenLaneAdapterError
from junctionlens.evaluator.official import (
    inspect_evaluator_image,
    load_evaluator_image_contract,
)

MAX_OFFICIAL_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_PARITY_FRAMES = 32
PARITY_TOLERANCE = 1e-9


class AdapterParityError(RuntimeError):
    """Raised when official projection or adapter parity cannot be trusted."""


def load_parity_selector(path: Path) -> tuple[tuple[str, str, str], ...]:
    """Load one bounded, versioned list of frozen source frame identifiers."""
    with path.open(encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict) or set(value) != {"frames", "schema_version"}:
        raise AdapterParityError("parity selector has invalid top-level keys")
    if value["schema_version"] != "junctionlens.openlane-parity-selector.v1":
        raise AdapterParityError("parity selector schema is unsupported")
    frames = value["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_PARITY_FRAMES:
        raise AdapterParityError("parity selector must contain between one and 32 frames")
    result: list[tuple[str, str, str]] = []
    for index, item in enumerate(frames):
        if not isinstance(item, dict) or set(item) != {"segment_id", "split_id", "timestamp"}:
            raise AdapterParityError(f"parity selector frame {index} has invalid keys")
        identifier = tuple(str(item[key]) for key in ("split_id", "segment_id", "timestamp"))
        if any(not part or Path(part).name != part for part in identifier):
            raise AdapterParityError(f"parity selector frame {index} has an unsafe identifier")
        try:
            int(identifier[2])
        except ValueError as error:
            raise AdapterParityError(
                f"parity selector frame {index} has a nonnumeric timestamp"
            ) from error
        result.append(cast(tuple[str, str, str], identifier))
    if len(result) != len(set(result)):
        raise AdapterParityError("parity selector contains duplicate frames")
    return tuple(result)


def _matrix(value: object) -> list[list[float]]:
    return cast(list[list[float]], np.asarray(value, dtype=np.float64).tolist())


def _points(value: object) -> list[list[float]]:
    return cast(list[list[float]], junctionlens_points_to_openlane(value).tolist())


def adapted_source_projection(frame: AdaptedFrame) -> dict[str, Any]:
    """Invert canonical normalization into the fields exposed by official Frame."""
    basis_inverse = invert_transform(OPENLANE_TO_JUNCTIONLENS)
    source_cameras: dict[str, Any] = {}
    for camera in frame.cameras:
        if not camera.valid:
            continue
        if camera.source_camera is None or camera.image_relative_path is None:
            raise AdapterParityError("valid camera lacks source identity")
        source_extrinsic = basis_inverse @ np.asarray(camera.t_vehicle_camera, dtype=np.float64)
        source_cameras[camera.source_camera] = {
            "image_path": camera.image_relative_path,
            "intrinsic": {
                "K": _matrix(camera.intrinsic),
                "distortion": list(camera.distortion_coefficients),
                "model": camera.distortion_model,
            },
            "extrinsic": {
                "rotation": source_extrinsic[:3, :3].tolist(),
                "translation": source_extrinsic[:3, 3].tolist(),
            },
        }
    pose = (
        basis_inverse
        @ np.asarray(frame.t_world_vehicle, dtype=np.float64)
        @ np.asarray(OPENLANE_TO_JUNCTIONLENS, dtype=np.float64)
    )
    annotation: dict[str, Any] | None
    if not frame.annotations_valid:
        annotation = None
    else:
        annotation = {
            "lane_segment": [
                {
                    "id": lane.source_object_id,
                    "centerline": _points(lane.centerline),
                    "left_laneline": _points(lane.left_boundary),
                    "right_laneline": _points(lane.right_boundary),
                    "left_laneline_type": lane.left_boundary_type,
                    "right_laneline_type": lane.right_boundary_type,
                    "is_intersection_or_connector": lane.is_intersection_or_connector,
                }
                for lane in frame.lanes
            ],
            "traffic_element": [
                {
                    "id": control.source_object_id,
                    "category": control.category,
                    "attribute": control.attribute,
                    "points": [list(point) for point in control.source_pixel_box.points],
                }
                for control in frame.traffic_controls
            ],
            "area": [
                {
                    "id": area.source_object_id,
                    "category": area.category,
                    "points": _points(area.points),
                }
                for area in frame.road_areas
            ],
            "topology_lsls": [list(row) for row in frame.topology_lane_lane],
            "topology_lste": [list(row) for row in frame.topology_lane_traffic],
        }
    return {
        "identifier": {
            "split_id": frame.key.split_id,
            "segment_id": frame.key.segment_id,
            "timestamp": str(frame.key.timestamp_ns),
        },
        "metadata_version": frame.source_metadata.metadata_version,
        "segment_id": frame.key.segment_id,
        "timestamp": frame.key.timestamp_ns,
        "source_name": frame.source_metadata.source_name,
        "source_segment_id": frame.source_metadata.source_segment_id,
        "cameras": source_cameras,
        "pose": {
            "rotation": pose[:3, :3].tolist(),
            "translation": pose[:3, 3].tolist(),
        },
        "annotation": annotation,
    }


def _docker() -> str:
    executable = shutil.which("docker")
    if executable is None:
        raise AdapterParityError("Docker CLI is unavailable")
    return executable


def _selector_payload(identifiers: Sequence[tuple[str, str, str]]) -> bytes:
    value = {
        "schema_version": "junctionlens.openlane-parity-selector.v1",
        "frames": [
            {"split_id": split, "segment_id": segment, "timestamp": timestamp}
            for split, segment, timestamp in identifiers
        ],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def run_official_projection(
    dataset_root: Path,
    identifiers: Sequence[tuple[str, str, str]],
    project_root: Path,
) -> Mapping[str, Any]:
    """Run official Frame getters in the locked networkless compatibility image."""
    project_root = project_root.resolve(strict=True)
    dataset_root = dataset_root.resolve(strict=True)
    contract = load_evaluator_image_contract(project_root)
    reference = str(contract["local_reference"])
    inspect_evaluator_image(
        project_root,
        reference,
        str(contract["config_sha256"]),
        str(contract["platform_manifest_sha256"]),
    )
    runner = (project_root / "containers/adapter_parity_runner.py").resolve(strict=True)
    staging_override = os.environ.get("JUNCTIONLENS_DOCKER_STAGING_ROOT")
    cache_root = (
        Path(staging_override).expanduser()
        if staging_override is not None
        else user_cache_path("junctionlens") / "adapter-parity"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise AdapterParityError("Docker staging root must be a real directory")
    with tempfile.TemporaryDirectory(prefix="request-", dir=cache_root) as temp:
        selector = Path(temp) / "selector.json"
        selector.write_bytes(_selector_payload(identifiers))
        command = [
            _docker(),
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "1g",
            "--cpus",
            "2",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",  # noqa: S108 - isolated tmpfs
            "--user",
            "65532:65532",
            "--mount",
            f"type=bind,src={dataset_root},dst=/dataset,readonly",
            "--mount",
            f"type=bind,src={selector},dst=/input/selector.json,readonly",
            "--mount",
            f"type=bind,src={runner},dst=/runner/adapter_parity_runner.py,readonly",
            "--entrypoint",
            "python",
            reference,
            "/runner/adapter_parity_runner.py",
            "/dataset",
            "/input/selector.json",
        ]
        result = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        raise AdapterParityError(
            f"official adapter projection failed with exit code {result.returncode}: "
            f"{result.stderr[:2048].strip()}"
        )
    if len(result.stdout.encode("utf-8")) > MAX_OFFICIAL_OUTPUT_BYTES:
        raise AdapterParityError("official adapter projection exceeds the output byte limit")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise AdapterParityError("official adapter projection returned invalid JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"devkit_version", "frames", "schema_version"}
        or value["schema_version"] != "junctionlens.openlane-official-projection.v1"
        or value["devkit_version"] != "2.1.0"
        or not isinstance(value["frames"], list)
        or len(value["frames"]) != len(identifiers)
    ):
        raise AdapterParityError("official adapter projection has an invalid contract")
    return cast(Mapping[str, Any], value)


def _compare(expected: object, observed: object, path: str = "frame") -> float:
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected is not observed:
            raise AdapterParityError(f"official adapter parity differs at {path}")
        return 0.0
    if isinstance(expected, int | float) and isinstance(observed, int | float):
        difference = abs(float(expected) - float(observed))
        if not np.isfinite(difference) or difference > PARITY_TOLERANCE:
            raise AdapterParityError(
                f"official adapter parity differs at {path}: absolute error {difference}"
            )
        return difference
    if isinstance(expected, dict) and isinstance(observed, dict):
        if set(expected) != set(observed):
            raise AdapterParityError(f"official adapter parity keys differ at {path}")
        return max(
            (_compare(expected[key], observed[key], f"{path}.{key}") for key in expected),
            default=0.0,
        )
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            raise AdapterParityError(f"official adapter parity length differs at {path}")
        return max(
            (
                _compare(expected_item, observed_item, f"{path}[{index}]")
                for index, (expected_item, observed_item) in enumerate(
                    zip(expected, observed, strict=True)
                )
            ),
            default=0.0,
        )
    if expected != observed:
        raise AdapterParityError(f"official adapter parity differs at {path}")
    return 0.0


def verify_official_parity(
    adapter: OpenLaneAdapter,
    selector_path: Path,
    project_root: Path,
) -> Mapping[str, Any]:
    """Compare every selected normalized field with the official devkit projection."""
    identifiers = load_parity_selector(selector_path)
    official = run_official_projection(adapter.root, identifiers, project_root)
    official_frames = cast(list[object], official["frames"])
    maximum_error = 0.0
    frame_hashes: list[str] = []
    for identifier, official_frame in zip(identifiers, official_frames, strict=True):
        try:
            adapted_frame = adapter.load_frame(*identifier)
        except OpenLaneAdapterError as error:
            raise AdapterParityError(str(error)) from error
        adapted = adapted_source_projection(adapted_frame)
        maximum_error = max(maximum_error, _compare(official_frame, adapted))
        canonical = json.dumps(
            official_frame, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        frame_hashes.append(hashlib.sha256(canonical).hexdigest())
    evidence = json.dumps(frame_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "junctionlens.openlane-adapter-parity-report.v1",
        "state": "ACCEPTED",
        "devkit_version": "2.1.0",
        "frame_count": len(identifiers),
        "maximum_absolute_numeric_error": maximum_error,
        "tolerance": PARITY_TOLERANCE,
        "official_projection_set_sha256": hashlib.sha256(evidence).hexdigest(),
    }


__all__ = [
    "AdapterParityError",
    "adapted_source_projection",
    "load_parity_selector",
    "run_official_projection",
    "verify_official_parity",
]

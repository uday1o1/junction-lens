"""Content-addressed persistence for the frozen synthetic graph corpus."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from junctionlens.contract import to_binary, to_json
from junctionlens.synthetic.generator import generate_corruptions, generate_scene_frames
from junctionlens.synthetic.models import scene_specs


class SyntheticCorpusError(ValueError):
    """Raised when generated corpus persistence or verification fails closed."""


@dataclass(frozen=True, slots=True)
class SyntheticCorpus:
    """One immutable in-memory set of relative paths and exact file bytes."""

    seed: int
    files: Mapping[str, bytes]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(
    path: str,
    payload: bytes,
    *,
    media_type: str,
    scene: str,
    frame_index: int,
    variant: str,
) -> dict[str, object]:
    return {
        "byte_size": len(payload),
        "frame_index": frame_index,
        "media_type": media_type,
        "path": path,
        "scene": scene,
        "sha256": _sha256(payload),
        "variant": variant,
    }


def generate_corpus(seed: int = 20_260_813) -> SyntheticCorpus:
    """Generate the complete V1 corpus and its self-describing manifest."""
    frames = generate_scene_frames(seed)
    corruptions = generate_corruptions(frames)
    files: dict[str, bytes] = {}
    records: list[dict[str, object]] = []

    def add_file(
        path: str,
        payload: bytes,
        *,
        media_type: str,
        scene: str,
        frame_index: int,
        variant: str,
    ) -> None:
        if path in files and files[path] != payload:
            raise SyntheticCorpusError(f"generated path collision with different bytes: {path}")
        if path in files:
            return
        files[path] = payload
        records.append(
            _record(
                path,
                payload,
                media_type=media_type,
                scene=scene,
                frame_index=frame_index,
                variant=variant,
            )
        )

    for frame in frames:
        scene = frame.scene_kind.value
        stem = f"graphs/{scene}/frame-{frame.frame_index:02d}"
        for variant, envelope in (
            ("ground-truth", frame.ground_truth),
            ("perfect", frame.perfect_prediction),
        ):
            add_file(
                f"{stem}.{variant}.pb",
                to_binary(envelope),
                media_type="application/x-protobuf",
                scene=scene,
                frame_index=frame.frame_index,
                variant=variant,
            )
            add_file(
                f"{stem}.{variant}.json",
                to_json(envelope).encode("utf-8"),
                media_type="application/json",
                scene=scene,
                frame_index=frame.frame_index,
                variant=variant,
            )
        for path, image in frame.camera_images.items():
            add_file(
                path,
                image,
                media_type="image/svg+xml",
                scene=scene,
                frame_index=frame.frame_index,
                variant="source-rendering",
            )

    for corruption in corruptions:
        scene = corruption.scene_kind.value
        variant = f"corrupt-{corruption.corruption.value}"
        stem = f"graphs/{scene}/frame-{corruption.frame_index:02d}.{variant}"
        add_file(
            f"{stem}.pb",
            to_binary(corruption.prediction),
            media_type="application/x-protobuf",
            scene=scene,
            frame_index=corruption.frame_index,
            variant=variant,
        )
        add_file(
            f"{stem}.json",
            to_json(corruption.prediction).encode("utf-8"),
            media_type="application/json",
            scene=scene,
            frame_index=corruption.frame_index,
            variant=variant,
        )

    shape_scenes: dict[str, list[str]] = {}
    for specification in scene_specs():
        for shape in specification.mandatory_shapes:
            shape_scenes.setdefault(shape, []).append(specification.kind.value)
    manifest = {
        "configuration_sha256": frames[0].ground_truth.producer.configuration_sha256,
        "corruptions": [corruption.corruption.value for corruption in corruptions],
        "content_file_count": len(files),
        "file_count": len(files) + 1,
        "files": sorted(records, key=lambda record: str(record["path"])),
        "generator": "junctionlens-synthetic-generator-v1",
        "mandatory_shapes": {key: sorted(value) for key, value in sorted(shape_scenes.items())},
        "schema": "junctionlens.synthetic-corpus/v1",
        "seed": seed,
        "temporal_sequences": [
            {
                "ego_translation_x_m": [0.0, 2.0],
                "frame_indices": [0, 1],
                "scene": "straight-control",
                "timestamps_ns": [
                    frame.ground_truth.graph.frame_key.timestamp_ns
                    for frame in frames
                    if frame.scene_kind.value == "straight-control"
                ],
            }
        ],
    }
    files["manifest.json"] = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return SyntheticCorpus(seed=seed, files=MappingProxyType(dict(sorted(files.items()))))


def _relative_files(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SyntheticCorpusError(f"synthetic corpus must not contain symlinks: {path}")
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
    return result


def verify_corpus(root: Path, *, seed: int = 20_260_813) -> SyntheticCorpus:
    """Verify that a persisted corpus exactly matches regenerated bytes and paths."""
    if root.is_symlink() or not root.is_dir():
        raise SyntheticCorpusError(f"synthetic corpus root is not a regular directory: {root}")
    expected = generate_corpus(seed)
    observed_paths = _relative_files(root)
    expected_paths = set(expected.files)
    missing = sorted(expected_paths - observed_paths)
    unexpected = sorted(observed_paths - expected_paths)
    if missing or unexpected:
        raise SyntheticCorpusError(
            f"synthetic corpus path mismatch: missing={missing}, unexpected={unexpected}"
        )
    for relative_path, payload in expected.files.items():
        observed = (root / relative_path).read_bytes()
        if observed != payload:
            raise SyntheticCorpusError(
                f"synthetic corpus byte mismatch for {relative_path}: "
                f"expected_sha256={_sha256(payload)}, observed_sha256={_sha256(observed)}"
            )
    return expected


def _atomic_write(root: Path, path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = root
    for part in path.relative_to(root).parts[:-1]:
        current /= part
        if current.is_symlink():
            raise SyntheticCorpusError(f"refusing to write through a symlink: {current}")
    if path.is_symlink():
        raise SyntheticCorpusError(f"refusing to replace a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_corpus(root: Path, *, seed: int = 20_260_813) -> SyntheticCorpus:
    """Write a corpus atomically while refusing stale or unexplained files."""
    if root.is_symlink():
        raise SyntheticCorpusError(f"refusing a symlinked synthetic corpus root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    corpus = generate_corpus(seed)
    observed_paths = _relative_files(root)
    unexpected = sorted(observed_paths - set(corpus.files))
    if unexpected:
        raise SyntheticCorpusError(
            f"refusing to overwrite a directory with stale files: {unexpected}"
        )
    for relative_path, payload in corpus.files.items():
        target = root / relative_path
        if target.is_file() and target.read_bytes() == payload:
            continue
        _atomic_write(root, target, payload)
    verify_corpus(root, seed=seed)
    return corpus

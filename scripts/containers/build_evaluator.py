#!/usr/bin/env python3
"""Build and qualify the reproducible official evaluator OCI image."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PYTHON_SOURCE = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from junctionlens.bootstrap import download_verified  # noqa: E402

SOURCE_DATE_EPOCH = "1698810385"
LOCAL_REFERENCE = "junctionlens/official-evaluator:v2.1.0"


class EvaluatorBuildError(RuntimeError):
    """Raised when image construction or qualification fails."""


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as source:
        payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise EvaluatorBuildError(f"expected a mapping in {path}")
    return payload


def _run(command: list[str], root: Path, timeout: int = 3_600) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise EvaluatorBuildError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout[-4096:]}\n{result.stderr[-4096:]}"
        )
    return result.stdout


def _docker() -> str:
    executable = shutil.which("docker")
    if executable is None:
        raise EvaluatorBuildError("Docker CLI is unavailable")
    return executable


def _prepare_context(root: Path) -> Path:
    data_lock = _yaml(root / "configs/data/openlane-v2-v2.1.lock.yaml")
    devkit = data_lock["devkit"]
    source = download_verified(
        str(devkit["source_archive"]),
        str(devkit["source_sha256"]),
        root / ".cache/sources/openlane-v2-v2.1.0.tar.gz",
    )
    context = root / ".cache/evaluator-build/context"
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    copies = {
        root / "containers/Containerfile.evaluator": context / "Dockerfile",
        root / "containers/evaluator-requirements.lock": context / "evaluator-requirements.lock",
        root / "containers/evaluator_runner.py": context / "evaluator_runner.py",
        root / "python/junctionlens/evaluator/payload.py": context / "evaluator_payload.py",
        source: context / "openlane-v2-v2.1.0.tar.gz",
    }
    for source_path, destination in copies.items():
        shutil.copyfile(source_path, destination)
    return context


def _context_sha256(context: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in context.rglob("*") if candidate.is_file()):
        relative = path.relative_to(context).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _read_oci(path: Path) -> dict[str, str]:
    with tarfile.open(path, mode="r:") as archive:
        members = {member.name: member for member in archive.getmembers()}
        index_member = members.get("index.json")
        if index_member is None or not index_member.isfile():
            raise EvaluatorBuildError("OCI export has no regular index.json")
        extracted = archive.extractfile(index_member)
        if extracted is None:
            raise EvaluatorBuildError("OCI index could not be read")
        index_bytes = extracted.read()
        index = json.loads(index_bytes)
        manifests = index.get("manifests", [])
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise EvaluatorBuildError("OCI index must contain exactly one platform manifest")
        manifest_digest = manifests[0].get("digest", "")
        if not isinstance(manifest_digest, str) or not manifest_digest.startswith("sha256:"):
            raise EvaluatorBuildError("OCI platform manifest digest is invalid")
        manifest_name = f"blobs/sha256/{manifest_digest.removeprefix('sha256:')}"
        manifest_member = members.get(manifest_name)
        if manifest_member is None or not manifest_member.isfile():
            raise EvaluatorBuildError("OCI platform manifest blob is absent")
        manifest_file = archive.extractfile(manifest_member)
        if manifest_file is None:
            raise EvaluatorBuildError("OCI platform manifest could not be read")
        manifest_bytes = manifest_file.read()
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_digest.removeprefix("sha256:"):
            raise EvaluatorBuildError("OCI platform manifest content hash does not match")
        manifest = json.loads(manifest_bytes)
        config_digest = manifest.get("config", {}).get("digest", "")
        if not isinstance(config_digest, str) or not config_digest.startswith("sha256:"):
            raise EvaluatorBuildError("OCI config digest is invalid")
    return {
        "config_sha256": config_digest.removeprefix("sha256:"),
        "oci_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "platform_manifest_sha256": manifest_digest.removeprefix("sha256:"),
    }


def _build_export(root: Path, context: Path, destination: Path, *, no_cache: bool) -> None:
    if destination.exists():
        destination.unlink()
    command = [
        str(root / ".tools/bin/docker-buildx"),
        "build",
        "--platform",
        "linux/amd64",
        "--provenance=false",
        "--sbom=false",
        "--build-arg",
        f"SOURCE_DATE_EPOCH={SOURCE_DATE_EPOCH}",
        "--output",
        f"type=oci,dest={destination},oci-mediatypes=true,rewrite-timestamp=true",
    ]
    if no_cache:
        command.append("--no-cache")
    command.append(str(context))
    _run(command, root)


def _load_image(root: Path, archive: Path, identity: Mapping[str, str]) -> None:
    _run([_docker(), "load", "--input", str(archive)], root)
    _run(
        [
            _docker(),
            "tag",
            f"sha256:{identity['platform_manifest_sha256']}",
            LOCAL_REFERENCE,
        ],
        root,
    )


def qualify(check_lock: bool) -> dict[str, Any]:
    """Build twice independently, compare OCI identities, load, and optionally check the lock."""
    root = _root()
    buildx = root / ".tools/bin/docker-buildx"
    if not buildx.is_file():
        raise EvaluatorBuildError("run ./tools/jl bootstrap-cpu before building containers")
    context = _prepare_context(root)
    context_sha256 = _context_sha256(context)
    output_root = root / ".cache/evaluator-build"
    first = output_root / "first.oci.tar"
    second = output_root / "second.oci.tar"
    _build_export(root, context, first, no_cache=True)
    first_identity = _read_oci(first)
    _build_export(root, context, second, no_cache=True)
    second_identity = _read_oci(second)
    if first_identity != second_identity:
        raise EvaluatorBuildError(
            f"independent OCI builds differ: {first_identity!r} != {second_identity!r}"
        )
    _load_image(root, first, first_identity)
    inspect = _run(
        [_docker(), "image", "inspect", "--format", "{{.Id}}", LOCAL_REFERENCE], root
    ).strip()
    accepted_ids = {
        f"sha256:{first_identity['config_sha256']}",
        f"sha256:{first_identity['platform_manifest_sha256']}",
    }
    if inspect not in accepted_ids:
        raise EvaluatorBuildError("loaded image identity differs from the OCI export")
    result: dict[str, Any] = {
        **first_identity,
        "build_context_sha256": context_sha256,
        "local_reference": LOCAL_REFERENCE,
        "source_date_epoch": int(SOURCE_DATE_EPOCH),
        "state": "ACCEPTED_LOCAL",
    }
    if check_lock:
        locked = _yaml(root / "containers/images.lock")["application_images"].get(
            "official_evaluator"
        )
        if locked != result:
            raise EvaluatorBuildError(f"evaluator image lock differs; observed {result!r}")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-lock", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(qualify(_arguments().check_lock), sort_keys=True))
    except (EvaluatorBuildError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"evaluator build error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

"""Offline and online verification for immutable dependency locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from junctionlens.bootstrap import download_verified

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_FETCH_BLOCK = re.compile(
    r"FetchContent_Declare\(\s*(?P<name>[a-zA-Z0-9_]+)\s+"
    r"URL\s+(?P<url>https://\S+)\s+"
    r"URL_HASH\s+SHA256=(?P<sha256>[0-9a-f]{64})",
    re.MULTILINE,
)
_OFFICIAL_DATASET_MD5 = {
    "openlane-v2-sample": "21c607fa5a1930275b7f1409b25042a0",
    "subset-a-metadata": "95bf28ccf22583d20434d75800be065d",
    "subset-a-map-element-bucket": "1c1f9d49ecd47d6bc5bf093f38fb68c9",
    "subset-a-image-0": "8ade7daeec1b64f8ab91a50c81d812f6",
    "subset-a-image-1": "c78e776f79e2394d2d5d95b7b5985e0f",
    "subset-a-image-2": "4bf09079144aa54cb4dcd5ff6e00cf79",
    "subset-a-image-3": "fd9e64345445975f462213b209632aee",
    "subset-a-image-4": "ae07e48c88ea2c3f6afbdf5ff71e9821",
    "subset-a-image-5": "df62c1f6e6b3fb2a2a0868c78ab19c92",
    "subset-a-image-6": "7bff1ce30329235f8e0f25f6f6653b8f",
    "subset-a-image-7": "c73af4a7aef2692b96e4e00795120504",
    "subset-a-image-8": "fb2f61e7309e0b48e2697e085a66a259",
    "subset-a-sd-map": "de22c7be880b667f1b3373ff665aac2e",
    "subset-b-metadata": "27696b1ed1d99b1f70fdb68f439dc87d",
}


class LockVerificationError(RuntimeError):
    """Raised when a lock is incomplete, mutable, or mismatched."""


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise LockVerificationError(f"expected an object in {path}")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise LockVerificationError(f"expected a mapping in {path}")
    return value


def _require_hash(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise LockVerificationError(f"{label} is not a valid lowercase hash")
    return value


def _validate_toolchains(payload: Mapping[str, Any]) -> None:
    tools = payload.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise LockVerificationError("toolchain lock has no tools")
    for name, raw_spec in tools.items():
        if not isinstance(raw_spec, dict):
            raise LockVerificationError(f"tool {name} is not an object")
        version = raw_spec.get("version")
        if (
            not isinstance(version, str)
            or not version
            or any(token in version for token in "*^~><")
        ):
            raise LockVerificationError(f"tool {name} does not have an exact version")
        assets = raw_spec.get("assets")
        if not isinstance(assets, dict) or set(assets) != {"darwin-arm64", "linux-x86_64"}:
            raise LockVerificationError(f"tool {name} must lock both supported platforms")
        for platform_name, raw_asset in assets.items():
            if not isinstance(raw_asset, dict):
                raise LockVerificationError(f"tool {name} asset {platform_name} is invalid")
            _require_hash(raw_asset.get("sha256"), _SHA256, f"{name}/{platform_name} sha256")
            url = raw_asset.get("url")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise LockVerificationError(f"tool {name} asset {platform_name} has an invalid URL")


def _validate_dataset(payload: Mapping[str, Any]) -> None:
    devkit = payload.get("devkit")
    if not isinstance(devkit, dict):
        raise LockVerificationError("dataset devkit lock is missing")
    if devkit.get("commit") != "d731a26bdbf34723dd915ad525c2c2eca19ed8a1":
        raise LockVerificationError("OpenLane-V2 commit does not match v2.1.0")
    _require_hash(devkit.get("source_sha256"), _SHA256, "OpenLane-V2 source sha256")
    if payload.get("acknowledgment_required") is not True:
        raise LockVerificationError("dataset terms acknowledgment must be required")
    if payload.get("redistribution_allowed") is not False:
        raise LockVerificationError("dataset redistribution must default to false")
    archives = payload.get("archives")
    if not isinstance(archives, list):
        raise LockVerificationError("dataset archive list is missing")
    observed = {}
    for raw_archive in archives:
        if not isinstance(raw_archive, dict):
            raise LockVerificationError("dataset archive entry is invalid")
        name = raw_archive.get("name")
        if not isinstance(name, str):
            raise LockVerificationError("dataset archive name is invalid")
        observed[name] = _require_hash(raw_archive.get("published_md5"), _MD5, f"{name} md5")
    if observed != _OFFICIAL_DATASET_MD5:
        raise LockVerificationError("dataset MD5 set differs from the pinned official data page")


def _validate_adapter_lock(root: Path, dataset: Mapping[str, Any]) -> None:
    adapter_lock = dataset.get("adapter")
    if not isinstance(adapter_lock, dict) or set(adapter_lock) != {
        "preprocessing_config_sha256",
        "version",
    }:
        raise LockVerificationError("dataset adapter lock is incomplete")
    config_path = root / "configs/data/openlane-v2-v2.1.adapter.yaml"
    config_bytes = config_path.read_bytes()
    observed_sha256 = hashlib.sha256(config_bytes).hexdigest()
    locked_sha256 = _require_hash(
        adapter_lock.get("preprocessing_config_sha256"),
        _SHA256,
        "OpenLane adapter config sha256",
    )
    if observed_sha256 != locked_sha256:
        raise LockVerificationError("OpenLane adapter config differs from its dataset lock")
    config = _load_yaml(config_path)
    if config.get("adapter_version") != adapter_lock.get("version"):
        raise LockVerificationError("OpenLane adapter version differs from its dataset lock")
    if config.get("schema_mode") != dataset.get("schema_mode"):
        raise LockVerificationError("OpenLane adapter schema mode differs from its dataset lock")
    if config.get("source_camera_mappings") != dataset.get("camera_slot_mappings"):
        raise LockVerificationError("OpenLane camera mappings differ from their dataset lock")


def _validate_split_policy_lock(root: Path, dataset: Mapping[str, Any]) -> None:
    split_lock = dataset.get("split_policy")
    if not isinstance(split_lock, dict) or set(split_lock) != {
        "committed_manifest",
        "committed_manifest_sha256",
        "config_sha256",
        "version",
    }:
        raise LockVerificationError("dataset split-policy lock is incomplete")
    if split_lock.get("version") != "v1":
        raise LockVerificationError("dataset split-policy version differs from V1")
    policy_path = root / "configs/data/openlane-v2-v2.1.split-v1.yaml"
    observed_policy = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    expected_policy = _require_hash(
        split_lock.get("config_sha256"),
        _SHA256,
        "OpenLane split-policy config sha256",
    )
    if observed_policy != expected_policy:
        raise LockVerificationError("OpenLane split policy differs from its dataset lock")
    manifest_relative = split_lock.get("committed_manifest")
    if manifest_relative != "configs/data/openlane-v2-v2.1.split-v1.json":
        raise LockVerificationError("OpenLane committed split-manifest path differs from V1")
    manifest_path = root / str(manifest_relative)
    expected_manifest = split_lock.get("committed_manifest_sha256")
    if expected_manifest is None:
        if manifest_path.exists() or manifest_path.is_symlink():
            raise LockVerificationError("OpenLane split manifest exists without a locked hash")
        return
    locked_manifest = _require_hash(
        expected_manifest,
        _SHA256,
        "OpenLane committed split-manifest sha256",
    )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise LockVerificationError("locked OpenLane split manifest is missing")
    observed_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if observed_manifest != locked_manifest:
        raise LockVerificationError("OpenLane committed split manifest differs from its lock")


def _validate_images(payload: Mapping[str, Any]) -> None:
    if payload.get("platform") != "linux/amd64":
        raise LockVerificationError("image lock must target linux/amd64")
    base_images = payload.get("base_images")
    if not isinstance(base_images, dict) or set(base_images) != {
        "cpu",
        "official_evaluator",
        "gpu",
    }:
        raise LockVerificationError("image lock does not contain every required base")
    for name, raw_image in base_images.items():
        if not isinstance(raw_image, dict):
            raise LockVerificationError(f"image {name} is invalid")
        digest = _require_hash(
            raw_image.get("platform_manifest_sha256"),
            _SHA256,
            f"image {name} platform digest",
        )
        reference = raw_image.get("reference")
        if not isinstance(reference, str) or not reference.endswith(digest):
            raise LockVerificationError(f"image {name} reference is not digest-pinned")
    runtime_sources = payload.get("runtime_sources")
    if not isinstance(runtime_sources, dict):
        raise LockVerificationError("runtime source lock is missing")
    ort = runtime_sources.get("onnxruntime")
    if not isinstance(ort, dict) or set(ort.get("submodules", {})) != {
        "onnx",
        "libprotobuf-mutator",
        "emsdk",
    }:
        raise LockVerificationError("ONNX Runtime recursive source identities are incomplete")
    applications = payload.get("application_images")
    if not isinstance(applications, dict):
        raise LockVerificationError("application image lock is missing")
    evaluator = applications.get("official_evaluator")
    if not isinstance(evaluator, dict) or evaluator.get("state") != "ACCEPTED_LOCAL":
        raise LockVerificationError("official evaluator image is not accepted locally")
    for field in (
        "build_context_sha256",
        "config_sha256",
        "oci_index_sha256",
        "platform_manifest_sha256",
    ):
        _require_hash(evaluator.get(field), _SHA256, f"official evaluator {field}")
    if evaluator.get("local_reference") != "junctionlens/official-evaluator:v2.1.0":
        raise LockVerificationError("official evaluator local reference is not frozen")
    if evaluator.get("source_date_epoch") != 1698810385:
        raise LockVerificationError("official evaluator source epoch is not frozen")


def _cmake_sources(path: Path) -> list[tuple[str, str, str]]:
    source = path.read_text(encoding="utf-8")
    matches = [
        (match.group("name"), match.group("url"), match.group("sha256"))
        for match in _FETCH_BLOCK.finditer(source)
    ]
    if len(matches) != 7:
        raise LockVerificationError("expected seven hash-pinned CMake source declarations")
    return matches


def validate_lock_set(root: Path) -> None:
    """Validate every lock structurally without network access."""
    _validate_toolchains(_load_json(root / "configs/toolchains/v1.lock.json"))
    dataset = _load_yaml(root / "configs/data/openlane-v2-v2.1.lock.yaml")
    _validate_dataset(dataset)
    _validate_adapter_lock(root, dataset)
    _validate_split_policy_lock(root, dataset)
    _validate_images(_load_yaml(root / "containers/images.lock"))
    _cmake_sources(root / "cmake/dependencies.cmake")


def _run(command: Iterable[str], cwd: Path) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise LockVerificationError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[:2048]}"
        )
    return result.stdout


def _verify_tool_archives(root: Path, toolchains: Mapping[str, Any]) -> None:
    tools = toolchains["tools"]
    for name, spec in tools.items():
        for platform_name, asset in spec["assets"].items():
            suffix = {
                "binary": ".bin",
                "tar.gz": ".tar.gz",
                "tar.xz": ".tar.xz",
                "zip": ".zip",
            }[asset["archive_type"]]
            cache = (
                root
                / ".cache/lock-verification"
                / f"{name}-{spec['version']}-{platform_name}{suffix}"
            )
            download_verified(asset["url"], asset["sha256"], cache)


def _verify_source_archives(root: Path, dataset: Mapping[str, Any]) -> None:
    for name, url, sha256 in _cmake_sources(root / "cmake/dependencies.cmake"):
        download_verified(url, sha256, root / ".cache/lock-verification" / f"{name}.tar.gz")
    devkit = dataset["devkit"]
    download_verified(
        devkit["source_archive"],
        devkit["source_sha256"],
        root / ".cache/lock-verification/openlane-v2-v2.1.0.tar.gz",
    )
    output = _run(
        ["git", "ls-remote", devkit["repository"], f"refs/tags/{devkit['tag']}"],
        root,
    )
    fields = output.strip().split()
    if not fields or fields[0] != devkit["commit"]:
        raise LockVerificationError("OpenLane-V2 tag no longer resolves to the locked commit")


def _verify_image_manifests(root: Path, images: Mapping[str, Any]) -> None:
    for name, image in images["base_images"].items():
        output = _run(["docker", "manifest", "inspect", image["source_tag"], "--verbose"], root)
        payload = json.loads(output)
        manifests = payload if isinstance(payload, list) else [payload]
        matching = [
            item
            for item in manifests
            if item.get("Descriptor", {}).get("platform", {}).get("os") == "linux"
            and item.get("Descriptor", {}).get("platform", {}).get("architecture") == "amd64"
        ]
        if len(matching) != 1:
            raise LockVerificationError(f"image {name} has no unique linux/amd64 manifest")
        observed = matching[0]["Descriptor"]["digest"]
        expected = f"sha256:{image['platform_manifest_sha256']}"
        if observed != expected:
            raise LockVerificationError(
                f"image {name} digest mismatch: expected {expected}, observed {observed}"
            )


def verify_online(root: Path) -> None:
    """Download every release archive and resolve every public identity."""
    toolchains = _load_json(root / "configs/toolchains/v1.lock.json")
    dataset = _load_yaml(root / "configs/data/openlane-v2-v2.1.lock.yaml")
    images = _load_yaml(root / "containers/images.lock")
    _verify_tool_archives(root, toolchains)
    _verify_source_archives(root, dataset)
    _verify_image_manifests(root, images)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online", action="store_true", help="resolve and hash every public source"
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point for repository lock verification."""
    root = Path.cwd().resolve()
    arguments = _parse_args()
    try:
        validate_lock_set(root)
        if arguments.online:
            verify_online(root)
    except (LockVerificationError, OSError, ValueError, yaml.YAMLError) as error:
        print(json.dumps({"status": "FAILED", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "PASSED", "online": arguments.online}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

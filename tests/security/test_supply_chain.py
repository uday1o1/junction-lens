"""Determinism and seeded-control tests for supply-chain security evidence."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/security/supply_chain.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("junctionlens_supply_chain", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sbom_is_deterministic_cyclonedx_17_and_contains_all_lock_ecosystems() -> None:
    module = _module()
    first = module.generate_sbom()
    second = module.generate_sbom()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["bomFormat"] == "CycloneDX"
    assert first["specVersion"] == "1.7"
    purls = {component["purl"] for component in first["components"]}
    assert any(value.startswith("pkg:pypi/") for value in purls)
    assert any(value.startswith("pkg:npm/") for value in purls)
    assert "pkg:generic/onnxruntime@1.25.0" in purls
    assert "pkg:generic/cuda-toolkit@12.8.1" in purls


def test_secret_rules_reject_seeded_tokens_and_allow_nearby_controls() -> None:
    module = _module()
    seeded = "AK" + "IA" + "A" * 16
    private_key = "-----BEGIN " + "PRIVATE KEY-----"
    controls = ("AKIB" + "A" * 16, "-----BEGIN PUBLIC KEY-----")

    assert module._SECRET_PATTERNS["aws-access-key"].search(seeded)
    assert module._SECRET_PATTERNS["private-key"].search(private_key)
    assert not module._SECRET_PATTERNS["aws-access-key"].search(controls[0])
    assert not module._SECRET_PATTERNS["private-key"].search(controls[1])


def test_advisory_exceptions_are_expiring_and_controlled() -> None:
    module = _module()
    exceptions = module.advisory_exceptions()

    assert {item["advisory_id"] for item in exceptions} == {
        "PYSEC-2025-194",
        "PYSEC-2026-1805",
        "PYSEC-2026-3447",
    }
    assert all(item["controls"] for item in exceptions)


def test_native_license_inventory_matches_locked_build_versions() -> None:
    module = _module()
    inventory = module.load_yaml_object(
        (ROOT / "configs/security/native-components.yaml").read_bytes(),
        "native inventory",
    )
    image_lock = module.load_yaml_object(
        (ROOT / "containers/images.lock").read_bytes(),
        "container image lock",
    )
    cmake = (ROOT / "cmake/dependencies.cmake").read_text(encoding="utf-8")

    def cmake_version(variable: str) -> str:
        match = re.search(rf'^set\({variable} "([^"]+)"\)$', cmake, re.MULTILINE)
        assert match is not None
        return match.group(1)

    versions = {item["name"]: item["version"] for item in inventory["components"]}
    expected = {
        "abseil-cpp": cmake_version("JUNCTIONLENS_ABSEIL_VERSION"),
        "cuda-toolkit": image_lock["runtime_sources"]["cuda"]["version"],
        "cudnn": image_lock["runtime_sources"]["cudnn"]["version"],
        "eigen": cmake_version("JUNCTIONLENS_EIGEN_VERSION"),
        "googletest": cmake_version("JUNCTIONLENS_GOOGLETEST_VERSION"),
        "onnxruntime": cmake_version("JUNCTIONLENS_ONNXRUNTIME_VERSION"),
        "opencv": cmake_version("JUNCTIONLENS_OPENCV_VERSION"),
        "protobuf": cmake_version("JUNCTIONLENS_PROTOBUF_RUNTIME_VERSION"),
        "tensorrt": image_lock["runtime_sources"]["tensorrt"]["version"],
    }
    assert versions == expected

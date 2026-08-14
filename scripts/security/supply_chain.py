#!/usr/bin/env python3
"""Generate deterministic supply-chain evidence and enforce repository security gates."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from junctionlens.security.parsing import (  # noqa: E402
    load_json,
    load_json_object,
    load_yaml_object,
)

_FORBIDDEN_LICENSES = re.compile(r"(?i)(?:^|\W)(?:AGPL|SSPL|GPL)(?:-|\W|$)")
_SECRET_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github-token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{70,255})\b"
    ),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "slack-token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "credential-url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@[^\s]+"),
}


class SupplyChainError(RuntimeError):
    """A stable fail-closed supply-chain gate failure."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
    )


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _command(
    arguments: list[str], *, allowed: frozenset[int] = frozenset({0})
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode not in allowed:
        detail = (completed.stderr or completed.stdout).strip()
        raise SupplyChainError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}: {detail}"
        )
    return completed


def _tool(name: str) -> str:
    candidates = (ROOT / ".tools/bin" / name, Path(sys.executable).parent / name)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    resolved = shutil.which(name)
    if resolved is None:
        raise SupplyChainError(f"required tool is unavailable: {name}")
    return resolved


def advisory_exceptions() -> list[dict[str, object]]:
    """Load time-limited, controlled vulnerability exceptions from the committed policy."""
    policy = load_yaml_object(
        (ROOT / "configs/security/advisory-exceptions.yaml").read_bytes(),
        "advisory exception policy",
    )
    values = policy.get("exceptions")
    if not isinstance(values, list):
        raise SupplyChainError("advisory exception policy has no exception list")
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    today = dt.date.today()
    for value in values:
        if not isinstance(value, dict):
            raise SupplyChainError("advisory exception must be an object")
        advisory_id = value.get("advisory_id")
        package = value.get("package")
        version = value.get("version")
        expires = value.get("expires")
        rationale = value.get("rationale")
        controls = value.get("controls")
        if not all(isinstance(item, str) and item for item in (advisory_id, package, version)):
            raise SupplyChainError("advisory exception identity is incomplete")
        if not isinstance(expires, str):
            raise SupplyChainError(f"advisory exception {advisory_id} has no ISO date")
        try:
            expiry_date = dt.date.fromisoformat(expires)
        except ValueError as error:
            raise SupplyChainError(
                f"advisory exception {advisory_id} has an invalid ISO date"
            ) from error
        if expiry_date < today:
            raise SupplyChainError(f"advisory exception {advisory_id} expired on {expires}")
        if not isinstance(rationale, str) or len(rationale) < 80:
            raise SupplyChainError(f"advisory exception {advisory_id} has no substantive rationale")
        if not isinstance(controls, list) or not controls:
            raise SupplyChainError(f"advisory exception {advisory_id} has no controls")
        for control in controls:
            if not isinstance(control, str) or not (ROOT / control).is_file():
                raise SupplyChainError(
                    f"advisory exception {advisory_id} names a missing control: {control}"
                )
        if str(advisory_id) in seen:
            raise SupplyChainError(f"duplicate advisory exception: {advisory_id}")
        seen.add(str(advisory_id))
        results.append(
            {
                **value,
                "expires": expiry_date.isoformat(),
            }
        )
    return sorted(results, key=lambda item: str(item["advisory_id"]))


def scan_secrets() -> dict[str, object]:
    """Scan tracked and prospective tracked files with high-confidence credential patterns."""
    tracked = _command(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    ).stdout.split("\0")
    findings: list[dict[str, object]] = []
    scanned = 0
    for relative in sorted(item for item in tracked if item):
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
            continue
        payload = path.read_bytes()
        if b"\0" in payload:
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule, pattern in _SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append({"path": relative, "line": line_number, "rule": rule})
    report: dict[str, object] = {
        "schema_version": "junctionlens.secret-scan.v1",
        "scanned_files": scanned,
        "findings": findings,
        "status": "PASS" if not findings else "FAIL",
    }
    if findings:
        raise SupplyChainError(f"secret scan found {len(findings)} high-confidence finding(s)")
    return report


def _python_license(distribution: importlib.metadata.Distribution) -> str:
    name = distribution.metadata.get("Name", "")
    expression = distribution.metadata.get("License-Expression")
    if expression:
        return expression.strip()
    value = distribution.metadata.get("License")
    if value and len(value) <= 80 and "\n" not in value:
        aliases = {"Apache 2.0": "Apache-2.0", "BSD": "BSD-3-Clause"}
        return aliases.get(value.strip(), value.strip())
    classifiers = distribution.metadata.get_all("Classifier") or []
    mapping = {
        "License :: OSI Approved :: Apache Software License": "Apache-2.0",
        "License :: OSI Approved :: BSD License": "BSD-3-Clause",
        "License :: OSI Approved :: MIT License": "MIT",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
        "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    }
    for classifier in classifiers:
        if classifier in mapping:
            return mapping[classifier]
    if name.lower() == "junctionlens":
        return "Apache-2.0"
    return "UNKNOWN"


def _node_license(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict) and isinstance(value.get("type"), str):
        return str(value["type"])
    if isinstance(value, list):
        values = sorted({_node_license(item) for item in value})
        return " OR ".join(item for item in values if item != "UNKNOWN") or "UNKNOWN"
    return "UNKNOWN"


def license_inventory() -> dict[str, object]:
    """Inventory the synchronized Python, Node, and locked native component licenses."""
    components: dict[tuple[str, str, str], dict[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            record = {
                "ecosystem": "pypi",
                "name": name,
                "version": distribution.version,
                "license": _python_license(distribution),
            }
            components[("pypi", name.lower(), distribution.version)] = record
    pnpm_lock = load_yaml_object((ROOT / "pnpm-lock.yaml").read_bytes(), "pnpm lockfile")
    locked_node_packages = pnpm_lock.get("packages")
    if not isinstance(locked_node_packages, dict):
        raise SupplyChainError("pnpm lockfile has no package mapping")
    allowed_node = set()
    for key in locked_node_packages:
        if isinstance(key, str) and "@" in key:
            name, version = key.rsplit("@", 1)
            allowed_node.add((name.lower(), version))
    package_files = list((ROOT / "node_modules/.pnpm").glob("*/node_modules/*/package.json"))
    package_files.extend((ROOT / "node_modules/.pnpm").glob("*/node_modules/@*/*/package.json"))
    for path in sorted(set(package_files)):
        try:
            value = load_json_object(path.read_bytes(), f"Node package metadata {path.name}")
        except ValueError:
            continue
        name = value.get("name")
        version = value.get("version")
        if (
            isinstance(name, str)
            and isinstance(version, str)
            and (name.lower(), version) in allowed_node
        ):
            components[("npm", name.lower(), version)] = {
                "ecosystem": "npm",
                "name": name,
                "version": version,
                "license": _node_license(value.get("license") or value.get("licenses")),
            }
    native = load_yaml_object(
        (ROOT / "configs/security/native-components.yaml").read_bytes(),
        "native component inventory",
    )
    native_components = native.get("components")
    if not isinstance(native_components, list):
        raise SupplyChainError("native component inventory has no component list")
    for value in native_components:
        if not isinstance(value, dict):
            raise SupplyChainError("native component inventory contains a non-object")
        name, version, license_value = value.get("name"), value.get("version"), value.get("license")
        if not all(isinstance(item, str) for item in (name, version, license_value)):
            raise SupplyChainError("native component inventory entry is incomplete")
        components[("native", str(name).lower(), str(version))] = {
            "ecosystem": "native",
            "name": str(name),
            "version": str(version),
            "license": str(license_value),
        }
    ordered = sorted(
        components.values(),
        key=lambda item: (item["ecosystem"], item["name"].lower(), item["version"]),
    )
    unacceptable = [
        item
        for item in ordered
        if item["license"] == "UNKNOWN" or _FORBIDDEN_LICENSES.search(item["license"])
    ]
    report: dict[str, object] = {
        "schema_version": "junctionlens.license-inventory.v1",
        "component_count": len(ordered),
        "components": ordered,
        "unacceptable": unacceptable,
        "status": "PASS" if not unacceptable else "FAIL",
    }
    if unacceptable:
        names = ", ".join(f"{item['ecosystem']}:{item['name']}" for item in unacceptable[:10])
        raise SupplyChainError(f"license inventory has unacceptable entries: {names}")
    return report


def _purl(ecosystem: str, name: str, version: str) -> str:
    return f"pkg:{ecosystem}/{quote(name, safe='@/')}@{quote(version, safe='')}"


def generate_sbom() -> dict[str, object]:
    """Create a deterministic CycloneDX 1.7 SBOM from all committed dependency locks."""
    components: dict[str, dict[str, object]] = {}
    uv_lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    for package in uv_lock.get("package", []):
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        purl = _purl("pypi", name, version)
        hashes = []
        source = package.get("sdist")
        if isinstance(source, dict) and isinstance(source.get("hash"), str):
            algorithm, _, content = source["hash"].partition(":")
            if algorithm == "sha256" and content:
                hashes.append({"alg": "SHA-256", "content": content})
        components[purl] = {
            "type": "library",
            "bom-ref": purl,
            "name": name,
            "version": version,
            "purl": purl,
            **({"hashes": hashes} if hashes else {}),
        }
    pnpm_lock = load_yaml_object((ROOT / "pnpm-lock.yaml").read_bytes(), "pnpm lockfile")
    packages = pnpm_lock.get("packages")
    if not isinstance(packages, dict):
        raise SupplyChainError("pnpm lockfile has no package mapping")
    for key, value in packages.items():
        if not isinstance(key, str) or "@" not in key or not isinstance(value, dict):
            continue
        name, version = key.rsplit("@", 1)
        purl = _purl("npm", name, version)
        hashes = []
        resolution = value.get("resolution")
        if isinstance(resolution, dict) and isinstance(resolution.get("integrity"), str):
            algorithm, _, encoded = resolution["integrity"].partition("-")
            if algorithm == "sha512" and encoded:
                try:
                    content = base64.b64decode(encoded, validate=True).hex()
                except ValueError as error:
                    raise SupplyChainError(f"invalid npm integrity for {key}") from error
                hashes.append({"alg": "SHA-512", "content": content})
        components[purl] = {
            "type": "library",
            "bom-ref": purl,
            "name": name,
            "version": version,
            "purl": purl,
            **({"hashes": hashes} if hashes else {}),
        }
    native = load_yaml_object(
        (ROOT / "configs/security/native-components.yaml").read_bytes(),
        "native component inventory",
    )
    native_components = native.get("components")
    if not isinstance(native_components, list):
        raise SupplyChainError("native component inventory has no component list")
    for value in native_components:
        if not isinstance(value, dict):
            raise SupplyChainError("native component inventory contains a non-object")
        name, version, license_value = value.get("name"), value.get("version"), value.get("license")
        if not all(isinstance(item, str) and item for item in (name, version, license_value)):
            raise SupplyChainError("native component inventory entry is incomplete")
        purl = _purl("generic", str(name), str(version))
        components[purl] = {
            "type": "library",
            "bom-ref": purl,
            "name": str(name),
            "version": str(version),
            "purl": purl,
            "licenses": [{"license": {"name": str(license_value)}}],
            "properties": [{"name": "junctionlens:ecosystem", "value": "native"}],
        }
    ordered = [components[key] for key in sorted(components)]
    identity = hashlib.sha256(_canonical(ordered)).hexdigest()
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "serialNumber": (
            f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'junctionlens-sbom:' + identity)}"
        ),
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": "pkg:pypi/junctionlens@0.1.0",
                "name": "junctionlens",
                "version": "0.1.0",
                "purl": "pkg:pypi/junctionlens@0.1.0",
            },
            "properties": [
                {
                    "name": "junctionlens:uv-lock-sha256",
                    "value": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
                },
                {
                    "name": "junctionlens:pnpm-lock-sha256",
                    "value": hashlib.sha256((ROOT / "pnpm-lock.yaml").read_bytes()).hexdigest(),
                },
            ],
        },
        "components": ordered,
    }


def audit_dependencies() -> dict[str, object]:
    """Query both package advisory services and fail on high or critical vulnerabilities."""
    exceptions = advisory_exceptions()
    with tempfile.TemporaryDirectory(prefix="junctionlens-audit-") as temporary:
        requirements = Path(temporary) / "requirements.txt"
        _command(
            [
                _tool("uv"),
                "export",
                "--locked",
                "--no-dev",
                "--extra",
                "cpu",
                "--extra",
                "analytics",
                "--extra",
                "service",
                "--no-emit-project",
                "--no-header",
                "--no-annotate",
                "--output-file",
                str(requirements),
            ]
        )
        python_command = [
            _tool("pip-audit"),
            "--requirement",
            str(requirements),
            "--disable-pip",
            "--no-deps",
            "--format",
            "json",
        ]
        for exception in exceptions:
            python_command.extend(("--ignore-vuln", str(exception["advisory_id"])))
        python_result = _command(
            python_command,
            allowed=frozenset({0, 1}),
        )
        try:
            python_report = load_json(python_result.stdout.encode(), "pip-audit output")
        except ValueError as error:
            raise SupplyChainError("pip-audit did not return valid JSON") from error
        if python_result.returncode not in {0, 1}:
            raise SupplyChainError("pip-audit infrastructure failed")
    node_result = _command(
        [_tool("pnpm"), "audit", "--audit-level", "high", "--json"],
        allowed=frozenset({0, 1}),
    )
    try:
        node_report = load_json_object(node_result.stdout.encode(), "pnpm audit output")
    except ValueError as error:
        raise SupplyChainError("pnpm audit did not return valid JSON") from error
    if "error" in node_report:
        raise SupplyChainError(f"pnpm audit infrastructure failed: {node_report['error']}")
    python_vulnerabilities = []
    dependencies = (
        python_report.get("dependencies", []) if isinstance(python_report, dict) else python_report
    )
    if isinstance(dependencies, list):
        for dependency in dependencies:
            if isinstance(dependency, dict):
                for vulnerability in dependency.get("vulns", []):
                    python_vulnerabilities.append(
                        {"package": dependency.get("name"), **vulnerability}
                    )
    metadata = node_report.get("metadata", {})
    vulnerability_counts = metadata.get("vulnerabilities", {}) if isinstance(metadata, dict) else {}
    node_blocking = (
        sum(int(vulnerability_counts.get(level, 0)) for level in ("high", "critical"))
        if isinstance(vulnerability_counts, dict)
        else 0
    )
    report = {
        "schema_version": "junctionlens.dependency-audit.v1",
        "python_vulnerabilities": python_vulnerabilities,
        "documented_exceptions": exceptions,
        "node_vulnerability_counts": vulnerability_counts,
        "status": "PASS" if not python_vulnerabilities and node_blocking == 0 else "FAIL",
    }
    if report["status"] != "PASS":
        raise SupplyChainError(
            f"dependency audit found {len(python_vulnerabilities)} Python and "
            f"{node_blocking} high/critical Node vulnerability(s)"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "licenses", "sbom", "secrets", "all-local"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "audit":
            value = audit_dependencies()
        elif arguments.command == "licenses":
            value = license_inventory()
        elif arguments.command == "sbom":
            value = generate_sbom()
        elif arguments.command == "secrets":
            value = scan_secrets()
        else:
            value = {
                "schema_version": "junctionlens.local-security-evidence.v1",
                "secret_scan": scan_secrets(),
                "license_inventory": license_inventory(),
                "sbom": generate_sbom(),
                "status": "PASS",
            }
    except (OSError, SupplyChainError, subprocess.SubprocessError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    if arguments.output is not None:
        _write(arguments.output, value)
    print(_canonical(value).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

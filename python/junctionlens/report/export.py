"""Reproducible offline evidence bundles derived from immutable comparisons."""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import platform
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from junctionlens.api.models import ServiceConfig
from junctionlens.api.repository import EvidenceReadError, EvidenceRepository
from junctionlens.registry.service import EvidenceRegistry
from junctionlens.registry.store import ArtifactReceipt, RegistryError, canonical_json_bytes
from junctionlens.security.parsing import ParseBoundaryError, ParseLimits, load_json_object

_REPORT_DATA_MEDIA_TYPE = "application/vnd.junctionlens.comparison-report-data+json"
_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
_SCENE_MEDIA_TYPE = "application/vnd.junctionlens.scene-bundle+json"
_MAX_REPORT_DATA_BYTES = 32 * 1024 * 1024
_MAX_PRIVATE_IMAGE_BYTES = 8 * 1024 * 1024
_IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_REQUIRED_BUNDLE_FILES = (
    "REPORT.html",
    "REPORT.json",
    "REPORT.md",
    "commands.jsonl",
    "counterexamples.json",
    "decision.json",
    "environment.json",
    "metrics.parquet",
    "slices.parquet",
)
_PACKAGE_NAMES = ("duckdb", "junctionlens", "pyarrow", "pydantic")
_LOCK_FILES = (
    "containers/images.lock",
    "pnpm-lock.yaml",
    "uv.lock",
)


class EvidenceBundleError(RuntimeError):
    """Raised when a report would be incomplete, unsafe, or non-reproducible."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ReportCell(_StrictModel):
    cell_id: str = Field(min_length=1, max_length=256)
    metric: str = Field(min_length=1, max_length=256)
    slice: str = Field(min_length=1, max_length=512)
    status: str = Field(min_length=1, max_length=128)
    reason_code: str = Field(min_length=1, max_length=256)
    support: dict[str, int]
    point_estimate: float | None
    interval: dict[str, float | None]
    margin: float
    finite_replicates: int | None = Field(ge=0)
    invalid_replicates: int | None = Field(ge=0)
    counterexample_query: str = Field(min_length=1, max_length=4096)


class _ComparisonReportData(_StrictModel):
    schema_version: Literal["junctionlens.comparison-report-data.v1"]
    status: str = Field(min_length=1, max_length=128)
    baseline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    charter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics_table_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    slice_table_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason_codes: tuple[str, ...]
    cells: tuple[_ReportCell, ...]
    primary_hypotheses: tuple[dict[str, JsonValue], ...]
    filtering_changes_release_status: Literal[False]

    @model_validator(mode="after")
    def validate_order_and_uniqueness(self) -> _ComparisonReportData:
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("report reason codes must be sorted and unique")
        identities = tuple(cell.cell_id for cell in self.cells)
        if len(identities) != len(set(identities)):
            raise ValueError("report cells must have unique identities")
        return self


@dataclass(frozen=True, slots=True)
class EvidenceBundleReceipt:
    """Stable identities for one materialized report bundle."""

    export_mode: Literal["public", "private"]
    comparison_manifest_sha256: str
    decision_manifest_sha256: str
    bundle_manifest_sha256: str
    output_directory: str
    archive: ArtifactReceipt
    file_sha256: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        archive = asdict(self.archive)
        return {
            "schema_version": "junctionlens.report-bundle-receipt.v1",
            "export_mode": self.export_mode,
            "comparison_manifest_sha256": self.comparison_manifest_sha256,
            "decision_manifest_sha256": self.decision_manifest_sha256,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "output_directory": self.output_directory,
            **archive,
            "immutable_path": (
                f"objects/sha256/{self.archive.payload_sha256[:2]}/"
                f"{self.archive.payload_sha256[2:]}"
            ),
            "file_sha256": dict(sorted(self.file_sha256.items())),
        }


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        return load_json_object(
            payload,
            label,
            ParseLimits(
                max_bytes=_MAX_REPORT_DATA_BYTES,
                max_depth=32,
                max_nodes=500_000,
                max_container_items=100_000,
                max_string_bytes=4 * 1024 * 1024,
            ),
        )
    except ParseBoundaryError as error:
        raise EvidenceBundleError(str(error)) from error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report_data(
    repository: EvidenceRepository,
    comparison_manifest_sha256: str,
) -> tuple[_ComparisonReportData, str]:
    artifact = repository.artifact(comparison_manifest_sha256)
    if artifact.kind != "comparison" or artifact.media_type != _REPORT_DATA_MEDIA_TYPE:
        raise EvidenceBundleError("comparison is not a persisted report-data artifact")
    payload = repository.open_payload(
        comparison_manifest_sha256,
        limit=_MAX_REPORT_DATA_BYTES,
    ).read()
    try:
        report = _ComparisonReportData.model_validate(
            _strict_json(payload, "comparison report data")
        )
    except ValueError as error:
        raise EvidenceBundleError(f"comparison report-data schema is invalid: {error}") from error
    required_parents = {
        report.decision_manifest_sha256,
        report.metrics_table_manifest_sha256,
        report.slice_table_manifest_sha256,
    }
    if not required_parents.issubset(artifact.parents):
        raise EvidenceBundleError("comparison report data is missing immutable table parents")
    return report, artifact.license_id


def _verified_parquet(
    repository: EvidenceRepository,
    manifest_sha256: str,
    *,
    expected_kind: str,
    decision_manifest_sha256: str,
) -> bytes:
    artifact = repository.artifact(manifest_sha256)
    if artifact.kind != expected_kind or artifact.media_type != _PARQUET_MEDIA_TYPE:
        raise EvidenceBundleError(f"{expected_kind} parent is not registered Parquet")
    if expected_kind == "comparison" and decision_manifest_sha256 not in artifact.parents:
        raise EvidenceBundleError("metrics table does not name the decision as an immutable parent")
    return repository.open_payload(
        manifest_sha256,
        limit=repository.config.max_metric_table_bytes,
    ).read()


def _decision_bytes(
    repository: EvidenceRepository,
    report: _ComparisonReportData,
) -> tuple[dict[str, Any], bytes]:
    decision = cast(dict[str, Any], repository.decision(report.decision_manifest_sha256))
    if decision.get("status") != report.status or decision.get("cells") != [
        cell.model_dump(mode="json") for cell in report.cells
    ]:
        raise EvidenceBundleError("comparison report data differs from its persisted decision")
    if decision.get("primary_hypotheses") != list(report.primary_hypotheses):
        raise EvidenceBundleError("comparison hypotheses differ from the persisted decision")
    payload = repository.open_payload(
        report.decision_manifest_sha256,
        limit=16 * 1024 * 1024,
    ).read()
    return decision, payload


def _safe_display(value: object) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)


def _markdown(value: object) -> str:
    text = _safe_display(value).replace("\n", " ").replace("\r", " ")
    for source, replacement in (("\\", "\\\\"), ("`", "\\`"), ("|", "\\|")):
        text = text.replace(source, replacement)
    return text


def _html(value: object) -> str:
    return html.escape(_safe_display(value), quote=True)


def _report_model(
    report: _ComparisonReportData,
    *,
    mode: Literal["public", "private"],
    private_images: Sequence[Mapping[str, JsonValue]],
    counterexample_summary: Mapping[str, JsonValue],
) -> dict[str, Any]:
    return {
        "schema_version": "junctionlens.evidence-report.v1",
        "export_mode": mode,
        "title": "JunctionLens Evidence Report",
        "status": report.status,
        "reason_codes": list(report.reason_codes),
        "identities": {
            "baseline_manifest_sha256": report.baseline_manifest_sha256,
            "candidate_manifest_sha256": report.candidate_manifest_sha256,
            "charter_sha256": report.charter_sha256,
            "decision_manifest_sha256": report.decision_manifest_sha256,
            "metrics_table_manifest_sha256": report.metrics_table_manifest_sha256,
            "slice_table_manifest_sha256": report.slice_table_manifest_sha256,
        },
        "cells": [cell.model_dump(mode="json") for cell in report.cells],
        "primary_hypotheses": list(report.primary_hypotheses),
        "counterexamples": dict(counterexample_summary),
        "privacy": {
            "dataset_frames_included": bool(private_images),
            "private_paths_included": False,
            "licensed_thumbnails_opt_in": mode == "private" and bool(private_images),
        },
        "private_images": list(private_images),
        "authoritative_decision_persisted": True,
        "decision_recalculated": False,
    }


def _render_markdown(model: Mapping[str, Any]) -> bytes:
    identities = cast(Mapping[str, Any], model["identities"])
    reason_codes = cast(list[str], model["reason_codes"])
    cells = cast(list[Mapping[str, Any]], model["cells"])
    private_images = cast(list[Mapping[str, Any]], model["private_images"])
    lines = [
        "# JunctionLens Evidence Report",
        "",
        f"- Export mode: `{_markdown(model['export_mode'])}`",
        f"- Persisted release status: `{_markdown(model['status'])}`",
        "- Decision source: immutable persisted artifact",
        "- Decision recalculated during report export: `false`",
        "",
        "## Immutable identities",
        "",
    ]
    lines.extend(f"- {key}: `{_markdown(value)}`" for key, value in sorted(identities.items()))
    lines.extend(["", "## Reason codes", ""])
    if reason_codes:
        lines.extend(f"- `{_markdown(value)}`" for value in reason_codes)
    else:
        lines.append("No gate failure reason codes were persisted.")
    lines.extend(
        [
            "",
            "## Gating cells",
            "",
            "| Cell | Metric | Slice | Status | Estimate | Interval | Margin | Reason |",
            "| --- | --- | --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for cell in cells:
        interval = cast(Mapping[str, Any], cell["interval"])
        interval_text = (
            f"[{_safe_display(interval.get('lower'))}, {_safe_display(interval.get('upper'))}]"
        )
        lines.append(
            "| "
            + " | ".join(
                _markdown(value)
                for value in (
                    cell["cell_id"],
                    cell["metric"],
                    cell["slice"],
                    cell["status"],
                    cell["point_estimate"],
                    interval_text,
                    cell["margin"],
                    cell["reason_code"],
                )
            )
            + " |"
        )
    lines.extend(["", "### Counterexample queries", ""])
    lines.extend(
        f"- `{_markdown(cell['cell_id'])}`: `{_markdown(cell['counterexample_query'])}`"
        for cell in cells
    )
    lines.extend(["", "## Privacy", ""])
    if model["export_mode"] == "public":
        lines.append("Dataset frames and private filesystem paths are excluded from this bundle.")
    elif private_images:
        lines.append("Licensed thumbnails were included by explicit private-export acknowledgment.")
    else:
        lines.append("Private mode was selected, but no licensed thumbnails were requested.")
    if private_images:
        lines.extend(["", "### Private thumbnails", ""])
        for image in private_images:
            lines.append(
                f"- [{_markdown(image['label'])}]({_markdown(image['relative_path'])}) - "
                f"license `{_markdown(image['license_id'])}`"
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _render_html(model: Mapping[str, Any]) -> bytes:
    identities = cast(Mapping[str, Any], model["identities"])
    reasons = cast(list[str], model["reason_codes"])
    cells = cast(list[Mapping[str, Any]], model["cells"])
    private_images = cast(list[Mapping[str, Any]], model["private_images"])
    identity_rows = "".join(
        f"<dt>{_html(key)}</dt><dd><code>{_html(value)}</code></dd>"
        for key, value in sorted(identities.items())
    )
    reason_items = (
        "".join(f"<li><code>{_html(value)}</code></li>" for value in reasons)
        if reasons
        else "<li>No gate failure reason codes were persisted.</li>"
    )
    cell_rows = []
    query_rows = []
    for cell in cells:
        interval = cast(Mapping[str, Any], cell["interval"])
        interval_text = (
            f"[{_safe_display(interval.get('lower'))}, {_safe_display(interval.get('upper'))}]"
        )
        cell_rows.append(
            "<tr>"
            + "".join(
                f"<td>{_html(value)}</td>"
                for value in (
                    cell["cell_id"],
                    cell["metric"],
                    cell["slice"],
                    cell["status"],
                    cell["point_estimate"],
                    interval_text,
                    cell["margin"],
                    cell["reason_code"],
                )
            )
            + "</tr>"
        )
        query_rows.append(
            f"<dt><code>{_html(cell['cell_id'])}</code></dt>"
            f"<dd><code>{_html(cell['counterexample_query'])}</code></dd>"
        )
    if model["export_mode"] == "public":
        privacy = "Dataset frames and private filesystem paths are excluded from this bundle."
    elif private_images:
        privacy = "Licensed thumbnails were included by explicit private-export acknowledgment."
    else:
        privacy = "Private mode was selected, but no licensed thumbnails were requested."
    image_markup = ""
    if private_images:
        image_markup = (
            '<h3>Private thumbnails</h3><div class="images">'
            + "".join(
                "<figure>"
                f'<img src="{_html(item["relative_path"])}" alt="{_html(item["label"])}">'
                f"<figcaption>{_html(item['label'])} - license "
                f"<code>{_html(item['license_id'])}</code>"
                "</figcaption></figure>"
                for item in private_images
            )
            + "</div>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy"
 content="default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline';
 base-uri 'none'; form-action 'none'">
<title>JunctionLens Evidence Report</title>
<style>
:root {{
 color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif;
 background: #081316; color: #e7f1f2;
}}
body {{ margin: 0 auto; max-width: 1120px; padding: 2rem; line-height: 1.5; }}
header, section {{
 background: #102126; border: 1px solid #29434a; border-radius: 12px;
 margin: 0 0 1rem; padding: 1.25rem;
}}
h1, h2, h3 {{ color: #f4c95d; margin-top: 0; }}
.status {{ color: #9ee0bf; font-size: 1.25rem; font-weight: 700; }}
dl {{ display: grid; grid-template-columns: minmax(180px, 1fr) 2fr; gap: .4rem 1rem; }}
dd {{ margin: 0; min-width: 0; overflow-wrap: anywhere; }}
table {{ border-collapse: collapse; display: block; overflow-x: auto; width: 100%; }}
th, td {{
 border-bottom: 1px solid #29434a; padding: .6rem; text-align: left;
 vertical-align: top;
}}
code {{ color: #b8e1e7; overflow-wrap: anywhere; }}
.images {{
 display: grid; gap: 1rem;
 grid-template-columns: repeat(auto-fit,minmax(260px,1fr));
}}
figure {{ margin: 0; }} img {{ height: auto; max-width: 100%; }}
@media (max-width: 620px) {{ body {{ padding: .75rem; }} dl {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header><p>Immutable comparison evidence</p><h1>JunctionLens Evidence Report</h1>
<p class="status" aria-label="Persisted release status">{_html(model["status"])}</p>
<p>Export mode: <code>{_html(model["export_mode"])}</code>.
The decision was read from immutable evidence and was not recalculated.</p></header>
<main>
<section><h2>Immutable identities</h2><dl>{identity_rows}</dl></section>
<section><h2>Reason codes</h2><ul>{reason_items}</ul></section>
<section><h2>Gating cells</h2>
<table tabindex="0" aria-label="Gating cell results"><thead><tr>
<th>Cell</th><th>Metric</th><th>Slice</th><th>Status</th>
<th>Estimate</th><th>Interval</th><th>Margin</th><th>Reason</th>
</tr></thead><tbody>{"".join(cell_rows)}</tbody></table>
<h3>Counterexample queries</h3><dl>{"".join(query_rows)}</dl></section>
<section><h2>Privacy</h2><p>{_html(privacy)}</p>{image_markup}</section>
</main>
</body>
</html>
"""
    return document.encode("utf-8")


def _environment(project_root: Path) -> dict[str, Any]:
    locks = []
    for relative in _LOCK_FILES:
        path = project_root / relative
        if path.is_file() and not path.is_symlink():
            locks.append(
                {
                    "path": relative,
                    "sha256": _sha256_file(path),
                    "byte_size": path.stat().st_size,
                }
            )
    packages = []
    for name in _PACKAGE_NAMES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        packages.append({"name": name, "version": version})
    return {
        "schema_version": "junctionlens.report-environment.v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {"machine": platform.machine(), "system": platform.system()},
        "packages": packages,
        "locks": locks,
        "environment_variables_included": False,
        "filesystem_paths_included": False,
    }


def _commands(
    comparison_manifest_sha256: str,
    mode: Literal["public", "private"],
    scene_manifest_sha256: str | None,
) -> bytes:
    report_argv = [
        "junctionlens",
        "report",
        "--comparison",
        comparison_manifest_sha256,
        "--mode",
        mode,
        "--artifact-root",
        "<artifact-root>",
        "--output-dir",
        "<output-directory>",
    ]
    if scene_manifest_sha256 is not None:
        report_argv.extend(["--scene", scene_manifest_sha256])
    if mode == "private" and scene_manifest_sha256 is not None:
        report_argv.append("--acknowledge-private-license")
    commands = [
        {
            "schema_version": "junctionlens.reproduction-command.v1",
            "purpose": "recreate this deterministic evidence bundle",
            "argv": report_argv,
        },
        {
            "schema_version": "junctionlens.reproduction-command.v1",
            "purpose": "open the self-contained report offline",
            "argv": ["<browser>", "<output-directory>/REPORT.html"],
        },
    ]
    return b"".join(canonical_json_bytes(command) + b"\n" for command in commands)


def _counterexamples(
    repository: EvidenceRepository,
    *,
    scene_manifest_sha256: str | None,
    decision_manifest_sha256: str,
    mode: Literal["public", "private"],
    private_license_acknowledged: bool,
    staging: Path,
) -> tuple[dict[str, Any], list[dict[str, JsonValue]], tuple[str, ...]]:
    if scene_manifest_sha256 is None:
        return (
            {
                "schema_version": "junctionlens.counterexample-export.v1",
                "export_mode": mode,
                "scene": None,
                "images": [],
            },
            [],
            (),
        )
    scene_artifact = repository.artifact(scene_manifest_sha256)
    if (
        scene_artifact.kind != "counterexample_bundle"
        or scene_artifact.media_type != _SCENE_MEDIA_TYPE
    ):
        raise EvidenceBundleError("scene is not a registered counterexample bundle")
    scene, scene_decision = repository.scene_bundle(scene_manifest_sha256)
    if scene.decision_manifest_sha256 != decision_manifest_sha256:
        raise EvidenceBundleError("counterexample scene belongs to a different decision")
    if scene_decision.get("decision_sha256") is None:
        raise EvidenceBundleError("counterexample scene decision has no immutable identity")
    if mode == "public":
        frames = []
        for frame in scene.frames:
            frames.append(
                {
                    "frame_id": frame.frame_id,
                    "segment_id": frame.segment_id,
                    "timestamp_ns": frame.timestamp_ns,
                    "cameras": [
                        {
                            "slot": camera.slot,
                            "label": camera.label,
                            "state": "OMITTED_PUBLIC",
                            "restriction_reason": (
                                camera.restriction_reason
                                or "Image omitted from the public evidence export."
                            ),
                        }
                        for camera in frame.cameras
                    ],
                }
            )
        return (
            {
                "schema_version": "junctionlens.counterexample-export.v1",
                "export_mode": "public",
                "scene": {
                    "title": scene.title,
                    "license_notice": scene.license_notice,
                    "frames": frames,
                },
                "images": [],
            },
            [],
            (scene_manifest_sha256,),
        )
    if not private_license_acknowledged:
        raise EvidenceBundleError("private scene export requires --acknowledge-private-license")
    images: list[dict[str, JsonValue]] = []
    image_parents: set[str] = set()
    image_directory = staging / "private-thumbnails"
    image_directory.mkdir(mode=0o700)
    for frame in scene.frames:
        for camera in frame.cameras:
            manifest_sha256 = camera.artifact_manifest_sha256
            if manifest_sha256 is None or manifest_sha256 in image_parents:
                continue
            artifact = repository.artifact(manifest_sha256)
            extension = _IMAGE_EXTENSIONS.get(artifact.media_type)
            if extension is None:
                raise EvidenceBundleError("private scene contains an unsupported image type")
            payload = repository.open_payload(
                manifest_sha256,
                limit=_MAX_PRIVATE_IMAGE_BYTES,
            ).read()
            relative = f"private-thumbnails/{manifest_sha256}{extension}"
            target = staging / relative
            target.write_bytes(payload)
            target.chmod(0o600)
            images.append(
                {
                    "label": camera.label,
                    "frame_id": frame.frame_id,
                    "camera_slot": camera.slot,
                    "relative_path": relative,
                    "source_manifest_sha256": manifest_sha256,
                    "license_id": artifact.license_id,
                    "source_label": "registered counterexample camera artifact",
                }
            )
            image_parents.add(manifest_sha256)
    images.sort(key=lambda item: cast(str, item["source_manifest_sha256"]))
    return (
        {
            "schema_version": "junctionlens.counterexample-export.v1",
            "export_mode": "private",
            "scene_manifest_sha256": scene_manifest_sha256,
            "scene_license_id": scene_artifact.license_id,
            "scene": scene.model_dump(mode="json"),
            "images": images,
        },
        images,
        tuple(sorted({scene_manifest_sha256, *image_parents})),
    )


def _safe_output_directory(
    output_directory: Path,
    *,
    mode: Literal["public", "private"],
    artifact_root: Path,
) -> Path:
    expanded = output_directory.expanduser()
    if expanded.exists() or expanded.is_symlink():
        raise EvidenceBundleError("report output directory already exists")
    try:
        parent = expanded.parent.resolve(strict=True)
    except OSError as error:
        raise EvidenceBundleError("report output parent must already exist") from error
    if expanded.parent.is_symlink() or not parent.is_dir():
        raise EvidenceBundleError("report output parent must be a real directory")
    output = parent / expanded.name
    if output.name in {"", ".", ".."}:
        raise EvidenceBundleError("report output directory name is invalid")
    if mode == "private":
        root = artifact_root.resolve(strict=True)
        if not output.is_relative_to(root):
            raise EvidenceBundleError("private report output must remain beneath artifact root")
    return output


def _write_files(staging: Path, files: Mapping[str, bytes]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, payload in sorted(files.items()):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o600)
        hashes[relative] = _sha256_bytes(payload)
    return hashes


def _deterministic_zip(staging: Path, paths: Sequence[str]) -> Path:
    archive_path = staging / "junctionlens-evidence-bundle.zip"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for relative in sorted(paths):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0x800
            archive.writestr(info, (staging / relative).read_bytes())
    archive_path.chmod(0o600)
    return archive_path


def export_evidence_bundle(
    *,
    artifact_root: Path,
    schema_path: Path,
    project_root: Path,
    comparison_manifest_sha256: str,
    output_directory: Path,
    mode: Literal["public", "private"] = "public",
    scene_manifest_sha256: str | None = None,
    private_license_acknowledged: bool = False,
) -> EvidenceBundleReceipt:
    """Export and register one deterministic comparison evidence bundle."""
    try:
        if mode == "public" and private_license_acknowledged:
            raise EvidenceBundleError("private license acknowledgment is invalid in public mode")
        output = _safe_output_directory(
            output_directory,
            mode=mode,
            artifact_root=artifact_root,
        )
        repository = EvidenceRepository(
            ServiceConfig(artifact_root=artifact_root, schema_path=schema_path)
        )
        report, license_id = _load_report_data(repository, comparison_manifest_sha256)
        decision, decision_payload = _decision_bytes(repository, report)
        metrics_payload = _verified_parquet(
            repository,
            report.metrics_table_manifest_sha256,
            expected_kind="comparison",
            decision_manifest_sha256=report.decision_manifest_sha256,
        )
        slices_payload = _verified_parquet(
            repository,
            report.slice_table_manifest_sha256,
            expected_kind="slice_table",
            decision_manifest_sha256=report.decision_manifest_sha256,
        )
        registry = EvidenceRegistry(artifact_root, schema_path)
    except EvidenceBundleError:
        raise
    except (EvidenceReadError, KeyError, RegistryError, OSError, TypeError, ValueError) as error:
        raise EvidenceBundleError(str(error)) from error
    temporary_path: Path | None = None
    try:
        temporary_path = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        counterexamples, private_images, counterexample_parents = _counterexamples(
            repository,
            scene_manifest_sha256=scene_manifest_sha256,
            decision_manifest_sha256=report.decision_manifest_sha256,
            mode=mode,
            private_license_acknowledged=private_license_acknowledged,
            staging=temporary_path,
        )
        model = _report_model(
            report,
            mode=mode,
            private_images=private_images,
            counterexample_summary={
                "included": scene_manifest_sha256 is not None,
                "scene_manifest_sha256": scene_manifest_sha256,
                "image_count": len(private_images),
            },
        )
        files = {
            "REPORT.html": _render_html(model),
            "REPORT.json": canonical_json_bytes(model) + b"\n",
            "REPORT.md": _render_markdown(model),
            "commands.jsonl": _commands(comparison_manifest_sha256, mode, scene_manifest_sha256),
            "counterexamples.json": canonical_json_bytes(counterexamples) + b"\n",
            "decision.json": decision_payload,
            "environment.json": canonical_json_bytes(_environment(project_root)) + b"\n",
            "metrics.parquet": metrics_payload,
            "slices.parquet": slices_payload,
        }
        file_hashes = _write_files(temporary_path, files)
        for path in temporary_path.rglob("private-thumbnails/*"):
            if path.is_file():
                file_hashes[path.relative_to(temporary_path).as_posix()] = _sha256_file(path)
        bundle_manifest = {
            "schema_version": "junctionlens.evidence-bundle-manifest.v1",
            "export_mode": mode,
            "comparison_manifest_sha256": comparison_manifest_sha256,
            "decision_manifest_sha256": report.decision_manifest_sha256,
            "status": decision["status"],
            "license_id": license_id,
            "files": [
                {
                    "path": relative,
                    "sha256": digest,
                    "byte_size": (temporary_path / relative).stat().st_size,
                }
                for relative, digest in sorted(file_hashes.items())
            ],
            "dataset_frames_included": bool(private_images),
            "private_paths_included": False,
        }
        manifest_payload = canonical_json_bytes(bundle_manifest) + b"\n"
        (temporary_path / "manifest.json").write_bytes(manifest_payload)
        (temporary_path / "manifest.json").chmod(0o600)
        file_hashes["manifest.json"] = _sha256_bytes(manifest_payload)
        sums = "".join(
            f"{digest}  {relative}\n" for relative, digest in sorted(file_hashes.items())
        ).encode("utf-8")
        (temporary_path / "SHA256SUMS").write_bytes(sums)
        (temporary_path / "SHA256SUMS").chmod(0o600)
        file_hashes["SHA256SUMS"] = _sha256_bytes(sums)
        expected = set(_REQUIRED_BUNDLE_FILES) | {"manifest.json", "SHA256SUMS"}
        if not expected.issubset(file_hashes):
            raise EvidenceBundleError("report bundle is missing a required file")
        archive_path = _deterministic_zip(temporary_path, tuple(file_hashes))
        parents = tuple(
            sorted(
                {
                    comparison_manifest_sha256,
                    report.decision_manifest_sha256,
                    report.metrics_table_manifest_sha256,
                    report.slice_table_manifest_sha256,
                    *counterexample_parents,
                }
            )
        )
        receipt = registry.put_file(
            archive_path,
            kind="evidence_report",
            media_type="application/zip",
            license_id=license_id,
            metadata={
                "bundle_manifest_sha256": file_hashes["manifest.json"],
                "comparison_manifest_sha256": comparison_manifest_sha256,
                "decision_manifest_sha256": report.decision_manifest_sha256,
                "export_mode": mode,
                "format": "junctionlens-evidence-bundle-v1",
                "status": report.status,
            },
            parents=parents,
        )
        for path in temporary_path.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        temporary_path.replace(output)
        temporary_path = None
    except EvidenceBundleError:
        raise
    except (EvidenceReadError, KeyError, RegistryError, OSError, TypeError, ValueError) as error:
        raise EvidenceBundleError(str(error)) from error
    finally:
        if temporary_path is not None:
            shutil.rmtree(temporary_path, ignore_errors=True)
    return EvidenceBundleReceipt(
        export_mode=mode,
        comparison_manifest_sha256=comparison_manifest_sha256,
        decision_manifest_sha256=report.decision_manifest_sha256,
        bundle_manifest_sha256=file_hashes["manifest.json"],
        output_directory=str(output),
        archive=receipt,
        file_sha256=dict(sorted(file_hashes.items())),
    )


__all__ = [
    "EvidenceBundleError",
    "EvidenceBundleReceipt",
    "export_evidence_bundle",
]

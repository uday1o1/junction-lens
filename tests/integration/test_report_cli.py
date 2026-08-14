"""End-to-end evidence report export through the public CLI."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from jsonschema import Draft202012Validator
from tests.helpers.report_evidence import create_report_evidence
from typer.testing import CliRunner

from junctionlens.cli.main import app

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "schemas/artifact-manifest-v1.schema.json"
REPORT_SCHEMA = ROOT / "schemas/report-v1.schema.json"
REQUIRED_FILES = {
    "REPORT.html",
    "REPORT.json",
    "REPORT.md",
    "SHA256SUMS",
    "commands.jsonl",
    "counterexamples.json",
    "decision.json",
    "environment.json",
    "manifest.json",
    "metrics.parquet",
    "slices.parquet",
}


def _arguments(artifact_root: Path, comparison: str, output: Path) -> list[str]:
    return [
        "report",
        "--comparison",
        comparison,
        "--artifact-root",
        str(artifact_root),
        "--schema",
        str(SCHEMA),
        "--output-dir",
        str(output),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_bundle_is_complete_escaped_and_byte_reproducible(tmp_path: Path) -> None:
    evidence = create_report_evidence(tmp_path)
    first_output = tmp_path / "public-a"
    second_output = tmp_path / "public-b"
    runner = CliRunner()

    first = runner.invoke(
        app,
        _arguments(evidence.artifact_root, evidence.comparison_manifest_sha256, first_output),
    )
    second = runner.invoke(
        app,
        _arguments(evidence.artifact_root, evidence.comparison_manifest_sha256, second_output),
    )

    assert first.exit_code == second.exit_code == 0, first.output
    first_receipt = json.loads(first.stdout)
    second_receipt = json.loads(second.stdout)
    for key in ("manifest_sha256", "payload_sha256", "bundle_manifest_sha256", "file_sha256"):
        assert first_receipt[key] == second_receipt[key]
    assert first_receipt["export_mode"] == "public"
    assert REQUIRED_FILES.issubset(path.name for path in first_output.iterdir())
    assert first_receipt["file_sha256"] == {
        relative: _sha256(first_output / relative) for relative in first_receipt["file_sha256"]
    }
    assert (first_output / "decision.json").read_bytes() == evidence.decision_bytes
    assert (first_output / "metrics.parquet").read_bytes() == evidence.metrics_bytes
    assert (first_output / "slices.parquet").read_bytes() == evidence.slices_bytes
    report = json.loads((first_output / "REPORT.json").read_bytes())
    schema = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
    assert report["privacy"] == {
        "dataset_frames_included": False,
        "licensed_thumbnails_opt_in": False,
        "private_paths_included": False,
    }
    assert report["decision_recalculated"] is False
    html = (first_output / "REPORT.html").read_text(encoding="utf-8")
    markdown = (first_output / "REPORT.md").read_text(encoding="utf-8")
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "<script" not in html.lower()
    assert 'src="http:' not in html.lower()
    assert 'src="https:' not in html.lower()
    assert "\\|overall" in markdown
    assert "\\`descending\\`" in markdown
    commands = (first_output / "commands.jsonl").read_text(encoding="utf-8")
    environment = (first_output / "environment.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in commands
    assert str(tmp_path) not in environment
    sums = (first_output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in sums:
        digest, relative = line.split("  ", maxsplit=1)
        assert digest == _sha256(first_output / relative)
    archive = first_output / "junctionlens-evidence-bundle.zip"
    assert archive.read_bytes() == (second_output / archive.name).read_bytes()
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == sorted(first_receipt["file_sha256"])
        assert {entry.date_time for entry in bundle.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_public_scene_is_redacted_and_private_scene_requires_acknowledgment(
    tmp_path: Path,
) -> None:
    evidence = create_report_evidence(tmp_path)
    runner = CliRunner()
    public_output = tmp_path / "public-scene"
    public = runner.invoke(
        app,
        [
            *_arguments(
                evidence.artifact_root,
                evidence.comparison_manifest_sha256,
                public_output,
            ),
            "--scene",
            evidence.scene_manifest_sha256,
        ],
    )

    assert public.exit_code == 0, public.output
    public_counterexamples = json.loads((public_output / "counterexamples.json").read_bytes())
    assert public_counterexamples["images"] == []
    assert "artifact_manifest_sha256" not in json.dumps(public_counterexamples)
    assert not (public_output / "private-thumbnails").exists()

    private_output = evidence.artifact_root / "private-without-ack"
    rejected = runner.invoke(
        app,
        [
            *_arguments(
                evidence.artifact_root,
                evidence.comparison_manifest_sha256,
                private_output,
            ),
            "--mode",
            "private",
            "--scene",
            evidence.scene_manifest_sha256,
        ],
    )
    assert rejected.exit_code == 2
    assert "acknowledge-private-license" in rejected.stderr
    assert not private_output.exists()

    accepted_output = evidence.artifact_root / "private-accepted"
    accepted = runner.invoke(
        app,
        [
            *_arguments(
                evidence.artifact_root,
                evidence.comparison_manifest_sha256,
                accepted_output,
            ),
            "--mode",
            "private",
            "--scene",
            evidence.scene_manifest_sha256,
            "--acknowledge-private-license",
        ],
    )
    assert accepted.exit_code == 0, accepted.output
    report = json.loads((accepted_output / "REPORT.json").read_bytes())
    assert report["privacy"]["dataset_frames_included"] is True
    assert report["privacy"]["licensed_thumbnails_opt_in"] is True
    image = report["private_images"][0]
    assert image["license_id"] == "CC-BY-NC-SA-4.0"
    assert image["source_manifest_sha256"] == evidence.image_manifest_sha256
    assert (accepted_output / image["relative_path"]).read_bytes() == evidence.image_bytes
    html = (accepted_output / "REPORT.html").read_text(encoding="utf-8")
    assert image["relative_path"] in html
    assert "Front &lt;center&gt;" in html


def test_report_refuses_clobber_and_private_output_outside_registry(tmp_path: Path) -> None:
    evidence = create_report_evidence(tmp_path)
    runner = CliRunner()
    existing = tmp_path / "existing"
    existing.mkdir()

    clobber = runner.invoke(
        app,
        _arguments(evidence.artifact_root, evidence.comparison_manifest_sha256, existing),
    )
    outside = runner.invoke(
        app,
        [
            *_arguments(
                evidence.artifact_root,
                evidence.comparison_manifest_sha256,
                tmp_path / "outside-private",
            ),
            "--mode",
            "private",
        ],
    )

    assert clobber.exit_code == outside.exit_code == 2
    assert "already exists" in clobber.stderr
    assert "beneath artifact root" in outside.stderr

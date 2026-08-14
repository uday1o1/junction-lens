"""Public immutable registry workflow tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from junctionlens.cli.main import app

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "schemas/artifact-manifest-v1.schema.json"


def test_registry_cli_put_inspect_provenance_and_gc_are_deterministic(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    payload = tmp_path / "payload.json"
    payload.write_text('{"measured":true}\n', encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"name":"synthetic"}\n', encoding="utf-8")
    put_arguments = [
        "registry",
        "put",
        "--input",
        str(payload),
        "--kind",
        "evidence_report",
        "--media-type",
        "application/json",
        "--license-id",
        "Apache-2.0",
        "--metadata",
        str(metadata),
        "--artifact-root",
        str(artifact_root),
        "--schema",
        str(SCHEMA),
    ]
    runner = CliRunner()

    first = runner.invoke(app, put_arguments)
    second = runner.invoke(app, put_arguments)

    assert first.exit_code == second.exit_code == 0
    assert first.stdout == second.stdout
    receipt = json.loads(first.stdout)
    inspect = runner.invoke(
        app,
        [
            "registry",
            "inspect",
            "--manifest",
            receipt["manifest_sha256"],
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
        ],
    )
    provenance = runner.invoke(
        app,
        [
            "registry",
            "provenance",
            "--manifest",
            receipt["manifest_sha256"],
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
        ],
    )
    gc = runner.invoke(
        app,
        [
            "registry",
            "gc",
            "--dry-run",
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
        ],
    )
    alias = runner.invoke(
        app,
        [
            "registry",
            "set-alias",
            "--alias",
            "reports/current",
            "--manifest",
            receipt["manifest_sha256"],
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
        ],
    )
    resolved = runner.invoke(
        app,
        [
            "registry",
            "resolve-alias",
            "--alias",
            "reports/current",
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
        ],
    )

    assert inspect.exit_code == provenance.exit_code == gc.exit_code == 0
    assert alias.exit_code == resolved.exit_code == 0
    assert json.loads(inspect.stdout)["kind"] == "evidence_report"
    assert json.loads(provenance.stdout)["artifacts"][0]["depth"] == 0
    assert json.loads(gc.stdout)["orphaned_objects"] == []
    assert json.loads(resolved.stdout)["manifest_sha256"] == receipt["manifest_sha256"]
    refused = runner.invoke(
        app,
        [
            "registry",
            "gc",
            "--artifact-root",
            str(artifact_root),
            "--schema",
            str(SCHEMA),
        ],
    )
    assert refused.exit_code == 2
    assert "requires --dry-run" in refused.stderr


def test_registry_cli_run_resume_is_environment_bound(tmp_path: Path) -> None:
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "schema_version": "junctionlens.run-identity.v1",
                "run_kind": "synthetic-evaluation",
                "parent_artifact_hashes": [],
                "dataset_manifest_sha256": "1" * 64,
                "split_manifest_sha256": "2" * 64,
                "model_profile_sha256": "3" * 64,
                "configuration_sha256": "4" * 64,
                "source_git_commit": "5" * 40,
                "source_dirty": False,
                "dependency_lock_hashes": {"uv.lock": "6" * 64},
                "container_image_digests": {"evaluator": "7" * 64},
                "seed": 20260813,
                "command_schema_version": "junctionlens.evaluate.v1",
                "execution_provider_profile": "cpu-reference",
            }
        ),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"
    arguments = [
        "registry",
        "resume",
        "--identity",
        str(identity),
        "--environment-fingerprint",
        "8" * 64,
        "--artifact-root",
        str(artifact_root),
        "--schema",
        str(SCHEMA),
    ]
    runner = CliRunner()

    created = runner.invoke(app, arguments)
    resumed = runner.invoke(app, arguments)

    assert created.exit_code == resumed.exit_code == 0
    assert json.loads(created.stdout)["state"] == "CREATED"
    assert json.loads(resumed.stdout)["state"] == "RESUMED"

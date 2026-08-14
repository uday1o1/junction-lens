"""Remote selected-study and pre-holdout charter orchestration tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts.gpu import qualify_core_evidence, qualify_faults, qualify_study

from junctionlens.faults.models import FAULT_REASON_CODES, FaultKind
from junctionlens.gate.charter import load_charter_draft

ROOT = Path(__file__).parents[2]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_value(experiment: str, seed: int, split_sha256: str) -> dict[str, Any]:
    draft = load_charter_draft(ROOT / "configs/gates/acceptance-v1.draft.yaml")
    freeze = {cell.id: 0.4 + (seed - 20260813) * 0.001 for cell in draft.cells}
    return {
        "schema_version": "junctionlens.selected-model-evaluation.v1",
        "state": "ACCEPTED",
        "experiment_id": experiment,
        "seed": seed,
        "source_partition": "model_selection",
        "internal_holdout_access_count": 0,
        "selection_split_manifest_sha256": split_sha256,
        "source_frame_manifest_sha256": "f" * 64,
        "checkpoint_sha256": hashlib.sha256(f"checkpoint-{experiment}-{seed}".encode()).hexdigest(),
        "selection_receipt_sha256": "a" * 64,
        "metrics": {
            "DET_l": 0.4,
            "DET_t": 0.4,
            "TOP_lt": 0.4,
            "wrong_control_assignment_rate": 0.2,
            "official_composite": 0.4,
            "negative_log_likelihood": 1.0,
        },
        "freeze_metrics": {
            key: value for key, value in freeze.items() if not key.startswith("runtime.")
        },
        "measured_example": {
            "frame_token": f"frame-{seed}",
            "source_domain": "argoverse2",
            "classification": "MEASURED_FAILURE",
            "severity_score": 1.0,
        },
    }


def test_power_simulation_is_deterministic_and_preholdout_only() -> None:
    variability = {"overall.DET_l": [0.4, 0.41, 0.39]}
    margins = {"overall.DET_l": 0.01}

    first = qualify_study._power_simulation(variability, margins)
    second = qualify_study._power_simulation(variability, margins)

    assert first == second
    assert first["replicates"] == 10_000
    assert first["candidate_results_used"] is False
    assert first["internal_holdout_used"] is False


def test_fault_qualification_exercises_full_public_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        kind = command[command.index("--kind") + 1]
        seed = int(command[command.index("--seed") + 1])
        fault = FaultKind(kind)
        receipt = {
            "state": (
                "CONTROL_PASSED" if fault is FaultKind.PERMUTE_NODES_CORRECTLY else "DETECTED"
            ),
            "fault_kind": kind,
            "primary_reason_code": FAULT_REASON_CODES[fault],
            "parent_manifest_sha256": command[command.index("--input") + 1],
            "derived_manifest_sha256": hashlib.sha256(f"{kind}-{seed}".encode()).hexdigest(),
            "counterexample_manifest_sha256": hashlib.sha256(
                f"counterexample-{kind}-{seed}".encode()
            ).hexdigest(),
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(receipt).encode(), b"")

    monkeypatch.setattr("scripts.gpu.qualify_faults.subprocess.run", fake_run)

    result = qualify_faults.qualify(ROOT, tmp_path / "faults")

    assert result["state"] == "ACCEPTED"
    assert result["case_count"] == len(FaultKind) * 3
    assert len(calls) == result["case_count"]
    assert all("junctionlens.cli.main" in command for command in calls)


def test_core_evidence_audit_binds_every_measured_gate(tmp_path: Path) -> None:
    data = tmp_path / "data"
    models = tmp_path / "models"
    study = tmp_path / "study"
    faults = tmp_path / "faults"
    for path in (data, models, study / "e0-final", faults):
        path.mkdir(parents=True, exist_ok=True)
    _write_json(
        data / "qualification.json",
        {"mechanical_state": "ACCEPTED", "segment_count": 700, "visual_audit_frame_count": 12},
    )
    micro = tmp_path / "micro.json"
    _write_json(
        micro,
        {
            "status": "PASSED",
            "frames": 32,
            "final_total_loss": 0.01,
            "checkpoint_sha256": "a" * 64,
        },
    )
    _write_json(
        models / "qualification.json",
        {"state": "TRAINING_AND_SELECTION_ACCEPTED", "internal_holdout_access_count": 0},
    )
    _write_json(
        study / "qualification.json",
        {
            "state": "ACCEPTED",
            "internal_holdout_access_count": 0,
            "e1_outcome": "REJECTED_BY_KEEP_GATE",
            "selected_experiment_id": "E0-independent",
        },
    )
    _write_json(study / "e0-final/model-manifest.json", {"seed_metrics": [{}, {}, {}]})
    (study / "acceptance-v1.yaml").write_text("frozen: true\n", encoding="utf-8")
    parity = tmp_path / "parity.json"
    performance = tmp_path / "performance.json"
    profiler = tmp_path / "profiler.json"
    provider = tmp_path / "provider.json"
    _write_json(
        parity,
        {
            "status": "PASSED",
            "raw_maximum_absolute_error": 0.001,
            "graph_maximum_absolute_error": 0.001,
        },
    )
    _write_json(performance, {"status": "PASSED"})
    _write_json(
        profiler,
        {"status": "PASSED", "profiled_benchmark_publishable": False},
    )
    _write_json(provider, {"status": "PASSED", "provider_profile": "cuda"})
    _write_json(
        faults / "qualification.json",
        {
            "state": "ACCEPTED",
            "detection_rate": 1.0,
            "case_count": 63,
            "detected_or_control_passed_count": 63,
            "v1_release_acceptance_run": False,
        },
    )

    result = qualify_core_evidence.qualify(
        Namespace(
            data=data,
            micro=micro,
            models=models,
            study=study,
            parity=parity,
            performance=performance,
            profiler=profiler,
            provider=provider,
            faults=faults,
            visual_signoff_sha256="b" * 64,
            source_commit="c" * 40,
            output=tmp_path / "core",
        )
    )

    assert result["state"] == "ACCEPTED"
    evidence = json.loads((tmp_path / "core/core-evidence.json").read_text())
    assert evidence["internal_holdout_access_count"] == 0
    assert evidence["fault_lab"]["case_count"] == 63


def test_study_qualification_uses_all_e0_seeds_and_freezes_charter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    data = tmp_path / "data"
    models = tmp_path / "models"
    output = tmp_path / "study"
    for path in (dataset, data, models):
        path.mkdir()
    split = data / "split-manifest.json"
    split.write_text("{}\n", encoding="utf-8")
    split_sha256 = _sha256(split)
    environment = tmp_path / "environment.json"
    performance = tmp_path / "performance.json"
    provider = tmp_path / "provider.json"
    _write_json(
        environment,
        {
            "failures": [],
            "selected_gpu": {"uuid": "GPU-test"},
            "driver_version": "570.26",
            "cuda_toolkit_version": "12.8",
            "cudnn_package_version": "9.14.0.64",
        },
    )
    _write_json(performance, {"status": "PASSED", "p95_latency_ms": 10.0})
    _write_json(provider, {"status": "PASSED", "provider_profile": "cuda"})
    labels: list[str] = []

    def fake_cli(label: str, arguments: list[str], **_kwargs: object) -> dict[str, Any]:
        labels.append(label)
        if label.startswith("evaluate-"):
            root = Path(arguments[arguments.index("--output-root") + 1])
            experiment = arguments[arguments.index("--experiment") + 1]
            seed = int(label.rsplit("-", 1)[-1])
            result = _selected_value(experiment, seed, split_sha256)
            _write_json(root / "metrics.json", {"state": "ACCEPTED"})
            _write_json(root / "prediction-manifest.json", {"state": "ACCEPTED"})
            result["evaluation_artifact_sha256"] = _sha256(root / "metrics.json")
            result["prediction_manifest_sha256"] = _sha256(root / "prediction-manifest.json")
            _write_json(root / "selected-evaluation.json", result)
        elif label == "finalize-e0":
            root = Path(arguments[arguments.index("--output-root") + 1])
            _write_json(root / "model-manifest.json", {"state": "THREE_SEED_BASELINE_COMPLETE"})
            (root / "MODEL_CARD.md").write_text("# Model card\n", encoding="utf-8")
            measured_path = Path(arguments[arguments.index("--measured-evidence") + 1])
            manifest = json.loads((root / "model-manifest.json").read_text())
            manifest["measured_evidence_sha256"] = _sha256(measured_path)
            _write_json(root / "model-manifest.json", manifest)
            _write_json(
                output / "receipts/finalize-e0.json",
                {
                    "state": "ACCEPTED",
                    "model_manifest_sha256": _sha256(root / "model-manifest.json"),
                    "model_card_sha256": _sha256(root / "MODEL_CARD.md"),
                },
            )
        elif label == "finalize-e1-study":
            path = Path(arguments[arguments.index("--output") + 1])
            evidence_path = Path(arguments[arguments.index("--evidence") + 1])
            _write_json(
                path,
                {
                    "state": "ACCEPTED",
                    "study_validity": "ACCEPTED",
                    "outcome": "REJECTED_BY_KEEP_GATE",
                    "selected_experiment_id": "E0-independent",
                    "source_partition": "model_selection",
                    "internal_holdout_access_count": 0,
                    "evidence_sha256": _sha256(evidence_path),
                },
            )
        elif label == "freeze-charter":
            path = Path(arguments[arguments.index("--output") + 1])
            path.write_text(
                "schema_version: junctionlens.acceptance-charter.v1\n", encoding="utf-8"
            )
        return {"state": "ACCEPTED"}

    monkeypatch.setattr(qualify_study, "_run_cli", fake_cli)

    class EqualValue:
        def __eq__(self, _other: object) -> bool:
            return True

        def __ne__(self, _other: object) -> bool:
            return False

    monkeypatch.setattr(
        qualify_study,
        "load_frozen_charter",
        lambda _path: SimpleNamespace(
            source_commit="1" * 40,
            baseline_run_manifest_sha256=EqualValue(),
            freeze_evidence=SimpleNamespace(m0_hardware_baseline_manifest_sha256=EqualValue()),
        ),
    )

    result = qualify_study.qualify(
        project_root=ROOT,
        dataset_root=dataset,
        data_qualification_root=data,
        models_root=models,
        environment_path=environment,
        performance_path=performance,
        provider_path=provider,
        output_root=output,
        source_commit="1" * 40,
    )

    assert result["state"] == "ACCEPTED"
    assert result["selected_experiment_id"] == "E0-independent"
    assert labels[:4] == [
        "evaluate-e0-20260813",
        "evaluate-e0-20260814",
        "evaluate-e0-20260815",
        "evaluate-e1-20260813",
    ]
    assert labels[-3:] == ["finalize-e0", "finalize-e1-study", "freeze-charter"]
    evidence = json.loads((output / "e0-measured-evidence.json").read_text())
    assert [item["seed"] for item in evidence["seed_metrics"]] == list(qualify_study.SEEDS)

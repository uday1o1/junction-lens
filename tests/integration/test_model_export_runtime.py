from __future__ import annotations

import json
import subprocess
from pathlib import Path

import onnx
import pytest
import torch

from junctionlens.model.export import ModelExportError, export_model, validate_exported_model
from junctionlens.model.parity import run_parity
from junctionlens.model.profile import load_m0_profile
from junctionlens.model.spike import M0GraphModel


@pytest.fixture(scope="module")
def exported_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("model-export")
    profile = load_m0_profile(Path("configs/model/m0-spike.yaml"))
    torch.manual_seed(profile.seed)
    checkpoint = root / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": M0GraphModel(profile).state_dict(),
            "profile_sha256": profile.canonical_sha256(),
            "profile": profile.model_dump(mode="json"),
            "seed": profile.seed,
        },
        checkpoint,
    )
    model_path = root / "model.onnx"
    report = export_model(profile, checkpoint, model_path)
    assert report["status"] == "PASSED"
    assert str(Path.cwd()).encode() not in model_path.read_bytes()
    return checkpoint, model_path


def test_export_runs_python_and_native_raw_output_parity(
    exported_model: tuple[Path, Path], tmp_path: Path
) -> None:
    checkpoint, model_path = exported_model
    runner = Path("build/cpu/junctionlens-onnx-probe")
    assert runner.is_file(), "run ./tools/jl build-cpu before the Python integration suite"
    profile = load_m0_profile(Path("configs/model/m0-spike.yaml"))
    report = run_parity(profile, checkpoint, model_path, runner, tmp_path / "parity.json")
    assert report["status"] == "PASSED"
    assert report["dynamic_batch_two"] == "PASSED"
    assert (
        max(
            item["pytorch_to_native_cpp_ort"]["maximum_absolute_error"]
            for item in report["tensors"]
        )
        <= 1e-4
    )


def test_native_metadata_validator_rejects_seeded_defect_and_control_passes(
    exported_model: tuple[Path, Path], tmp_path: Path
) -> None:
    _, model_path = exported_model
    profile = load_m0_profile(Path("configs/model/m0-spike.yaml"))
    runner = Path("build/cpu/junctionlens-onnx-probe").resolve()
    common = [
        str(runner),
        "--expected-profile-sha256",
        profile.canonical_sha256(),
        "--frame-index",
        "7",
    ]
    control = subprocess.run(
        [*common, "--model", str(model_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert control.returncode == 0
    assert json.loads(control.stdout)["status"] == "PASSED"

    corrupted = onnx.load_model(model_path)
    for item in corrupted.metadata_props:
        if item.key == "junctionlens.profile_sha256":
            item.value = "0" * 64
    corrupted_path = tmp_path / "wrong-metadata.onnx"
    onnx.save_model(corrupted, corrupted_path)
    defect = subprocess.run(
        [*common, "--model", str(corrupted_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert defect.returncode != 0
    assert "model metadata differs: junctionlens.profile_sha256" in defect.stderr


def test_python_validator_rejects_missing_metadata(exported_model: tuple[Path, Path]) -> None:
    _, model_path = exported_model
    profile = load_m0_profile(Path("configs/model/m0-spike.yaml"))
    model = onnx.load_model(model_path)
    retained = [
        item for item in model.metadata_props if item.key != "junctionlens.output_contract_sha256"
    ]
    del model.metadata_props[:]
    model.metadata_props.extend(retained)
    broken = model_path.with_name("missing-metadata.onnx")
    onnx.save_model(model, broken)
    with pytest.raises(ModelExportError, match="output_contract_sha256"):
        validate_exported_model(broken, profile)


def test_same_checkpoint_exports_byte_identical_model(
    exported_model: tuple[Path, Path], tmp_path: Path
) -> None:
    checkpoint, model_path = exported_model
    profile = load_m0_profile(Path("configs/model/m0-spike.yaml"))
    repeated = tmp_path / "repeated.onnx"
    export_model(profile, checkpoint, repeated)
    assert repeated.read_bytes() == model_path.read_bytes()

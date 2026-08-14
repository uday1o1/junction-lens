from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from junctionlens.model.contract import input_contract, output_contract
from junctionlens.model.profile import M0ModelProfile, load_m0_profile
from junctionlens.model.spike import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    M0GraphModel,
    project_image_center_rays,
)
from junctionlens.model.synthetic import make_micro_inputs

PROFILE_PATH = Path("configs/model/m0-spike.yaml")


def test_profile_is_strict_and_hash_is_canonical() -> None:
    profile = load_m0_profile(PROFILE_PATH)
    payload = profile.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        M0ModelProfile.model_validate(payload)
    reordered = json.loads(json.dumps(profile.model_dump(mode="json"), sort_keys=True))
    assert M0ModelProfile.model_validate(reordered).canonical_sha256() == profile.canonical_sha256()


def test_tensor_contract_matches_full_model_surface() -> None:
    profile = load_m0_profile(PROFILE_PATH)
    assert tuple(item.name for item in input_contract(profile)) == INPUT_NAMES
    assert tuple(item.name for item in output_contract(profile)) == OUTPUT_NAMES
    assert input_contract(profile)[0].shape == ("batch", 2, 8, 3, 384, 640)
    assert output_contract(profile)[-1].shape == ("batch", 64, 96)


def test_model_outputs_have_frozen_shapes_and_bounded_scales() -> None:
    profile = load_m0_profile(PROFILE_PATH)
    inputs = make_micro_inputs(profile, torch.tensor([3, 5]), spatial_size=8)
    model = M0GraphModel(profile).eval()
    with torch.inference_mode():
        outputs = dict(zip(OUTPUT_NAMES, model(*inputs), strict=True))
    for contract in output_contract(profile):
        expected = tuple(2 if item == "batch" else item for item in contract.shape)
        assert tuple(outputs[contract.name].shape) == expected
        assert torch.isfinite(outputs[contract.name]).all()
    assert (outputs["lane_geometry_scales"] > 0).all()
    assert (outputs["traffic_box_scales"] > 0).all()
    assert (outputs["area_geometry_scales"] > 0).all()
    diagonal = outputs["lane_successor_logits"].diagonal(dim1=1, dim2=2)
    assert torch.equal(diagonal, torch.full_like(diagonal, -20.0))


def test_micro_input_identity_is_spatially_equivalent() -> None:
    profile = load_m0_profile(PROFILE_PATH)
    frame = torch.tensor([23])
    small = make_micro_inputs(profile, frame, spatial_size=8)
    full = make_micro_inputs(profile, frame, spatial_size=(384, 640))
    assert torch.equal(small[0].mean(dim=(-1, -2)), full[0].mean(dim=(-1, -2)))
    for small_value, full_value in zip(small[1:], full[1:], strict=True):
        assert torch.equal(small_value, full_value)


def test_calibrated_projection_maps_center_ray_to_vehicle_frame() -> None:
    profile = load_m0_profile(PROFILE_PATH)
    inputs = make_micro_inputs(profile, torch.tensor([1]), spatial_size=8)
    rays = project_image_center_rays(
        inputs[2], inputs[3], profile.input.height, profile.input.width
    )
    assert rays.shape == (1, 2, 8, 3)
    assert torch.allclose(rays[..., 0], torch.zeros_like(rays[..., 0]))
    assert torch.allclose(rays[..., 1], torch.zeros_like(rays[..., 1]))
    assert torch.allclose(rays[..., 2], torch.ones_like(rays[..., 2]))

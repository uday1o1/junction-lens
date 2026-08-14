"""OpenLane official-devkit parity projection tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from junctionlens.data.openlane import OpenLaneAdapter
from junctionlens.data.parity import (
    AdapterParityError,
    adapted_source_projection,
    load_parity_selector,
    verify_official_parity,
)

_CONFIG = Path("configs/data/openlane-v2-v2.1.adapter.yaml")


def _selector(path: Path) -> Path:
    selector = path / "selector.yaml"
    selector.write_text(
        yaml.safe_dump(
            {
                "schema_version": "junctionlens.openlane-parity-selector.v1",
                "frames": [{"split_id": "train", "segment_id": "segment-1", "timestamp": "100"}],
            }
        ),
        encoding="utf-8",
    )
    return selector


def test_parity_gate_accepts_identical_official_projection(
    openlane_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parity gate compares the complete frozen projection and reports only hashes."""
    adapter = OpenLaneAdapter(openlane_root, _CONFIG)
    frame = adapted_source_projection(adapter.load_frame("train", "segment-1", "100"))

    def official(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {
            "schema_version": "junctionlens.openlane-official-projection.v1",
            "devkit_version": "2.1.0",
            "frames": [copy.deepcopy(frame)],
        }

    monkeypatch.setattr("junctionlens.data.parity.run_official_projection", official)
    report = verify_official_parity(adapter, _selector(tmp_path), Path.cwd())
    assert report["state"] == "ACCEPTED"
    assert report["frame_count"] == 1
    assert report["maximum_absolute_numeric_error"] <= 1e-12
    assert set(report) == {
        "schema_version",
        "state",
        "devkit_version",
        "frame_count",
        "maximum_absolute_numeric_error",
        "tolerance",
        "official_projection_set_sha256",
    }


def test_parity_gate_rejects_seeded_topology_difference(
    openlane_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nearby official control proves the seeded topology mismatch is detected."""
    adapter = OpenLaneAdapter(openlane_root, _CONFIG)
    official_frame = adapted_source_projection(adapter.load_frame("train", "segment-1", "100"))
    corrupted = copy.deepcopy(official_frame)
    corrupted["annotation"]["topology_lste"][0][0] = 0
    monkeypatch.setattr(
        "junctionlens.data.parity.run_official_projection",
        lambda *_args, **_kwargs: {
            "schema_version": "junctionlens.openlane-official-projection.v1",
            "devkit_version": "2.1.0",
            "frames": [corrupted],
        },
    )
    with pytest.raises(AdapterParityError, match="topology_lste"):
        verify_official_parity(adapter, _selector(tmp_path), Path.cwd())


def test_parity_selector_rejects_traversal(tmp_path: Path) -> None:
    """Frozen selectors cannot escape the registered dataset root."""
    selector = _selector(tmp_path)
    payload = yaml.safe_load(selector.read_text(encoding="utf-8"))
    payload["frames"][0]["segment_id"] = "../outside"
    selector.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(AdapterParityError, match="unsafe identifier"):
        load_parity_selector(selector)

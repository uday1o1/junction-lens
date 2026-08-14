"""Unrestricted product demonstration safety and graph-evidence tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from junctionlens.cli.main import app
from junctionlens.demo.synthetic import _write_paired_graphs
from junctionlens.evaluator.custom import load_graph_pairs
from junctionlens.v1 import scene_control_graph_pb2 as scg


def test_demo_refuses_to_replace_an_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    output.mkdir()
    marker = output / "owned-by-caller.txt"
    marker.write_text("preserve me\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["demo-synthetic", "--output", str(output)])

    assert result.exit_code == 2
    assert "demo output already exists" in result.stderr
    assert marker.read_text(encoding="utf-8") == "preserve me\n"


def test_demo_materializes_real_paired_control_swap_evidence(tmp_path: Path) -> None:
    ground_truth, baseline, candidate = _write_paired_graphs(tmp_path)

    baseline_pairs = load_graph_pairs(ground_truth, baseline)
    candidate_pairs = load_graph_pairs(ground_truth, candidate)

    assert len(baseline_pairs) == len(candidate_pairs) == 200
    assert {json.loads(pair.segment_token)["segment_id"] for pair in baseline_pairs} == {
        f"demo-segment-{index:03d}" for index in range(200)
    }
    for clean, fault in zip(baseline_pairs, candidate_pairs, strict=True):
        assert clean.frame_token == fault.frame_token
        assert clean.prediction.graph.lanes == fault.prediction.graph.lanes
        assert clean.prediction.graph.traffic_controls == fault.prediction.graph.traffic_controls
        clean_edges = {
            (edge.source_node_id, edge.target_node_id)
            for edge in clean.prediction.graph.edges
            if edge.edge_type == scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE
        }
        fault_edges = {
            (edge.source_node_id, edge.target_node_id)
            for edge in fault.prediction.graph.edges
            if edge.edge_type == scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE
        }
        assert len(clean_edges) == len(fault_edges) == 3
        assert clean_edges.isdisjoint(fault_edges)

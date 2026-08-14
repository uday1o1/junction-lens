import type {
  SceneBundleDetail,
  SceneFrame,
  SceneGraphLayer,
} from "../src/api";

const HASH = "a".repeat(64);
const DECISION_HASH = "b".repeat(64);

function graph(offset: number): SceneGraphLayer {
  return {
    lanes: [
      {
        node_id: `lane-${offset}`,
        confidence: 0.91,
        points: [
          { x: 2, y: offset },
          { x: 12, y: offset + 0.5 },
          { x: 24, y: offset + 1 },
        ],
      },
      {
        node_id: `lane-next-${offset}`,
        confidence: 0.87,
        points: [
          { x: 24, y: offset + 1 },
          { x: 35, y: offset + 3 },
        ],
      },
    ],
    controls: [
      {
        node_id: `control-${offset}`,
        x: 18,
        y: offset - 4,
        control_type: "traffic_light",
        state: "red",
        confidence: 0.94,
      },
    ],
    edges: [
      {
        edge_id: `successor-${offset}`,
        edge_type: "lane_successor",
        source_node_id: `lane-${offset}`,
        target_node_id: `lane-next-${offset}`,
        confidence: 0.88,
      },
      {
        edge_id: `control-edge-${offset}`,
        edge_type: "control_applies_to_lane",
        source_node_id: `control-${offset}`,
        target_node_id: `lane-${offset}`,
        confidence: 0.93,
      },
    ],
  };
}

function frame(index: number): SceneFrame {
  return {
    frame_id: `frame-${String(index).padStart(2, "0")}`,
    segment_id: "synthetic-intersection",
    timestamp_ns: String(1_725_000_000_000_000_000n + BigInt(index)),
    cameras: [
      {
        slot: "front-center",
        label: "Front center",
        artifact_manifest_sha256: HASH,
        restriction_reason: null,
      },
      {
        slot: "front-left",
        label: "Front left",
        artifact_manifest_sha256: null,
        restriction_reason: "Licensed image excluded from this export.",
      },
    ],
    ground_truth: graph(index),
    baseline: graph(index + 2),
    candidate: graph(index + 1),
  };
}

export const SCENE_DETAIL: SceneBundleDetail = {
  schema_version: "junctionlens.api-scene-bundle.v1",
  manifest_sha256: "c".repeat(64),
  bundle: {
    schema_version: "junctionlens.scene-bundle.v1",
    title: "Synthetic intersection control regression",
    decision_manifest_sha256: "d".repeat(64),
    license_notice: "Synthetic unrestricted cameras",
    frames: [frame(0), frame(1)],
  },
  decision: {
    schema_version: "junctionlens.gate-decision.v1",
    decision_sha256: DECISION_HASH,
    status: "FAIL_REGRESSION",
    integrity_reason_codes: [],
    infrastructure_reason_codes: [],
    performance_reason_codes: [],
    cells: [
      {
        status: "FAIL_REGRESSION",
        reason_code: "GATE_REGRESSION_CI_BELOW_MARGIN",
      },
    ],
  },
};

export const SCENE_PAGE = {
  schema_version: "junctionlens.api-artifact-page.v1",
  items: [{ manifest_sha256: SCENE_DETAIL.manifest_sha256 }],
};

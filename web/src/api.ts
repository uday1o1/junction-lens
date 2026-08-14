export type ScenePoint = Readonly<{ x: number; y: number }>;

export type SceneLane = Readonly<{
  node_id: string;
  points: readonly ScenePoint[];
  confidence: number;
}>;

export type SceneControl = Readonly<{
  node_id: string;
  x: number;
  y: number;
  control_type: string;
  state: string;
  confidence: number;
}>;

export type SceneEdge = Readonly<{
  edge_id: string;
  edge_type: "lane_successor" | "control_applies_to_lane";
  source_node_id: string;
  target_node_id: string;
  confidence: number;
}>;

export type SceneGraphLayer = Readonly<{
  lanes: readonly SceneLane[];
  controls: readonly SceneControl[];
  edges: readonly SceneEdge[];
}>;

export type SceneCamera = Readonly<{
  slot: string;
  label: string;
  artifact_manifest_sha256: string | null;
  restriction_reason: string | null;
}>;

export type SceneFrame = Readonly<{
  frame_id: string;
  segment_id: string;
  timestamp_ns: string;
  cameras: readonly SceneCamera[];
  ground_truth: SceneGraphLayer;
  baseline: SceneGraphLayer;
  candidate: SceneGraphLayer;
}>;

export type SceneBundle = Readonly<{
  schema_version: "junctionlens.scene-bundle.v1";
  title: string;
  decision_manifest_sha256: string;
  license_notice: string;
  frames: readonly SceneFrame[];
}>;

export type Decision = Readonly<{
  schema_version: "junctionlens.gate-decision.v1";
  decision_sha256: string;
  status: string;
  integrity_reason_codes?: readonly string[];
  infrastructure_reason_codes?: readonly string[];
  performance_reason_codes?: readonly string[];
  cells?: readonly Readonly<Record<string, unknown>>[];
}>;

export type SceneBundleDetail = Readonly<{
  schema_version: "junctionlens.api-scene-bundle.v1";
  manifest_sha256: string;
  bundle: SceneBundle;
  decision: Decision;
}>;

const HASH = /^[0-9a-f]{64}$/u;
const DECIMAL = /^(0|[1-9][0-9]*)$/u;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be nonempty text`);
  }
  return value;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
  return value;
}

function probability(value: unknown, label: string): number {
  const parsed = finite(value, label);
  if (parsed < 0 || parsed > 1) {
    throw new Error(`${label} must be a probability`);
  }
  return parsed;
}

function hash(value: unknown, label: string): string {
  const parsed = text(value, label);
  if (!HASH.test(parsed)) {
    throw new Error(`${label} must be a lowercase SHA-256`);
  }
  return parsed;
}

function decimal(value: unknown, label: string): string {
  const parsed = text(value, label);
  if (!DECIMAL.test(parsed)) {
    throw new Error(`${label} must be a canonical decimal string`);
  }
  return parsed;
}

function array(value: unknown, label: string): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  return value;
}

function point(value: unknown): ScenePoint {
  if (!isObject(value)) throw new Error("scene point must be an object");
  return { x: finite(value.x, "point x"), y: finite(value.y, "point y") };
}

function lane(value: unknown): SceneLane {
  if (!isObject(value)) throw new Error("scene lane must be an object");
  const points = array(value.points, "lane points").map(point);
  if (points.length < 2)
    throw new Error("scene lane needs at least two points");
  return {
    node_id: text(value.node_id, "lane identity"),
    points,
    confidence: probability(value.confidence, "lane confidence"),
  };
}

function control(value: unknown): SceneControl {
  if (!isObject(value)) throw new Error("scene control must be an object");
  return {
    node_id: text(value.node_id, "control identity"),
    x: finite(value.x, "control x"),
    y: finite(value.y, "control y"),
    control_type: text(value.control_type, "control type"),
    state: text(value.state, "control state"),
    confidence: probability(value.confidence, "control confidence"),
  };
}

function edge(value: unknown): SceneEdge {
  if (!isObject(value)) throw new Error("scene edge must be an object");
  const edgeType = text(value.edge_type, "edge type");
  if (edgeType !== "lane_successor" && edgeType !== "control_applies_to_lane") {
    throw new Error("scene edge type is unsupported");
  }
  return {
    edge_id: text(value.edge_id, "edge identity"),
    edge_type: edgeType,
    source_node_id: text(value.source_node_id, "edge source"),
    target_node_id: text(value.target_node_id, "edge target"),
    confidence: probability(value.confidence, "edge confidence"),
  };
}

function graph(value: unknown): SceneGraphLayer {
  if (!isObject(value)) throw new Error("scene graph must be an object");
  return {
    lanes: array(value.lanes, "graph lanes").map(lane),
    controls: array(value.controls, "graph controls").map(control),
    edges: array(value.edges, "graph edges").map(edge),
  };
}

function camera(value: unknown): SceneCamera {
  if (!isObject(value)) throw new Error("scene camera must be an object");
  const artifact = value.artifact_manifest_sha256;
  const reason = value.restriction_reason;
  if ((artifact === null) === (reason === null)) {
    throw new Error("camera must be available or restricted");
  }
  return {
    slot: text(value.slot, "camera slot"),
    label: text(value.label, "camera label"),
    artifact_manifest_sha256:
      artifact === null ? null : hash(artifact, "camera artifact"),
    restriction_reason:
      reason === null ? null : text(reason, "camera restriction"),
  };
}

function frame(value: unknown): SceneFrame {
  if (!isObject(value)) throw new Error("scene frame must be an object");
  return {
    frame_id: text(value.frame_id, "frame identity"),
    segment_id: text(value.segment_id, "segment identity"),
    timestamp_ns: decimal(value.timestamp_ns, "frame timestamp"),
    cameras: array(value.cameras, "frame cameras").map(camera),
    ground_truth: graph(value.ground_truth),
    baseline: graph(value.baseline),
    candidate: graph(value.candidate),
  };
}

function decision(value: unknown): Decision {
  if (!isObject(value)) throw new Error("persisted decision must be an object");
  if (value.schema_version !== "junctionlens.gate-decision.v1") {
    throw new Error("persisted decision schema is unsupported");
  }
  return {
    ...value,
    schema_version: "junctionlens.gate-decision.v1",
    decision_sha256: hash(value.decision_sha256, "decision identity"),
    status: text(value.status, "decision status"),
  };
}

function sceneDetail(value: unknown): SceneBundleDetail {
  if (
    !isObject(value) ||
    value.schema_version !== "junctionlens.api-scene-bundle.v1"
  ) {
    throw new Error("scene API response schema is unsupported");
  }
  if (
    !isObject(value.bundle) ||
    value.bundle.schema_version !== "junctionlens.scene-bundle.v1"
  ) {
    throw new Error("scene bundle schema is unsupported");
  }
  const frames = array(value.bundle.frames, "scene frames").map(frame);
  if (frames.length === 0) throw new Error("scene bundle is empty");
  return {
    schema_version: "junctionlens.api-scene-bundle.v1",
    manifest_sha256: hash(value.manifest_sha256, "scene manifest"),
    bundle: {
      schema_version: "junctionlens.scene-bundle.v1",
      title: text(value.bundle.title, "scene title"),
      decision_manifest_sha256: hash(
        value.bundle.decision_manifest_sha256,
        "scene decision manifest",
      ),
      license_notice: text(value.bundle.license_notice, "scene license notice"),
      frames,
    },
    decision: decision(value.decision),
  };
}

async function getJson(url: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok)
    throw new Error(`evidence API returned HTTP ${response.status}`);
  return response.json();
}

export async function loadScene(
  signal: AbortSignal,
): Promise<SceneBundleDetail | null> {
  const requested = new URLSearchParams(window.location.search).get("scene");
  let manifest = requested;
  if (manifest === null) {
    const page = await getJson("/api/v1/scenes?limit=1", signal);
    if (
      !isObject(page) ||
      page.schema_version !== "junctionlens.api-artifact-page.v1"
    ) {
      throw new Error("scene index schema is unsupported");
    }
    const items = array(page.items, "scene index items");
    if (items.length === 0) return null;
    const first = items[0];
    if (!isObject(first)) throw new Error("scene index item is invalid");
    manifest = hash(first.manifest_sha256, "scene index manifest");
  } else {
    manifest = hash(manifest, "requested scene manifest");
  }
  return sceneDetail(await getJson(`/api/v1/scenes/${manifest}`, signal));
}

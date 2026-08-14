/** Lossless ProtoJSON handling for unsigned 64-bit graph identifiers. */

declare const unsigned64Brand: unique symbol;

export type Unsigned64String = string & { readonly [unsigned64Brand]: true };

const unsigned64Pattern = /^(0|[1-9][0-9]{0,19})$/;
const maximumUnsigned64 = 18_446_744_073_709_551_615n;

export function unsigned64String(
  value: unknown,
  path: string,
): Unsigned64String {
  if (
    typeof value !== "string" ||
    !unsigned64Pattern.test(value) ||
    BigInt(value) > maximumUnsigned64
  ) {
    throw new TypeError(`${path} must be a canonical uint64 decimal string`);
  }
  return value as Unsigned64String;
}

type JsonRecord = Record<string, unknown>;

function record(value: unknown, path: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  return value as JsonRecord;
}

function records(value: unknown, path: string): JsonRecord[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${path} must be an array`);
  }
  return value.map((item, index) => record(item, `${path}[${index}]`));
}

export interface GraphIdentityProjection {
  readonly nodeIds: readonly Unsigned64String[];
  readonly edgeIds: readonly Unsigned64String[];
  readonly trackIds: readonly Unsigned64String[];
}

export function graphIdentityProjection(
  protoJson: unknown,
): GraphIdentityProjection {
  const envelope = record(protoJson, "envelope");
  const graph = record(envelope.graph, "graph");
  const nodes = [
    ...records(graph.lanes, "graph.lanes"),
    ...records(graph.traffic_controls, "graph.traffic_controls"),
    ...records(graph.road_areas, "graph.road_areas"),
  ];
  const edges = records(graph.edges, "graph.edges");
  const tracks = records(graph.tracks, "graph.tracks");
  return {
    nodeIds: nodes.map((node, index) =>
      unsigned64String(node.node_id, `nodes[${index}].node_id`),
    ),
    edgeIds: edges.map((edge, index) =>
      unsigned64String(edge.edge_id, `edges[${index}].edge_id`),
    ),
    trackIds: tracks.map((track, index) =>
      unsigned64String(track.track_id, `tracks[${index}].track_id`),
    ),
  };
}

import { useId } from "react";

import type { SceneFrame, SceneGraphLayer, ScenePoint } from "./api";

export type LayerVisibility = Readonly<{
  groundTruth: boolean;
  baseline: boolean;
  candidate: boolean;
}>;

type NamedLayer = Readonly<{
  id: "ground-truth" | "baseline" | "candidate";
  label: string;
  graph: SceneGraphLayer;
  visible: boolean;
}>;

const WIDTH = 760;
const HEIGHT = 560;
const PADDING = 48;

function middle(points: readonly ScenePoint[]): ScenePoint {
  return points[Math.floor(points.length / 2)] ?? { x: 0, y: 0 };
}

export function GraphView({
  frame,
  visibility,
}: Readonly<{ frame: SceneFrame; visibility: LayerVisibility }>) {
  const titleId = useId();
  const descriptionId = useId();
  const markerPrefix = useId().replaceAll(":", "");
  const layers: readonly NamedLayer[] = [
    {
      id: "ground-truth",
      label: "Ground truth",
      graph: frame.ground_truth,
      visible: visibility.groundTruth,
    },
    {
      id: "baseline",
      label: "Baseline",
      graph: frame.baseline,
      visible: visibility.baseline,
    },
    {
      id: "candidate",
      label: "Candidate",
      graph: frame.candidate,
      visible: visibility.candidate,
    },
  ];
  const visible = layers.filter((layer) => layer.visible);
  const points = visible.flatMap((layer) => [
    ...layer.graph.lanes.flatMap((lane) => lane.points),
    ...layer.graph.controls.map((control) => ({ x: control.x, y: control.y })),
  ]);
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs, -5);
  const maxX = Math.max(...xs, 5);
  const minY = Math.min(...ys, -5);
  const maxY = Math.max(...ys, 5);
  const rangeX = Math.max(maxX - minX, 1);
  const rangeY = Math.max(maxY - minY, 1);
  const project = (point: ScenePoint) => ({
    x: PADDING + ((point.x - minX) / rangeX) * (WIDTH - PADDING * 2),
    y: HEIGHT - PADDING - ((point.y - minY) / rangeY) * (HEIGHT - PADDING * 2),
  });

  return (
    <svg
      aria-describedby={descriptionId}
      aria-labelledby={titleId}
      className="graph-canvas"
      role="img"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
    >
      <title id={titleId}>Bird's-eye lane and control graph comparison</title>
      <desc id={descriptionId}>
        Visible prediction layers use distinct colors and line patterns.
        Arrowheads show directed graph edges.
      </desc>
      <defs>
        {layers.map((layer) => (
          <marker
            id={`${markerPrefix}-${layer.id}-arrow`}
            key={layer.id}
            markerHeight="8"
            markerWidth="8"
            orient="auto"
            refX="7"
            refY="4"
            viewBox="0 0 8 8"
          >
            <path
              className={`arrow-fill layer-${layer.id}`}
              d="M 0 0 L 8 4 L 0 8 z"
            />
          </marker>
        ))}
        <pattern
          height="32"
          id={`${markerPrefix}-grid`}
          patternUnits="userSpaceOnUse"
          width="32"
        >
          <path className="graph-grid-line" d="M 32 0 L 0 0 0 32" fill="none" />
        </pattern>
      </defs>
      <rect fill={`url(#${markerPrefix}-grid)`} height={HEIGHT} width={WIDTH} />
      <line
        className="ego-axis"
        x1={WIDTH / 2}
        x2={WIDTH / 2}
        y1={24}
        y2={HEIGHT - 24}
      />
      {visible.map((layer) => {
        const nodes = new Map<string, ScenePoint>();
        for (const lane of layer.graph.lanes)
          nodes.set(lane.node_id, middle(lane.points));
        for (const control of layer.graph.controls) {
          nodes.set(control.node_id, { x: control.x, y: control.y });
        }
        return (
          <g
            className={`graph-layer layer-${layer.id}`}
            data-testid={`layer-${layer.id}`}
            key={layer.id}
          >
            {layer.graph.edges.map((edge) => {
              const source = nodes.get(edge.source_node_id);
              const target = nodes.get(edge.target_node_id);
              if (source === undefined || target === undefined) return null;
              const from = project(source);
              const to = project(target);
              return (
                <line
                  className={`graph-edge edge-${edge.edge_type}`}
                  data-edge-type={edge.edge_type}
                  key={edge.edge_id}
                  markerEnd={`url(#${markerPrefix}-${layer.id}-arrow)`}
                  x1={from.x}
                  x2={to.x}
                  y1={from.y}
                  y2={to.y}
                >
                  <title>{`${layer.label} ${edge.edge_type.replaceAll("_", " ")}, confidence ${edge.confidence.toFixed(2)}`}</title>
                </line>
              );
            })}
            {layer.graph.lanes.map((lane) => {
              const path = lane.points
                .map((point, index) => {
                  const location = project(point);
                  return `${index === 0 ? "M" : "L"} ${location.x.toFixed(2)} ${location.y.toFixed(2)}`;
                })
                .join(" ");
              return (
                <path
                  className="graph-lane"
                  d={path}
                  fill="none"
                  key={lane.node_id}
                >
                  <title>{`${layer.label} lane ${lane.node_id}, confidence ${lane.confidence.toFixed(2)}`}</title>
                </path>
              );
            })}
            {layer.graph.controls.map((control) => {
              const location = project({ x: control.x, y: control.y });
              return (
                <g
                  className="graph-control"
                  key={control.node_id}
                  transform={`translate(${location.x} ${location.y})`}
                >
                  <rect height="18" rx="3" width="18" x="-9" y="-9" />
                  <path d="M -4 0 L -1 4 L 5 -5" fill="none" />
                  <title>{`${layer.label} ${control.control_type.replaceAll("_", " ")} ${control.state}, confidence ${control.confidence.toFixed(2)}`}</title>
                </g>
              );
            })}
          </g>
        );
      })}
      <g
        aria-hidden="true"
        className="ego-marker"
        transform={`translate(${WIDTH / 2} ${HEIGHT - 38})`}
      >
        <path d="M 0 -16 L 11 12 L 0 7 L -11 12 z" />
      </g>
    </svg>
  );
}

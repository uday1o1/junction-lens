import { useEffect, useState } from "react";

import { loadScene } from "./api";
import type { SceneBundleDetail, SceneCamera } from "./api";
import { presentDecision } from "./decision";
import { GraphView } from "./GraphView";
import type { LayerVisibility } from "./GraphView";

type LoadState =
  | Readonly<{ state: "loading" }>
  | Readonly<{ state: "empty" }>
  | Readonly<{ state: "error"; message: string }>
  | Readonly<{ state: "ready"; detail: SceneBundleDetail }>;

function CameraCard({
  camera,
  frameId,
}: Readonly<{ camera: SceneCamera; frameId: string }>) {
  if (camera.artifact_manifest_sha256 === null) {
    return (
      <article
        aria-label={`${camera.label} image restricted`}
        className="camera-card camera-restricted"
      >
        <div aria-hidden="true" className="restricted-mark">
          Restricted
        </div>
        <h3>{camera.label}</h3>
        <p>{camera.restriction_reason}</p>
        <span className="camera-state">Image not exported</span>
      </article>
    );
  }
  return (
    <article className="camera-card">
      <img
        alt={`${camera.label}, synchronized source view for frame ${frameId}`}
        src={`/api/v1/images/${camera.artifact_manifest_sha256}`}
      />
      <div className="camera-caption">
        <h3>{camera.label}</h3>
        <span className="camera-state">Available</span>
      </div>
    </article>
  );
}

function LoadingState() {
  return (
    <main aria-busy="true" aria-live="polite" className="state-page">
      <span className="eyebrow">Local evidence</span>
      <h1>Loading registered scene</h1>
      <div aria-hidden="true" className="loading-bar" />
      <p>Verifying the immutable scene and persisted decision.</p>
    </main>
  );
}

function EmptyState() {
  return (
    <main className="state-page">
      <span className="eyebrow">No registered scenes</span>
      <h1>The evidence viewer is ready</h1>
      <p>
        Register a counterexample scene bundle to inspect graph differences
        here.
      </p>
    </main>
  );
}

function ErrorState({ message }: Readonly<{ message: string }>) {
  return (
    <main className="state-page" role="alert">
      <span className="eyebrow">Evidence unavailable</span>
      <h1>The scene could not be opened</h1>
      <p>{message}</p>
      <p>The release decision was not recalculated.</p>
    </main>
  );
}

function Viewer({ detail }: Readonly<{ detail: SceneBundleDetail }>) {
  const [frameIndex, setFrameIndex] = useState(0);
  const [visibility, setVisibility] = useState<LayerVisibility>({
    groundTruth: true,
    baseline: true,
    candidate: true,
  });
  useEffect(() => {
    const navigate = (event: KeyboardEvent) => {
      const delta =
        event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
      if (delta === 0) return;
      event.preventDefault();
      setFrameIndex((current) =>
        Math.max(0, Math.min(detail.bundle.frames.length - 1, current + delta)),
      );
    };
    window.addEventListener("keydown", navigate);
    return () => window.removeEventListener("keydown", navigate);
  }, [detail.bundle.frames.length]);
  const frame = detail.bundle.frames[frameIndex];
  if (frame === undefined)
    return <ErrorState message="The selected frame is missing." />;
  const decision = presentDecision(detail.decision);
  const setBoundedFrame = (next: number) => {
    setFrameIndex(Math.max(0, Math.min(detail.bundle.frames.length - 1, next)));
  };
  const toggle = (key: keyof LayerVisibility) => {
    setVisibility((current) => ({ ...current, [key]: !current[key] }));
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <a
          aria-label="JunctionLens evidence viewer home"
          className="brand"
          href="/"
        >
          <span aria-hidden="true" className="brand-mark">
            JL
          </span>
          <span>JunctionLens</span>
        </a>
        <div className="evidence-identity">
          <span>Scene evidence</span>
          <code title={detail.manifest_sha256}>
            {detail.manifest_sha256.slice(0, 12)}
          </code>
        </div>
      </header>

      <main className="workspace" tabIndex={-1}>
        <section aria-labelledby="scene-title" className="scene-heading">
          <div>
            <span className="eyebrow">Control-graph comparison</span>
            <h1 id="scene-title">{detail.bundle.title}</h1>
            <p>
              Segment <strong>{frame.segment_id}</strong> · frame{" "}
              {frameIndex + 1} of {detail.bundle.frames.length}
            </p>
          </div>
          <aside
            aria-label="Persisted release decision"
            className={`decision-card tone-${decision.tone}`}
          >
            <span>Persisted decision</span>
            <strong>{decision.label}</strong>
            <code title={detail.decision.decision_sha256}>
              {detail.decision.decision_sha256.slice(0, 12)}
            </code>
            {decision.reasons.length > 0 ? (
              <details>
                <summary>
                  {decision.reasons.length} reason code
                  {decision.reasons.length === 1 ? "" : "s"}
                </summary>
                <ul>
                  {decision.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </details>
            ) : null}
          </aside>
        </section>

        <section aria-labelledby="camera-title" className="panel camera-panel">
          <div className="panel-heading">
            <div>
              <span className="section-index">01</span>
              <h2 id="camera-title">Synchronized cameras</h2>
            </div>
            <span className="license-label">
              {detail.bundle.license_notice}
            </span>
          </div>
          <div className="camera-grid">
            {frame.cameras.map((camera) => (
              <CameraCard
                camera={camera}
                frameId={frame.frame_id}
                key={camera.slot}
              />
            ))}
          </div>
        </section>

        <section aria-labelledby="graph-title" className="panel graph-panel">
          <div className="panel-heading graph-heading">
            <div>
              <span className="section-index">02</span>
              <h2 id="graph-title">Bird's-eye graph</h2>
            </div>
            <fieldset className="layer-controls">
              <legend>Visible layers</legend>
              <label className="toggle-ground-truth">
                <input
                  checked={visibility.groundTruth}
                  onChange={() => toggle("groundTruth")}
                  type="checkbox"
                />
                <span aria-hidden="true" /> Ground truth
              </label>
              <label className="toggle-baseline">
                <input
                  checked={visibility.baseline}
                  onChange={() => toggle("baseline")}
                  type="checkbox"
                />
                <span aria-hidden="true" /> Baseline
              </label>
              <label className="toggle-candidate">
                <input
                  checked={visibility.candidate}
                  onChange={() => toggle("candidate")}
                  type="checkbox"
                />
                <span aria-hidden="true" /> Candidate
              </label>
            </fieldset>
          </div>
          <div className="graph-layout">
            <GraphView frame={frame} visibility={visibility} />
            <aside aria-label="Current graph counts" className="graph-summary">
              <span className="eyebrow">Current frame</span>
              <dl>
                <div>
                  <dt>Candidate lanes</dt>
                  <dd>{frame.candidate.lanes.length}</dd>
                </div>
                <div>
                  <dt>Baseline lanes</dt>
                  <dd>{frame.baseline.lanes.length}</dd>
                </div>
                <div>
                  <dt>Candidate controls</dt>
                  <dd>{frame.candidate.controls.length}</dd>
                </div>
                <div>
                  <dt>Candidate edges</dt>
                  <dd>{frame.candidate.edges.length}</dd>
                </div>
              </dl>
              <p>
                Arrowheads encode direction. Layer patterns remain distinct
                without color.
              </p>
            </aside>
          </div>
        </section>

        <nav aria-label="Frame navigation" className="frame-navigation">
          <button
            disabled={frameIndex === 0}
            onClick={() => setBoundedFrame(frameIndex - 1)}
            type="button"
          >
            <span aria-hidden="true">←</span> Previous frame
          </button>
          <div aria-live="polite">
            <strong>{frame.frame_id}</strong>
            <span>Timestamp {frame.timestamp_ns} ns</span>
          </div>
          <button
            disabled={frameIndex === detail.bundle.frames.length - 1}
            onClick={() => setBoundedFrame(frameIndex + 1)}
            type="button"
          >
            Next frame <span aria-hidden="true">→</span>
          </button>
        </nav>
      </main>
    </div>
  );
}

export function App() {
  const [loadState, setLoadState] = useState<LoadState>({ state: "loading" });
  useEffect(() => {
    const controller = new AbortController();
    void loadScene(controller.signal)
      .then((detail) => {
        setLoadState(
          detail === null ? { state: "empty" } : { state: "ready", detail },
        );
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setLoadState({
            state: "error",
            message:
              error instanceof Error ? error.message : "Unknown evidence error",
          });
        }
      });
    return () => controller.abort();
  }, []);
  if (loadState.state === "loading") return <LoadingState />;
  if (loadState.state === "empty") return <EmptyState />;
  if (loadState.state === "error")
    return <ErrorState message={loadState.message} />;
  return <Viewer detail={loadState.detail} />;
}

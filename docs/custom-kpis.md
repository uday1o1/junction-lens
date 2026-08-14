# Custom graph and temporal KPIs

JunctionLens computes custom graph and temporal KPIs only after freezing one `CustomMatchV1` artifact.
These pairs are custom associations and are never described as official OpenLane-V2 matches.

## Public workflow

Run the custom path with separate roots containing schema-valid ground-truth and prediction Protobuf envelopes.

```console
junctionlens evaluate \
  --ground-truth /path/to/ground-truth \
  --predictions /path/to/predictions \
  --artifact-root /path/to/artifacts
```

The command recursively loads `.pb` files, filters by graph role, and requires exactly one ground-truth and one prediction envelope for every exact frame key.
Missing, extra, or duplicate frames fail before matching.
Evaluator association never compares numeric prediction node IDs with ground-truth node IDs.

The command prints a versioned JSON receipt containing payload and manifest SHA-256 identities for the match map, frame KPI table, and segment KPI table.
The two KPI tables use Apache Parquet and retain numerator, denominator, support, direction, unit, status, and nullable value for every row.
An empty eligible population is stored as `value = null`, `denominator = 0`, and `status = EMPTY_DENOMINATOR`.

## CustomMatchV1 evidence

The host projects validated Protobuf geometry into a bounded JSON request.
The digest-verified Python 3.8 image calls the pinned upstream OpenLane-V2 v2.1.0 distance primitives directly.
The container has no network, a read-only root filesystem, no Linux capabilities, no-new-privileges, bounded memory and processes, and an unprivileged numeric user.

Each prediction association record preserves:

- Raw existence confidence.
- Complete quantized geometry key.
- Decoder-query index and final sorted position.
- Every ground-truth source ID and official-primitive distance cost.
- The strict threshold result for every pair.
- The selected pair or a stable rejection reason.
- A stable unmatched reason when no pair is selected.

The host validates the input hash, frozen policy, complete frame and object populations, ordering, candidate coverage, strict threshold decisions, and one-to-one selections before metrics can consume the artifact.

## Graph populations

Control-edge precision uses thresholded predicted edges whose matched endpoints belong to an eligible matched control population.
Control-edge recall uses ground-truth control-to-lane edges whose endpoints are both matched.
Wrong-control assignment selects the highest raw-probability thresholded outgoing edge with target node ID as the deterministic tie breaker.
Controls with no thresholded outgoing edge affect recall but do not enter the wrong-assignment denominator.

Reachability enumerates distinct matched ground-truth lane pairs connected within one to three directed successor hops.
Path blocking evaluates each eligible ground-truth source lane once.
Real cycles, merges, and splits remain valid graph structures.
Spurious successor rate includes only predicted successor edges whose endpoints are both matched.

Successor endpoint gaps use every thresholded predicted successor edge.
Median, P90, and P95 use the frozen linear type-7 quantile definition.
The above-threshold rate uses a strict gap greater than 2.0 meters.

## Temporal populations

Presence flicker uses each stable source-identified ground-truth node in a run of at least three consecutive annotated frames.
Each transition records whether the `CustomMatchV1` presence state changed.

Successor and control-edge flip rates use persistent ground-truth edges whose endpoint source identities exist in both adjacent frames.
An edge state is present only when both predictions are matched and their raw edge probability reaches 0.5.

Geometry jitter includes persistent matched lanes and road areas with valid ego poses.
Previous ground-truth and prediction geometry is transformed into the current vehicle frame, each geometry is resampled to 20 points, and the per-node sample is the root-mean-square norm of prediction-change residual minus ground-truth change.
The segment table reports median, P90, and P95 samples.

An ID switch is counted when a currently matched prediction changes track ID while the prior matched track remains active in the current prediction graph.
An unmatched intervening frame is a miss rather than a switch.
A reacquisition after the prior track is absent or terminated starts a new fragment.
The result is normalized as switches per 100 unique eligible ground-truth tracks.

## Reproducibility and limitations

The repository gate invokes the public workflow twice on the frozen synthetic corpus and requires identical payload hashes for all three artifacts.
The Python implementation is checked with hand-calculated clean and faulted graphs, empty populations, property bounds, temporal alternation, and live-track switch cases.
The C++20 `libjunctionlens_eval` primitives consume the same language-neutral golden values for ratios, linear quantiles, reachability, state flips, and endpoint gaps.

The repository-owned synthetic corpus contains only two frames in its temporal scene, so the public-path corpus deliberately exercises empty three-frame presence support.
Three-frame temporal fault goldens are tested in memory until licensed persistent-identity data is available.
No licensed-data temporal result is claimed by this milestone.

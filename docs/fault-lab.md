# Structural fault lab

The JunctionLens fault lab proves that the contract, graph metrics, temporal checks, calibration checks, runtime checks, and release evidence detect declared failures that object-detection summaries can hide.

`junctionlens fault` reads one clean immutable `prediction_bundle`, derives one child without modifying its parent, independently checks the observed invariant change, and writes a counterexample report.
The child manifest has the original bundle as its parent.
The counterexample report has both the original and child manifests as parents.

## Command

```bash
uv run junctionlens fault \
  --input <prediction-bundle-manifest-sha256> \
  --kind swap-control-edges \
  --seed 20260813 \
  --artifact-root artifacts
```

The command exits with code 0 only when the nearby clean control passes and the independent analyzer observes every required predicate for the declared transformation.
Malformed inputs, missing transformation preconditions, invalid clean controls, or missed detectors exit with code 2.
The output JSON identifies the original bundle, derived bundle, counterexample report, and stable primary reason code.

## Mandatory matrix

| Transformation | Expected primary reason code |
| --- | --- |
| `swap-control-edges` | `FAULT_CONTROL_ASSIGNMENT_CHANGED` |
| `drop-control-edges` | `FAULT_CONTROL_EDGE_RECALL_DEGRADED` |
| `drop-successor-chain` | `FAULT_REACHABILITY_DEGRADED` |
| `add-spurious-successors` | `FAULT_SPURIOUS_SUCCESSOR_ADDED` |
| `permute-nodes-correctly` | `CONTROL_GRAPH_PERMUTATION_INVARIANT` |
| `permute-nodes-without-edges` | `FAULT_TOPOLOGY_NODE_ORDER_MISMATCH` |
| `duplicate-node-id` | `CONTRACT_NODE_ID_DUPLICATE` |
| `dangling-edge` | `CONTRACT_EDGE_DANGLING` |
| `jitter-lanes` | `FAULT_LANE_GEOMETRY_JITTER` |
| `flip-boundaries` | `FAULT_LANE_BOUNDARIES_FLIPPED` |
| `corrupt-extrinsic` | `FAULT_CAMERA_EXTRINSIC_CHANGED` |
| `zero-uncertainty` | `CONTRACT_UNCERTAINTY_SCALE` |
| `inflate-uncertainty` | `FAULT_UNCERTAINTY_INFLATED` |
| `temperature-collapse` | `FAULT_CALIBRATION_OVERCONFIDENT` |
| `inject-nan` | `CONTRACT_NONFINITE` |
| `alternate-edge-confidence` | `FAULT_TEMPORAL_EDGE_FLIP` |
| `alternate-node-presence` | `FAULT_TEMPORAL_PRESENCE_FLICKER` |
| `reuse-track-id` | `FAULT_TRACK_ID_REUSED_AFTER_TERMINATION` |
| `force-provider-fallback` | `GATE_INTEGRITY_PROVIDER_FALLBACK` |
| `delay-postprocess` | `GATE_PERFORMANCE_P95_LATENCY_BUDGET` |
| `leak-buffer` | `GATE_PERFORMANCE_UNBOUNDED_MEMORY_GROWTH` |

The correct node permutation is a nonfault control.
Its wire order must change while its normalized graph and metrics remain invariant.
The original synthetic bundle is a second nearby control and must pass contract, provider, latency, memory, and history checks before any transformation runs.

## Flagship swapped-control evidence

The swapped-control transform exchanges the governed lanes for two distinct controls and preserves every node tensor and node geometry byte.
The counterexample report therefore records zero `DET_l` and `DET_t` deltas from structural node equivalence while control-edge recall falls from 1.0 to 0.0 and wrong-control assignment rises from 0.0 to 1.0.
The fault assertion marks `overall.control_edge_recall` as `FAIL_REGRESSION`.
It is explicitly labeled as a fault-cell assertion, not a V1 holdout acceptance run, so it cannot be mistaken for measured release evidence.

## Verification

```bash
./tools/jl test-integration-faults
./tools/jl verify-m7-3-local
```

The focused gate exercises all 21 transformations under seeds 20260813, 20260814, and 20260815 for 63 of 63 intended detections.
It also verifies the clean controls, monotonic control-edge drop severity, immutable parent links, byte-identical reruns, the public CLI, and the flagship swapped-control evidence.

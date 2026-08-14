# E1 learned topology

E1 keeps the E0 image encoder, ground-plane projection, shared query decoder, and node heads unchanged while replacing fitted geometric edge rules with learned directed heads.

The exact extension is frozen in [e1-joint-v1.yaml](../configs/model/e1-joint-v1.yaml), which binds the E0 profile by SHA-256.

## Directed heads

The lane-successor head scores every ordered lane pair with separate source and target projections plus endpoint displacement, distance, and directed heading features.

The diagonal is fixed to a finite rejection logit because the audited V1 dataset contract does not permit lane self-edges.

The control-applicability head uses separate control and lane projections plus the projected lane endpoints relative to the normalized `FRONT_CENTER` control box.

Invalid projected endpoints retain explicit visibility features instead of being silently treated as a valid location.

No nearest-neighbor rule overwrites either learned matrix.

## Matching and objective

Lanes, controls, and areas retain separate Hungarian node assignment.

The matched target identities induce directed query-indexed edge targets, so a query permutation changes both axes consistently.

Only pairs whose endpoints are matched participate in topology loss, and unmatched node queries remain supervised by their node existence heads.

Lane-successor and control-to-lane edges use separately logged focal losses with positive weights scanned only from `model_training`.

Positive lane-successor edges also receive the declared endpoint-continuity loss.

The combined E1 objective retains every E0 node and uncertainty loss and adds topology weights 3.0, 5.0, and 1.0 exactly as specified in the build plan.

## Axis contract

The internal control tensor is always control-major and lane-minor with shape `[control, lane]`.

The official OpenLane adapter transposes it exactly once into lane-major and traffic-minor shape `[lane, traffic]`.

Both conversion directions require declared lane and control counts, so the asymmetric two-control by three-lane seeded defect fails with a stable ordering exception instead of being accepted by shape coincidence.

## Learning diagnostics

Run the public hardware-independent diagnostic with:

```bash
uv run junctionlens model verify-topology \
  --output artifacts/m5/topology-diagnostic.json
```

The `oracle-nodes` mode fits the production topology heads from fixed node features and reports topology learning in isolation.

The `predicted-nodes` mode uses a nonidentity lane and control assignment, optimizes the predicted query representations with the production heads, and proves that topology gradients reach those representations.

Both modes use seed 20260813 and must reach at least 0.95 F1 for lane-successor and control-to-lane edges within 5,000 steps.

Oracle-node results are diagnostic evidence and must never be presented as end-to-end perception quality.

Run the complete local package gate with:

```bash
./tools/jl verify-m5-1-local
```

The licensed full-corpus E1 experiment and E0 comparison are the separate M5.2 target gate described in [e1-experiment.md](e1-experiment.md).

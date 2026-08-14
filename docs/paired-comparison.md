# Paired comparison and release decisions

`junctionlens compare` is the only public workflow that turns two immutable model evidence arms into a V1 release decision.
It pairs baseline and candidate evidence on exact frame tokens, materializes the frozen slice values, rebuilds pooled primitives by segment, applies the frozen charter, and stores the resulting evidence graph in the local registry.

## Input contract

Both `--baseline` and `--candidate` identify `prediction_bundle` artifact manifests in the selected registry.
Their payload media type is `application/vnd.junctionlens.comparison-arm+json` and their schema version is `junctionlens.comparison-arm.v1`.

Each arm declares its dataset, split, preprocessing, postprocessing, evaluator, metric-registry, slice-registry, integrity, and runtime identities.
Each arm manifest must link to at least one verified parent artifact so the registry can prove its source lineage.
Each frame contains its stable frame token, segment ID, timestamp, authoritative slice values, and metric primitives.
Ratio primitives contain numerators and denominators, average-precision primitives contain ground-truth counts and ranked predictions, and adaptive-ECE primitives contain confidence and correctness observations.
Runtime metrics contain the ten independently validated paired trial blocks and remain outside the segment bootstrap.

The comparison rejects duplicate keys, duplicate frame tokens, nonfinite required values, unknown fields, malformed primitives, and unverified registry objects.
These input-format failures exit with code 2 because no trustworthy diagnostic evidence can be derived from them.

## Exact pairing and integrity

The engine compares complete frame-token sets before statistical evaluation.
It also verifies the segment ID, timestamp, slice values, metric eligibility, ground-truth support, frozen registry hashes, split identity, and runtime trial identities for each paired input.
A mismatch produces a persisted `FAIL_INTEGRITY` or `BLOCKED_INFRASTRUCTURE` decision and a nonzero exit after the diagnostic artifacts have been written.
The engine never turns an input mismatch into a smaller passing population.

For each metric-and-slice charter cell, the engine groups eligible frames by `segment_id`.
Ratio numerators and denominators are summed inside each segment and pooled again inside every bootstrap draw.
Average-precision predictions receive frame-qualified identities before replicate-qualified identities are added by the bootstrap.
Nonlinear calibration observations are reconstructed for every draw rather than averaging per-segment final scores.

The comparison uses 10,000 deterministic replicates with seed 20260813.
Every gating cell receives the charter's Bonferroni-adjusted two-sided interval.
The decision uses a float64 roundoff tolerance only at the exact declared margin boundary.

## Command

```bash
uv run junctionlens compare \
  --baseline <baseline-manifest-sha256> \
  --candidate <candidate-manifest-sha256> \
  --charter configs/gates/acceptance-v1.yaml \
  --artifact-root artifacts
```

Exit code 0 means the persisted release status is `PASS`.
Exit code 3 means a complete diagnostic comparison was persisted with another release status.
Exit code 2 means the command could not construct trustworthy evidence.

The JSON receipt identifies these immutable artifacts:

- A deterministic `slices.parquet` logical table.
- Canonical gate evidence.
- The authoritative release decision.
- A deterministic `metrics.parquet` logical table.
- Read-only report data containing persisted cells, hypotheses, and reason codes.

Dashboard filters consume the report data and never recalculate release status.
The decision artifact remains the only release authority.

## Local verification

```bash
./tools/jl test-integration-comparison
./tools/jl verify-m7-2-local
```

The focused suite covers every V1 release status, both metric directions, the exact noninferiority boundary, family-wise adjustment, deterministic reruns, slice materialization, and a seeded frame-set mismatch.

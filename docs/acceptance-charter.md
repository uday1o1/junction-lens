# Acceptance charter and release decisions

JunctionLens separates the policy freeze from candidate evaluation so candidate results cannot influence V1 margins, support thresholds, or hypotheses.

The draft policy is [acceptance-v1.draft.yaml](../configs/gates/acceptance-v1.draft.yaml).

The final `acceptance-v1.yaml` is intentionally absent until the three measured E0 seed runs and M0 accelerated hardware baseline exist.

## Freeze evidence

The freeze input is an immutable `baseline_freeze_evidence` artifact in the repository content-addressed store.

It must identify E0, the three predeclared seeds, variability for every non-performance cell, the M0 hardware baseline, a pre-holdout power simulation, and the predeclared product-priority hash.

Only `model_training` and `model_selection` may appear as source partitions.

The command rejects any internal-holdout access, candidate-derived power input, missing seed, incomplete variability table, or proposed margin that is looser than the draft.

Run the freeze from a clean source checkout after the required measured artifacts exist:

```bash
uv run junctionlens gate freeze \
  --draft configs/gates/acceptance-v1.draft.yaml \
  --baseline-run artifacts://runs/<baseline-manifest-sha256> \
  --output configs/gates/acceptance-v1.yaml \
  --signer <local-signer-identity>
```

The output binds the source commit, draft, baseline artifact, metric registry, slice registry, signer, timestamp, and all freeze inputs with SHA-256 identities.

Its self-hash covers the complete policy body, and evaluation commands refuse a charter whose content no longer matches that hash.

## Statistical comparison

Accuracy and calibration cells pair candidate and baseline on the exact same segment IDs.

The implementation draws 10,000 segment clusters with PCG64 seed 20260813 and recomputes each pooled metric inside every draw.

Ratio estimators sum primitive numerators and denominators before division.

Average precision reconstructs the global ranked list and qualifies repeated segment and prediction IDs by draw occurrence.

Adaptive ECE reconstructs deterministic equal-count bins from replicate-qualified observations.

Fewer than 9,900 finite replicates makes the cell insufficient.

Runtime uses ten paired AB or BA trial blocks with 200 warmup frames and 2,000 measured frames per trial.

The balanced ten-block order is frozen in the charter, and evidence with a different order is blocked as an infrastructure mismatch.

Fewer than eight valid runtime pairs produces `BLOCKED_INFRASTRUCTURE`.

Every gating interval uses a two-sided type-7 percentile interval with Bonferroni alpha `0.05 / N`, where `N` is the frozen count of gating cells.

Higher-is-better cells use candidate minus baseline, while lower-is-better cells reverse that sign.

Runtime comparisons use the direction-normalized relative paired difference and remain outside the segment bootstrap.

## Decision semantics

A cell passes when its adjusted lower bound is at least the negative frozen margin.

A cell fails when its adjusted upper bound is below the negative frozen margin.

A cell is insufficient when the interval crosses the margin or its declared support is too small.

The persisted release status is exactly one of `PASS`, `FAIL_INTEGRITY`, `FAIL_REGRESSION`, `FAIL_PERFORMANCE`, `INSUFFICIENT_EVIDENCE`, or `BLOCKED_INFRASTRUCTURE`.

Integrity failure has highest precedence, followed by invalid infrastructure, performance failure, regression, and insufficient evidence.

Every non-passing cell includes a stable reason code, support counts, point estimate, adjusted interval, margin, and counterexample query.

Primary superiority hypotheses are reported separately because noninferiority does not prove improvement.

Apply a frozen charter with:

```bash
uv run junctionlens gate decide \
  --charter configs/gates/acceptance-v1.yaml \
  --evidence artifacts/<run>/gate-evidence.json \
  --output artifacts/<run>/release-decision.json
```

The command writes one immutable decision, returns exit code 3 for a valid non-passing decision, and refuses to replace an existing decision.

## Local package gate

Run the hardware-independent implementation gate with:

```bash
./tools/jl verify-m4-2-local
```

This gate covers pooled bootstrap behavior, pairing failures, finite-replicate handling, charter contamination controls, the public freeze and decide commands, and seeded pass, regression, insufficient, integrity, performance, and infrastructure outcomes.

The final V1 charter remains a target-only artifact until measured E0 variability and the M0 GPU baseline satisfy the freeze contract.

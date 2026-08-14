# E1 joint-topology experiment

Milestone 5.2 tests whether learned topology is worth retaining over the frozen E0 independent-linking baseline.
The study protocol and model outcome are separate states.
A valid study is accepted when its predeclared seed, partition isolation, artifact identities, evaluator identities, and immutable evidence all pass.
The outcome is `PROMOTED` only when every keep gate passes, and otherwise is `REJECTED_BY_KEEP_GATE` with E0 retained.

## Full-corpus training

E1 uses only seed 20260813 for screening and release selection.
The training command requires the checksum-registered full OpenLane-V2 corpus, the immutable training split, and an accepted topology diagnostic from Milestone 5.1.
It uses the exact E0 optimizer and node-loss recipe while adding the frozen E1 topology objective and model-training-only edge weights.

```console
junctionlens model train-e1 \
  --dataset-root "$OPENLANE_V2_ROOT" \
  --split-manifest artifacts/private/split-v1.json \
  --topology-diagnostic artifacts/m5/topology-diagnostic.json \
  --output-root artifacts/models/e1/20260813
```

Every epoch checkpoint is retained with deterministic random-number state, optimizer state, source commit, profile hashes, split identities, and training-only node and edge statistics.
Every frame record logs each unweighted node and topology loss, the weighted total, gradient norm, learning rates, and optimizer step.
Resume verifies the last checkpoint hash and every frozen run identity before restoring state.

## Frozen selection

Evaluate each retained checkpoint on `model_selection` and write the exact split hash with every score.
The selection rule is lane-control topology descending, official composite descending, negative log-likelihood ascending, then epoch ascending.

```console
junctionlens model select-e1 \
  --run-root artifacts/models/e1/20260813 \
  --scores artifacts/private/e1-selection-scores.json \
  --selection-split-manifest-sha256 "$SELECTION_SPLIT_SHA256"
```

The command verifies that every scored epoch exists and writes one immutable selection receipt.
It rejects repeated epochs, nonfinite metrics, split mismatches, missing checkpoints, and attempts to replace a different receipt.

## Keep-gate evidence

The comparison uses E0 and E1 seed 20260813 on the exact same model-selection manifest and source-frame manifest.
Evidence binds both selected checkpoint hashes, both selection receipts, both prediction manifests, both evaluation artifacts, and the official and custom evaluator configurations.
Internal holdout access must be exactly zero.

E1 is promoted only when all four conditions hold:

- `TOP_lt` improves by at least 0.02 absolute.
- `DET_l` decreases by no more than 0.01 absolute.
- `DET_t` decreases by no more than 0.01 absolute.
- Wrong-control assignment rate is strictly lower than E0.

```console
junctionlens model finalize-e1-study \
  --baseline-run-root artifacts/models/e0/20260813 \
  --candidate-run-root artifacts/models/e1/20260813 \
  --evidence artifacts/private/e1-study-evidence.json \
  --output artifacts/models/e1/study-report.json
```

The immutable report records each measured baseline, candidate, delta, threshold, decision, and stable reason code.
When any keep gate fails, the report remains a valid accepted study, records `REJECTED_BY_KEEP_GATE`, and selects `E0-independent` without weakening the threshold.

## Evidence boundary

Run the hardware-independent package gate with `./tools/jl verify-m5-2-local`.
Local tests prove strict selection, exact threshold boundaries, negative-result handling, contamination rejection, immutable reports, and fail-closed licensed-data entry points.
They do not claim E1 convergence, predictive improvement, or promotion.
Those claims require the registered licensed corpus and target GPU run.

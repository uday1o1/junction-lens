# E0 independent baseline

E0 is the frozen comparator for learned topology experiments.
It shares the multi-camera node architecture used by later candidates and does not contain learned lane-successor or control-to-lane heads.

## Architecture

The image encoder is a random-initialized EfficientNet-B0 feature pyramid with stride 8, 16, and 32 features projected to 128 channels.
Calibration maps current-frame camera features into the 200 by 160 JunctionLens ground-plane grid.
A masked learned softmax excludes invalid and geometrically invisible cameras from its normalization.
The projection module has separate slow reference and vectorized implementations with a 1e-6 local parity gate.

The shared four-layer transformer decoder uses 96 lane queries, 64 traffic-control queries, and 32 road-area queries at hidden dimension 256.
Type-specific heads predict existence, categories, geometry, marginal Laplace scales, and diagnostic track embeddings.
E0 ignores previous-frame inputs and does not claim temporal behavior.

## Independent topology rules

Lane-successor candidates use source-endpoint to target-startpoint distance and directed heading difference.
Self-edges are always excluded.
Control-to-lane candidates project each lane into the control source image and use the last visible lane point's distance to the half-open control box plus its image-up heading difference.
Control edges are emitted in canonical control-major, lane-minor order.

Both distance and heading thresholds are selected only from the model-training partition.
The bounded search maximizes edge F1 and resolves ties by the smallest distance threshold followed by the smallest heading threshold.
The fitted artifact records the training split hash, profile hash, observation hash, primitive counts, and selected thresholds.

## Training and selection

Run one predeclared seed only after the full licensed dataset and immutable split manifest are registered.

```console
junctionlens model train-e0 \
  --dataset-root "$OPENLANE_V2_ROOT" \
  --split-manifest artifacts/private/split-v1.json \
  --seed 20260813 \
  --output-root artifacts/models/e0/20260813
```

The command refuses an unregistered root and rejects any training-statistics request outside `model_training`.
It logs every unweighted head loss, weighted total, gradient norm, learning rate, frame identity, and optimizer step.
The implementation uses AdamW, the frozen backbone multiplier, 1,000-step warmup, cosine decay, gradient accumulation, clipping at 1.0, and the declared CUDA precision preference.

Every epoch is retained until model-selection evaluation writes exact topology, official-composite, and negative-log-likelihood scores.
The selection command orders epochs lexicographically by topology descending, official composite descending, NLL ascending, and epoch ascending.
Calibration and internal-holdout partitions are not accepted by the training or selection-statistics paths.

The baseline is complete only after all three predeclared seeds finish under one source commit, profile, training split, and selection split.
Finalization requires measured limitations and at least one measured failed example before it renders the model card.
Dataset-derived weights remain private pending a separate license review.

## Current evidence boundary

Local CPU tests establish strict configuration, backbone feature taps, projection parity, matching and loss gradients, threshold-fit isolation, deterministic selection, and fail-closed artifact assembly.
They do not establish full-corpus convergence, seed repeatability, predictive quality, or accelerated performance.
Those claims require the licensed dataset and target GPU qualification bundle.

# Calibration and runtime evidence

JunctionLens computes probability, geometry-uncertainty, and runtime evidence with deterministic populations and ordering.
Calibration metrics consume frozen `CustomMatchV1` outcomes rather than changing associations after observing metric results.

## Calibration workflow

Run the public calibration metric path with a bounded JSON input.

```console
junctionlens calibrate --input calibration-input.json
```

Binary Brier score is mean squared probability error.
Multiclass Brier score is the mean sum of squared class-probability errors.
Negative log likelihood clips only for evaluation to `[1e-7, 1 - 1e-7]` and reports how many evaluated probabilities crossed that clip boundary.

Adaptive ECE uses at most 15 equal-count bins.
Items sort by ascending confidence and then stable item ID, and earlier bins receive the remainder when the population is not divisible by the bin count.
Every bin reports count, mean confidence, mean correctness, and inclusive rank bounds.

AURC sorts by descending confidence and then stable item ID.
It integrates cumulative risk over coverage with the trapezoidal rule from the explicit origin `(0, 0)`.
The stable ID rule makes equal-confidence results reproducible and prevents input order from changing evidence.

Geometry coverage evaluates scalar coordinate residuals against the two-sided marginal Laplace interval.
The 90 percent half-width is `scale * factor * ln(10)` and the full interval width is twice that value.
Coverage includes residuals exactly on the boundary.
Median and P90 interval widths use linear type-7 quantiles.

## Runtime workflow

Run runtime evidence through the public evaluator command.

```console
junctionlens evaluate --runtime-input runtime-input.json
```

The runtime document declares the exact monotonic clock source used for every sample.
Every sample declares a phase, iteration, duration in milliseconds, and either `warmup` or `measured` kind.
Duplicate phase-kind-iteration identities are rejected.
Every reported phase must contain at least one measured sample.

Warmup and measured samples produce separate count, minimum, maximum, mean, median, P90, P95, and P99 records.
Throughput uses measured samples only and is the reciprocal of their mean duration.
Warmup values never enter measured latency or throughput statistics.

Profiler measurements are not runtime benchmark inputs.
Later runtime milestones attach hardware, provider-assignment, memory, and benchmark-validity manifests before these distributions can support a release claim.

## Verification scope

Analytic fixtures cover binary and multiclass Brier and NLL, clipping counts, ECE bins, deterministic AURC ties, exact Laplace boundaries, interval widths, and warmup separation.
The C++20 evaluator library independently checks binary Brier, NLL clipping, and marginal Laplace coverage against the same formulas.
This milestone establishes metric correctness and deterministic evidence structure, not a latency or calibration-quality claim for a trained model.

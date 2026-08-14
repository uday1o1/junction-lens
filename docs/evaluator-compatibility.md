# Official evaluator compatibility

The Python 3.8 compatibility image is the sole owner of official OpenLane-V2 v2.1 metric values in JunctionLens.
The container uses the untouched upstream evaluator source from commit `d731a26bdbf34723dd915ad525c2c2eca19ed8a1`.
The source archive, Python wheels, base image, build context, OCI index, platform manifest, and image config are hash or digest locked.
The image performs no operating-system package installation beyond its pinned base filesystem.

The historical OR-Tools 9.2.9972 wheel is no longer present on the owning live package index.
ADR 0004 records the constrained substitution of OR-Tools 9.3.10497 and the fixture gate that protects the metric behavior.
It also records the same-version headless OpenCV substitution that removes unused GUI dependencies from this server-only image.

## Build and verify

```sh
./tools/jl bootstrap-cpu
./tools/jl build-evaluator --check-lock
./tools/jl test-evaluator
./tools/jl verify-m3-1
```

The build command performs two independent no-cache Linux amd64 OCI exports.
It accepts the image only when the OCI index, platform manifest, and config digests match across both builds.
It then loads the qualified archive directly and verifies the daemon-visible image identity against the lock.

The host wrapper validates a bounded strict JSON schema before invoking Docker.
The evaluator runs as an unprivileged numeric user with no network, a read-only root, all capabilities dropped, `no-new-privileges`, process, CPU, and memory limits, a bounded scratch tmpfs, and a read-only input mount.
No externally supplied pickle enters the compatibility container.

The runner converts trusted JSON lists to the exact declared NumPy adjacency shapes before calling the untouched evaluator.
This includes `(0, 0)` lane topology and `(0, T)` lane-control topology for empty prediction sets.
Without that shape preservation, upstream NumPy infers a one-dimensional `(0,)` array and the v2.1 topology path raises an index error.

## Threshold-specific upstream artifacts

The versioned output preserves the matching arrays injected by OpenLane-V2 v2.1 for lane thresholds 1.0, 2.0, and 3.0 meters and traffic-element distance threshold 0.75.
Each artifact includes the upstream `idx_match_gt`, confidence array, and ten confidence thresholds plus lossless ground-truth and prediction ID arrays used to interpret those indices.
Unmatched upstream NaN indices are serialized as JSON `null`.

The untouched evaluator does not inject area matching arrays, so JunctionLens does not invent official area associations.
Later custom associations are separately versioned as `CustomMatchV1` and are never labeled official.

The modern host validates the returned environment, metric bounds, frame set, exact thresholds, source IDs, float32 confidences, artifact dimensions, match-index range, and one-to-one ground-truth use.
Every output records OpenCV runtime 5.0.0 and distribution `opencv-python-headless==5.0.0.93` alongside the other compatibility versions.
The public wrapper and a direct restricted container invocation read the same request and compare every returned JSON value with an absolute tolerance of 1e-12 for finite numbers and exact equality for JSON `null` behavior.

## Frozen fixture results

The following values were produced by the official compatibility image from repository-owned synthetic inputs.
They are correctness fixtures, not model-quality or runtime-performance results.

| Case | DET_l | DET_a | DET_t | TOP_ll | TOP_lt | OLUS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Perfect | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Empty scene | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.600000 |
| Empty predictions | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Partial predictions | 0.545455 | 0.500000 | 0.538462 | 0.000000 | 0.533333 | 0.462843 |
| Duplicate confidence | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Correctly permuted order | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Permuted nodes without matrices | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.906111 | 0.790380 |
| High-confidence false lane | 0.666667 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.933333 |
| Lane geometry corruption | 0.272727 | 1.000000 | 1.000000 | 0.000000 | 0.466667 | 0.591171 |
| Area geometry corruption | 1.000000 | 0.500000 | 1.000000 | 1.000000 | 1.000000 | 0.900000 |
| Traffic box corruption | 1.000000 | 1.000000 | 0.923077 | 1.000000 | 0.923810 | 0.976845 |
| Lane topology corruption | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 1.000000 | 0.941421 |
| Control topology corruption | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.923810 | 0.992230 |

The topology-only corruptions preserve all three detection components.
Correctly permuting prediction arrays with both adjacency axes preserves every metric and resolved upstream association.
Permuting prediction arrays without their matrices preserves node detection but degrades both topology components.
The high-confidence false lane remains unmatched at every official lane threshold and is normalized to JSON `null` in the artifact.

These are correctness results rather than performance measurements.
The verification report records metric values, artifact hashes, parity counts, and control states without mixing in throughput timing.

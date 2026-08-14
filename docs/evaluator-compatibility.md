# Official evaluator compatibility

The Python 3.8 compatibility image is the sole owner of official OpenLane-V2 v2.1 metric values in JunctionLens.
The container uses the untouched upstream evaluator source from commit `d731a26bdbf34723dd915ad525c2c2eca19ed8a1`.
The source archive, Python wheels, base image, Debian snapshot packages, build context, OCI index, platform manifest, and image config are hash or digest locked.

The historical OR-Tools 9.2.9972 wheel is no longer present on the owning live package index.
ADR 0004 records the constrained substitution of OR-Tools 9.3.10497 and the fixture gate that protects the metric behavior.

## Build and verify

```sh
./tools/jl bootstrap-cpu
./tools/jl build-evaluator --check-lock
./tools/jl test-evaluator
```

The build command performs two independent no-cache Linux amd64 OCI exports.
It accepts the image only when the OCI index, platform manifest, and config digests match across both builds.
It then loads the qualified archive directly and verifies the daemon-visible image identity against the lock.

The host wrapper validates a bounded strict JSON schema before invoking Docker.
The evaluator runs as an unprivileged numeric user with no network, a read-only root, all capabilities dropped, `no-new-privileges`, process, CPU, and memory limits, a bounded scratch tmpfs, and a read-only input mount.
No externally supplied pickle enters the compatibility container.

## Frozen fixture results

The following values were produced by the official compatibility image from repository-owned synthetic inputs.
They are correctness fixtures, not model-quality or runtime-performance results.

| Case | DET_l | DET_a | DET_t | TOP_ll | TOP_lt | OLUS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Perfect | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Lane geometry corruption | 0.272727 | 1.000000 | 1.000000 | 0.000000 | 0.466667 | 0.591171 |
| Area geometry corruption | 1.000000 | 0.500000 | 1.000000 | 1.000000 | 1.000000 | 0.900000 |
| Traffic box corruption | 1.000000 | 1.000000 | 0.923077 | 1.000000 | 0.923810 | 0.976845 |
| Lane topology corruption | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 1.000000 | 0.941421 |
| Control topology corruption | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.923810 | 0.992230 |

The topology-only corruptions preserve all three detection components.
The evaluator also returns threshold-specific upstream matching artifacts in the versioned JSON output.

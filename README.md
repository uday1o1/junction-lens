# JunctionLens

JunctionLens is a local-first research and model-development product for control-aware road-scene graphs.
Implementation is paused at the safe boundary recorded in `BUILD_PLAN.md`.
The unrestricted CPU product path is implemented through deterministic synthetic evidence.
Licensed-data, trained-model, accelerated-runtime, and final-release claims remain blocked until their named gates pass.

JunctionLens is not a vehicle controller, a safety case, or a certification product.

## Verified local bootstrap

```sh
./tools/jl bootstrap-cpu
./tools/jl verify-m0-1
```

The command uses repository-locked CPU toolchains and dependencies.
`junctionlens doctor --json` reports absent dataset, GPU, CUDA, and TensorRT capabilities separately instead of assuming they exist.

## Unrestricted demonstration

```sh
./tools/jl demo-synthetic
./tools/jl inspect-demo --artifact-root artifacts/demo
uv run --locked junctionlens serve --artifact-root artifacts/demo --open-browser
```

The demo compares 200 real procedural segments and 600 eligible lane-to-control edges.
It keeps baseline and candidate node detections identical while rotating only the candidate control associations.
The graph cells reject that candidate for the intended regression reason, and the independent fault lab detects `swap-control-edges` while its nearby clean control passes.

On a CPU-only machine, the overall release status truthfully remains `BLOCKED_INFRASTRUCTURE` because accelerated runtime evidence is absent.
See [Unrestricted synthetic demonstration](docs/synthetic-demo.md) for the generated evidence, report, viewer, and clean-checkout workflow.

## Dataset and official metrics

Licensed OpenLane-V2 data is never downloaded automatically or committed to this repository.
See [Dataset and license boundary](docs/dataset-and-license.md) for the explicit acknowledgment, checksum registration, and audit workflow.
See [OpenLane-V2 adapter](docs/openlane-adapter.md) for lazy image loading, canonical camera tensors, and official-devkit parity.
See [Data manifests and V1 splits](docs/data-manifests-and-splits.md) for streaming content identity, immutable storage, and the segment-isolated split contract.
See [Visual and statistical data audit](docs/data-audit.md) for private calibration overlays, BEV labels, aggregate distributions, and slice-support previews.

See [Acceptance charter and release decisions](docs/acceptance-charter.md) for the pre-holdout policy freeze, paired bootstrap, stable reason codes, and immutable decision workflow.
See [Paired comparison and release decisions](docs/paired-comparison.md) for exact-frame pairing, slice materialization, persisted report data, and the public comparison command.
See [Structural fault lab](docs/fault-lab.md) for the mandatory graph, geometry, calibration, temporal, and runtime corruption matrix.
See [Native inference runtime](docs/runtime.md) for the bounded C++ batch path, accelerated-provider design, preprocessing contract, output provenance, and parity evidence.
See [Remote GPU qualification](docs/gpu-qualification.md) for secure source synchronization, target preflight, resumable phases, and the consolidated hardware handoff.
See [Local evidence service](docs/local-evidence-service.md) for the loopback-only read API and production browser workflow.
See [Scene viewer](docs/scene-viewer.md) for synchronized camera and graph regression inspection.
See [Reproducible evidence reports](docs/evidence-reports.md) for deterministic public and acknowledged private offline bundles.
See [Supply-chain security](docs/supply-chain-security.md) for parser hardening, vulnerability policy, secret scanning, license inventory, and deterministic SBOM commands.

See [E1 learned topology](docs/e1-learned-topology.md) for the directed edge heads, canonical matrix ordering, topology objective, and oracle-node versus predicted-node diagnostics.

The official metric path runs the untouched OpenLane-V2 v2.1 evaluator in a reproducible restricted Python 3.8 image.
See [Official evaluator compatibility](docs/evaluator-compatibility.md) for its trust boundary, exact build command, and repository-owned corruption evidence.

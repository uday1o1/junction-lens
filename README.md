# JunctionLens

JunctionLens is a local-first research and model-development product for control-aware road-scene graphs.
It is under active implementation according to the acceptance gates in `BUILD_PLAN.md`.

The first verified workflow is established in Milestone 0.
Measured model, runtime, and release claims will be added only after their named evidence gates pass.

JunctionLens is not a vehicle controller, a safety case, or a certification product.

## Verified local bootstrap

```sh
./tools/jl bootstrap-cpu
./tools/jl verify-m0-1
```

The command uses repository-locked CPU toolchains and dependencies.
`junctionlens doctor --json` reports absent dataset, GPU, CUDA, and TensorRT capabilities separately instead of assuming they exist.

## Dataset and official metrics

Licensed OpenLane-V2 data is never downloaded automatically or committed to this repository.
See [Dataset and license boundary](docs/dataset-and-license.md) for the explicit acknowledgment, checksum registration, and audit workflow.
See [OpenLane-V2 adapter](docs/openlane-adapter.md) for lazy image loading, canonical camera tensors, and official-devkit parity.
See [Data manifests and V1 splits](docs/data-manifests-and-splits.md) for streaming content identity, immutable storage, and the segment-isolated split contract.
See [Visual and statistical data audit](docs/data-audit.md) for private calibration overlays, BEV labels, aggregate distributions, and slice-support previews.

See [Acceptance charter and release decisions](docs/acceptance-charter.md) for the pre-holdout policy freeze, paired bootstrap, stable reason codes, and immutable decision workflow.
See [Paired comparison and release decisions](docs/paired-comparison.md) for exact-frame pairing, slice materialization, persisted report data, and the public comparison command.
See [Structural fault lab](docs/fault-lab.md) for the mandatory graph, geometry, calibration, temporal, and runtime corruption matrix.
See [Native CPU inference runtime](docs/runtime.md) for the bounded C++ batch path, preprocessing contract, output provenance, and parity evidence.

See [E1 learned topology](docs/e1-learned-topology.md) for the directed edge heads, canonical matrix ordering, topology objective, and oracle-node versus predicted-node diagnostics.

The official metric path runs the untouched OpenLane-V2 v2.1 evaluator in a reproducible restricted Python 3.8 image.
See [Official evaluator compatibility](docs/evaluator-compatibility.md) for its trust boundary, exact build command, and repository-owned corruption evidence.

# JunctionLens Build Plan

Status: implementation in progress with the complete local core verified and the consolidated licensed-data and GPU qualification blocked on external prerequisites.

## Implementation stop status on 2026-08-14

This section records the exact safe-boundary state from which implementation resumes.
It does not weaken any milestone deliverable or acceptance gate below.
The work-state terms have the meanings defined in Section 28.1.

### Verified local evidence

The repository entry point `./tools/jl verify-local` completed with exit code 0 in 1,251.1 seconds on 2026-08-14 with an isolated local Docker context supplied through `DOCKER_CONTEXT`.
That gate exercised the public evaluator image build and fixtures, formatting, linting, static analysis, Python tests, native tests, sanitizers, 5,000 seeded fuzz iterations, web tests, security checks, reproducibility checks, integrations, reports, and the documented synthetic workflows.
The official-evaluator build now uses two independent fresh BuildKit 0.30.0 daemons from the locked OCI index `sha256:0168606be2315b7c807a03b3d8aa79beefdb31c98740cebdffdfeebf31190c9f`.
Both fresh builders reproduced config `502743806a0c85e837c42e0ffdb30e926e232093b02fad5282909c1754daea33`, platform manifest `82b77905f1fd2b2429aacdf9346f4d6e5d68b2007f9bf6d5570be9156bab812e`, OCI index `728c12638feda9eb0e856a178ff638a5cf02600d1983ad5073aea6343b9fe820`, and unchanged build-context hash `287bdaff92733cc4d7ba6e9ef7be9462d3d111edc42ab2d02043c969186df95d`.

| Plan scope | Local work package | Target-only gate | Exact status at this boundary |
| --- | --- | --- | --- |
| M0.1 | `IMPLEMENTED_LOCAL` | `ACCEPTED` | Toolchain, source, archive, runtime, and OCI locks plus truthful doctor probes are implemented and locally verified. |
| M0.2 and M0.3 | `IMPLEMENTED_LOCAL` | `DEFERRED_HARDWARE` | The evaluator, sample adapter, model spike, export, CPU runtime, and provider probes are verified locally, while licensed full-data and NVIDIA target evidence is not accepted. |
| M1.1 through M1.3 | `IMPLEMENTED_LOCAL` | `ACCEPTED` | The protobuf contract, validators, geometry core, and unrestricted synthetic corpus pass their complete local gates. |
| M2.1 through M2.3 | `IMPLEMENTED_LOCAL` | `DEFERRED_HARDWARE` | Adapter, manifest, split, statistical-audit, and visual-audit software is verified, while the licensed selector, frozen full-data manifest, and human visual inspection are not accepted. |
| M3.1 through M3.3 | `IMPLEMENTED_LOCAL` | `ACCEPTED` | Official fixture parity, graph and temporal KPIs, calibration evidence, and runtime KPI controls pass locally without a licensed-data performance claim. |
| M4.1 through M5.2 | `IMPLEMENTED_LOCAL` | `DEFERRED_HARDWARE` | E0, the acceptance charter, learned topology, and E1 experiment machinery pass local and synthetic gates, while full-data training, selection, and comparison evidence is not accepted. |
| M6.1 through M6.3 | `PENDING_LOCAL` | `BLOCKED` | Temporal fusion, calibration promotion, and the frozen E2/E3 study may begin only after the portfolio core target gate is accepted and its measured budget promotion gate passes. |
| M7.1 through M7.3 | `IMPLEMENTED_LOCAL` | `ACCEPTED` | Registry, comparison, decision, and mandatory seeded fault-lab paths pass their local acceptance gates. |
| M8.1 | `IMPLEMENTED_LOCAL` | `ACCEPTED` | The C++20 CPU runtime, metadata validation, parity, bounded streaming path, and CPU performance controls pass locally. |
| M8.2 and M8.3 | `IMPLEMENTED_LOCAL` | `DEFERRED_HARDWARE` | CUDA, conditional TensorRT, benchmark, profiler, stability, and fail-closed qualification automation is implemented, while no accelerated target result is accepted. |
| M9.1 through M9.3 | `IMPLEMENTED_LOCAL` | `ACCEPTED` | CLI, API, thin comparison viewer, and deterministic evidence reports pass local gates and real-browser synthetic inspection. |
| M10.1 and M10.2 | `IMPLEMENTED_LOCAL` | `ACCEPTED` | Security hardening, supply-chain checks, clean-checkout synthetic automation, and the complete local verification entry point pass. |
| Development-complete core checkpoint | `IMPLEMENTED_LOCAL` | `BLOCKED` | Every hardware-independent core package is implemented, but the checkpoint is not accepted without the consolidated licensed-data, GPU, profiler, and signed visual-review evidence. |
| Conditional TensorRT and richer-dashboard extensions | `PENDING_LOCAL` | `BLOCKED` | Work is held behind accepted core evidence and the Section 29 promotion order. |
| M11.1 and M11.2 | `PENDING_LOCAL` | `BLOCKED` | The frozen final matrix and independent reproduction require an accepted selected model and accelerated core bundle. |
| M12.1 and M12.2 | `PENDING_LOCAL` | `BLOCKED` | Final public claims, measured tables, portfolio artifacts, and the release audit require accepted M11 evidence. |

### Exact external blockers

The consolidated core target gate is `BLOCKED` because no compatible NVIDIA GPU SSH target is configured, no existing licensed OpenLane-V2 root is configured on that target, no machine-local license acknowledgment exists, and no human visual-audit signoff exists.
No milestone, checkpoint, accelerated result, trained-model claim, or release claim is accepted by inference from the local suite.

### Precise resume procedure

From a clean checkout of `origin/main`, first create the ignored license receipt with the exact accepted terms:

```sh
uv run --locked junctionlens data acknowledge \
  --accept-term Argoverse-2-terms \
  --accept-term CC-BY-NC-SA-4.0 \
  --accept-term nuScenes-terms \
  --confirm-restricted-noncommercial-use
```

Configure the external target and run the one consolidated core entry point:

```sh
export JUNCTIONLENS_GPU_HOST='<private-ssh-alias>'
export JUNCTIONLENS_REMOTE_ROOT='.junctionlens/qualification'
export JUNCTIONLENS_REMOTE_DATA_ROOT='<absolute-existing-licensed-openlane-v2-root>'
./scripts/gpu/qualify_remote.sh --profile core
```

The first complete core run is expected to return `LICENSED_VISUAL_AUDIT_REVIEW_REQUIRED` after retrieving its independently completed evidence.
Inspect the retrieved private bundle and, only if every assertion is genuinely satisfied, record the human signoff and resume the same content-addressed run:

```sh
uv run --locked junctionlens data signoff-visual-audit \
  --bundle '<downloaded-result>/licensed-data/private-visual-audit' \
  --accept-camera-projection-alignment \
  --accept-bev-geometry-alignment \
  --accept-label-identity-and-topology \
  --confirm-private-data-handling
./scripts/gpu/qualify_remote.sh --profile core
```

Resume M6 only if the retrieved core `status.json` is exactly `PASSED` and the measured promotion gate authorizes the conditional work.
After M6 and every remaining local extension gate is implemented, committed, and pushed, resume target qualification with `./scripts/gpu/qualify_remote.sh --profile full-v1`.

Planning snapshot: 2026-08-13.

JunctionLens is a local-first perception development product for control-aware road-scene graphs.
It ingests a calibrated multi-camera sequence, predicts lane geometry and topology, identifies traffic controls and road areas, associates each control with the lanes it governs, maintains temporal identity, and reports calibrated uncertainty.
It also compares candidate and baseline model releases through deterministic AV-specific KPIs, fault injection, paired statistical analysis, runtime evidence, and inspectable counterexamples.

The product is designed for researchers and engineers who need to answer two connected questions.

> Did the model understand the structure and controls of this road scene?

> Does the declared evidence support advancing the candidate relative to the frozen baseline under the benchmark policy?

JunctionLens is not a lane-detection notebook, a generic experiment dashboard, or a vehicle controller.
Its central engineering contribution is the combination of an owned graph-perception model with an owned release-evidence system that can detect structural regressions hidden by aggregate detection scores.

## 1. Product thesis

Detecting a traffic light is insufficient when the light is associated with the wrong lane.
Detecting lane fragments is insufficient when their successor graph blocks the route through an intersection.
Improving a global score is insufficient when performance regresses on a sparse domain, connector, small control, or low-luminance slice.

JunctionLens asks:

> Can a compact, calibrated, temporal multi-camera model reduce lane-to-control association errors while an independent release gate catches meaningful graph and domain regressions that aggregate scores miss?

The primary falsifiable modeling claim is:

- Joint lane and control graph modeling improves lane-to-control topology over independent detection followed by nearest-geometry linking without materially degrading lane or traffic-control detection.

The secondary falsifiable modeling claim is:

- Ego-motion-aligned temporal fusion reduces presence and topology flicker without materially degrading the frozen official benchmark metrics.

The release-system claim is:

- A predeclared, segment-paired release policy detects seeded control-association and graph-continuity regressions that leave node-detection scores nearly unchanged.

The claims are retained only if the numerical gates in this plan pass.

## 2. Target user and primary journey

The first user is an autonomy perception engineer evaluating a candidate scene-graph model before a larger simulation, mapping, or vehicle-integration stage.

The primary user journey is:

1. Run `junctionlens doctor` to verify toolchains, data contracts, model artifacts, and available execution providers.
2. Register a licensed OpenLane-V2 dataset root without copying the dataset into the repository.
3. Run `junctionlens data audit` to verify checksums, calibration, sequence identity, distributions, and split isolation.
4. Train or register a baseline model artifact.
5. Train or register a candidate model artifact that emits the canonical `SceneControlGraph` schema.
6. Run both models on the same frozen evaluation manifest.
7. Open the comparison report or local dashboard.
8. Inspect aggregate metrics, predeclared slices, uncertainty, latency, and release status.
9. Open ranked counterexamples to see multi-camera images, a BEV graph, control-association edges, and candidate-versus-baseline differences.
10. Export a self-contained redacted evidence bundle with content hashes, commands, model cards, and the release decision.

The first successful demonstration must show two prediction sets with nearly identical node-detection results where one contains incorrect lane-to-control edges and JunctionLens rejects it for the intended reason.

## 3. Public role evidence and skill alignment

The user-provided new-graduate autonomous mapping description emphasizes C++, Python, trajectory data analysis, graph algorithms, computational geometry, data preparation, proof-of-concept tooling, benchmarking, computer vision, robotics sensors, Linux, and systems fundamentals.
Public NVIDIA documentation independently shows that path perception represents drivable paths with left and right edges and adjacent-path semantics.
Public NVIDIA documentation also describes wait-condition perception as detecting intersections, traffic lights, and traffic signs, followed by light and sign classification.
OpenLane-V2 makes the coupling explicit by evaluating lane geometry, lane topology, traffic elements, and lane-to-traffic-element topology together.

JunctionLens produces direct evidence for the following responsibilities without copying a proprietary implementation:

| Responsibility | Project evidence |
| --- | --- |
| C++ systems engineering | C++20 graph evaluator, ONNX Runtime inference runtime, bounded streaming path, sanitizers, and profiling. |
| Python engineering | Dataset adapters, training, calibration, statistical evaluation, registry, API, and CLI. |
| Graph algorithms | Directed successor graphs, lane-control bipartite edges, matching, reachability, tracking, and regression analysis. |
| Computational geometry | Calibrated projection, polyline distance, endpoint continuity, BEV transforms, and uncertainty coverage. |
| Computer vision and ML | Multi-camera BEV features, query-based graph decoding, temporal fusion, multi-task losses, and calibration. |
| Data and KPI infrastructure | Immutable manifests, segment-disjoint splits, Parquet results, DuckDB queries, release policy, and counterexample bundles. |
| Performance engineering | C++ inference, provider auditing, host-device transfer accounting, latency distributions, memory peaks, and Nsight evidence. |
| Reliability | Schema validation, property tests, fault injection, fail-closed gates, provenance, and reproducible reruns. |

The repository and public materials must not identify internal teams, channels, interviewers, ticket identifiers, private systems, or nonpublic project details.
The project must stand on its public problem statement, public data, public APIs, and measured implementation.

## 4. Standalone product boundary

JunctionLens is independently useful to any team that produces lane-and-control graph predictions.
The canonical input and output schemas are not tied to OpenLane-V2 or to one model architecture.
The bundled reference model makes the product demonstrable from end to end.
External models can participate by emitting the same versioned prediction schema.

### 4.1 Required portfolio V1 core

The portfolio V1 core includes:

- A repository-owned synthetic scene-graph fixture generator.
- A pinned OpenLane-V2 v2.1 adapter and checksum manifest.
- Segment-disjoint train, selection, calibration, internal-holdout, and external-diagnostic manifests.
- A compact multi-camera lane-and-control graph reference model.
- A frozen independent-heads plus geometric-linking baseline.
- A versioned protobuf `SceneControlGraph` contract.
- A compatibility-isolated official-metric runner.
- C++ implementations of custom geometry and topology KPIs.
- A content-addressed experiment and artifact registry.
- A predeclared release-policy engine.
- Deterministic graph and metadata fault injection.
- A C++ ONNX Runtime inference path with CPU and CUDA execution profiles.
- A local CLI, FastAPI service, DuckDB analytics layer, deterministic static report, and thin read-only React scene viewer.
- A consolidated remote-GPU qualification workflow.
- Reproducible benchmark, model-card, architecture, limitations, and demonstration artifacts.

### 4.2 Conditional V1.1 work packages

The full plan also specifies ego-motion-aligned temporal fusion, track KPIs, probability and geometry calibration, conditional TensorRT qualification, the richer comparative dashboard, and the subset B diagnostic.
These work packages begin only after the core data, E0 and E1 model, evaluator, registry, fault-lab, CPU runtime, CUDA runtime, static report, and thin viewer gates pass within the frozen budget.
A failed V1.1 experiment produces a truthful negative result or deferral and must not delay the portfolio V1 core.
Section 29 defines the exact execution order even though later sections retain subject-based milestone numbers.

### 4.3 Explicit V1 non-goals

V1 does not include:

- Vehicle actuation, planning, or a stop/go decision.
- A claim of functional-safety certification.
- Online learning or model updates from user traffic.
- Production ingestion from real vehicle middleware.
- LiDAR, radar, raw GPS, or raw IMU fusion.
- Full HD-map construction or global map merging.
- OpenDRIVE generation.
- Photorealistic world generation.
- A full autonomous-driving simulator.
- Distributed training.
- Kubernetes, Kafka, Redis, or a cloud data lake.
- Multi-tenant authentication or hosted software as a service.
- Automated downloading or redistribution of restricted datasets.
- A proprietary-model compatibility layer.
- Mobile or embedded-board deployment.
- A claim that TensorRT accelerates the complete graph unless provider evidence proves it.
- A public release of dataset-derived weights until a separate license review permits it.

Anything in this list requires a separate plan after V1 is complete.

## 5. Novelty boundary and prior art

Lane detection, traffic-light detection, multi-camera BEV perception, graph prediction, calibration, experiment tracking, and dashboards all exist.
JunctionLens must not claim to invent those categories.

OpenLane-V2 already defines joint lane and traffic-element topology and provides official aggregate metrics.
Research models already compete on that benchmark.
General experiment platforms already store runs and plot scalar metrics.
Simulation frameworks already compute driving-quality and runtime telemetry.

JunctionLens has a defensible portfolio niche because it combines:

- A compact, owned reference model rather than only wrapping a paper repository.
- Explicit lane-to-control applicability as a first-class failure mode.
- Temporal graph stability and calibrated uncertainty as release evidence.
- Segment-paired candidate-versus-baseline regression decisions.
- Source-domain and long-tail slice gates with minimum-support rules.
- A fault lab that proves aggregate scores can hide structural mistakes.
- C++ inference and geometry evaluation connected to the same protobuf contract as Python training.
- A user-facing graph-diff workflow that ranks inspectable counterexamples.

Before the public README is finalized, `docs/prior-art.md` must compare at least:

- OpenLane-V2 and its official evaluator.
- The official OpenLane-V2 reference baseline.
- At least two current open lane-topology models.
- A general experiment-tracking product.
- A public AV evaluation or simulation framework.
- A public traffic-light or wait-condition perception reference.

The comparison must identify what is reused, what is independently implemented, what is measured, and what is deliberately out of scope.

### 5.1 Rejected project alternatives

A generic lane detector was rejected because it is common and omits topology, control applicability, uncertainty, and release infrastructure.
A traffic-light classifier was rejected because the difficult product error is often which light applies to which lane rather than merely the light color.
A generic KPI dashboard was rejected because it owns neither AV semantics nor the model and geometry that produce the evidence.
An OpenDRIVE validator was rejected because it validates authored maps rather than camera-derived scene structure and model releases.
A full simulation or world-generation project was rejected because it targets another layer, requires much more compute, and would make the work primarily an integration exercise.
A fork of an existing AV simulator was rejected because it would resemble maintainer work and obscure the independent model and evaluator contribution.
A state-of-the-art paper reproduction was rejected because dependency archaeology and leaderboard chasing would dominate the product and still omit release infrastructure.

## 6. Truthful claims and safety boundary

The public product description must state that JunctionLens is a research and model-development tool.
It must not be connected to vehicle actuation.
It must not represent a safety case, certification, or production-readiness approval.

Permitted claims require named evidence and may include:

- The joint model improves a named lane-control topology metric over the frozen geometric-linking baseline on a named frozen split.
- Temporal fusion reduces a named flicker metric under the frozen experiment protocol.
- Calibration improves named probabilistic metrics on a separate calibration and holdout protocol.
- The release gate detects each declared injected fault for the intended reason.
- The CUDA profile reaches a named latency and memory result on named hardware and software.
- The TensorRT profile executes a measured fraction of graph nodes on TensorRT on a named environment.
- A clean checkout reproduces a named synthetic or licensed-data workflow.

Prohibited claims include:

- The model is safe to drive a vehicle.
- The model understands every traffic rule.
- The product handles all geographies, weather, signs, or intersections.
- Confidence equals probability of safety.
- Better aggregate score means safer behavior.
- The model is real-time without an end-to-end latency protocol.
- The product uses TensorRT end to end when ONNX Runtime partitions nodes to another provider.
- Results generalize beyond the tested datasets and slices.
- The product is affiliated with or endorsed by a named employer.

## 7. Data source, version, and license contract

### 7.1 Primary dataset

OpenLane-V2 is the mandatory public dataset for the reference model.
It provides calibrated multi-camera images, ego pose, sequence identity, 3D lane centerlines, lane boundaries and types, intersection or connector labels, traffic lights and signs with attributes, pedestrian crossings, road boundaries, lane-lane topology, lane-control topology, and optional SD-map inputs.

The repository must pin the OpenLane-V2 v2.1.0 devkit tag at commit:

```text
d731a26bdbf34723dd915ad525c2c2eca19ed8a1
```

The devkit code is Apache-2.0.
The dataset is CC BY-NC-SA 4.0 and also requires agreement to the upstream nuScenes and Argoverse 2 terms.
The code repository must not download full data until the user explicitly confirms those terms.
The Git repository must never contain dataset images, annotations, preprocessed pickles, thumbnails derived from restricted images, or unreviewed trained weights.

### 7.2 Dataset lock file

`configs/data/openlane-v2-v2.1.lock.yaml` must include:

- The devkit repository URL, tag, and exact commit.
- The schema mode `lane-segment-v2.1`.
- The official source page URL.
- Every selected archive name, byte size when published, and official MD5 checksum.
- A local SHA-256 computed after download.
- The license identifiers and the timestamp of local user acknowledgment.
- The adapter version and preprocessing configuration hash.
- The camera-slot mapping for each source dataset.
- The set of enabled splits and their purpose.
- A `redistribution_allowed: false` default.

The following official checksums must be seeded into the lock and independently verified against the official data page during Milestone 0:

| Archive | Published checksum |
| --- | --- |
| OpenLane-V2 sample | `21c607fa5a1930275b7f1409b25042a0` |
| subset A metadata | `95bf28ccf22583d20434d75800be065d` |
| subset A map-element bucket | `1c1f9d49ecd47d6bc5bf093f38fb68c9` |
| subset A SD map | `de22c7be880b667f1b3373ff665aac2e` |
| subset B metadata | `27696b1ed1d99b1f70fdb68f439dc87d` |

The bootstrap script must populate and verify every selected image-shard checksum from the official page rather than relying on filenames alone.
A checksum mismatch is a hard failure and must never be treated as a warning.

### 7.3 Dataset profiles

The repository must support three profiles:

`synthetic` uses only repository-owned generated geometry, controls, calibration, and prediction fixtures.
It is mandatory for CI and the unrestricted public demo.

`sample` uses the approximately 300 MB official OpenLane-V2 sample after license acknowledgment.
It is used for adapter, visualization, evaluator, micro-overfit, and interface validation.

`full` uses the explicitly selected subset A archives and optional subset B diagnostic archives.
It is used for portfolio model training and final evidence.

CI must never assume that `sample` or `full` exists.

### 7.4 Split isolation

All learned-model partitions must be grouped by complete `segment_id`.
No frame from one segment may occur in more than one partition.

OpenLane-V2 subset A exposes 700 training segments.
The committed V1 split manifest assigns exactly 350 segments to model training, 80 to model selection, 70 to probability and geometry-scale calibration, and 200 to the frozen internal holdout.
The 200-segment holdout is required so the overall paired release comparison can meet its minimum-support rule.
The split generator may use only segment identity and authoritative source metadata that exist before partition assignment.
It must not derive a stratification field from labels, images, model outputs, or statistics computed inside any eventual partition.
The generator freezes the ordered segment lists and their SHA-256 values before any per-partition model result is inspected, and it never reshuffles V1.

Before the acceptance charter is sealed and while the already assigned holdout remains unopened, `junctionlens gate power` must use train and model-selection segments only to simulate the declared clustered estimator under plausible effect sizes and observed support.
If the simulation cannot support the planned family of gating cells, the project must reduce the number of gating slices, move weakly supported slices to exploratory status, or revise the declared improvement threshold before any holdout inference.
The project must never weaken a margin, merge a slice, or redefine support after observing candidate holdout results.

The official subset A validation split is an external benchmark used only after the architecture, losses, model-selection rule, calibrators, slice definitions, and numerical acceptance charter are frozen.
The official test set must not be discussed as evaluated unless labels or a valid server result exist.

Subset B is an optional frozen source-domain diagnostic.
It must never be mixed into subset A training or used to claim an official subset A leaderboard result.

### 7.5 Slice provenance

A slice name must describe only an observed or reproducibly derived property.
Geographic labels are allowed only when authoritative dataset metadata supports them.
Otherwise the product uses a source-domain identifier.

The V1 slice registry includes:

- Source domain.
- Intersection or connector presence.
- Merge or split topology.
- Lane-graph degree bucket.
- Lane curvature bucket.
- Traffic-control pixel-area bucket.
- Long-range projection proxy.
- Crosswalk presence.
- Traffic-control attribute group.
- Low-luminance proxy.
- Camera-availability pattern.
- Lane-count complexity bucket.

The low-luminance proxy must not be labeled as night.
An occlusion label must not be invented from small boxes.
Any later learned or heuristic slice requires its own versioned definition, tests, and freeze event.

## 8. System architecture

JunctionLens has six bounded subsystems.

1. The data layer validates licensed inputs, normalizes coordinates and cameras, creates immutable frame manifests, and exposes deterministic PyTorch datasets.
2. The model layer trains baseline, joint, temporal, and calibrated graph models.
3. The C++ runtime loads an exported ONNX model, preprocesses frames, runs an audited execution-provider chain, postprocesses queries, assigns temporal tracks, and emits protobuf graphs.
4. The evaluation layer matches graphs, reproduces official metrics, computes custom KPIs, creates slice tables, and performs paired statistical comparison.
5. The registry stores immutable manifests and Parquet artifacts and indexes them through DuckDB.
6. The local product layer exposes the CLI, FastAPI read API, and React graph-diff dashboard.

The runtime data flow is:

```text
licensed frames + calibration + pose
        |
        v
validated SensorFrame sequence
        |
        v
multi-camera encoder -> calibrated BEV -> temporal fusion -> query decoder
        |
        v
raw lane, control, area, and topology tensors
        |
        v
C++ postprocessing + track assignment
        |
        v
SceneControlGraph protobuf
        |
        +--> evaluator --> Parquet KPI tables --> release decision
        |
        +--> local API --> graph-diff dashboard --> evidence bundle
```

Raw dataset objects, normalized model inputs, model outputs, matched evaluation objects, and aggregated KPIs must remain separate types.
The implementation must not reuse one mutable dictionary across those boundaries.

## 9. Exact technology stack and dependency policy

### 9.1 Core stack

The initial reference toolchain is:

| Surface | Pinned reference |
| --- | --- |
| Linux runtime | Ubuntu 24.04 LTS, x86-64 |
| C++ | C++20 |
| Primary compiler | GCC 13 |
| Sanitizer and analysis compiler | Clang 18 |
| Build system | CMake 3.31 and Ninja 1.12 |
| Python | CPython 3.12 |
| Python resolver | `uv` with a committed `uv.lock` |
| PyTorch | 2.8.0 |
| torchvision | 0.23.0 |
| ONNX | 1.18.0 with opset 18 export |
| ONNX Runtime | 1.25.0 |
| CUDA reference | CUDA 12.8 |
| cuDNN reference | 9.14.0.64 |
| TensorRT reference | 10.14.1.48 when the conditional profile is enabled |
| Protobuf compiler and runtime | protoc 31.1 and libprotobuf 6.31.1 |
| C++ bindings | pybind11 3.0 |
| Linear algebra | Eigen 3.4 |
| Image processing | OpenCV 4.11 |
| Python analytics | Polars 1.31, PyArrow 20, DuckDB 1.3 |
| Python service | FastAPI 0.116 and Pydantic 2.11 |
| Web runtime | Node.js 22 LTS |
| Web application | React 19, TypeScript 5.8, Vite 7 |
| Python tests | pytest 8.4 and Hypothesis 6.x |
| C++ tests | GoogleTest 1.16 and RapidCheck at a pinned commit |

Patch versions not shown in the table must be resolved and pinned in the first Milestone 0 lock-file commit.
The implementation agent may update a reference version before dependent code exists only when current official compatibility documentation requires it.
Any update requires an ADR, exact new pin, source link, compatibility test, and regenerated locks.
Dependency versions must not float after Milestone 0.

The GPU runtime image builds ONNX Runtime 1.25.0 from its exact release source against CUDA 12.8, cuDNN 9.14.0.64, and the declared C++ ABI.
The conditional TensorRT image uses TensorRT 10.14.1.48, whose upstream-tested Linux CUDA 12 package is built for CUDA 12.9, only after a Milestone 0 driver and runtime compatibility probe passes.
The exact ONNX Runtime source archive hash, compiler flags, CUDA architectures, provider flags, shared-library hashes, and OCI digest are frozen in `containers/images.lock`.
The CPU artifact is built without CUDA or TensorRT providers and has no GPU-library dependency.
The GPU artifact is separate because the CUDA provider has hard load-time CUDA and cuDNN dependencies even when a session requests CPU execution.
No CPU CI job may install or load the GPU artifact.

The official OpenLane-V2 evaluator is not installed in the CPython 3.12 application environment.
It runs untouched in a digest-pinned Python 3.8 compatibility container with the exact v2.1 requirements, including NumPy from 1.22 through 1.23, SciPy 1.8.0, OR-Tools 9.2.9972, and Shapely 2.0.0.
Only JunctionLens-generated, schema-validated evaluator inputs may cross into that container because the upstream interface consumes pickle data.
The compatibility runner has no network, mounts inputs read-only, writes only to a bounded scratch output, and returns JSON metrics plus threshold-specific upstream matching artifacts.
The CPython 3.12 wrapper and any modernized helper are orchestration layers only and must prove fixture parity against the untouched compatibility container.

### 9.2 Dependency ownership

C++ dependencies are pinned through `cmake/dependencies.cmake` with release archive hashes or exact commits.
Python dependencies are pinned by `pyproject.toml` and `uv.lock`.
Web dependencies are pinned by `package.json` and `pnpm-lock.yaml`.
OCI images are referenced by digest in `containers/images.lock`.
External model backbones are recorded with source, version, weight hash, license, and preprocessing contract.

The project must not vendor large third-party repositories.
Generated protobuf files are reproducible build outputs unless the selected language tooling requires checked-in browser types.
Any checked-in generated file must be labeled and updated only through its generator command.

### 9.3 Execution-provider profiles

The provider profiles are exact and ordered.

`cpu-reference` registers only `CPUExecutionProvider` and uses FP32.
It is mandatory on developer machines and CI for correctness and portability.
It is not required to meet the real-time gate.

`cuda` registers `CUDAExecutionProvider` followed by `CPUExecutionProvider` and uses FP16 model weights where validated.
It is the mandatory accelerated V1 profile.
Every model node expected on CUDA must be audited.
Unexpected CPU fallback makes the accelerated result invalid.

`tensorrt` registers `TensorrtExecutionProvider`, then `CUDAExecutionProvider`, then `CPUExecutionProvider` as recommended by ONNX Runtime documentation.
It is conditional rather than mandatory.
The report must state the node and subgraph partition assigned to each provider.
TensorRT engine and timing caches are content-addressed by model hash, provider options, GPU compute capability, TensorRT version, CUDA version, driver compatibility class, and shape profile.
No cache may be reused across a mismatched key.

If TensorRT rejects an operator but CUDA covers it within the frozen latency and memory limits, the project remains valid and reports a partial TensorRT profile.
If CUDA unexpectedly falls back to CPU for a mandatory model operator, the accelerated profile fails until the graph or build is corrected.
If only CPU works, model and evaluator development may continue, but the accelerated portfolio claim remains blocked.

### 9.4 ONNX contract

The exported model uses ONNX opset 18.
The batch dimension may be dynamic from 1 to the frozen evaluation batch maximum.
The time dimension, camera-slot dimension, image height, image width, BEV dimensions, query counts, and point counts are fixed by `ModelProfileV1`.
Data-dependent-shape operators, Python control flow, custom operators, and runtime nonmaximum suppression are prohibited in the mandatory graph unless Milestone 0 proves support across CPU and CUDA profiles.
Postprocessing, thresholding, matching, and track assignment remain in C++.

The exporter must run ONNX shape inference, `onnx.checker`, CPU inference, CUDA inference when present, provider trace capture, and Python-versus-C++ parity before registering an artifact.

The build asserts that `protoc`, C++ headers, and `libprotobuf` all report the exact 31.1 or 6.31.1 release family selected in the lock.
The C++ runtime performs the same version check at process startup and fails before parsing persisted data on a mismatch.
Separate compatibility fixtures prove Python generated-code operation and browser ProtoJSON handling without requiring those language packages to share the C++ runtime binary.

## 10. Canonical protobuf contract

`proto/junctionlens/v1/scene_control_graph.proto` is the source of truth for cross-language data exchange.
The schema package is `junctionlens.v1`.
Persisted data uses a top-level `SceneControlGraphEnvelope` containing `schema_major`, `schema_minor`, `ProducerInfo`, and one `SceneControlGraph` payload.
Nested domain messages do not duplicate schema or producer fields.
Every enum begins with an explicit zero-valued `_UNSPECIFIED` member.
Validators reject `_UNSPECIFIED` where the enclosing field is required.
Removed fields reserve both their field numbers and field names forever.

The schema must define the following messages and enums.

### 10.1 Identity and provenance

`FrameKey` contains:

- `dataset_id`.
- `dataset_version`.
- `split_id`.
- `segment_id`.
- `timestamp_ns`.
- `source_domain`.
- `calibration_sha256`.
- `frame_manifest_sha256`.

`ArtifactRef` contains:

- `kind`.
- `sha256`.
- `byte_size`.
- `media_type`.
- `relative_uri`.
- `license_id`.

`ProducerInfo` contains:

- Git commit and dirty-state flag.
- Model artifact hash.
- Configuration hash.
- Runtime build hash.
- Execution-provider profile.
- Provider assignment digest.
- Random seed where applicable.

### 10.2 Sensor contract

`CameraSlot` is an enum with:

- `FRONT_CENTER`.
- `FRONT_LEFT`.
- `FRONT_RIGHT`.
- `SIDE_LEFT`.
- `SIDE_RIGHT`.
- `REAR_LEFT`.
- `REAR_CENTER`.
- `REAR_RIGHT`.

`CameraFrame` contains:

- The camera slot.
- A validity flag.
- The original image artifact reference.
- Capture timestamp in nanoseconds.
- Original width and height.
- The 3 by 3 intrinsic matrix.
- The 4 by 4 `T_vehicle_camera` transform.
- Distortion-model identity and coefficients when present.
- The deterministic resize, crop, and pad transform.

`SensorFrame` contains:

- One `FrameKey`.
- Exactly one entry for each ordered camera slot.
- The 4 by 4 `T_world_vehicle` pose.
- A pose-validity flag.
- The data-adapter version.

Missing cameras use a present slot with `valid=false`.
Camera-slot order must never be inferred from repeated-field order.

### 10.3 Graph node contract

`Point3d` uses double-precision meters in persisted messages.
`Polyline3d` contains ordered points, a confidence, and optional per-point uncertainty.

`LaneSegment` contains:

- `node_id`.
- Optional `track_id`.
- An ordered 3D centerline.
- Ordered left and right 3D boundaries.
- Left and right boundary-type distributions.
- Intersection-or-connector probability.
- Existence confidence.
- Per-point Laplace scales in meters.
- A decoder-query index for debugging.

`TrafficControl` contains:

- `node_id`.
- Optional `track_id`.
- The source camera slot.
- An optional `SourcePixelBox` containing the four original source numeric coordinates, source box convention, and source image dimensions without normalization or rounding.
- A normalized source-image bounding box.
- A control-category distribution.
- A control-attribute distribution.
- Existence confidence.
- Calibrated class and attribute confidence.
- A decoder-query index for debugging.

The OpenLane-V2 adapter assigns the published traffic-element box to `FRONT_CENTER` because the v2.1 annotations and visualizer define one front-view box rather than one box per camera.
The official evaluator adapter reads ground-truth coordinates from `SourcePixelBox` and never reconstructs them from normalized model coordinates.
Predicted boxes are converted from the half-open model contract exactly once under the frozen image dimensions and are retained as a separate evaluator-input artifact.
An external adapter may select another source camera only when its source contract proves that mapping.

`RoadArea` contains:

- `node_id`.
- Optional `track_id`.
- A category distribution for pedestrian crossing or road boundary.
- An ordered 3D polyline or polygon according to the category.
- Existence confidence.
- Geometry uncertainty.

### 10.4 Edge and track contract

`GraphEdgeType` contains:

- `LANE_SUCCESSOR`.
- `CONTROL_APPLIES_TO_LANE`.

`GraphEdge` contains:

- `edge_id`.
- `edge_type`.
- `source_node_id`.
- `target_node_id`.
- Raw probability.
- Calibrated probability.
- A binary decision under the frozen threshold set.
- Optional uncertainty metadata.

For `LANE_SUCCESSOR`, source and target are both lane nodes and direction follows travel.
For `CONTROL_APPLIES_TO_LANE`, source is the traffic-control node and target is the governed lane node.

`TemporalTrack` contains:

- `track_id`.
- Node type.
- Current node ID.
- First and last timestamps.
- Age in observed frames.
- Missed-frame count.
- Termination reason.
- Matching cost components.

### 10.5 Graph container

`SceneControlGraphEnvelope` contains:

- `schema_major`.
- `schema_minor`.
- `ProducerInfo`.
- One `SceneControlGraph` payload.

`SceneControlGraph` contains:

- `FrameKey`.
- Repeated lane, traffic-control, road-area, edge, and track messages.
- Raw tensor artifact references when debug export is enabled.
- Validation warnings that do not alter release status.

Protobuf binary is the immutable interchange representation, but protobuf binary bytes are not treated as a canonical serialization across implementations.
Deterministic application hashes are computed from a separately specified canonical logical projection rather than raw protobuf wire bytes.
ProtoJSON is provided for debugging and external integrations.
Because ProtoJSON encodes `uint64` values as decimal strings, the TypeScript client must retain node, edge, and track IDs as strings and must never coerce them to JavaScript numbers.
Parquet tables are derived analytics projections and never replace the protobuf source artifact.

### 10.6 ID semantics

Ground-truth dataset IDs remain namespaced by dataset, split, segment, timestamp, and node type.
They must never be assumed globally unique.

Predicted `node_id` values are frame-local unsigned 64-bit values and are unique across all node types in the frame.
The most significant eight bits encode the frozen node-type code, and the remaining 56 bits encode the deterministic type-local ordinal plus one.
They are assigned deterministically after filtering by node type, descending uncalibrated existence score, quantized geometry key, and decoder-query index.
The query index is the final tie breaker.

Predicted `track_id` values are segment-local unsigned 64-bit values assigned by the temporal tracker.
They must remain stable through tolerated misses and must never be reused after termination.

An imported ground-truth node carries a lossless `source_object_id` in adapter metadata.
When the dataset proves persistence across frames, its tracking key is `(dataset_id, split_id, segment_id, node_type, source_object_id)` and deliberately excludes timestamp.
Milestone 0 validates persistence separately for lane, traffic-control, and road-area identities by checking collisions, reuse, and geometric continuity.
If one node type lacks stable source identity, V1 either defines and freezes a geometry-based ground-truth association with hand-audited fixtures or disables temporal KPIs for that type.
Timestamp-namespaced frame IDs must never be mistaken for temporal ground-truth identity.

`edge_id` is a deterministic hash of schema major, frame identity, edge type, source node ID, and target node ID.
IDs provide referential integrity only.
Evaluator matching must not use numeric ID equality between prediction and ground truth.

Duplicate node IDs, dangling edges, invalid type combinations, reused terminated track IDs, and nonfinite values are hard schema failures.

## 11. Coordinate, geometry, and time conventions

JunctionLens uses one canonical right-handed vehicle frame.

- Positive X points forward.
- Positive Y points left.
- Positive Z points up.
- Distances are meters.
- Angles are radians.
- Timestamps are signed 64-bit nanoseconds.

The notation `T_target_source` maps a homogeneous point from `source` coordinates into `target` coordinates.
Therefore `T_vehicle_camera` maps camera coordinates to vehicle coordinates.
Adapters must explicitly invert source transforms when a dataset publishes the opposite direction.

Image coordinates use positive U to the right and positive V downward with the origin at the upper-left pixel corner.
The adapter preserves every original integer pixel box losslessly for the official evaluator.
The model contract uses continuous half-open boxes `[u_min, v_min, u_max, v_max)` relative to the uncropped original image.
Normalization divides horizontal coordinates by image width and vertical coordinates by image height, never by width minus one or height minus one.
The maxima are exclusive and normalized values lie in the closed numeric range from zero to one.
Resize, crop, and padding are represented by a separate homogeneous image transform.

OpenLane-V2 ground coordinates use positive X to the right, positive Y forward, and positive Z up.
JunctionLens therefore maps a source point by `[x_jl, y_jl, z_jl] = [y_ol, -x_ol, z_ol]`.
For column vectors, the homogeneous matrix `T_junctionlens_openlane` contains the corresponding axis permutation and sign change and multiplies the point on the left.
For row-vector tensor code, the implementation uses the transpose and records that convention in the API name.
Dataset transform-direction normalization and the OpenLane-to-JunctionLens basis change are tested independently before they are composed.

Lane centerline and boundary points are ordered in legal travel direction.
Left and right are defined while looking forward along travel direction.
The V1 model profile predicts exactly 11 points for each lane centerline and boundary.
Areas may use their schema-declared variable point count with a frozen maximum.

The canonical BEV range is:

- X from negative 20 meters inclusive to 80 meters exclusive.
- Y from negative 40 meters inclusive to 40 meters exclusive.
- Cell size 0.5 meters.
- Grid shape 200 by 160.

Each temporal input is transformed into the current vehicle frame with ego poses before fusion.
If either pose is missing or fails validation, temporal fusion is disabled for that pair and the reason is recorded.
The system must never silently treat an invalid pose as identity.

Coordinate tests must cover:

- Known camera rays.
- Projection and back-projection on a synthetic ground plane.
- Transform inversion and composition.
- Dataset-to-canonical and canonical-to-dataset round trips.
- Left and right boundary orientation.
- Ego-motion alignment.
- Resize, crop, pad, and box round trips.
- Border-touching, full-image, and one-pixel half-open box IoU goldens.
- OpenLane basis-change axis goldens and inverse round trips.
- Timestamp ordering and duplicate timestamps.

The numerical coordinate gate is a maximum 1e-6 homogeneous-transform identity error, 1e-5 meter synthetic round-trip error, and 0.25 pixel projection error on declared fixtures.

## 12. Reference model specification

### 12.1 Input profile

`ModelProfileV1` consumes:

- Batch dimension B.
- Two ordered timestamps with the current frame last.
- Eight canonical camera slots.
- RGB tensors resized and padded to 384 by 640.
- A Boolean camera-validity tensor.
- Per-camera intrinsics after the image transform.
- Per-camera `T_vehicle_camera` transforms.
- Previous-to-current ego-motion transform and validity.

The tensor shape is logically `[B, 2, 8, 3, 384, 640]`.
Invalid camera slots contain zeros and are masked before aggregation.
The normalization mean, standard deviation, color order, interpolation method, resize policy, and pad value are part of the model artifact rather than implicit code defaults.

### 12.2 Image encoder and BEV projection

The reference image encoder is an EfficientNet-B0 feature pyramid shared by every camera and timestamp.
Only publicly licensed backbone weights with a recorded hash may be used.
An ablation must report training from random initialization if pretrained-weight terms complicate redistribution.

Features at strides 8, 16, and 32 are projected to 128 channels and fused to the stride-8 scale.
The projection module uses camera calibration to sample features into the frozen 200 by 160 BEV grid.
The first implementation uses a deterministic ground-plane inverse-perspective projection plus a learned height-residual channel.
It does not claim full volumetric reconstruction.

Camera contributions are aggregated with a masked learned softmax whose normalization excludes invalid cameras.
The projection path must have both a slow reference implementation and a vectorized implementation.
They must agree within the frozen numerical tolerance.

### 12.3 Temporal fusion

The previous BEV tensor is warped into the current vehicle frame with the validated ego-motion transform.
A convolutional gated recurrent unit fuses previous and current BEV features.
The single-frame model is obtained by replacing the previous feature with a learned null state and setting the temporal-validity mask to false.

Photometric augmentations must use the same sampled parameters across both timestamps and all camera views when temporal consistency would otherwise be invalidated.
Geometric augmentations are allowed only when calibration, poses, boxes, polylines, and topology remain mathematically consistent.
Independent random crops and unmodeled horizontal flips are prohibited in V1.

### 12.4 Query decoder

The frozen initial capacities are:

- 96 lane queries.
- 64 traffic-control queries.
- 32 road-area queries.
- Four transformer decoder layers.
- Hidden dimension 256.
- Eight attention heads.

Milestone 0 must audit the complete training-label distributions.
If more than 0.1 percent of eligible frames exceed a capacity, the capacity must be increased before model training and then frozen in `ModelProfileV1`.
Query capacity must never clip labels silently.

Lane queries predict:

- Existence logit.
- Eleven 3D centerline points.
- Eleven 3D left-boundary points.
- Eleven 3D right-boundary points.
- Left and right boundary-type logits.
- Intersection-or-connector logit.
- Three coordinatewise Laplace log scales for every centerline and boundary point.
- A normalized temporal-track embedding.

Traffic-control queries predict:

- Existence logit.
- A normalized `FRONT_CENTER` box for the OpenLane-V2 V1 profile.
- Control-category logits.
- Control-attribute logits.
- Four coordinatewise box Laplace log scales.
- A normalized temporal-track embedding.

Road-area queries predict:

- Existence logit.
- Area-category logits.
- Frozen-count 3D points and a valid-point mask.
- Three coordinatewise geometry Laplace log scales for every valid point.
- A normalized temporal-track embedding.

The lane-successor head scores every ordered lane-query pair with a bilinear projection plus endpoint-geometry features.
Self-edges are masked unless the dataset audit demonstrates a valid self-edge convention.
The control-applicability head scores every traffic-control to lane-query pair with a bilinear projection plus projected relative-position features.
No nearest-neighbor postprocessor may overwrite learned topology in the joint model.

The canonical learned tensor is control-major and lane-minor.
When exporting to the OpenLane-V2 evaluator, which stores a lane-by-traffic-element matrix, the adapter applies exactly `topology_lste[lane_index, traffic_index] = P(CONTROL_APPLIES_TO_LANE(control_index, lane_index))`.
Import applies the inverse transpose into control-major canonical edges.
An asymmetric two-control by three-lane golden fixture must fail if either direction is transposed incorrectly.

### 12.5 Matching and training losses

Ground-truth assignment uses separate Hungarian matching for lanes, controls, and areas.
Matching costs are computed only from train-time allowed labels.

The default lane matching cost is:

```text
2.0 * existence focal cost
+ 5.0 * centerline L1 cost
+ 2.0 * boundary Chamfer cost
+ 1.0 * connector classification cost
```

The default traffic-control matching cost is:

```text
2.0 * existence focal cost
+ 5.0 * box L1 cost
+ 2.0 * generalized IoU cost
+ 2.0 * attribute classification cost
```

The default area matching cost is:

```text
2.0 * existence focal cost
+ 4.0 * symmetric Chamfer cost
+ 1.0 * area classification cost
```

The optimized training objective contains:

- Lane existence focal loss with weight 2.0.
- Centerline smooth-L1 and ordered-point loss with weight 5.0.
- Left and right boundary geometry loss with weight 2.0.
- Boundary-type cross-entropy with weight 1.0.
- Connector binary cross-entropy with weight 1.0.
- Traffic-control existence focal loss with weight 2.0.
- Traffic-control box L1 loss with weight 5.0.
- Traffic-control generalized-IoU loss with weight 2.0.
- Control-category and attribute loss with combined weight 2.0.
- Area existence, category, and geometry loss with combined weight 3.0.
- Lane-successor focal binary loss with weight 3.0.
- Control-applicability focal binary loss with weight 5.0.
- Successor endpoint-continuity loss with weight 1.0.
- Ego-motion-compensated temporal geometry loss with weight 1.0.
- Temporal embedding contrastive loss with weight 0.5.
- Coordinatewise geometry Laplace negative log-likelihood with weight 1.0.

Class and edge positive weights are computed from the training partition only and written into the run manifest.
They must not be recomputed from validation or holdout data.

The implementation must log every unweighted loss and gradient norm separately.
A decreasing weighted sum alone is not sufficient evidence that each head learns.

### 12.6 Training recipe

The frozen default training recipe is:

- AdamW.
- Base learning rate `2e-4`.
- Backbone learning-rate multiplier `0.1` when pretrained weights are used.
- Weight decay `0.01`.
- Linear warmup for 1,000 optimizer steps.
- Cosine decay to `2e-6`.
- Maximum 50 epochs.
- Batch size 1 sequence per GPU.
- Gradient accumulation for an effective batch size of 8.
- Gradient norm clipping at 1.0.
- BF16 automatic mixed precision when the target supports it.
- FP16 automatic mixed precision with dynamic loss scaling otherwise.
- FP32 CPU smoke training.
- Seed 20260813 for the primary run.

Early stopping is based on the model-selection partition after a minimum of 20 epochs and patience of 8 evaluations.
Checkpoint selection is lexicographic by lane-control topology, then official composite score, then negative log-likelihood.
A scalar weighted score must not be invented after viewing results.

Seed 20260813 is the sole predeclared release-decision seed for both E0 and the selected candidate.
Seeds 20260814 and 20260815 are robustness replications and can support a variability table but cannot determine the release result.
One seed is used for screening experiments.
The final artifact for seed 20260813 is selected on the model-selection partition, registered, and frozen before any internal-holdout inference.
The project must never select a best seed, checkpoint, threshold, or calibrator from holdout results.

### 12.7 Micro-overfit gate

Before full training, the model must overfit 32 fixed sample frames.
After at most 5,000 optimizer steps, it must satisfy all of the following on the same 32 frames:

- At least 90 percent reduction from the median first-100-step total loss.
- At least 98 percent matched node-category accuracy.
- Median matched centerline point error at most 0.25 meters.
- Lane-successor F1 at least 0.95 under oracle node matching.
- Lane-control topology F1 at least 0.95 under oracle node matching.
- No NaN, Inf, exploding gradient, or mixed-precision overflow remaining at termination.

Failure blocks full training until the data, coordinate, matching, capacity, or loss defect is resolved.
The gate must not be weakened to accommodate an unexplained failure.

## 13. Baselines, experiments, and ablations

### 13.1 Required experiment matrix

`E0-independent` uses the same image encoder and node heads but does not learn lane-lane or lane-control topology.
It constructs successor edges from frozen endpoint-distance and heading rules.
It constructs control edges by projecting each lane centerline into the control source image and applying a frozen box-to-visible-lane endpoint and heading rule.
Its rules are fitted only on the training partition and frozen before model-selection evaluation.

`E1-joint` adds learned lane-successor and lane-control heads without temporal fusion.

`E2-temporal` adds ego-motion-aligned two-frame fusion and track embeddings.

`E3-calibrated` freezes E2 weights and fits probability temperatures and geometry-scale correction on the calibration partition.
E1 and E2 keep gates are scored on the model-selection partition.
E3 calibrators are fit on the calibration partition and scored for selection on the model-selection partition without refitting.

The final V1 model is E3 if all keep gates pass.

### 13.2 Diagnostic model modes

The evaluator supports three diagnostic modes:

`oracle-nodes` supplies ground-truth nodes to a topology head so edge learning can be isolated.

`predicted-nodes` evaluates the complete end-to-end system.

`oracle-calibration` is prohibited as a published result and is used only in unit fixtures.

Every result table must label the mode.
Oracle-node metrics must not be presented as end-to-end perception quality.

### 13.3 Mandatory ablations

The final report includes:

- E0 versus E1 for joint topology.
- E1 versus E2 for temporal fusion.
- E2 raw versus E3 calibrated for probability and geometry uncertainty.
- Ground-plane projection reference versus vectorized projection for correctness and speed.
- Pretrained versus random-initialized backbone when licensing permits the comparison.
- Camera-mask robustness with one predeclared missing-camera pattern.
- Optional SD-map adapter only if the core V1 gates already pass.

The optional SD-map adapter is not allowed to delay the development-complete portfolio core.

### 13.4 Keep gates

E1 is retained only if, on the model-selection partition:

- `TOP_lt` improves by at least 2.0 absolute percentage points over E0.
- `DET_l` is no more than 1.0 percentage point worse than E0.
- `DET_t` is no more than 1.0 percentage point worse than E0.
- Wrong-control assignment rate is lower than E0.

E2 receives provisional selection approval only if:

- Temporal presence flicker decreases by at least 10 percent relative to E1.
- Lane-control edge-flip rate decreases by at least 10 percent relative to E1.
- The official composite score is no more than 1.0 percentage point worse than E1.
- Its exported graph passes CPU parity and has a Milestone 0 cost estimate consistent with the frozen runtime budget.

E2 receives final model-selection approval in Milestone 8 only after its accelerated runtime is compared with E1 on the frozen model-selection manifest and passes the absolute runtime limits.
If E2 fails that promotion, E1 becomes the candidate before any holdout inference.

E3 is retained only if:

- Edge Brier score improves by at least 5 percent relative to raw E2 confidence.
- Adaptive ECE does not worsen for any primary head with sufficient support.
- The segment-cluster adjusted confidence interval for 90 percent geometry coverage lies entirely between 87 and 93 percent overall.
- Official predictive metrics remain numerically unchanged within 1e-6 because they continue to use the frozen raw existence and edge ranking scores.

If a keep gate fails, the project publishes the failed hypothesis and uses the preceding accepted configuration.
It must not fabricate a positive model claim.
The keep gates select an architecture before any candidate is evaluated on the frozen internal holdout.
The internal holdout is used once for the final selected candidate-versus-baseline decision in Milestone 11.

## 14. Temporal tracking and calibration

### 14.1 Tracker

The C++ temporal tracker matches current predictions to live tracks after ego-motion alignment.
It uses Hungarian assignment with the frozen cost:

```text
0.45 * normalized geometry distance
+ 0.25 * cosine embedding distance
+ 0.20 * class-distribution distance
+ 0.10 * topology-neighborhood distance
```

Matches exceeding the node-type threshold are rejected.
Unmatched confident nodes create new tracks.
Tracks survive at most three consecutive missed frames.
A terminated track ID is never reused in the segment.

Tracker thresholds are selected on the model-selection partition and frozen before the calibration and holdout runs.
Ground-truth IDs are used only to score tracking and never as tracker inputs.

### 14.2 Probability calibration

The calibrator fits one positive scalar temperature for each of:

- Lane existence.
- Traffic-control existence.
- Road-area existence.
- Boundary type.
- Connector state.
- Control category.
- Control attribute.
- Lane-successor edges.
- Control-to-lane edges.

Temperatures minimize held-out negative log-likelihood on the calibration partition.
No temperature may be fit on the final internal holdout, official validation, or subset B diagnostic.
Calibrated probabilities are stored alongside rather than over raw scores.
Binary positive-temperature transforms must preserve raw ordering.
Multiclass calibration must preserve the predicted class for each item, but confidence ordering across different items is not assumed invariant.

Calibration artifacts include head identity, source model hash, fit-manifest hash, optimizer configuration, temperature, sample count, class support, pre-metrics, post-metrics, and code hash.

### 14.3 Geometry uncertainty

The model predicts one positive coordinatewise Laplace scale through `softplus(raw_scale) + 1e-4` meters for every X, Y, and Z value of every modeled geometry point.
One nonnegative multiplicative factor per geometry head is fitted on the calibration partition to target 90 percent marginal point coverage.
Coverage is reported overall and by predeclared distance and source-domain slices.
For a two-sided Laplace interval with scale `b`, the 90 percent marginal half-width is exactly `b * ln(10)`.
The coverage keep gate uses a segment-cluster adjusted confidence interval around the coverage estimate and requires its lower bound to be at least 87 percent and its upper bound to be at most 93 percent without an interval-width regression beyond the frozen margin.

The report must distinguish marginal calibration from simultaneous polyline coverage.
V1 does not claim a 90 percent joint confidence region for an entire lane.

### 14.4 Selective prediction

JunctionLens computes risk-coverage curves by sorting nodes or edges from highest to lowest calibrated confidence and measuring task error as lower-confidence predictions are abstained.
It reports area under the risk-coverage curve.
It does not automatically remove low-confidence graph elements from a control system.
Abstention is an evaluation concept in V1.

## 15. Evaluation architecture and official compatibility

### 15.1 Evaluation stages

Evaluation proceeds in an immutable sequence:

1. Validate input manifests and schema.
2. Load ground truth and predictions by exact `FrameKey`.
3. Reject missing, duplicate, or extra frames unless the run policy explicitly permits partial diagnostics.
4. Convert protobuf nodes to evaluator-native immutable structures.
5. Compute official metrics and their threshold-specific matches through the untouched Python 3.8 compatibility evaluator.
6. Compute `CustomMatchV1` once for custom topology, temporal, and calibration metrics.
7. Verify any modern Python orchestration output against the compatibility evaluator and verify C++ custom primitives against language-neutral goldens.
8. Freeze the `CustomMatchV1` artifact and preserve the upstream threshold-specific matches separately.
9. Write frame and segment metrics to Parquet.
10. Aggregate by the frozen registry of slices.
11. Compare candidate and baseline with paired segment statistics.
12. Apply the frozen release policy.
13. Materialize counterexamples and the evidence bundle.

### 15.2 Official metrics

JunctionLens reports the current OpenLane-V2 v2.1 metrics without renaming or redefining them:

- Lane-segment detection `DET_l`.
- Area detection `DET_a`.
- Traffic-element detection `DET_t`.
- Lane-lane topology `TOP_ll`.
- Lane-traffic-element topology `TOP_lt`.
- The official composite `OLUS` and every component used by the pinned evaluator.

Official discrete Frechet, Chamfer, IoU, distance thresholds, node matching, topology averaging, and transform function are owned by the pinned v2.1 specification.
The repository must not silently substitute a faster approximation in official result tables.

### 15.3 Official evaluator ownership and parity gate

The Python 3.8 compatibility container is the sole owner of official metric values and official matching artifacts.
JunctionLens does not reimplement the full official evaluator in C++ and does not make a C++ official-parity claim.
The modern Python wrapper must match the compatibility container's JSON outputs exactly on the frozen fixtures, except for explicitly normalized NaN serialization.

The test fixture includes empty, perfect, partial, duplicate-confidence, permuted-order, and adversarial prediction sets.
For every finite official metric, the wrapper and compatibility-container outputs must differ by at most 1e-12 absolute after reading the same returned JSON values.
NaN and empty-denominator behavior must match exactly after the declared JSON normalization.

Correctly permuting prediction arrays and their adjacency matrices must leave metrics invariant.
Permuting arrays without matrices must fail the seeded integrity test or cause the expected topology degradation.

Until parity passes, official results cannot drive a release decision.

### 15.4 CustomMatchV1

The official evaluator may form different confidence-greedy matches at different metric thresholds, so JunctionLens never invents one "official match map" for custom KPIs.
`CustomMatchV1` is a separate, versioned, deterministic association policy and is always labeled custom.
Predictions are processed by descending raw existence confidence, then quantized geometry key, then decoder-query index.
Each prediction greedily takes the lowest-cost unmatched ground-truth object of the same type that passes the frozen threshold.
Lane matching uses the official lane-segment distance primitive at the fixed 2.0 meter threshold.
Traffic-control matching uses the official traffic-element distance, defined as one minus IoU, at the exact distance threshold 0.75.
Road-area matching uses the official area-distance primitive at the fixed 1.0 meter threshold.
Equal costs select the lexicographically smallest lossless ground-truth source ID.
Predictions outside threshold remain unmatched false positives, ground-truth objects left unmatched remain false negatives, and no object participates in more than one pair.
The match artifact records raw scores, sorted order, every eligible cost, selected pair, rejected pair reason, and unmatched reason.
Custom KPI documentation names `CustomMatchV1` and must not describe its pairs as official matches.

## 16. Custom KPI definitions

Custom KPIs use the frozen `CustomMatchV1` association unless a metric explicitly states otherwise.
Each KPI defines its eligible population, denominator, direction, unit, and missing-data behavior in `configs/metrics/v1.yaml`.

### 16.1 Control-association KPIs

Let an eligible ground-truth control have at least one ground-truth governed lane and a matched predicted control.
Let the eligible governed lanes be those with matched predicted lane nodes.

`control_edge_precision` is the number of thresholded predicted control-to-lane edges whose matched ground-truth pair exists divided by all thresholded predicted edges between matched eligible nodes.

`control_edge_recall` is the number of ground-truth control-to-lane edges recovered by a thresholded edge between their matched predictions divided by all ground-truth edges whose endpoints are matched.

`wrong_control_assignment_rate` considers eligible controls with at least one thresholded predicted edge.
It is the fraction whose highest-scoring thresholded predicted lane maps to no ground-truth governed lane for that control.
Controls with no thresholded edge are counted by edge recall rather than as wrong assignments.

`confident_wrong_control_rate` uses the same numerator but additionally requires calibrated edge probability at least 0.9.
It exposes harmful overconfidence separately from ordinary error.

### 16.2 Lane-topology KPIs

`reachability_recall_h3` enumerates matched ground-truth lane pairs reachable within one to three directed successor hops.
It is the fraction that remains reachable within one to three hops in the matched predicted graph.

`path_blocking_rate_h3` considers each matched ground-truth lane that has at least one ground-truth continuation within three hops.
It is the fraction for which no matched predicted continuation exists within three predicted hops.

`successor_endpoint_gap_m` is the Euclidean distance from the final centerline point of a predicted source lane to the first centerline point of its predicted successor.
It is reported as median, P90, P95, and the fraction above the frozen geometry threshold.

`spurious_successor_rate` is the fraction of thresholded predicted successor edges between matched lanes that do not correspond to a ground-truth successor edge under `CustomMatchV1`.

Real road graphs may merge, split, and contain cycles.
JunctionLens must not treat acyclicity, fixed degree, or reciprocal edges as universal validity rules.

### 16.3 Temporal KPIs

`presence_flicker_rate` uses ground-truth nodes present for at least three consecutive annotated frames.
It is the number of predicted presence-state transitions after matching divided by eligible adjacent-frame transitions.

`successor_edge_flip_rate` uses ground-truth successor edges whose endpoints persist in consecutive frames.
It is the fraction of adjacent transitions where the thresholded predicted state changes after node matching.

`control_edge_flip_rate` is defined analogously for persistent control-to-lane edges.

`geometry_jitter_m` compares the change in ego-motion-aligned predicted geometry with the corresponding change in ground-truth geometry for a persistent matched node.
It reports the norm of the residual change rather than penalizing true scene motion.

`id_switches_per_100_tracks` counts one switch when the current matched prediction for an eligible persistent ground-truth track has a different predicted track ID from its previous matched prediction while the previous predicted track remains live.
Unmatched intervening frames count as misses, not switches, and a reacquisition after the previous track has terminated starts a new fragment rather than a switch.
It is normalized by 100 eligible ground-truth tracks.

### 16.4 Calibration KPIs

Calibration populations are defined only after `CustomMatchV1` is frozen for the frame.
Node-existence calibration includes matched predictions as positive outcomes and unmatched predictions as negative outcomes.
Unmatched ground-truth nodes have no predicted probability and are reported separately as missed detections rather than assigned an invented zero score.
Edge calibration conditional on perception includes only candidate endpoint pairs whose endpoints are both matched.
Separate end-to-end edge metrics retain the penalties caused by missed endpoints.

`brier_score` is the mean squared error for a binary head and the multiclass sum of squared class-probability errors for a multiclass head.

`negative_log_likelihood` clips probabilities to `[1e-7, 1 - 1e-7]` only for numerical evaluation and reports the unclipped saturation count.

`adaptive_ece_15` uses 15 equal-count bins with deterministic tie handling for binary heads and top-label confidence and correctness for multiclass heads.
It is reported with bin counts and must not be the only calibration metric.

`aurc` is the trapezoidal area under the risk-coverage curve with deterministic confidence tie ordering.

`geometry_coverage_90` is the empirical fraction of scalar point-coordinate residuals inside the predicted marginal 90 percent Laplace interval.

`geometry_interval_width_m` reports the median and P90 interval width so coverage cannot be improved by unbounded uncertainty.

### 16.5 Runtime KPIs

The runtime records:

- Decode time when image decode belongs to the measured profile.
- Resize, normalize, and pack time.
- Host-to-device transfer time.
- Model inference time.
- Device-to-host transfer time.
- Postprocessing time.
- Tracking and serialization time.
- End-to-end input-ready to graph-ready latency.
- Throughput.
- Startup and warmup time.
- Peak resident host memory.
- Peak device memory.
- Execution-provider node counts and fallbacks.
- Cache hit and miss state.

Latency summaries report count, mean, median, P90, P95, P99, maximum, warmup count, and the exact clock source.
Average latency alone is prohibited.

## 17. Statistical and numerical acceptance charter

### 17.1 Freeze event

`configs/gates/acceptance-v1.yaml` is drafted before candidate holdout results exist.
It is finalized after E0 baseline variability and M0 hardware baselines are measured but before E1, E2, or E3 is evaluated on the frozen internal holdout.

The freeze command is:

```bash
uv run junctionlens gate freeze \
  --draft configs/gates/acceptance-v1.draft.yaml \
  --baseline-run artifacts://runs/<baseline-run-id> \
  --output configs/gates/acceptance-v1.yaml
```

Freezing writes the charter SHA-256, baseline hashes, metric registry hash, slice registry hash, software commit, timestamp, and signer identity supplied by the local user.
The final charter is read-only to evaluation commands.

Any material change creates `acceptance-v2.yaml` and invalidates direct V1 release comparisons.
The tool must never edit expected margins automatically after a candidate result.

### 17.2 Statistical unit

The independent resampling unit is `segment_id`.
Frames, nodes, and edges within a segment are not treated as independent samples.

Candidate and baseline are paired on the exact same eligible segments.
The paired difference is resampled with 10,000 segment-cluster bootstrap replicates using seed 20260813.
The bootstrap implementation and version are part of the report.

Each replicate draws the original paired segment set with replacement and recomputes the complete pooled metric for candidate and baseline inside that replicate.
It does not average precomputed per-segment AP values or pretend frames are independent.
Official average-precision metrics reconstruct the global ranked prediction list for every replicate.
When one source segment is drawn more than once, each draw receives a deterministic replicate-qualified segment and frame identity so repeated observations are retained without evaluator key collisions.
Ratio metrics sum their primitive numerators and denominators across drawn segment occurrences and divide once per replicate.
Nonlinear graph and calibration metrics are likewise recomputed from their primitive observations rather than averaged from final segment summaries.
An estimator with a zero eligible denominator emits `INVALID_EMPTY_REPLICATE` for that replicate.
A gating cell is `INSUFFICIENT_EVIDENCE` when fewer than 9,900 of 10,000 replicates are finite, and otherwise its interval uses the finite replicates while reporting the invalid count.

Runtime uses a separate repeated-trial estimand and is never inserted into the segment bootstrap.
After one stabilization and cache-state protocol, the runner executes ten paired trial blocks with candidate and baseline order assigned by a frozen balanced AB or BA schedule.
Each trial performs 200 warmup frames followed by 2,000 measured frames from the same manifest and records input order.
The comparison is the paired trial-block difference, with a 10,000-replicate bootstrap over the ten blocks and the frozen seed 20260813.
Thermal, clock, contention, and cache invalidity fail the affected block, and fewer than eight valid pairs produces `BLOCKED_INFRASTRUCTURE`.

### 17.3 Multiple comparisons

The charter identifies the finite set of gating metric-and-slice cells.
For N gating cells, each cell uses a Bonferroni-adjusted two-sided interval with family alpha 0.05 and per-cell alpha `0.05 / N`.
Exploratory slices use ordinary 95 percent intervals and are visibly labeled non-gating.

### 17.4 Minimum support

An overall release comparison requires at least 200 paired segments.
A gating slice requires at least 30 paired segments.
A lane-control edge metric additionally requires at least 500 eligible ground-truth edges.
A temporal metric requires at least 500 eligible adjacent-frame transitions across at least 30 segments.

When support is lower, the result is `INSUFFICIENT_EVIDENCE`.
The tool may report the estimate but may not pass the cell, merge it into another slice after seeing the result, or claim noninferiority.

### 17.5 Decision semantics

For a higher-is-better metric with candidate-minus-baseline delta and noninferiority margin M:

- The cell passes when the adjusted lower confidence bound is at least negative M.
- The cell fails for regression when the adjusted upper confidence bound is below negative M.
- The cell is insufficient when the interval crosses negative M.

For a lower-is-better metric, signs are reversed before applying the same logic.

The release status is one of:

- `PASS`.
- `FAIL_INTEGRITY`.
- `FAIL_REGRESSION`.
- `FAIL_PERFORMANCE`.
- `INSUFFICIENT_EVIDENCE`.
- `BLOCKED_INFRASTRUCTURE`.

Only `PASS` authorizes the report to call the candidate accepted under the V1 benchmark policy.
Acceptance under a noninferiority cell is not evidence of improvement.
Public language must say "accepted under the benchmark policy" unless the corresponding predeclared superiority hypothesis also passes.

The primary holdout superiority hypothesis requires the Bonferroni-adjusted lower confidence bound for candidate-minus-E0 `TOP_lt` to be at least positive 1.0 percentage point.
The associated detection protection requires the adjusted lower bounds for `DET_l` and `DET_t` to remain at least negative 1.0 percentage point.
If E2 is selected, the temporal claim additionally requires the adjusted lower bound on relative reduction in `presence_flicker_rate` versus E1 to be at least 5 percent while the official composite passes its declared noninferiority cell.
If E3 is selected, the calibration claim additionally requires the adjusted lower bound on relative reduction in control-edge Brier score versus raw E2 to be at least 2 percent while predictive ranks remain frozen.
These hypotheses, comparators, metric direction transforms, and family membership are serialized in the charter before holdout inference.

### 17.6 Initial numerical margins

The draft charter starts with these maximum tolerated regressions:

| Metric | Margin |
| --- | ---: |
| `DET_l` | 1.0 percentage point |
| `DET_t` | 1.0 percentage point |
| `DET_a` | 1.0 percentage point |
| `TOP_ll` | 0.5 percentage point |
| `TOP_lt` | 0.5 percentage point |
| `control_edge_recall` | 0.5 percentage point |
| `wrong_control_assignment_rate` | 0.5 percentage point increase |
| `reachability_recall_h3` | 0.5 percentage point |
| `path_blocking_rate_h3` | 0.5 percentage point increase |
| `adaptive_ece_15` | 0.01 absolute increase |
| P95 end-to-end latency | 5 percent increase while remaining under the absolute budget |
| Peak device memory | 5 percent increase while remaining under the absolute budget |

The freeze tool may tighten a margin based on baseline repeatability.
It must not loosen a draft margin without an ADR written before candidate holdout evaluation.
All baseline variability, power analysis, and any margin revision use only training and model-selection data or predeclared pseudo-holdouts derived from them.
Neither E0 nor a candidate may run on the internal holdout until the one-shot Milestone 11 evaluation.

### 17.7 Numerical integrity

All persisted continuous metrics use float64 aggregation.
Input FP16 or FP32 values are promoted before distance and bootstrap calculations.
The report records NaN, Inf, clipped-probability, empty-denominator, and unmatched-node counts.
Silent dropping of nonfinite observations is prohibited.

Metric tables have a deterministic sort order.
Identical content hashes and deterministic profiles must produce bit-identical JSON and Parquet logical values.
GPU results documented as nondeterministic use a named tolerance and must still preserve release status across three reruns.

## 18. Artifact registry and reproducibility

### 18.1 Local-first storage

V1 uses a local content-addressed artifact store under an ignored root selected by `JUNCTIONLENS_ARTIFACT_ROOT`.
The default is `./artifacts` for development.

Blobs are stored at:

```text
objects/sha256/<first-two-hex>/<remaining-hex>
```

Manifests are immutable JSON validated by `schemas/artifact-manifest-v1.schema.json`.
DuckDB indexes metadata and Parquet tables but is never the sole copy of provenance.

The registry supports these artifact kinds:

- Dataset lock.
- Frame manifest.
- Split manifest.
- Run configuration.
- Checkpoint.
- Calibrator.
- ONNX model.
- Execution-provider trace.
- Prediction bundle.
- Match map.
- Frame KPI table.
- Segment KPI table.
- Slice table.
- Comparison.
- Release decision.
- Counterexample bundle.
- Benchmark.
- Profiler capture.
- Model card.
- Evidence report.

### 18.2 Run identity

A run ID is the SHA-256 of canonical JSON containing:

- Run kind.
- Parent artifact hashes.
- Dataset and split manifest hashes.
- Model profile and configuration hashes.
- Source Git commit and dirty state.
- Complete dependency-lock hashes.
- Container image digests when used.
- Seed.
- Command schema version.
- Execution-provider profile.

Human-readable aliases are mutable pointers and are never evidence identifiers.

### 18.3 Atomicity and concurrency

Artifacts are written to a same-filesystem temporary path, flushed, hashed, and atomically renamed.
The containing directory is synchronized before the registry reports success.
An existing hash path is verified rather than overwritten.

DuckDB has one writer protected by an advisory lock with owner PID, host fingerprint, creation time, and heartbeat.
Read-only dashboard connections may run concurrently.
A stale lock is reclaimed only after the owning process and host condition are validated and the action is logged.

Interrupted runs resume only when all declared input hashes and the environment compatibility fingerprint still match.
Otherwise a new run ID is mandatory.

### 18.4 Reproduction bundle

Every accepted comparison produces:

- `manifest.json`.
- `decision.json`.
- `metrics.parquet`.
- `slices.parquet`.
- `counterexamples.json`.
- `commands.jsonl`.
- `environment.json`.
- `REPORT.html`.
- `REPORT.md`.
- `SHA256SUMS`.

The public bundle excludes dataset frames and private paths by default.
It may include repository-owned synthetic renderings and plots derived only from scalar aggregates.
A separately reviewed private bundle may include licensed thumbnails.

## 19. Release-policy engine

### 19.1 Integrity stage

The gate first rejects:

- Dataset or split hash mismatch.
- Candidate and baseline frame-set mismatch.
- Schema-major incompatibility.
- Duplicate or dangling IDs.
- Invalid coordinate or calibration metadata.
- NaN or Inf in required tensors or metrics.
- Missing calibrator for a calibrated profile.
- Official evaluator compatibility or wrapper-output failure.
- Incomplete provenance.
- Data leakage between partitions.
- Changed metric or slice registry after the charter freeze.
- Unapproved partial inference.
- Unexpected execution-provider fallback in an accelerated gate.

An integrity failure stops statistical acceptance but still produces a diagnostic report.

### 19.2 Accuracy and calibration stage

The engine evaluates all primary overall cells and every supported predeclared gating slice.
Primitive metrics gate independently.
The official composite is reported but cannot compensate for a failing lane-control or topology primitive.

Calibration gates use the exact same prediction ranks and only compare probability transformations.
If calibration changes node or edge ordering, the calibrator artifact is invalid.

### 19.3 Runtime stage

Accelerated release evidence uses the same hardware identity, power policy, provider configuration, input profile, warmup, sample count, and contention limits for candidate and baseline.
A hardware mismatch produces `BLOCKED_INFRASTRUCTURE` rather than a performance result.

The absolute accelerated V1 budgets are:

- At least 10 complete graph outputs per second at batch 1.
- P95 input-ready to serialized-graph-ready latency at most 100 milliseconds.
- P99 input-ready to serialized-graph-ready latency at most 125 milliseconds.
- Peak device memory at most 6 GiB.
- No unbounded host or device memory growth over 10,000 frames.
- No unexpected CPU provider nodes.

Each paired trial block uses 200 warmup frames and 2,000 measured frames from the frozen manifest under the Section 17.2 AB or BA schedule.
Image decoding is reported in a separate full-file profile and excluded only from the explicit predecoded profile.
Both labels must be visible.

### 19.4 Decision and reason codes

Every failed or insufficient cell emits a stable reason code, metric, slice, support, point estimate, interval, margin, and counterexample query.
The decision is deterministic from the evidence bundle and charter.

Changing a dashboard filter cannot change release status.
The API serves the persisted decision rather than recalculating it with client parameters.

## 20. Fault lab

`junctionlens fault` creates a derived prediction bundle with one declared transformation and a parent hash.
Faults never modify the original bundle.

### 20.1 Mandatory graph faults

- `swap-control-edges` swaps the governed lanes for two matched controls while preserving node tensors.
- `drop-control-edges` removes a controlled fraction of true lane-control links.
- `drop-successor-chain` removes one edge from an otherwise reachable three-hop lane path.
- `add-spurious-successors` inserts geometry-plausible but incorrect lane edges.
- `permute-nodes-correctly` permutes nodes and all matrix axes as a nonfault control.
- `permute-nodes-without-edges` permutes nodes but leaves adjacency order unchanged.
- `duplicate-node-id` creates a referential-integrity failure.
- `dangling-edge` points to an absent node.

### 20.2 Mandatory geometry and calibration faults

- `jitter-lanes` applies a seeded lateral and longitudinal perturbation.
- `flip-boundaries` swaps left and right lane boundaries.
- `corrupt-extrinsic` perturbs one camera transform while preserving schema validity.
- `zero-uncertainty` creates overconfident geometry intervals.
- `inflate-uncertainty` creates uninformative wide intervals.
- `temperature-collapse` makes probabilities overconfident without changing ranks.
- `inject-nan` inserts a nonfinite value at a declared path.

### 20.3 Mandatory temporal and runtime faults

- `alternate-edge-confidence` crosses the edge threshold on alternating frames.
- `alternate-node-presence` creates presence flicker.
- `reuse-track-id` reuses a terminated track ID.
- `force-provider-fallback` exports or configures one seeded unsupported node in a test-only model.
- `delay-postprocess` injects bounded latency into the benchmark-only test runtime.
- `leak-buffer` is a test-only allocator fixture used to prove long-run memory detection.

### 20.4 Fault acceptance

Every mandatory fault has one expected primary reason code and permitted secondary effects.
The nonfault permutation control must preserve all metrics within parity tolerance.
Each deterministic fault must be detected in 100 percent of its seeded fixtures.
Nearby clean controls must pass.

The portfolio demo must include at least one `swap-control-edges` case where `DET_l`, `DET_t`, and node geometry remain unchanged while the lane-control release cell fails.

## 21. C++ inference and evaluation runtime

### 21.1 Components

The C++20 library is split into:

- `libjunctionlens_contract` for generated protobuf types and validation.
- `libjunctionlens_geometry` for transforms, curves, projection, distances, and matching primitives.
- `libjunctionlens_eval` for custom KPIs and language-neutral metric goldens.
- `libjunctionlens_infer` for preprocessing, ONNX Runtime, postprocessing, and provider audit.
- `libjunctionlens_track` for temporal identity.
- `junctionlens-runtime` for batch and streaming CLI execution.
- `junctionlens_cpp` for narrow pybind11 bindings.

Python must not call private C++ symbols through `ctypes`.
The pybind surface is typed, versioned, and tested against protobuf artifacts.

### 21.2 Buffer ownership

The streaming runtime uses a fixed pool of frame slots.
Each slot moves through:

```text
FREE -> DECODING -> PREPROCESSING -> INFERENCE -> POSTPROCESSING -> SERIALIZING -> FREE
```

Each transition has one owner, an explicit completion event, and an error transition back through cleanup.
No stage retains a pointer after returning the slot.
Queues are bounded and export current depth and high-water mark.

The offline evaluator blocks rather than dropping frames.
The live preview profile may drop an oldest not-yet-inferred sequence when its declared freshness budget is exceeded.
Any dropped frame is recorded and invalidates accuracy evaluation on that run.

### 21.3 Device tensors and synchronization

The CUDA profile uses ONNX Runtime I/O binding for device-resident input and output when Milestone 0 proves the path.
Custom CUDA preprocessing is not required in V1.
If OpenCV CUDA is introduced later, it requires its own correctness and synchronization plan.

Every asynchronous transfer or provider run has an owned CUDA stream or documented default-stream rule.
The runtime synchronizes only at declared ownership boundaries.
Benchmark timings use CUDA events for device phases and `CLOCK_MONOTONIC_RAW` for host end-to-end phases.

### 21.4 Provider audit

`junctionlens-runtime doctor` prints available providers, library versions, GPU identity, model input and output metadata, model hash, and expected provider coverage.
The runtime captures verbose ONNX Runtime partition logs in a redacted artifact.
The parser converts those logs into a stable provider-assignment summary and retains the raw log hash.

The parser is qualified only for the exact ONNX Runtime build hash frozen in the GPU image.
Its fixtures include raw logs from that build, every expected provider, and the declared CPU exceptions for shape and metadata operations.
An ONNX Runtime version or build-hash change invalidates the parser fixture and requires requalification before provider assignment can gate a result.
The project makes no claim that verbose provider-log text is stable across ONNX Runtime releases.

The mandatory CUDA artifact includes the expected set or count of nodes assigned to CUDA.
A later run with a different assignment is rejected until requalified.

### 21.5 Numerical parity

Python PyTorch, Python ONNX Runtime CPU, C++ ONNX Runtime CPU, and C++ postprocessing must satisfy:

- FP32 raw-output maximum absolute error at most 1e-4.
- FP32 raw-output maximum relative error at most 1e-4 for values with magnitude above 1e-3.
- Identical filtered node and edge counts under frozen threshold fixtures.
- Identical deterministic node and edge IDs.
- Identical protobuf logical content except declared producer metadata.

The CUDA FP16 profile must satisfy maximum absolute error 5e-3 and preserve every discrete output in the frozen parity corpus.
Any discrete difference requires case-level review and a profile-specific tolerance decision before release.

### 21.6 Profiling

The final runtime evidence includes:

- Nsight Systems timeline with NVTX ranges for decode, preprocess, copy, inference, postprocess, track, and serialize.
- ONNX Runtime profiling output.
- Nsight Compute capture only for a confirmed hot custom or provider kernel that the project can interpret.
- CPU flame graph for postprocessing and evaluation.
- Allocation and memory-high-water reports.

Profiler overhead runs are separate from benchmark runs.
Numbers measured under a profiler must not be substituted into the release latency table.

## 22. Python CLI and service

The public CLI is built with Typer and exposes:

```text
junctionlens doctor
junctionlens data register
junctionlens data audit
junctionlens data split
junctionlens train
junctionlens export
junctionlens infer
junctionlens evaluate
junctionlens calibrate
junctionlens compare
junctionlens gate freeze
junctionlens gate decide
junctionlens fault
junctionlens report
junctionlens serve
```

Every command supports `--help`, structured JSON status, nonzero failure exit codes, and an optional human-readable view.
Commands that create artifacts print the artifact hash and immutable path.

Configuration uses validated YAML loaded into frozen Pydantic models.
Unknown keys are errors.
Environment variables are restricted to secrets, paths, and machine-local overrides rather than experiment hyperparameters.

FastAPI is a read-only local evidence API in V1.
It serves runs, artifacts, metric tables, release decisions, and redacted image proxies from registered roots.
It does not launch training or arbitrary commands.

## 23. Dashboard and visible product behavior

The React application is a static single-page client served by the local FastAPI process.
It must remain useful without a network connection after dependencies and artifacts exist.

### 23.1 Required views

The run index shows model identity, dataset, status, metrics, provider, hardware class, and provenance completeness.

The comparison view shows:

- Persisted release status and reason codes.
- Candidate-minus-baseline deltas and adjusted confidence intervals.
- Overall and gating-slice tables.
- Accuracy, topology, calibration, temporal, and runtime panels.
- Support counts and insufficient-evidence indicators.

The scene inspector shows:

- Synchronized available camera frames.
- Current-frame BEV lane geometry.
- Lane-successor arrows.
- Control nodes and control-to-lane edges.
- Candidate, baseline, and ground-truth toggles.
- Calibrated confidence and geometry intervals.
- Previous and next sequence controls.
- A clear licensed-image indicator.

The counterexample view ranks cases by a frozen severity function defined before the final evaluation.
It supports reason-code filtering but does not recalculate the release decision.

The runtime view shows phase latency distributions, throughput, memory, provider assignment, warmup, and benchmark validity.

### 23.2 Severity ranking

Counterexample severity is lexicographic rather than a post hoc weighted score:

1. Integrity fault.
2. Confident wrong control assignment.
3. Path-blocking regression.
4. Control-edge recall regression.
5. Lane topology regression.
6. Calibration regression.
7. Geometry or detection regression.

Within a class, cases sort by candidate-minus-baseline error, then calibrated confidence, then frame identity.

### 23.3 Accessibility and usability

The dashboard supports keyboard navigation, visible focus, semantic tables, non-color status labels, and a high-contrast graph palette.
Graph edges have distinct patterns in addition to color.
Loading, empty, restricted-image, insufficient-evidence, and error states must be visually tested.

The final acceptance uses a real browser at desktop and narrow widths.
Screenshots alone are not sufficient.

## 24. Security, privacy, and data handling

### 24.1 Service boundary

The service binds only to `127.0.0.1` in V1 and rejects non-loopback host configuration.
Remote serving, authentication, CORS configuration, and external TLS termination are not V1 capabilities.

The API has no arbitrary file-read endpoint.
Artifact paths are resolved below registered canonical roots, reject traversal, reject escaping symlinks, and use `openat`-style safe access where feasible.
Image responses enforce declared media types and byte limits.

### 24.2 Browser controls

The service sets a restrictive Content Security Policy, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, and frame-denial headers.
The React client does not render artifact text with raw HTML.
Report HTML escapes all model, dataset, slice, and path strings.
No third-party scripts, fonts, analytics, or CDN assets are loaded.

### 24.3 Dataset privacy

Restricted dataset frames remain in the user-selected licensed root.
The artifact registry stores references and hashes rather than copying full frames.
Private thumbnails are opt-in, ignored by Git, and labeled with source and license.
Public report export excludes them by default.

Logs redact user-specific roots, signed URLs, and unrelated environment values.
Environment capture uses an allowlist.

### 24.4 Supply chain

CI runs dependency vulnerability scanning, secret scanning, license inventory, and an SBOM generator.
Downloaded archives and source releases require checksums.
OCI images require digests.
The public release checklist blocks known critical vulnerabilities in reachable runtime dependencies unless a documented non-reachability assessment exists.

### 24.5 Parser hardening

Protobuf, JSON, YAML, Parquet, image, and archive inputs enforce byte, dimension, recursion, row, and object-count limits.
Malformed external artifacts produce typed errors rather than crashes.
C++ parsers and validators run under ASan, UBSan, and fuzz smoke tests.

## 25. Repository layout

The target tree is:

```text
junction-lens/
├── AGENTS.md
├── BUILD_PLAN.md
├── CMakeLists.txt
├── CMakePresets.json
├── LICENSE
├── README.md
├── SECURITY.md
├── pyproject.toml
├── uv.lock
├── package.json
├── pnpm-lock.yaml
├── cmake/
│   ├── dependencies.cmake
│   └── sanitizers.cmake
├── containers/
│   ├── Containerfile.cpu
│   ├── Containerfile.gpu
│   └── images.lock
├── proto/
│   └── junctionlens/v1/scene_control_graph.proto
├── schemas/
│   ├── artifact-manifest-v1.schema.json
│   └── report-v1.schema.json
├── cpp/
│   ├── include/junctionlens/
│   │   ├── contract/
│   │   ├── geometry/
│   │   ├── eval/
│   │   ├── infer/
│   │   └── track/
│   ├── src/
│   ├── bindings/
│   └── tests/
├── python/
│   └── junctionlens/
│       ├── cli/
│       ├── contract/
│       ├── data/
│       ├── model/
│       ├── training/
│       ├── calibration/
│       ├── evaluation/
│       ├── registry/
│       ├── report/
│       └── service/
├── web/
│   ├── src/
│   ├── tests/
│   └── playwright/
├── configs/
│   ├── data/
│   ├── model/
│   ├── experiments/
│   ├── metrics/
│   ├── slices/
│   └── gates/
├── fixtures/
│   ├── synthetic/
│   ├── golden/
│   └── faults/
├── tests/
│   ├── contract/
│   ├── integration/
│   ├── end_to_end/
│   ├── security/
│   └── reproducibility/
├── benchmarks/
│   ├── manifests/
│   └── analysis/
├── scripts/
│   ├── bootstrap/
│   ├── ci/
│   └── gpu/
├── docs/
│   ├── architecture.md
│   ├── data-contract.md
│   ├── dataset-and-license.md
│   ├── metrics.md
│   ├── benchmark-protocol.md
│   ├── prior-art.md
│   ├── model-card.md
│   ├── safety-and-limitations.md
│   └── interview-walkthrough.md
└── .github/workflows/
```

`artifacts/`, `data/`, `checkpoints/`, `engines/`, `profiles/`, and private thumbnails are ignored.
Required source, schemas, configs, and synthetic fixtures must not be hidden by broad ignore patterns.

## 26. Clean developer commands

The repository provides these stable commands through `justfile` or a repository-owned `tools/jl` wrapper:

```bash
./tools/jl bootstrap-cpu
./tools/jl configure-cpu
./tools/jl build-cpu
./tools/jl format-check
./tools/jl lint
./tools/jl typecheck
./tools/jl test-cpp
./tools/jl test-python
./tools/jl test-web
./tools/jl test-contract
./tools/jl test-integration-synthetic
./tools/jl test-security
./tools/jl test-reproducibility
./tools/jl verify-local
```

GPU commands are:

```bash
./tools/jl doctor-gpu
./tools/jl build-gpu
./tools/jl test-gpu
./tools/jl benchmark-gpu
./scripts/gpu/qualify_remote.sh
```

The primary unrestricted demo is:

```bash
./tools/jl demo-synthetic
uv run junctionlens serve --artifact-root artifacts/demo --open-browser
```

The licensed-data workflow is:

```bash
uv run junctionlens data register \
  --lock configs/data/openlane-v2-v2.1.lock.yaml \
  --root "$OPENLANE_V2_ROOT"

uv run junctionlens data audit --dataset openlane-v2-v2.1
uv run junctionlens infer --run configs/experiments/final-v1.yaml
uv run junctionlens compare --baseline <baseline-hash> --candidate <candidate-hash>
```

Commands must not require activation of a manually managed virtual environment.
They must fail with an actionable message when a licensed dataset or GPU prerequisite is absent.

## 27. Test and CI strategy

### 27.1 Unit and property tests

C++ tests cover:

- Matrix and transform math.
- Polyline distance and interpolation.
- Matching and tie behavior.
- Graph referential integrity.
- KPI denominators and empty cases.
- Tracker lifecycle and ID reuse.
- ONNX metadata validation.
- Buffer-pool transitions and cleanup.
- Malformed protobuf inputs.

Python tests cover:

- Data adapter normalization.
- Segment split isolation.
- Augmentation consistency.
- Hungarian targets and loss masks.
- Calibration fit and ranking invariance.
- Bootstrap pairing and decision semantics.
- Artifact hashing and atomicity.
- Fault transforms and reason codes.
- CLI failure exits and structured status.

Property tests cover:

- Transform inversion and composition.
- Correct node and adjacency permutation invariance.
- Metric bounds.
- Perfect prediction optimum.
- Monotonic fault severity for controlled drop rates.
- Registry idempotence.
- Tracker ID uniqueness.

### 27.2 Integration tests

Synthetic end-to-end tests perform:

```text
generate -> validate -> infer fixture -> evaluate -> compare -> gate -> report -> serve
```

The CPU ONNX path runs a tiny deterministic model artifact.
The web test opens the resulting report and scene inspector in a real browser.

Licensed-data integration tests are opt-in and use sample-manifest selectors.
Their absence is skipped only outside a release qualification and must never fabricate a pass.

### 27.3 C++ quality gates

The C++ pipeline runs:

- `clang-format --dry-run --Werror`.
- `clang-tidy` on project translation units.
- GCC warnings with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror` after documented third-party exclusions.
- ASan and UBSan tests.
- ThreadSanitizer on CPU-only concurrency tests in a separate build.
- Coverage for evaluator and tracker core.
- A bounded libFuzzer smoke corpus for protobuf and manifest parsing.

### 27.4 Python and web quality gates

Python runs Ruff formatting and lint, mypy strict mode on public packages, pytest, Hypothesis, and coverage.
Web runs formatter, ESLint, TypeScript no-emit checking, Vitest, accessibility assertions, and Playwright.

The initial coverage floors are 85 percent branch coverage for the Python evaluation, registry, and gate packages, 80 percent line coverage for C++ evaluator and tracker code, and 80 percent statement coverage for dashboard decision rendering.
Coverage is a floor rather than a substitute for real-path tests.

### 27.5 CI jobs

Pull-request CI uses only public source and synthetic fixtures.
It includes:

- Dependency-lock verification.
- Generated-file drift check.
- License and secret scan.
- C++ GCC release build and tests.
- C++ Clang sanitizer build and tests.
- Python lint, types, unit, and property tests.
- Web lint, unit, accessibility, and browser tests.
- Contract and protobuf compatibility.
- Synthetic end-to-end workflow.
- Deterministic report hash check.
- CPU ONNX inference parity.
- OCI build without GPU execution.

Nightly or manually authorized private CI may use licensed sample data and a GPU runner.
Its artifacts remain private unless redacted.

### 27.6 Flake policy

A flaky test is a failure.
Retries may collect evidence but may not convert an initial failure to a passing release gate.
Randomized tests log their seed.
Any quarantined test requires an owner, issue, reason, and expiry date, and no quarantined test may cover a release-critical path.

## 28. Local-first GPU handoff

Training, CUDA inference, TensorRT qualification, Nsight profiling, and accelerated benchmarks require a compatible remote Linux GPU environment.
The repository must not contain a private hostname, address, username, home path, dataset path, or credential.
Machine-local values are supplied through ignored environment configuration.

### 28.1 Work-state model

A local work package is `PENDING_LOCAL` or `IMPLEMENTED_LOCAL`.
A target-only gate is `DEFERRED_HARDWARE`, `BLOCKED`, `FAILED`, or `ACCEPTED`.
A milestone is complete only when every local work package is `IMPLEMENTED_LOCAL` and every applicable target-only gate is `ACCEPTED`.

The implementation agent may continue later hardware-independent work after a target gate becomes `DEFERRED_HARDWARE`.
It may not mark the milestone, portfolio checkpoint, performance result, or public claim accepted by assumption.

The implementation agent must complete every meaningful macOS or CPU test, source file, fixture, analysis path, dashboard path, and automation script before requiring remote interaction.

### 28.2 Stable entry points

The repository owns:

- `scripts/gpu/qualify_remote.sh` as the single local entry point.
- `scripts/gpu/run_remote_qualification.sh` as the noninteractive resumable remote runner.

The local entry point consumes:

- `JUNCTIONLENS_GPU_HOST` for an SSH alias.
- `JUNCTIONLENS_REMOTE_ROOT` for an optional path below the resolved remote home.
- `JUNCTIONLENS_REMOTE_DATA_ROOT` for an existing licensed dataset root.
- `JUNCTIONLENS_GPU_UUID` only when an explicit GPU override is necessary.

It must never supply secret values on a remote command line.

### 28.3 Source synchronization

The local script must:

1. Refuse an unexplained dirty worktree.
2. Record the Git commit, submodule commits, dependency locks, image digests, tracked paths, file modes, symlink targets, and file SHA-256 values.
3. Create a source bundle only from declared Git-tracked files.
4. Reject absolute archive paths, traversal, special files, and escaping symbolic links.
5. Derive a content-addressed remote directory below the validated remote root.
6. Transfer without deleting unrelated remote content.
7. Verify the manifest before and after safe extraction.
8. Acquire one idempotency lock per source bundle.
9. Start or resume the remote runner in `tmux`, with a tested `nohup` fallback.
10. Reconnect and poll without losing the remote log.
11. Retrieve results into a new no-clobber ignored local directory.
12. Verify remote and local `SHA256SUMS`.
13. Exit nonzero unless the top-level result is exactly `PASSED`.

The script must not transfer datasets, private caches, credentials, model stores unrelated to the run, or ignored artifacts.

### 28.4 Remote preflight

The remote runner records:

- Host and OS identity.
- CPU and memory.
- Every visible GPU model, UUID, compute capability, memory, and health state.
- Selected GPU and selection reason.
- Driver, CUDA runtime, CUDA toolkit, cuDNN, TensorRT, and ONNX Runtime versions.
- Compiler, CMake, Ninja, Python, Node, container, and browser versions.
- Dataset-lock and selected-archive verification state.
- Disk-space and inode availability.
- Competing GPU process, utilization, clock, power, thermal, ECC, and Xid state.

The default GPU selection policy chooses the lexicographically smallest healthy UUID that satisfies the model memory and compute requirements.
An override is recorded and validated.

The runner acquires an exclusive lock keyed by GPU UUID for performance phases.
It refuses a live lock and reclaims a stale lock only after owner PID, session, heartbeat, and bundle identity checks.

### 28.5 Remote phases

The resumable remote phases are:

1. Environment and data preflight.
2. Clean dependency and container resolution.
3. C++ and Python GPU build.
4. Complete CPU and synthetic verification on Linux.
5. CUDA provider availability and node-assignment audit.
6. TensorRT provider probe and conditional partition audit.
7. CUDA numerical parity.
8. Dataset adapter and official evaluator compatibility on the licensed sample.
9. Thirty-two-frame micro-overfit.
10. Baseline and candidate training or resume from content-matched checkpoints.
11. Calibration and frozen evaluations.
12. Fault-lab qualification.
13. Accelerated latency, throughput, memory, and long-run stability.
14. Nsight Systems and selected Nsight Compute captures.
15. Evidence bundle, redaction, hashes, and final report.

A phase is reused only when its source, executable, configuration, dataset, model, environment, and declared-output hashes match.
Phase transitions are written atomically and durably.

The runner continues independent diagnostic phases after a nonfatal failure so one session returns the complete problem list.
It writes `USER_ACTION_REQUIRED.md` once when a license, package, disk, permission, or external prerequisite prevents progress.
It must not ask for dozens of manual commands.

### 28.6 Benchmark validity

The performance runner samples utilization, memory, temperature, clocks, power, throttle reasons, ECC state, Xid messages, and competing processes throughout each benchmark.
The acceptance charter freezes contention, thermal, clock, and error bounds after M0.
A contaminated run is `BLOCKED_INFRASTRUCTURE` and cannot publish performance numbers.

Training may share a GPU if declared.
Release latency and memory qualification requires exclusive use.

### 28.7 Remote outputs

Every phase directory contains:

- Redacted command.
- Named environment allowlist without secret values.
- Redacted stdout and stderr.
- Start and end timestamps.
- Duration.
- Structured status and reason.
- Input and output hashes.

The top-level result contains:

- `status.json`.
- `environment.json`.
- `commands.jsonl`.
- `junit.xml`.
- `benchmarks.json`.
- `provider-assignment.json`.
- `SHA256SUMS`.
- `REPORT.md`.
- `USER_ACTION_REQUIRED.md` when blocked.

If the implementation environment can use SSH, the implementation agent runs the entry point, retrieves results, fixes in-scope defects, and repeats it without asking the user to relay intermediate commands.
If the implementation environment cannot use SSH, the only requested user action is running `./scripts/gpu/qualify_remote.sh` from the repository root and returning control after it retrieves the bundle.

## 29. Milestone execution rules

The implementation follows this dependency order: Milestones 0 through 5, Milestone 7, the CPU and CUDA portions of Milestone 8, Milestones 9 and 10 at core depth, the development-complete core checkpoint, conditional Milestone 6, the TensorRT and richer-dashboard extensions, Milestone 11, and Milestone 12.
This order deliberately builds the evaluator and runtime before any temporal model can receive final promotion.
Each submilestone has a focused commit after its gate passes.
No commit may combine unverified work from a later milestone merely for convenience.
When a target-only gate requires clean Git-tracked source, a focused qualification commit may be created and pushed after its complete local work package reaches `IMPLEMENTED_LOCAL`.
That qualification commit does not mark the submilestone `ACCEPTED`, and the accepted target evidence receives a separate focused commit after the target-only gate passes.

The preferred commit format is Conventional Commits.
After every verified focused commit, push the current branch to `origin` immediately and confirm that the remote commit matches before continuing.
A push failure is resolved before accumulating additional unpushed milestone commits.

Generated data, restricted assets, models, profiler reports, engines, caches, and private evidence are never committed.

## 30. Milestone 0: feasibility, evidence, and locks

### 30.1 M0.1: toolchain and source locks

Deliver:

- Repository instructions and license skeleton.
- Exact Python, C++, web, protobuf, ONNX Runtime, CUDA, and optional TensorRT locks.
- OCI image digest lock.
- OpenLane-V2 v2.1.0 source and license lock.
- `junctionlens doctor` skeleton that reports real rather than assumed capabilities.
- ADRs for any compatibility-driven changes from this plan.

Validate:

- A clean CPU bootstrap resolves from locks.
- All release archives and images match hashes.
- `doctor --json` validates against its schema.
- Missing GPU, dataset, and TensorRT states are reported distinctly.

Focused commit:

```text
chore: pin toolchains and evidence sources
```

### 30.2 M0.2: dataset and evaluator feasibility

Deliver:

- License-acknowledgment workflow.
- Sample adapter.
- Camera-slot and coordinate normalization.
- Label-capacity audit.
- Digest-pinned Python 3.8 official-evaluator compatibility image and restricted wrapper.
- Fixed perfect and corrupted prediction fixtures.

Validate:

- Official sample checksums pass.
- Every camera and pose transform passes range and round-trip checks.
- Sequence IDs are stable where the dataset contract says they are stable.
- Lane, traffic-control, and road-area source IDs are audited separately for cross-frame persistence, collision, and reuse before any temporal KPI is enabled.
- Query capacities cover at least 99.9 percent of eligible audited frames or are increased before freeze.
- Perfect fixtures reach their expected optimum.
- Seeded corruptions alter the expected official components.

Focused commit:

```text
feat(data): prove openlane adapter and evaluator
```

### 30.3 M0.3: model and deployment spike

Deliver:

- Minimal reference encoder, projection, and query outputs.
- Thirty-two-frame micro-overfit command.
- ONNX opset-18 exporter.
- C++ ONNX Runtime loader and metadata validator.
- CPU and available CUDA provider trace.
- TensorRT provider probe when available.

Validate:

- The micro-overfit gate in Section 12.7 passes.
- ONNX checker and shape inference pass.
- PyTorch and C++ CPU raw-output parity passes.
- The CUDA profile has no unexpected CPU nodes.
- TensorRT node partition is recorded without being assumed complete.
- M0 benchmark records a credible path to the absolute V1 latency and memory budgets.
- M0 records measured training sequences per second, peak training memory, checkpoint bytes, evaluator segments per second, and expected artifact growth.
- `configs/budgets/v1.yaml` freezes a maximum of 120 GPU-hours and 400 GiB of generated artifacts before full training.
- E1, E2, and E3 screen with seed 20260813 only, and only E0 plus the selected final architecture receive the two robustness seeds.
- The estimated experiment matrix, checkpoint cadence, retained-artifact policy, and screening-to-final promotion rule fit inside those limits with at least 20 percent contingency.

If the M0 extrapolation exceeds either budget, the project applies the Section 43 model and scope-reduction ladder before full training.
An over-budget run aborts at its next safe checkpoint and cannot consume contingency silently.

Focused commit:

```text
spike(model): prove graph inference path
```

### 30.4 M0 decision

Proceed when all available M0 local gates pass and target-only gates are accepted or explicitly `DEFERRED_HARDWARE` under Section 28.

Kill or pivot when:

- Required dataset labels or calibration cannot be validated.
- License terms prevent the intended private research use.
- The model cannot overfit the tiny fixed set after data and loss debugging.
- CPU ONNX export cannot reproduce PyTorch outputs.
- CUDA execution requires unsupported custom operators with no maintainable equivalent.
- A realistic fixed-shape profile cannot fit within 6 GiB after the model simplification ladder is exhausted.

TensorRT failure alone removes the TensorRT profile rather than killing JunctionLens.

## 31. Milestone 1: contracts, geometry, and synthetic truth

### 31.1 M1.1: protobuf and validators

Implement the complete V1 protobuf schema, generated bindings, JSON conversion, schema validators, ID rules, size limits, and compatibility tests.

Acceptance requires:

- C++ and Python parse the same golden messages.
- Invalid IDs, edges, transforms, boxes, and nonfinite values fail with stable reason codes.
- Schema round trips preserve logical content.
- A forward-compatible unknown minor field is tolerated while a major mismatch fails.

Focused commit:

```text
feat(contract): define scene control graph v1
```

### 31.2 M1.2: geometry core

Implement transforms, projection, polyline interpolation, discrete Frechet distance, Chamfer distance, IoU, endpoint features, and deterministic Hungarian wrappers.

Acceptance requires all numerical coordinate gates, property tests, malformed-input tests, and C++ sanitizer tests to pass.

Focused commit:

```text
feat(geometry): add calibrated graph primitives
```

### 31.3 M1.3: synthetic generator

Implement deterministic roads, merges, splits, intersections, controls, crosswalks, camera calibration, temporal ego motion, perfect predictions, and controlled corruptions.

Acceptance requires:

- Generated protobuf validates.
- Rendered projections agree with analytic truth.
- All mandatory graph shapes occur in the frozen fixture corpus.
- Repeated generation with one seed is byte-identical.

Focused commit:

```text
test(fixtures): add synthetic graph truth corpus
```

## 32. Milestone 2: production data path

### 32.1 M2.1: complete adapter

Implement lazy image loading, metadata parsing, Map Element Bucket conversion, calibration normalization, camera masks, ego pose, area labels, topology matrices, and source-domain metadata.

Acceptance requires comparison against the official devkit on a frozen sample and no eager loading of the full dataset.

Focused commit:

```text
feat(data): normalize openlane graph sequences
```

### 32.2 M2.2: manifests and splits

Implement content manifests, deterministic segment grouping, stratification, split audit, leakage rejection, and immutable registry artifacts.

Acceptance requires zero segment overlap, stable hashes across reruns, expected allocation counts, and a deliberate seeded-leak failure.

Focused commit:

```text
feat(data): freeze segment isolated splits
```

### 32.3 M2.3: visual and statistical audit

Implement calibration overlays, BEV label rendering, class and capacity distributions, missing-camera patterns, geometry ranges, topology support, and slice-support previews.

Acceptance requires manual inspection of a frozen diverse sample plus automated projection and range tests.

Focused commit:

```text
feat(data): add dataset audit and visual checks
```

## 33. Milestone 3: evaluator and KPI core

### 33.1 M3.1: official evaluator compatibility

Implement the pinned Python 3.8 compatibility container, trusted-input wrapper, threshold-specific upstream artifact capture, and modern Python output check.

Acceptance requires every Section 15.3 compatibility item and order-invariance control to pass.

Focused commit:

```text
feat(eval): reproduce openlane v2 metrics
```

### 33.2 M3.2: graph and temporal KPIs

Implement `CustomMatchV1` and the exact custom definitions from Section 16 with immutable association artifacts and per-frame and per-segment outputs.

Acceptance requires hand-calculated goldens, property bounds, empty-denominator behavior, and monotonic controlled faults.

Focused commit:

```text
feat(eval): add control graph and temporal kpis
```

### 33.3 M3.3: calibration and runtime KPIs

Implement Brier, NLL, adaptive ECE, AURC, geometry coverage, interval width, and runtime distributions.

Acceptance requires analytic fixtures, deterministic ties, clipping counts, and separation of warmup from measured samples.

Focused commit:

```text
feat(eval): add uncertainty and runtime evidence
```

## 34. Milestone 4: baseline model and charter freeze

### 34.1 M4.1: independent baseline

Build and train E0 with the shared node architecture and frozen geometric edge rules.

Acceptance requires:

- Stable training under the declared recipe.
- Complete model and dataset manifests.
- No use of selection, calibration, or holdout data in training statistics.
- Three repeatable baseline seeds for the final configuration.
- A model card with limitations and failed examples.

Focused commit:

```text
feat(model): establish independent graph baseline
```

### 34.2 M4.2: acceptance charter

Implement paired segment bootstrap, multiple-comparison adjustment, support rules, margin semantics, freeze command, and persisted reason codes.

Freeze V1 only from baseline variability measured on training and model-selection data, the pre-holdout power simulation, and predeclared product priorities.
The internal holdout is unavailable to this milestone by command-level policy.

Acceptance requires seeded pass, fail, insufficient, integrity, performance, and infrastructure cases.

Focused commit:

```text
feat(gate): freeze versioned release charter
```

## 35. Milestone 5: joint graph model

### 35.1 M5.1: learned topology

Implement lane-successor and control-to-lane heads, edge matching, class imbalance handling, topology losses, and query masking.

Acceptance requires oracle-node topology learning, full predicted-node training, and no topology-matrix ordering defect.

Focused commit:

```text
feat(model): learn lane and control topology
```

### 35.2 M5.2: E1 experiment

Train E1 under the frozen selection rule and compare it with E0 on the model-selection partition.

Acceptance requires the E1 keep gates in Section 13.4.
If they fail, retain E0 and publish the negative result while fixing only demonstrated correctness defects.

Focused commit:

```text
test(model): validate joint topology hypothesis
```

## 36. Milestone 6: temporal model and uncertainty

### 36.1 M6.1: temporal fusion and tracker

Implement ego-motion warp, ConvGRU, temporal training masks, track embeddings, C++ tracker, track lifecycle, and temporal KPI fixtures.

Acceptance requires pose-invalid fallback, track-ID integrity, temporal numerical checks, and no future-frame leakage.

Focused commit:

```text
feat(model): add temporal graph consistency
```

### 36.2 M6.2: calibration

Implement head temperatures, geometry scales, calibration artifacts, ranking-invariance checks, and selective-risk analysis.

Acceptance requires disjoint calibration data and every E3 keep gate.

Focused commit:

```text
feat(model): calibrate graph uncertainty
```

### 36.3 M6.3: frozen E2 and E3 study

Run seed-20260813 screening experiments on the training, model-selection, and calibration partitions, generate complete selection tables, and select E1, E2, or E3 only through the frozen rules.
Do not evaluate any candidate on the internal holdout during this milestone.

Focused commit:

```text
test(model): freeze temporal calibration study
```

## 37. Milestone 7: registry, comparison, and fault lab

### 37.1 M7.1: registry

Implement content storage, manifests, atomic writes, locking, DuckDB indexing, resume, garbage-collection dry run, and provenance views.

Acceptance requires crash injection, concurrent-reader tests, stale-lock tests, and byte-identical reruns.

Focused commit:

```text
feat(registry): add immutable evidence store
```

### 37.2 M7.2: comparison and decision

Implement exact-frame pairing, slice materialization, bootstrap intervals, support rules, persisted decisions, and report data.

Acceptance requires every release status, metric direction, margin boundary, and multiple-comparison fixture.

Focused commit:

```text
feat(gate): decide paired model releases
```

### 37.3 M7.3: fault lab

Implement all Section 20 faults and nonfault controls.

Acceptance requires 100 percent intended detection, nearby clean passes, immutable parent links, and the flagship swapped-control demonstration.

Focused commit:

```text
feat(faults): prove structural regression gates
```

## 38. Milestone 8: production inference runtime

### 38.1 M8.1: CPU runtime

Implement preprocessing, ONNX Runtime session, postprocessing, protobuf output, batch CLI, buffer ownership, and CPU parity.

Acceptance requires malformed-input handling, clean shutdown, repeated load and unload, and exact discrete parity.

Focused commit:

```text
feat(runtime): add cxx graph inference pipeline
```

### 38.2 M8.2: CUDA profile and conditional TensorRT extension

Implement I/O binding, provider options, provider trace parser, cache keys, CUDA parity, and conditional TensorRT profile.

Core acceptance requires no unexpected CUDA-profile CPU fallback and requires the model-selection runtime promotion for E2 when E2 is under consideration.
TensorRT work starts only after the CUDA core passes and accepts truthful partial partition reporting as an extension result.

Focused commit:

```text
perf(runtime): qualify accelerated providers
```

### 38.3 M8.3: performance and stability

Implement phase timing, NVTX, memory high-water tracking, 10,000-frame stability, benchmark validity, and profiler automation.

Acceptance requires every absolute runtime budget in Section 19.3 on the qualification environment.

Focused commit:

```text
perf(runtime): meet graph latency budget
```

## 39. Milestone 9: user-facing service and dashboard

### 39.1 M9.1: CLI and API

Finish every public command, read-only FastAPI routes, schemas, pagination, artifact-root security, and persisted decision serving.

Acceptance requires CLI and API end-to-end tests, non-loopback rejection tests, and stable error contracts.

Focused commit:

```text
feat(app): expose evidence cli and api
```

### 39.2 M9.2: comparison dashboard

At core depth, implement a thin read-only scene viewer for synchronized available cameras, BEV lanes, controls, graph edges, candidate and baseline toggles, persisted status, and frame navigation.
After the V1.1 promotion gate, add the full run index, KPI intervals, slice tables, temporal controls, calibration panels, counterexample workflow, and runtime view.

Core acceptance requires real browser tests, accessibility checks, restricted-image states, and correct persisted decision rendering for the thin viewer.
Extension acceptance applies the same checks to every richer comparative view.

Focused commit:

```text
feat(web): visualize control graph regressions
```

### 39.3 M9.3: evidence reports

Implement deterministic Markdown, HTML, JSON, and Parquet report outputs with public and private export modes.

Acceptance requires repeatable hashes, escaping tests, no external assets, and a clean offline open.

Focused commit:

```text
feat(report): export reproducible evidence bundles
```

## 40. Milestone 10: security and full system qualification

### 40.1 M10.1: security hardening

Complete path containment, parser limits, headers, token redaction, dependency scan, SBOM, fuzz smoke, secret scan, and license inventory.

Focused commit:

```text
security: harden local evidence service
```

### 40.2 M10.2: synthetic clean-checkout acceptance

From a clean checkout, execute bootstrap, build, full local verification, synthetic generation, evaluation, comparison, gate, report, and browser inspection.

Acceptance requires no manual file edits, hidden cache dependency, stale generated file, missing source, or unexplained worktree change.

Focused commit:

```text
test(e2e): verify clean synthetic workflow
```

### 40.3 Development-complete core checkpoint

The development-complete portfolio core exists only after the core-depth gates in the Section 29 execution order pass.
It requires:

- A working synthetic unrestricted demo.
- A working licensed-data path when data is available.
- An owned baseline and frozen candidate artifact or a truthful negative result.
- Official evaluator compatibility evidence.
- Custom graph and runtime evidence.
- A fault-lab proof.
- C++ CPU and accepted accelerated runtime.
- A visible thin graph-diff scene viewer.
- A self-contained evidence report.

This checkpoint is not a holdout acceptance and must not call a candidate final, improved, or accepted.
Temporal, calibration, TensorRT, richer-dashboard, and source-domain extension work proceeds only if the checkpoint and budget promotion gate pass.

## 41. Milestone 11: frozen final benchmark

### 41.1 M11.1: final matrix

Run the predeclared seed-20260813 E0 and selected final candidate once on the frozen internal holdout.
Run seeds 20260814 and 20260815 only as robustness replications after the seed-20260813 release decision is irreversibly persisted, and never use them to replace that decision.
Run the selected final artifact once on the official subset A validation benchmark.
Before any M11 inference, the registry freezes the exact seed-20260813 candidate artifact that will be used for both internal holdout and official validation.
The official-validation result cannot select a different checkpoint or feed back into the holdout decision.
Run the optional subset B diagnostic only if its terms, data, and support are valid.

The matrix includes official, control-association, topology, temporal, calibration, source-domain, long-tail, fault, latency, throughput, and memory tables.

No model, threshold, calibrator, slice, or margin changes are allowed after viewing these results without declaring a new experiment version.

Focused commit:

```text
test(benchmark): publish frozen v1 evidence
```

### 41.2 M11.2: independent reproduction

Reproduce the synthetic workflow in a fresh CPU environment and the accelerated benchmark from the content-addressed remote bundle.
Verify every retrieved hash and rerun the release decision from artifacts alone.

Focused commit:

```text
test(repro): verify v1 evidence independently
```

## 42. Milestone 12: portfolio and release documentation

### 42.1 Public documentation

Complete:

- README with a concise problem, demo, architecture, measured results, and limitations.
- Architecture document with data-flow and ownership diagrams.
- Data contract and coordinate reference.
- Dataset and license guide.
- Metric and statistical protocol.
- Prior-art matrix.
- Model card.
- Safety and limitations document.
- Reproduction guide.
- Security policy.
- Three-minute interview walkthrough.

All claims must trace to a committed or content-addressed evidence bundle.
All screenshots must be public-safe or explicitly private.

Focused commit:

```text
docs: publish junctionlens v1 evidence
```

### 42.2 Final audit

Audit for:

- TODO, FIXME, placeholder, stub, and mocked-success paths.
- Restricted data and derived thumbnails.
- Secrets and personal paths.
- Internal names and private context.
- Floating dependencies and mutable image tags.
- Missing license notices.
- Stale results or screenshots.
- Unsupported public claims.
- Untracked required source.
- Generated artifacts accidentally staged.
- Unexplained dirty worktree state.

Run the complete verification suite again after documentation changes.

Focused commit:

```text
chore: finalize v1 release audit
```

## 43. Kill, pivot, and scope-reduction ladder

The project must preserve its product thesis while reducing implementation risk in a declared order.

### 43.1 Model memory or latency pressure

Apply these reductions in order:

1. Reduce hidden dimension from 256 to 192.
2. Reduce decoder layers from four to three.
3. Reduce feature channels while preserving query capacities.
4. Reduce BEV resolution from 0.5 to 0.625 meters and re-freeze the model profile before training.
5. Replace the backbone with a smaller documented architecture.
6. Disable the optional SD-map adapter.

Do not reduce query capacities below audited label coverage.
Do not shrink the camera set by silently discarding valid source cameras.
Do not remove lane-control topology because it is the product-defining output.

### 43.2 Temporal failure

If E2 fails its keep gate after correctness review, publish E1 as the final model and retain temporal evaluation as a documented negative experiment.
Do not force temporal fusion into V1 merely to satisfy the initial design.

### 43.3 Calibration failure

If probability calibration fails, publish raw scores as uncalibrated and disable calibrated release cells.
The final release status is then `INSUFFICIENT_EVIDENCE` for any policy requiring calibrated probabilities.
Do not relabel raw confidence as probability.

### 43.4 TensorRT failure

If TensorRT cannot cover enough graph nodes or meet stability requirements, retain CUDA EP as the accelerated V1 path.
Report the failed TensorRT profile with provider evidence.

### 43.5 Dataset or license failure

If full OpenLane-V2 access is unavailable, complete the entire unrestricted synthetic product and sample adapter but do not claim a trained portfolio model or final generalization evidence.
The data-dependent milestones remain blocked rather than passed.

### 43.6 Product kill condition

JunctionLens should be abandoned or substantially replanned if public data cannot support lane-to-control topology, if the owned model cannot learn beyond the independent baseline after validated training, and if the fault lab cannot show that the release system adds evidence beyond official metrics.
Those three failures would remove the project’s distinctive contribution.

## 44. Portfolio artifacts and interview story

The final portfolio package includes:

- A 30-second visual teaser showing a correct and wrong lane-control association.
- A three-minute narrated workflow from registered models to release decision.
- One architecture diagram showing data, model, runtime, evaluator, registry, and dashboard boundaries.
- One model diagram showing cameras, BEV projection, temporal fusion, queries, and graph heads.
- One table comparing E0, E1, E2, and E3 under frozen metrics.
- One fault-study table proving intended detection.
- One calibration reliability plot and risk-coverage plot.
- One source-domain and long-tail slice heatmap.
- One runtime latency distribution and Nsight timeline.
- One public-safe evidence bundle.
- A concise model card and limitations page.

The interview explanation is:

1. A detector can be right about objects and wrong about road structure.
2. JunctionLens represents lanes and controls as one temporal uncertain graph.
3. The model learns topology jointly rather than linking independent detections only by distance.
4. The evaluator measures control applicability, reachability, stability, calibration, and runtime.
5. The release policy is frozen before candidate holdout results and resamples by segment.
6. The fault lab proves why those gates matter.
7. The same protobuf crosses Python training, C++ inference, evaluation, and the dashboard.

The story must include one failed hypothesis, tradeoff, or pivot if one occurred.
It must not pretend every experiment succeeded.

## 45. Definition of done

JunctionLens V1 is complete only when:

- Every milestone and applicable submilestone is accepted.
- Every unresolved target-only gate is accepted rather than deferred.
- A clean CPU checkout passes the full synthetic workflow.
- The licensed data path passes when required for model claims.
- Official evaluator compatibility and wrapper-output checks pass.
- The baseline and final selected model are frozen and reproducible.
- Any calibration or temporal claim matches its keep-gate and superiority outcome, and omitted claims are labeled conditional extensions.
- The fault lab catches every mandatory fault.
- The release decision reproduces from immutable artifacts.
- The C++ CPU and CUDA paths meet their parity contracts.
- The accelerated profile meets the absolute latency, throughput, memory, fallback, and stability budgets.
- TensorRT claims are limited to measured provider coverage.
- The thin viewer and every implemented dashboard extension have been inspected in a real browser.
- Security, license, secret, and restricted-data audits pass.
- Public documentation contains no internal or personal context.
- Every result and claim points to evidence.
- Git history contains focused verified commits.
- The final worktree contains no unexplained changes.

## 46. Authoritative public sources

### 46.1 Problem and product evidence

- NVIDIA DRIVE Labs describes WaitNet, LightNet, and SignNet as components for detecting and classifying intersections, traffic lights, and traffic signs in wait-condition perception: https://developer.nvidia.com/blog/?p=14568
- NVIDIA DriveWorks documents PathNet inputs, path edges, adjacent paths, confidence, and known domain and geometry limitations: https://docs.nvidia.com/drive/archive/driveworks-3.0/pathnet_mainsection.html
- NVIDIA publicly describes PathNet and RoadNet as identifying the path an autonomous vehicle takes in an AV model-training system: https://developer.nvidia.com/blog/federated-learning-in-autonomous-vehicles-using-cross-border-training/
- NVIDIA publicly explains that path, open-road, lane, map, and wait-condition networks contribute different road-scene evidence: https://blogs.nvidia.com/blog/self-driving-cars-make-decisions/

### 46.2 Dataset and metrics

- OpenLane-V2 repository and task overview: https://github.com/OpenDriveLab/OpenLane-V2
- OpenLane-V2 v2.1 metric definitions: https://github.com/OpenDriveLab/OpenLane-V2/blob/master/docs/metrics.md
- OpenLane-V2 data hierarchy, official checksums, and download sizes: https://github.com/OpenDriveLab/OpenLane-V2/blob/master/data/README.md
- OpenLane-V2 data semantics and topology format: https://github.com/OpenDriveLab/OpenLane-V2/blob/master/docs/getting_started.md
- OpenLane-V2 v2.1 evaluator thresholds and matching implementation: https://github.com/OpenDriveLab/OpenLane-V2/blob/v2.1.0/openlanev2/lanesegment/evaluation/evaluate.py
- OpenLane-V2 v2.1 compatibility requirements: https://github.com/OpenDriveLab/OpenLane-V2/blob/v2.1.0/requirements.txt
- OpenLane-V2 paper: https://papers.neurips.cc/paper_files/paper/2023/file/3c0a4c8c236144f1b99b7e1531debe9c-Paper-Datasets_and_Benchmarks.pdf
- Argoverse 2 HD-map lane-graph semantics: https://argoverse.github.io/user-guide/api/hd_maps.html

### 46.3 Runtime and evaluation architecture

- ONNX Runtime 1.25 roadmap and release status: https://onnxruntime.ai/roadmap
- ONNX Runtime v1.25.0 release: https://github.com/microsoft/onnxruntime/releases/tag/v1.25.0
- ONNX Runtime v1.25.0 TensorRT CI version pins: https://github.com/microsoft/onnxruntime/blob/v1.25.0/tools/ci_build/github/azure-pipelines/templates/common-variables.yml
- ONNX Runtime CUDA and cuDNN load-time dependency design: https://github.com/microsoft/onnxruntime/blob/v1.25.0/docs/CUDA_cuDNN_Optional_Design.md
- ONNX Runtime CUDA execution provider: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX Runtime TensorRT execution provider and fallback recommendation: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- ONNX Runtime device tensors and I/O binding: https://onnxruntime.ai/docs/performance/device-tensor.html
- Protobuf ProtoJSON mapping for 64-bit integers: https://protobuf.dev/programming-guides/json/
- NVIDIA AlpaSim public architecture separates runtime logs from evaluation: https://github.com/NVlabs/alpasim/blob/main/docs/DESIGN.md
- NVIDIA AlpaSim public plugin documentation includes evaluation scorers: https://github.com/NVlabs/alpasim/blob/main/docs/PLUGIN_SYSTEM.md
- NVIDIA AlpaSim public telemetry documentation describes persisted Prometheus runtime evidence: https://github.com/NVlabs/alpasim/blob/main/docs/TELEMETRY.md

### 46.4 Source-use rules

Official documentation, standards, pinned source, and dataset code own behavioral contracts.
Secondary articles may inform prior-art discovery but must not override an owning specification.
Every source used for a public claim must be dated or version-pinned in the evidence notes.
Long quotations are unnecessary.
The project should paraphrase behavior and link the owning page.

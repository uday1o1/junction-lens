# Remote GPU qualification

JunctionLens owns one local entry point for Linux GPU qualification.
It transfers only a deterministic manifest of Git-tracked source files and never transfers datasets, credentials, caches, models, engines, or ignored artifacts.

## Prerequisites

Configure an SSH alias for a compatible Ubuntu 24.04 x86-64 machine with an NVIDIA GPU, driver 570.26 or newer, CUDA 12.8, and cuDNN 9.14.0.64.
The remote account needs `python3`, `git`, `ssh`, `scp`, and network access to the exact locked public dependency sources.
The `runtime-performance`, `core`, and `full-v1` profiles also require the Nsight Systems `nsys` CLI.
TensorRT 10.14.1.48 remains conditional and does not block the mandatory CUDA result.

Set machine-local values in the invoking environment.
Do not commit them.

```sh
export JUNCTIONLENS_GPU_HOST=my-private-ssh-alias
export JUNCTIONLENS_REMOTE_ROOT=.junctionlens/qualification
export JUNCTIONLENS_REMOTE_DATA_ROOT=/existing/licensed/openlane-v2
./scripts/gpu/qualify_remote.sh --profile runtime-cuda
```

`JUNCTIONLENS_REMOTE_DATA_ROOT` is optional for the `runtime-cuda` profile because that profile uses only repository-owned synthetic parity inputs.
It is also optional for `runtime-performance`, which uses the same repository-owned synthetic inputs for the absolute runtime gate.
`JUNCTIONLENS_GPU_UUID` may select one visible GPU explicitly.
Without an override, the runner selects the lexicographically smallest healthy UUID that satisfies the memory requirement.

## Source synchronization

The local entry point refuses a dirty worktree.
It records the Git commit, Git modes, symbolic-link targets, submodule commits, dependency-lock hashes, image-lock hash, file sizes, and file SHA-256 values.
It creates a deterministic tar containing only declared tracked regular files and safe relative symbolic links.

The remote extractor verifies the manifest and archive hash before extraction, rejects absolute paths, traversal, special files, undeclared members, duplicate members, escaping symbolic links, size changes, mode changes, and post-extraction digest changes.
Each source manifest and profile maps to one content-addressed workspace below the validated remote root.
Existing unrelated remote content is never deleted.

The runner uses `tmux` when available and a PID-checked `nohup` fallback otherwise.
Repeated invocations reconnect to the same content-addressed run.
Phase reuse requires identical source, command, profile, selected GPU, and declared input hashes.
Adding a signed visual-review receipt invalidates only the review and downstream model-evidence phases.
It does not repeat dependency resolution, builds, mechanical data auditing, parity, performance, profiling, or fault qualification.

## Result contract

The entry point polls the remote status, retrieves one new no-clobber result under `.junctionlens/qualification`, verifies every entry in `SHA256SUMS`, prints the local result path, and exits nonzero unless the top-level status is exactly `PASSED`.

Each phase records its redacted command, named non-secret environment allowlist, stdout, stderr, timestamps, duration, return code, structured status, and input hash.
The top-level bundle contains `status.json`, `environment.json`, `commands.jsonl`, `junit.xml`, `benchmarks.json`, `provider-assignment.json`, `REPORT.md`, and `SHA256SUMS`.
Blocked and failed runs also contain `USER_ACTION_REQUIRED.md` with the exact reason code and the one resumption command.

The `runtime-cuda` profile is the Milestone 8.2 target gate only.
It does not accept the later portfolio core checkpoint or full V1 release.
The Milestone 8.3 target gate uses:

```sh
./scripts/gpu/qualify_remote.sh --profile runtime-performance
```

That profile inherits CUDA correctness, then runs the frozen latency and stability protocol under continuous GPU telemetry and records separate Nsight Systems and ONNX Runtime profiles.
It exits nonzero for a failed budget, an incomplete 10,000-frame run, profiler failure, or contaminated environment.

## Portfolio-core qualification

The `core` profile requires the checksum-verified full dataset and an explicit ignored license receipt.
Create the receipt once from the repository root:

```sh
uv run --locked junctionlens data acknowledge \
  --accept-term Argoverse-2-terms \
  --accept-term CC-BY-NC-SA-4.0 \
  --accept-term nuScenes-terms \
  --confirm-restricted-noncommercial-use
```

Then run the consolidated entry point:

```sh
./scripts/gpu/qualify_remote.sh --profile core
```

The core profile audits the licensed adapter and official evaluator, verifies the 32-frame feasibility gate, trains and selects all predeclared E0 and E1 runs, freezes the E0 model card and acceptance charter, runs the complete seeded fault matrix, and binds accelerated runtime and profiler evidence into a private portfolio-core report.
It never accesses the internal holdout.

The first run can truthfully stop at `LICENSED_VISUAL_AUDIT_REVIEW_REQUIRED` after completing every independent phase.
The retrieved `USER_ACTION_REQUIRED.md` identifies the private audit bundle and gives the exact signoff and resume commands.
Reviewing the files is a human decision and must not be automated or inferred.
After a valid signoff, rerunning the same core command resumes the hash-matched workspace.

The `full-v1` profile remains blocked at the conditional extension boundary until this exact core evidence bundle is accepted and the Milestone 6 promotion work is authorized by its measured result.

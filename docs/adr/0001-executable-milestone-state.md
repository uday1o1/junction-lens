# ADR 0001: Executable milestone and qualification state

Status: Accepted

Date: 2026-08-13

## Context

`BUILD_PLAN.md` is the implementation authority, but an independent pre-implementation review found several literal sequencing contradictions.
The contradictions prevent source from reaching required hardware, prevent valid negative studies from completing, and make committed data locks machine-specific.
This ADR makes the smallest corrections needed to execute the plan while preserving every product, evidence, and acceptance objective.

## Decision

### Local and target states

A local work package transitions from `PENDING_LOCAL` to `IMPLEMENTED_LOCAL` after its complete local gate passes.
A target-only gate retains the states `DEFERRED_HARDWARE`, `BLOCKED`, `FAILED`, and `ACCEPTED`.
A milestone is `ACCEPTED` only when every local package is `IMPLEMENTED_LOCAL` and every applicable target-only gate is `ACCEPTED`.
`IMPLEMENTED_LOCAL` never implies hardware or milestone acceptance.

### Qualification commits

When a target gate requires Git-tracked source, a focused qualification commit may be created and pushed after all local gates pass.
Its evidence must state `IMPLEMENTED_LOCAL` and must not claim milestone acceptance.
The exact pushed commit is qualified remotely.
A follow-up focused evidence commit records the accepted target result and completes the milestone.
This exception breaks the source-transfer deadlock without relaxing either local or target verification.

### Scoped remote profiles

The single public entry point remains `scripts/gpu/qualify_remote.sh`.
The runner supports versioned `m0.3`, `runtime-cuda`, `runtime-performance`, `core`, and `full-v1` profiles as later phase handlers become available.
The `runtime-cuda` profile isolates the Milestone 8.2 target gate so a passing accelerated-runtime qualification cannot be confused with the later portfolio core checkpoint.
The `runtime-performance` profile extends that accepted correctness path with the Milestone 8.3 absolute budgets, stability run, benchmark validity audit, and non-benchmark profiler captures.
`PASSED` means every phase required by the selected profile passed.
The `full-v1` profile remains the final consolidated qualification contract.

### Staged model gates

M0 proves node classification, node geometry, loss stability, ONNX export, CPU parity, and available provider feasibility for the E0 architecture slice.
M5 adds the Section 12.7 topology thresholds when E1 owns the topology heads.
M6 adds temporal loss, leakage, stability, and calibration gates when E2 and E3 own those behaviors.
M0 freezes the E1, E2, and E3 budget allocation and promotion policy but does not claim that later architectures have already run.
Each deferred threshold is applied in full when its owning implementation exists.

### Study validity and promotion outcome

E1, E2, E3, and conditional TensorRT studies have separate execution and outcome states.
A study is accepted when its predeclared protocol, data isolation, integrity checks, and evidence pass.
Its outcome is either `PROMOTED` or `REJECTED_BY_KEEP_GATE`.
A valid negative result can therefore complete the study without fabricating a successful hypothesis.

### Extension runtime sequence

M6.1 gives E2 provisional model-selection status.
M8.2 and M8.3 are then repeated for E2 before E3 is fit.
If E2 remains eligible, M6.2 and M6.3 complete calibration and final selection.
The final selected artifact repeats runtime parity and absolute budget checks before M11.

### License acknowledgment

The committed dataset lock records license identifiers and `acknowledgment_required: true`.
The acknowledgment timestamp and terms digest are stored in an ignored machine-local receipt.
Dataset registration records only a redacted receipt hash in the immutable artifact manifest.
This preserves a stable committed lock and an auditable local consent trail.

### Hardware feasibility threshold

M0 freezes the qualification hardware identity or class, software stack, and model shape profile.
Projected final P95 latency and peak device memory must remain inside their absolute budgets with at least 20 percent contingency.
An estimate without those declared inputs and margins is not acceptance evidence.

### Host and target evidence

macOS arm64 is a CPU portability profile only.
Ubuntu 24.04 x86-64 remains the reference CPU and accelerated runtime target.
No macOS result supports a CUDA, TensorRT, Linux compiler, or release-performance claim.

## Consequences

No acceptance criterion is removed or weakened.
Some target-dependent milestones may have a pushed `IMPLEMENTED_LOCAL` commit before their later acceptance evidence commit.
Machine-local license consent no longer dirties a clean checkout.
Negative experiments remain truthful completed studies while the preceding accepted configuration remains the candidate.

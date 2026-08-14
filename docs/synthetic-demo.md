# Unrestricted synthetic demonstration

The primary public demonstration builds real graph evidence without licensed images or a GPU.
It generates 200 deterministic procedural intersection segments with 600 eligible lane-to-control edges.
The baseline and candidate contain identical lanes and traffic-control nodes.
Only the candidate's control-to-lane associations are rotated to the wrong lanes.

Run the complete workflow from the repository root:

```bash
./tools/jl bootstrap-cpu
./tools/jl build-cpu
./tools/jl demo-synthetic
./tools/jl inspect-demo --artifact-root artifacts/demo
```

The demo performs synthetic generation, custom evaluation, immutable arm registration, paired comparison, a standalone gate replay, the flagship fault-lab check, deterministic public report export, and real-browser inspection.
The browser screenshot is written to `artifacts/demo/browser-inspection.png`.
The offline evidence report is written to `artifacts/demo/public-report/REPORT.html`.

To inspect the generated registry interactively, run:

```bash
uv run --locked junctionlens serve \
  --artifact-root artifacts/demo \
  --open-browser
```

The lane-control cells reject the swapped candidate with `GATE_REGRESSION_CI_BELOW_MARGIN`.
The independent fault lab reports `FAULT_CONTROL_ASSIGNMENT_CHANGED` while its nearby clean control passes.

The overall release status remains `BLOCKED_INFRASTRUCTURE` on the CPU-only demo because accelerated runtime qualification has not run.
This is deliberate and prevents synthetic accuracy evidence from being presented as an accepted V1 model or performance result.
The demo is not licensed-data generalization evidence, a trained-model result, a vehicle safety case, or a final release decision.

Run the exact candidate-index clean-checkout gate with:

```bash
git add <candidate source files>
./tools/jl verify-m10-2
```

The gate exports the exact Git index into a fresh temporary checkout.
It uses checkout-local dependency caches, installs the locked browser, runs the complete local verification suite, recreates the demo, inspects it in Chromium, and rejects any unexplained source-tree change.

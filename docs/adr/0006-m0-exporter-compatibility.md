# ADR 0006: Qualify the M0 dynamo exporter boundary

## Status

Accepted for the M0 feasibility model.

## Context

The plan freezes PyTorch 2.8.0, ONNX 1.18.0, ONNX opset 18, and ONNX Runtime 1.25.0.
It does not freeze the separate ONNX Script and ONNX IR packages used by the PyTorch dynamo exporter.

Resolving ONNX Script 0.3.2 against the current ONNX IR 1.0.0 release failed inside the version-conversion pass after successful graph translation.
Pinning the release-era ONNX IR 0.1.4 removed that failure, but the older exporter stack froze leading dimensions in linear query heads.
The current ONNX Script 0.7.1 and ONNX IR 1.0.0 pair completes translation and preserves the dynamic query-head batch dimension after linear heads are expressed as flattened last-dimension operations.

PyTorch 2.8 still writes a static batch-one annotation for the two outputs that apply `softplus` after a dynamic reshape.
ONNX Runtime executes those tensors correctly at batch two but reconstructs the stale batch-one annotation while optimizing a session.

Direct PyTorch and ONNX Runtime FP32 execution also differed by several millionths for a few raw values just above magnitude `1e-3`.
Those values met the absolute parity limit but slightly exceeded the independent relative parity limit.

## Decision

Pin ONNX Script 0.7.1 and ONNX IR 1.0.0 exactly.
Keep PyTorch, ONNX, opset, and ONNX Runtime at the plan versions.

The exporter runs the ONNX checker and strict shape inference before correcting only the two known graph-output batch annotations.
The saved model declares the dynamic batch contract.
Acceptance additionally executes every output at batch two, so an annotation-only correction cannot hide a fixed-shape graph.
The native validator accepts ONNX Runtime's optimized-session batch-one report only for `lane_geometry_scales` and `area_geometry_scales` and still validates every other dimension and tensor exactly.

PyTorch exporter debug metadata contains absolute source paths and is not part of the runtime contract.
The exporter removes node stack traces, debug metadata properties, and documentation strings before hashing the model while retaining the JunctionLens model metadata.
The same checkpoint therefore produces the same model bytes from different checkout paths.

The M0 tensor boundary maps values with magnitude below `0.05` to canonical zero through a straight-through operation.
The deadband is part of the hashed M0 profile.
It keeps training gradients intact, removes numerically meaningless near-zero signs, and permits both the absolute and relative FP32 parity limits to gate independently.

## Evidence

- [PyTorch 2.8 ONNX exporter](https://docs.pytorch.org/docs/2.8/onnx.html)
- [ONNX Script 0.7.1](https://github.com/microsoft/onnxscript/releases/tag/v0.7.1)
- [ONNX IR 1.0.0](https://github.com/onnx/ir-py/releases/tag/v1.0.0)
- [ONNX Runtime 1.25.0 C++ API](https://onnxruntime.ai/docs/api/c/)

The integration gate runs strict checker and shape inference, a batch-two ONNX Runtime execution, every raw-output comparison through Python and native C++ ONNX Runtime, and seeded metadata corruption controls.

## Consequences

The M0 graph adds only standard opset-18 operators and no custom runtime dependency.
The deadband is limited to the feasibility profile and must be re-evaluated with the final model's coordinate and confidence distributions.
An exporter, ONNX Script, ONNX IR, or ONNX Runtime version change invalidates this qualification and requires the complete export and parity gate again.

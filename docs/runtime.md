# Native inference runtime

Milestone 8.1 provides a production C++20 CPU pipeline behind the public `junctionlens infer` command.
The pipeline decodes repository-relative image artifacts, materializes the frozen two-timestamp tensor profile, runs ONNX Runtime 1.25.0 through `CPUExecutionProvider`, applies deterministic graph postprocessing, and writes SceneControlGraph V1 protobuf envelopes.

## Build

Bootstrap the exact CPU tools before configuring the runtime.

```sh
./tools/jl bootstrap-cpu
./tools/jl configure-cpu
./tools/jl build-cpu
```

The build activates only the `core`, `imgproc`, and `imgcodecs` modules from the already locked OpenCV 4.11.0 source archive.
JPEG, PNG, and portable pixmap codec support is built from the pinned source tree without GUI, video, Python, or network features in the runtime.

## Batch input

The input list is UTF-8 text with one protobuf envelope path per line.
Blank lines and lines beginning with `#` are ignored.
Relative paths are resolved from the input list directory.
Each envelope must contain a valid `SensorFrame` whose `FrameKey` exactly matches the graph key.

Every valid camera image uses a repository-relative `ArtifactRef` URI under `--asset-root`.
The runtime resolves symlinks, rejects parent traversal, requires a regular file, and checks the declared byte size and SHA-256 before decoding pixels.
The offline path blocks on its fixed buffer pool and never drops a frame.

## Run

```sh
junctionlens infer \
  --model artifacts/m0/model-spike/model.onnx \
  --input-list artifacts/runtime/inputs.txt \
  --asset-root /path/to/registered/assets \
  --output-dir artifacts/runtime/predictions
```

The Python command records the current Git commit and dirty state, the canonical model-profile digest, the native binary digest, the model artifact digest, and the CPU provider-assignment digest.
It delegates inference to `junctionlens-runtime` without a shell and validates the versioned JSON receipt before returning success.

`--repeat-loads` performs bounded construction and destruction of independent ONNX Runtime sessions before the batch.
It exists to qualify clean repeated model load and unload behavior and accepts values from 1 through 100.
`--buffer-slots` controls the fixed offline pool and accepts values from 1 through 1024.

## Preprocessing contract

The runtime requires eight canonical camera slots at both timestamps.
It decodes valid images as 8-bit color, converts BGR decoder output to RGB, applies bilinear letterbox fit to 640 by 384, fills padding with pixel value zero, and normalizes with the frozen ImageNet mean and standard deviation.
Invalid cameras retain zero tensors and a false camera-valid mask.
Intrinsics are transformed by the exact realized resize and padding matrix.
The first frame in a segment duplicates the current timestamp and sets `temporal_valid` false.
Later strictly increasing frames use `inverse(T_world_current_vehicle) * T_world_previous_vehicle` and set `temporal_valid` only when both poses are valid.

## Postprocessing contract

Node and edge decisions use raw sigmoid probability greater than or equal to 0.5.
Predicted nodes sort by descending raw existence probability, the complete frozen quantized geometry key, and decoder query index.
Node IDs encode the concrete node type in the high eight bits and the type-local ordinal plus one in the low 56 bits.
Edge IDs use the same SHA-256 logical projection as the Python V1 contract.
Class distributions use stable softmax, road-area point masks retain all thresholded points with a deterministic two-point minimum, and every emitted uncertainty scale remains aligned with its geometry.

## Buffer ownership and failure behavior

Each acquired slot follows `FREE -> DECODING -> PREPROCESSING -> INFERENCE -> POSTPROCESSING -> SERIALIZING -> FREE`.
A lease owns exactly one slot and returns it to `FREE` during exception unwinding.
The runtime reports queue depth high-water and refuses clean shutdown unless every slot is free.

Output files are serialized to a sibling temporary file and atomically renamed only after a successful flush.
Existing outputs and stale temporary files are never overwritten.
Malformed protobuf, invalid tensor shape, nonfinite model output, digest mismatch, unsafe path, model-metadata mismatch, and output-contract failure return nonzero with a stable runtime reason code.

## Verification

```sh
./tools/jl verify-m8-1-local
```

The native tests cover SHA-256 vectors, every legal buffer transition, skipped-stage rejection, exception cleanup, real image decode, RGB normalization, timestamp duplication, and malformed dimensions.
The integration gate exercises `junctionlens infer` on a real two-frame protobuf batch, loads and unloads three independent sessions, validates every output through the Python contract, and compares it with an independent Python ONNX Runtime postprocessor.
The comparison requires exact field presence, exact node and edge counts, exact integer and enum values, exact deterministic IDs and binary decisions, and floating-point agreement within the frozen `1e-4` raw-output tolerance.
Seeded malformed-protobuf, corrupted-image, and attempted-overwrite cases must fail while their nearby control passes.

## Accelerated profile implementation state

Milestone 8.2 adds an isolated Linux x86-64 GPU build without changing the CPU artifact or its dependencies.
The GPU preset requires the locked CUDA 12.8 minor line and the exact ONNX Runtime 1.25.0 GPU release artifact.
The local macOS package is `IMPLEMENTED_LOCAL`, while CUDA and TensorRT results remain `DEFERRED_HARDWARE` until the `runtime-cuda` remote profile passes on the qualification target.

The mandatory `cuda` profile registers `CUDAExecutionProvider` followed by `CPUExecutionProvider`.
The runtime rejects the profile if any model node is assigned to CPU.
The conditional `tensorrt` profile registers TensorRT, CUDA, and CPU in that order, requires at least one TensorRT node, permits truthful partial CUDA coverage, and still rejects every CPU node.

Accelerated inference allocates input tensors through the ONNX Runtime CUDA allocator and binds every input and output with `Ort::IoBinding`.
Host-to-device and device-to-host copies use synchronous CUDA calls as explicit ownership boundaries.
The CUDA provider keeps `do_copy_in_default_stream` enabled, and the runtime does not introduce a custom asynchronous stream in V1.

`junctionlens-runtime doctor` reports the observed provider list, exact ONNX Runtime shared-library SHA-256, GPU identity, model hash, input and output names, provider node counts, provider-log hash, assignment digest, cache key, and I/O-binding state.
The provider parser accepts only the exact ONNX Runtime 1.25.0 log grammar at the `VerifyEachNodeIsAssignedToAnEp` boundary and requires a concrete shared-library hash.
Changing the runtime binary invalidates prior parser qualification evidence.

The TensorRT engine and timing cache directory is the SHA-256 of the model, provider options, GPU compute capability, TensorRT version, CUDA version, driver compatibility class, device ID, and fixed shape profile.
Existing symbolic-link cache roots are rejected, and `trt_force_timing_cache` remains disabled.

The local accelerated implementation gate is:

```sh
./tools/jl verify-m8-2-local
```

This command proves the exact parser fixtures, fallback rejection, partial TensorRT reporting, cache invalidation, secure source synchronization, public CPU control path, formatting, linting, type checking, native tests, and inherited correctness gates.
It does not claim CUDA correctness, performance, or TensorRT coverage on macOS.

# ADR 0003: Runtime compatibility pins

Status: Accepted

Date: 2026-08-13

## Context

Milestone 0 verified every reference version against current owning documentation and pinned upstream source.
Most reference pins are directly compatible.
Three details require explicit handling so later builds do not make broader claims than the evidence.

## Decision

ONNX 1.18.0 remains the application exporter and checker package.
ONNX Runtime 1.25.0 separately owns an internal ONNX 1.21.0 dependency in its source build.
The two roles are locked independently and are never described as one shared ONNX library.

The ONNX Runtime source archive is an identity and audit artifact, not a complete recursive source tree.
GPU and CPU source builds use the exact release commit and separately verify the pinned ONNX, libprotobuf-mutator, and Emscripten submodule commits before building.

Protobuf `protoc` 31.1 corresponds to C++ and Python runtime 6.31.1.
The C++ build and runtime require that exact release match.
The version check normalizes the compiler and runtime numbering schemes and rejects adjacent patch releases.

The mandatory CUDA execution-provider profile uses CUDA 12.8 and cuDNN 9.14.0.64.
Qualification requires an NVIDIA driver at least 570.26 so the project does not rely on ambiguous forward-compatibility behavior.

TensorRT 10.14.1.48 is officially packaged for CUDA 12.9.
ONNX Runtime 1.25.0 upstream CI deliberately tests that package in its CUDA 12.8 TensorRT build.
JunctionLens therefore treats this as a conditional, fail-closed, ORT-tested cross-minor combination rather than a native NVIDIA CUDA 12.8 TensorRT support claim.
If the provider probe fails, the optional TensorRT profile moves to its own CUDA 12.9 Update 1 image or is truthfully rejected without affecting mandatory CUDA acceptance.

## Evidence

- [ONNX Runtime 1.25.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.25.0)
- [ONNX Runtime CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
- [ONNX Runtime 1.25.0 TensorRT CI](https://github.com/microsoft/onnxruntime/blob/v1.25.0/.github/workflows/linux_tensorrt_ci.yml)
- [cuDNN 9.14 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.14.0/reference/support-matrix.html)
- [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
- [TensorRT 10.14.1 support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [Protobuf cross-version runtime guarantee](https://protobuf.dev/support/cross-version-runtime-guarantee/#cpp)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/#v280)

## Consequences

No reference version is changed.
Provider and runtime claims remain narrower than the verified compatibility evidence.
GPU builds must verify recursive source identity and cannot build from the bare ONNX Runtime archive alone.

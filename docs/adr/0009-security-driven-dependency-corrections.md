# ADR 0009: Correct obsolete dependency pins after the M10 security audit

## Status

Accepted on 2026-08-14.

## Context

The M0 compatibility study froze PyTorch 2.8.0, torchvision 0.23.0, ONNX 1.18.0, PyArrow 20.0.0, FastAPI 0.116.1, and the transitive Starlette 0.47.3 release.
Those pins were feasible when selected, but the M10 advisory scan found vulnerabilities published after that compatibility study.
The findings include a high-severity memory corruption and potential code execution path in the exact `torch.load(..., weights_only=True)` API used for JunctionLens checkpoints.
They also include high-severity denial-of-service paths in Starlette `FileResponse` and `StaticFiles`, both of which the local viewer uses.
ONNX 1.18.0 has a critical hub trust bypass and multiple model parsing and external-data findings.
PyArrow 20.0.0 is affected by a high-severity Arrow C++ use-after-free, although the vulnerable pre-buffered IPC API is not exposed through Python.

## Decision

Replace PyTorch 2.8.0 and torchvision 0.23.0 with the official compatible PyTorch 2.11.0 and torchvision 0.26.0 pair.
The official wheel matrix retains CPU and CUDA 12.8 variants, so this change preserves the target platform objective.
Replace ONNX 1.18.0 with ONNX 1.22.0 while retaining opset 18 export and ONNX Runtime 1.25.0 as an independently versioned runtime.
Replace PyArrow 20.0.0 with the first fixed 23.0.1 release.
Replace FastAPI 0.116.1 with 0.141.1 and pin Starlette 1.3.1 directly so all known Starlette findings are fixed.

Keep the exact protobuf 6.31.1 release boundary temporarily.
PYSEC-2026-1805 affects only `google.protobuf.json_format.ParseDict` with nested `Any` messages.
JunctionLens does not call that API or use `Any` and instead uses generated binary messages plus separately bounded strict JSON parsing.
The time-limited, machine-checked exception is recorded in `configs/security/advisory-exceptions.yaml` and expires on 2026-11-14.
The complete protoc, generated code, Python runtime, C++ headers, and libprotobuf pin must move together when that exception is reassessed.
Pip-audit also reports PYSEC-2025-194 against PyTorch 2.11.0 even though the current OSV record declares 2.6.0-NA as the last affected release and identifies only `torch.jit.script`, which JunctionLens does not call.
That upstream metadata inconsistency is retained as a separate time-limited medium-severity exception rather than omitted from evidence.
PyTorch 2.11 also constrains its transitive setuptools dependency below the release that fixes PYSEC-2026-3447.
That medium-severity finding affects only setuptools source-distribution file selection, while JunctionLens builds with Hatchling and never invokes the affected setuptools API.
It is retained as a third time-limited exception.

## Evidence

- [PyTorch 2.11.0 and torchvision 0.26.0 installation matrix](https://pytorch.org/get-started/previous-versions/#v2110)
- [PyTorch weights-only unpickler advisory](https://osv.dev/vulnerability/PYSEC-2026-2286)
- [ONNX hub trust bypass advisory](https://osv.dev/vulnerability/PYSEC-2026-103)
- [PyArrow use-after-free advisory](https://osv.dev/vulnerability/PYSEC-2026-113)
- [Starlette range-header denial-of-service advisory](https://osv.dev/vulnerability/PYSEC-2026-1942)
- [Protobuf recursion-depth bypass advisory](https://osv.dev/vulnerability/PYSEC-2026-1805)

## Consequences

All export, parity, model, service, report, and browser gates must run again before acceptance.
Earlier M0 performance numbers do not transfer to PyTorch 2.11.0 and cannot be presented as results for the corrected dependency set.
GPU performance and provider qualification remain target-only gates and must be rerun on the NVIDIA machine.

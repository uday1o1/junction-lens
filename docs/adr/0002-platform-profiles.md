# ADR 0002: Reference and portability platform profiles

Status: Accepted

Date: 2026-08-13

## Context

The reference stack targets Ubuntu 24.04 LTS on x86-64 with GCC 13 and Clang 18.
Initial implementation is running on a macOS arm64 host with Apple Clang and no NVIDIA device.

## Decision

The `linux-x86_64-reference` profile owns release compiler, sanitizer, official evaluator container, CUDA, TensorRT, and performance evidence.
The `macos-arm64-portability` profile owns native CPU development, schema, data-contract, model smoke, evaluator primitives, CLI, service, browser, and reproducibility checks that are platform-independent.
Portable checks may pass on macOS without being relabeled as Linux evidence.
Linux-only and accelerated gates remain target-only until the exact reference or qualification environment accepts them.

The doctor reports observed compiler, architecture, runtime, providers, data state, and target readiness independently.
It never reports a configured reference pin as an observed capability.

## Consequences

The project can finish meaningful local work without a local NVIDIA GPU.
The final definition of done still requires accepted Ubuntu x86-64 CPU and NVIDIA accelerated evidence.

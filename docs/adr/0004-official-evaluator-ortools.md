# ADR 0004: Use the nearest available OR-Tools evaluator dependency

## Status

Accepted for the OpenLane-V2 v2.1.0 compatibility image.

## Context

The pinned OpenLane-V2 v2.1.0 `setup.py` requests OR-Tools 9.2.9972.
The public PyPI simple index no longer publishes that release, so current resolvers cannot produce a reproducible installation from the owning public package index.
The archived PyPI project record proves that Python 3.8 Linux x86-64 wheels once existed, but using an unaffiliated deleted-package mirror would weaken the supply-chain contract.

The lane-segment evaluator imports the centerline evaluator package, which imports OR-Tools during module initialization.
The lane-segment metric implementation does not otherwise call OR-Tools.

## Decision

Keep the plan's digest-pinned CPython 3.8 Linux x86-64 image and untouched OpenLane-V2 v2.1.0 evaluator source.
Resolve OR-Tools 9.3.10497, the immediately following public release with a Python 3.8 Linux x86-64 wheel, instead of the deleted 9.2.9972 package.
Lock every resolved wheel hash.

The non-headless OpenCV wheel imported transitively by the untouched package requires `libgl1`, `libglib2.0-0`, and `libxcb1` in the slim base image.
Install those packages at exact versions from the immutable Debian snapshot dated 2024-09-26 that is recorded in the pinned base image's own source configuration.

The official-evaluator image is accepted only after perfect, corrupted, and order-permutation fixtures prove the expected metric behavior.
The compatibility correction is recorded in the evaluator environment report and must not be described as an exact upstream dependency environment.

## Evidence

- OpenLane-V2 v2.1.0 dependency declaration: <https://github.com/OpenDriveLab/OpenLane-V2/blob/v2.1.0/setup.py>
- Current OR-Tools project index: <https://pypi.org/simple/ortools/>
- Archived OR-Tools 9.2.9972 release metadata: <https://pypi.org/project/ortools/9.2.9972/>
- Available OR-Tools 9.3.10497 release: <https://pypi.org/project/ortools/9.3.10497/>
- Debian snapshot service: <https://snapshot.debian.org/>

## Consequences

The evaluator remains on Python 3.8 and the official metric code stays byte-for-byte identical to the pinned v2.1.0 source.
The environment has one explicit dependency deviation and cannot claim exact reproduction of the historical upstream installation.
Any fixture mismatch blocks official metric ownership until the deviation is replaced or explained by stronger evidence.

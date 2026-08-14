# ADR 0004: Bound official-evaluator compatibility substitutions

## Status

Accepted for the OpenLane-V2 v2.1.0 compatibility image.

## Context

The pinned OpenLane-V2 v2.1.0 `setup.py` requests OR-Tools 9.2.9972.
The public PyPI simple index no longer publishes that release, so current resolvers cannot produce a reproducible installation from the owning public package index.
The archived PyPI project record proves that Python 3.8 Linux x86-64 wheels once existed, but using an unaffiliated deleted-package mirror would weaken the supply-chain contract.

The lane-segment evaluator imports the centerline evaluator package, which imports OR-Tools during module initialization.
The lane-segment metric implementation does not otherwise call OR-Tools.

The evaluator also imports upstream I/O code that imports `cv2`, even though trusted dictionary input never calls image or GUI functions.
The original desktop OpenCV wheel requires a large X11 and Mesa runtime chain in the slim image.
The previously pinned Debian snapshot installation reproduced those system libraries until its APT signature path became invalid at the 2026-08-14 UTC qualification boundary.
Direct `gpgv` verification of the archived release bytes still succeeded, but APT rejected all three repositories, so the build path was no longer reproducible and could not remain accepted.

## Decision

Keep the plan's digest-pinned CPython 3.8 Linux x86-64 image and untouched OpenLane-V2 v2.1.0 evaluator source.
Resolve OR-Tools 9.3.10497, the immediately following public release with a Python 3.8 Linux x86-64 wheel, instead of the deleted 9.2.9972 package.
Lock every resolved wheel hash.

Use `opencv-python-headless` 5.0.0.93 in place of `opencv-python` 5.0.0.93.
Both distributions provide the `cv2` import, while the headless flavor intentionally omits GUI functionality and its X11 dependency chain.
The evaluator does not call `cv2.imshow` or any other GUI API.
Install no additional Debian packages in the compatibility image.

The official-evaluator image is accepted only after perfect, corrupted, and order-permutation fixtures prove the expected metric behavior.
The compatibility correction is recorded in the evaluator environment report and must not be described as an exact upstream dependency environment.

## Evidence

- OpenLane-V2 v2.1.0 dependency declaration: <https://github.com/OpenDriveLab/OpenLane-V2/blob/v2.1.0/setup.py>
- Current OR-Tools project index: <https://pypi.org/simple/ortools/>
- Archived OR-Tools 9.2.9972 release metadata: <https://pypi.org/project/ortools/9.2.9972/>
- Available OR-Tools 9.3.10497 release: <https://pypi.org/project/ortools/9.3.10497/>
- OpenCV headless package guidance: <https://pypi.org/project/opencv-python-headless/>
- Debian snapshot usage and `Valid-Until` guidance: <https://snapshot.debian.org/>

## Consequences

The evaluator remains on Python 3.8 and the official metric code stays byte-for-byte identical to the pinned v2.1.0 source.
The environment has two explicit dependency substitutions and cannot claim exact reproduction of the historical upstream installation.
Removing the APT step makes the operating-system filesystem entirely owned by the pinned base-image digest.
Any fixture mismatch blocks official metric ownership until the deviation is replaced or explained by stronger evidence.

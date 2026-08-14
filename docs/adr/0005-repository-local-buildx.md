# ADR 0005: Pin a repository-local OCI builder

## Status

Accepted for reproducible OCI image exports.

## Context

The detected Docker Engine exposes only the legacy builder because its optional Buildx CLI plugin is absent.
The legacy builder cannot export an OCI layout and embeds unstable build metadata, so it cannot satisfy the image-digest lock or clean-build reproducibility gates.
Installing an unpinned host-global plugin would make developer and CI behavior diverge.

## Decision

Lock Docker Buildx 0.34.1 for macOS arm64 and Linux x86-64 in the repository toolchain lock.
The bootstrap installs the verified binary below ignored `.tools` state.
Container automation exposes it through a machine-local Docker configuration and never modifies the user's global Docker configuration.

OCI builds use `SOURCE_DATE_EPOCH`, the OCI exporter, OCI media types, and timestamp rewriting.
The manifest and config digests are read from the exported OCI index and frozen in `containers/images.lock` after two independent builds agree.

## Evidence

- Docker OCI exporter: <https://docs.docker.com/build/exporters/oci-docker/>
- Docker reproducible builds: <https://docs.docker.com/build/ci/github-actions/reproducible-builds/>
- Docker Buildx v0.34.1: <https://github.com/docker/buildx/releases/tag/v0.34.1>
- Docker Desktop Buildx v0.34.1: <https://github.com/docker/buildx-desktop/releases/tag/v0.34.1-desktop.1>

## Consequences

The builder becomes an exact repository input rather than an assumed workstation capability.
The local Docker daemon remains optional for CPU-only work, but evaluator image qualification fails closed when it is absent.

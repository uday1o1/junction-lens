# ADR 0005: Pin a repository-local OCI builder

## Status

Accepted for reproducible OCI image exports.

## Context

The detected Docker Engine exposes only the legacy builder because its optional Buildx CLI plugin is absent.
The legacy builder cannot export an OCI layout and embeds unstable build metadata, so it cannot satisfy the image-digest lock or clean-build reproducibility gates.
Installing an unpinned host-global plugin would make developer and clean-checkout behavior diverge.

## Decision

Lock Docker Buildx 0.34.1 for macOS arm64 and Linux x86-64 in the repository toolchain lock.
Lock the multi-platform BuildKit 0.30.0 daemon image by OCI index digest in `containers/images.lock`.
The bootstrap installs the verified Buildx binary below ignored `.tools` state.
Each independent evaluator build uses a fresh `docker-container` builder backed by the locked BuildKit image and repository-scoped temporary Buildx configuration.
The automation verifies the running builder's version, driver, and exact configured image before it accepts an export.
It removes the temporary builder after each export and never modifies the user's global Docker configuration.

OCI builds use `SOURCE_DATE_EPOCH`, the OCI exporter, OCI media types, and timestamp rewriting.
The manifest and config digests are read from the exported OCI index and frozen in `containers/images.lock` after builds from two independent fresh builders agree.

## Evidence

- Docker OCI exporter: <https://docs.docker.com/build/exporters/oci-docker/>
- Docker reproducible builds: <https://docs.docker.com/build/ci/github-actions/reproducible-builds/>
- Docker Buildx v0.34.1: <https://github.com/docker/buildx/releases/tag/v0.34.1>
- Docker Desktop Buildx v0.34.1: <https://github.com/docker/buildx-desktop/releases/tag/v0.34.1-desktop.1>
- Moby BuildKit v0.30.0: <https://github.com/moby/buildkit/releases/tag/v0.30.0>

## Consequences

Both the Buildx client and BuildKit daemon become exact repository inputs rather than assumed workstation capabilities.
Daemon-local cache history cannot affect the locked evaluator identity because each comparison build starts with a fresh builder state.
The local Docker daemon remains optional for CPU-only work, but evaluator image qualification fails closed when it is absent.

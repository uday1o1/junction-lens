# Data manifests and V1 splits

JunctionLens freezes data identity before model results can influence partition membership.
The full-profile workflow streams one bounded provenance record per frame into the ignored local artifact store and retains no annotation geometry, image pixels, or image paths in that frame-manifest payload.
Each record contains source identity hashes, frame metadata and calibration hashes, camera availability, pose validity, and annotation availability.

## Immutable storage

The local registry stores payloads and their validated artifact manifests under `objects/sha256` paths.
Writes use a same-filesystem temporary file, file synchronization, an exclusive hard link, and directory synchronization.
An existing hash path is verified and never overwritten.
Artifact manifests conform to `schemas/artifact-manifest-v1.schema.json` and contain no wall-clock field, so identical content and provenance reproduce the same hash.

The artifact root defaults to `artifacts` and may be changed with `--artifact-root`.
That root is ignored because full-profile frame records derive from restricted dataset metadata.

## Frozen V1 policy

The policy is `configs/data/openlane-v2-v2.1.split-v1.yaml`.
It permits only complete `segment_id` and authoritative `source_domain` metadata to influence assignment.
Annotations, images, label statistics, model outputs, and statistics computed inside a future partition are explicitly forbidden assignment inputs.

The algorithm sorts source domains lexically, apportions each domain across the remaining partition capacities with deterministic Hamilton largest remainders, and orders records within each domain by a domain-separated SHA-256 score.
The result contains exactly 350 model-training segments, 80 model-selection segments, 70 calibration segments, and 200 frozen internal-holdout segments.
Every ordered list, the complete segment catalog, the source dataset manifest, and the source frame artifact are hash-bound.

The official validation split remains an external benchmark.
The official test split remains disabled without a valid server result.
Subset B is not eligible for the subset-A learning partitions.

## Licensed full-profile workflow

First create a streaming frame manifest from the checksum-registered full profile:

```sh
uv run --locked junctionlens data manifest \
  --profile full \
  --artifact-root artifacts
```

Use the returned artifact-manifest hash to freeze and export the committed V1 split:

```sh
uv run --locked junctionlens data freeze-splits \
  --frame-manifest-sha256 FRAME_MANIFEST_SHA256 \
  --artifact-root artifacts \
  --export configs/data/openlane-v2-v2.1.split-v1.json
```

Audit the exported file independently:

```sh
uv run --locked junctionlens data audit-splits \
  --manifest configs/data/openlane-v2-v2.1.split-v1.json
```

An existing export accepts an identical rerun and rejects different bytes.
The dataset lock keeps `committed_manifest_sha256` unset until the licensed 700-segment source gate runs.
The split file and its hash must be added to the lock in the same qualification commit.

## Local package gate

Run the hardware-independent implementation gate with:

```sh
./tools/jl verify-m2-2-local
```

This gate exercises a repository-owned 700-segment catalog through the public freeze and audit commands.
It proves exact allocations, stable hashes across reruns, zero overlap, immutable registry behavior, and a deliberate hash-consistent leakage failure.
It does not substitute synthetic identifiers for the final licensed V1 manifest.

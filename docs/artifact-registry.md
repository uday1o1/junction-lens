# Immutable evidence registry

JunctionLens stores evidence below the local ignored artifact root selected by `JUNCTIONLENS_ARTIFACT_ROOT`, with `./artifacts` as the development default.
Immutable payloads and manifests remain the provenance authority, while DuckDB is a rebuildable query index.

## Object and manifest identity

Every payload is SHA-256 hashed and stored at `objects/sha256/<first-two>/<remaining-hex>`.
Its canonical JSON manifest records the artifact kind, byte size, media type, license, metadata, and sorted parent manifest hashes.
The manifest is schema-validated and stored through the same content-addressed path.
An existing object is rehashed and size-checked instead of being overwritten.
Files are copied through same-filesystem temporary files, flushed, published atomically, made read-only, and followed by a directory synchronization.

Use the public command to store and index an artifact:

```console
junctionlens registry put \
  --input artifacts/private/metrics.parquet \
  --kind segment_kpi_table \
  --media-type application/vnd.apache.parquet \
  --license-id CC-BY-NC-SA-4.0 \
  --metadata artifacts/private/metrics-metadata.json \
  --parent "$MATCH_MANIFEST_SHA256" \
  --artifact-root artifacts
```

Repeating the command with byte-identical content and provenance returns the same payload and manifest identities.

## DuckDB index and locking

The mutable DuckDB database indexes artifact metadata, parent edges, aliases, and operational run state.
It never contains the sole copy of artifact provenance.
The `artifact_summary` and `provenance_edges` views support read-only queries, and transitive provenance is returned in deterministic depth and hash order.

One advisory writer lock protects every DuckDB mutation.
The durable owner record contains a random lock identifier, PID, redacted host fingerprint, creation time, and heartbeat.
A live same-host process blocks a second writer.
A different host also blocks recovery because its process state cannot be verified locally.
Only a proven absent same-host PID may be reclaimed, and each recovery is appended to the synchronized recovery event log.
Query-only registry methods do not acquire the writer lock and can observe committed artifacts concurrently within the owning service process through DuckDB MVCC.
DuckDB does not support mixed read-write access from multiple processes, so a second process may connect only between writer sessions or in an all-read-only deployment.

Inspect one artifact or its complete ancestry with:

```console
junctionlens registry inspect --manifest "$MANIFEST_SHA256" --artifact-root artifacts
junctionlens registry provenance --manifest "$MANIFEST_SHA256" --artifact-root artifacts
```

Human-readable aliases are mutable convenience pointers and are explicitly labeled as non-evidence.
Use `registry set-alias` and `registry resolve-alias` for navigation, but use the returned manifest SHA-256 in every evidence record.

## Exact run resume

A run ID is the SHA-256 of canonical JSON containing the run kind, parent artifacts, dataset and split manifests, model and command configuration, source commit and dirty state, dependency locks, image digests, seed, command schema, and execution-provider profile.
The first resume command stores that identity as an immutable `run_configuration` artifact.
A later invocation returns `RESUMED` only when the full run identity and environment compatibility fingerprint match.
A changed environment fails closed and requires a new compatible run rather than silently loading old state.

```console
junctionlens registry resume \
  --identity artifacts/private/run-identity.json \
  --environment-fingerprint "$ENVIRONMENT_SHA256" \
  --artifact-root artifacts
```

## Recovery and garbage-collection audit

The index can be reconstructed from verified immutable manifests when mutable DuckDB rows are missing.

```console
junctionlens registry rebuild-index --artifact-root artifacts
```

V1 garbage collection is deliberately audit-only.
The command reports unindexed objects, interrupted staging files, and reclaimable bytes without deleting any data.

```console
junctionlens registry gc --dry-run --artifact-root artifacts
```

Omitting `--dry-run` is rejected.
Crash-injection tests prove that a failure after object publication or before the DuckDB commit leaves immutable bytes available for a byte-identical retry and visible to the dry-run audit.

Run the complete local package gate with `./tools/jl verify-m7-1-local`.

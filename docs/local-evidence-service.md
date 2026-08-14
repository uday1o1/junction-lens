# Local evidence service

JunctionLens exposes registered evidence through a read-only local API.
The service does not train models, run inference, recalculate release decisions, or accept artifact mutations.

## Start or validate the service

Create or populate a registry through the documented evaluation and comparison commands first.
Validate the root without opening a socket:

```bash
uv run --locked junctionlens serve \
  --artifact-root artifacts/demo \
  --check
```

Start the service:

```bash
uv run --locked junctionlens serve \
  --artifact-root artifacts/demo
```

The default service validates and serves the production client from `web/dist`.
Run `pnpm run build:web` before starting it.
Pass `--api-only` only when a browser client is intentionally unnecessary.
The V1 service binds only to `127.0.0.1`.
Any other `--host` value is rejected before a socket is opened.
Use `--open-browser` to open the local product after startup.

## API contract

The API prefix is `/api/v1`.
List routes use stable `offset` and `limit` pagination, with a maximum page size of 100.
Every returned artifact is cross-checked against its immutable manifest and content hash.

| Route | Purpose |
| --- | --- |
| `GET /api/v1/health` | Report registry readability and object counts. |
| `GET /api/v1/runs` | List persisted run identities and terminal states. |
| `GET /api/v1/artifacts` | List registered immutable artifacts, optionally filtered by kind. |
| `GET /api/v1/artifacts/{manifest}` | Return one verified artifact manifest view. |
| `GET /api/v1/artifacts/{manifest}/content` | Download one bounded verified payload. |
| `GET /api/v1/metrics/{manifest}` | Page through one registered Parquet metric table. |
| `GET /api/v1/decisions` | List persisted release-decision artifacts. |
| `GET /api/v1/decisions/{manifest}` | Return the exact persisted decision body after identity verification. |
| `GET /api/v1/images/{manifest}` | Proxy one bounded registered PNG, JPEG, or WebP artifact. |
| `GET /api/v1/scenes` | List registered immutable scene bundles. |
| `GET /api/v1/scenes/{manifest}` | Return one strict scene bundle with its verified persisted decision. |

All mutation methods return a stable read-only error.
Unknown routes, missing artifacts, invalid parameters, unsafe registry state, and unsupported image types use the versioned `junctionlens.api-error.v1` envelope.

## Artifact-root containment

The selected artifact root must be an existing real directory with a readable registry index.
A symlink supplied as the root is rejected.
Payload paths are derived from validated lowercase SHA-256 identities rather than user-provided path fragments.
Object directory and payload symlinks are rejected by the registry verifier.
Response file descriptors are opened without following a final symlink, hashed before use, and retained until response consumption.
General artifact, image, and metric-table responses have separate byte limits.

## Persisted decision serving

The browser and API consume the registered `release_decision` payload.
The service verifies the decision's own `decision_sha256` before returning it.
Changing a client-side filter cannot change the served release status.

Create an immutable JSON report snapshot from a decision with:

```bash
uv run --locked junctionlens report \
  --decision <decision-manifest-sha256> \
  --artifact-root artifacts/demo
```

The command prints the report manifest hash, payload hash, and immutable relative path.

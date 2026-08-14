# Reproducible evidence reports

JunctionLens exports an immutable comparison as a self-contained offline evidence bundle.
The exporter reads the persisted release decision and registered Parquet parents without recalculating release status or changing table rows.

## Public export

Pass the comparison report-data manifest printed by `junctionlens compare`.
The output directory must not already exist.

```sh
uv run --locked junctionlens report \
  --comparison <comparison-report-data-manifest-sha256> \
  --artifact-root artifacts/demo \
  --output-dir artifacts/demo/public-report
```

The command prints the registered ZIP manifest hash, ZIP payload hash, immutable object path, bundle manifest hash, and the SHA-256 of each exported file.
The public mode excludes camera image payloads and private filesystem paths.
An optional `--scene` adds redacted counterexample metadata while continuing to omit every camera artifact identity and image byte.

## Bundle contract

Each bundle contains these required files:

- `manifest.json` records the immutable inputs, privacy mode, status, license boundary, file hashes, and byte sizes.
- `decision.json` is the exact verified persisted decision payload.
- `metrics.parquet` and `slices.parquet` are exact copies of the registered comparison tables.
- `counterexamples.json` records either an explicit empty state, a public redacted scene summary, or an acknowledged private scene.
- `commands.jsonl` contains canonical argument arrays with path placeholders rather than machine-local paths.
- `environment.json` records bounded platform, package, and repository-lock identities without reading arbitrary environment variables.
- `REPORT.json` is the validated machine-readable report.
- `REPORT.md` is the portable text report.
- `REPORT.html` is the responsive offline report with inline first-party styling and no script, font, analytics, CDN, or network dependency.
- `SHA256SUMS` covers every report file other than itself and the outer ZIP.

The deterministic stored ZIP uses sorted members, fixed permissions, a fixed ZIP epoch, and no compressor-dependent output.
Running the same export under the same locked environment produces the same file hashes, ZIP payload hash, and registered artifact manifest.

## Private export

Private mode can include registered camera thumbnails from a counterexample scene.
It requires an explicit license acknowledgment and an output directory beneath the selected artifact root.

```sh
uv run --locked junctionlens report \
  --comparison <comparison-report-data-manifest-sha256> \
  --scene <counterexample-scene-manifest-sha256> \
  --mode private \
  --acknowledge-private-license \
  --artifact-root artifacts/demo \
  --output-dir artifacts/demo/private-report
```

Each included image is copied from a verified registered object and labeled with its source manifest, camera slot, frame, and license identifier.
The HTML uses only bundle-relative image paths.
Private mode never serializes an absolute source path or signed URL.

Do not publish, commit, or redistribute a private bundle unless the source dataset terms allow it and the bundle has received the required review.

## Offline review and verification

Open `REPORT.html` directly from the output directory after disconnecting from the network.
No local JunctionLens service is required for this report path.

Verify every listed payload with:

```sh
cd artifacts/demo/public-report
shasum -a 256 -c SHA256SUMS
```

The registered outer ZIP is `junctionlens-evidence-bundle.zip` in the materialized directory.
Its immutable registry copy is the path printed by the report command.

## Decision-only compatibility snapshot

The earlier decision-only command remains available for API compatibility:

```sh
uv run --locked junctionlens report \
  --decision <decision-manifest-sha256> \
  --artifact-root artifacts/demo
```

That command registers one canonical JSON snapshot.
It is not a complete reproduction bundle and should not be used in place of `--comparison` for release evidence.

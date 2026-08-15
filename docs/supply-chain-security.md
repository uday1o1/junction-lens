# Supply-chain security

JunctionLens derives local security evidence from the committed Python and pnpm lockfiles, the native component inventory, and the repository working set.
Generated evidence is written below ignored `artifacts/security/` and is not source-controlled.

## Stable commands

Run both advisory ecosystems and write the machine-readable result:

```sh
./tools/jl scan-dependencies
```

Generate a deterministic CycloneDX 1.7 SBOM covering Python, Node, and native dependencies:

```sh
./tools/jl generate-sbom
```

Run the complete security gate:

```sh
./tools/jl test-security
```

Run repeatability checks for the SBOM, source bundle, and offline report:

```sh
./tools/jl test-reproducibility
```

`scan-dependencies` exports exact hash-locked runtime requirements without resolving a second environment, queries pip-audit and pnpm, and blocks unresolved high or critical findings.
Network, registry, malformed-report, and timeout failures fail the command.

The secret scan examines Git-tracked files and unignored prospective tracked files.
It skips binary and oversized files, reports only paths and rule names, and never prints matched credential text.

The license inventory covers synchronized Python distributions, pnpm-lock packages present in the synchronized node modules tree, and native components declared in `configs/security/native-components.yaml`.
Unknown, AGPL, SSPL, and GPL license results fail closed until the component metadata or policy is corrected.

## Advisory exceptions

`configs/security/advisory-exceptions.yaml` is the only supported exception mechanism.
An exception must be exact, time-limited, substantive, and tied to repository controls.
The dependency report retains every active exception so a passing result cannot hide the accepted residual risk.
The complete dependency correction rationale is recorded in ADR 0009.

## Parser and service controls

JSON and YAML inputs have byte, depth, node, container, and string bounds and reject duplicate keys.
JSON additionally rejects non-finite numbers, and YAML rejects aliases before construction.
Registered Parquet responses have byte, row, column, nested-value, and container-item limits.
Image responses verify the actual decoded format, dimensions, pixels, and byte count before serving content.
Archive extraction preflights member count, member size, total size, compression ratio, type, traversal, and symlink containment before creating a destination.
The Protobuf validation boundary limits serialized bytes, reflected message depth, repeated values, strings, and total visited fields.

The C++ parser mutation smoke performs 5,000 deterministic mutations under ASan and UBSan through `./tools/jl test-fuzz`.
An optional coverage-guided `junctionlens-contract-fuzz` target remains available with `JUNCTIONLENS_ENABLE_LIBFUZZER=ON` on Clang installations that ship the libFuzzer runtime.

The loopback API uses a restrictive content security policy, frame denial, no-sniff, no-referrer, permissions, opener, resource, and no-store controls.
Artifact reads originate from verified registry identities and no endpoint accepts an arbitrary path.
The service disables access logs, returns generic internal errors, and redacts local roots, URL credentials, authorization values, signed URL queries, and secret-like assignments from configuration diagnostics.

## Local verification contract

The repository-owned security commands synchronize exact Python and Node dependency locks, run parser and seeded-control tests, inventory licenses, scan secrets, query both advisory services, and generate the deterministic SBOM.
The commands do not upload generated evidence or restricted artifacts.

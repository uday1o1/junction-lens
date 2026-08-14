# Security policy

## Supported version

Security fixes are applied to the current `main` branch during V1 development.
No released version is currently supported.

## Reporting a vulnerability

Do not include vulnerability details, credentials, signed URLs, restricted dataset content, or private artifact paths in a public issue.
Use the private security-advisory interface for this GitHub repository when it is available.
If that interface is unavailable, contact the repository owner privately through GitHub before sharing technical details.

Include the affected commit, the smallest safe reproduction, the expected impact, and whether restricted data or a GPU target is required.
Remove secrets and user-specific paths from logs before attaching them.

## Product boundary

JunctionLens is a local research and evidence tool, not a vehicle controller, safety case, or certification product.
The V1 service accepts only `127.0.0.1`, has no authentication or remote-serving mode, and exposes read-only registered artifacts.
Treat a failure to contain artifact reads, a bypass of registered content identities, malformed-input memory corruption, credential disclosure, or report-script injection as a security issue.

Licensed OpenLane-V2 data remains outside the repository under a user-selected root.
Do not attach licensed frames, private thumbnails, model weights derived from restricted data, or evaluator outputs containing restricted content to a report.

## Local verification

Run the complete local security gate with:

```sh
./tools/jl test-security
```

Run the deterministic evidence gate with:

```sh
./tools/jl test-reproducibility
```

These commands fail closed on parser-boundary tests, sanitizer mutation smoke, prospective secret findings, unacceptable licenses, expired advisory exceptions, high or critical dependency findings, and nondeterministic SBOM output.
The dependency scan requires access to the owning package advisory services, and an infrastructure or timeout failure is not a passing result.

## Vulnerability exceptions

Time-limited exceptions live in `configs/security/advisory-exceptions.yaml`.
Each exception must identify one exact package and advisory, state a substantive non-reachability assessment, name executable or source controls, and have an unexpired review date.
The security gate rejects expired, duplicate, incomplete, or uncontrolled exceptions.

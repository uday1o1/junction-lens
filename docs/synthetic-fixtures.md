# Synthetic truth corpus

The V1 synthetic corpus is unrestricted repository-owned test data for graph, projection, temporal, evaluator, and product workflows.
It contains straight controlled roads, a merge, a split, an intersection, a crosswalk, two temporal frames, perfect predictions, and three single-fault corruptions.

Every frame contains all eight canonical camera slots with exact intrinsic and rigid extrinsic calibration.
Repository-owned SVG source images are rendered from the ground-truth graph through those calibrations.
The persisted frame geometry is expressed in the current vehicle frame, while `T_world_vehicle` records the declared two-meter temporal ego motion.

The frozen seed is `20260813`.
The committed manifest records every relative path, byte size, media type, variant, and SHA-256 digest.
Generation uses deterministic Protobuf serialization and sorted JSON, and verification regenerates every byte rather than trusting the manifest alone.

Generate an ignored working copy with:

```bash
./tools/jl generate-synthetic
```

Verify the committed corpus with:

```bash
./tools/jl verify-synthetic
```

The generator refuses unexplained stale files, symlink traversal, non-regular corpus roots, and byte drift.
The controlled corruptions are `drop-control`, `break-topology`, and `shift-lane`.
Each corruption remains schema-valid so later evaluator tests distinguish semantic failure from malformed input.

# Visual and statistical data audit

The V1 data audit combines private image-level inspection with aggregate-only statistical evidence.
It is a data-quality workflow, not a claim that a model is safe or production-ready.

## Private visual bundle

`junctionlens data visual-audit` renders every available camera for the frozen selector in `configs/data/openlane-v2-v2.1.audit-v1.yaml`.
Camera PNGs retain licensed source pixels and add green lane centerlines, blue lane boundaries, yellow areas, red traffic-control boxes, and a cyan intrinsic principal-point marker.
The canonical BEV SVG uses the frozen negative-20-to-80-meter forward range and negative-40-to-40-meter lateral range.
It renders lane and area geometry, positive lane-lane edges, and positive lane-control support markers.

The generated bundle is private by default and lives under the ignored `artifacts` root.
Its index and file manifest mark source-image overlays and derived label geometry as private.
Do not commit or publish those files.

Run the registered sample workflow with:

```sh
uv run --locked junctionlens data visual-audit \
  --profile sample \
  --output artifacts/data-audit/openlane-v2-v2.1-sample
```

The command emits `PENDING_HUMAN_INSPECTION` even when automated checks pass.
A reviewer must open the generated `index.html`, inspect camera projection alignment and BEV orientation, and verify that source boxes, boundaries, areas, and topology markers agree with the selected frames.
An infrastructure failure or an uninspected bundle is not accepted evidence.

## Aggregate report

`summary.json` contains no pixels, image paths, object IDs, or annotation coordinates.
It records complete class distributions, capacity histograms, camera-availability patterns, geometry extrema, hard-range violations, canonical-BEV coverage, and positive topology support.
The audit streams frames and decodes at most one front-camera image at a time for the explicitly named low-luminance proxy.
That proxy is never labeled night.

Slice provenance is frozen in `configs/slices/v1.yaml`.
The preview includes source domain, connector presence, merge or split topology, graph degree, curvature, traffic-control pixel area, long range, crosswalk presence, attribute group, low luminance, camera availability, and lane-count complexity.
Segment counts remain separate from frame counts so future statistical gates can use `segment_id` as the independent unit.

## Automated gate

Run the hardware-independent package with:

```sh
./tools/jl verify-m2-3-local
```

The gate covers a 0.25-pixel analytic projection golden, rendering contracts, exact slice definitions, complete aggregate fields, hard geometry ranges, an extreme seeded range defect, and the real public CLI bundle path.
The repository-owned fixture bundle supports unrestricted implementation inspection.
Milestone acceptance still requires manual inspection of the frozen licensed selector.

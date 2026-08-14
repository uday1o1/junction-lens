# Scene comparison viewer

The JunctionLens scene viewer is a thin read-only interface for inspecting graph evidence that already exists in the artifact registry.
It never recalculates release status or changes registered artifacts.

## Build and open the viewer

Build the pinned offline web application:

```bash
pnpm run build:web
```

Start the local service with a registered artifact root:

```bash
uv run --locked junctionlens serve \
  --artifact-root artifacts/demo \
  --open-browser
```

The service validates `web/dist` before binding to `127.0.0.1`.
Use `--api-only` only when the JSON API is intentionally required without the browser client.

## Scene bundle contract

The viewer reads immutable `counterexample_bundle` artifacts with media type `application/vnd.junctionlens.scene-bundle+json`.
The payload schema version is `junctionlens.scene-bundle.v1`.
Each bundle contains a title, a persisted decision manifest identity, a license notice, and one or more synchronized frames.
Each frame contains camera-slot availability plus separate ground-truth, baseline, and candidate graph layers.
Lane nodes carry BEV polylines and confidence.
Control nodes carry BEV position, type, state, and confidence.
Edges are either lane-successor or control-applies-to-lane relations with explicit source and target identities.

The release-decision manifest must be an immutable parent of the scene bundle.
Every available camera must reference a registered raster-image artifact.
Every unavailable camera must provide a bounded restriction reason instead of an artifact path.
Nanosecond timestamps remain canonical decimal strings so browser number conversion cannot lose precision.

## Visible behavior

The header shows the registered scene identity and the exact persisted decision identity.
The decision label and reason codes come from the persisted `release_decision` payload.
Client filters cannot change that label.

Available camera slots show synchronized registered images for the selected frame.
Restricted slots show an explicit non-image state and the recorded reason.
The browser never receives a licensed dataset filesystem path.

The BEV panel shows directed lane-successor and control-to-lane edges with arrowheads.
Ground truth, baseline, and candidate use different colors and different line patterns.
Native checkboxes toggle each layer without modifying evidence.
Previous and next buttons and the left and right arrow keys navigate the frozen frame sequence.

## Accessibility and browser verification

The application uses semantic headings, navigation, fieldsets, native checkboxes, buttons, figures, and status disclosures.
Keyboard focus is visible against the dark background.
The graph includes an accessible name and description.
Status and layer identity never depend on color alone.

The browser gate runs the production build through the production FastAPI application in pinned Chromium.
It checks desktop and narrow viewports, Axe results, console errors, page errors, horizontal overflow, restricted-image behavior, graph toggles, directed edges, persisted status, buttons, and keyboard navigation.
The response must retain the restrictive content security policy while the test context alone bypasses that policy to inject Axe.

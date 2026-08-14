# ADR 0008: Custom match ordering and KPI thresholds

Status: Accepted

## Context

BUILD_PLAN.md defines `CustomMatchV1`, requires a quantized geometry key, and refers to thresholded graph edges and a frozen endpoint-gap threshold.
It does not assign numeric values to the geometry quantization, graph-edge threshold, or endpoint-gap threshold.
Those values must be fixed before custom KPI fixtures can be interpreted or compared.

The official evaluator can choose different matches at different metric thresholds.
Its threshold-specific matches therefore cannot be relabeled as a single official association map for custom metrics.

## Decision

`CustomMatchV1` runs inside the digest-verified OpenLane-V2 v2.1.0 evaluator image and calls its imported `lane_segment_distance`, `traffic_element_distance`, and `area_distance` functions directly.
Lane and road-area geometry keys use half-away-from-zero 1 millimeter quantization.
Traffic-control normalized boxes use half-away-from-zero 0.000001 quantization.
Predictions sort by descending raw existence confidence, the complete quantized geometry key, and decoder-query index.
The fixed match thresholds use strict less-than comparison at 2.0 meters for lanes, 0.75 IoU distance for controls, and 1.0 meter area distance.
Equal costs select the lexicographically smallest lossless ground-truth source object ID.

Thresholded graph KPIs use raw edge probability greater than or equal to 0.5.
The confident-wrong control threshold is calibrated probability greater than or equal to 0.9, as specified by BUILD_PLAN.md.
The successor endpoint-gap alert threshold is 2.0 meters.
Metric quantiles use linear type-7 interpolation.
Temporal geometry is resampled to 20 points before ego-motion-aligned residual calculation.

These values are recorded in `configs/metrics/v1.yaml` and are part of the V1 metric contract.
Changing any value requires a new metric contract version and regenerated goldens.

## Consequences

The custom association artifact remains explicitly distinct from the official threshold-specific artifacts.
Every prediction record contains its complete ordering evidence, all ground-truth costs, threshold decisions, pair decisions, and unmatched reason.
The evaluator image identity changes when the matching implementation changes, so a clean dual build and image-lock update are mandatory.
Custom matching remains local and networkless, but it requires the qualified evaluator image.

The selected numeric values are the smallest plan correction needed to make the previously underspecified terms executable.
They do not alter the project objective or any official OpenLane-V2 metric.

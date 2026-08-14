# Geometry core

JunctionLens uses a right-handed vehicle frame with positive X forward, positive Y left, and positive Z up.
The notation `T_target_source` maps column-vector points from the source frame into the target frame.
All public transform functions validate finite, rigid, right-handed matrices before use.

OpenLane-V2 points use positive X right, positive Y forward, and positive Z up.
The exact source-to-canonical basis mapping is `[x_jl, y_jl, z_jl] = [y_ol, -x_ol, z_ol]`.
Temporal alignment maps previous-frame points through the previous world pose and the inverse current world pose.

Image coordinates have positive U right and positive V down from the upper-left pixel corner.
Model boxes are continuous half-open extents `[u_min, v_min, u_max, v_max)`.
Resize, crop, and padding are represented by a homogeneous image transform, separate from camera calibration.
Box transformation evaluates all four corners and returns their finite axis-aligned envelope.

The Python API is exported by `junctionlens.geometry`.
The corresponding native API is declared in `cpp/include/junctionlens/geometry/geometry.hpp` and built as `junctionlens::geometry`.
Both implementations provide rigid transforms, calibrated projection and plane back-projection, image-box operations, polyline resampling, discrete Frechet and Chamfer distances, endpoint features, and deterministic rectangular Hungarian assignment.

Malformed shapes, nonfinite values, singular calibration, invalid rigid transforms, degenerate polylines, reversed boundaries, and nonincreasing timestamps fail closed.
The native API exposes stable `GEOMETRY_*` reason codes through `GeometryError::reason_code()`.

Run the complete milestone gate with:

```bash
./tools/jl verify-m1-2
```

The gate checks the frozen transform, meter, and pixel tolerances in release and ASan/UBSan builds, exercises randomized Python properties, and compares assignment costs with SciPy's independent implementation.

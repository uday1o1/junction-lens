# SceneControlGraph V1 contract

The canonical schema is `proto/junctionlens/v1/scene_control_graph.proto`.
Binary Protobuf is the immutable interchange representation, and strict ProtoJSON is the debugging and integration representation.

## Coordinates and time

Persisted 3D points use meters and double precision.
The vehicle frame is right-handed with X forward, Y left, and Z up.
`T_vehicle_camera` maps camera-frame points into the vehicle frame.
`T_world_vehicle` maps vehicle-frame points into the world frame.
Every rigid transform is a row-major 4 by 4 affine matrix with an orthonormal rotation, determinant +1, and final row `[0, 0, 0, 1]`.
Image transforms are row-major 3 by 3 matrices that map original pixel coordinates into model coordinates.
Timestamps are signed 64-bit nanoseconds and must be nonnegative in persisted frame identities.

Normalized boxes use source-image coordinates divided by the frozen source width and height.
The normalized model convention is half-open `[x_min, y_min, x_max, y_max)` with positive area inside `[0, 1]`.
`SourcePixelBox` separately retains the original numeric coordinates, source convention, and source image dimensions without rounding.

## Identities

Predicted node IDs are frame-local unsigned 64-bit integers.
The high eight bits contain the frozen node type code and the low 56 bits contain the zero-based type-local ordinal plus one.
Node IDs are unique across lane, traffic-control, and road-area nodes in one graph.

Track IDs are nonzero, segment-local unsigned 64-bit integers.
The tracker must not reuse an ID after termination.
Edge IDs are the first eight bytes of a length-prefixed SHA-256 projection over schema major, frame identity, edge type, source ID, and target ID.

Ground-truth source IDs remain in `AdapterMetadata` and are namespaced by dataset, split, segment, timestamp, and node type.
They are not treated as globally unique graph IDs.

## Compatibility and limits

Binary readers accept future minor fields for schema major 1 and preserve unknown wire fields.
Strict ProtoJSON readers reject unknown fields because ProtoJSON cannot preserve them safely.
Raw serialized bytes are not canonical across languages.
Logical hashes use the known-field canonical JSON projection specified by ADR 0007.

The fail-closed input and collection limits are declared in `configs/contracts/v1.yaml`.
Validation failures expose stable `CONTRACT_*` reason codes and a field path.

## Commands

Generate local Python bindings with `./tools/jl generate-bindings`.
Validate a binary graph with `junctionlens contract validate --input graph.pb`.
Convert a binary graph to ProtoJSON with `junctionlens contract convert --input graph.pb --output graph.json --from binary --to json`.
Run the complete milestone gate with `./tools/jl verify-m1-1`.

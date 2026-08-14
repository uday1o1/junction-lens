# ADR 0007: Protobuf compatibility boundary

Status: Accepted

Date: 2026-08-13

## Context

The V1 graph must cross Python, C++, and browser boundaries without treating implementation-specific wire bytes as a canonical identity.
Current Protobuf documentation states that deterministic serialization is not canonical across languages or builds.
Binary parsers retain unknown fields, while ProtoJSON cannot preserve an unknown field that has no descriptor.
The C++ runtime also requires an exact generated-code and runtime release match.

## Decision

`proto/junctionlens/v1/scene_control_graph.proto` is the sole schema source.
The repository-local `protoc` 31.1 compiler generates Python 6.31.1 source and typing stubs during bootstrap, and CMake generates C++ bindings into the build tree.
Generated language files are not committed.
The C++ library compiles and links Protobuf 6.31.1 from its hash-pinned release archive with Abseil fixed at commit `9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f`.
Compile-time and startup checks reject a different C++ Protobuf release.

Binary envelopes with schema major 1 and a future schema minor are accepted when all known V1 fields validate.
Unknown binary fields remain attached to the parsed message when it is reserialized.
A schema major other than 1 fails with `CONTRACT_SCHEMA_MAJOR`.
ProtoJSON rejects unknown fields because accepting and dropping them would falsely imply preservation.

Application identities use SHA-256 over a sorted, compact JSON projection of all known logical fields.
They never use raw Protobuf bytes as a cross-language canonical hash.
ProtoJSON `uint64` values remain decimal strings at the TypeScript boundary and are never coerced to JavaScript numbers.

Binary and JSON inputs are limited to 64 MiB before parsing.
The remaining collection and string limits are frozen in `configs/contracts/v1.yaml` and enforced by the validators.

## Consequences

Adding a backward-compatible field increments the schema minor and preserves existing field numbers and names.
Removing a field requires reserving both its number and name.
A breaking semantic or wire change requires a new schema major and a deliberate migration path.
Known-field logical hashes remain comparable between Python and C++ even when their wire field order differs.

## References

- [Protobuf serialization documentation](https://protobuf.dev/programming-guides/serialization-not-canonical/)
- [ProtoJSON format](https://protobuf.dev/programming-guides/json/)
- [Protobuf cross-version runtime guarantee](https://protobuf.dev/support/cross-version-runtime-guarantee/#cpp)

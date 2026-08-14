import { describe, expect, it } from "vitest";

import golden from "../../tests/fixtures/contract/v1/golden.json";
import { graphIdentityProjection, unsigned64String } from "../src/contract/ids";

describe("V1 ProtoJSON identities", () => {
  it("retains every uint64 identity as a decimal string", () => {
    const identities = graphIdentityProjection(golden);
    expect(identities.nodeIds).toEqual([
      "72057594037927937",
      "144115188075855873",
      "216172782113783809",
    ]);
    expect(identities.trackIds).toEqual(["101", "102", "103"]);
    expect(typeof identities.edgeIds[0]).toBe("string");
  });

  it("rejects number coercion, noncanonical decimals, and overflow", () => {
    expect(() => unsigned64String(72057594037927937, "node_id")).toThrow(
      "decimal string",
    );
    expect(() => unsigned64String("01", "node_id")).toThrow("decimal string");
    expect(() => unsigned64String("18446744073709551616", "node_id")).toThrow(
      "decimal string",
    );
  });
});

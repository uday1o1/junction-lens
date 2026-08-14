import { describe, expect, it } from "vitest";

import type { Decision } from "../src/api";
import { presentDecision } from "../src/decision";

function decision(status: string): Decision {
  return {
    schema_version: "junctionlens.gate-decision.v1",
    decision_sha256: "a".repeat(64),
    status,
    integrity_reason_codes: ["Z_REASON", "A_REASON"],
    cells: [{ status: "FAIL_REGRESSION", reason_code: "A_REASON" }],
  };
}

describe("persisted decision presentation", () => {
  it.each([
    ["PASS", "Pass", "pass"],
    ["FAIL_INTEGRITY", "Fail integrity", "fail"],
    ["FAIL_REGRESSION", "Fail regression", "fail"],
    ["FAIL_PERFORMANCE", "Fail performance", "fail"],
    ["INSUFFICIENT_EVIDENCE", "Insufficient evidence", "insufficient"],
    ["BLOCKED_INFRASTRUCTURE", "Blocked infrastructure", "blocked"],
    ["FUTURE_STATUS", "Unknown decision", "fail"],
  ] as const)("renders %s without changing it", (status, label, tone) => {
    expect(presentDecision(decision(status))).toEqual({
      label,
      tone,
      reasons: ["A_REASON", "Z_REASON"],
    });
  });
});

import type { Decision } from "./api";

export type DecisionPresentation = Readonly<{
  label: string;
  tone: "pass" | "fail" | "blocked" | "insufficient";
  reasons: readonly string[];
}>;

const LABELS: Readonly<Record<string, string>> = {
  PASS: "Pass",
  FAIL_INTEGRITY: "Fail integrity",
  FAIL_REGRESSION: "Fail regression",
  FAIL_PERFORMANCE: "Fail performance",
  INSUFFICIENT_EVIDENCE: "Insufficient evidence",
  BLOCKED_INFRASTRUCTURE: "Blocked infrastructure",
};

function textList(value: unknown): readonly string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

export function presentDecision(decision: Decision): DecisionPresentation {
  const cellReasons = (decision.cells ?? []).flatMap((cell) => {
    const reason = cell.reason_code;
    return cell.status !== "PASS" && typeof reason === "string" ? [reason] : [];
  });
  const reasons = [
    ...textList(decision.integrity_reason_codes),
    ...textList(decision.infrastructure_reason_codes),
    ...textList(decision.performance_reason_codes),
    ...cellReasons,
  ];
  const tone =
    decision.status === "PASS"
      ? "pass"
      : decision.status === "BLOCKED_INFRASTRUCTURE"
        ? "blocked"
        : decision.status === "INSUFFICIENT_EVIDENCE"
          ? "insufficient"
          : "fail";
  return {
    label: LABELS[decision.status] ?? "Unknown decision",
    tone,
    reasons: [...new Set(reasons)].sort(),
  };
}

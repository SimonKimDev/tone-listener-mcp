import type { AnalysisPreflight, Interval } from "./api";

export type AnalysisSourceInput = {
  kind: "youtube" | "audio_file";
  youtube_url?: string;
  upload_id?: string;
  duration_seconds?: number;
  intervals: Interval[];
  refresh: boolean;
};

export type ConfirmedPlan = { inputKey: string; plan: AnalysisPreflight };

// Identity of the exact request a preflight answered; any edit makes that preflight stale.
export function analysisInputKey(input: AnalysisSourceInput): string {
  return JSON.stringify([
    input.kind,
    input.youtube_url ?? null,
    input.upload_id ?? null,
    input.duration_seconds ?? null,
    input.intervals.map((item) => [item.start_seconds, item.end_seconds]),
    input.refresh,
  ]);
}

// The request bound the user is confirming, or null when no preflight matches the current input.
export function confirmedRequestBound(
  confirmed: ConfirmedPlan | null,
  input: AnalysisSourceInput,
): number | null {
  if (!confirmed || confirmed.inputKey !== analysisInputKey(input)) return null;
  return confirmed.plan.max_new_requests;
}

export function describePlan(plan: AnalysisPreflight): string {
  const parts = [];
  if (plan.reused) parts.push(`${plan.reused} reused from saved listening`);
  if (plan.shared_in_flight) parts.push(`${plan.shared_in_flight} joining analysis already running`);
  parts.push(plan.max_new_requests === 0
    ? "no new Gemini request"
    : `up to ${plan.max_new_requests} new Gemini request${plan.max_new_requests === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

export function describeIntervalPlan(item: AnalysisPreflight["intervals"][number]): string {
  if (item.status === "cache") {
    return `reuse saved result${item.analyzed_at ? ` from ${new Date(item.analyzed_at).toLocaleString()}` : ""}`;
  }
  if (item.status === "shared_in_flight") return "join analysis already running";
  return "new Gemini request";
}

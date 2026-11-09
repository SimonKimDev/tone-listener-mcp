import { describe, expect, it } from "vitest";
import type { AnalysisPreflight } from "./api";
import { analysisInputKey, confirmedRequestBound, describePlan, type AnalysisSourceInput } from "./analysisPlan";

const input: AnalysisSourceInput = {
  kind: "youtube",
  youtube_url: "https://youtu.be/401iS5McuoU",
  duration_seconds: 60,
  intervals: [{ start_seconds: 0, end_seconds: 5 }, { start_seconds: 5, end_seconds: 10 }],
  refresh: false,
};

const plan: AnalysisPreflight = {
  model: "offline-model",
  refresh: false,
  intervals: [
    { interval: input.intervals[0]!, status: "cache", analyzed_at: "2026-11-08T00:00:00+00:00" },
    { interval: input.intervals[1]!, status: "shared_in_flight", analyzed_at: null },
  ],
  reused: 1,
  shared_in_flight: 1,
  max_new_requests: 0,
  provider_contacted: false,
};

describe("analysis preflight confirmation", () => {
  it("confirms only the exact input the preflight answered", () => {
    const confirmed = { inputKey: analysisInputKey(input), plan };
    expect(confirmedRequestBound(confirmed, input)).toBe(0);
    expect(confirmedRequestBound(confirmed, { ...input })).toBe(0);
  });

  it("treats any changed bound, source, upload or refresh choice as unconfirmed", () => {
    const confirmed = { inputKey: analysisInputKey(input), plan };
    const changed: AnalysisSourceInput[] = [
      { ...input, intervals: [{ start_seconds: 0, end_seconds: 5 }, { start_seconds: 5, end_seconds: 10.5 }] },
      { ...input, youtube_url: "https://youtu.be/aaaaaaaaaaa" },
      { ...input, refresh: true },
      { ...input, kind: "audio_file", youtube_url: undefined, upload_id: "other" },
    ];
    for (const value of changed) expect(confirmedRequestBound(confirmed, value)).toBeNull();
    expect(confirmedRequestBound(null, input)).toBeNull();
  });

  it("separates reused and shared work from new requests", () => {
    expect(describePlan(plan)).toBe("1 reused from saved listening · 1 joining analysis already running · no new Gemini request");
    expect(describePlan({ ...plan, reused: 0, shared_in_flight: 0, max_new_requests: 3 })).toBe("up to 3 new Gemini requests");
  });
});

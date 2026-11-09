import { describe, expect, it, vi } from "vitest";
import { DI_SAMPLE_LABELS, DI_SAMPLES, type RefinementPlan, type RefinementRecord, type ToneProject } from "./api";
import { adoptable, claimUpload, needsReanalysis, planMatches, startAudio } from "./RefinementPanel";

const project = { id: "p", revision: 3, source: { kind: "youtube" }, chain: { revision: 2, nodes: [] } } as unknown as ToneProject;
const plan = { project_id: "p", project_revision: 3, model: "gemini-3.6-flash", samples: ["riff-1", "riff-2"] } as unknown as RefinementPlan;

describe("refinement confirmation", () => {
  it("only offers a run for the exact plan that was previewed", () => {
    expect(planMatches(plan, project, "gemini-3.6-flash", ["riff-1", "riff-2"])).toBe(true);
    expect(planMatches(plan, project, "gemini-4-argon", ["riff-1", "riff-2"])).toBe(false);
    expect(planMatches(plan, project, "gemini-3.6-flash", ["riff-1"])).toBe(false);
    expect(planMatches(plan, { ...project, revision: 4 } as ToneProject, "gemini-3.6-flash", ["riff-1", "riff-2"])).toBe(false);
    expect(planMatches(null, project, "gemini-3.6-flash", ["riff-1"])).toBe(false);
    const local = { ...plan, search: "local" } as RefinementPlan;
    expect(planMatches(local, project, "gemini-3.6-flash", ["riff-1", "riff-2"], null, "local")).toBe(true);
    expect(planMatches(local, project, "gemini-3.6-flash", ["riff-1", "riff-2"])).toBe(false);
    expect(planMatches(plan, project, "gemini-3.6-flash", ["riff-1", "riff-2"], null, "local")).toBe(false);
  });

  it("binds an uploaded-audio plan to the exact reselected file", () => {
    const fileProject = { ...project, source: { kind: "audio_file", content_sha256: "a".repeat(64) } } as unknown as ToneProject;
    const filePlan = { ...plan, reference: { kind: "audio_file", upload_id: "u1" } } as unknown as RefinementPlan;
    expect(planMatches(filePlan, fileProject, "gemini-3.6-flash", ["riff-1", "riff-2"], "u1")).toBe(true);
    expect(planMatches(filePlan, fileProject, "gemini-3.6-flash", ["riff-1", "riff-2"], "u2")).toBe(false);
    expect(planMatches(filePlan, fileProject, "gemini-3.6-flash", ["riff-1", "riff-2"], null)).toBe(false);
    expect(needsReanalysis(fileProject)).toBe(false);
    expect(needsReanalysis({ ...fileProject, source: { kind: "audio_file" } } as unknown as ToneProject)).toBe(true);
    expect(needsReanalysis(project)).toBe(false);
  });

  it("disables adoption for dirty or earlier chain revisions", () => {
    const record = { project_id: "p", project_revision: 3 } as RefinementRecord;
    expect(adoptable(record, project, false)).toBe(true);
    expect(adoptable(record, project, true)).toBe(false);
    expect(adoptable(record, { ...project, revision: 4 } as ToneProject, false)).toBe(false);
  });
});

describe("refinement panel lifecycle", () => {
  it("deletes a late upload once the panel no longer owns it", () => {
    const discard = vi.fn();
    expect(claimUpload(false, "late", discard)).toBe(false);
    expect(discard).toHaveBeenCalledWith("late");
    expect(claimUpload(true, "kept", discard)).toBe(true);
    expect(discard).toHaveBeenCalledTimes(1);
  });

  it("marks playing only after play() resolves and surfaces rejection", async () => {
    const started = vi.fn();
    const failed = vi.fn();
    await startAudio({ play: vi.fn().mockResolvedValue(undefined) }, started, failed);
    expect(started).toHaveBeenCalledTimes(1);
    expect(failed).not.toHaveBeenCalled();
    const blocked = new Error("NotAllowedError");
    await startAudio({ play: vi.fn().mockRejectedValue(blocked) }, started, failed);
    expect(started).toHaveBeenCalledTimes(1);
    expect(failed).toHaveBeenCalledWith(blocked);
    await startAudio(null, started, failed);
    expect(failed).toHaveBeenCalledTimes(2);
  });
});

describe("bundled DI samples", () => {
  it("keeps the riffs first and labels every sample, crediting the CC BY leads", () => {
    expect(DI_SAMPLES.slice(0, 2)).toEqual(["riff-1", "riff-2"]);
    expect(DI_SAMPLES).toContain("lead-1");
    expect(DI_SAMPLE_LABELS["lead-2"]).toContain("CC BY 4.0");
    const leads = { ...plan, samples: ["lead-1", "lead-2"] } as unknown as RefinementPlan;
    expect(planMatches(leads, project, "gemini-3.6-flash", ["lead-1", "lead-2"])).toBe(true);
    expect(planMatches(leads, project, "gemini-3.6-flash", ["riff-1", "riff-2"])).toBe(false);
  });
});

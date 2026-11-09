// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { api, DI_SAMPLE_LABELS, DI_SAMPLES, type JobStatus, type RefinementPlan, type RefinementRecord, type ToneProject } from "./api";
import * as jobs from "./jobs";
import { adoptable, claimUpload, needsReanalysis, planMatches, RefinementPanel, startAudio } from "./RefinementPanel";

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
  it("transfers the reference to an accepted job before the panel unmounts", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    const container = document.createElement("div");
    const root = createRoot(container);
    const fileProject: ToneProject = {
      ...project, source: { ...project.source, kind: "audio_file", content_sha256: "a".repeat(64) },
    };
    const filePlan: RefinementPlan = {
      ...plan, reference: { kind: "audio_file", upload_id: "reference" }, variants: [],
      max_new_requests: 2, uploads_per_request: 3,
    };
    const job: JobStatus = {
      id: "job", kind: "refinement", status: "queued", progress: 0, message: "Queued",
    };
    let finish!: (result: JobStatus) => void;
    vi.spyOn(api, "refinements").mockResolvedValue([]);
    vi.spyOn(api, "upload").mockResolvedValue({
      upload_id: "reference", display_name: "reference.wav", duration_seconds: 10,
      sample_rate: 16000, channels: 1, file_size_bytes: 100, intervals: [], expires_in_seconds: 1800,
    });
    vi.spyOn(api, "refinementPlan").mockResolvedValue(filePlan);
    const submit = vi.spyOn(api, "refine").mockResolvedValue(job);
    const discard = vi.spyOn(api, "discardUpload").mockResolvedValue(undefined);
    vi.spyOn(jobs, "waitForJob").mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    let unmounted = false;
    try {
      await act(async () => root.render(createElement(RefinementPanel, {
        project: fileProject, dirty: false, disabled: false, onAdopted: vi.fn(),
        guard: (callback) => callback,
      })));
      const file = container.querySelector<HTMLInputElement>('input[type="file"]')!;
      Object.defineProperty(file, "files", { value: [new File(["offline"], "reference.wav")] });
      await act(async () => file.dispatchEvent(new Event("change", { bubbles: true })));
      const planButton = [...container.querySelectorAll("button")].find(
        (button) => button.textContent === "Plan comparison · no request",
      )!;
      await act(async () => planButton.click());
      const consent = container.querySelector<HTMLInputElement>(".refine-plan input[type=checkbox]")!;
      await act(async () => consent.click());
      const run = [...container.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Run blinded comparison",
      )!;
      await act(async () => run.click());
      expect(submit).toHaveBeenCalledOnce();
      await act(async () => root.unmount());
      unmounted = true;
      expect(discard).not.toHaveBeenCalled();
    } finally {
      if (!unmounted) await act(async () => root.unmount());
      if (finish) await act(async () => finish({ ...job, status: "completed" }));
      vi.restoreAllMocks();
      vi.unstubAllGlobals();
    }
  });

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

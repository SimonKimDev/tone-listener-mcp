import { describe, expect, it } from "vitest";
import { automaticIntervals, createUploadOwner } from "./SourceInput";

describe("full precision source intervals", () => {
  for (const duration of [543131 / 44100, 1.2346, 247.0006, 9.9999, 30.0004]) {
    it(`never overruns or overlaps duration ${duration}`, () => {
      const intervals = automaticIntervals(duration);
      expect(intervals.length).toBeGreaterThan(0);
      expect(intervals.length).toBeLessThanOrEqual(3);
      for (let index = 0; index < intervals.length; index += 1) {
        const interval = intervals[index]!;
        expect(interval.end_seconds).toBeLessThanOrEqual(duration);
        expect(interval.end_seconds).toBeGreaterThan(interval.start_seconds);
        expect(interval.end_seconds - interval.start_seconds).toBeLessThanOrEqual(30);
        if (index > 0) expect(interval.start_seconds).toBeGreaterThanOrEqual(intervals[index - 1]!.end_seconds);
      }
    });
  }
});

describe("upload ownership", () => {
  function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
  }

  function harness() {
    const owner = createUploadOwner();
    const state = { upload: null as string | null, uploading: false, error: "" };
    const select = (request: Promise<string>) => owner.run(() => request, {
      onStart: () => { state.upload = null; state.uploading = true; state.error = ""; },
      onSuccess: (result) => { state.upload = result; },
      onError: (reason) => { state.error = String(reason); },
      onSettled: () => { state.uploading = false; },
    });
    return { owner, state, select };
  }

  it("keeps the latest file when it resolves before an older upload", async () => {
    const { state, select } = harness();
    const first = deferred<string>();
    const latest = deferred<string>();
    const firstRun = select(first.promise);
    const latestRun = select(latest.promise);
    latest.resolve("latest.wav");
    await latestRun;
    first.resolve("first.wav");
    await firstRun;
    expect(state).toEqual({ upload: "latest.wav", uploading: false, error: "" });
  });

  it("ignores an older completion while the latest upload is still pending", async () => {
    const { state, select } = harness();
    const first = deferred<string>();
    const latest = deferred<string>();
    const firstRun = select(first.promise);
    const latestRun = select(latest.promise);
    first.resolve("first.wav");
    await firstRun;
    expect(state).toEqual({ upload: null, uploading: true, error: "" });
    latest.resolve("latest.wav");
    await latestRun;
    expect(state).toEqual({ upload: "latest.wav", uploading: false, error: "" });
  });

  it("ignores an older failure after a newer upload starts", async () => {
    const { state, select } = harness();
    const first = deferred<string>();
    const latest = deferred<string>();
    const firstRun = select(first.promise);
    void select(latest.promise);
    first.reject("first failed");
    await firstRun;
    expect(state).toEqual({ upload: null, uploading: true, error: "" });
  });

  it("disowns a pending upload when the source changes", async () => {
    const { owner, state, select } = harness();
    const pending = deferred<string>();
    const run = select(pending.promise);
    owner.invalidate();
    state.uploading = false;
    pending.resolve("abandoned.wav");
    await run;
    expect(state).toEqual({ upload: null, uploading: false, error: "" });
  });
});

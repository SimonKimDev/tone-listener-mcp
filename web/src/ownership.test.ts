import { describe, expect, it } from "vitest";
import { createGeneration, guardedBy } from "./ownership";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

// Mirrors App: loads track project-open requests; displays bump when the shown view/project changes.
function navigator() {
  const loads = createGeneration();
  const displays = createGeneration();
  const state = { view: "new" as "new" | "workbench", project: null as string | null, preview: null as string | null, draftName: "" };
  const show = (view: "new" | "workbench", project: string | null) => {
    if (view !== state.view || project !== state.project) displays.next();
    state.view = view;
    state.project = project;
  };
  return {
    state,
    displays,
    async load(request: Promise<{ project: string; preview: string | null }>) {
      const token = loads.next();
      const result = await request;
      if (!loads.owns(token)) return;
      show("workbench", result.project);
      state.preview = result.preview;
    },
    beginNewDraft() {
      loads.next();
      show("new", state.project);
    },
    setPreview() {
      return guardedBy(displays, displays.current(), (preview: string | null) => { state.preview = preview; });
    },
  };
}

describe("navigation ownership", () => {
  it("keeps the latest project when an older open resolves last", async () => {
    const nav = navigator();
    const a = deferred<{ project: string; preview: string | null }>();
    const b = deferred<{ project: string; preview: string | null }>();
    const openA = nav.load(a.promise);
    const openB = nav.load(b.promise);
    b.resolve({ project: "B", preview: "b-preview" });
    await openB;
    a.resolve({ project: "A", preview: "a-preview" });
    await openA;
    expect(nav.state).toMatchObject({ view: "workbench", project: "B", preview: "b-preview" });
  });

  it("does not leave a new draft when a pending open resolves", async () => {
    const nav = navigator();
    const a = deferred<{ project: string; preview: string | null }>();
    const openA = nav.load(a.promise);
    nav.beginNewDraft();
    nav.state.draftName = "unsaved";
    a.resolve({ project: "A", preview: null });
    await openA;
    expect(nav.state).toMatchObject({ view: "new", project: null, draftName: "unsaved" });
  });

  it("drops a preview finished for a previous project", async () => {
    const nav = navigator();
    await nav.load(Promise.resolve({ project: "A", preview: null }));
    const setPreviewForA = nav.setPreview();
    await nav.load(Promise.resolve({ project: "B", preview: "b-preview" }));
    setPreviewForA("a-preview");
    expect(nav.state).toMatchObject({ project: "B", preview: "b-preview" });
  });

  it("applies results while the same view remains current", async () => {
    const nav = navigator();
    await nav.load(Promise.resolve({ project: "A", preview: null }));
    nav.setPreview()("a-preview");
    expect(nav.state.preview).toBe("a-preview");
  });
});

// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactFlowProps } from "@xyflow/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, type AdapterCatalog, type ChainNode, type ToneProject } from "./api";
import { ChainEditor } from "./ChainEditor";

vi.mock("@xyflow/react", () => ({
  Background: () => null,
  Handle: () => null,
  Position: { Left: "left", Right: "right" },
  ReactFlow: ({ nodes, onNodeClick }: ReactFlowProps) => (
    <div>
      {nodes?.map((node) => (
        <button key={node.id} data-node={node.id} data-draggable={node.draggable}
          onClick={(event) => onNodeClick?.(event, node)}>{node.id}</button>
      ))}
    </div>
  ),
}));

const catalog: AdapterCatalog = {
  adapter_id: "soundshed", adapter_label: "Soundshed", engine_version: "1.7.3",
  resources: [],
  blocks: [{
    kind: "gain", label: "Gain", description: "Trim",
    parameters: [{
      id: "gain_db", label: "Gain", minimum: -24, maximum: 24,
      default: 0, unit: "dB", description: "Output trim",
    }],
  }],
};

function node(id: string, kind: ChainNode["kind"]): ChainNode {
  return {
    id, kind, title: id, enabled: true, asset: null, note: null, note_role: "blocking",
    params: kind === "gain" ? { gain_db: 0 } : {},
  };
}

function project(): ToneProject {
  const interval = { start_seconds: 0, end_seconds: 10 };
  return {
    id: "project-a", schema_version: 1, revision: 1, name: "Draft",
    source: { kind: "audio_file", display_name: "DI.wav", duration_seconds: 10, selected_intervals: [interval] },
    analysis: [{ interval, model: "offline", unavailable: "Offline fixture" }],
    selected_analysis_index: 0, synthetic_fixture: true,
    intent: { summary: "Fixture", observations: [], uncertainties: [], gear_notes: [] },
    chain: { revision: 1, nodes: [node("input", "input"), node("gain-a", "gain"), node("gain-b", "gain"), node("output", "output")] },
    target: { adapter_id: "soundshed", engine_version: "1.7.3", resources: {} },
    created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
  };
}

let container: HTMLDivElement;
let root: Root;

function button(text: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((item) => item.textContent === text);
  if (!found) throw new Error(`Missing button: ${text}`);
  return found;
}

function order(): string[] {
  return [...container.querySelectorAll<HTMLElement>("[data-node]")].map((item) => item.dataset.node!);
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("chain editing", () => {
  it("keeps both endpoints fixed while allowing interior processors to move", async () => {
    await act(async () => root.render(
      <ChainEditor project={project()} catalog={catalog} onSaved={vi.fn()} onDirtyChange={vi.fn()} />,
    ));
    for (const id of ["input", "output"]) {
      await act(async () => button(id).click());
      expect(button("← Move").disabled).toBe(true);
      expect(button("Move →").disabled).toBe(true);
      await act(async () => { button("← Move").click(); button("Move →").click(); });
      expect(order()).toEqual(["input", "gain-a", "gain-b", "output"]);
    }
    await act(async () => button("gain-a").click());
    await act(async () => button("Move →").click());
    expect(order()).toEqual(["input", "gain-b", "gain-a", "output"]);
    await act(async () => button("← Move").click());
    expect(order()).toEqual(["input", "gain-a", "gain-b", "output"]);
  });

  it.each(["success", "failure"])("locks every chain mutation until save %s settles", async (outcome) => {
    let resolve!: (value: ToneProject) => void;
    let reject!: (reason: Error) => void;
    const pending = new Promise<ToneProject>((res, rej) => { resolve = res; reject = rej; });
    const update = vi.spyOn(api, "updateChain").mockReturnValue(pending);
    const saved = vi.fn();
    const initial = project();
    await act(async () => root.render(
      <ChainEditor project={initial} catalog={catalog} onSaved={saved} onDirtyChange={vi.fn()} />,
    ));
    await act(async () => button(" Enabled").click());
    await act(async () => button("Save chain").click());
    expect(update).toHaveBeenCalledOnce();
    expect(container.querySelectorAll("fieldset:disabled")).toHaveLength(2);
    for (const control of container.querySelectorAll("fieldset input, fieldset select, fieldset textarea, fieldset button")) {
      expect(control.matches(":disabled")).toBe(true);
    }
    for (const graphNode of container.querySelectorAll<HTMLElement>("[data-node]")) {
      expect(graphNode.dataset.draggable).toBe("false");
    }
    await act(async () => { button("Add block").click(); button("Remove").click(); button("Move →").click(); });
    expect(order()).toEqual(["input", "gain-a", "gain-b", "output"]);
    const updated = { ...initial, revision: 2, chain: { revision: 2, nodes: update.mock.calls[0]![1].nodes } };
    await act(async () => {
      if (outcome === "success") resolve(updated);
      else reject(new Error("Save failed."));
      await pending.catch(() => undefined);
    });
    expect(container.querySelectorAll("fieldset:disabled")).toHaveLength(0);
    if (outcome === "success") expect(saved).toHaveBeenCalledWith(updated);
    else {
      expect(saved).not.toHaveBeenCalled();
      expect(container.textContent).toContain("Save failed.");
      expect(button("Save chain").disabled).toBe(false);
    }
  });
});

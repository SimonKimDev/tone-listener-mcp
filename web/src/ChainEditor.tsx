import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeChange,
  type NodeProps,
} from "@xyflow/react";
import {
  api,
  type AdapterCatalog,
  type ChainNode,
  type ToneProject,
} from "./api";

type GraphData = {
  title: string;
  kind: ChainNode["kind"];
  assetName?: string;
  enabled: boolean;
  omitted?: boolean;
};
type GraphNode = Node<GraphData, "tone">;
type BlockKind = Exclude<ChainNode["kind"], "input" | "output" | "gear_note">;

const PROCESSOR_KINDS: BlockKind[] = [
  "nam_fx", "nam_amp", "cab_ir", "parametric_eq", "room_reverb", "digital_delay", "chorus", "gain",
];

function isProcessorKind(kind: ChainNode["kind"]): kind is BlockKind {
  return PROCESSOR_KINDS.includes(kind as BlockKind);
}

function SignalNode({ data, selected }: NodeProps<GraphNode>) {
  return (
    <div className={`signal-node kind-${data.kind} ${selected ? "selected" : ""} ${data.enabled ? "" : "bypassed"}`}>
      {data.kind !== "input" && <Handle type="target" position={Position.Left} isConnectable={false} />}
      <div className="node-topline">
        <span className="node-kind">{nodeLabel(data.kind)}</span>
        <span className={`node-status ${data.enabled ? "on" : ""}`}>{data.enabled ? "ON" : "BYPASSED"}</span>
      </div>
      <strong className="node-title">{data.title}</strong>
      {data.assetName && <span className="node-asset">{data.assetName}</span>}
      {data.kind === "gear_note" && <span className="node-asset">{data.omitted ? "Omitted from render · uncertain optional effect" : "Needs review · no substitution"}</span>}
      {data.kind !== "output" && <span className="node-port">SERIAL SIGNAL</span>}
      {data.kind !== "output" && <Handle type="source" position={Position.Right} isConnectable={false} />}
    </div>
  );
}

const nodeTypes = { tone: SignalNode };

function nodeLabel(kind: ChainNode["kind"]): string {
  const labels: Record<ChainNode["kind"], string> = {
    input: "SOURCE",
    nam_fx: "PEDAL NAM",
    nam_amp: "AMP NAM",
    cab_ir: "CABINET IR",
    parametric_eq: "EQUALIZER",
    room_reverb: "REVERB",
    digital_delay: "DELAY",
    chorus: "MODULATION",
    gain: "UTILITY",
    gear_note: "GEAR NOTE",
    output: "OUTPUT",
  };
  return labels[kind];
}

function defaultParams(kind: BlockKind, catalog: AdapterCatalog): Record<string, number> {
  const definition = catalog.blocks.find((block) => block.kind === kind);
  return Object.fromEntries((definition?.parameters ?? []).map((spec) => [spec.id, spec.default]));
}

function compatibleResource(kind: BlockKind, catalog: AdapterCatalog) {
  const resourceType = kind === "cab_ir" ? "ir" : "nam";
  return catalog.resources.filter((resource) => {
    if (resource.asset.resource_type !== resourceType) return false;
    if (kind === "nam_fx" && !["pedal", "unknown"].includes(resource.asset.capture_scope)) return false;
    if (kind === "nam_amp" && !["amp", "unknown"].includes(resource.asset.capture_scope)) return false;
    return true;
  });
}

function validate(nodes: ChainNode[], catalog: AdapterCatalog, forRender = true): string[] {
  const errors: string[] = [];
  if (nodes[0]?.kind !== "input" || nodes.at(-1)?.kind !== "output") {
    errors.push("Input and Output must remain fixed at the ends.");
  }
  if (nodes.filter((node) => node.kind === "input").length !== 1 || nodes.filter((node) => node.kind === "output").length !== 1) {
    errors.push("Input and Output must each appear exactly once.");
  }
  if (new Set(nodes.map((node) => node.id)).size !== nodes.length) errors.push("Node IDs must be unique.");
  if (nodes.length > 16) errors.push("The serial chain limit is 16 blocks.");
  for (const node of nodes) {
    if (node.kind === "gear_note") {
      if ((forRender && node.note_role !== "omitted_optional") || !node.note) errors.push(`${node.title}: ${node.note ?? "Explain this unresolved gear note."}`);
      continue;
    }
    const block = catalog.blocks.find((item) => item.kind === node.kind);
    for (const [id, value] of Object.entries(node.params ?? {})) {
      const parameter = block?.parameters?.find((item) => item.id === id);
      if (!parameter || !Number.isFinite(value) || value < parameter.minimum || value > parameter.maximum) {
        errors.push(`${node.title}: ${id} is not a supported value.`);
      }
    }
    if (["nam_fx", "nam_amp", "cab_ir"].includes(node.kind)) {
      const expected = node.kind === "cab_ir" ? "ir" : "nam";
      if (node.asset?.resource_type !== expected) errors.push(`${node.title}: choose a compatible local resource.`);
      else if (!catalog.resources.some((item) =>
        item.asset.resource_id === node.asset?.resource_id
        && item.asset.resource_type === node.asset?.resource_type
        && item.asset.sha256 === node.asset?.sha256)) {
        errors.push(`${node.title}: resource is unavailable or has changed.`);
      }
    }
  }
  return [...new Set(errors)];
}

function formatValue(value: number, unit: string): string {
  const digits = unit === "Hz" || unit === "ms" ? 0 : 2;
  return `${Number(value.toFixed(digits))} ${unit}`;
}

export function ChainEditor({
  project,
  catalog,
  onSaved,
  onDirtyChange,
}: {
  project: ToneProject;
  catalog: AdapterCatalog;
  onSaved: (project: ToneProject) => void;
  onDirtyChange: (dirty: boolean) => void;
}) {
  const [nodes, setNodes] = useState<ChainNode[]>(project.chain.nodes);
  const [selectedId, setSelectedId] = useState(project.chain.nodes[1]?.id ?? "input");
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [addKind, setAddKind] = useState<BlockKind>("parametric_eq");
  const [replacementKind, setReplacementKind] = useState<BlockKind>("nam_fx");
  const [replacementResource, setReplacementResource] = useState("");

  const dirty = JSON.stringify(nodes) !== JSON.stringify(project.chain.nodes);
  const errors = useMemo(() => validate(nodes, catalog), [nodes, catalog]);
  const saveErrors = useMemo(() => validate(nodes, catalog, false), [nodes, catalog]);
  const selected = nodes.find((node) => node.id === selectedId) ?? nodes[0];
  const selectedBlock = selected && catalog.blocks.find((block) => block.kind === selected.kind);
  const resourcesForSelected = selected && isProcessorKind(selected.kind)
    ? compatibleResource(selected.kind, catalog)
    : [];
  const graphNodes = useMemo<GraphNode[]>(() => nodes.map((node, index) => ({
    id: node.id,
    type: "tone",
    position: positions[node.id] ?? { x: 54 + index * 272, y: 80 },
    draggable: node.kind !== "input" && node.kind !== "output",
    focusable: true,
    ariaRole: "button",
    ariaLabel: `${nodeLabel(node.kind)}: ${node.title}`,
    selected: node.id === selectedId,
    data: {
      title: node.title,
      kind: node.kind,
      assetName: node.asset?.display_name,
      enabled: node.enabled,
      omitted: node.kind === "gear_note" && node.note_role === "omitted_optional",
    },
  })), [nodes, positions, selectedId]);
  const edges = useMemo<Edge[]>(() => nodes.slice(0, -1).map((node, index) => ({
    id: `edge-${node.id}-${nodes[index + 1]?.id}`,
    source: node.id,
    target: nodes[index + 1]?.id ?? "",
    type: "smoothstep",
    animated: false,
  })), [nodes]);

  useEffect(() => onDirtyChange(dirty), [dirty, onDirtyChange]);

  const onNodesChange = useCallback((changes: NodeChange<GraphNode>[]) => {
    for (const change of changes) {
      if (change.type === "select" && change.selected) setSelectedId(change.id);
    }
    const moved = changes.filter((change) => change.type === "position" && change.position);
    if (moved.length) {
      setPositions((current) => {
        const next = { ...current };
        for (const change of moved) {
          if (change.type === "position" && change.position) next[change.id] = change.position;
        }
        return next;
      });
    }
  }, []);

  useEffect(() => {
    if (selected?.kind === "gear_note") {
      setReplacementResource(compatibleResource(replacementKind, catalog)[0]?.asset.resource_id ?? "");
    }
  }, [selected?.id, selected?.kind, replacementKind, catalog]);

  function reorderByPosition(dragged: GraphNode) {
    if (dragged.id === "input" || dragged.id === "output") return;
    const position = { ...positions, [dragged.id]: dragged.position };
    const middle = nodes.slice(1, -1).slice().sort((left, right) => {
      const leftX = position[left.id]?.x ?? nodes.findIndex((node) => node.id === left.id) * 272;
      const rightX = position[right.id]?.x ?? nodes.findIndex((node) => node.id === right.id) * 272;
      return leftX - rightX;
    });
    setNodes([nodes[0]!, ...middle, nodes.at(-1)!]);
    setPositions({});
  }

  function updateNode(id: string, update: Partial<ChainNode>) {
    setNodes((current) => current.map((node) => node.id === id ? { ...node, ...update } : node));
  }

  function moveSelected(direction: -1 | 1) {
    if (!selected) return;
    const index = nodes.findIndex((node) => node.id === selected.id);
    const next = index + direction;
    if (index <= 0 || next >= nodes.length - 1) return;
    setNodes((current) => {
      const copy = [...current];
      [copy[index], copy[next]] = [copy[next]!, copy[index]!];
      return copy;
    });
    setPositions({});
  }

  function addBlock(kind: BlockKind) {
    if (nodes.length >= 16) return;
    const options = compatibleResource(kind, catalog);
    let asset: ChainNode["asset"] = null;
    if (["nam_fx", "nam_amp", "cab_ir"].includes(kind)) {
      if (!options.length) {
        setError(`No compatible local resource is available for ${nodeLabel(kind)}.`);
        return;
      }
      asset = options[0]!.asset;
    }
    const block = catalog.blocks.find((item) => item.kind === kind);
    const node: ChainNode = {
      id: `${kind}-${crypto.randomUUID().slice(0, 8)}`,
      kind,
      title: block?.label ?? nodeLabel(kind),
      enabled: true,
      params: defaultParams(kind, catalog),
      asset,
      note: null,
      note_role: "blocking",
    };
    setNodes((current) => [...current.slice(0, -1), node, current.at(-1)!]);
    setSelectedId(node.id);
    setError("");
  }

  function replaceGearNote(kind: BlockKind, resourceId: string) {
    if (!selected || selected.kind !== "gear_note") return;
    const block = catalog.blocks.find((item) => item.kind === kind);
    const resource = compatibleResource(kind, catalog).find((item) => item.asset.resource_id === resourceId);
    const assetRequired = ["nam_fx", "nam_amp", "cab_ir"].includes(kind);
    if (assetRequired && !resource) return;
    const replacement: ChainNode = {
      id: selected.id,
      kind,
      title: block?.label ?? nodeLabel(kind),
      enabled: true,
      params: defaultParams(kind, catalog),
      asset: resource?.asset ?? null,
      note: null,
      note_role: "blocking",
    };
    updateNode(selected.id, replacement);
  }

  async function saveChain() {
    if (saveErrors.length || !dirty) return;
    setSaving(true);
    setError("");
    try {
      const updated = await api.updateChain(project.id, {
        base_revision: project.revision,
        name: project.name,
        nodes,
      });
      setNodes(updated.chain.nodes);
      setPositions({});
      onSaved(updated);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save this chain.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="chain-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">02 / SIGNAL PATH</p>
          <h2>Shape the chain</h2>
        </div>
        <div className="chain-actions">
          <select
            aria-label="Block to add"
            value={addKind}
            onChange={(event) => setAddKind(event.target.value as BlockKind)}
            disabled={nodes.length >= 16}
          >
            {PROCESSOR_KINDS.map((kind) => (
              <option key={kind} value={kind}>+ {nodeLabel(kind)}</option>
            ))}
          </select>
          <button className="small-button add-block-button" onClick={() => addBlock(addKind)} disabled={nodes.length >= 16}>Add block</button>
          <button className={`secondary-button ${dirty ? "unsaved" : ""}`} onClick={() => void saveChain()} disabled={!dirty || saving || saveErrors.length > 0}>
            {saving ? "Saving…" : dirty ? "Save chain" : "Saved"}
          </button>
        </div>
      </div>
      <div className="chain-meta">
        <span><i className="status-dot" /> {nodes.length} blocks</span>
        <span>Drag a processor to reorder · Input and Output are fixed</span>
        <span>Project r{project.revision} / Chain r{project.chain.revision}</span>
      </div>
      <div className="flow-scroll">
        <div className="flow-canvas" style={{ width: Math.max(1000, nodes.length * 272 + 120) }}>
          <ReactFlow
            nodes={graphNodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onNodeClick={(_, node) => setSelectedId(node.id)}
            onNodeDragStop={(_, node) => reorderByPosition(node as GraphNode)}
            nodesConnectable={false}
            elementsSelectable
            panOnDrag
            zoomOnScroll={false}
            fitView={false}
            minZoom={0.65}
            maxZoom={1}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#292d2a" gap={24} size={1} />
          </ReactFlow>
        </div>
      </div>
      <div className="chain-detail-layout">
        <div className="node-editor">
          {selected ? (
            <>
              <div className="node-editor-head">
                <span className={`kind-mark mark-${selected.kind}`}>{nodeLabel(selected.kind)}</span>
                <div className="node-editor-buttons">
                  <button className="small-button" onClick={() => moveSelected(-1)} disabled={nodes.findIndex((node) => node.id === selected.id) <= 1}>← Move</button>
                  <button className="small-button" onClick={() => moveSelected(1)} disabled={nodes.findIndex((node) => node.id === selected.id) >= nodes.length - 2}>Move →</button>
                  {selected.kind !== "input" && selected.kind !== "output" && (
                    <button className="small-button remove-button" onClick={() => {
                      setNodes((current) => current.filter((node) => node.id !== selected.id));
                      setSelectedId(nodes[Math.max(0, nodes.findIndex((node) => node.id === selected.id) - 1)]?.id ?? "input");
                    }}>Remove</button>
                  )}
                </div>
              </div>
              {selected.kind === "gear_note" ? (
                <div className="gear-replace">
                  <div className="field-label">Resolve this positional gear note</div>
                  <div className="replace-row">
                    <select value={replacementKind} onChange={(event) => {
                      const next = event.target.value as BlockKind;
                      setReplacementKind(next);
                      setReplacementResource(compatibleResource(next, catalog)[0]?.asset.resource_id ?? "");
                    }}>
                      {PROCESSOR_KINDS.map((kind) => <option key={kind} value={kind}>{nodeLabel(kind)}</option>)}
                    </select>
                    {["nam_fx", "nam_amp", "cab_ir"].includes(replacementKind) && (
                      <select value={replacementResource} onChange={(event) => setReplacementResource(event.target.value)}>
                        {compatibleResource(replacementKind, catalog).map((resource) => (
                          <option key={resource.asset.resource_id} value={resource.asset.resource_id}>
                            {resource.asset.display_name}
                          </option>
                        ))}
                      </select>
                    )}
                    <button className="small-button" onClick={() => replaceGearNote(replacementKind, replacementResource)}>Replace</button>
                  </div>
                  <label className="editor-field"><span>Positional gear recommendation / limitation</span>
                    <textarea value={selected.note ?? ""} maxLength={1000} onChange={(event) => updateNode(selected.id, { note: event.target.value })} />
                  </label>
                  <p className="field-hint">This note is preserved in the exported description. Replacement is explicit; nothing is substituted automatically.</p>
                </div>
              ) : (
                <>
                  <div className="editor-grid">
                    <label className="editor-field">
                      <span>Block label</span>
                      <input value={selected.title} maxLength={120} onChange={(event) => updateNode(selected.id, { title: event.target.value })} />
                    </label>
                    <label className="bypass-control">
                      <span>Processor</span>
                      <button className={`toggle ${selected.enabled ? "enabled" : ""}`} onClick={() => updateNode(selected.id, { enabled: !selected.enabled })} aria-pressed={selected.enabled} disabled={selected.kind === "input" || selected.kind === "output"}>
                        <span /> {selected.enabled ? "Enabled" : "Bypassed"}
                      </button>
                    </label>
                  </div>
                  {selected.asset && (
                    <label className="editor-field resource-picker">
                      <span>Local Soundshed resource</span>
                      <select
                        value={selected.asset.resource_id}
                        onChange={(event) => {
                          const resource = resourcesForSelected.find((item) => item.asset.resource_id === event.target.value);
                          if (resource) updateNode(selected.id, { asset: resource.asset });
                        }}
                      >
                        {resourcesForSelected.map((resource) => (
                          <option key={resource.asset.resource_id} value={resource.asset.resource_id}>
                            {resource.asset.display_name} · {resource.provider}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                  {selectedBlock?.parameters?.length ? (
                    <div className="parameter-grid">
                      {selectedBlock.parameters.map((parameter) => {
                        const value = selected.params?.[parameter.id] ?? parameter.default;
                        const step = parameter.unit === "Hz" || parameter.unit === "ms" ? 1 : 0.01;
                        return (
                          <label className="parameter-control" key={parameter.id}>
                            <span className="parameter-label">
                              <strong>{parameter.label}</strong><output>{formatValue(value, parameter.unit)}</output>
                            </span>
                            <input
                              type="range"
                              min={parameter.minimum}
                              max={parameter.maximum}
                              step={step}
                              value={value}
                              aria-label={`${parameter.label}: ${parameter.description}`}
                              onChange={(event) => updateNode(selected.id, {
                                params: { ...selected.params, [parameter.id]: Number(event.target.value) },
                              })}
                            />
                            <small>{parameter.description}</small>
                          </label>
                        );
                      })}
                    </div>
                  ) : <p className="field-hint">{selectedBlock?.description}</p>}
                  <label className="editor-field"><span>Gear recommendation note (not rig identification)</span>
                    <textarea value={selected.note ?? ""} maxLength={1000} onChange={(event) => updateNode(selected.id, { note: event.target.value || null })} />
                  </label>
                </>
              )}
            </>
          ) : <p className="muted">Select a chain block to edit it.</p>}
        </div>
        <aside className={`validation-panel ${errors.length ? "has-errors" : "valid"}`}>
          <span className="validation-title">{errors.length ? "Needs attention" : "Chain ready"}</span>
          {errors.length ? (
            <ul>{errors.slice(0, 4).map((item) => <li key={item}>{item}</li>)}</ul>
          ) : (
            <p>One serial path · actual local NAM and IR resources · target-specific controls</p>
          )}
          {error && <p className="inline-warning">{error}</p>}
        </aside>
      </div>
    </section>
  );
}

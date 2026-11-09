import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type AdapterCatalog,
  type Health,
  type PreviewSnapshot,
  type ProjectSummary,
  type ToneProject,
} from "./api";
import { ChainEditor } from "./ChainEditor";
import { createGeneration, guardedBy } from "./ownership";
import { PreviewPlayer } from "./PreviewPlayer";
import { SourceInput } from "./SourceInput";

type View = "workbench" | "new";

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [catalog, setCatalog] = useState<AdapterCatalog | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [project, setProject] = useState<ToneProject | null>(null);
  const [preview, setPreview] = useState<PreviewSnapshot | null>(null);
  const [chainDirty, setChainDirty] = useState(false);
  const [view, setView] = useState<View>("new");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedInterval, setSelectedInterval] = useState(0);
  const [selectingProfile, setSelectingProfile] = useState(false);
  // loads: latest project-open request wins. displays: bumps whenever the shown view/project changes,
  // so async results started under an earlier view cannot replace the current one.
  const loads = useRef(createGeneration());
  const displays = useRef(createGeneration());
  const displayed = useRef<{ view: View; projectId: string | null }>({ view: "new", projectId: null });

  function display(nextView: View, nextProject: ToneProject | null) {
    const projectId = nextProject?.id ?? null;
    if (displayed.current.view !== nextView || displayed.current.projectId !== projectId) {
      displays.current.next();
      displayed.current = { view: nextView, projectId };
    }
  }

  useEffect(() => {
    setSelectedInterval(project?.selected_analysis_index ?? 0);
  }, [project?.id, project?.selected_analysis_index]);

  const refreshProjects = useCallback(async () => {
    setProjects(await api.projects());
  }, []);

  const loadProject = useCallback(async (id: string) => {
    const token = loads.current.next();
    setError("");
    try {
      const [nextProject, projectPreviews] = await Promise.all([
        api.project(id),
        api.previews(id),
      ]);
      if (!loads.current.owns(token)) return;
      display("workbench", nextProject);
      setProject(nextProject);
      setChainDirty(false);
      setPreview(projectPreviews.find((item) =>
        item.project_revision === nextProject.revision
        && item.chain_revision === nextProject.chain.revision,
      ) ?? null);
      setView("workbench");
    } catch (reason) {
      if (loads.current.owns(token)) setError(reason instanceof Error ? reason.message : "Could not open this local project.");
    }
  }, []);

  useEffect(() => {
    let active = true;
    void Promise.all([
      api.health(),
      api.catalog(),
      api.projects(),
    ])
      .then(async ([nextHealth, nextCatalog, nextProjects]) => {
        if (!active) return;
        setHealth(nextHealth);
        setCatalog(nextCatalog);
        setProjects(nextProjects);
        if (nextProjects.length) await loadProject(nextProjects[0]!.id);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "Could not connect to the local workbench.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [loadProject]);

  async function handleNewProject(nextProject: ToneProject) {
    loads.current.next();
    display("workbench", nextProject);
    setProject(nextProject);
    setPreview(null);
    setChainDirty(false);
    setView("workbench");
    await refreshProjects();
  }

  function handleSaved(nextProject: ToneProject) {
    setProject(nextProject);
    setPreview(null);
    setChainDirty(false);
    void refreshProjects();
  }

  function beginNewDraft() {
    loads.current.next();
    display("new", project);
    setError("");
    setView("new");
  }

  function showWorkbench() {
    if (!project) return beginNewDraft();
    loads.current.next();
    display("workbench", project);
    setView("workbench");
  }

  // Results owned by an earlier view are dropped; server-side saves still appear in the draft list.
  const epoch = displays.current.current();
  function owned<A extends unknown[]>(callback: (...args: A) => void) {
    return (...args: A) => {
      if (displays.current.owns(epoch)) callback(...args);
      else void refreshProjects();
    };
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={showWorkbench} aria-label="Tone Listener Workbench home">
          <span className="brand-symbol">∿</span>
          <span>TONE<span className="brand-light">LISTENER</span><small>LOCAL WORKBENCH</small></span>
        </button>
        <nav className="topnav">
          <button className={view === "workbench" ? "nav-active" : ""} onClick={showWorkbench} disabled={!project}>Workbench</button>
          <button className={view === "new" ? "nav-active" : ""} onClick={beginNewDraft}>New analysis <span>＋</span></button>
        </nav>
        <div className="topbar-status">
          <span className={`status-dot ${health?.renderer_available ? "green" : "amber"}`} />
          <span>{health?.renderer_available ? "LOCAL ENGINE" : "ENGINE SETUP NEEDED"}</span>
          <span className="topbar-divider" />
          <span className="model-name">{health?.model ?? "Connecting…"}</span>
        </div>
      </header>

      {loading ? (
        <main className="loading-screen"><span className="spinner" /><p>Opening local workbench…</p></main>
      ) : view === "new" ? (
        <SourceInput health={health} onProject={owned((nextProject: ToneProject) => void handleNewProject(nextProject))} />
      ) : project && catalog ? (
        <main className="workbench-main">
          <section className="project-bar">
            <div className="project-title-block">
              <span className="eyebrow">LOCAL PROJECT <span>·</span> SOUNDSHED GUITAR</span>
              <h1>{project.name}</h1>
              <div className="project-subline">
                <span>{project.source.display_name}</span>
                <span>·</span>
                <span>{project.analysis.length} reference interval{project.analysis.length === 1 ? "" : "s"}</span>
                <span>·</span>
                <span>Engine {project.target.engine_version}</span>
              </div>
            </div>
            <div className="project-switcher">
              <label htmlFor="project-select">Saved drafts</label>
              <select id="project-select" value={project.id} onChange={(event) => void loadProject(event.target.value)}>
                {projects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
            </div>
          </section>

          {error && <div className="error-banner global-error" role="alert">{error}<button onClick={() => setError("")}>×</button></div>}
          {!health?.renderer_available && (
            <div className="setup-banner">
              <strong>Native renderer is not validated.</strong>
              <span>{health?.renderer_error ?? "Build the VST3 host and install Soundshed Guitar 1.7.3. Preview and export require the DSP state-sensitivity self-test."}</span>
            </div>
          )}

          <div className="workbench-columns">
            <div className="workbench-left">
              <ChainEditor
                key={`${project.id}:${project.revision}`}
                project={project}
                catalog={catalog}
                onSaved={owned(handleSaved)}
                onDirtyChange={setChainDirty}
              />
              <PreviewPlayer
                key={project.id}
                project={project}
                preview={preview}
                setPreview={guardedBy(displays.current, epoch, setPreview)}
                dirty={chainDirty}
                disabled={!health?.renderer_available}
              />
            </div>
            <aside className="analysis-sidebar">
              <div className="sidebar-card intent-card">
                <p className="eyebrow">01 / REFERENCE READ</p>
                <h2>Listening notes</h2>
                <label className="editor-field">
                  <span>Recipe target interval</span>
                  <select value={selectedInterval} disabled={chainDirty || selectingProfile}
                    onChange={(event) => setSelectedInterval(Number(event.target.value))}>
                    {project.analysis.map((segment, index) => (
                      <option key={index} value={index}>Interval {index + 1} · {segment.interval.start_seconds.toFixed(1)}–{segment.interval.end_seconds.toFixed(1)}s · {segment.profile?.gain ?? segment.listening?.gain_level ?? "unavailable"}</option>
                    ))}
                  </select>
                </label>
                <button className="small-button" disabled={chainDirty || selectingProfile || selectedInterval === project.selected_analysis_index}
                  onClick={async () => {
                    setSelectingProfile(true);
                    const applySaved = owned(handleSaved);
                    const showError = guardedBy(displays.current, epoch, setError);
                    try {
                      applySaved(await api.selectProfile(project.id, project.revision, selectedInterval));
                    } catch (reason) {
                      showError(reason instanceof Error ? reason.message : "Could not select the target interval.");
                    } finally {
                      setSelectingProfile(false);
                    }
                  }}>Use interval & regenerate chain</button>
                <p className="field-hint">Regeneration replaces the saved chain and invalidates its preview. Only this interval supplies the recipe; other tones are not blended.</p>
                <p className="intent-summary">{project.intent.summary}</p>
                <div className="axis-list">
                  {project.intent.observations.map((observation) => {
                    const [label, ...rest] = observation.split(":");
                    return <div key={observation}><span>{label}</span><strong>{rest.join(":").trim()}</strong></div>;
                  })}
                </div>
                {project.intent.uncertainties.length > 0 && (
                  <div className="uncertainty-box">
                    <span>UNCERTAINTIES</span>
                    <ul>{project.intent.uncertainties.slice(0, 5).map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>
                  </div>
                )}
                {project.intent.gear_notes.map((note) => <p className="gear-note" key={note}>◇ {note}</p>)}
                <div className="analysis-meta">
                  {project.synthetic_fixture && <p className="inline-warning">OFFLINE SYNTHETIC FIXTURE · Invented profiles, not Gemini listening or tone-match evidence.</p>}
                  <span>{[...new Set(project.analysis.map((item) => item.model))].join(", ")}</span>
                  <span>{project.analysis.filter((item) => item.unavailable).length} unavailable</span>
                </div>
                <details className="interval-details">
                  <summary>Per-interval evidence & local metrics</summary>
                  {project.analysis.map((item, index) => {
                    const metrics = item.metrics;
                    const level = metrics && typeof metrics.level === "object" && metrics.level !== null
                      ? metrics.level as Record<string, unknown>
                      : null;
                    const rms = level?.rms_dbfs;
                    return (
                      <div className="interval-evidence" key={`${index}-${item.interval.start_seconds}`}>
                        <strong>Clip {index + 1} · {item.interval.start_seconds.toFixed(1)}–{item.interval.end_seconds.toFixed(1)} sec</strong>
                        <small>Model: {item.model}</small>
                        {item.profile ? (
                          <>
                            <p>{item.profile.gain} · {item.profile.brightness} · {item.profile.mids} mids · guitar {item.profile.guitar_dominance}</p>
                            {item.profile.evidence.slice(0, 3).map((evidence, evidenceIndex) => (
                              <small key={`${evidenceIndex}-${evidence.timestamp}`}>@ {evidence.timestamp.toFixed(1)}s · {evidence.observation}</small>
                            ))}
                          </>
                        ) : item.listening ? (
                          <>
                            <p>{item.listening.summary}</p>
                            {item.listening.evidence.slice(0, 3).map((evidence, evidenceIndex) => (
                              <small key={`${evidenceIndex}-${evidence.timestamp}`}>@ {evidence.timestamp.toFixed(1)}s · {evidence.observation}</small>
                            ))}
                          </>
                        ) : <p className="inline-warning">{item.unavailable ?? "Listening unavailable; no tone was inferred."}</p>}
                        {item.failure_info && (
                          <small>Provider failure: {item.failure_info.category.replaceAll("_", " ")}
                            {typeof item.failure_info.retry_after_seconds === "number"
                              && ` · Retry-After ${Math.ceil(item.failure_info.retry_after_seconds)} seconds`}
                          </small>
                        )}
                        {item.reuse?.kind === "cache" && (
                          <small>Reused saved listening from {new Date(item.reuse.analyzed_at).toLocaleString()} · same input, bounds, model and prompt · no new request</small>
                        )}
                        {item.reuse?.kind === "shared_in_flight" && (
                          <small>Shared result of an identical analysis already running · no separate request from this draft</small>
                        )}
                        {item.provider_attempted === false && !item.reuse && <small>No provider request was made for this interval.</small>}
                        {typeof rms === "number" && <small>Local decoded audio · RMS {rms.toFixed(1)} dBFS</small>}
                        {project.source.kind === "youtube" && <small>YouTube input · no extracted WAV or local DSP metrics</small>}
                      </div>
                    );
                  })}
                </details>
              </div>
              <div className="sidebar-card source-card-small">
                <p className="eyebrow">SOURCE & PROVENANCE</p>
                <div className="source-summary">
                  <span className="source-type">{project.source.kind === "youtube" ? "YT" : "DI"}</span>
                  <div><strong>{project.source.display_name}</strong><small>{project.source.kind === "youtube" ? "URL sent to Gemini · no local extraction" : `${project.source.duration_seconds?.toFixed(2)} sec · uploaded intervals only`}</small></div>
                </div>
                {project.source.url && <a className="source-link" href={project.source.url} target="_blank" rel="noreferrer">Open reference ↗</a>}
                <div className="interval-summary">
                  {project.source.selected_intervals.map((item, index) => (
                    <span key={index}>Clip {index + 1} · {item.start_seconds.toFixed(1)}–{item.end_seconds.toFixed(1)} sec</span>
                  ))}
                </div>
                <p className="field-hint">No waveform or local metrics are claimed for YouTube sources. Tone notes are cautious observations, not original-gear identification.</p>
              </div>
              <div className="sidebar-card provenance-card">
                <p className="eyebrow">PRESET & ENGINE</p>
                <div className="provenance-row"><span>Target</span><strong>Soundshed Guitar</strong></div>
                <div className="provenance-row"><span>Plugin</span><strong>{project.target.engine_version}</strong></div>
                <div className="provenance-row"><span>Resource mode</span><strong>Local references only</strong></div>
                <div className="provenance-row"><span>Model controls</span><strong>Fixed in captures</strong></div>
                <p className="field-hint">Capture Drive/Tone settings are baked into the NAM. Surrounding trims and EQ are separate controls.</p>
              </div>
            </aside>
          </div>
        </main>
      ) : (
        <main className="empty-workbench">
          <div className="empty-mark">∿</div>
          <p className="eyebrow">NO LOCAL DRAFTS YET</p>
          <h1>Start with a reference.<br /><em>Build from there.</em></h1>
          <button className="primary-button" onClick={beginNewDraft}>Create a tone draft <span>↗</span></button>
        </main>
      )}
      <footer className="app-footer">
        <span>LOCAL HOST · 127.0.0.1</span>
        <span>Preview uses actual dry DI through the installed Soundshed chain</span>
        <span>NO AUDIO LEAVES WITHOUT YOUR CONFIRMATION</span>
      </footer>
    </div>
  );
}

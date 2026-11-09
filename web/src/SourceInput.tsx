import { useEffect, useRef, useState } from "react";
import {
  analysisInputKey,
  confirmedRequestBound,
  describeIntervalPlan,
  describePlan,
  type AnalysisSourceInput,
  type ConfirmedPlan,
} from "./analysisPlan";
import { createGeneration, guardedBy } from "./ownership";
import { api, type AnalysisPreflight, type Health, type Interval, type JobStatus, type ToneProject, type UploadResult } from "./api";
import { terminalJobStates, waitForJob } from "./jobs";

type VideoDuration = {
  id: string | null;
  duration: number | null;
  loading: boolean;
  error: string | null;
};

function youtubeVideoId(value: string): string | null {
  try {
    const url = new URL(value.trim());
    if (url.protocol !== "https:" || url.username || url.password || url.port) return null;
    const host = url.hostname.toLowerCase();
    let id = "";
    if (host === "youtu.be" || host === "www.youtu.be") {
      id = url.pathname.slice(1).split("/")[0] ?? "";
    } else if (["youtube.com", "www.youtube.com", "m.youtube.com"].includes(host)) {
      const parts = url.pathname.split("/").filter(Boolean);
      if (parts.length === 1 && parts[0] === "watch") id = url.searchParams.get("v") ?? "";
      if (parts.length === 2 && ["shorts", "embed", "live"].includes(parts[0] ?? "")) id = parts[1] ?? "";
    }
    return /^[A-Za-z0-9_-]{11}$/.test(id) ? id : null;
  } catch {
    return null;
  }
}

function YouTubeDuration({
  url,
  onDuration,
}: {
  url: string;
  onDuration: (duration: number | null) => void;
}) {
  const [state, setState] = useState<VideoDuration>({
    id: null,
    duration: null,
    loading: false,
    error: null,
  });
  const frame = useRef<HTMLIFrameElement | null>(null);
  const id = youtubeVideoId(url);

  useEffect(() => {
    onDuration(null);
    if (!id) {
      setState({ id: null, duration: null, loading: false, error: null });
      return;
    }
    let disposed = false;
    const timeout = window.setTimeout(() => {
      if (!disposed) {
        setState({ id, duration: null, loading: false, error: "YouTube duration is unavailable." });
        onDuration(null);
      }
    }, 20_000);
    setState({ id, duration: null, loading: true, error: null });
    const receive = (event: MessageEvent) => {
      if (disposed || event.source !== frame.current?.contentWindow || event.data?.videoId !== id) return;
      if (event.data.type === "tone-duration" && Number.isFinite(event.data.duration)
          && event.data.duration > 0 && event.data.duration <= 86_400) {
        window.clearTimeout(timeout);
        setState({ id, duration: event.data.duration, loading: false, error: null });
        onDuration(event.data.duration);
      }
    };
    window.addEventListener("message", receive);
    return () => {
      disposed = true;
      window.clearTimeout(timeout);
      window.removeEventListener("message", receive);
    };
    // The effect restarts only when the video ID changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, onDuration]);

  if (!id) return null;
  return (
    <div className="youtube-player-wrap">
      <iframe key={id} ref={frame} title="Official reference video player"
        className="youtube-player" src={`/youtube-player.html?v=${id}`}
        sandbox="allow-scripts" allow="encrypted-media; fullscreen" />
      {state.loading && <span className="muted small">Reading duration from the official YouTube player…</span>}
      {state.error && <span className="inline-warning">{state.error} Enter intervals manually.</span>}
      {state.duration && <span className="muted small">Player duration: {formatSeconds(state.duration)}</span>}
    </div>
  );
}

function formatSeconds(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function automaticIntervals(duration: number): Interval[] {
  if (!Number.isFinite(duration) || duration <= 0) return [];
  const count = Math.min(3, Math.max(1, Math.floor(duration / 10)));
  const length = Math.min(30, duration / count);
  const starts = count === 1
    ? [Math.max(0, (duration - length) / 2)]
    : Array.from({ length: count }, (_, index) => ((duration - length) / (count - 1)) * index);
  return starts.map((start) => ({
    start_seconds: start,
    end_seconds: Math.min(duration, start + length),
  }));
}

export type UploadHandlers<T> = {
  onStart: () => void;
  onSuccess: (result: T) => void;
  onError: (reason: unknown) => void;
  onSettled: () => void;
};

// Only the most recently started upload may change source state; invalidate() also disowns it.
export function createUploadOwner() {
  let generation = 0;
  return {
    invalidate() {
      generation += 1;
    },
    async run<T>(request: () => Promise<T>, handlers: UploadHandlers<T>) {
      generation += 1;
      const mine = generation;
      handlers.onStart();
      try {
        const result = await request();
        if (mine === generation) handlers.onSuccess(result);
      } catch (reason) {
        if (mine === generation) handlers.onError(reason);
      } finally {
        if (mine === generation) handlers.onSettled();
      }
    },
  };
}

export function SourceInput({
  health,
  onProject,
}: {
  health: Health | null;
  onProject: (project: ToneProject) => void;
}) {
  const [kind, setKind] = useState<"youtube" | "audio_file">("youtube");
  const [url, setUrl] = useState("");
  const [duration, setDuration] = useState<number | null>(null);
  const [intervals, setIntervals] = useState<Interval[]>([]);
  const [editedIntervals, setEditedIntervals] = useState(false);
  const [upload, setUpload] = useState<UploadResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [consent, setConsent] = useState(false);
  const [name, setName] = useState("");
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState("");
  const [working, setWorking] = useState(false);
  const [refresh, setRefresh] = useState(false);
  const [confirmedPlan, setConfirmedPlan] = useState<ConfirmedPlan | null>(null);
  const [planError, setPlanError] = useState("");
  const [planRevision, setPlanRevision] = useState(0);
  const preflights = useRef(createGeneration());
  const videoId = youtubeVideoId(url);

  useEffect(() => {
    if (duration !== null && !editedIntervals) {
      setIntervals(automaticIntervals(duration));
    }
  }, [duration, editedIntervals]);

  const uploadOwner = useRef(createUploadOwner());

  useEffect(() => () => uploadOwner.current.invalidate(), []);

  function abandonPendingUpload() {
    uploadOwner.current.invalidate();
    setUploading(false);
  }

  async function onFileSelected(file?: File) {
    if (!file) {
      abandonPendingUpload();
      setUpload(null);
      setIntervals([]);
      setEditedIntervals(false);
      setDuration(null);
      return;
    }
    await uploadOwner.current.run(() => api.upload(file), {
      onStart: () => {
        setError("");
        setUpload(null);
        setIntervals([]);
        setEditedIntervals(false);
        setDuration(null);
        setUploading(true);
      },
      onSuccess: (result) => {
        setUpload(result);
        setDuration(result.duration_seconds);
        setIntervals(result.intervals);
        setName((current) => current || result.display_name.replace(/\.[^.]+$/, ""));
      },
      onError: (reason) => {
        setError(reason instanceof Error ? reason.message : "Audio upload failed.");
      },
      onSettled: () => setUploading(false),
    });
  }

  function updateInterval(index: number, field: keyof Interval, value: string) {
    setEditedIntervals(true);
    setIntervals((current) => current.map((item, itemIndex) => (
      itemIndex === index
        ? { ...item, [field]: value === "" ? 0 : Number(value) }
        : item
    )));
  }

  function addInterval() {
    if (intervals.length >= 3) return;
    setEditedIntervals(true);
    const last = intervals.at(-1);
    const start = last?.end_seconds ?? 0;
    const end = duration === null ? start + 30 : Math.min(duration, start + 30);
    if (end <= start) return;
    setIntervals((current) => [...current, { start_seconds: start, end_seconds: end }]);
  }

  function removeInterval(index: number) {
    setEditedIntervals(true);
    setIntervals((current) => current.filter((_, itemIndex) => itemIndex !== index));
  }

  async function startAnalysis() {
    const bound = confirmedRequestBound(confirmedPlan, analysisInput);
    if (bound === null) return;
    setError("");
    setWorking(true);
    setJob(null);
    try {
      let initial: JobStatus;
      try {
        initial = await api.analyze({
          ...analysisInput,
          external_consent: true,
          max_new_requests: bound,
          name: name.trim() || undefined,
        });
      } catch (reason) {
        // The server rejects a dispatch whose reuse plan changed; show the fresh counts to reconfirm.
        setConfirmedPlan(null);
        setPlanRevision((value) => value + 1);
        throw reason;
      }
      const finished = await waitForJob(initial, setJob);
      if (finished.status === "failed") throw new Error(finished.error ?? "Analysis failed.");
      if (finished.status === "cancelled") throw new Error("Analysis cancelled.");
      const project = finished.result?.project as ToneProject | undefined;
      if (!project) throw new Error("Analysis finished without a saved project.");
      onProject(project);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Analysis failed.");
    } finally {
      setWorking(false);
      setPlanRevision((value) => value + 1);
    }
  }

  async function cancelAnalysis() {
    if (!job) return;
    try {
      setJob(await api.cancelJob(job.id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel analysis.");
    }
  }

  const intervalsValid = intervals.length > 0
    && intervals.every((item) =>
      Number.isFinite(item.start_seconds)
      && Number.isFinite(item.end_seconds)
      && item.start_seconds >= 0
      && item.end_seconds > item.start_seconds
      && item.end_seconds - item.start_seconds <= 30
      && (duration === null || item.end_seconds <= duration))
    && intervals.every((item, index) =>
      index === 0 || item.start_seconds >= (intervals[index - 1]?.end_seconds ?? 0));
  const sourceReady = kind === "youtube" ? Boolean(videoId) : Boolean(upload);
  const analysisInput: AnalysisSourceInput = {
    kind,
    ...(kind === "youtube"
      ? { youtube_url: url, duration_seconds: duration ?? undefined }
      : { upload_id: upload?.upload_id, duration_seconds: upload?.duration_seconds }),
    intervals,
    refresh,
  };
  const inputKey = analysisInputKey(analysisInput);
  const canPlan = sourceReady && intervalsValid && !uploading;
  const requestBound = confirmedRequestBound(confirmedPlan, analysisInput);

  useEffect(() => {
    const token = preflights.current.next();
    setPlanError("");
    if (!canPlan) {
      setConfirmedPlan(null);
      return;
    }
    const input = JSON.parse(JSON.stringify(analysisInput)) as AnalysisSourceInput;
    const timer = window.setTimeout(() => {
      api.preflightAnalysis(input)
        .then(guardedBy(preflights.current, token, (plan: AnalysisPreflight) =>
          setConfirmedPlan({ inputKey: analysisInputKey(input), plan })))
        .catch(guardedBy(preflights.current, token, (reason: unknown) => {
          setConfirmedPlan(null);
          setPlanError(reason instanceof Error ? reason.message : "Could not check reusable results.");
        }));
    }, 250);
    return () => window.clearTimeout(timer);
    // analysisInput is fully described by inputKey.
  }, [inputKey, canPlan, planRevision]);

  return (
    <section className="source-screen">
      <div className="source-hero">
        <p className="eyebrow">LOCAL TONE WORKBENCH <span>·</span> SOUNDSHED ADAPTER</p>
        <h1>Build the tone.<br /><em>Then hear it.</em></h1>
        <p className="hero-copy">
          Analyze a reference, shape a real serial signal chain, and audition dry guitar through the installed Soundshed engine.
        </p>
      </div>
      <div className="source-card">
        <div className="source-card-head">
          <div>
            <span className="step-index">01</span>
            <h2>Choose a reference</h2>
          </div>
          <div className="local-chip"><span className="live-dot" /> LOCAL ONLY</div>
        </div>
        <div className="source-tabs" role="tablist">
          <button className={kind === "youtube" ? "active" : ""} onClick={() => {
            abandonPendingUpload();
            setKind("youtube"); setIntervals([]); setEditedIntervals(false); setUpload(null); setDuration(null); setConsent(false);
          }}>
            <span className="tab-icon">▶</span> YouTube link
          </button>
          <button className={kind === "audio_file" ? "active" : ""} onClick={() => {
            setKind("audio_file"); setIntervals([]); setEditedIntervals(false); setDuration(null); setConsent(false);
          }}>
            <span className="tab-icon">↑</span> Audio file
          </button>
        </div>
        {kind === "youtube" ? (
          <div className="source-fields">
            <label className="field-label" htmlFor="youtube-url">Public YouTube video URL</label>
            <input
              id="youtube-url"
              type="url"
              value={url}
              placeholder="https://www.youtube.com/watch?v=…"
              onChange={(event) => {
                setUrl(event.target.value);
                setDuration(null);
                setIntervals([]);
                setEditedIntervals(false);
              }}
            />
            {videoId && <YouTubeDuration url={url} onDuration={setDuration} />}
            {url && !videoId && <span className="inline-warning">Enter a public YouTube video URL, not a playlist or channel.</span>}
            <p className="field-hint">The official embedded player provides duration. The app does not download the video or extract its audio.</p>
          </div>
        ) : (
          <div className="source-fields">
            <label className="drop-zone">
              <span className="upload-icon">↑</span>
              <strong>{upload?.display_name ?? "Choose a local audio file"}</strong>
              <span>{upload
                ? `${upload.sample_rate.toLocaleString()} Hz · ${upload.channels} channel${upload.channels === 1 ? "" : "s"} · ${formatSeconds(upload.duration_seconds)}`
                : "WAV, FLAC, OGG or MP3 · up to 100 MiB · mono or stereo"}</span>
              <input
                type="file"
                accept="audio/*,.wav,.flac,.ogg,.mp3"
                onChange={(event) => void onFileSelected(event.target.files?.[0])}
              />
            </label>
            {uploading && <span className="muted small">Copying to private local app storage…</span>}
            <p className="field-hint">Only the selected interval is prepared for Gemini after you confirm below.</p>
          </div>
        )}

        <div className="interval-section">
          <div className="section-row">
            <div>
              <span className="field-label">Analysis intervals</span>
              <p className="field-hint">Up to 3 disjoint clips, each 30 seconds or shorter.</p>
            </div>
            {duration !== null && !editedIntervals && (
              <span className="auto-badge">AUTO-SELECTED</span>
            )}
          </div>
          {intervals.map((interval, index) => (
            <div className="interval-row" key={`interval-${index}`}>
              <span className="interval-index">0{index + 1}</span>
              <label>
                <span>Start · sec</span>
                <input
                  type="number" min="0" max={duration ?? 86400} step="0.1"
                  value={interval.start_seconds}
                  onChange={(event) => updateInterval(index, "start_seconds", event.target.value)}
                />
              </label>
              <span className="interval-arrow">→</span>
              <label>
                <span>End · sec</span>
                <input
                  type="number" min="0.1" max={duration ?? 86430} step="0.1"
                  value={interval.end_seconds}
                  onChange={(event) => updateInterval(index, "end_seconds", event.target.value)}
                />
              </label>
              <button className="icon-button" aria-label="Remove interval" onClick={() => removeInterval(index)}>×</button>
            </div>
          ))}
          {intervals.length === 0 && (
            <div className="empty-interval">Waiting for the player duration, or add an interval manually.</div>
          )}
          <button className="text-button" onClick={addInterval} disabled={intervals.length >= 3}>
            + Add interval
          </button>
        </div>

        <div className="project-name-row">
          <label className="field-label" htmlFor="project-name">Draft name <span className="muted">(optional)</span></label>
          <input id="project-name" value={name} maxLength={120} placeholder="e.g. Warm edge-of-breakup" onChange={(event) => setName(event.target.value)} />
        </div>

        <div className="consent-box">
          <input id="gemini-consent" type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
          <label htmlFor="gemini-consent">
            I agree to send {kind === "youtube" ? "this public video URL and selected intervals" : "the selected audio clips"} to Google Gemini using <strong>{health?.model ?? "the configured model"}</strong> for analysis.
            No account key is exposed to the browser. Review the exact clips below before starting.
          </label>
        </div>
        <div className="privacy-note">
          <span className="privacy-icon">◇</span>
          <span>
            Your audio, project and previews remain in app-owned local storage. The Gemini request is external; no tone profile, artist name or expected result is added.
          </span>
        </div>

        {error && <div className="error-banner" role="alert">{error}</div>}
        {job && !terminalJobStates.has(job.status) && (
          <div className="job-progress">
            <div className="progress-top"><span>{job.message}</span><button className="text-button danger-text" onClick={() => void cancelAnalysis()}>Cancel</button></div>
            <div className="progress-track"><span style={{ width: `${Math.max(8, job.progress * 100)}%` }} /></div>
          </div>
        )}
        {job?.status === "partial" && <div className="inline-warning">{job.message} Intervals with unavailable listening are retained as explicit limitations.</div>}

        <label className="refresh-option">
          <input type="checkbox" checked={refresh} onChange={(event) => setRefresh(event.target.checked)} />
          Listen again instead of reusing saved results for these exact intervals (sends new requests)
        </label>
        {intervalsValid && sourceReady && (
          <div className="confirmation">
            <strong>Ready to analyze · {confirmedPlan?.plan.model ?? health?.model ?? "model unavailable"}</strong>
            <span>{intervals.map((item, index) => {
              const status = requestBound !== null ? confirmedPlan?.plan.intervals[index] : undefined;
              return `#${index + 1} ${item.start_seconds.toFixed(1)}–${item.end_seconds.toFixed(1)}s${status ? ` (${describeIntervalPlan(status)})` : ""}`;
            }).join("  ·  ")}</span>
            {requestBound !== null && confirmedPlan
              ? <span className="request-count">Request check (no Gemini contact): {describePlan(confirmedPlan.plan)}. Reused results are earlier listening, not a new analysis.</span>
              : <span className="request-count">{planError || "Checking reusable results…"}</span>}
          </div>
        )}
        <button
          className="primary-button analyze-button"
          disabled={!sourceReady || !intervalsValid || !consent || uploading || working || requestBound === null || !health?.model_configured}
          onClick={() => void startAnalysis()}
        >
          {working ? "Analyzing reference…" : "Analyze & create draft"}
          <span>↗</span>
        </button>
        {!health?.model_configured && <p className="inline-warning">Add a Gemini key to the local config file to enable analysis.</p>}
      </div>
    </section>
  );
}

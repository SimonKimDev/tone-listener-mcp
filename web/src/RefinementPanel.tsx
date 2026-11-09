import { useEffect, useRef, useState } from "react";
import {
  api,
  DI_SAMPLES,
  type DiSample,
  type JobStatus,
  type JudgeModel,
  type RefinementPlan,
  type RefinementRecord,
  type ToneProject,
} from "./api";
import { waitForJob } from "./jobs";

export type SearchMode = "explore" | "local";

const SEARCH_LABELS: Record<SearchMode, string> = {
  explore: "Explore: other recipes and broad EQ",
  local: "Fine-tune: same gear, individual and joint input-trim / upper-EQ moves",
};

const AXES = ["gain", "brightness", "bass", "mids", "attack", "compression", "ambience"] as const;

export function planMatches(
  plan: RefinementPlan | null,
  project: ToneProject,
  model: JudgeModel,
  samples: DiSample[],
  uploadId: string | null = null,
  search: SearchMode = "explore",
): plan is RefinementPlan {
  return Boolean(
    plan
    && plan.project_id === project.id
    && plan.project_revision === project.revision
    && plan.model === model
    && plan.samples.join() === samples.join()
    && (plan.search ?? "explore") === search
    && (project.source.kind !== "audio_file" || (uploadId !== null && plan.reference.upload_id === uploadId)),
  );
}

/** Uploaded-audio projects analyzed before fingerprints cannot verify a reselected file. */
export function needsReanalysis(project: ToneProject): boolean {
  return project.source.kind === "audio_file" && !project.source.content_sha256;
}

export function adoptable(record: RefinementRecord, project: ToneProject, dirty: boolean): boolean {
  return !dirty && record.project_id === project.id && record.project_revision === project.revision;
}

/** Keep a late upload only while the request that started it still owns the panel; otherwise delete it. */
export function claimUpload(owned: boolean, uploadId: string, discard: (id: string) => void): boolean {
  if (!owned) discard(uploadId);
  return owned;
}

/** Report playing only after the media element actually started; surface a rejected play(). */
export async function startAudio(
  audio: Pick<HTMLAudioElement, "play"> | null,
  onStarted: () => void,
  onFailed: (reason: unknown) => void,
): Promise<void> {
  if (!audio) {
    onFailed(new Error("The player was not ready; use the player controls."));
    return;
  }
  try {
    await audio.play();
    onStarted();
  } catch (reason) {
    onFailed(reason);
  }
}

export function RefinementPanel({
  project,
  dirty,
  disabled,
  onAdopted,
  guard,
}: {
  project: ToneProject;
  dirty: boolean;
  disabled: boolean;
  onAdopted: (project: ToneProject) => void;
  guard: <A extends unknown[]>(callback: (...args: A) => void) => (...args: A) => void;
}) {
  const [model, setModel] = useState<JudgeModel>("gemini-3.6-flash");
  const [samples, setSamples] = useState<DiSample[]>(["riff-1", "riff-2"]);
  const [search, setSearch] = useState<SearchMode>("explore");
  const [plan, setPlan] = useState<RefinementPlan | null>(null);
  const [consent, setConsent] = useState(false);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [records, setRecords] = useState<RefinementRecord[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [playingKey, setPlayingKey] = useState("");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [reselected, setReselected] = useState<{ id: string; name: string } | null>(null);
  const uploadRef = useRef<string | null>(null);
  const mountedRef = useRef(true);
  const selectionRef = useRef(0);
  const pendingPlayRef = useRef<string | null>(null);
  const fileSource = project.source.kind === "audio_file";
  const legacy = needsReanalysis(project);
  const record = records[0] ?? null;
  const confirmed = planMatches(plan, project, model, samples, reselected?.id ?? null, search);

  // The server deletes a reselected file after the run or any refusal; discard it if the panel is abandoned.
  const forgetUpload = (discard: boolean) => {
    const id = uploadRef.current;
    uploadRef.current = null;
    setReselected(null);
    if (discard && id) void api.discardUpload(id).catch(() => undefined);
  };
  const discardQuietly = (id: string) => { void api.discardUpload(id).catch(() => undefined); };
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      selectionRef.current += 1;
      pendingPlayRef.current = null;
      if (uploadRef.current) discardQuietly(uploadRef.current);
      uploadRef.current = null;
    };
  }, []);

  async function reselect(file: File | undefined) {
    const selection = ++selectionRef.current;
    forgetUpload(true);
    setPlan(null);
    setConsent(false);
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const uploaded = await api.upload(file);
      const owned = mountedRef.current && selection === selectionRef.current;
      if (!claimUpload(owned, uploaded.upload_id, discardQuietly)) return;
      uploadRef.current = uploaded.upload_id;
      guard(setReselected)({ id: uploaded.upload_id, name: uploaded.display_name });
    } catch (reason) {
      show(describe(reason, "Could not read that file."));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    let active = true;
    api.refinements(project.id)
      .then((items) => { if (active) setRecords(items); })
      .catch((reason: unknown) => {
        if (!active) return;
        setRecords([]);
        setError(reason instanceof Error ? `Could not load earlier comparisons: ${reason.message}` : "Could not load earlier comparisons.");
      });
    return () => { active = false; };
  }, [project.id]);

  useEffect(() => {
    setPlan(null);
    setConsent(false);
  }, [project.revision, project.chain.revision, model, samples.join(), search]);

  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);

  // Runs after commit, so the <audio> element for this URL is mounted before play() is requested.
  useEffect(() => {
    const key = pendingPlayRef.current;
    if (!audioUrl || !key) return;
    pendingPlayRef.current = null;
    void startAudio(
      audioRef.current,
      () => { if (mountedRef.current) setPlayingKey(key); },
      (reason) => {
        if (!mountedRef.current) return;
        setPlayingKey("");
        setError(reason instanceof Error ? `Playback failed: ${reason.message}` : "Playback was blocked; use the player controls.");
      },
    );
  }, [audioUrl]);

  const show = guard(setError);
  const describe = (reason: unknown, fallback: string) => reason instanceof Error ? reason.message : fallback;

  async function preparePlan() {
    setBusy(true);
    setError("");
    const apply = guard(setPlan);
    try {
      apply(await api.refinementPlan(project.id, {
        project_revision: project.revision, chain_revision: project.chain.revision, model, samples, search,
        ...(reselected ? { upload_id: reselected.id } : {}),
      }));
    } catch (reason) {
      if (fileSource) forgetUpload(false);
      show(describe(reason, "Could not plan a refinement."));
    } finally {
      setBusy(false);
    }
  }

  async function run() {
    if (!confirmed || !consent) return;
    setBusy(true);
    setError("");
    const update = guard(setJob);
    const applyRecords = guard(setRecords);
    try {
      const started = await api.refine(project.id, {
        project_revision: project.revision, chain_revision: project.chain.revision, model, samples, search,
        plan_sha256: plan.plan_sha256, max_new_requests: plan.max_new_requests, external_consent: true,
        ...(reselected ? { upload_id: reselected.id } : {}),
      });
      // The accepted job owns the reference until its terminal cleanup, even if this panel unmounts.
      if (fileSource) forgetUpload(false);
      const finished = await waitForJob(started, update);
      if (finished.status === "failed") show(finished.error ?? "Refinement failed.");
      applyRecords(await api.refinements(project.id));
      guard(setPlan)(null);
      guard(setConsent)(false);
    } catch (reason) {
      show(describe(reason, "Could not start the refinement."));
    } finally {
      if (fileSource) forgetUpload(false);
      setBusy(false);
    }
  }

  async function play(variantId: string, sample: DiSample) {
    if (!record) return;
    const key = `${variantId}:${sample}`;
    try {
      const blob = await api.refinementAudio(project.id, record.id, variantId, sample);
      if (!mountedRef.current) return;
      const url = URL.createObjectURL(blob);
      pendingPlayRef.current = key;
      setPlayingKey("");
      setAudioUrl(url);
    } catch (reason) {
      show(describe(reason, "Could not load that render."));
    }
  }

  async function adopt(variantId: string) {
    if (!record) return;
    setBusy(true);
    const applied = guard(onAdopted);
    try {
      applied(await api.adoptVariant(project.id, record.id, project.revision, variantId));
    } catch (reason) {
      show(describe(reason, "Could not adopt that variant."));
    } finally {
      setBusy(false);
    }
  }

  const running = job !== null && (job.status === "queued" || job.status === "running");
  const labels = Object.fromEntries((record?.variants ?? []).map((item) => [item.id, item]));

  return (
    <section className="refine-section" aria-label="Reference comparison">
      <div className="refine-header">
        <div>
          <p className="eyebrow">04 / COMPARE WITH REFERENCE</p>
          <h2>Blinded refinement</h2>
        </div>
        <span className="refine-free">Local edits, previews & export never call Gemini</span>
      </div>
      <p className="field-hint">
        Renders a few local variants of this chain through Soundshed and asks one blinded, level-matched
        comparison per DI sample against the selected reference interval. Only gain direction is calibrated;
        brightness and ranking are weak evidence and ambience is not usable. Nothing here is a listener verdict.
        Audition clips are RMS-matched to each other with shared headroom (as the judge hears them), so a louder
        variant does not sound better just by volume; equal RMS is still not identical perceived loudness.
      </p>
      <div className="refine-controls">
        <label className="editor-field">
          <span>Judge model</span>
          <select value={model} disabled={busy || running} onChange={(event) => setModel(event.target.value as JudgeModel)}>
            <option value="gemini-3.6-flash">gemini-3.6-flash</option>
            <option value="gemini-4-argon">gemini-4-argon</option>
          </select>
        </label>
        <label className="editor-field">
          <span>Candidates</span>
          <select value={search} disabled={busy || running} onChange={(event) => setSearch(event.target.value as SearchMode)}>
            <option value="explore">{SEARCH_LABELS.explore}</option>
            <option value="local">{SEARCH_LABELS.local}</option>
          </select>
        </label>
        {DI_SAMPLES.map((sample) => (
          <label className="refine-check" key={sample}>
            <input type="checkbox" checked={samples.includes(sample)} disabled={busy || running}
              onChange={(event) => setSamples((current) => {
                const next = event.target.checked ? [...current, sample] : current.filter((item) => item !== sample);
                return DI_SAMPLES.filter((item) => next.includes(item));
              })} />
            {sample}
          </label>
        ))}
        {fileSource && !legacy && (
          <label className="editor-field">
            <span>Reselect analyzed file</span>
            <input type="file" accept="audio/*" disabled={busy || running}
              onChange={(event) => void reselect(event.target.files?.[0])} />
          </label>
        )}
        <button className="small-button"
          disabled={busy || running || dirty || disabled || samples.length === 0 || legacy || (fileSource && !reselected)}
          onClick={() => void preparePlan()}>Plan comparison · no request</button>
      </div>
      {fileSource && !legacy && (
        <p className="field-hint">
          Uploaded audio is not kept after analysis. Reselect the same file: it is checked against the analyzed
          file's content hash, sent once per request as the reference, and deleted when the comparison ends.
          {reselected && <> Selected: <strong>{reselected.name}</strong> (verified when you plan).</>}
        </p>
      )}
      {legacy && (
        <p className="inline-warning">
          This project was analyzed before source fingerprints were recorded, so a reselected file cannot be verified.
          Analyze the file again (cached intervals are reused without new requests) and refine the new project.
        </p>
      )}
      {search === "local" && (
        <p className="field-hint">
          Fine-tune keeps every device, resource, order and note of the saved chain. Individual moves change one
          setting: the first NAM capture's input trim ±3/±6 dB (not its fixed captured Drive/Tone) or the upper EQ
          ±3 dB (2 kHz and 8 kHz bands). Joint moves apply ±6 dB trim and ±3 dB EQ together (four combinations).
          Moves that clamp to an identical chain are listed once; the plan below shows the exact candidate and
          upload count before anything is sent. Adopt a result, then plan again to search around it. A moved setting is not an improvement unless the judged
          comparison says so; recipe alternatives are in Explore and are not a measured gain order.
        </p>
      )}
      {dirty && <p className="inline-warning">Save the chain before planning; unsaved edits are not compared.</p>}
      {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError("")}>×</button></div>}

      {confirmed && (
        <div className="refine-plan">
          <div className="refine-count">
            <strong>{plan.max_new_requests}</strong>
            <span>maximum new generation request{plan.max_new_requests === 1 ? "" : "s"} · {plan.uploads_per_request} uploads each · no retries, no reuse</span>
          </div>
          <ul className="refine-variants">
            {plan.variants.map((item) => (
              <li key={item.id}><strong>{item.label}</strong>{item.role === "control" && <em> control</em>} — {item.description}</li>
            ))}
          </ul>
          <p className="field-hint">{plan.disclosure} Reference: {plan.reference.kind === "audio_file" ? `reselected file ${reselected?.name ?? ""} (content verified)` : String(plan.reference.url)} {Number(plan.reference.start_seconds).toFixed(1)}–{Number(plan.reference.end_seconds).toFixed(1)} s.</p>
          <label className="refine-check">
            <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
            Send these renders and the reference interval to {plan.model} ({plan.max_new_requests} request{plan.max_new_requests === 1 ? "" : "s"} maximum)
          </label>
          <button className="primary-button" disabled={!consent || busy || running || dirty} onClick={() => void run()}>
            Run blinded comparison
          </button>
        </div>
      )}
      {running && job && (
        <div className="refine-progress">
          <span>{job.message}</span>
          <button className="small-button" onClick={() => void api.cancelJob(job.id)}>Cancel</button>
          <small>A request already sent still counts and its result is saved.</small>
        </div>
      )}

      {record && (
        <div className="refine-results">
          <div className="refine-result-meta">
            <span>{record.model}</span>
            <span>{SEARCH_LABELS[(record.search ?? "explore") as SearchMode].split(":")[0]}</span>
            <span>{new Date(record.created_at).toLocaleString()}</span>
            <span>{record.status}</span>
            <span>{record.new_generate_requests} of {record.confirmed_max_new_requests} confirmed request(s) used</span>
            {!adoptable(record, project, dirty) && <span className="inline-warning">From an earlier chain revision; adopt disabled</span>}
          </div>
          {record.suggestion && (
            <p className={`refine-suggestion ${record.suggestion.variant_id ? "has-suggestion" : ""}`}>
              {record.suggestion.variant_id
                ? <>Suggested: <strong>{labels[record.suggestion.variant_id]?.label}</strong>. </>
                : <>No change suggested. </>}
              {record.suggestion.basis}
            </p>
          )}
          {(record.judgments ?? []).map((judgment) => (
            <div className="refine-sample" key={judgment.sample}>
              <h3>{judgment.sample} · {judgment.status}{judgment.control_passed === true ? " · control passed" : judgment.control_passed === false ? " · control FAILED (not usable)" : ""}</h3>
              {judgment.reason && <p className="inline-warning">{judgment.reason}</p>}
              {judgment.control_detail && <small>{judgment.control_detail} · reference guitar {judgment.reference_guitar_audible ?? "n/a"}</small>}
              {(judgment.observations ?? []).length > 0 && (
                <table className="refine-table">
                  <thead>
                    <tr><th>Variant vs reference</th>{AXES.map((axis) => <th key={axis} className={axis === "ambience" ? "uncalibrated" : ""}>{axis}</th>)}<th>rank</th><th /></tr>
                  </thead>
                  <tbody>
                    {[...(judgment.observations ?? [])].sort((a, b) => a.rank - b.rank).map((item) => (
                      <tr key={item.variant_id}>
                        <td>{labels[item.variant_id]?.label ?? item.variant_id}</td>
                        {AXES.map((axis) => <td key={axis} className={axis === "ambience" ? "uncalibrated" : ""}>{item[axis].replace("_", " ")}</td>)}
                        <td>{item.rank}</td>
                        <td>
                          <button className="small-button" onClick={() => void play(item.variant_id, judgment.sample)}>
                            {playingKey === `${item.variant_id}:${judgment.sample}` ? "▶ playing" : "▶"}
                          </button>
                          {labels[item.variant_id]?.role === "variant" && (
                            <button className="small-button" disabled={busy || !adoptable(record, project, dirty)}
                              onClick={() => void adopt(item.variant_id)}>Adopt</button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ))}
          {audioUrl && <audio ref={audioRef} src={audioUrl} controls className="refine-audio"
            onPause={() => setPlayingKey("")} onEnded={() => setPlayingKey("")} />}
          <details>
            <summary>Evidence strength & limitations</summary>
            <ul>{Object.entries(record.evidence ?? {}).map(([axis, text]) => <li key={axis}><strong>{axis}</strong>: {text}</li>)}</ul>
            <ul>{(record.limitations ?? []).map((text) => <li key={text}>{text}</li>)}</ul>
          </details>
        </div>
      )}
    </section>
  );
}

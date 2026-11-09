import { useEffect, useRef, useState } from "react";
import { api, DI_SAMPLE_LABELS, DI_SAMPLES, type DiSample, type JobStatus, type PreviewSnapshot, type ToneProject } from "./api";
import { terminalJobStates, waitForJob } from "./jobs";
import { padChannelForPlayback, startTogether, stopSources } from "./playback";

type ActivePlayback = {
  drySource: AudioBufferSourceNode;
  wetSource: AudioBufferSourceNode;
  dryGain: GainNode;
  wetGain: GainNode;
  monitorGain: GainNode;
};

function dbToGain(decibels: number): number {
  return Math.pow(10, decibels / 20);
}

export function PreviewPlayer({
  project,
  preview,
  setPreview,
  dirty,
  disabled,
}: {
  project: ToneProject;
  preview: PreviewSnapshot | null;
  setPreview: (preview: PreviewSnapshot | null) => void;
  dirty: boolean;
  disabled: boolean;
}) {
  const [sample, setSample] = useState<DiSample>("riff-1");
  const [dryWet, setDryWet] = useState(1);
  const [monitorDb, setMonitorDb] = useState(-18);
  const [loop, setLoop] = useState(false);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState("");
  const [working, setWorking] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [audioLoading, setAudioLoading] = useState(false);
  const audioData = useRef<{ dry: AudioBuffer; wet: AudioBuffer } | null>(null);
  const context = useRef<AudioContext | null>(null);
  const playback = useRef<ActivePlayback | null>(null);
  const unresolvedNotes = project.chain.nodes.some((node) => node.kind === "gear_note" && node.note_role !== "omitted_optional");
  const omittedEffects = project.chain.nodes.filter((node) => node.kind === "gear_note" && node.note_role === "omitted_optional");
  const isCurrentPreview = Boolean(
    preview
    && preview.project_id === project.id
    && preview.project_revision === project.revision
    && preview.chain_revision === project.chain.revision
    && preview.engine_version === project.target.engine_version
    && preview.dsp_conditions.state_canary_version === 1
    && !disabled
    && !unresolvedNotes
    && !dirty,
  );

  useEffect(() => {
    stop();
    audioData.current = null;
    if (!isCurrentPreview || !preview) return;
    let active = true;
    setAudioLoading(true);
    setError("");
    void Promise.all([
      api.audio(project.id, preview.id),
      api.sample(preview.sample_name),
    ])
      .then(async ([wetBlob, dryBlob]) => {
        const audioContext = context.current ?? new AudioContext();
        context.current = audioContext;
        const [wet, unpaddedDry] = await Promise.all([
          audioContext.decodeAudioData(await wetBlob.arrayBuffer()),
          audioContext.decodeAudioData(await dryBlob.arrayBuffer()),
        ]);
        const dry = audioContext.createBuffer(
          unpaddedDry.numberOfChannels, wet.length, wet.sampleRate,
        );
        for (let channel = 0; channel < dry.numberOfChannels; channel += 1) {
          dry.copyToChannel(padChannelForPlayback(unpaddedDry.getChannelData(channel), wet.length), channel);
        }
        if (active) audioData.current = { dry, wet };
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "Could not load preview audio.");
      })
      .finally(() => {
        if (active) setAudioLoading(false);
      });
    return () => {
      active = false;
    };
    // The sample is part of the immutable preview snapshot.
  }, [project.id, preview?.id, isCurrentPreview]);

  useEffect(() => {
    const active = playback.current;
    if (active) {
      active.dryGain.gain.setTargetAtTime(1 - dryWet, context.current?.currentTime ?? 0, 0.02);
      active.wetGain.gain.setTargetAtTime(dryWet, context.current?.currentTime ?? 0, 0.02);
      active.monitorGain.gain.setTargetAtTime(dbToGain(monitorDb), context.current?.currentTime ?? 0, 0.02);
    }
  }, [dryWet, monitorDb]);

  useEffect(() => {
    if (playback.current) {
      playback.current.drySource.loop = loop;
      playback.current.wetSource.loop = loop;
    }
  }, [loop]);

  useEffect(() => () => {
    stop();
    void context.current?.close();
  }, []);

  function stop() {
    const active = playback.current;
    if (active) {
      stopSources([active.drySource, active.wetSource]);
      active.dryGain.disconnect();
      active.wetGain.disconnect();
      active.monitorGain.disconnect();
      playback.current = null;
    }
    setPlaying(false);
  }

  async function play() {
    if (!audioData.current) return;
    stop();
    const audioContext = context.current ?? new AudioContext();
    context.current = audioContext;
    await audioContext.resume();
    const drySource = audioContext.createBufferSource();
    const wetSource = audioContext.createBufferSource();
    const dryGain = audioContext.createGain();
    const wetGain = audioContext.createGain();
    const monitorGain = audioContext.createGain();
    drySource.buffer = audioData.current.dry;
    wetSource.buffer = audioData.current.wet;
    drySource.loop = loop;
    wetSource.loop = loop;
    dryGain.gain.value = 1 - dryWet;
    wetGain.gain.value = dryWet;
    monitorGain.gain.value = dbToGain(monitorDb);
    drySource.connect(dryGain).connect(monitorGain);
    wetSource.connect(wetGain).connect(monitorGain);
    monitorGain.connect(audioContext.destination);
    const active = { drySource, wetSource, dryGain, wetGain, monitorGain };
    playback.current = active;
    startTogether(drySource, wetSource, audioContext.currentTime, () => {
      if (playback.current === active) stop();
    });
    setPlaying(true);
  }

  async function render() {
    if (dirty || disabled || unresolvedNotes) return;
    setError("");
    setWorking(true);
    setJob(null);
    stop();
    try {
      const initial = await api.preview(project.id, {
        project_revision: project.revision,
        chain_revision: project.chain.revision,
        sample_name: sample,
      });
      const finished = await waitForJob(initial, setJob);
      if (finished.status === "failed") throw new Error(finished.error ?? "Preview render failed.");
      if (finished.status === "cancelled") throw new Error("Preview render cancelled.");
      if (finished.status === "stale") throw new Error("The project changed during rendering; render again.");
      const result = finished.result?.preview as PreviewSnapshot | undefined;
      if (!result) throw new Error("Renderer completed without a preview snapshot.");
      setPreview(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Preview render failed.");
    } finally {
      setWorking(false);
    }
  }

  async function cancel() {
    if (!job) return;
    try {
      setJob(await api.cancelJob(job.id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel rendering.");
    }
  }

  return (
    <section className="preview-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">03 / ACTUAL RENDER</p>
          <h2>Preview the chain</h2>
        </div>
        <div className="engine-pill">
          <span className="engine-mark">S</span>
          <span><strong>Soundshed Guitar</strong><small>{project.target.engine_version} · VST3 · 44.1 kHz</small></span>
        </div>
      </div>
      <div className="preview-panel">
        <div className="preview-main">
          <div className="preview-status-row">
            <span className="render-badge"><span /> REAL ENGINE RENDER</span>
            <span className="muted small">Dry DI → current chain → stereo WAV</span>
          </div>
          <div className="render-controls">
            <label className="editor-field sample-select">
              <span>Dry guitar sample</span>
              <select value={sample} onChange={(event) => {
                const next = event.target.value as DiSample;
                setSample(next);
                if (preview && preview.sample_name !== next) setPreview(null);
              }} disabled={working}>
                {DI_SAMPLES.map((name) => <option key={name} value={name}>{DI_SAMPLE_LABELS[name]}</option>)}
              </select>
            </label>
            <button className="primary-button render-button" disabled={dirty || disabled || unresolvedNotes || working} onClick={() => void render()}>
              {working ? "Rendering…" : "Render preview"} <span>▷</span>
            </button>
          </div>
          <p className="field-hint di-credit">
            Riffs: dry-guitar-samples by Daniel Farrell (CC0). Leads: excerpts of the Guitar-TECHS dataset by
            H. Pedroza, W. Abreu, R. M. Corey and I. R. Roman (ICASSP 2025, Zenodo 14963133),{" "}
            <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>;
            trimmed and resampled to 44.1 kHz, no gain.
          </p>
          {dirty && <div className="inline-warning">Save the chain before rendering. Unsaved controls never replace a current preview.</div>}
          {unresolvedNotes && <div className="inline-warning">Resolve or explicitly remove positional gear notes before rendering. They are not silently replaced by audio processors.</div>}
          {omittedEffects.length > 0 && <div className="inline-warning">This preview and export OMIT uncertain optional effect(s): {omittedEffects.map((node) => node.title).join(", ")}. Nothing was substituted; add one explicitly if you hear it.</div>}
          {job && !terminalJobStates.has(job.status) && (
            <div className="job-progress">
              <div className="progress-top"><span>{job.message}</span><button className="text-button danger-text" onClick={() => void cancel()}>Cancel</button></div>
              <div className="progress-track"><span style={{ width: `${Math.max(8, job.progress * 100)}%` }} /></div>
            </div>
          )}
          {error && <div className="error-banner" role="alert">{error}</div>}
          {preview && isCurrentPreview && (
            <>
              <div className="preview-ready">
                <span className="check-mark">✓</span>
                <div><strong>Preview ready</strong><span>{preview.duration_seconds.toFixed(2)} sec · {preview.sample_rate.toLocaleString()} Hz · {preview.audio_sha256.slice(0, 12)}…</span></div>
                <button className="small-button" onClick={() => setPreview(null)}>Clear</button>
              </div>
              {preview.peak > 1 && (
                <div className="inline-warning" role="alert">
                  DSP peak: +{(20 * Math.log10(preview.peak)).toFixed(1)} dBFS.
                  {" "}The float preview preserves this hot output without normalization.
                  Lower the saved Output trim for headroom; monitor volume does not change the exported preset.
                </div>
              )}
              <div className="player-controls">
                <button className="play-button" onClick={() => playing ? stop() : void play()} disabled={audioLoading || !audioData.current}>
                  {playing ? "Ⅱ" : "▶"}
                </button>
                <div className="player-control-grid">
                  <label className="range-control">
                    <span><strong>Dry / Wet</strong><output>{Math.round(dryWet * 100)}% wet</output></span>
                    <input type="range" min="0" max="1" step="0.01" value={dryWet} onChange={(event) => setDryWet(Number(event.target.value))} />
                  </label>
                  <label className="range-control">
                    <span><strong>Monitor level</strong><output>{monitorDb} dB · post-DSP</output></span>
                    <input type="range" min="-36" max="-6" step="1" value={monitorDb} onChange={(event) => setMonitorDb(Number(event.target.value))} />
                  </label>
                  <label className="loop-control">
                    <input type="checkbox" checked={loop} onChange={(event) => setLoop(event.target.checked)} />
                    <span>Loop sample</span>
                  </label>
                </div>
              </div>
              <p className="field-hint">Dry/Wet and monitor level are playback-only. The exported chain remains unchanged. Playback starts only when you press Play.</p>
              {preview.peak * dbToGain(monitorDb) > 1 && (
                <div className="inline-warning" role="alert">The wet signal can exceed playback headroom at this monitor level. Lower monitor volume or saved Output trim.</div>
              )}
            </>
          )}
          {audioLoading && <div className="muted small">Loading rendered and dry audio for local playback…</div>}
        </div>
        <aside className="render-facts">
          <span className="facts-title">Render conditions</span>
          <dl>
            <div><dt>Engine</dt><dd>{preview?.engine_version ?? project.target.engine_version}</dd></div>
            <div><dt>Model size</dt><dd>Full · no calibration</dd></div>
            <div><dt>Global FX</dt><dd>Bypassed</dd></div>
            <div><dt>Mixer slots</dt><dd>One allowlisted preset</dd></div>
            <div><dt>User profile</dt><dd>Untouched</dd></div>
          </dl>
          {preview && isCurrentPreview && <button className="export-button" onClick={async () => {
            try {
              const blob = await api.exportPreset(project.id, preview.id);
              const url = URL.createObjectURL(blob);
              const link = document.createElement("a");
              link.href = url;
              link.download = `${project.name}.soundshed.preset`;
              link.click();
              window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
            } catch (reason) {
              setError(reason instanceof Error ? reason.message : "Export failed.");
            }
          }}>Export .soundshed.preset <span>↓</span></button>}
          <p className="field-hint">Export references your local resource library. NAM and IR audio are not redistributed; the file is not portable without those resources.</p>
        </aside>
      </div>
    </section>
  );
}

#!/usr/bin/env python3
"""Separate, bounded accuracy phase for the same-gear local search (case-01 development data only).

Phase ``tone-accuracy-local-v1`` has its OWN durable registry entry and ledger (ceiling 12 generation attempts
including failures, no retries, first HTTP 429 pauses the phase). It never reads or writes the ledger, freeze or
artifacts of ``tone-copier-final-v1``; the final phase's case-01 files are copied in read-only and bound by hash.

Base: the final phase's selected ``pushed_clean_amp`` chain for case-01 (19.2-30 s). Reference profile: the final
phase's case-01 reference judgment, REUSED as development evidence (no fresh reference call; not independent).

A/B (8 relative calls): Flash and Argon x riff-1/riff-2 x two fresh label seeds (A, B), each judging the same fixed,
pre-rendered candidates (current + Fine-tune moves + wrong-gain control) with the shipped RMS-matched relative judge.
Per model, a candidate is supported only if ``suggest_from`` picks the same non-current candidate in A and in B.

C (4 absolute-profile calls, Flash) runs only if all 8 A/B judgments are usable and the union of per-model supported
candidates has exactly one member. Baseline: the historical final-phase WAVs, prepared as the listener prepares them
(16 kHz mono) and left at that level. Candidate: prepared the same way and scaled by one scalar to the baseline's
prepared RMS for the same DI. A copy that would exceed full scale blocks C before any call.

Subcommands: ``freeze --final-dir DIR`` (local), ``plan`` (no requests), ``run --live`` (paid; only after approval).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import secrets
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tone_copier_batch as driver
import tone_copier_final as final

from tone_listener_mcp import audio_io
from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.workbench import evaluation_v3, refinement
from tone_listener_mcp.workbench.adapters import soundshed as soundshed_module
from tone_listener_mcp.workbench.models import ToneProject

PHASE = "tone-accuracy-local-v1"
CEILING = 12
CASE = "case-01"
BASE_VARIANT = "pushed_clean_amp"
PROFILE_MODEL = final.PROFILE_MODEL
RELATIVE_MODELS = refinement.JUDGE_MODELS
SAMPLES = final.SAMPLES
SEEDS = ("A", "B")
UPLOAD_RATE = 16000
_, VIDEO, INTERVAL, _ = next(case for case in final.CASES if case[0] == CASE)
URL = f"https://www.youtube.com/watch?v={VIDEO}"
RELATIVE_STEPS = tuple((seed, model, sample) for seed in SEEDS for model in RELATIVE_MODELS for sample in SAMPLES)
C_STEPS = tuple((role, sample) for sample in SAMPLES for role in ("baseline", "candidate"))
assert len(RELATIVE_STEPS) + len(C_STEPS) == CEILING
FINAL_INPUTS = {
    "base_project": f"{CASE}/variants/{BASE_VARIANT}.project.json",
    "artifacts_manifest": f"{CASE}/artifacts.json",
    "reference_judgment": f"judgments/{CASE}:reference.json",
    "report": "reports/report-01.json",
    **{f"baseline_wav_{sample}": f"{CASE}/renders/{BASE_VARIANT}-{sample}.wav" for sample in SAMPLES},
    **{f"historical_eval_{sample}": f"judgments/{CASE}:eval-{sample}.json" for sample in SAMPLES},
}
LIMITATIONS = (
    "Case-01 is development data (it shaped pushed_clean_amp and the selection rules); nothing here is held out.",
    "The case-01 reference profile is reused from tone-copier-final-v1; it is not a fresh or independent reference.",
    "Every outcome is a model judgment, not a listener verdict; no result is a perceptual-match claim.",
    ("Relative support (A/B), categorical improvement (C) and perceptual matching are separate claims; "
     "the relative judge's calibration does not validate the absolute-profile judge."),
    ("C uploads are matched COPIES prepared for the listener (16 kHz mono, candidate scaled to the baseline's RMS); "
     "they are not raw native renders. Equal RMS is not equal loudness."),
    "Ambience/effects are reported separately and are not validated.",
)


def registry_path() -> Path:
    return final.registry_path()


def _now() -> str:
    return final._now()


def _sha(data: bytes) -> str:
    return driver._sha(data)


def _file_sha(path: Path) -> str:
    return _sha(Path(path).read_bytes())


def _pcm_sha(samples: np.ndarray) -> str:
    return _sha(np.ascontiguousarray(samples, dtype="<f4").tobytes())


def config() -> dict:
    return {
        "phase": PHASE, "ceiling_generate_attempts_including_failures": CEILING,
        "case": CASE, "youtube_id": VIDEO, "interval": list(INTERVAL),
        "provenance": ("DEVELOPMENT data: case-01 shaped pushed_clean_amp and the selection rules; the final phase's "
                       "case-01 reference profile is reused, not re-listened. Not held out; not independent."),
        "base": f"tone-copier-final-v1 {CASE} selected variant '{BASE_VARIANT}' (read-only copy, bound by hash)",
        "candidates": ("refinement.build_variants(search='local') on the base chain with the reused reference "
                       "profile: current, NAM input trim -6/-3/+3/+6 dB, upper-EQ tilt +/-3 dB, wrong-gain control. "
                       "Rendered once with both DIs and bound by hash before any request."),
        "relative_steps": [{"tag": _relative_tag(*step), "seed_set": step[0], "model": step[1], "sample": step[2],
                            "kind": "relative"} for step in RELATIVE_STEPS],
        "relative_protocol": ("Shipped relative judge (YouTube interval anchor, RMS matched to "
                              f"{refinement.relative_judge.TARGET_RMS_DBFS} dBFS, blinded labels). Each call draws a "
                              "fresh label seed; a B seed is redrawn locally until its label mapping differs from A "
                              "for the same model and DI."),
        "support_rule": ("Per model and seed set, summarize_judgment on both DIs then suggest_from. A model supports a "
                         "candidate only if A and B both suggest the same non-current candidate. Failed/invalid "
                         "judgments are missing evidence, never agreement."),
        "c_gate": ("C runs only if all 8 A/B judgments are usable (status 'judged' AND control_passed is True) and "
                   "the union of per-model "
                   "supported candidates has exactly one member. Different candidates across models, or none, skip "
                   "C; no model is preferred. Support is recorded as cross-model (both models) or one-model."),
        "c_steps": [{"tag": _c_tag(role, sample), "role": role, "sample": sample, "model": PROFILE_MODEL,
                     "kind": "absolute_profile"} for role, sample in C_STEPS],
        "c_preparation": ("audio_io.prepare_upload (mono mean, resample to 16 kHz). Baseline = historical final-phase "
                          "WAV at its prepared level (no rescaling). Candidate scaled by one scalar to the baseline's "
                          "prepared RMS for the same DI. Prepared copies must be invariant under the listener's own "
                          "preparation and within full scale (peak <= 1.0), else C is blocked before any request. "
                          "Scalar and raw/prepared/upload hashes are recorded."),
        "c_rubric": {
            "grader": "evaluation_v3.compare_profiles_v3(reused reference, judged copy, omitted effects)",
            "beats_per_di": ("guitar clear in both grades; every core axis known in both (else inconclusive); gain gap and brightness gap each do not worsen; "
                             "gain+brightness gap sum strictly decreases; core known-failure (threshold) count does "
                             "not increase; gain+brightness+mids distance does not increase"),
            "beats": "true only if beats on BOTH DIs; false if any DI is determinably not beaten; else inconclusive",
            "baseline_gap_reproduction": ("the baseline re-evaluation is compared with the historical final-phase "
                          "gain and brightness gaps on both DIs. Reproduction is one recheck of the baseline only; "
                          "candidate profiles are NOT repeated, so it does not exclude candidate-judgment noise. "
                          "Non-reproduction means baseline evaluator variability is visible (which is "
                          "not evidence that there was no real improvement)"),
            "thresholds": evaluation_v3.RULES,
        },
        "stop_rules": [
            "Any HTTP 429 pauses the phase; remaining steps are not_attempted; nothing is retried or substituted.",
            "A reserved tag is never re-sent (a crash after reservation counts as attempted).",
            "Stored artifacts that differ from their hashes block the phase instead of being regraded.",
        ],
        "limitations": list(LIMITATIONS),
    }


def _relative_tag(seed: str, model: str, sample: str) -> str:
    return f"{seed}:{model}:{sample}"


def _c_tag(role: str, sample: str) -> str:
    return f"C:{role}:{sample}"


def _inputs_identity(root: Path) -> dict:
    return {name: _file_sha(root / "inputs" / Path(rel).name) for name, rel in FINAL_INPUTS.items()}


def identity(renderer, root: Path, variants: dict) -> dict:
    files = {"accuracy_runner": __file__, "final_runner_helpers": inspect.getfile(final),
             "refinement_selection": inspect.getfile(refinement),
             "soundshed_adapter_recipes": inspect.getfile(soundshed_module)}
    value = driver.frozen_identity(None)
    value.update({name: _file_sha(Path(path)) for name, path in files.items()})
    value.update({"engine": getattr(renderer, "engine_version", None),
                  "worker_sha256": getattr(renderer, "_worker_sha256", None),
                  "plugin_sha256": getattr(renderer, "_plugin_sha256", None),
                  "inputs": _inputs_identity(root), "variant_chain_sha256": variants, "config": config()})
    return value


class Phase(driver.Batch):
    """Registry-bound ledger, identical in mechanism to the final phase but with its own id and ceiling.

    Subclasses (later development phases) override ``phase_id``, ``ceiling`` and ``identity``; each id has its own
    registry entry, ledger, lock and freeze history, so no phase can reuse or reset another's budget.
    """

    phase_id = PHASE
    ceiling = CEILING

    def binding_note(self, binding: str) -> dict:
        return {"phase": self.phase_id, "binding": binding,
                "note": "The authoritative ledger, tags and freeze are kept in the per-user phase registry."}

    def identity(self, renderer, variants: dict) -> dict:
        return identity(renderer, self.root, variants)

    def __init__(self, output: Path, create: bool = False):
        output = Path(output).resolve()
        registry = registry_path()
        registry.mkdir(parents=True, exist_ok=True)
        os.chmod(registry, 0o700)
        name, ceiling = self.phase_id, self.ceiling
        marker, state = registry / f"{name}.json", registry / name
        with final._flock(registry / f".{name}.lock"):
            if marker.exists():
                info = driver._load(marker)
                if info["output"] != str(output):
                    raise SystemExit(f"{name} is registered to another directory; its ceiling cannot be reset.")
            else:
                if not create:
                    raise SystemExit(f"{name} is not initialized; run `freeze` first.")
                if state.exists():
                    raise SystemExit("Phase state exists without its registry entry; refusing (fail closed).")
                if output.exists() and any(output.iterdir()):
                    raise SystemExit("A new phase needs a new or empty output directory.")
                info = {"phase": name, "output": str(output), "ceiling": ceiling,
                        "binding": secrets.token_hex(16), "created_at": _now()}
                state.mkdir(mode=0o700)
                driver._save(state / "ledger.json", {"ceiling": ceiling, "attempts": [], "provider_pause": None,
                                                     "binding": info["binding"]})
                output.mkdir(parents=True, exist_ok=True)
                driver._save(output / "phase-binding.json", self.binding_note(info["binding"]))
                driver._save(marker, info)
            self._verify(info, output, state)
        self.root, self.state = output, state
        self.ledger_path, self.lock_path = state / "ledger.json", state / ".ledger.lock"

    def _verify(self, info: dict, output: Path, state: Path) -> None:
        ledger_path, binding_path = state / "ledger.json", output / "phase-binding.json"
        if not ledger_path.exists():
            raise SystemExit("Authoritative phase ledger is missing; refusing to start a new one (fail closed).")
        ledger = driver._load(ledger_path)
        if info.get("phase", self.phase_id) != self.phase_id or info.get("ceiling") != self.ceiling or \
                ledger.get("binding") != info["binding"] or ledger.get("ceiling") != self.ceiling:
            raise SystemExit("Authoritative phase ledger does not match the registry; refusing (fail closed).")
        if not binding_path.exists() or driver._load(binding_path).get("binding") != info["binding"]:
            raise SystemExit("Output directory is missing or was replaced; its phase ledger cannot restart at zero.")
        if driver._load(binding_path).get("phase") != self.phase_id:
            raise SystemExit(f"Output binding does not name {self.phase_id}; refusing (fail closed).")

    tag_used = final.Phase.tag_used
    _freeze_record = final.Phase._freeze_record
    current_freeze = final.Phase.current_freeze

    def freeze_phase(self, renderer, variants: dict, revise: bool = False, reason: str | None = None) -> str:
        value = self.identity(renderer, variants)
        digest = _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        with self._locked():
            record = self._freeze_record()
            current = record["current"]
            if current and current["sha256"] == digest:
                return digest
            if current:
                ledger = driver._load(self.ledger_path)
                if not revise:
                    raise SystemExit("Frozen identity changed; this phase cannot be refrozen or reset "
                                     "(`freeze --revise` is allowed only before any reservation).")
                if ledger["attempts"] or ledger["provider_pause"]:
                    raise SystemExit("Revision refused: the phase already has reservations; the freeze is final.")
                record["history"].append({**current, "status": "superseded_unrun", "superseded_at": _now(),
                                          "reason": reason or "unspecified pre-spend revision",
                                          "attempts_at_supersession": 0})
            supersedes = record["history"][-1]["sha256"] if record["history"] else None
            record["current"] = {"sha256": digest, "frozen_at": _now(), "supersedes": supersedes, "reason": reason,
                                 "identity": value}
            driver._save(self.state / "freeze.json", record)
        driver._save(self.root / "freeze.json", {"note": "Copy; the authoritative freeze lives with the ledger.",
                                                  **record})
        return digest


class Blocked(Exception):
    pass


def import_inputs(phase: Phase, final_dir: Path) -> dict:
    """Copy the final phase's case-01 inputs (read-only) once; afterwards only verify the copies."""
    inputs = phase.root / "inputs"
    manifest_path = inputs / "inputs.json"
    if manifest_path.exists():
        manifest = driver._load(manifest_path)
        for name, entry in manifest["files"].items():
            if _file_sha(inputs / entry["file"]) != entry["sha256"]:
                raise Blocked(f"Imported input {name} changed; refusing.")
        return manifest
    inputs.mkdir(exist_ok=True)
    files = {}
    for name, rel in FINAL_INPUTS.items():
        source = Path(final_dir) / rel
        data = source.read_bytes()
        target = inputs / Path(rel).name
        target.write_bytes(data)
        os.chmod(target, 0o400)
        files[name] = {"file": target.name, "source": f"tone-copier-final-v1/{rel}", "sha256": _sha(data)}
    artifacts = json.loads((inputs / Path(FINAL_INPUTS["artifacts_manifest"]).name).read_text())
    base = artifacts["variants"][BASE_VARIANT]
    for sample in SAMPLES:
        if base["renders"][sample]["file_sha256"] != files[f"baseline_wav_{sample}"]["sha256"]:
            raise Blocked(f"Historical baseline WAV {sample} does not match the final-phase artifact manifest.")
    if base["project_sha256"] != files["base_project"]["sha256"]:
        raise Blocked("Historical base project does not match the final-phase artifact manifest.")
    report = json.loads((inputs / Path(FINAL_INPUTS["report"]).name).read_text())
    case = next(item for item in report["cases"] if item["case"] == CASE)
    if case["selection"]["selected"] != BASE_VARIANT:
        raise Blocked("The final phase did not select the expected base variant.")
    manifest = {"source_phase": "tone-copier-final-v1", "case": CASE, "imported_at": _now(), "files": files,
                "note": "Read-only copies; the final phase's own ledger/freeze/artifacts are never touched."}
    driver._save(manifest_path, manifest)
    return manifest


def _input(phase: Phase, name: str) -> Path:
    return phase.root / "inputs" / Path(FINAL_INPUTS[name]).name


def reference_profile(phase: Phase):
    profile = final._profile(driver._load(_input(phase, "reference_judgment")))
    if profile is None:
        raise Blocked("Reused reference judgment has no profile.")
    return profile


def base_project(phase: Phase) -> ToneProject:
    return ToneProject.model_validate(driver._load(_input(phase, "base_project")))


def historical(phase: Phase, reference) -> dict:
    """Historical final-phase grades of the baseline, recomputed and checked against the frozen report."""
    report = driver._load(_input(phase, "report"))
    case = next(item for item in report["cases"] if item["case"] == CASE)
    omitted = final._omitted(base_project(phase).chain)
    out = {}
    for sample in SAMPLES:
        recorded = case["evaluation"]["per_di"][sample]["grade_v3"]
        profile = final._profile(driver._load(_input(phase, f"historical_eval_{sample}")))
        recomputed = evaluation_v3.compare_profiles_v3(reference, profile, omitted) if profile else None
        if recomputed is None or recomputed["core"] != recorded["core"]:
            raise Blocked(f"Historical {sample} grade does not recompute from its stored judgment.")
        out[sample] = {"core": recorded["core"], "guitar_clear_in_both": recorded["guitar_clear_in_both"],
                       "gaps": _gaps(recorded)}
    return out


# The spent run evaluated these candidates; later Fine-tune additions (joint moves) are not part of this phase.
LOCAL_V1_VARIANT_IDS = ("current", "trim_minus6", "trim_minus3", "trim_plus3", "trim_plus6",
                        "eq_brighter", "eq_darker", "control")


def build(adapter, phase: Phase) -> list:
    reference = reference_profile(phase)
    variants = refinement.build_variants(adapter, base_project(phase).chain, reference, "local")
    return [item for item in variants if item.id in LOCAL_V1_VARIANT_IDS]


def materialize(phase: Phase, variants, adapter, renderer, freeze: str, judged_tags=None) -> dict:
    """Render every candidate with both DIs once, bind by hash, then only verify and reuse.

    ``judged_tags``: the phase's request tags whose reservation means the stored candidates were judged
    (default: this phase's relative steps); missing candidates are then never re-rendered.
    """
    judged_tags = [_relative_tag(*step) for step in RELATIVE_STEPS] if judged_tags is None else judged_tags
    directory = phase.root / "candidates"
    manifest_path = directory / "artifacts.json"
    expected = {item.id: item.chain_sha256 for item in variants}
    if manifest_path.exists():
        manifest = driver._load(manifest_path)
        if {key: value["chain_sha256"] for key, value in manifest["variants"].items()} != expected:
            raise Blocked("Stored candidates do not match the frozen candidate set; refusing to regrade.")
        if manifest["engine_version"] != renderer.engine_version:
            raise Blocked("Stored candidates were rendered by a different engine.")
        for vid, entry in manifest["variants"].items():
            if _file_sha(directory / entry["project"]) != entry["project_sha256"]:
                raise Blocked(f"Stored project for {vid} changed.")
            for sample, render in entry["renders"].items():
                if _file_sha(directory / render["wav"]) != render["file_sha256"]:
                    raise Blocked(f"Stored audio for {vid}/{sample} changed; refusing to regrade.")
        _require_current_is_baseline(phase, directory, manifest)
        return manifest
    if any(phase.tag_used(tag) for tag in judged_tags):
        raise Blocked("Judged candidates are missing; refusing to grade replacement audio.")
    if directory.exists():
        shutil.rmtree(directory)  # never judged: no relative step was ever reserved
    partial = phase.root / "candidates.partial"
    if partial.exists():
        shutil.rmtree(partial)
    (partial / "renders").mkdir(parents=True)
    (partial / "variants").mkdir()
    project = base_project(phase)
    manifest = {"case": CASE, "freeze_sha256_at_render": freeze, "engine_version": renderer.engine_version,
                "created_at": _now(), "variants": {}}
    for variant in variants:
        candidate = final._variant_project(project, variant.chain)
        project_rel = f"variants/{variant.id}.project.json"
        driver._save(partial / project_rel, candidate.model_dump(mode="json"))
        entry = {"role": variant.role, "label": variant.label, "description": variant.description,
                 "chain_sha256": variant.chain_sha256, "project": project_rel,
                 "project_sha256": _file_sha(partial / project_rel),
                 "assets": [{"node": node.id, **node.asset.model_dump(mode="json")}
                            for node in variant.chain.nodes if node.asset is not None],
                 "omitted_optional": final._omitted(variant.chain), "renders": {}}
        for sample in SAMPLES:
            try:
                rendered = renderer.render(candidate, sample, partial / "renders")
            except Exception as error:
                raise Blocked(f"Local render failed for {variant.id}/{sample}: {type(error).__name__}") from error
            wav = partial / "renders" / f"{variant.id}-{sample}.wav"
            os.replace(partial / "renders" / f"{rendered.snapshot.id}.wav", wav)
            digest = _file_sha(wav)
            if digest != rendered.snapshot.audio_sha256:
                raise Blocked(f"Rendered file hash differs from its snapshot for {variant.id}/{sample}.")
            samples = rendered.samples.reshape(len(rendered.samples), -1)
            entry["renders"][sample] = {
                "wav": f"renders/{variant.id}-{sample}.wav", "file_sha256": digest,
                "snapshot": rendered.snapshot.model_dump(mode="json"),
                "output_peak_all_channels": float(np.max(np.abs(samples))),
                "samples_over_full_scale_all_channels": int(np.sum(np.abs(samples) > 1.0))}
        manifest["variants"][variant.id] = entry
    manifest["current_matches_historical_baseline_pcm"] = _current_matches(phase, partial, manifest)
    _require_current_is_baseline(phase, partial, manifest)
    os.replace(partial, directory)
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    driver._save(manifest_path, manifest)
    return manifest


def _current_matches(phase: Phase, directory: Path, manifest: dict) -> dict:
    """Per DI: the re-rendered current chain equals the historical baseline (rate, channels and PCM)."""
    out = {}
    for sample in SAMPLES:
        new, rate = sf.read(directory / manifest["variants"]["current"]["renders"][sample]["wav"], dtype="float32",
                            always_2d=True)
        old, old_rate = sf.read(_input(phase, f"baseline_wav_{sample}"), dtype="float32", always_2d=True)
        out[sample] = bool(rate == old_rate and new.shape == old.shape and np.array_equal(new, old))
    return out


def _require_current_is_baseline(phase: Phase, directory: Path, manifest: dict) -> None:
    matches = _current_matches(phase, directory, manifest)
    if not all(matches.values()):
        raise Blocked("Re-rendered current chain does not reproduce the historical baseline audio "
                      f"(rate/channels/PCM) on every DI {matches}; refusing before any request.")


def _clip(path: Path, source: str) -> AudioClip:
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    return AudioClip(samples=samples.T, sample_rate=rate, source=source, start_seconds=0)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def prepare_matched(phase: Phase, directory_name: str, sources: dict, header: dict,
                    samples: tuple = SAMPLES) -> dict:
    """Matched upload copies: 'baseline' at its prepared (16 kHz mono) level; every other role scaled by one scalar
    to the same DI's prepared baseline RMS (per DI in ``samples``). ``sources``: {role: callable(sample) -> raw WAV path}.

    A copy over full scale, a silent clip or a copy the listener's own preparation would change blocks the
    preparation (``record['blocked']``) before any request. Once written, the record is only verified.
    """
    directory = phase.root / directory_name
    record_path = directory / "preparation.json"
    roles = list(sources)
    if record_path.exists():
        record = driver._load(record_path)
        if {key: value for key, value in record.items() if key in header} != header:
            raise Blocked("Stored upload preparation is for a different candidate set; refusing.")
        for sample in samples:
            for role in roles:
                entry = record["samples"][sample][role]
                if entry.get("upload_file") and _file_sha(directory / entry["upload_file"]) != \
                        entry["upload_file_sha256"]:
                    raise Blocked(f"Stored upload copy {role}/{sample} changed; refusing.")
        return record
    directory.mkdir(exist_ok=True)
    record = {**header, "treatment": "declared new-phase comparison treatment; not a retrospective change",
              "label": "matched upload COPIES, not raw native renders", "samples": {}, "blocked": None}
    problems = []
    for sample in samples:
        prepared = {}
        rows = {}
        for role in roles:
            raw = sources[role](sample)
            clip = audio_io.prepare_upload(_clip(raw, f"{role}-{sample}"))
            prepared[role] = clip.samples[0]
            rows[role] = {"raw_file": os.path.relpath(raw, phase.root), "raw_file_sha256": _file_sha(raw),
                          "prepared_pcm_sha256": _pcm_sha(clip.samples[0]),
                          "prepared_rms_dbfs": float(20 * np.log10(max(_rms(clip.samples[0]), 1e-12))),
                          "prepared_peak": float(np.max(np.abs(clip.samples[0])))}
        base_rms = _rms(prepared["baseline"])
        silent = [role for role in roles if _rms(prepared[role]) <= 1e-9]
        if base_rms <= 1e-9 or silent:
            problems.append(f"{sample}: prepared audio is silent ({', '.join(silent)}); matching undefined")
            record["samples"][sample] = rows
            continue
        uploads = {}
        for role in roles:
            scalar = 1.0 if role == "baseline" else base_rms / _rms(prepared[role])
            uploads[role] = prepared[role] if role == "baseline" else prepared[role] * scalar
            rows[role]["scalar"] = float(scalar)
            if role != "baseline":
                rows[role]["scalar_db"] = float(20 * np.log10(scalar))
        for role, values in uploads.items():
            pcm = values.astype("<f4")
            peak = float(np.max(np.abs(pcm)))
            rows[role]["upload_peak"] = peak
            rows[role]["upload_rms_dbfs"] = float(20 * np.log10(_rms(pcm)))
            if peak > 1.0:
                problems.append(f"{sample}/{role}: matched copy peak {peak:.4f} exceeds full scale")
                continue
            again = audio_io.prepare_upload(AudioClip(pcm[None, :], UPLOAD_RATE, role, 0)).samples[0]
            if not np.array_equal(again.astype("<f4"), pcm):
                problems.append(f"{sample}/{role}: copy is not invariant under the listener's preparation")
                continue
            name = f"{role}-{sample}.upload.wav"
            sf.write(directory / name, pcm, UPLOAD_RATE, subtype="FLOAT")
            os.chmod(directory / name, 0o400)
            rows[role].update(upload_file=name, upload_file_sha256=_file_sha(directory / name),
                              upload_pcm_sha256=_pcm_sha(pcm))
        record["samples"][sample] = rows
    if problems:
        record["blocked"] = "Upload preparation blocked before any request: " + "; ".join(problems)
    driver._save(record_path, record)
    return record


def prepare_c(phase: Phase, candidate_id: str, manifest: dict) -> dict:
    """Prepare matched upload copies (baseline at its prepared level; candidate scaled to its RMS)."""
    sources = {"baseline": lambda sample: _input(phase, f"baseline_wav_{sample}"),
               "candidate": lambda sample: phase.root / "candidates" /
               manifest["variants"][candidate_id]["renders"][sample]["wav"]}
    header = {"candidate": candidate_id, "baseline": f"historical tone-copier-final-v1 {BASE_VARIANT} WAVs"}
    record = prepare_matched(phase, "c-inputs", sources, header)
    if record["blocked"]:
        record["blocked"] = record["blocked"].replace("Upload preparation blocked", "C preparation blocked", 1)
    return record


def _gaps(grade: dict | None) -> dict | None:
    if grade is None:
        return None
    axes = grade["core"]["axes"]
    gaps = {axis: axes[axis]["difference"] for axis in ("gain", "brightness", "mids")}
    known = None not in gaps.values()
    return {**gaps, "gain_plus_brightness": gaps["gain"] + gaps["brightness"] if known else None,
            "gain_brightness_mids": sum(gaps.values()) if known else None,
            "known_failure_count": len(grade["core"]["known_failures"])}


def beats_on_di(baseline: dict | None, candidate: dict | None) -> dict:
    if baseline is None or candidate is None:
        return {"beats": None, "reason": "a C judgment is unavailable"}
    if not (baseline["guitar_clear_in_both"] and candidate["guitar_clear_in_both"]):
        return {"beats": None, "reason": "guitar not clearly audible in both grades"}
    b, c = _gaps(baseline), _gaps(candidate)
    unknown = sorted({axis for grade in (baseline, candidate) for axis in evaluation_v3.CORE_AXES
                      if grade["core"]["axes"][axis]["difference"] is None})
    if unknown:
        return {"beats": None, "reason": f"incomplete core evidence (uncertain: {', '.join(unknown)})",
                "baseline": b, "candidate": c}
    checks = {
        "gain_gap_not_worse": c["gain"] <= b["gain"],
        "brightness_gap_not_worse": c["brightness"] <= b["brightness"],
        "gain_plus_brightness_strictly_smaller": c["gain_plus_brightness"] < b["gain_plus_brightness"],
        "known_failure_count_not_increased": c["known_failure_count"] <= b["known_failure_count"],
        "gain_brightness_mids_not_increased": c["gain_brightness_mids"] <= b["gain_brightness_mids"],
    }
    return {"beats": all(checks.values()), "checks": checks, "baseline": b, "candidate": c}


def axis_deltas(baseline: dict | None, candidate: dict | None) -> dict | None:
    if baseline is None or candidate is None:
        return None
    out = {}
    for axis in evaluation_v3.CORE_AXES:
        before, after = baseline["core"]["axes"][axis], candidate["core"]["axes"][axis]
        change = None if None in (before["difference"], after["difference"]) else \
            after["difference"] - before["difference"]
        out[axis] = {"reference": before["reference"], "baseline": before["candidate"],
                     "candidate": after["candidate"], "baseline_gap": before["difference"],
                     "candidate_gap": after["difference"], "gap_change": change}
    return out


def support(per_model: dict) -> dict:
    """Pre-registered C gate over per-model A/B suggestions."""
    complete = all(item["usable"] for item in per_model.values())
    supported = {model: item["supported"] for model, item in per_model.items()}
    union = sorted({value for value in supported.values() if value})
    if not complete:
        return {"run_c": False, "candidate": None, "supported_by_model": supported, "union": union,
                "reason": "A/B evidence unusable (a judgment failed, was invalid/not attempted, or its wrong-gain control did not "
                          "pass); C skipped"}
    if len(union) != 1:
        reason = ("no model supported a candidate" if not union else
                  "models supported different candidates; disagreement is not agreement and no model is preferred")
        return {"run_c": False, "candidate": None, "supported_by_model": supported, "union": union, "reason": reason}
    (candidate,) = union
    models = sorted(model for model, value in supported.items() if value == candidate)
    basis = "cross-model" if len(models) == len(per_model) else "one-model"
    return {"run_c": True, "candidate": candidate, "supported_by_model": supported, "union": union,
            "support_basis": basis, "supporting_models": models,
            "reason": (f"{basis} support ({', '.join(models)})" +
                       ("" if basis == "cross-model" else "; the other model supported no candidate"))}


def _draw_seed(identifiers: list[str], avoid: dict | None) -> int:
    while True:
        seed = secrets.randbelow(2**31)
        if avoid is None or refinement.relative_judge.blind_labels(identifiers, seed) != avoid:
            return seed


def _caller(phase: Phase, steps: dict, freeze: str, log: Callable[[str], None]) -> Callable[..., dict | None]:
    """One provider request per tag: reserve, send once, settle and record; never re-sent or retried.

    A recorded judgment is reused; a reserved-but-unrecorded tag, a paused phase (HTTP 429) or a reached ceiling
    returns None with an explicit step status.
    """
    judgments_dir = phase.root / "judgments"

    def call(tag: str, kind: str, model: str, send: Callable[[], dict], **detail) -> dict | None:
        recorded = judgments_dir / f"{tag}.json"
        if recorded.exists():
            result = driver._load(recorded)
            steps[tag] = {"status": final._status(result), "recorded_earlier": True}
            return result
        ledger = phase.ledger
        if phase.tag_used(tag, ledger):
            steps[tag] = {"status": "attempted_unrecorded", "note": "Reserved earlier; never re-sent."}
            return None
        if ledger["provider_pause"]:
            steps[tag] = {"status": "not_attempted", "reason": "phase paused after HTTP 429"}
            return None
        if len(ledger["attempts"]) >= ledger["ceiling"]:
            steps[tag] = {"status": "not_attempted", "reason": "phase ceiling reached"}
            return None
        entry = phase.reserve(kind, model, {"tag": tag, "freeze_sha256": freeze, **detail})
        try:
            result = send()
        except Exception as error:  # noqa: BLE001 - counted and recorded, never retried
            result = {"failure_info": {"category": "exception", "exception_type": type(error).__name__},
                      "request_counts": None}
        result.update(tag=tag, freeze_sha256=freeze, model_requested=model, request_detail=detail)
        phase.settle(entry, result)
        driver._record(phase, tag, result)
        steps[tag] = {"status": final._status(result), "http_status": entry.get("http_status")}
        log(f"{tag}: {steps[tag]['status']}")
        return result

    return call


def run(phase: Phase, adapter, renderer, providers, freeze: str, log: Callable[[str], None] = print) -> dict:
    report = {"phase": PHASE, "freeze_sha256": freeze, "case": CASE, "interval": list(INTERVAL),
              "provenance": config()["provenance"], "steps": {}, "limitations": list(LIMITATIONS)}
    steps = report["steps"]
    judgments_dir = phase.root / "judgments"

    call = _caller(phase, steps, freeze, log)

    def finish(outcome: str, reason: str | None = None) -> dict:
        for seed, model, sample in RELATIVE_STEPS:
            steps.setdefault(_relative_tag(seed, model, sample), {"status": "not_attempted", "reason": reason})
        for role, sample in C_STEPS:
            steps.setdefault(_c_tag(role, sample), {"status": "not_attempted", "reason": reason})
        report["outcome"] = outcome
        if reason:
            report["reason"] = reason
        return report

    try:
        if not (phase.root / "inputs" / "inputs.json").exists():
            raise Blocked("Imported final-phase inputs are missing; run `freeze --final-dir` first.")
        import_inputs(phase, None)  # verify-only once the manifest exists
        reference = reference_profile(phase)
        report["historical_baseline"] = historical(phase, reference)
        variants = build(adapter, phase)
        manifest = materialize(phase, variants, adapter, renderer, freeze)
    except Blocked as error:
        return finish("blocked", str(error))
    report["candidates"] = {"manifest": "candidates/artifacts.json",
                            "manifest_sha256": _file_sha(phase.root / "candidates" / "artifacts.json"),
                            "current_matches_historical_baseline_pcm":
                                manifest["current_matches_historical_baseline_pcm"],
                            "variants": {vid: {"role": entry["role"], "label": entry["label"],
                                               "chain_sha256": entry["chain_sha256"]}
                                         for vid, entry in manifest["variants"].items()}}
    identifiers = sorted(manifest["variants"])
    clips = {sample: {vid: _clip(phase.root / "candidates" / entry["renders"][sample]["wav"], vid)
                      for vid, entry in manifest["variants"].items()} for sample in SAMPLES}
    hot = refinement.control_is_hotter(reference)
    summaries: dict[tuple[str, str], list] = {}
    for seed_set, model, sample in RELATIVE_STEPS:
        tag = _relative_tag(seed_set, model, sample)
        earlier = judgments_dir / f"{_relative_tag('A', model, sample)}.json"
        avoid = driver._load(earlier).get("label_mapping") if seed_set == "B" and earlier.exists() else None
        seed = _draw_seed(identifiers, avoid)
        result = call(tag, "relative", model,
                      lambda model=model, sample=sample, seed=seed: providers.relative(
                          model=model, candidates=clips[sample], seed=seed, youtube=(URL, *INTERVAL)),
                      sample=sample, seed_set=seed_set, seed=seed)
        summary = (refinement.summarize_judgment(sample, result, hot) if result else
                   refinement.SampleJudgment(sample=sample, status="not_attempted", reason=steps[tag]["status"]))
        summaries.setdefault((seed_set, model), []).append(summary)
    per_model = {}
    for model in RELATIVE_MODELS:
        sets = {}
        for seed_set in SEEDS:
            judged = summaries[(seed_set, model)]
            suggestion = refinement.suggest_from(judged)
            sets[seed_set] = {"suggested": suggestion.variant_id, "basis": suggestion.basis,
                              "judgments": [item.model_dump(mode="json") for item in judged]}
        usable = all(item["status"] == "judged" and item["control_passed"] is True
                     for seed_set in SEEDS for item in sets[seed_set]["judgments"])
        a, b = sets["A"]["suggested"], sets["B"]["suggested"]
        per_model[model] = {"usable": usable, "seed_sets": sets,
                            "supported": a if usable and a is not None and a == b and a != "current" else None}
    gate = support(per_model)
    report["relative"] = {"per_model": per_model, "gate": gate,
                          "claim": "relative support only; not a categorical or perceptual-match claim"}
    if not gate["run_c"]:
        return finish("c_skipped", gate["reason"])
    candidate = gate["candidate"]
    try:
        preparation = prepare_c(phase, candidate, manifest)
    except Blocked as error:
        return finish("blocked", str(error))
    report["c_preparation"] = {"record": "c-inputs/preparation.json",
                               "record_sha256": _file_sha(phase.root / "c-inputs" / "preparation.json"),
                               **preparation}
    if preparation["blocked"]:
        return finish("c_preparation_blocked", preparation["blocked"])
    omitted = final._omitted(base_project(phase).chain)
    grades: dict[str, dict] = {}
    for role, sample in C_STEPS:
        entry = preparation["samples"][sample][role]
        upload = _clip(phase.root / "c-inputs" / entry["upload_file"], f"{role}-{sample}")
        result = call(_c_tag(role, sample), "absolute_profile", PROFILE_MODEL,
                      lambda upload=upload: providers.profile(upload, PROFILE_MODEL),
                      role=role, sample=sample, candidate=candidate if role == "candidate" else BASE_VARIANT,
                      upload_pcm_sha256=entry["upload_pcm_sha256"])
        profile = final._profile(result)
        grades.setdefault(sample, {})[role] = (evaluation_v3.compare_profiles_v3(reference, profile, omitted)
                                               if profile else None)
    per_di, reproduction = {}, {}
    for sample in SAMPLES:
        baseline, cand = grades[sample].get("baseline"), grades[sample].get("candidate")
        history = report["historical_baseline"][sample]["gaps"]
        reproduced = None if baseline is None or not baseline["guitar_clear_in_both"] else (
            _gaps(baseline)["gain"] == history["gain"] and _gaps(baseline)["brightness"] == history["brightness"])
        reproduction[sample] = {"historical_gaps": history, "reevaluated_gaps": _gaps(baseline),
                             "reproduced": reproduced}
        per_di[sample] = {
            "baseline": {"grade_v3": baseline, "core_decision": final.core_decision(baseline)},
            "candidate": {"grade_v3": cand, "core_decision": final.core_decision(cand)},
            "axis_deltas": axis_deltas(baseline, cand), "beats": beats_on_di(baseline, cand)}
    verdicts = [per_di[sample]["beats"]["beats"] for sample in SAMPLES]
    beats = False if False in verdicts else True if all(verdicts) else None
    reproduced = all(item["reproduced"] is True for item in reproduction.values())
    report["c"] = {
        "candidate": candidate, "support_basis": gate["support_basis"], "per_di": per_di,
        "baseline_reproduction": reproduction, "baseline_gaps_reproduced": reproduced,
        "candidate_profiles_repeated": False,
        "beats_baseline_categorically": beats,
        "aggregate_core": {role: evaluation_v3.aggregate_v3(
            [{"outcome": per_di[sample][role]["core_decision"]["outcome"]} for sample in SAMPLES])
            for role in ("baseline", "candidate")},
        "interpretation": (
            "Categorical comparison under the frozen v3 rubric only. " +
            ("The baseline recheck reproduced the historical gain/brightness gaps on both DIs. " if reproduced else
             "The baseline recheck did NOT reproduce the historical gain/brightness gaps (or was unavailable): "
             "baseline evaluator variability is visible; this does not show that there was no real improvement. ") +
            "Candidate profiles were judged once (not repeated), so candidate-judgment noise is not excluded."),
        "perceptual_match_claim": False, "effects": "reported separately; not validated"}
    if beats is None:
        outcome = "c_inconclusive"
    else:
        outcome = ("candidate_beats" if beats else "candidate_does_not_beat") + (
            "_baseline_reproduced" if reproduced else "_baseline_not_reproduced")
    return finish(outcome)


def export_candidates(phase: Phase, manifest: dict, adapter) -> dict:
    """Export each non-control candidate's preset (references the local library; nothing bundled)."""
    out = {}
    for vid, entry in manifest["variants"].items():
        if entry["role"] == "control":
            continue
        out[vid] = final.export_selected(phase.root / "candidates", manifest, vid, adapter)
    return out


def cmd_freeze(args, adapter=None, renderer=None) -> str:
    from tone_listener_mcp.workbench.renderer import NativeRenderer

    phase = Phase(args.output, create=True)
    try:
        import_inputs(phase, args.final_dir)
    except Blocked as error:
        raise SystemExit(str(error)) from error
    adapter = adapter or soundshed_module.SoundshedAdapter()
    renderer = renderer or NativeRenderer(adapter)
    if hasattr(renderer, "initialize"):
        renderer.initialize()
    reference = reference_profile(phase)
    historical(phase, reference)
    variants = {item.id: item.chain_sha256 for item in build(adapter, phase)}
    digest = phase.freeze_phase(renderer, variants, revise=getattr(args, "revise", False),
                                reason=getattr(args, "reason", None))
    print(digest)
    return digest


def cmd_plan(args, adapter=None) -> dict:
    phase = Phase(args.output)
    adapter = adapter or soundshed_module.SoundshedAdapter()
    variants = build(adapter, phase)
    ledger = phase.ledger
    record = phase._freeze_record()
    plan = {
        "freeze_sha256": phase.current_freeze(),
        "freeze_history": [{"sha256": item["sha256"], "status": item["status"], "reason": item.get("reason")}
                           for item in record["history"]],
        "ledger": {"ceiling": ledger["ceiling"], "attempted": len(ledger["attempts"]),
                   "provider_pause": ledger["provider_pause"]},
        **config(),
        "candidate_set": [{"id": item.id, "role": item.role, "label": item.label, "description": item.description,
                           "chain_sha256": item.chain_sha256} for item in variants],
        "uploads_per_relative_call": len(variants), "uploads_per_profile_call": 1,
        "max_new_generate_requests": {"relative_A_B": len(RELATIVE_STEPS), "absolute_C_if_gated": len(C_STEPS),
                                      "total": len(RELATIVE_STEPS) + len(C_STEPS)},
        "historical_baseline_gaps": {sample: value["gaps"] for sample, value in
                                     historical(phase, reference_profile(phase)).items()},
        "command": "python3 scripts/tone_accuracy_local.py --output <phase-dir> run --live",
    }
    driver._save(phase.root / "plan.json", plan)
    print(json.dumps(plan, indent=1))
    return plan


def cmd_run(args, adapter=None, renderer=None, providers=None) -> dict:
    if providers is None:
        driver._require_live(args)
    phase = Phase(args.output)
    if phase.current_freeze() is None:
        raise SystemExit("Run `freeze` and review `plan` first.")
    # One run at a time per phase: materialization, grading, exports and report persistence are serialized;
    # a later run reuses recorded judgments instead of overlapping.
    with final._flock(phase.state / ".run.lock"):
        return _run_locked(phase, adapter, renderer, providers)


def _run_locked(phase: Phase, adapter, renderer, providers) -> dict:
    from tone_listener_mcp.workbench.renderer import NativeRenderer

    adapter = adapter or soundshed_module.SoundshedAdapter()
    renderer = renderer or NativeRenderer(adapter)
    if hasattr(renderer, "initialize"):
        renderer.initialize()
    variants = {item.id: item.chain_sha256 for item in build(adapter, phase)}
    freeze = phase.freeze_phase(renderer, variants)
    report = run(phase, adapter, renderer, providers or final.Providers(), freeze)
    manifest_path = phase.root / "candidates" / "artifacts.json"
    if manifest_path.exists():
        report["exports"] = export_candidates(phase, driver._load(manifest_path), adapter)
    ledger = phase.ledger
    report["ledger"] = {"ceiling": ledger["ceiling"], "attempted": len(ledger["attempts"]),
                        "by_model": {model: sum(item["model"] == model for item in ledger["attempts"])
                                     for model in sorted({PROFILE_MODEL, *RELATIVE_MODELS})},
                        "failed": sum(item["status"] == "failed" for item in ledger["attempts"]),
                        "unsettled": sum(item["status"] == "sent" for item in ledger["attempts"]),
                        "provider_pause": ledger["provider_pause"]}
    reports = phase.root / "reports"
    reports.mkdir(exist_ok=True)
    driver._save(reports / f"report-{len(list(reports.iterdir())) + 1:02d}.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--final-dir", type=Path, required=True,
                        help="tone-copier-final-v1 output directory (read only).")
    freeze.add_argument("--revise", action="store_true", help="Supersede an unrun freeze (zero reservations only).")
    freeze.add_argument("--reason")
    sub.add_parser("plan")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    {"freeze": cmd_freeze, "plan": cmd_plan, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

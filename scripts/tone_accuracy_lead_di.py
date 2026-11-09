#!/usr/bin/env python3
"""Bounded DI-material diagnostic for case-01 on two lead-register DIs (development data only).

Phase ``tone-accuracy-lead-di-v1`` has its OWN registry entry, ledger, lock and freeze history (ceiling 6 generation
attempts including failures; no retries; first HTTP 429 pauses the phase). It reuses the ledger/lock/preparation
mechanisms of ``tone-accuracy-local-v1`` and never writes ``tone-copier-final-v1``, ``tone-accuracy-local-v1`` or
``tone-accuracy-single-v1``; their files are copied in read-only and bound by hash.

Pre-registered HYPOTHESIS (not an established fact): case-01's categorical "fail" of the selected
``pushed_clean_amp`` chain is largely a DI-material confound (the bundled riffs are chordal; the reference is a lead
line). The chain is unchanged; only the dry input differs (Guitar-TECHS lead-1/lead-2, CC BY 4.0).

Judge: the ORIGINAL single-clip neutral ToneProfile listener (gemini-3.6-flash, unchanged prompt/schema) graded with
the unchanged v3 rubric against the REUSED final-phase case-01 reference profile. No relative calls, no new reference.

Steps (6 calls maximum; nothing is retried or substituted):
  1. baseline        x lead-1, lead-2
  2. control         x lead-1, lead-2   (stop at the first control that is not rejected as hotter: invalid_control)
  3. baseline_repeat x lead-1, lead-2   (skipped when step 1 already excludes both positive outcomes)

Outcomes (fixed before any call): invalid_control, material_confound_supported, partial_material_effect,
no_material_effect. None of them is a tone improvement, causal proof, six-case acceptance or a recommendation.

Subcommands: ``freeze --final-dir DIR --single-report FILE`` (local renders/preparation, no requests),
``plan`` (no requests), ``run --live`` (paid; only after explicit approval).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tone_accuracy_local as local
import tone_accuracy_single as single
import tone_copier_batch as driver
import tone_copier_final as final

from tone_listener_mcp.workbench import evaluation_v3, refinement
from tone_listener_mcp.workbench.adapters import soundshed as soundshed_module

PHASE = "tone-accuracy-lead-di-v1"
CEILING = 6
CASE, BASE_VARIANT, INTERVAL = local.CASE, local.BASE_VARIANT, local.INTERVAL
SAMPLES = ("lead-1", "lead-2")
RIFF_SAMPLES = local.SAMPLES
PROFILE_MODEL = final.PROFILE_MODEL
VARIANT_IDS = ("current", "control")
UPLOAD_ROLES = ("baseline", "control")
STEP_TAGS = (*(f"baseline:{s}" for s in SAMPLES), *(f"control:{s}" for s in SAMPLES),
             *(f"baseline_repeat:{s}" for s in SAMPLES))
assert len(STEP_TAGS) == CEILING
OUTCOMES = ("invalid_control", "material_confound_supported", "partial_material_effect", "no_material_effect")
SINGLE_V1_REPORT_SHA256 = "84190b5d674c04d4d02bcab9fc3043c0d4fd7e5ed897dea165236c355e06eed0"
DI_DIR = Path(refinement.__file__).resolve().parent / "assets" / "di"
HYPOTHESIS = ("case-01's categorical 'fail' of the selected pushed_clean_amp chain is largely a DI-material "
              "confound (chordal riffs vs a lead-line reference). This is a hypothesis under test, not an "
              "established cause; DSP feature spreads alone do not establish it.")
SAFEGUARDS = (
    "No outcome is a tone improvement: the chain is unchanged and only the dry input differs.",
    ("material_confound_supported means only that the frozen baseline's categorical score is performance/material "
     "dependent; it is not causal proof that the original failure was wrong and not six-case acceptance."),
    ("The original riff failures (tone-copier-final-v1 and tone-accuracy-single-v1) remain the recorded results; "
     "they are shown alongside and are not replaced, relabelled or superseded."),
    "No recommendation, automatic adoption or replacement benchmark is produced by any outcome.",
    ("Lead DIs differ from the riffs in guitar (Sire T7 vs the riff guitar), player, interface and material; a "
     "difference cannot be attributed to register/material alone."),
    ("Guitar-TECHS publishes the pickup as 'Bridge-Neck' for the whole player-03 session; the pickup used for each "
     "excerpt is not verified."),
)
LIMITATIONS = (
    ("Case-01 is development data (it shaped pushed_clean_amp and these rules); the lead DIs were chosen after "
     "seeing the riff results; nothing here is held out."),
    "The case-01 reference profile is reused from tone-copier-final-v1; it is not a fresh or independent reference.",
    "Every outcome is one model's categorical judgment, not a listener verdict; no result is a perceptual match.",
    ("Uploads are matched COPIES (16 kHz mono; the control scaled to the same-DI baseline RMS); they are not raw "
     "native renders, and equal RMS is not equal loudness. Native WAVs are unchanged."),
    ("Each baseline is judged twice per lead DI (initial + one repeat); two judgments do not characterise the "
     "judge's variability, and the control is judged once."),
    "Ambience/effects are reported separately and are not validated.",
)


def config() -> dict:
    return {
        "phase": PHASE, "ceiling_generate_attempts_including_failures": CEILING, "case": CASE,
        "youtube_id": local.VIDEO, "interval": list(INTERVAL), "hypothesis": HYPOTHESIS,
        "provenance": ("DEVELOPMENT data, reused evidence: the final phase's case-01 reference profile and selected "
                       "chain are reused, not re-listened or re-selected."),
        "base": f"tone-copier-final-v1 {CASE} selected variant '{BASE_VARIANT}' (read-only copy, bound by hash)",
        "dis": {"samples": list(SAMPLES), "source": "Guitar-TECHS player 03 (CC BY 4.0); see assets/di/provenance.json"},
        "model": PROFILE_MODEL,
        "judge": "final.Providers.profile: original neutral ToneProfile prompt/schema (unchanged); no relative calls",
        "rubric": "evaluation_v3.compare_profiles_v3(reused reference, judged copy, omitted effects); thresholds "
                  "unchanged", "thresholds": evaluation_v3.RULES,
        "variants": {"baseline": "current = the unchanged pushed_clean_amp chain rendered on each lead DI",
                     "control": "refinement wrong-gain control (refinement.build_variants(search='local'))"},
        "upload_preparation": (
            "audio_io.prepare_upload (mono mean, 16 kHz). Baseline = this phase's native lead render of the unchanged "
            "chain at its prepared level (no rescaling; there is no historical lead render). Control scaled by one "
            "scalar to the same DI's prepared baseline RMS. Copies must be within full scale, non-silent and "
            "invariant under the listener's preparation, else the phase is blocked before any request. The repeat "
            "re-sends the baseline's identical upload copy as a fresh request."),
        "riff_comparison": {
            "source": f"tone-accuracy-single-v1 report (sha256 {SINGLE_V1_REPORT_SHA256}), read-only copy",
            "value": "baseline gain+brightness+mids distance per riff (re-profiled in single-v1)",
            "threshold": "the MINIMUM of the two riff values; a lead value must be strictly below it (below both)"},
        "steps": [
            {"step": 1, "tags": [f"baseline:{s}" for s in SAMPLES], "calls": 2,
             "rule": "both are always judged (no early stop between them)"},
            {"step": 2, "tags": [f"control:{s}" for s in SAMPLES], "calls": 2,
             "gate": ("final.control_rejection: guitar clear in both and gain ordinal gap > "
                      f"{evaluation_v3.RULES['maximum_axis_difference']} in the intended (hotter) direction, per DI. "
                      "Missing/uncertain control evidence is not a rejection. The first control not rejected "
                      "ends the phase as invalid_control.")},
            {"step": 3, "tags": [f"baseline_repeat:{s}" for s in SAMPLES], "calls": 2,
             "rule": ("judged only if the controls were rejected on both lead DIs AND step 1 leaves a positive "
                      "outcome possible (every lead DI has complete core evidence with a core pass or a "
                      "gain+brightness+mids distance strictly below the riff threshold); otherwise not spent")},
        ],
        "outcomes": {
            "invalid_control": "a control is not rejected as hotter on either lead DI (including missing evidence)",
            "material_confound_supported": ("baseline core-passes on BOTH lead DIs, initially AND on repeat "
                                            "(guitar clear, every core axis known)"),
            "partial_material_effect": ("not material_confound_supported, but baseline gain+brightness+mids is "
                                        "strictly below the single-v1 riff threshold on BOTH lead DIs, initially "
                                        "AND on repeat, with complete core evidence; all raw axis gaps reported"),
            "no_material_effect": ("otherwise, including missing/incomplete baseline evidence (reported as "
                                   "evidence_complete=false, never as a pass)"),
            "procedural": ("paused (HTTP 429), stopped (reserved tag unrecorded), blocked/preparation_blocked are "
                           "not classifications"),
        },
        "stop_rules": [
            "Any HTTP 429 pauses the phase; remaining steps are not_attempted; nothing is retried or substituted.",
            "A reserved tag is never re-sent (a crash after reservation counts as attempted).",
            "Stored renders/upload copies that differ from their hashes block the phase instead of being regraded.",
        ],
        "safeguards": list(SAFEGUARDS),
        "limitations": list(LIMITATIONS),
    }


def _di_identity() -> dict:
    return {name: local._file_sha(DI_DIR / name) for name in ("provenance.json", *(f"{s}.wav" for s in SAMPLES))}


def identity(renderer, root: Path, variants: dict) -> dict:
    files = {"lead_di_runner": __file__, "single_phase_helpers": inspect.getfile(single),
             "local_phase_helpers": inspect.getfile(local), "final_runner_helpers": inspect.getfile(final),
             "refinement_selection": inspect.getfile(refinement), "evaluation_v3": inspect.getfile(evaluation_v3),
             "soundshed_adapter_recipes": inspect.getfile(soundshed_module)}
    value = driver.frozen_identity(None)
    value.update({name: local._file_sha(Path(path)) for name, path in files.items()})
    value.update({"engine": getattr(renderer, "engine_version", None),
                  "worker_sha256": getattr(renderer, "_worker_sha256", None),
                  "plugin_sha256": getattr(renderer, "_plugin_sha256", None),
                  "inputs": {**local._inputs_identity(root), "single_v1_report": _single_report_sha(root)},
                  "dis": _di_identity(), "variant_chain_sha256": variants, "config": config()})
    return value


class Phase(local.Phase):
    phase_id = PHASE
    ceiling = CEILING

    def identity(self, renderer, variants: dict) -> dict:
        return identity(renderer, self.root, variants)


Blocked = local.Blocked


def _single_report_path(root: Path) -> Path:
    return root / "inputs" / "single-v1-report.json"


def _single_report_sha(root: Path) -> str | None:
    path = _single_report_path(root)
    return local._file_sha(path) if path.exists() else None


def import_single_report(phase: Phase, source: Path | None) -> dict:
    """Copy the accepted single-v1 report (read-only) once; afterwards only verify the copy."""
    target = _single_report_path(phase.root)
    if not target.exists():
        if source is None:
            raise Blocked("The single-v1 report has not been imported; run `freeze --single-report` first.")
        data = Path(source).read_bytes()
        if local._sha(data) != SINGLE_V1_REPORT_SHA256:
            raise Blocked("The supplied single-v1 report is not the accepted report (sha256 mismatch).")
        target.write_bytes(data)
        os.chmod(target, 0o400)
    if local._file_sha(target) != SINGLE_V1_REPORT_SHA256:
        raise Blocked("Imported single-v1 report changed; refusing.")
    return driver._load(target)


def riff_comparison(phase: Phase) -> dict:
    report = import_single_report(phase, None)
    per_riff = {}
    for sample in RIFF_SAMPLES:
        entry = report["baseline"][sample]
        gaps = local._gaps(entry["grade_v3"])
        if gaps is None or gaps["gain_brightness_mids"] is None:
            raise Blocked(f"single-v1 baseline {sample} has no complete gain+brightness+mids distance.")
        per_riff[sample] = {"gaps": gaps, "raw_axes": raw_axes(entry["grade_v3"]),
                            "core_decision": entry["core_decision"]}
    return {"source": f"tone-accuracy-single-v1 report sha256 {SINGLE_V1_REPORT_SHA256} (outcome "
                      f"{report['outcome']})", "per_riff": per_riff,
            "threshold_gain_brightness_mids": min(v["gaps"]["gain_brightness_mids"] for v in per_riff.values()),
            "note": "Riff failures stay the recorded results; this phase does not replace them."}


def raw_axes(grade: dict | None) -> dict | None:
    if grade is None:
        return None
    return {axis: {key: grade["core"]["axes"][axis][key] for key in ("reference", "candidate", "difference")}
            for axis in evaluation_v3.CORE_AXES}


def build(adapter, phase: Phase) -> list:
    reference = local.reference_profile(phase)
    chain = local.base_project(phase).chain
    variants = refinement.build_variants(adapter, chain, reference, "local")
    chosen = [item for item in variants if item.id in VARIANT_IDS]
    if tuple(sorted(item.id for item in chosen)) != tuple(sorted(VARIANT_IDS)):
        raise Blocked(f"Baseline/control are unavailable on the base chain: {[item.id for item in chosen]}")
    current = next(item for item in chosen if item.id == "current")
    if current.chain_sha256 != adapter.chain_sha256(chain):
        raise Blocked("The baseline is not the unchanged final-phase chain; refusing.")
    return chosen


def materialize(phase: Phase, variants, renderer, freeze: str) -> dict:
    """Render baseline and control on both lead DIs once, bind by hash, then only verify and reuse."""
    directory = phase.root / "candidates"
    manifest_path = directory / "artifacts.json"
    expected = {item.id: item.chain_sha256 for item in variants}
    if manifest_path.exists():
        manifest = driver._load(manifest_path)
        if {key: value["chain_sha256"] for key, value in manifest["variants"].items()} != expected:
            raise Blocked("Stored renders do not match the frozen baseline/control; refusing to regrade.")
        if manifest["engine_version"] != renderer.engine_version:
            raise Blocked("Stored renders were made by a different engine.")
        if manifest["dis"] != _di_identity():
            raise Blocked("Bundled lead DI files changed since rendering; refusing.")
        for vid, entry in manifest["variants"].items():
            if local._file_sha(directory / entry["project"]) != entry["project_sha256"]:
                raise Blocked(f"Stored project for {vid} changed.")
            for sample, render in entry["renders"].items():
                if local._file_sha(directory / render["wav"]) != render["file_sha256"]:
                    raise Blocked(f"Stored audio for {vid}/{sample} changed; refusing to regrade.")
        return manifest
    if any(phase.tag_used(tag) for tag in STEP_TAGS):
        raise Blocked("Judged renders are missing; refusing to grade replacement audio.")
    if directory.exists():
        shutil.rmtree(directory)  # never judged: no step was ever reserved
    partial = phase.root / "candidates.partial"
    if partial.exists():
        shutil.rmtree(partial)
    (partial / "renders").mkdir(parents=True)
    (partial / "variants").mkdir()
    project = local.base_project(phase)
    manifest = {"case": CASE, "freeze_sha256_at_render": freeze, "engine_version": renderer.engine_version,
                "dis": _di_identity(), "created_at": local._now(), "variants": {}}
    for variant in variants:
        candidate = final._variant_project(project, variant.chain)
        project_rel = f"variants/{variant.id}.project.json"
        driver._save(partial / project_rel, candidate.model_dump(mode="json"))
        entry = {"role": variant.role, "label": variant.label, "description": variant.description,
                 "chain_sha256": variant.chain_sha256, "project": project_rel,
                 "project_sha256": local._file_sha(partial / project_rel),
                 "omitted_optional": final._omitted(variant.chain), "renders": {}}
        for sample in SAMPLES:
            try:
                rendered = renderer.render(candidate, sample, partial / "renders")
            except Exception as error:
                raise Blocked(f"Local render failed for {variant.id}/{sample}: {type(error).__name__}") from error
            wav = partial / "renders" / f"{variant.id}-{sample}.wav"
            os.replace(partial / "renders" / f"{rendered.snapshot.id}.wav", wav)
            digest = local._file_sha(wav)
            if digest != rendered.snapshot.audio_sha256:
                raise Blocked(f"Rendered file hash differs from its snapshot for {variant.id}/{sample}.")
            samples = rendered.samples.reshape(len(rendered.samples), -1)
            if not np.all(np.isfinite(samples)):
                raise Blocked(f"Non-finite render for {variant.id}/{sample}; refusing.")
            entry["renders"][sample] = {
                "wav": f"renders/{variant.id}-{sample}.wav", "file_sha256": digest,
                "snapshot": rendered.snapshot.model_dump(mode="json"),
                "output_peak_all_channels": float(np.max(np.abs(samples))),
                "samples_over_full_scale_all_channels": int(np.sum(np.abs(samples) > 1.0))}
        manifest["variants"][variant.id] = entry
    os.replace(partial, directory)
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    driver._save(manifest_path, manifest)
    return manifest


def prepare(phase: Phase, manifest: dict) -> dict:
    def source(role: str):
        vid = "current" if role == "baseline" else role
        return lambda sample: phase.root / "candidates" / manifest["variants"][vid]["renders"][sample]["wav"]

    header = {"roles": list(UPLOAD_ROLES),
              "baseline": "this phase's native lead render of the unchanged pushed_clean_amp chain (no rescaling)"}
    return local.prepare_matched(phase, "uploads", {role: source(role) for role in UPLOAD_ROLES}, header,
                                 samples=SAMPLES)


def materialize_and_prepare(phase: Phase, adapter, renderer, freeze: str) -> tuple[dict, dict]:
    variants = build(adapter, phase)
    used = [tag for tag in STEP_TAGS if phase.tag_used(tag)]
    if used and not (phase.root / "uploads" / "preparation.json").exists():
        raise Blocked("Judged upload copies are missing; refusing to prepare replacements.")
    manifest = materialize(phase, variants, renderer, freeze)
    preparation = prepare(phase, manifest)
    binding = single._material_binding(manifest, preparation)
    path = phase.state / "materialized.json"
    if path.exists():
        if driver._load(path)["binding"] != binding:
            raise Blocked("Rendered audio/upload copies differ from those bound before any request; refusing.")
    else:
        if used:
            raise Blocked("Material binding is missing after requests were reserved; refusing (fail closed).")
        driver._save(path, {"binding": binding, "bound_at": local._now(), "freeze_sha256": freeze})
    return manifest, preparation


def baseline_evidence(grade: dict | None, threshold: int) -> dict:
    """Per-judgment baseline evidence; incomplete evidence can never support a positive outcome."""
    decision = final.core_decision(grade)
    complete = grade is not None and grade["guitar_clear_in_both"] and single._core_complete(grade)
    gaps = local._gaps(grade)
    gbm = gaps["gain_brightness_mids"] if gaps else None
    return {"grade_v3": grade, "core_decision": decision, "raw_axes": raw_axes(grade), "gaps": gaps,
            "evidence_complete": complete,
            "core_pass": complete and decision["outcome"] == "pass",
            "gain_brightness_mids_below_riff_threshold": (gbm < threshold) if complete and gbm is not None else None}


def classify(initial: dict, repeat: dict | None) -> dict:
    """Pre-registered classification given valid controls; ``repeat`` is None when repeats were not spent."""
    judgments = [initial[s] for s in SAMPLES] + ([repeat[s] for s in SAMPLES] if repeat else [])
    complete = repeat is not None and all(item["evidence_complete"] for item in judgments)
    if complete and all(item["core_pass"] for item in judgments):
        return {"outcome": "material_confound_supported", "evidence_complete": True}
    if complete and all(item["gain_brightness_mids_below_riff_threshold"] is True for item in judgments):
        return {"outcome": "partial_material_effect", "evidence_complete": True}
    return {"outcome": "no_material_effect",
            "evidence_complete": all(item["evidence_complete"] for item in judgments) and repeat is not None}


def positive_possible(initial: dict) -> bool:
    return all(item["evidence_complete"] and (item["core_pass"] or
                                              item["gain_brightness_mids_below_riff_threshold"] is True)
               for item in (initial[s] for s in SAMPLES))


def run(phase: Phase, adapter, renderer, providers, freeze: str, log: Callable[[str], None] = print) -> dict:
    report = {"phase": PHASE, "freeze_sha256": freeze, "case": CASE, "interval": list(INTERVAL),
              "model": PROFILE_MODEL, "hypothesis": HYPOTHESIS, "provenance": config()["provenance"], "steps": {},
              "safeguards": list(SAFEGUARDS), "limitations": list(LIMITATIONS),
              "recommendation": None, "adoption": None, "replacement_benchmark": False,
              "tone_improvement_claim": False, "perceptual_match_claim": False}
    steps = report["steps"]
    call = local._caller(phase, steps, freeze, log)

    def finish(outcome: str, reason: str | None = None, **extra) -> dict:
        for tag in STEP_TAGS:
            steps.setdefault(tag, {"status": "not_attempted", "reason": reason or outcome})
        report["outcome"] = outcome
        report.update(extra)
        if reason:
            report["reason"] = reason
        return report

    try:
        if not _single_report_path(phase.root).exists() or not (phase.root / "inputs" / "inputs.json").exists():
            raise Blocked("Imported inputs are missing; run `freeze --final-dir --single-report` first.")
        local.import_inputs(phase, None)
        reference = local.reference_profile(phase)
        report["riff_results_unchanged"] = {"tone-copier-final-v1": local.historical(phase, reference),
                                            "tone-accuracy-single-v1": riff_comparison(phase)}
        threshold = report["riff_results_unchanged"]["tone-accuracy-single-v1"]["threshold_gain_brightness_mids"]
        manifest, preparation = materialize_and_prepare(phase, adapter, renderer, freeze)
    except Blocked as error:
        return finish("blocked", str(error))
    report["materials"] = {"candidates_manifest_sha256": local._file_sha(phase.root / "candidates" / "artifacts.json"),
                           "upload_preparation_sha256": local._file_sha(phase.root / "uploads" / "preparation.json"),
                           "dis": manifest["dis"], "preparation": preparation}
    if preparation["blocked"]:
        return finish("preparation_blocked", preparation["blocked"])
    omitted = final._omitted(local.base_project(phase).chain)
    hot = refinement.control_is_hotter(reference)

    def profile(tag: str, role: str, sample: str) -> tuple[dict | None, bool]:
        row = preparation["samples"][sample][role]
        upload = local._clip(phase.root / "uploads" / row["upload_file"], f"{role}-{sample}")
        result = call(tag, "absolute_profile", PROFILE_MODEL, lambda: providers.profile(upload, PROFILE_MODEL),
                      sample=sample, role=role, upload_pcm_sha256=row["upload_pcm_sha256"])
        if result is None:
            return None, False
        judged = final._profile(result)
        return (evaluation_v3.compare_profiles_v3(reference, judged, omitted) if judged else None), True

    def halted(tag: str) -> dict:
        status = steps[tag]
        return finish("paused" if phase.ledger["provider_pause"] else "stopped",
                      f"{tag}: {status.get('reason') or status.get('note') or status['status']}")

    def judged(tag: str, role: str, sample: str):
        grade, attempted = profile(tag, role, sample)
        if not attempted:
            return None, halted(tag)
        if phase.ledger["provider_pause"]:
            return None, finish("paused", f"{tag}: HTTP 429; phase paused (no retry)")
        return grade, None

    initial: dict[str, dict] = {}
    report["baseline"] = {"initial": initial}
    for sample in SAMPLES:
        grade, stop = judged(f"baseline:{sample}", "baseline", sample)
        if stop:
            return stop
        initial[sample] = baseline_evidence(grade, threshold)
    report["control"] = {}
    for sample in SAMPLES:
        tag = f"control:{sample}"
        grade, stop = judged(tag, "control", sample)
        if stop:
            return stop
        gate = single.control_gate(grade, hot)
        report["control"][sample] = {"grade_v3": grade, "raw_axes": raw_axes(grade), "gate": gate,
                                     "judgment_status": steps[tag]["status"]}
        if not gate["passed"]:
            return finish("invalid_control", f"{sample}: control not rejected as hotter "
                          f"({gate.get('reason') or 'gain gap/direction criterion not met'}); repeats not spent")
    if not positive_possible(initial):
        verdict = classify(initial, None)
        return finish(verdict["outcome"], "initial baseline judgments exclude both positive outcomes; "
                      "repeats not spent", classification=verdict)
    repeat: dict[str, dict] = {}
    report["baseline"]["repeat"] = repeat
    for sample in SAMPLES:
        grade, stop = judged(f"baseline_repeat:{sample}", "baseline", sample)
        if stop:
            return stop
        repeat[sample] = baseline_evidence(grade, threshold)
    verdict = classify(initial, repeat)
    return finish(verdict["outcome"], classification=verdict)


def cmd_freeze(args, adapter=None, renderer=None) -> str:
    from tone_listener_mcp.workbench.renderer import NativeRenderer

    phase = Phase(args.output, create=True)
    try:
        local.import_inputs(phase, args.final_dir)
        import_single_report(phase, args.single_report)
        riff_comparison(phase)
    except Blocked as error:
        raise SystemExit(str(error)) from error
    adapter = adapter or soundshed_module.SoundshedAdapter()
    renderer = renderer or NativeRenderer(adapter)
    if hasattr(renderer, "initialize"):
        renderer.initialize()
    local.historical(phase, local.reference_profile(phase))
    try:
        variants = {item.id: item.chain_sha256 for item in build(adapter, phase)}
    except Blocked as error:
        raise SystemExit(str(error)) from error
    revise = getattr(args, "revise", False)
    digest = phase.freeze_phase(renderer, variants, revise=revise, reason=getattr(args, "reason", None))
    if revise and not phase.ledger["attempts"]:
        (phase.state / "materialized.json").unlink(missing_ok=True)
        for name in ("candidates", "uploads"):
            shutil.rmtree(phase.root / name, ignore_errors=True)
    try:
        _, preparation = materialize_and_prepare(phase, adapter, renderer, digest)
    except Blocked as error:
        raise SystemExit(str(error)) from error
    if preparation["blocked"]:
        print(preparation["blocked"], file=sys.stderr)
    print(digest)
    return digest


def cmd_plan(args, adapter=None) -> dict:
    phase = Phase(args.output)
    adapter = adapter or soundshed_module.SoundshedAdapter()
    variants = build(adapter, phase)
    ledger = phase.ledger
    record = phase._freeze_record()
    preparation = driver._load(phase.root / "uploads" / "preparation.json")
    manifest = driver._load(phase.root / "candidates" / "artifacts.json")
    reference = local.reference_profile(phase)
    plan = {
        "freeze_sha256": phase.current_freeze(),
        "freeze_history": [{"sha256": item["sha256"], "status": item["status"], "reason": item.get("reason")}
                           for item in record["history"]],
        "ledger": {"ceiling": ledger["ceiling"], "attempted": len(ledger["attempts"]),
                   "provider_pause": ledger["provider_pause"]},
        **config(),
        "variant_set": [{"id": item.id, "role": item.role, "label": item.label, "chain_sha256": item.chain_sha256}
                        for item in variants],
        "control_expected_direction": {True: "hotter", False: "cleaner", None: "either"}[
            refinement.control_is_hotter(reference)],
        "renders": {vid: {sample: {key: render[key] for key in ("file_sha256", "output_peak_all_channels",
                                                                 "samples_over_full_scale_all_channels")}
                          for sample, render in entry["renders"].items()}
                    for vid, entry in manifest["variants"].items()},
        "materials": {"candidates_manifest_sha256": local._file_sha(phase.root / "candidates" / "artifacts.json"),
                      "upload_preparation_sha256": local._file_sha(phase.root / "uploads" / "preparation.json"),
                      "binding": driver._load(phase.state / "materialized.json")["binding"],
                      "preparation_blocked": preparation["blocked"], "dis": manifest["dis"]},
        "riff_results_unchanged": {"tone-copier-final-v1": local.historical(phase, reference),
                                   "tone-accuracy-single-v1": riff_comparison(phase)},
        "uploads_per_profile_call": 1,
        "max_new_generate_requests": {"baseline": 2, "control": 2, "baseline_repeat": 2, "total": CEILING},
        "command": "python3 scripts/tone_accuracy_lead_di.py --output <phase-dir> run --live",
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
    with final._flock(phase.state / ".run.lock"):
        return _run_locked(phase, adapter, renderer, providers)


def _run_locked(phase: Phase, adapter, renderer, providers) -> dict:
    from tone_listener_mcp.workbench.renderer import NativeRenderer

    adapter = adapter or soundshed_module.SoundshedAdapter()
    renderer = renderer or NativeRenderer(adapter)
    if hasattr(renderer, "initialize"):
        renderer.initialize()
    try:
        variants = {item.id: item.chain_sha256 for item in build(adapter, phase)}
    except Blocked as error:
        raise SystemExit(str(error)) from error
    freeze = phase.freeze_phase(renderer, variants)
    report = run(phase, adapter, renderer, providers or final.Providers(), freeze)
    ledger = phase.ledger
    report["ledger"] = {"ceiling": ledger["ceiling"], "attempted": len(ledger["attempts"]),
                        "by_model": {PROFILE_MODEL: sum(item["model"] == PROFILE_MODEL
                                                        for item in ledger["attempts"])},
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
    freeze.add_argument("--single-report", type=Path, required=True,
                        help="Accepted tone-accuracy-single-v1 reports/report-01.json (read only).")
    freeze.add_argument("--revise", action="store_true", help="Supersede an unrun freeze (zero reservations only).")
    freeze.add_argument("--reason")
    sub.add_parser("plan")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    {"freeze": cmd_freeze, "plan": cmd_plan, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

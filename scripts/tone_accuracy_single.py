#!/usr/bin/env python3
"""Bounded single-clip accuracy phase for fixed same-gear Fine-tune candidates (case-01 development data only).

Phase ``tone-accuracy-single-v1`` has its OWN registry entry, ledger, lock and freeze history (ceiling 12 generation
attempts including failures; no retries; first HTTP 429 pauses the phase). It reuses the mechanisms of
``tone-accuracy-local-v1`` (whose spent 8/12 ledger is never touched or repurposed) and never writes
``tone-copier-final-v1``; the final phase's case-01 files are copied in read-only and bound by hash.

Judge: the ORIGINAL single-clip neutral ToneProfile listener (gemini-3.6-flash, unchanged prompt/schema) graded with
the unchanged v3 rubric against the REUSED final-phase case-01 reference profile. No relative calls, no new reference.

Fixed before any call: candidates ``trim_plus6``, ``eq_brighter`` and ``joint_plus6_brighter`` on the final phase's
selected ``pushed_clean_amp`` chain, plus the deliberately wrong-gain control. Uploads are matched copies: baseline =
historical WAV at its prepared (16 kHz mono) level; every other clip scaled to the same DI's baseline RMS.

Steps (each gate stops the phase immediately; nothing is retried):
  1. baseline x 2 DIs   - guitar clear, all core axes known, historical gain/brightness gaps reproduced
  2. control  x 2 DIs   - final-phase control criterion (gain gap > 1 in the intended direction, guitar clear)
  3. 3 candidates x 2   - eligible only if it beats this phase's baseline on BOTH DIs (conservative rule)
  4. repeat   x 2 DIs   - the single ranked-first eligible candidate, judged again; repeat-supported only if both
                          repeats beat the baseline too.

Subcommands: ``freeze --final-dir DIR`` (local renders/preparation, no requests), ``plan`` (no requests),
``run --live`` (paid; only after explicit approval).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tone_accuracy_local as local
import tone_copier_batch as driver
import tone_copier_final as final

from tone_listener_mcp.workbench import evaluation_v3, refinement
from tone_listener_mcp.workbench.adapters import soundshed as soundshed_module
from tone_listener_mcp.workbench.models import PreviewSnapshot, ToneProject

PHASE = "tone-accuracy-single-v1"
CEILING = 12
CASE, BASE_VARIANT, SAMPLES, INTERVAL = local.CASE, local.BASE_VARIANT, local.SAMPLES, local.INTERVAL
PROFILE_MODEL = final.PROFILE_MODEL
CANDIDATES = ("trim_plus6", "eq_brighter", "joint_plus6_brighter")
VARIANT_IDS = ("current", *CANDIDATES, "control")
UPLOAD_ROLES = ("baseline", "control", *CANDIDATES)
GATE_STEPS = (("baseline", ("baseline",)), ("control", ("control",)), ("candidates", CANDIDATES))
STEP_TAGS = (*(f"baseline:{s}" for s in SAMPLES), *(f"control:{s}" for s in SAMPLES),
             *(f"candidate:{c}:{s}" for c in CANDIDATES for s in SAMPLES), *(f"repeat:{s}" for s in SAMPLES))
assert len(STEP_TAGS) == CEILING
LEGACY_LOCAL_V1_SIDECAR = (
    "tone-accuracy-local-v1/candidates/selected/selected.json names 'eq_darker' only because the generic export "
    "helper rewrites that file for every exported candidate and eq_darker was the last non-control candidate "
    "enumerated. It is NOT selection evidence: local-v1 ended c_skipped with no supported candidate. The file is "
    "preserved unchanged; this phase writes all-candidate exports and a separate evidence-backed "
    "recommendation.json only after a repeat-supported result.")
LIMITATIONS = (
    ("Case-01 is development data (it shaped pushed_clean_amp, the selection rules and these fixed candidates); "
    "nothing here is held out."),
    "The case-01 reference profile is reused from tone-copier-final-v1; it is not a fresh or independent reference.",
    "Every outcome is one model's categorical judgment, not a listener verdict; no result is a perceptual match.",
    "Categorical improvement over the baseline, an actual core pass and a perceptual match are separate claims.",
    ("Uploads are matched COPIES (16 kHz mono; non-baseline clips scaled to the same-DI baseline RMS); they are not "
     "raw native renders, and equal RMS is not equal loudness. Native WAVs and preset parameters are unchanged."),
    ("The baseline recheck is one judgment per DI; only the chosen candidate is repeated, so the other candidates' "
     "judgment noise is not excluded."),
    "Ambience/effects are reported separately and are not validated.",
)


def config() -> dict:
    return {
        "phase": PHASE, "ceiling_generate_attempts_including_failures": CEILING, "case": CASE,
        "youtube_id": local.VIDEO, "interval": list(INTERVAL),
        "provenance": ("DEVELOPMENT data, reused evidence: case-01 shaped pushed_clean_amp, the selection rules and "
                       "these candidates; the final phase's case-01 reference profile is reused, not re-listened."),
        "base": f"tone-copier-final-v1 {CASE} selected variant '{BASE_VARIANT}' (read-only copy, bound by hash)",
        "model": PROFILE_MODEL,
        "judge": "final.Providers.profile: original neutral ToneProfile prompt/schema (unchanged); no relative calls",
        "rubric": "evaluation_v3.compare_profiles_v3(reused reference, judged copy, omitted effects); thresholds "
                  "unchanged", "thresholds": evaluation_v3.RULES,
        "candidates": {"fixed_before_calls": list(CANDIDATES), "control": "refinement wrong-gain control",
                       "source": "refinement.build_variants(search='local') filtered to these ids"},
        "upload_preparation": (
            "audio_io.prepare_upload (mono mean, 16 kHz). Baseline = historical final-phase WAV at its prepared "
            "level (no rescaling). Control and candidates scaled by one scalar to the same DI's prepared baseline "
            "RMS. Copies must be within full scale (peak <= 1.0), non-silent and invariant under the listener's "
            "preparation, else the phase is blocked before any request. Scalars and raw/prepared/upload hashes are "
            "recorded; the repeat re-sends the chosen candidate's identical upload copy as a fresh request."),
        "steps": [
            {"step": 1, "tags": [f"baseline:{s}" for s in SAMPLES], "calls": 2,
             "gate": ("guitar clear in both, every core axis known, and gain AND brightness gaps equal the "
                      "historical final-phase gaps, per DI; checked after each call (stop immediately)")},
            {"step": 2, "tags": [f"control:{s}" for s in SAMPLES], "calls": 2,
             "gate": ("final.control_rejection: guitar clear in both and gain ordinal gap > "
                      f"{evaluation_v3.RULES['maximum_axis_difference']} in the intended direction, per DI")},
            {"step": 3, "tags": [f"candidate:{c}:{s}" for c in CANDIDATES for s in SAMPLES], "calls": 6,
             "eligible": ("beats this phase's re-profiled baseline on BOTH DIs: guitar clear; every core axis known "
                          "in both grades; gain gap and brightness gap each not worse; their sum strictly smaller; "
                          "core threshold-failure count not increased; gain+brightness+mids distance not increased"),
             "ranking": ["core pass on both DIs first", "fewer total core threshold failures (both DIs)",
                         "smaller total gain+brightness+mids distance (both DIs)", "fewer parameter changes",
                         "stable id (lexicographic)"],
             "stop": "no eligible candidate: stop without repeats"},
            {"step": 4, "tags": [f"repeat:{s}" for s in SAMPLES], "calls": 2,
             "claim": ("repeat-supported categorical improvement only if both repeats ALSO beat the baseline under "
                       "the same rule; actual core pass is reported separately; neither is a perceptual match")},
        ],
        "stop_rules": [
            "Any HTTP 429 pauses the phase; remaining steps are not_attempted; nothing is retried or substituted.",
            "A failed/invalid judgment is missing evidence: a gate it belongs to fails closed.",
            "A reserved tag is never re-sent (a crash after reservation counts as attempted).",
            "Stored renders/upload copies that differ from their hashes block the phase instead of being regraded.",
        ],
        "exports": ("every non-control candidate (current + 3) exported separately under exports/; "
                    "recommendation.json is written only for a repeat-supported candidate"),
        "legacy_local_v1_sidecar": LEGACY_LOCAL_V1_SIDECAR,
        "limitations": list(LIMITATIONS),
    }


def identity(renderer, root: Path, variants: dict) -> dict:
    files = {"single_runner": __file__, "local_phase_helpers": inspect.getfile(local),
             "final_runner_helpers": inspect.getfile(final), "refinement_selection": inspect.getfile(refinement),
             "evaluation_v3": inspect.getfile(evaluation_v3), "soundshed_adapter_recipes": inspect.getfile(soundshed_module)}
    value = driver.frozen_identity(None)
    value.update({name: local._file_sha(Path(path)) for name, path in files.items()})
    value.update({"engine": getattr(renderer, "engine_version", None),
                  "worker_sha256": getattr(renderer, "_worker_sha256", None),
                  "plugin_sha256": getattr(renderer, "_plugin_sha256", None),
                  "inputs": local._inputs_identity(root), "variant_chain_sha256": variants, "config": config()})
    return value


class Phase(local.Phase):
    phase_id = PHASE
    ceiling = CEILING

    def identity(self, renderer, variants: dict) -> dict:
        return identity(renderer, self.root, variants)


Blocked = local.Blocked


def build(adapter, phase: Phase) -> list:
    reference = local.reference_profile(phase)
    variants = refinement.build_variants(adapter, local.base_project(phase).chain, reference, "local")
    chosen = [item for item in variants if item.id in VARIANT_IDS]
    if tuple(sorted(item.id for item in chosen)) != tuple(sorted(VARIANT_IDS)):
        raise Blocked(f"Fixed candidate set is unavailable on the base chain: {[item.id for item in chosen]}")
    return chosen


def param_changes(base_chain, chain) -> list[dict]:
    before = {node.id: node.params for node in base_chain.nodes}
    return [{"node": node.id, "param": key, "from": before.get(node.id, {}).get(key), "to": value}
            for node in chain.nodes for key, value in node.params.items()
            if before.get(node.id, {}).get(key) != value]


def prepare(phase: Phase, manifest: dict) -> dict:
    sources = {"baseline": lambda sample: local._input(phase, f"baseline_wav_{sample}")}
    for role in UPLOAD_ROLES[1:]:
        sources[role] = (lambda sample, role=role: phase.root / "candidates" /
                         manifest["variants"][role]["renders"][sample]["wav"])
    header = {"roles": list(UPLOAD_ROLES),
              "baseline": f"historical tone-copier-final-v1 {BASE_VARIANT} WAVs (PCM-identical to current)"}
    return local.prepare_matched(phase, "uploads", sources, header)


def _material_binding(manifest: dict, preparation: dict) -> dict:
    """Deterministic content identity of the renders and upload copies (no timestamps)."""
    return {"renders": {vid: {sample: render["file_sha256"] for sample, render in entry["renders"].items()}
                        for vid, entry in sorted(manifest["variants"].items())},
            "uploads": {sample: {role: {key: row.get(key) for key in ("scalar", "upload_pcm_sha256",
                                                                      "upload_file_sha256", "upload_peak")}
                                 for role, row in rows.items()} for sample, rows in preparation["samples"].items()},
            "blocked": preparation["blocked"]}


def materialize_and_prepare(phase: Phase, adapter, renderer, freeze: str) -> tuple[dict, dict]:
    variants = build(adapter, phase)
    used = [tag for tag in STEP_TAGS if phase.tag_used(tag)]
    if used and not (phase.root / "uploads" / "preparation.json").exists():
        raise Blocked("Judged upload copies are missing; refusing to prepare replacements.")
    manifest = local.materialize(phase, variants, adapter, renderer, freeze, judged_tags=STEP_TAGS)
    preparation = prepare(phase, manifest)
    binding = _material_binding(manifest, preparation)
    path = phase.state / "materialized.json"
    if path.exists():
        if driver._load(path)["binding"] != binding:
            raise Blocked("Rendered candidates/upload copies differ from those bound before any request; refusing.")
    else:
        if used:
            raise Blocked("Material binding is missing after requests were reserved; refusing (fail closed).")
        driver._save(path, {"binding": binding, "bound_at": local._now(), "freeze_sha256": freeze})
    return manifest, preparation


def _core_complete(grade: dict | None) -> bool:
    return grade is not None and all(grade["core"]["axes"][axis]["difference"] is not None
                                     for axis in evaluation_v3.CORE_AXES)


def baseline_gate(grade: dict | None, history: dict) -> dict:
    if grade is None:
        return {"passed": False, "reason": "baseline judgment unavailable or invalid"}
    if not grade["guitar_clear_in_both"]:
        return {"passed": False, "reason": "guitar not clearly audible in both judgments"}
    if not _core_complete(grade):
        return {"passed": False, "reason": "incomplete core evidence (an axis is uncertain)"}
    gaps = local._gaps(grade)
    reproduced = gaps["gain"] == history["gain"] and gaps["brightness"] == history["brightness"]
    return {"passed": reproduced, "reevaluated_gaps": gaps, "historical_gaps": history,
            "reason": None if reproduced else "historical gain/brightness gaps not reproduced"}


def control_gate(grade: dict | None, hot: bool | None) -> dict:
    verdict = final.control_rejection(grade, hot)
    return {"passed": verdict.get("rejected") is True, **verdict}


def rank_key(item: dict) -> tuple:
    return (not item["core_pass_both"], item["known_failures_total"], item["gain_brightness_mids_total"],
            item["parameter_changes"], item["id"])


def evaluate_candidates(grades: dict, baseline: dict, changes: dict) -> dict:
    """Per-candidate evidence vs this phase's baseline grades, eligibility and the frozen ranking."""
    out = {}
    for cid in CANDIDATES:
        per_di = {}
        for sample in SAMPLES:
            grade = grades.get(cid, {}).get(sample)
            per_di[sample] = {"grade_v3": grade, "core_decision": final.core_decision(grade),
                              "axis_deltas": local.axis_deltas(baseline[sample], grade),
                              "beats": local.beats_on_di(baseline[sample], grade)}
        verdicts = [per_di[sample]["beats"]["beats"] for sample in SAMPLES]
        gaps = [local._gaps(per_di[sample]["grade_v3"]) for sample in SAMPLES]
        eligible = all(value is True for value in verdicts)
        out[cid] = {
            "id": cid, "per_di": per_di, "eligible": eligible,
            "beats_both_dis": False if False in verdicts else True if eligible else None,
            "core_pass_both": all(per_di[s]["core_decision"]["outcome"] == "pass" for s in SAMPLES),
            "known_failures_total": sum(g["known_failure_count"] for g in gaps) if None not in gaps else None,
            "gain_brightness_mids_total": (sum(g["gain_brightness_mids"] for g in gaps)
                                           if None not in gaps and all(g["gain_brightness_mids"] is not None
                                                                       for g in gaps) else None),
            "parameter_changes": len(changes[cid]), "changes": changes[cid]}
    ranked = sorted((item for item in out.values() if item["eligible"]), key=rank_key)
    return {"candidates": out, "eligible_ranked": [item["id"] for item in ranked],
            "chosen": ranked[0]["id"] if ranked else None}


def run(phase: Phase, adapter, renderer, providers, freeze: str, log: Callable[[str], None] = print) -> dict:
    report = {"phase": PHASE, "freeze_sha256": freeze, "case": CASE, "interval": list(INTERVAL),
              "model": PROFILE_MODEL, "provenance": config()["provenance"], "steps": {},
              "legacy_local_v1_sidecar": LEGACY_LOCAL_V1_SIDECAR, "limitations": list(LIMITATIONS),
              "recommendation": None, "perceptual_match_claim": False}
    steps = report["steps"]
    call = local._caller(phase, steps, freeze, log)

    def finish(outcome: str, reason: str | None = None) -> dict:
        for tag in STEP_TAGS:
            steps.setdefault(tag, {"status": "not_attempted", "reason": reason or outcome})
        report["outcome"] = outcome
        if reason:
            report["reason"] = reason
        return report

    try:
        if not (phase.root / "inputs" / "inputs.json").exists():
            raise Blocked("Imported final-phase inputs are missing; run `freeze --final-dir` first.")
        local.import_inputs(phase, None)
        reference = local.reference_profile(phase)
        history = local.historical(phase, reference)
        manifest, preparation = materialize_and_prepare(phase, adapter, renderer, freeze)
    except Blocked as error:
        return finish("blocked", str(error))
    report["historical_baseline"] = history
    report["materials"] = {"candidates_manifest_sha256": local._file_sha(phase.root / "candidates" / "artifacts.json"),
                           "upload_preparation_sha256": local._file_sha(phase.root / "uploads" / "preparation.json"),
                           "current_matches_historical_baseline_pcm":
                               manifest["current_matches_historical_baseline_pcm"], "preparation": preparation}
    if preparation["blocked"]:
        return finish("preparation_blocked", preparation["blocked"])
    omitted = final._omitted(local.base_project(phase).chain)
    hot = refinement.control_is_hotter(reference)

    def profile(tag: str, role: str, sample: str, **detail) -> tuple[dict | None, bool]:
        row = preparation["samples"][sample][role]
        upload = local._clip(phase.root / "uploads" / row["upload_file"], f"{role}-{sample}")
        result = call(tag, "absolute_profile", PROFILE_MODEL, lambda: providers.profile(upload, PROFILE_MODEL),
                      sample=sample, role=role, upload_pcm_sha256=row["upload_pcm_sha256"], **detail)
        if result is None:
            return None, False
        judged = final._profile(result)
        return (evaluation_v3.compare_profiles_v3(reference, judged, omitted) if judged else None), True

    def halted(tag: str) -> dict:
        status = steps[tag]
        return finish("paused" if phase.ledger["provider_pause"] else "stopped",
                      f"{tag}: {status.get('reason') or status.get('note') or status['status']}")

    def gate_failed(tag: str, outcome: str, reason: str) -> dict:
        if phase.ledger["provider_pause"]:
            return finish("paused", f"{tag}: HTTP 429; phase paused (no retry)")
        if steps[tag]["status"] != "ok":
            return finish(f"{outcome}_missing_evidence", f"{tag}: judgment {steps[tag]['status']}; fails closed")
        return finish(outcome, reason)

    baseline: dict[str, dict | None] = {}
    report["baseline"] = {}
    for sample in SAMPLES:
        tag = f"baseline:{sample}"
        grade, attempted = profile(tag, "baseline", sample)
        if not attempted:
            return halted(tag)
        gate = baseline_gate(grade, history[sample]["gaps"])
        baseline[sample] = grade
        report["baseline"][sample] = {"grade_v3": grade, "core_decision": final.core_decision(grade), "gate": gate}
        if not gate["passed"]:
            return gate_failed(tag, "baseline_not_reproduced",
                               f"{sample}: {gate['reason']} (measurement limitation; no retry)")
    report["control"] = {}
    for sample in SAMPLES:
        tag = f"control:{sample}"
        grade, attempted = profile(tag, "control", sample)
        if not attempted:
            return halted(tag)
        gate = control_gate(grade, hot)
        report["control"][sample] = {"grade_v3": grade, "gate": gate}
        if not gate["passed"]:
            return gate_failed(tag, "control_not_rejected",
                               f"{sample}: {gate.get('reason') or 'gain gap/direction criterion not met'}")
    grades: dict[str, dict] = {}
    for cid in CANDIDATES:
        for sample in SAMPLES:
            tag = f"candidate:{cid}:{sample}"
            grade, attempted = profile(tag, cid, sample, candidate=cid)
            if not attempted:
                return halted(tag)
            grades.setdefault(cid, {})[sample] = grade
    base_chain = local.base_project(phase).chain
    variants = {item.id: item for item in build(adapter, phase)}
    changes = {cid: param_changes(base_chain, variants[cid].chain) for cid in CANDIDATES}
    evaluation = evaluate_candidates(grades, baseline, changes)
    report["candidates"] = evaluation
    chosen = evaluation["chosen"]
    if phase.ledger["provider_pause"]:
        return finish("paused", "HTTP 429 during candidate judgments; phase paused (no retry)")
    if chosen is None:
        return finish("no_eligible_candidate", "no candidate beat the baseline on both DIs; repeats not spent")
    repeats = {}
    for sample in SAMPLES:
        tag = f"repeat:{sample}"
        recorded = phase.root / "judgments" / f"{tag}.json"
        if recorded.exists() and driver._load(recorded).get("request_detail", {}).get("candidate") != chosen:
            return finish("blocked", f"{tag} was recorded for a different candidate; refusing.")
        grade, attempted = profile(tag, chosen, sample, candidate=chosen)
        if not attempted:
            return halted(tag)
        repeats[sample] = {"grade_v3": grade, "core_decision": final.core_decision(grade),
                           "axis_deltas": local.axis_deltas(baseline[sample], grade),
                           "beats": local.beats_on_di(baseline[sample], grade)}
    supported = all(repeats[sample]["beats"]["beats"] is True for sample in SAMPLES)
    paused = bool(phase.ledger["provider_pause"])
    initial = evaluation["candidates"][chosen]
    report["repeat"] = {
        "candidate": chosen, "per_di": repeats, "repeat_supported_improvement": supported,
        "core_pass_both_dis": {"initial": initial["core_pass_both"],
                               "repeat": all(repeats[s]["core_decision"]["outcome"] == "pass" for s in SAMPLES)},
        "claims": {"categorical_improvement_vs_baseline": "repeat-supported" if supported else "not supported",
                   "actual_core_pass": "reported separately above; improvement is not a core pass",
                   "perceptual_match": False}}
    if paused and not supported:
        return finish("paused", "HTTP 429 during repeats; phase paused (no retry)")
    return finish("repeat_supported_improvement" if supported else "repeat_not_supported")


def export_all(phase: Phase, manifest: dict, adapter) -> dict:
    """Export every non-control candidate separately; existing files are reused only if equivalent."""
    directory = phase.root / "candidates"
    out_dir = phase.root / "exports"
    out_dir.mkdir(exist_ok=True)
    out = {}
    for vid, entry in sorted(manifest["variants"].items()):
        if entry["role"] == "control":
            continue
        project = ToneProject.model_validate(driver._load(directory / entry["project"]))
        out[vid] = {}
        for sample, render in entry["renders"].items():
            snapshot = PreviewSnapshot.model_validate(render["snapshot"])
            path = out_dir / f"{vid}-{sample}.soundshed.preset"
            try:
                archive = adapter.export_archive(project, snapshot)
            except Exception as error:  # noqa: BLE001 - export failure is reported, not hidden
                out[vid][sample] = {"status": "failed", "reason": f"{type(error).__name__}: {str(error)[:200]}"}
                continue
            if path.exists():
                existing = path.read_bytes()
                content = final._archive_content(existing)
                if content is None or content != final._archive_content(archive):
                    out[vid][sample] = {"status": "conflict", "reason": "existing preset differs; not overwritten",
                                        "existing_sha256": local._file_sha(path)}
                    continue
                archive = existing
            else:
                path.write_bytes(archive)
                os.chmod(path, 0o600)
            out[vid][sample] = {"status": "exported", "path": f"exports/{path.name}", "sha256": local._sha(archive),
                                "preview_audio_sha256": render["file_sha256"],
                                "validation": final._validate_archive(archive, project.chain, snapshot, adapter)}
    return out


def recommendation(report: dict, exports: dict) -> dict | None:
    """Evidence-backed recommendation; only for a completed, repeat-supported run."""
    if report.get("outcome") != "repeat_supported_improvement":
        return None
    chosen = report["repeat"]["candidate"]
    return {"phase": PHASE, "freeze_sha256": report["freeze_sha256"], "candidate": chosen,
            "basis": ("beat this phase's re-profiled baseline on both DIs, ranked first among eligible candidates, "
                      "and both independent repeats also beat the baseline (v3 categorical rubric)"),
            "core_pass_both_dis": report["repeat"]["core_pass_both_dis"],
            "parameter_changes": report["candidates"]["candidates"][chosen]["changes"],
            "presets": exports.get(chosen), "evidence": [f"judgments/{tag}.json" for tag in STEP_TAGS],
            "claims": report["repeat"]["claims"], "limitations": list(LIMITATIONS)}


def cmd_freeze(args, adapter=None, renderer=None) -> str:
    from tone_listener_mcp.workbench.renderer import NativeRenderer

    phase = Phase(args.output, create=True)
    try:
        local.import_inputs(phase, args.final_dir)
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
        # Zero reservations: renders/copies may be rebuilt under the revised freeze.
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
    base_chain = local.base_project(phase).chain
    ledger = phase.ledger
    record = phase._freeze_record()
    preparation = driver._load(phase.root / "uploads" / "preparation.json")
    reference = local.reference_profile(phase)
    plan = {
        "freeze_sha256": phase.current_freeze(),
        "freeze_history": [{"sha256": item["sha256"], "status": item["status"], "reason": item.get("reason")}
                           for item in record["history"]],
        "ledger": {"ceiling": ledger["ceiling"], "attempted": len(ledger["attempts"]),
                   "provider_pause": ledger["provider_pause"]},
        **config(),
        "candidate_set": [{"id": item.id, "role": item.role, "label": item.label, "description": item.description,
                           "chain_sha256": item.chain_sha256, "parameter_changes": param_changes(base_chain, item.chain)}
                          for item in variants],
        "control_expected_direction": {True: "hotter", False: "cleaner", None: "either"}[
            refinement.control_is_hotter(reference)],
        "materials": {"candidates_manifest_sha256": local._file_sha(phase.root / "candidates" / "artifacts.json"),
                      "upload_preparation_sha256": local._file_sha(phase.root / "uploads" / "preparation.json"),
                      "binding": driver._load(phase.state / "materialized.json")["binding"],
                      "preparation_blocked": preparation["blocked"]},
        "uploads_per_profile_call": 1,
        "max_new_generate_requests": {"baseline": 2, "control": 2, "candidates": 6, "repeat": 2, "total": CEILING},
        "historical_baseline_gaps": {sample: value["gaps"] for sample, value in
                                     local.historical(phase, reference).items()},
        "command": "python3 scripts/tone_accuracy_single.py --output <phase-dir> run --live",
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
    manifest_path = phase.root / "candidates" / "artifacts.json"
    exports = export_all(phase, driver._load(manifest_path), adapter) if manifest_path.exists() else {}
    report["exports"] = exports
    advice = recommendation(report, exports)
    if advice is not None:
        driver._save(phase.root / "recommendation.json", advice)
        report["recommendation"] = "recommendation.json"
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
    freeze.add_argument("--revise", action="store_true", help="Supersede an unrun freeze (zero reservations only).")
    freeze.add_argument("--reason")
    sub.add_parser("plan")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    {"freeze": cmd_freeze, "plan": cmd_plan, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

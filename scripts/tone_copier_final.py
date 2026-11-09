#!/usr/bin/env python3
"""Final multi-reference tone-copier evaluation using the SHIPPED product path.

For every case the first fresh reference listening is also the product's analysis input; the chain
comes from ``SoundshedAdapter.default_chain`` and candidates from ``refinement.build_variants``; the
selection is ``refinement.summarize_judgment`` + ``suggest_from`` exactly as the web workflow does.
Nothing is hand-tuned per case.

Budget: one durable phase ledger with a global ceiling of 48 generation attempts INCLUDING failures,
across all cases, models and processes. The authoritative ledger, reserved tags and freeze (with its
history) live in a per-user phase registry outside the output directory, bound to the output by a
random binding id; a missing, moved or recreated piece fails closed, so the ceiling never resets. A
freeze may be revised only while zero attempts are reserved, and the superseded freeze is retained.
Any HTTP 429 pauses the whole phase; nothing is retried or substituted. Judgments are never reused.

Per case, variant projects, full native snapshots and WAVs are bound by hash before any selector
request; a resume verifies and reuses them and never grades replacement audio.

Subcommands: ``freeze [--revise]`` (local only), ``migrate-legacy``, ``plan`` (no requests),
``run --live`` (paid; only after explicit approval).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import inspect
import io
import json
import os
import secrets
import shutil
import sys
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tone_copier_batch as driver

from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.listener import ToneProfile
from tone_listener_mcp.workbench import evaluation_v3, refinement
from tone_listener_mcp.workbench.adapters import soundshed as soundshed_module
from tone_listener_mcp.workbench.models import (
    Interval,
    PreviewSnapshot,
    SourceRef,
    TargetBinding,
    ToneProject,
)

PHASE = "tone-copier-final-v1"
CEILING = 48
PROFILE_MODEL = "gemini-3.6-flash"
SELECTOR_MODEL = "gemini-4-argon"
SAMPLES = ("riff-1", "riff-2")
# (case, YouTube id, interval, provenance). No case is pristine.
CASES = (
    ("case-01", "401iS5McuoU", (19.2, 30.0),
     "TUNING case: v3 batches 1-2 derived the pushed_clean_amp variant and selection rules from it."),
    ("case-02", "qU1qy9831nw", (0.0, 30.0),
     "v1/v2 development case (outcomes seen); not used for v3 tuning. Interval preregistered 0-30 s."),
    ("case-03", "YfBn7XhGerc", (0.0, 30.0),
     "v1/v2 development case (outcomes seen, repeated); not used for v3 tuning. Interval preregistered 0-30 s."),
    ("case-04", "lD6LbzTZrvs", (0.0, 30.0),
     "v1/v2 development case (outcomes seen); not used for v3 tuning. Interval preregistered 0-30 s."),
    ("case-05", "6FfQqDl_7yY", (0.0, 30.0),
     "v2 held-out whose outcome was INSPECTED; not pristine. Interval preregistered 0-30 s."),
    ("case-06", "MIc-6iOC0VU", (0.0, 30.0),
     "v2 held-out whose outcome was INSPECTED; not pristine. Interval preregistered 0-30 s."),
)
# Ordered so an early stop (429 / blocked case) keeps the most informative evidence.
STEPS = (
    ("reference", "profile", PROFILE_MODEL,
     "Fresh reference listening of the YouTube interval; also the product analysis input."),
    ("select-riff-1", "relative", SELECTOR_MODEL,
     "Shipped blinded relative judgment of all variants + wrong-gain control vs the reference (riff-1)."),
    ("select-riff-2", "relative", SELECTOR_MODEL, "Same for riff-2; selection needs both DIs to agree."),
    ("eval-riff-1", "profile", PROFILE_MODEL, "Absolute v3 profile of the SELECTED candidate render, riff-1."),
    ("eval-riff-2", "profile", PROFILE_MODEL, "Absolute v3 profile of the SELECTED candidate render, riff-2."),
    ("control-riff-1", "profile", PROFILE_MODEL, "Absolute v3 profile of the wrong-gain control, riff-1."),
    ("reference-repeat", "profile", PROFILE_MODEL, "Independent fresh repeat of the reference listening."),
    ("preview-repeat-riff-1", "profile", PROFILE_MODEL, "Independent repeat of eval-riff-1 (stability)."),
)
CALLS_PER_CASE = len(STEPS)
assert CALLS_PER_CASE * len(CASES) == CEILING
OMITTED_FIELDS = ("reverb", "delay", "modulation")
LIMITATIONS = (
    "Every outcome is a model judgment, not a listener verdict; no result is a perceptual-identity claim.",
    ("Ambience/effects are reported separately and are NOT validated: batch2 strong-ambience controls failed "
     "for both models, so even categorical effect agreement is not ambience matching."),
    ("Selection ranking and small-brightness detection were weak in development; the selected variant is not "
     "claimed optimal."),
    "Case-01 is a tuning case; cases 02-06 had prior outcomes seen; none is pristine held-out data.",
    ("Renders are RMS level-matched for the judge; durations differ by native tails, so equal RMS is not "
     "equal loudness."),
)


def registry_path() -> Path:
    override = os.environ.get("TONE_EVAL_PHASE_REGISTRY")
    if override:
        return Path(override)
    return Path.home() / "Library" / "Application Support" / "Tone Listener MCP" / "eval-phases"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@contextlib.contextmanager
def _flock(path: Path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def config() -> dict:
    return {
        "phase": PHASE, "ceiling_generate_attempts_including_failures": CEILING,
        "profile_model": PROFILE_MODEL, "selector_model": SELECTOR_MODEL, "samples": list(SAMPLES),
        "cases": [{"case": case, "youtube_id": video, "interval": list(interval), "provenance": note}
                  for case, video, interval, note in CASES],
        "steps_per_case": [{"tag": tag, "kind": kind, "model": model, "purpose": purpose}
                           for tag, kind, model, purpose in STEPS],
        "eligibility": ("Shipped refinement eligibility before any selector request: clear reference guitar, valid "
                        "current chain, at least one distinct variant and a distinct wrong-gain control."),
        "selection": ("refinement.build_variants -> summarize_judgment (both DIs) -> suggest_from; "
                      "no suggestion => keep the product default chain ('current')."),
        "grading": {
            "per_di_core": ("compare_profiles_v3 core outcome; guitar not clear in both => inconclusive "
                            "(observed differences still reported); missing/failed DI => explicit inconclusive"),
            "aggregate_core": "aggregate_v3 worst-first over BOTH planned DIs",
            "control_rejected": ("structured: guitar clear in both, gain difference > "
                                 f"{evaluation_v3.RULES['maximum_axis_difference']} step(s) in the deliberate "
                                 "direction (hotter/cleaner)"),
            "repeat_stable": "guitar clear in both and v3 core outcome pass between the two independent judgments",
            "acceptance": ("fail if any DI core fails; otherwise core_supported_effects_not_validated only when "
                           "both DIs pass core AND the control is rejected AND reference and preview repeats are "
                           "stable; else inconclusive. Never a perceptual-match pass."),
            "effects": "reported separately; not validated",
        },
        "artifacts": ("Per case: immutable variant projects, full native snapshots and WAVs bound by hash before "
                      "any selector request; resume verifies and reuses them. Selected variant's "
                      ".soundshed.preset exported per DI and validated against the chain and preview."),
        "stop_rules": [
            "Any HTTP 429 pauses the phase; remaining steps are recorded not_attempted; no retries.",
            ("A failed/invalid reference, non-clear guitar dominance, or failed eligibility blocks the case; "
             "its remaining calls stay unspent."),
            "A previously reserved tag is never re-sent (a crash after reservation counts as attempted).",
        ],
        "limitations": list(LIMITATIONS),
    }


def identity(renderer) -> dict:
    files = {
        "final_runner": __file__,
        "refinement_selection": inspect.getfile(refinement),
        "soundshed_adapter_recipes": inspect.getfile(soundshed_module),
    }
    base = driver.frozen_identity(None)
    base.update({name: driver._sha(Path(path).read_bytes()) for name, path in files.items()})
    base.update({"engine": getattr(renderer, "engine_version", None),
                 "worker_sha256": getattr(renderer, "_worker_sha256", None),
                 "plugin_sha256": getattr(renderer, "_plugin_sha256", None),
                 "config": config()})
    return base


class Phase(driver.Batch):
    """Phase ledger whose authoritative counter, tags and freeze live OUTSIDE the output directory.

    The registry entry, the state directory and the output binding share one random binding id. Any
    missing or replaced piece (renamed/recreated output, deleted ledger) fails closed instead of
    starting a fresh ledger, so the ceiling cannot be reset by moving, restarting or refreezing.
    """

    def __init__(self, output: Path, create: bool = False):
        output = Path(output).resolve()
        registry = registry_path()
        registry.mkdir(parents=True, exist_ok=True)
        os.chmod(registry, 0o700)
        marker = registry / f"{PHASE}.json"
        state = registry / PHASE
        with _flock(registry / f".{PHASE}.lock"):
            if marker.exists():
                info = driver._load(marker)
                if info["output"] != str(output):
                    raise SystemExit(f"{PHASE} is registered to another directory; its ceiling cannot be reset.")
                if "binding" not in info:
                    raise SystemExit(f"{PHASE} uses the legacy output-local ledger; run `migrate-legacy` first.")
            else:
                if not create:
                    raise SystemExit(f"{PHASE} is not initialized; run `freeze` first.")
                if state.exists():
                    raise SystemExit("Phase state exists without its registry entry; refusing (fail closed).")
                if output.exists() and any(output.iterdir()):
                    raise SystemExit("A new phase needs a new or empty output directory.")
                info = {"phase": PHASE, "output": str(output), "ceiling": CEILING,
                        "binding": secrets.token_hex(16), "created_at": _now()}
                state.mkdir(mode=0o700)
                driver._save(state / "ledger.json", {"ceiling": CEILING, "attempts": [], "provider_pause": None,
                                                     "binding": info["binding"]})
                output.mkdir(parents=True, exist_ok=True)
                driver._save(output / "phase-binding.json", _binding_note(info["binding"]))
                driver._save(marker, info)
            self._verify(info, output, state)
        self.root = output
        self.state = state
        self.ledger_path = state / "ledger.json"
        self.lock_path = state / ".ledger.lock"

    @staticmethod
    def _verify(info: dict, output: Path, state: Path) -> None:
        ledger_path, binding_path = state / "ledger.json", output / "phase-binding.json"
        if not ledger_path.exists():
            raise SystemExit("Authoritative phase ledger is missing; refusing to start a new one (fail closed).")
        ledger = driver._load(ledger_path)
        if ledger.get("binding") != info["binding"] or ledger.get("ceiling") != CEILING:
            raise SystemExit("Authoritative phase ledger does not match the registry; refusing (fail closed).")
        if not binding_path.exists() or driver._load(binding_path).get("binding") != info["binding"]:
            raise SystemExit("Output directory is missing or was replaced; its phase ledger cannot restart at zero.")

    def tag_used(self, tag: str, ledger: dict | None = None) -> bool:
        ledger = ledger if ledger is not None else self.ledger
        return any(item.get("tag") == tag for item in ledger["attempts"]) or (
            self.root / "judgments" / f"{tag}.json").exists()

    def _freeze_record(self) -> dict:
        path = self.state / "freeze.json"
        return driver._load(path) if path.exists() else {"current": None, "history": []}

    def current_freeze(self) -> str | None:
        current = self._freeze_record()["current"]
        return current["sha256"] if current else None

    def freeze_final(self, renderer, revise: bool = False, reason: str | None = None) -> str:
        value = identity(renderer)
        digest = driver._sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
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


def _binding_note(binding: str) -> dict:
    return {"phase": PHASE, "binding": binding,
            "note": "The authoritative ledger, tags and freeze are kept in the per-user phase registry."}


def migrate_legacy(output: Path) -> dict:
    """Move a ZERO-attempt legacy output-local phase under the registry; prior freeze kept as history."""
    output = Path(output).resolve()
    registry = registry_path()
    marker, state = registry / f"{PHASE}.json", registry / PHASE
    with _flock(registry / f".{PHASE}.lock"):
        if not marker.exists():
            raise SystemExit("No legacy phase registration exists.")
        info = driver._load(marker)
        if "binding" in info:
            raise SystemExit("Phase was already migrated.")
        if info["output"] != str(output):
            raise SystemExit(f"{PHASE} is registered to another directory.")
        if state.exists():
            raise SystemExit("Phase state already exists; refusing (fail closed).")
        legacy_ledger_path, legacy_freeze_path = output / "ledger.json", output / "freeze.json"
        legacy = driver._load(legacy_ledger_path)
        judgments = output / "judgments"
        if (legacy.get("attempts") or legacy.get("provider_pause") or legacy.get("ceiling") != CEILING
                or (judgments.exists() and any(judgments.iterdir()))):
            raise SystemExit("Legacy phase has reservations or judgments; migration refused (fail closed).")
        legacy_freeze = driver._load(legacy_freeze_path)
        history = output / "history" / "legacy-01"
        history.mkdir(parents=True)
        moved = {}
        for name in ("ledger.json", "freeze.json", "plan.json", ".ledger.lock"):
            if (output / name).exists():
                moved[name] = driver._sha((output / name).read_bytes())
                os.replace(output / name, history / name)
        binding = secrets.token_hex(16)
        state.mkdir(mode=0o700)
        driver._save(state / "ledger.json", {
            "ceiling": CEILING, "attempts": [], "provider_pause": None, "binding": binding,
            "migrated_from_legacy": {"verified_zero_attempts": True, "migrated_at": _now(),
                                     "files_sha256": moved}})
        sha = legacy_freeze.pop("sha256")
        driver._save(state / "freeze.json", {"current": None, "history": [{
            "sha256": sha, "identity": legacy_freeze, "status": "superseded_unrun",
            "superseded_at": _now(), "attempts_at_supersession": 0,
            "reason": "legacy output-local ledger replaced by authoritative registry ledger before any spend"}]})
        driver._save(output / "phase-binding.json", _binding_note(binding))
        driver._save(marker, {**info, "binding": binding, "migrated_at": _now()})
    return {"legacy_freeze_sha256": sha, "moved": moved}


class Providers:
    """Real provider calls; replaced by fakes in offline tests."""

    def reference(self, url: str, start: float, end: float, model: str) -> dict:
        from tone_listener_mcp.listener import listen_youtube_tone_profile

        os.environ["TONE_LISTENER_MODEL"] = model
        return listen_youtube_tone_profile(url, start, end)

    def profile(self, clip: AudioClip, model: str) -> dict:
        from tone_listener_mcp.listener import listen_tone_profile

        os.environ["TONE_LISTENER_MODEL"] = model
        return listen_tone_profile(clip)

    def relative(self, *, model: str, candidates: dict, seed: int, youtube) -> dict:
        return refinement.relative_judge.judge_relative(model=model, candidates=candidates, seed=seed,
                                                        youtube=youtube)


def _profile(result: dict | None) -> ToneProfile | None:
    value = (result or {}).get("listening")
    return ToneProfile.model_validate(value) if isinstance(value, dict) else None


def _omitted(chain) -> list[str]:
    titles = [node.title for node in chain.nodes if node.kind == "gear_note" and node.note_role == "omitted_optional"]
    return [field for field in OMITTED_FIELDS if any(field.title() in title for title in titles)]


def _status(result: dict) -> str:
    return "failed" if result.get("failure_info") else "invalid" if result.get("invalid") else "ok"


def _file_sha(path: Path) -> str:
    return driver._sha(path.read_bytes())


def core_decision(grade: dict | None, missing_reason: str | None = None) -> dict:
    """Per-DI core outcome honoring the audibility precondition; missing evidence is explicit."""
    if grade is None:
        return {"outcome": "inconclusive", "reason": missing_reason or "evaluation unavailable", "observed": None}
    core = grade["core"]
    if not grade["guitar_clear_in_both"]:
        return {"outcome": "inconclusive", "reason": "guitar not clearly audible in both judgments",
                "observed_known_failures": core["known_failures"]}
    return {"outcome": core["outcome"], "known_failures": core["known_failures"],
            "unknown_axes": core["unknown_axes"]}


def control_rejection(grade: dict | None, hot: bool | None) -> dict:
    threshold = evaluation_v3.RULES["maximum_axis_difference"]
    if grade is None:
        return {"rejected": None, "reason": "control judgment unavailable"}
    if not grade["guitar_clear_in_both"]:
        return {"rejected": None, "reason": "guitar not clearly audible; control not interpretable"}
    gain = grade["core"]["axes"]["gain"]
    if gain["difference"] is None:
        return {"rejected": None, "reason": "gain uncertain in a judgment", "gain": gain}
    order = evaluation_v3.AXES["gain"]
    signed = order.index(gain["candidate"]) - order.index(gain["reference"])
    direction_ok = hot is None or (signed > 0 if hot else signed < 0)
    return {"rejected": gain["difference"] > threshold and direction_ok, "gain": gain, "signed_steps": signed,
            "expected_direction": None if hot is None else "hotter" if hot else "cleaner",
            "threshold_steps": threshold}


def repeat_stability(first: ToneProfile | None, second: ToneProfile | None) -> dict:
    if first is None or second is None:
        return {"stable": None, "reason": "repeat or its counterpart unavailable"}
    grade = evaluation_v3.compare_profiles_v3(first, second, [])
    stable = grade["guitar_clear_in_both"] and grade["core"]["outcome"] == "pass"
    return {"stable": stable, "core": grade["core"], "guitar_clear_in_both": grade["guitar_clear_in_both"],
            "effects_not_validated": grade["effects"]}


def acceptance(aggregate: str, control: dict, reference_repeat: dict, preview_repeat: dict) -> dict:
    validity = []
    if control.get("rejected") is not True:
        validity.append("wrong-gain control not rejected or unavailable: evaluator not validated for this case")
    if reference_repeat.get("stable") is not True:
        validity.append("reference repeat not shown stable")
    if preview_repeat.get("stable") is not True:
        validity.append("preview repeat not shown stable")
    if aggregate == "fail":
        outcome = "fail"
    elif aggregate == "pass" and not validity:
        outcome = "core_supported_effects_not_validated"
    else:
        outcome = "inconclusive"
    reasons = ([] if aggregate in ("pass", "fail") else [f"core aggregate {aggregate}"]) + validity
    return {"outcome": outcome, "reasons": reasons, "perceptual_match_claim": False,
            "effects": "reported separately; not validated"}


def _validate_archive(archive: bytes, chain, snapshot: PreviewSnapshot, adapter) -> dict:
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        names = handle.namelist()
        payload = json.loads(handle.read("preset.json"))
    graph = payload["preset"]["scenes"][0]["graph"]["nodes"]
    expected_ids = [("__input__" if node.kind == "input" else "__output__" if node.kind == "output" else node.id)
                    for node in chain.nodes if node.kind != "gear_note"]
    expected_assets = sorted((node.id, node.asset.resource_type, node.asset.resource_id)
                             for node in chain.nodes if node.asset is not None)
    graph_assets = sorted((item["id"], resource["resourceType"], resource["resourceId"])
                          for item in graph for resource in item.get("resources", []))
    checks = {
        "only_preset_json": names == ["preset.json"],
        "no_bundled_resources": payload.get("resources") == [],
        "graph_matches_chain_order": [item["id"] for item in graph] == expected_ids,
        "resources_match_chain_assets": graph_assets == expected_assets,
        "preview_matches_chain": snapshot.chain_sha256 == adapter.chain_sha256(chain),
    }
    return {"valid": all(checks.values()), "checks": checks,
            "portability": ("References the local Soundshed resource library by id; NAM/IR files are not "
                            "included. Standalone import is not claimed.")}


class CaseBlocked(Exception):
    pass


def _variant_project(project: ToneProject, chain) -> ToneProject:
    return project.model_copy(update={"chain": chain, "target": TargetBinding(
        adapter_id=project.target.adapter_id, engine_version=project.target.engine_version,
        resources={node.id: node.asset for node in chain.nodes if node.asset is not None})})


def materialize(phase: Phase, name: str, directory: Path, project: ToneProject, variants, adapter, renderer,
                freeze: str) -> dict:
    """Render once, bind by hash, then only ever verify and reuse."""
    manifest_path = directory / "artifacts.json"
    expected = {item.id: item.chain_sha256 for item in variants}
    paid = [f"{name}:{tag}" for tag, *_ in STEPS[1:]]
    if manifest_path.exists():
        manifest = driver._load(manifest_path)
        if manifest["freeze_sha256"] != freeze or {k: v["chain_sha256"] for k, v in
                                                    manifest["variants"].items()} != expected:
            raise CaseBlocked("Stored artifacts do not match the current variants/freeze; refusing to regrade.")
        if manifest["engine_version"] != renderer.engine_version:
            raise CaseBlocked("Stored artifacts were rendered by a different engine.")
        for vid, entry in manifest["variants"].items():
            if _file_sha(directory / entry["project"]) != entry["project_sha256"]:
                raise CaseBlocked(f"Stored project for {vid} changed.")
            for sample, render in entry["renders"].items():
                if _file_sha(directory / render["wav"]) != render["file_sha256"]:
                    raise CaseBlocked(f"Stored audio for {vid}/{sample} changed; refusing to regrade.")
        return manifest
    if any(phase.tag_used(tag) for tag in paid):
        raise CaseBlocked("Judged artifacts are missing; refusing to grade replacement audio.")
    for leftover in (directory / "renders.partial", directory / "renders", directory / "variants"):
        if leftover.exists():
            shutil.rmtree(leftover)  # never judged: no paid step of this case was ever reserved
    partial = directory / "renders.partial"
    partial.mkdir(parents=True)
    (directory / "variants").mkdir()
    manifest = {"case": name, "freeze_sha256": freeze, "engine_version": renderer.engine_version,
                "created_at": _now(), "variants": {}}
    for variant in variants:
        candidate = _variant_project(project, variant.chain)
        project_rel = f"variants/{variant.id}.project.json"
        driver._save(directory / project_rel, candidate.model_dump(mode="json"))
        entry = {"role": variant.role, "label": variant.label, "chain_sha256": variant.chain_sha256,
                 "project": project_rel, "project_sha256": _file_sha(directory / project_rel),
                 "assets": [{"node": node.id, **node.asset.model_dump(mode="json")}
                            for node in variant.chain.nodes if node.asset is not None],
                 "omitted_optional": _omitted(variant.chain), "renders": {}}
        for sample in SAMPLES:
            try:
                rendered = renderer.render(candidate, sample, partial)
            except Exception as error:
                raise CaseBlocked(f"Local render failed for {variant.id}/{sample}: {type(error).__name__}") \
                    from error
            wav = partial / f"{variant.id}-{sample}.wav"
            os.replace(partial / f"{rendered.snapshot.id}.wav", wav)
            digest = _file_sha(wav)
            if digest != rendered.snapshot.audio_sha256:
                raise CaseBlocked(f"Rendered file hash differs from its snapshot for {variant.id}/{sample}.")
            entry["renders"][sample] = {"wav": f"renders/{variant.id}-{sample}.wav", "file_sha256": digest,
                                        "snapshot": rendered.snapshot.model_dump(mode="json")}
        manifest["variants"][variant.id] = entry
    os.replace(partial, directory / "renders")
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    driver._save(manifest_path, manifest)
    return manifest


def _load_clip(directory: Path, render: dict, source: str) -> AudioClip:
    samples, rate = sf.read(directory / render["wav"], dtype="float32", always_2d=True)
    return AudioClip(samples=samples.T, sample_rate=rate, source=source, start_seconds=0)


def _archive_content(data: bytes) -> dict | None:
    """Member names mapped to parsed JSON (or raw bytes); ZIP timestamps and compression are ignored."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            content = {}
            for name in archive.namelist():
                raw = archive.read(name)
                try:
                    content[name] = json.loads(raw)
                except ValueError:
                    content[name] = raw
            return content
    except (zipfile.BadZipFile, OSError):
        return None


def export_selected(directory: Path, manifest: dict, selected: str, adapter) -> dict:
    entry = manifest["variants"][selected]
    project = ToneProject.model_validate(driver._load(directory / entry["project"]))
    out = directory / "selected"
    out.mkdir(exist_ok=True)
    driver._save(out / "selected.json", {"variant": selected, "project": entry["project"],
                                         "chain_sha256": entry["chain_sha256"]})
    exports = {}
    for sample, render in entry["renders"].items():
        snapshot = PreviewSnapshot.model_validate(render["snapshot"])
        path = out / f"{selected}-{sample}.soundshed.preset"
        try:
            archive = adapter.export_archive(project, snapshot)
        except Exception as error:  # noqa: BLE001 - export failure is reported, not hidden
            exports[sample] = {"status": "failed", "reason": f"{type(error).__name__}: {str(error)[:200]}"}
            continue
        if path.exists():
            existing = path.read_bytes()
            content = _archive_content(existing)
            if content is None or content != _archive_content(archive):
                exports[sample] = {"status": "conflict", "reason": "existing preset differs; not overwritten",
                                   "existing_sha256": _file_sha(path)}
                continue
            # Same parsed payload, only ZIP metadata differs: keep the original bytes and their hash.
            archive = existing
        else:
            path.write_bytes(archive)
            os.chmod(path, 0o600)
        exports[sample] = {"status": "exported", "path": f"selected/{path.name}", "sha256": driver._sha(archive),
                           "preview_audio_sha256": render["file_sha256"],
                           "validation": _validate_archive(archive, project.chain, snapshot, adapter)}
    return exports


def run_case(phase: Phase, case: tuple, adapter, renderer, providers: Providers, freeze: str,
             log: Callable[[str], None] = print) -> dict:
    name, video, (start, end), provenance = case
    url = f"https://www.youtube.com/watch?v={video}"
    directory = phase.root / name
    directory.mkdir(exist_ok=True)
    summary = {"case": name, "interval": [start, end], "provenance": provenance, "steps": {}}
    steps = summary["steps"]

    def call(tag: str, kind: str, model: str, send: Callable[[], dict], **detail) -> dict | None:
        full = f"{name}:{tag}"
        recorded = phase.root / "judgments" / f"{full}.json"
        if recorded.exists():
            result = driver._load(recorded)
            steps[tag] = {"status": _status(result), "recorded_earlier": True}
            return result
        ledger = phase.ledger
        if phase.tag_used(full, ledger):
            steps[tag] = {"status": "attempted_unrecorded", "note": "Reserved earlier; never re-sent."}
            return None
        if ledger["provider_pause"]:
            steps[tag] = {"status": "not_attempted", "reason": "phase paused after HTTP 429"}
            return None
        if len(ledger["attempts"]) >= ledger["ceiling"]:
            steps[tag] = {"status": "not_attempted", "reason": "phase ceiling reached"}
            return None
        entry = phase.reserve(kind, model, {"tag": full, "freeze_sha256": freeze, **detail})
        try:
            result = send()
        except Exception as error:  # noqa: BLE001 - the attempt is counted and recorded, never retried
            result = {"failure_info": {"category": "exception", "exception_type": type(error).__name__},
                      "request_counts": None}
        result.update(tag=full, freeze_sha256=freeze, model_requested=model)
        phase.settle(entry, result)
        driver._record(phase, full, result)
        steps[tag] = {"status": _status(result), "http_status": entry.get("http_status")}
        log(f"{full}: {steps[tag]['status']}")
        return result

    def blocked(reason: str) -> dict:
        summary["outcome"] = "blocked"
        summary["reason"] = reason
        for tag, *_ in STEPS[1:]:
            steps.setdefault(tag, {"status": "not_attempted", "reason": f"case blocked: {reason}"})
        return summary

    reference = call("reference", "reference_profile", PROFILE_MODEL,
                     lambda: providers.reference(url, start, end, PROFILE_MODEL), interval=[start, end])
    ref_profile = _profile(reference)
    if ref_profile is None:
        return blocked("no valid fresh reference profile")
    if ref_profile.guitar_dominance != "clear":
        return blocked(f"reference guitar dominance {ref_profile.guitar_dominance!r}; product refuses refinement")

    interval = Interval(start_seconds=start, end_seconds=end)
    chain = adapter.default_chain([{"profile": ref_profile}])
    errors = adapter.validate_chain(chain)
    if errors:
        return blocked("default chain does not validate: " + "; ".join(errors)[:300])
    variants = refinement.build_variants(adapter, chain, ref_profile)
    roles = {item.role for item in variants}
    if not {"current", "variant", "control"} <= roles:
        return blocked("no distinct local variant and wrong-gain control (shipped refinement eligibility)")
    by_id = {item.id: item for item in variants}
    project = soundshed_module.new_project(
        source=SourceRef(kind="youtube", url=url, display_name=f"Reference {name}", selected_intervals=[interval]),
        analysis=[{"interval": interval.model_dump(), "model": PROFILE_MODEL, "profile": ref_profile.model_dump()}],
        intent={"summary": "Final evaluation product project."}, chain=chain,
        engine_version=renderer.engine_version, name=f"Final {name}",
    )
    stored_project = directory / "base-project.json"
    if stored_project.exists():
        project = ToneProject.model_validate(driver._load(stored_project))
        if adapter.chain_sha256(project.chain) != adapter.chain_sha256(chain):
            return blocked("stored base project differs from the shipped default chain")
    else:
        driver._save(stored_project, project.model_dump(mode="json"))
    try:
        manifest = materialize(phase, name, directory, project, variants, adapter, renderer, freeze)
    except CaseBlocked as error:
        return blocked(str(error))
    summary["artifacts"] = {"manifest": f"{name}/artifacts.json",
                            "manifest_sha256": _file_sha(directory / "artifacts.json"),
                            "variants": {vid: {"role": entry["role"], "chain_sha256": entry["chain_sha256"]}
                                         for vid, entry in manifest["variants"].items()}}
    clips = {sample: {vid: _load_clip(directory, entry["renders"][sample], vid)
                      for vid, entry in manifest["variants"].items()} for sample in SAMPLES}
    hot = refinement.control_is_hotter(ref_profile)
    judgments = []
    for sample in SAMPLES:
        result = call(f"select-{sample}", "relative", SELECTOR_MODEL,
                      lambda sample=sample: providers.relative(
                          model=SELECTOR_MODEL, candidates=clips[sample], seed=secrets.randbelow(2**31),
                          youtube=(url, start, end)), sample=sample)
        judgments.append(refinement.summarize_judgment(sample, result, hot) if result else
                         refinement.SampleJudgment(sample=sample, status="not_attempted",
                                                   reason=steps[f"select-{sample}"]["status"]))
    suggestion = refinement.suggest_from(judgments)
    selected = suggestion.variant_id or "current"
    summary["selection"] = {"selected": selected, "basis": suggestion.basis, "evidence": "weak",
                            "judgments": [item.model_dump(mode="json") for item in judgments]}
    summary["exports"] = export_selected(directory, manifest, selected, adapter)

    def grade(result: dict | None, variant: str) -> dict | None:
        profile = _profile(result)
        if profile is None:
            return None
        return evaluation_v3.compare_profiles_v3(ref_profile, profile, _omitted(by_id[variant].chain))

    per_di = {}
    for sample in SAMPLES:
        result = call(f"eval-{sample}", "absolute_profile", PROFILE_MODEL,
                      lambda sample=sample: providers.profile(clips[sample][selected], PROFILE_MODEL),
                      candidate=selected, sample=sample)
        full = grade(result, selected)
        per_di[sample] = {"step_status": steps[f"eval-{sample}"]["status"], "grade_v3": full,
                          "core_decision": core_decision(full, f"evaluation {steps[f'eval-{sample}']['status']}")}
    control_result = call("control-riff-1", "absolute_profile", PROFILE_MODEL,
                          lambda: providers.profile(clips["riff-1"]["control"], PROFILE_MODEL),
                          candidate="control", sample="riff-1")
    control_grade = grade(control_result, "control")
    repeat = call("reference-repeat", "reference_profile", PROFILE_MODEL,
                  lambda: providers.reference(url, start, end, PROFILE_MODEL), interval=[start, end])
    preview_repeat = call("preview-repeat-riff-1", "absolute_profile", PROFILE_MODEL,
                          lambda: providers.profile(clips["riff-1"][selected], PROFILE_MODEL),
                          candidate=selected, sample="riff-1")
    eval_first = phase.root / "judgments" / f"{name}:eval-riff-1.json"
    eval_profile = _profile(driver._load(eval_first)) if eval_first.exists() else None
    aggregate = evaluation_v3.aggregate_v3([{"outcome": item["core_decision"]["outcome"]}
                                            for item in per_di.values()])
    control = {"grade": {"vs_reference": control_grade}, **control_rejection(control_grade, hot)}
    reference_stability = repeat_stability(ref_profile, _profile(repeat))
    preview_stability = repeat_stability(eval_profile, _profile(preview_repeat))
    summary["evaluation"] = {
        "per_di": per_di, "aggregate_core": aggregate,
        "effects_reported_separately_not_validated": {
            sample: (item["grade_v3"] or {}).get("effects") for sample, item in per_di.items()},
    }
    summary["control"] = control
    summary["stability"] = {"reference_repeat": reference_stability, "preview_repeat": preview_stability}
    summary["acceptance"] = acceptance(aggregate, control, reference_stability, preview_stability)
    summary["outcome"] = summary["acceptance"]["outcome"]
    return summary


def cmd_plan(args) -> None:
    phase = Phase(args.output)
    ledger = phase.ledger
    plan = {
        "freeze_sha256": phase.current_freeze(),
        "freeze_history": [{"sha256": item["sha256"], "status": item["status"], "reason": item.get("reason")}
                           for item in phase._freeze_record()["history"]],
        "ledger": {"ceiling": ledger["ceiling"], "attempted": len(ledger["attempts"]),
                   "provider_pause": ledger["provider_pause"]},
        **config(),
        "per_case_max_new_generate_requests": CALLS_PER_CASE,
        "total_max_new_generate_requests": CALLS_PER_CASE * len(CASES),
        "command": "python3 scripts/tone_copier_final.py --output <phase-dir> run --live",
    }
    driver._save(phase.root / "plan.json", plan)
    print(json.dumps(plan, indent=1))


def cmd_freeze(args, adapter=None, renderer=None) -> str:
    from tone_listener_mcp.workbench.renderer import NativeRenderer

    phase = Phase(args.output, create=True)
    adapter = adapter or soundshed_module.SoundshedAdapter()
    renderer = renderer or NativeRenderer(adapter)
    if hasattr(renderer, "initialize"):
        renderer.initialize()
    digest = phase.freeze_final(renderer, revise=getattr(args, "revise", False), reason=getattr(args, "reason", None))
    print(digest)
    return digest


def cmd_migrate(args) -> None:
    print(json.dumps(migrate_legacy(args.output), indent=1))


def cmd_run(args, adapter=None, renderer=None, providers: Providers | None = None) -> dict:
    from tone_listener_mcp.workbench.renderer import NativeRenderer

    if providers is None:
        driver._require_live(args)
    phase = Phase(args.output)
    if phase.current_freeze() is None:
        raise SystemExit("Run `freeze` and review `plan` first.")
    adapter = adapter or soundshed_module.SoundshedAdapter()
    renderer = renderer or NativeRenderer(adapter)
    if hasattr(renderer, "initialize"):
        renderer.initialize()
    freeze = phase.freeze_final(renderer)
    providers = providers or Providers()
    selected = [case for case in CASES if not args.case or case[0] in args.case]
    report = {"phase": PHASE, "freeze_sha256": freeze, "cases": [], "limitations": list(LIMITATIONS)}
    for case in selected:
        report["cases"].append(run_case(phase, case, adapter, renderer, providers, freeze))
    ledger = phase.ledger
    report["ledger"] = {"ceiling": ledger["ceiling"], "attempted": len(ledger["attempts"]),
                        "by_model": {model: sum(item["model"] == model for item in ledger["attempts"])
                                     for model in (PROFILE_MODEL, SELECTOR_MODEL)},
                        "failed": sum(item["status"] == "failed" for item in ledger["attempts"]),
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
    freeze.add_argument("--revise", action="store_true", help="Supersede an unrun freeze (zero reservations only).")
    freeze.add_argument("--reason")
    sub.add_parser("plan")
    sub.add_parser("migrate-legacy")
    run = sub.add_parser("run")
    run.add_argument("--live", action="store_true")
    run.add_argument("--case", action="append", choices=[case for case, *_ in CASES])
    args = parser.parse_args(argv)
    {"freeze": cmd_freeze, "plan": cmd_plan, "migrate-legacy": cmd_migrate, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

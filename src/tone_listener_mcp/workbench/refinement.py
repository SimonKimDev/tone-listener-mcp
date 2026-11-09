"""Budgeted, blinded refinement of the current chain against the analyzed reference interval.

Local work (variant chains, native renders) is free. The only provider work is one blinded,
level-matched relative judgment per selected DI sample, sent only after the user confirms the
exact plan hash and request count. Judgments are never cached or reused.

Evidence strength comes from offline calibration (tone-copier v3 batches 1-2): gain direction was
detected 6/6 per model; small brightness changes were detected poorly (Flash 0/4, Argon 2/4); and
neither model detected known reverb changes, so ambience is shown but never used for suggestions.
Closest-first ordering was unstable across repeats and is weak evidence, never an optimum.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from tone_listener_mcp.listener import ToneProfile
from tone_listener_mcp.workbench import relative_judge
from tone_listener_mcp.workbench.adapters.soundshed import _PARAMETERS, SoundshedAdapter
from tone_listener_mcp.workbench.evaluation import AXES
from tone_listener_mcp.workbench.models import ChainDraft, DiSample, StrictModel

JUDGE_MODELS = ("gemini-3.6-flash", "gemini-4-argon")
GAIN_ORDER = AXES["gain"]
EVIDENCE = {
    "gain": "calibrated: direction detected 6/6 per model on known DSP changes",
    "brightness": "weak: small brightness changes detected 0/4 (Flash) and 2/4 (Argon)",
    "bass": "uncalibrated", "mids": "uncalibrated", "attack": "uncalibrated", "compression": "uncalibrated",
    "ambience": "unusable: neither model detected known reverb changes; not used for suggestions",
    "rank": "weak: closest-first ordering was unstable across repeats; not an optimum",
}
LESS, MORE = {"less", "much_less"}, {"more", "much_more"}
# Development-derived topology (tone-copier v3 batches 1-2, case-01 only; weak rank evidence): a clean
# amp capture driven hard (no pedal) was ranked closest more often than the stock edge/crunch/overdrive
# recipes. It is offered as one candidate for these gain classes, never applied without a judgment.
PUSHED_CLEAN_GAINS = ("edge-of-breakup", "crunch", "overdrive")
PUSHED_CLEAN = {"amp": {"inputGain": 9.0}, "cabinet": {"highCutHz": 10000.0},
                "profile-eq": {"highGain": 3.0}, "reverb": {"mix": 0.24}}
VariantRole = Literal["current", "variant", "control"]
SearchMode = Literal["explore", "local"]
# Local search keeps every block, resource and order of the current chain and moves only real parameters.
# Input trim of the FIRST enabled NAM capture changes the level driving every downstream capture, which is the
# saturation control actually available (the captures' own Drive/Tone are fixed). Steps are symmetric dB moves.
LOCAL_TRIM_STEPS_DB = (-6.0, -3.0, 3.0, 6.0)
LOCAL_EQ_STEP_DB = 3.0
# Joint moves combine the outer trim steps with the EQ tilt (four sign combinations), using the same operators.
LOCAL_JOINT_TRIM_STEPS_DB = (6.0, -6.0)
SEARCH_DESCRIPTIONS = {
    "explore": "Alternative equipment recipes plus brightness moves; recipe names are not a measured gain order.",
    "local": ("Same equipment, order and notes as the current chain; only the first NAM capture's input trim "
              "(\u00b13/\u00b16 dB), a \u00b13 dB upper-EQ tilt (2 kHz + 8 kHz), or a joint \u00b16 dB trim + "
              "\u00b13 dB tilt moves. "
              "Adopt a result and plan again to continue."),
}


class RefinementVariant(StrictModel):
    id: str
    role: VariantRole
    label: str
    description: str
    chain: ChainDraft
    chain_sha256: str


class RefinementPlan(StrictModel):
    project_id: str
    project_revision: int
    base_chain_sha256: str
    model: str
    samples: list[DiSample]
    reference: dict[str, object]
    variants: list[RefinementVariant]
    search: SearchMode = "explore"
    max_new_requests: int
    uploads_per_request: int
    plan_sha256: str
    evidence: dict[str, str] = Field(default_factory=lambda: dict(EVIDENCE))
    disclosure: str


def _shift_gain(profile: ToneProfile, steps: int) -> ToneProfile | None:
    index = GAIN_ORDER.index(profile.gain) + steps
    if not 0 <= index < len(GAIN_ORDER):
        return None
    return profile.model_copy(update={"gain": GAIN_ORDER[index]})


def _with_params(chain: ChainDraft, changes: dict[str, dict[str, float]]) -> ChainDraft | None:
    """Relative parameter moves on existing nodes, clamped to the adapter's real ranges."""
    nodes = []
    touched = False
    for node in chain.nodes:
        update = changes.get(node.kind)
        if update and node.enabled:
            specs = {spec.id: spec for spec in _PARAMETERS[node.kind]}
            params = dict(node.params)
            for key, delta in update.items():
                if key in params and key in specs:
                    value = params[key] * delta if key == "highCutHz" else params[key] + delta
                    params[key] = float(min(specs[key].maximum, max(specs[key].minimum, value)))
                    touched = True
            node = node.model_copy(update={"params": params})
        nodes.append(node)
    return ChainDraft(revision=chain.revision, nodes=nodes) if touched else None


def _keep_notes(base: ChainDraft, recipe: ChainDraft) -> ChainDraft:
    """Recipe variants keep the omitted-effect notes of the current chain visible."""
    existing = {node.id for node in recipe.nodes}
    notes = [node for node in base.nodes if node.kind == "gear_note" and node.note_role == "omitted_optional"
             and node.id not in existing]
    nodes = [node for node in recipe.nodes if node.kind != "output"] + notes + [
        node for node in recipe.nodes if node.kind == "output"]
    return ChainDraft(revision=base.revision, nodes=nodes)


def _pushed_clean(adapter: SoundshedAdapter, base: ChainDraft, profile: ToneProfile) -> ChainDraft | None:
    bright = profile.brightness in ("bright", "slightly-bright", "balanced")
    recipe = adapter.default_chain([{"profile": profile.model_copy(update={
        "gain": "clean", **({"brightness": "balanced"} if bright else {})})}])
    nodes = []
    for node in recipe.nodes:
        update = PUSHED_CLEAN.get(node.id)
        if update and (bright or node.id == "amp"):
            node = node.model_copy(update={"params": {**node.params, **{
                key: value for key, value in update.items() if key in node.params}}})
        nodes.append(node)
    draft = _keep_notes(base, ChainDraft(revision=base.revision, nodes=nodes))
    return None if adapter.validate_chain(draft) else draft


def _first_capture(chain: ChainDraft):
    return next((node for node in chain.nodes if node.enabled and node.kind in ("nam_fx", "nam_amp")), None)


def _set_param(chain: ChainDraft, node_id: str, key: str, delta: float) -> ChainDraft | None:
    """Absolute-from-current move of one existing parameter, clamped to the adapter range; None if unchanged."""
    nodes, changed = [], False
    for node in chain.nodes:
        if node.id == node_id and key in node.params:
            spec = next(item for item in _PARAMETERS[node.kind] if item.id == key)
            value = float(min(spec.maximum, max(spec.minimum, node.params[key] + delta)))
            if value != node.params[key]:
                node = node.model_copy(update={"params": {**node.params, key: value}})
                changed = True
        nodes.append(node)
    return ChainDraft(revision=chain.revision, nodes=nodes) if changed else None


def _wrong_control(adapter: SoundshedAdapter, profile: ToneProfile | None) -> ChainDraft | None:
    if profile is None or profile.gain not in GAIN_ORDER:
        return None
    hot = bool(control_is_hotter(profile))
    control_profile = profile.model_copy(update={"gain": "high-gain" if hot else "clean",
                                                 "brightness": "dark" if hot else "bright",
                                                 "guitar_dominance": "clear"})
    control = adapter.default_chain([{"profile": control_profile}])
    return None if adapter.validate_chain(control) else control


def _local_moves(adapter: SoundshedAdapter, chain: ChainDraft) -> list[tuple[str, str, str, ChainDraft]]:
    moves: list[tuple[str, str, str, ChainDraft]] = []
    capture = _first_capture(chain)
    if capture is not None and "inputGain" in capture.params:
        for step in LOCAL_TRIM_STEPS_DB:
            draft = _set_param(chain, capture.id, "inputGain", step)
            if draft is None:
                continue
            sign = "plus" if step > 0 else "minus"
            new = next(node for node in draft.nodes if node.id == capture.id).params["inputGain"]
            moves.append((f"trim_{sign}{abs(int(step))}", f"Input trim {step:+g} dB",
                          (f"Same chain; '{capture.title}' input trim {capture.params['inputGain']:+g} \u2192 {new:+g} dB "
                          "(more/less drive into the fixed capture and everything after it)."), draft))
    # Upper-band tilt: the 8 kHz shelf alone sits mostly above typical cab high cuts, so 2 kHz moves with it.
    eq = next((node for node in chain.nodes if node.enabled and node.kind == "parametric_eq"), None)
    targets = [(eq.id, "highMidGain"), (eq.id, "highGain")] if eq is not None else (
        [(capture.id, "treble")] if capture is not None and "treble" in capture.params else [])

    def tilt(draft: ChainDraft, step: float) -> tuple[ChainDraft, list[str]]:
        moved = []
        for node_id, key in targets:
            nxt = _set_param(draft, node_id, key, step)
            if nxt is not None:
                draft, moved = nxt, [*moved, key]
        return draft, moved

    tilts = ((LOCAL_EQ_STEP_DB, "brighter", "Upper EQ +3 dB"), (-LOCAL_EQ_STEP_DB, "darker", "Upper EQ \u22123 dB"))
    for step, name, label in tilts:
        draft, moved = tilt(chain, step)
        if moved:
            moves.append((f"eq_{name}", label,
                          f"Same chain; only '{targets[0][0]}' {'/'.join(moved)} moved {step:+g} dB (clamped to range).",
                          draft))
    # Joint moves: the same two operators applied together. A clamped no-op part makes the chain identical to a
    # single-axis move, which _unique then drops (the single-axis id is kept).
    if capture is not None and "inputGain" in capture.params and targets:
        for trim in LOCAL_JOINT_TRIM_STEPS_DB:
            trimmed = _set_param(chain, capture.id, "inputGain", trim)
            if trimmed is None:
                continue
            new = next(node for node in trimmed.nodes if node.id == capture.id).params["inputGain"]
            for step, name, label in tilts:
                draft, moved = tilt(trimmed, step)
                if not moved:
                    continue
                sign = "plus" if trim > 0 else "minus"
                moves.append((f"joint_{sign}{abs(int(trim))}_{name}", f"Input trim {trim:+g} dB + {label}",
                              (f"Same chain; joint move: '{capture.title}' input trim {capture.params['inputGain']:+g} "
                               f"\u2192 {new:+g} dB and '{targets[0][0]}' {'/'.join(moved)} {step:+g} dB "
                               "(each clamped to range)."), draft))
    return [move for move in moves if not adapter.validate_chain(move[3])]


def build_variants(adapter: SoundshedAdapter, chain: ChainDraft, profile: ToneProfile | None,
                   search: SearchMode = "explore") -> list[RefinementVariant]:
    def variant(identifier: str, role: VariantRole, label: str, description: str, draft: ChainDraft):
        return RefinementVariant(id=identifier, role=role, label=label, description=description,
                                 chain=draft, chain_sha256=adapter.chain_sha256(draft))

    variants = [variant("current", "current", "Current chain", "The chain as it is now.", chain)]
    if search == "local":
        variants += [variant(identifier, "variant", label, description, draft)
                     for identifier, label, description, draft in _local_moves(adapter, chain)]
        control = _wrong_control(adapter, profile)
        if control is not None:
            hot = bool(control_is_hotter(profile))
            variants.append(variant(
                "control", "control", "Deliberately wrong control",
                f"Expected to be judged {'MORE' if hot else 'LESS'} gain and not closest; checks the judgment itself.",
                control))
        return _unique(variants)
    if profile is not None and profile.gain in GAIN_ORDER:
        for steps, identifier, label in ((-1, "recipe_cleaner", "One gain step cleaner recipe"),
                                         (1, "recipe_hotter", "One gain step hotter recipe")):
            shifted = _shift_gain(profile, steps)
            if shifted is None:
                continue
            recipe = adapter.default_chain([{"profile": shifted}])
            if adapter.validate_chain(recipe):
                continue
            variants.append(variant(identifier, "variant", label,
                                    f"Replaces equipment with the local '{shifted.gain}' recipe for the selected profile.",
                                    _keep_notes(chain, recipe)))
    if profile is not None and profile.gain in PUSHED_CLEAN_GAINS:
        pushed = _pushed_clean(adapter, chain, profile)
        if pushed is not None:
            variants.append(variant(
                "pushed_clean_amp", "variant", "Pushed clean amp (development-derived)",
                "Clean amp capture driven +9 dB with no pedal; derived from one development reference "
                "(weak evidence), offered only as a candidate.", pushed))
    for identifier, label, changes in (
        ("brighter", "Brighter", {"parametric_eq": {"highGain": 4, "highMidGain": 2}, "cab_ir": {"highCutHz": 1.3}}),
        ("darker", "Darker", {"parametric_eq": {"highGain": -4, "highMidGain": -2}, "cab_ir": {"highCutHz": 0.7}}),
    ):
        draft = _with_params(chain, changes)
        if draft is not None and not adapter.validate_chain(draft):
            variants.append(variant(identifier, "variant", label,
                                    "Same equipment; high shelf/high-mid EQ and cabinet high cut changed.", draft))
    control = _wrong_control(adapter, profile)
    if control is not None:
        hot = bool(control_is_hotter(profile))
        variants.append(variant(
            "control", "control", "Deliberately wrong control",
            f"Expected to be judged {'MORE' if hot else 'LESS'} gain and not closest; checks the judgment itself.",
            control))
    return _unique(variants)


def _unique(variants: list[RefinementVariant]) -> list[RefinementVariant]:
    unique: dict[str, RefinementVariant] = {}
    for item in variants:
        unique.setdefault(item.chain_sha256, item)  # an identical chain is the same audio, not new evidence
    return list(unique.values())


def plan_sha256(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def judge_identity() -> str:
    return plan_sha256({"instructions": relative_judge.RELATIVE_INSTRUCTIONS,
                        "schema": relative_judge.RelativeJudgment.model_json_schema(),
                        "max_output_tokens": relative_judge.MAX_OUTPUT_TOKENS,
                        "level_rms_dbfs": relative_judge.TARGET_RMS_DBFS})


def control_check(observed: dict[str, dict], hot: bool | None) -> dict:
    control = observed.get("control")
    if control is None or hot is None:
        return {"available": False, "passed": None, "detail": "No control in this judgment; it is unvalidated."}
    expected = MORE if hot else LESS
    passed = control["gain"] in expected and control["rank"] != 1
    return {"available": True, "passed": passed,
            "detail": f"control gain judged {control['gain']}, rank {control['rank']}"}


DISTANCE = {"similar": 0, "less": 1, "more": 1, "much_less": 2, "much_more": 2}


def _closer(item: dict, variant: str, axis: str) -> bool:
    """The variant's own judged distance from the reference is strictly smaller than the current chain's."""
    mine, current = DISTANCE.get(item["observed"][variant][axis]), DISTANCE.get(item["observed"]["current"][axis])
    return mine is not None and current is not None and mine < current


def _not_worse(item: dict, variant: str, axis: str) -> bool:
    mine, current = DISTANCE.get(item["observed"][variant][axis]), DISTANCE.get(item["observed"]["current"][axis])
    return mine is not None and current is not None and mine <= current


def _ranked_above_current(item: dict, variant: str) -> bool:
    return item["observed"][variant]["rank"] < item["observed"]["current"]["rank"]


def _best(per_sample: list[dict], names: list[str], axis: str) -> str:
    return min(names, key=lambda name: (sum(DISTANCE[item["observed"][name][axis]] for item in per_sample),
                                        sum(item["observed"][name]["rank"] for item in per_sample), name))


def suggest(per_sample: list[dict]) -> dict:
    """Suggest only a variant whose OWN measured judgments improve on the current chain in every sample.

    A recipe's name or the current chain's direction is never enough: the proposed variant must be judged
    closer on the axis, not worse on calibrated gain, and ranked above the current chain in every sample.
    """
    if not per_sample or not all(item.get("observed") and item["control"].get("passed") for item in per_sample):
        return {"variant": None,
                "basis": "Not every judgment passed its built-in control and audibility check; nothing is suggested."}
    if not all("current" in item["observed"] for item in per_sample):
        return {"variant": None, "basis": "The current chain was not judged in every sample; nothing is suggested."}
    shared = set.intersection(*(set(item["observed"]) for item in per_sample)) - {"current", "control"}
    names = sorted(shared)
    gain = [name for name in names if all(
        _closer(item, name, "gain") and _ranked_above_current(item, name) for item in per_sample)]
    if gain:
        return {"variant": _best(per_sample, gain, "gain"), "basis": (
            "Judged closer to the reference on gain (calibrated axis) than the current chain and ranked above it "
            "in every sample.")}
    if all(DISTANCE.get(item["observed"]["current"]["gain"]) == 0 for item in per_sample):
        closest = {min(item["observed"], key=lambda key: item["observed"][key]["rank"]) for item in per_sample}
        if len(closest) == 1:
            (winner,) = closest
            if winner in shared and all(item["observed"][winner]["gain"] == "similar" for item in per_sample):
                return {"variant": winner, "basis": (
                    "Ranked closest in every sample with gain judged similar (calibrated axis); "
                    "ranking itself is WEAK evidence.")}
    bright = [name for name in names if all(
        _closer(item, name, "brightness") and _not_worse(item, name, "gain") and _ranked_above_current(item, name)
        for item in per_sample)]
    if bright:
        return {"variant": _best(per_sample, bright, "brightness"), "basis": (
            "Judged closer on brightness (WEAK evidence), not worse on gain, and ranked above the current chain "
            "in every sample.")}
    return {"variant": None, "basis": "No variant measurably improved on the current chain in every sample; keep it."}


def control_is_hotter(profile: ToneProfile | None) -> bool | None:
    if profile is None or profile.gain not in GAIN_ORDER:
        return None
    return GAIN_ORDER.index(profile.gain) <= GAIN_ORDER.index("overdrive")


class VariantRender(StrictModel):
    sample: DiSample
    audio_sha256: str
    duration_seconds: float
    rms: float
    peak: float


class VariantSummary(StrictModel):
    id: str
    role: VariantRole
    label: str
    description: str
    chain_sha256: str
    renders: list[VariantRender] = Field(default_factory=list)


class VariantObservation(StrictModel):
    variant_id: str
    gain: str
    brightness: str
    bass: str
    mids: str
    attack: str
    compression: str
    ambience: str
    rank: int


class SampleJudgment(StrictModel):
    sample: DiSample
    status: Literal["judged", "invalid", "failed", "not_attempted"]
    reason: str | None = None
    reference_guitar_audible: str | None = None
    closest_is_close_match: str | None = None
    observations: list[VariantObservation] = Field(default_factory=list)
    control_passed: bool | None = None
    control_detail: str | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    generate_requests: int = 0
    upload_requests: int = 0
    notes: list[str] = Field(default_factory=list)


class RefinementSuggestion(StrictModel):
    variant_id: str | None
    basis: str


class RefinementRecord(StrictModel):
    id: str
    status: Literal["running", "completed", "partial", "cancelled", "failed"]
    project_id: str
    project_revision: int
    chain_revision: int
    base_chain_sha256: str
    plan_sha256: str
    model: str
    reference: dict[str, object]
    samples: list[DiSample]
    search: SearchMode = "explore"
    confirmed_max_new_requests: int
    new_generate_requests: int = 0
    new_upload_requests: int = 0
    created_at: str
    completed_at: str | None = None
    variants: list[VariantSummary]
    judgments: list[SampleJudgment] = Field(default_factory=list)
    suggestion: RefinementSuggestion | None = None
    evidence: dict[str, str] = Field(default_factory=lambda: dict(EVIDENCE))
    limitations: list[str] = Field(default_factory=lambda: list(LIMITATIONS))


LIMITATIONS = (
    "Every direction and rank is a model judgment, not a listener verdict or perceptual identity.",
    "The reference is a full mix; the judge hears 16 kHz mono audio, so nothing above 8 kHz is judged.",
    "Candidates are RMS level-matched; durations differ by native effect tails, so equal RMS is not equal loudness.",
    "Judgments are never cached or reused; each confirmed run makes fresh, separately counted requests.",
    "Omitted uncertain effects stay omitted in every variant; none was invented.",
)


def observations(result: dict) -> list[VariantObservation]:
    return [VariantObservation(variant_id=identifier, **values)
            for identifier, values in relative_judge.unblind(result).items()]


def summarize_judgment(sample: str, result: dict, hot: bool | None) -> SampleJudgment:
    counts = result.get("request_counts") or {}
    failure = result.get("failure_info") or {}
    base = {"sample": sample, "generate_requests": int(counts.get("generate", 0)),
            "upload_requests": int(counts.get("upload", 0)), "finish_reason": result.get("finish_reason"),
            "http_status": failure.get("http_status") if isinstance(failure.get("http_status"), int) else None}
    judgment = result.get("judgment")
    if judgment is None:
        kind = failure.get("category") or failure.get("exception_type") or "unknown"
        return SampleJudgment(status="failed", reason=f"Judgment unavailable ({kind}); not retried.", **base)
    common = {"reference_guitar_audible": judgment.get("reference_guitar_audible"),
              "closest_is_close_match": judgment.get("closest_is_close_match"),
              "notes": [str(note)[:300] for note in judgment.get("notes", [])][:6]}
    if result.get("invalid"):
        return SampleJudgment(status="invalid", reason="; ".join(result["invalid"])[:300], **common, **base)
    observed = observations(result)
    check = control_check({item.variant_id: item.model_dump() for item in observed}, hot)
    passed = check["passed"]
    if common["reference_guitar_audible"] != "clear":
        passed = False
        check["detail"] = f"reference guitar audibility {common['reference_guitar_audible']!r}; not usable"
    return SampleJudgment(status="judged", observations=observed, control_passed=passed,
                          control_detail=check["detail"], **common, **base)


def suggest_from(judgments: list[SampleJudgment]) -> RefinementSuggestion:
    missing = [item.sample for item in judgments if item.status != "judged"]
    if not judgments or missing:
        return RefinementSuggestion(variant_id=None, basis=(
            "Every planned DI sample needs a usable judgment; missing evidence for "
            f"{', '.join(missing) or 'all samples'} is not agreement. Nothing is suggested."))
    per_sample = [{"observed": {item.variant_id: item.model_dump() for item in judgment.observations},
                   "control": {"passed": judgment.control_passed}}
                  for judgment in judgments if judgment.status == "judged"]
    result = suggest(per_sample)
    return RefinementSuggestion(variant_id=result["variant"], basis=result["basis"])

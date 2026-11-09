"""Tone-copier grading v3: separate core tone from uncertain effects.

v2 (``evaluation.py``) stays frozen for its historical results. v3 differs in two ways:
a definitive large core mismatch is a fail even when another axis is unknown, and
uncertain optional effects can never upgrade partial evidence into an overall pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from tone_listener_mcp.listener import ToneProfile
from tone_listener_mcp.workbench.evaluation import AXES

RUBRIC_VERSION = "tone-copier-v3-2026-11-08"
CORE_AXES = tuple(AXES)
EFFECTS = ("reverb", "delay", "modulation")
PROMINENCE = ["none", "subtle", "moderate", "prominent"]
RULES = {
    "version": RUBRIC_VERSION,
    "maximum_axis_difference": 1,
    "maximum_combined_gain_brightness_mids_difference": 2,
    "known_large_core_mismatch": "fail, even when other axes are uncertain",
    "known_partial_combined_difference": "fail when the known part of gain+brightness+mids already exceeds 2",
    "uncertain_core_axis_without_known_failure": "inconclusive",
    "guitar_not_clear": "inconclusive precondition; known differences are still reported",
    "uncertain_or_possible_effect": "effects unresolved; overall at best core_match_effects_unresolved, never pass",
    "prominent_effect_contradiction": "fail only when both presences are known (present vs absent)",
    "unknown_presence_effect_difference": "unresolved, never a contradiction",
    "modulation_family_contradiction": "fail only when modulation is present in both with known families",
    "unknown_outcome_aggregation": "rejected (ValueError); never defaults to pass",
    "identity_claim": "same-model categorical regression signal only, not perceptual identity or gear identification",
}
OUTCOMES = ("pass", "core_match_effects_unresolved", "inconclusive", "fail")


def compare_profiles_v3(
    reference: ToneProfile, candidate: ToneProfile, omitted_effects: Iterable[str] = ()
) -> dict:
    omitted = sorted(set(omitted_effects))
    core = {}
    unknown, failures = [], []
    for axis, ordered in AXES.items():
        left, right = getattr(reference, axis), getattr(candidate, axis)
        difference = None if "uncertain" in (left, right) else abs(ordered.index(left) - ordered.index(right))
        core[axis] = {"reference": left, "candidate": right, "difference": difference}
        if difference is None:
            unknown.append(axis)
        elif difference > 1:
            failures.append(f"{axis} differs by {difference} steps")
    known_core = [core[axis]["difference"] for axis in ("gain", "brightness", "mids")
                  if core[axis]["difference"] is not None]
    if sum(known_core) > 2:
        failures.append(f"gain+brightness+mids known difference {sum(known_core)} exceeds 2")
    core_outcome = "fail" if failures else "inconclusive" if unknown else "pass"

    effects, unresolved, mismatches = {}, [], []
    for field in EFFECTS:
        left, right = getattr(reference, field), getattr(candidate, field)
        is_unresolved = any(item.presence in ("possible", "uncertain") or item.prominence == "uncertain"
                            for item in (left, right))
        known_presence = all(item.presence in ("present", "absent") for item in (left, right))
        contradiction = known_presence and left.presence != right.presence and any(
            item.presence == "present" and item.prominence in ("moderate", "prominent") for item in (left, right)
        )
        delta = None if is_unresolved else abs(PROMINENCE.index(left.prominence) - PROMINENCE.index(right.prominence))
        status = "mismatch" if contradiction or (delta is not None and delta > 1) else (
            "unresolved" if is_unresolved else "match")
        effects[field] = {
            "reference": left.model_dump(), "candidate": right.model_dump(), "difference": delta,
            "contradiction": contradiction, "status": status, "omitted_in_candidate_chain": field in omitted,
        }
        if status == "mismatch":
            mismatches.append(f"{field} {'contradiction' if contradiction else f'differs by {delta}'}")
        elif status == "unresolved":
            unresolved.append(field)
    if any(item.modulation.presence != "absent" for item in (reference, candidate)):
        both_present = all(item.modulation.presence == "present" for item in (reference, candidate))
        family_uncertain = "uncertain" in (reference.modulation_type, candidate.modulation_type)
        families_differ = reference.modulation_type != candidate.modulation_type
        family_contradiction = (
            both_present and families_differ and not family_uncertain
            and any(item.modulation.prominence in ("moderate", "prominent") for item in (reference, candidate))
        )
        presence_unknown = any(item.modulation.presence in ("possible", "uncertain") for item in (reference, candidate))
        family_unresolved = family_uncertain or (families_differ and presence_unknown)
        if family_contradiction:
            mismatches.append("modulation family contradiction")
        elif (family_unresolved and "modulation" not in unresolved
              and effects["modulation"]["status"] != "mismatch"):
            unresolved.append("modulation")
    guitar_clear = reference.guitar_dominance == "clear" and candidate.guitar_dominance == "clear"
    if not guitar_clear:
        outcome = "inconclusive"
    elif failures or mismatches:
        outcome = "fail"
    elif core_outcome == "inconclusive":
        outcome = "inconclusive"
    elif unresolved:
        outcome = "core_match_effects_unresolved"
    else:
        outcome = "pass"
    return {
        "rubric_version": RUBRIC_VERSION,
        "outcome": outcome,
        "guitar_clear_in_both": guitar_clear,
        "core": {"outcome": core_outcome, "axes": core, "known_failures": failures, "unknown_axes": unknown},
        "effects": {"axes": effects, "mismatches": mismatches, "unresolved": unresolved},
        "perceptual_match_claim": False,
        "limitations": RULES["identity_claim"],
    }


def aggregate_v3(results: Iterable[Mapping]) -> str:
    """Worst-first aggregation; partial evidence never becomes an overall pass."""
    outcomes = [item.get("outcome") for item in results]
    unknown = sorted({str(item) for item in outcomes if item not in OUTCOMES})
    if unknown:
        raise ValueError(f"Unknown or unperformed v3 outcomes cannot be aggregated: {', '.join(unknown)}")
    if not outcomes:
        return "inconclusive"
    for outcome in ("fail", "inconclusive", "core_match_effects_unresolved"):
        if outcome in outcomes:
            return outcome
    return "pass"

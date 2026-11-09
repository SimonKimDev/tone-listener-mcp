"""Frozen conservative categorical rubric for opt-in, blinded live evaluation."""

from __future__ import annotations

from tone_listener_mcp.listener import ToneProfile

RUBRIC_VERSION = 1
AXES = {
    "gain": ["clean", "edge-of-breakup", "crunch", "overdrive", "distortion", "high-gain"],
    "brightness": ["dark", "slightly-dark", "balanced", "slightly-bright", "bright"],
    "bass_level": ["low", "typical", "high"],
    "bass_tightness": ["loose", "neutral", "tight"],
    "mids": ["scooped", "neutral", "present", "forward"],
    "attack": ["soft", "rounded", "balanced", "sharp"],
    "compression": ["dynamic", "moderate", "compressed", "squashed"],
    "sustain": ["short", "typical", "long"],
}
RULES = {
    "version": RUBRIC_VERSION,
    "maximum_axis_difference": 1,
    "maximum_combined_gain_brightness_mids_difference": 2,
    "guitar_dominance": "clear required in both judgments",
    "uncertain_axis_or_possible_effect": "inconclusive, never a pass",
    "prominent_effect_contradiction": "fail",
    "negative_control_pass_or_inconclusive": "uncalibrated evaluator; inconclusive",
    "reference_or_preview_repeat_not_pass": "unstable evaluator; inconclusive",
    "identity_claim": "same-model regression signal only, not perceptual identity or gear identification",
}


def compare_profiles(reference: ToneProfile, candidate: ToneProfile) -> dict:
    differences = {}
    uncertain = (
        reference.guitar_dominance != "clear" or candidate.guitar_dominance != "clear"
    )
    failed = False
    for axis, ordered in AXES.items():
        left, right = getattr(reference, axis), getattr(candidate, axis)
        difference = None if "uncertain" in (left, right) else abs(ordered.index(left) - ordered.index(right))
        differences[axis] = {"reference": left, "candidate": right, "difference": difference}
        uncertain |= difference is None
        failed |= difference is not None and difference > 1
    core = [differences[field]["difference"] for field in ("gain", "brightness", "mids")]
    if all(value is not None for value in core):
        failed |= sum(core) > 2
    levels = ["none", "subtle", "moderate", "prominent"]
    for field in ("reverb", "delay", "modulation"):
        left, right = getattr(reference, field), getattr(candidate, field)
        unresolved = any(item.presence in ("possible", "uncertain") or item.prominence == "uncertain"
                         for item in (left, right))
        contradiction = (
            left.presence != right.presence
            and any(item.presence == "present" and item.prominence in ("moderate", "prominent")
                    for item in (left, right))
        )
        delta = None if unresolved else abs(levels.index(left.prominence) - levels.index(right.prominence))
        differences[field] = {
            "reference": left.model_dump(), "candidate": right.model_dump(),
            "difference": delta, "contradiction": contradiction,
        }
        uncertain |= unresolved
        failed |= contradiction or (delta is not None and delta > 1)
    if any(item.modulation.presence != "absent" for item in (reference, candidate)):
        unknown_family = "uncertain" in (reference.modulation_type, candidate.modulation_type)
        family_contradiction = (
            reference.modulation_type != candidate.modulation_type and not unknown_family
            and any(item.modulation.prominence in ("moderate", "prominent")
                    for item in (reference, candidate))
        )
        uncertain |= unknown_family
        failed |= family_contradiction
        differences["modulation_type"] = {
            "reference": reference.modulation_type, "candidate": candidate.modulation_type,
            "contradiction": family_contradiction,
        }
    return {
        "outcome": "inconclusive" if uncertain else "fail" if failed else "pass",
        "axis_differences": differences,
        "limitations": RULES["identity_claim"],
    }

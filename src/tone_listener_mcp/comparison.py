"""Loudness-matched comparison of candidate B against reference A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.listener import Progress, listen_comparison, listen_youtube_comparison
from tone_listener_mcp.metrics import (
    BANDS,
    GATE_ABSOLUTE_DBFS,
    GATING_METHOD,
    WINDOW_SECONDS,
    gated_power,
    measure,
)

BAND_LABELS = {
    "below_80": "below 80 Hz",
    "80_250": "80-250 Hz",
    "250_800": "250-800 Hz",
    "800_2000": "800 Hz-2 kHz",
    "2000_5000": "2-5 kHz",
    "5000_8000": "5-8 kHz",
    "above_8000": "above 8 kHz",
}


BAND_CHARACTER = {
    "below_80": "sub-bass weight and rumble",
    "80_250": "low-end thump and boom",
    "250_800": "body, boxiness and mud",
    "800_2000": "midrange honk and note presence",
    "2000_5000": "bite, pick attack and harshness",
    "5000_8000": "fizz and sizzle",
    "above_8000": "air, hiss and cymbal-like fizz",
}


MatchingStrategy = Literal["active", "whole"]
MATCHING_METHODS = {
    "active": f"Active-region RMS: {GATING_METHOD}",
    "whole": "Whole-clip RMS, including silence and tails",
}
MISMATCH_TOLERANCE = 0.1
# Bands below this relative energy (dB re: total spectral energy) in both A and B are
# numerically unstable: tiny absolute changes produce large ratios, so they get no hints.
BAND_EVIDENCE_FLOOR_DB = -50.0
HINT_THRESHOLD_DB = 1.0


def band_evidence(band_a: dict, band_b: dict) -> str:
    if not band_a["available"] or not band_b["available"]:
        return "unavailable"
    levels = [band["relative_db"] for band in (band_a, band_b) if band["relative_db"] is not None]
    if not levels or max(levels) < BAND_EVIDENCE_FLOOR_DB:
        return "insufficient_energy"
    return "sufficient"


def delta(a: float | None, b: float | None) -> float | None:
    return b - a if a is not None and b is not None else None


def matching_power(clip: AudioClip, strategy: MatchingStrategy) -> float:
    if strategy == "active":
        return gated_power(clip)[0]
    if strategy == "whole":
        return float(np.mean(clip.samples.astype(np.float64) ** 2))
    raise ValueError("matching must be 'active' or 'whole'")


def activity(a: AudioClip, b: AudioClip) -> dict:
    """Report duration and gated-activity differences that make tonal comparison less reliable."""
    seconds = [gated_power(clip)[1] for clip in (a, b)]

    def differs(x: float, y: float) -> bool:
        return abs(x - y) > max(WINDOW_SECONDS, MISMATCH_TOLERANCE * max(x, y))

    return {
        "duration_seconds_a": a.duration,
        "duration_seconds_b": b.duration,
        "active_seconds_a": seconds[0],
        "active_seconds_b": seconds[1],
        "duration_mismatch": differs(a.duration, b.duration),
        "activity_mismatch": differs(*seconds),
    }


def loudness_match(
    a: AudioClip, b: AudioClip, strategy: MatchingStrategy = "active"
) -> tuple[AudioClip, dict]:
    a_power, b_power = matching_power(a, strategy), matching_power(b, strategy)
    if not a_power or not b_power:
        return b, {
            "applied": False,
            "strategy": strategy,
            "method": MATCHING_METHODS[strategy],
            "gain_db_applied_to_b": None,
            "reason": (
                "Active-region RMS matching is undefined: a clip has no window above "
                f"{GATE_ABSOLUTE_DBFS:g} dBFS"
                if strategy == "active"
                else "RMS matching is undefined when either clip is silent"
            ),
        }
    gain = float(np.sqrt(a_power / b_power))
    matched = AudioClip(
        b.samples.astype(np.float64) * gain, b.sample_rate, b.source, b.start_seconds
    )
    return matched, {
        "applied": True,
        "strategy": strategy,
        "method": MATCHING_METHODS[strategy],
        "gain_db_applied_to_b": float(20 * np.log10(gain)),
        "reason": None,
    }


def compare(
    a: AudioClip,
    b: AudioClip,
    focus: str | None = None,
    matching: MatchingStrategy = "active",
    progress: Progress | None = None,
) -> dict:
    matched_b, match_info = loudness_match(a, b, matching)
    match_info["activity"] = activity(a, b)
    metrics_a, metrics_b = measure(a), measure(b)
    matched_metrics_b = measure(matched_b)
    spectrum_a, spectrum_b = metrics_a["spectrum"], matched_metrics_b["spectrum"]
    # Relative band powers use all frequencies through each clip's Nyquist.
    same_bandwidth = spectrum_a["nyquist_hz"] == spectrum_b["nyquist_hz"]
    band_deltas = {
        name: (
            delta(spectrum_a["bands"][name]["relative_db"], spectrum_b["bands"][name]["relative_db"])
            if same_bandwidth
            else None
        )
        for name, _, _ in BANDS
    }
    crest_key = "gated_crest_factor_db" if matching == "active" else "crest_factor_db"
    crest_delta = delta(metrics_a["level"][crest_key], matched_metrics_b["level"][crest_key])
    evidence = {
        name: (
            band_evidence(spectrum_a["bands"][name], spectrum_b["bands"][name])
            if same_bandwidth
            else "unavailable"
        )
        for name, _, _ in BANDS
    }
    hints = []
    for name, value in band_deltas.items():
        if evidence[name] == "sufficient" and value is not None and abs(value) >= HINT_THRESHOLD_DB:
            direction = "cut" if value > 0 else "boost"
            label = BAND_LABELS[name]
            observation = f"B has {value:+.1f} dB relative band energy versus A in {label}."
            alternative = (
                "Saturation harmonics, performance and ambience can also change band energy; "
                "compare drive/dynamics first. This is not a diagnosis of EQ."
            )
            listen_for = (
                f"Whether B's {BAND_CHARACTER[name]} now matches A without other bands "
                "becoming exposed."
            )
            hints.append({
                "family": "eq",
                "band": name,
                "band_label": label,
                "direction": direction,
                "delta_db": value,
                "observation": observation,
                "alternative_explanation": alternative,
                "listen_for": listen_for,
                "text": (
                    f"{observation} {alternative} Only if EQ is the cause, try a relative "
                    f"{direction} in B; this is not a knob value or proof that EQ can recreate A."
                ),
            })
    hints.sort(key=lambda hint: -abs(hint["delta_db"]))
    for priority, hint in enumerate(hints, 1):
        hint["priority"] = priority
    stereo_a, stereo_b = metrics_a["stereo"], matched_metrics_b["stereo"]
    stereo_deltas = (
        {key: delta(stereo_a[key], stereo_b[key])
         for key in ("lr_correlation", "side_mid_energy_ratio_db")}
        if stereo_a is not None and stereo_b is not None else None
    )
    result = {
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "matched_metrics_b": matched_metrics_b,
        "loudness_match": match_info,
        "deltas_b_minus_a": {
            "relative_band_db": band_deltas,
            "band_evidence": evidence,
            "band_evidence_floor_db": BAND_EVIDENCE_FLOOR_DB,
            "centroid_hz": delta(spectrum_a["centroid_hz"], spectrum_b["centroid_hz"]),
            "rolloff_85_hz": delta(spectrum_a["rolloff_85_hz"], spectrum_b["rolloff_85_hz"]),
            "crest_factor_db": crest_delta,
            "crest_factor_basis": crest_key,
            "stereo": stereo_deltas,
        },
        "eq_direction_hints": hints,
        "cautions": [
            "Use the same DI/phrase; pitch, performance and arrangement affect spectral differences.",
            "Relative band energies are not calibrated EQ knob settings.",
        ],
    }
    if not match_info["applied"]:
        result["cautions"].append(match_info["reason"])
    skipped = [BAND_LABELS[name] for name, value in evidence.items()
               if value == "insufficient_energy"]
    if skipped:
        result["cautions"].append(
            "No EQ hints for bands with negligible energy in both A and B (below "
            f"{BAND_EVIDENCE_FLOOR_DB:.0f} dB relative): {', '.join(skipped)}. Their raw deltas "
            "are numerically unstable and are reported only for completeness."
        )
    if match_info["activity"]["duration_mismatch"] or match_info["activity"]["activity_mismatch"]:
        result["cautions"].append(
            "A and B differ in duration or active (gated) duration; trim both to the same phrase. "
            + ("Active-region matching ignores silence and sub-gate tails, but longer ambience "
               "tails still affect spectra." if matching == "active" else
               "Whole-clip matching lets silence and tails change the applied gain.")
        )
    if not same_bandwidth:
        result["cautions"].append(
            "Relative band energies use different Nyquist frequencies, so band deltas "
            "and EQ direction hints are unavailable."
        )
    result["available_analyses"] = {
        "semantic_listening": True,
        "rms_matching": True,
        "spectral_deltas": True,
    }
    return {**result, **listen_comparison(a, matched_b, focus, matching, progress)}


@dataclass(frozen=True)
class YouTubeReference:
    """A public YouTube interval used as reference A; no local waveform is available."""

    url: str
    start: float | None = None
    end: float | None = None


Reference = AudioClip | YouTubeReference

LOCAL_REFERENCE_REQUIRED = (
    "Requires a local reference waveform; a YouTube reference is only listened to by Gemini, "
    "never downloaded or decoded locally"
)


def compare_reference(
    reference: Reference,
    candidate: AudioClip,
    focus: str | None = None,
    matching: MatchingStrategy = "active",
    progress: Progress | None = None,
) -> dict:
    """Compare candidate B with a local or YouTube reference A.

    `available_analyses` states which parts a source combination supports: RMS matching
    and spectral deltas need both waveforms locally; semantic listening needs a Gemini key.
    """
    if isinstance(reference, AudioClip):
        return compare(reference, candidate, focus, matching, progress)
    listened = listen_youtube_comparison(
        reference.url, reference.start, reference.end, candidate, focus, progress
    )
    return {
        "available_analyses": {
            "semantic_listening": True,
            "rms_matching": False,
            "spectral_deltas": False,
        },
        "unavailable_analyses": {
            "rms_matching": LOCAL_REFERENCE_REQUIRED,
            "spectral_deltas": LOCAL_REFERENCE_REQUIRED,
        },
        "metrics_b": measure(candidate),
        "cautions": [
            (
                "A and B are not loudness-matched; the YouTube reference uses a lossy codec and "
                "often a full mix, so judge tone direction rather than level or top-end detail."
            ),
            "Use the same riff or phrase; performance and arrangement affect comparisons.",
            (
                "For objective RMS matching and spectral deltas, use compare_audio with a local "
                "recording of the reference that you are permitted to use."
            ),
        ],
        **listened,
    }

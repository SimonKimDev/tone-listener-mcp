"""Loudness-matched comparison of candidate B against reference A."""

from __future__ import annotations

import numpy as np

from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.listener import listen_comparison
from tone_listener_mcp.metrics import BANDS, measure

BAND_LABELS = {
    "below_80": "below 80 Hz",
    "80_250": "80-250 Hz",
    "250_800": "250-800 Hz",
    "800_2000": "800 Hz-2 kHz",
    "2000_5000": "2-5 kHz",
    "5000_8000": "5-8 kHz",
    "above_8000": "above 8 kHz",
}


def delta(a: float | None, b: float | None) -> float | None:
    return b - a if a is not None and b is not None else None


def loudness_match(a: AudioClip, b: AudioClip) -> tuple[AudioClip, dict]:
    a_power = float(np.mean(a.samples.astype(np.float64) ** 2))
    b_power = float(np.mean(b.samples.astype(np.float64) ** 2))
    if not a_power or not b_power:
        return b, {
            "applied": False,
            "gain_db_applied_to_b": None,
            "reason": "RMS matching is undefined when either clip is silent",
        }
    gain = float(np.sqrt(a_power / b_power))
    matched = AudioClip(
        b.samples.astype(np.float64) * gain, b.sample_rate, b.source, b.start_seconds
    )
    return matched, {
        "applied": True,
        "gain_db_applied_to_b": float(20 * np.log10(gain)),
        "reason": None,
    }


def compare(a: AudioClip, b: AudioClip, focus: str | None = None) -> dict:
    matched_b, matching = loudness_match(a, b)
    metrics_a, metrics_b = measure(a), measure(b)
    matched_metrics_b = measure(matched_b)
    spectrum_a, spectrum_b = metrics_a["spectrum"], matched_metrics_b["spectrum"]
    band_deltas = {
        name: delta(
            spectrum_a["bands"][name]["relative_db"],
            spectrum_b["bands"][name]["relative_db"],
        )
        for name, _, _ in BANDS
    }
    hints = []
    for name, value in band_deltas.items():
        if value is not None and abs(value) >= 1:
            direction = "Cut" if value > 0 else "Boost"
            hints.append(
                f"B has {value:+.1f} dB relative band energy versus A in {BAND_LABELS[name]}. "
                f"{direction} this region in B toward A; this is an EQ direction, not a knob value."
            )
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
        "loudness_match": matching,
        "deltas_b_minus_a": {
            "relative_band_db": band_deltas,
            "centroid_hz": delta(spectrum_a["centroid_hz"], spectrum_b["centroid_hz"]),
            "rolloff_85_hz": delta(spectrum_a["rolloff_85_hz"], spectrum_b["rolloff_85_hz"]),
            "crest_factor_db": delta(
                metrics_a["level"]["crest_factor_db"], matched_metrics_b["level"]["crest_factor_db"]
            ),
            "stereo": stereo_deltas,
        },
        "eq_direction_hints": hints,
        "cautions": [
            "Use the same DI/phrase; pitch, performance and arrangement affect spectral differences.",
            "Relative band energies are not calibrated EQ knob settings.",
        ],
    }
    if not matching["applied"]:
        result["cautions"].append(matching["reason"])
    return {**result, **listen_comparison(a, matched_b, focus)}

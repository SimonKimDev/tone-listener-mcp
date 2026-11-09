"""Local level and dynamics measurements; undefined values are JSON null."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from tone_listener_mcp.audio_io import AudioClip

BANDS = (
    ("below_80", 0, 80),
    ("80_250", 80, 250),
    ("250_800", 250, 800),
    ("800_2000", 800, 2000),
    ("2000_5000", 2000, 5000),
    ("5000_8000", 5000, 8000),
    ("above_8000", 8000, float("inf")),
)


WINDOW_SECONDS = 0.05
GATE_ABSOLUTE_DBFS = -70.0
GATE_RELATIVE_DB = -20.0
GATING_METHOD = (
    "Unweighted 50 ms windows above -70 dBFS and within 20 dB of their mean "
    "(BS.1770-style two-stage gate; not LUFS)"
)


def power_db(power: float) -> float | None:
    return float(10 * np.log10(power)) if power > 0 else None


def gated_power(clip: AudioClip) -> tuple[float, float]:
    """Return mean power of the active (gated) region and the gated duration in seconds.

    Silence padding and decaying tails below the gate do not change this power, so
    loudness matching on it compares the played portions of two clips.
    """
    samples = clip.samples.astype(np.float64)
    frames = samples.shape[1]
    window = max(1, round(clip.sample_rate * WINDOW_SECONDS))
    starts = np.arange(0, frames, window)
    sums = np.add.reduceat(np.sum(samples**2, axis=0), starts)
    counts = np.diff(np.append(starts, frames)) * samples.shape[0]
    powers = sums / counts
    gate = powers > 10 ** (GATE_ABSOLUTE_DBFS / 10)
    if not gate.any():
        return 0.0, 0.0
    ungated_mean = np.sum(sums[gate]) / np.sum(counts[gate])
    gate &= powers >= ungated_mean * 10 ** (GATE_RELATIVE_DB / 10)
    gated_frames = np.sum(counts[gate]) / samples.shape[0]
    return float(np.sum(sums[gate]) / np.sum(counts[gate])), float(gated_frames / clip.sample_rate)


def level_metrics(clip: AudioClip) -> dict:
    samples = clip.samples.astype(np.float64)
    power = float(np.mean(samples**2))
    peak_power = float(np.max(np.abs(samples)) ** 2)
    active_power, active_seconds = gated_power(clip)
    window = max(1, round(clip.sample_rate * WINDOW_SECONDS))
    short_powers = np.array(
        [np.mean(samples[:, offset : offset + window] ** 2)
         for offset in range(0, samples.shape[1], window)]
    )
    percentiles = np.percentile(short_powers, [10, 50, 90])
    short_db = [power_db(float(value)) for value in percentiles]
    active = short_powers[short_powers > 1e-9]
    active_db = 10 * np.log10(active) if active.size else np.array([])
    robust_range = (
        float(np.diff(np.percentile(active_db, [10, 90]))[0]) if active.size >= 3 else None
    )
    return {
        "peak_dbfs": power_db(peak_power),
        "rms_dbfs": power_db(power),
        "crest_factor_db": power_db(peak_power / power) if power else None,
        "gated_rms_dbfs": power_db(active_power),
        "gated_crest_factor_db": power_db(peak_power / active_power) if active_power else None,
        "gated_duration_seconds": active_seconds,
        "gating_method": GATING_METHOD,
        "short_term_window_seconds": window / clip.sample_rate,
        "short_term_rms_dbfs": dict(zip(("p10", "p50", "p90"), short_db)),
        "short_term_rms_range_db": robust_range,
        "short_term_range_method": "p90-p10 of window dBFS above -90 dBFS; minimum 3 windows",
        "active_window_fraction": float(active.size / short_powers.size),
        "silent_window_fraction": float(np.mean(short_powers == 0)),
    }


def spectrum_metrics(clip: AudioClip) -> dict:
    frequencies, density = welch(
        clip.samples.astype(np.float64),
        fs=clip.sample_rate,
        nperseg=min(4096, clip.samples.shape[1]),
        detrend=False,
        axis=-1,
    )
    # Average channel power, not waveforms: antiphase stereo must not disappear.
    powers = np.mean(density, axis=0)
    total = float(np.sum(powers))
    nyquist_hz = clip.sample_rate / 2
    bands = {}
    for name, low, high in BANDS:
        available = low < nyquist_hz
        selected = (frequencies >= low) & (frequencies < high)
        energy = float(np.sum(powers[selected]))
        bands[name] = {
            "available": available,
            "effective_range_hz": [low, min(high, nyquist_hz)] if available else None,
            "relative_db": power_db(energy / total) if total and available else None,
        }
    positive = np.maximum(powers, np.finfo(float).tiny)
    return {
        "method": "Welch, Hann window, up to 4096 frames, channel-averaged power",
        "nyquist_hz": nyquist_hz,
        "resolution_hz": clip.sample_rate / min(4096, clip.samples.shape[1]),
        "centroid_hz": float(np.sum(frequencies * powers) / total) if total else None,
        "rolloff_85_hz": (
            float(frequencies[np.searchsorted(np.cumsum(powers), total * 0.85)])
            if total else None
        ),
        "flatness": float(np.exp(np.mean(np.log(positive))) / np.mean(powers))
        if total else None,
        "bands": bands,
    }


def stereo_metrics(clip: AudioClip) -> dict | None:
    if clip.samples.shape[0] != 2:
        return None
    left, right = clip.samples.astype(np.float64)
    mid = (left + right) / 2
    side = (left - right) / 2
    mid_power = float(np.mean(mid**2))
    side_power = float(np.mean(side**2))
    variable = np.std(left) > 0 and np.std(right) > 0
    return {
        "lr_correlation": float(np.corrcoef(left, right)[0, 1]) if variable else None,
        "side_mid_energy_ratio_db": power_db(side_power / mid_power) if mid_power else None,
        "mid_rms_dbfs": power_db(mid_power),
        "side_rms_dbfs": power_db(side_power),
    }


def measure(clip: AudioClip) -> dict:
    return {
        "clip": clip.metadata(),
        "level": level_metrics(clip),
        "spectrum": spectrum_metrics(clip),
        "stereo": stereo_metrics(clip),
    }

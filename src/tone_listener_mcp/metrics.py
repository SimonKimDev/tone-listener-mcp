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


def power_db(power: float) -> float | None:
    return float(10 * np.log10(power)) if power > 0 else None


def level_metrics(clip: AudioClip) -> dict:
    samples = clip.samples.astype(np.float64)
    power = float(np.mean(samples**2))
    peak_power = float(np.max(np.abs(samples)) ** 2)
    window = max(1, round(clip.sample_rate * 0.05))
    short_powers = np.array(
        [np.mean(samples[:, offset : offset + window] ** 2)
         for offset in range(0, samples.shape[1], window)]
    )
    percentiles = np.percentile(short_powers, [10, 50, 90])
    short_db = [power_db(float(value)) for value in percentiles]
    return {
        "peak_dbfs": power_db(peak_power),
        "rms_dbfs": power_db(power),
        "crest_factor_db": power_db(peak_power / power) if power else None,
        "short_term_window_seconds": window / clip.sample_rate,
        "short_term_rms_dbfs": dict(zip(("p10", "p50", "p90"), short_db)),
        "short_term_rms_range_db": (
            power_db(float(np.max(short_powers) / np.min(short_powers)))
            if np.min(short_powers) > 0 else None
        ),
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
    bands = {}
    for name, low, high in BANDS:
        available = low < clip.sample_rate / 2
        selected = (frequencies >= low) & (frequencies < high)
        energy = float(np.sum(powers[selected]))
        bands[name] = {
            "available": available,
            "relative_db": power_db(energy / total) if total and available else None,
        }
    positive = np.maximum(powers, np.finfo(float).tiny)
    return {
        "method": "Welch, Hann window, up to 4096 frames, channel-averaged power",
        "nyquist_hz": clip.sample_rate / 2,
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


def measure(clip: AudioClip) -> dict:
    return {
        "clip": clip.metadata(),
        "level": level_metrics(clip),
        "spectrum": spectrum_metrics(clip),
    }

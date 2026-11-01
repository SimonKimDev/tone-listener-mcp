"""Local level and dynamics measurements; undefined values are JSON null."""

from __future__ import annotations

import numpy as np

from tone_listener_mcp.audio_io import AudioClip


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


def measure(clip: AudioClip) -> dict:
    return {"clip": clip.metadata(), "level": level_metrics(clip)}

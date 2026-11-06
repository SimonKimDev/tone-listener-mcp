"""Generate known gain, drive, EQ and ambience changes without network access."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

from tone_listener_mcp.audio_io import load_audio


def synthetic_phrase(seconds: float, rate: int = 48000) -> np.ndarray:
    samples = np.zeros(round(seconds * rate), dtype=np.float64)
    for onset in np.arange(0, seconds, 0.5):
        offset = round(onset * rate)
        t = np.arange(min(round(rate * 0.45), len(samples) - offset)) / rate
        note = sum(np.sin(2 * np.pi * 220 * harmonic * t) / harmonic
                   for harmonic in range(1, 13))
        envelope = (1 - np.exp(-t * 500)) * np.exp(-t * 6)
        samples[offset : offset + len(t)] += note * envelope * 0.2
    return samples


def generate(output: Path, seconds: float = 4, source: str | None = None) -> dict[str, Path]:
    if not math.isfinite(seconds) or not 0.1 <= seconds <= 30:
        raise ValueError("seconds must be finite and between 0.1 and 30")
    if source:
        loaded = load_audio(source, end=seconds)
        samples = loaded.samples.astype(np.float64)
        rate = loaded.sample_rate
    else:
        samples = synthetic_phrase(seconds)[None, :]
        rate = 48000
    low = sosfilt(butter(2, min(1000, rate / 4), fs=rate, output="sos"), samples, axis=-1)
    impulse = np.zeros(round(rate * 0.25) + 1)
    impulse[0] = 1
    for delay, amplitude in ((0.035, 0.32), (0.071, 0.24), (0.113, 0.18), (0.173, 0.12)):
        impulse[round(delay * rate)] = amplitude
    variants = {
        "reference": samples,
        "gain_quiet": samples * 0.5,
        "gain_loud": samples * 1.5,
        "drive_light": np.tanh(samples * 2),
        "drive_heavy": np.tanh(samples * 8),
        "eq_dark": 0.5 * samples + 0.5 * low,
        "eq_bright": samples + (samples - low),
        "reverb": np.stack([fftconvolve(channel, impulse) for channel in samples]),
    }
    peak = max(float(np.max(np.abs(variant))) for variant in variants.values())
    headroom = min(1, 0.95 / peak) if peak else 1
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, variant in variants.items():
        path = output / f"{name}.wav"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite validation recording: {path}")
        sf.write(path, (variant * headroom).T, rate, subtype="FLOAT")
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", help="Optional DI recording; defaults to a synthetic plucked phrase")
    parser.add_argument("--seconds", type=float, default=4)
    args = parser.parse_args()
    for name, path in generate(args.output, args.seconds, args.source).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

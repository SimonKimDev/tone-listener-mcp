"""Bounded, channel-preserving audio loading."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # [channels, frames], float32
    sample_rate: int
    source: str
    start_seconds: float = 0.0

    @property
    def duration(self) -> float:
        return self.samples.shape[1] / self.sample_rate

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "sample_rate": self.sample_rate,
            "channels": self.samples.shape[0],
            "duration_seconds": self.duration,
            "start_seconds": self.start_seconds,
        }


def load_audio(path: str, start: float | None = None, end: float | None = None) -> AudioClip:
    """Read a selected interval, rejecting clips over the configured duration limit."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Audio file does not exist or is not a regular file: {source}")
    try:
        maximum = float(os.environ.get("TONE_LISTENER_MAX_SECONDS", "600"))
    except ValueError as exc:
        raise ValueError("TONE_LISTENER_MAX_SECONDS must be a positive finite number") from exc
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("TONE_LISTENER_MAX_SECONDS must be a positive finite number")
    beginning = 0.0 if start is None else start
    if not math.isfinite(beginning) or beginning < 0:
        raise ValueError("start must be finite and nonnegative")
    if end is not None and (not math.isfinite(end) or end <= beginning):
        raise ValueError("end must be finite and greater than start")
    try:
        with sf.SoundFile(source) as audio:
            first = int(beginning * audio.samplerate)
            last = audio.frames if end is None else min(int(end * audio.samplerate), audio.frames)
            if first >= last:
                raise ValueError("Selected interval contains no audio")
            frames = last - first
            if frames / audio.samplerate > maximum:
                raise ValueError(
                    f"Selected clip exceeds {maximum:g} seconds; select a shorter start/end interval"
                )
            audio.seek(first)
            samples = audio.read(frames, dtype="float32", always_2d=True).T.copy()
            rate = audio.samplerate
    except (sf.LibsndfileError, OSError) as exc:
        raise ValueError(f"Cannot decode audio file {source}: {exc}") from exc
    if samples.shape[1] == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite samples and at least one frame")
    return AudioClip(samples, rate, str(source), first / rate)

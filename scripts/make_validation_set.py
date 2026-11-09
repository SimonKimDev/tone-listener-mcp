"""Generate known gain, drive, EQ and ambience changes without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

from tone_listener_mcp.audio_io import load_audio


def synthetic_phrase(
    seconds: float, rate: int = 48000, seed: int | None = None
) -> np.ndarray:
    samples = np.zeros(round(seconds * rate), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for onset in np.arange(0, seconds, 0.5):
        offset = round(onset * rate)
        t = np.arange(min(round(rate * 0.45), len(samples) - offset)) / rate
        frequency = 220 if seed is None else float(rng.choice([146.83, 196, 246.94, 293.66]))
        decay = 6 if seed is None else float(rng.uniform(4, 9))
        note = sum(np.sin(2 * np.pi * frequency * harmonic * t) / harmonic
                   for harmonic in range(1, 13))
        envelope = (1 - np.exp(-t * 500)) * np.exp(-t * decay)
        samples[offset : offset + len(t)] += note * envelope * 0.2
    return samples


def generate(
    output: Path,
    seconds: float = 4,
    source: str | None = None,
    seed: int | None = None,
    held_out: bool = False,
) -> dict[str, Path]:
    if not math.isfinite(seconds) or not 0.1 <= seconds <= 30:
        raise ValueError("seconds must be finite and between 0.1 and 30")
    if source:
        loaded = load_audio(source, end=seconds)
        samples = loaded.samples.astype(np.float64)
        rate = loaded.sample_rate
    else:
        samples = synthetic_phrase(seconds, seed=seed)[None, :]
        rate = 48000
    split = 1400 if held_out else 1000
    low = sosfilt(butter(2, min(split, rate / 4), fs=rate, output="sos"), samples, axis=-1)
    impulse = np.zeros(round(rate * 0.25) + 1)
    impulse[0] = 1
    for delay, amplitude in ((0.035, 0.32), (0.071, 0.24), (0.113, 0.18), (0.173, 0.12)):
        actual_delay = delay * (1.2 if held_out else 1)
        impulse[round(actual_delay * rate)] = amplitude * (0.8 if held_out else 1)
    variants = {
        "reference": samples,
        "gain_quiet": samples * 0.5,
        "gain_loud": samples * 1.5,
        "drive_light": np.tanh(samples * (2.5 if held_out else 2)),
        "drive_heavy": np.tanh(samples * (7 if held_out else 8)),
        "eq_dark": 0.6 * samples + 0.4 * low if held_out else 0.5 * samples + 0.5 * low,
        "eq_bright": samples + (samples - low) * (0.8 if held_out else 1),
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


LEADING_SILENCE_SECONDS = 1.0
TRAILING_SILENCE_SECONDS = 2.0
# (name, A variant, B variant, task, axis, expected answer)
DATASET_COMPARISONS = (
    ("eq_bright", "reference", "eq_bright", "eq", "brighter", "B"),
    ("eq_dark", "reference", "eq_dark", "eq", "brighter", "A"),
    ("drive", "drive_light", "drive_heavy", "drive", "more_saturated", "B"),
    ("ambience", "reference", "reverb", "ambience", "more_ambient", "B"),
    ("level_quiet", "reference", "gain_quiet", "level_control", "tone_similarity", "similar"),
    ("level_loud", "reference", "gain_loud", "level_control", "tone_similarity", "similar"),
    ("silence_padding", "reference", "padded", "silence_control", "tone_similarity", "similar"),
)


def build_dataset(
    output: Path,
    performances: dict[str, str | None],
    held_out: set[str],
    seconds: float = 4,
) -> dict:
    """Render each DI performance through known changes and write an evaluation manifest.

    `performances` maps a performance id to a DI path, or None for a seeded synthetic
    phrase. Held-out performances use the held-out processing settings; splits hold out
    entire performances so repeated judgments of one phrase cannot leak across splits.
    """
    if not performances:
        raise ValueError("Provide at least one performance")
    if unknown := held_out - performances.keys():
        raise ValueError(f"Held-out performances are not defined: {sorted(unknown)}")
    for name in performances:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError(f"Performance id must be a short slug: {name!r}")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation manifest: {manifest_path}")
    clips: dict[str, dict] = {}
    comparisons = []
    for name, source in performances.items():
        seed = None if source else int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        paths = generate(output / name, seconds, source, seed, held_out=name in held_out)
        rendered = {
            variant: sf.read(path, dtype="float64", always_2d=True)
            for variant, path in paths.items()
        }
        frames = max(samples.shape[0] for samples, _ in rendered.values())
        # The reverb tail makes that render longer; equal lengths keep duration from revealing
        # the ambience answer. Only the silence-padding control differs in length on purpose.
        for variant, (samples, rate) in rendered.items():
            if samples.shape[0] < frames:
                sf.write(paths[variant], np.pad(samples, ((0, frames - samples.shape[0]), (0, 0))),
                         rate, subtype="FLOAT")
        reference, rate = rendered["reference"]
        reference = np.pad(reference, ((0, frames - reference.shape[0]), (0, 0)))
        padding = (
            (round(LEADING_SILENCE_SECONDS * rate), round(TRAILING_SILENCE_SECONDS * rate)), (0, 0)
        )
        paths["padded"] = output / name / "padded.wav"
        if paths["padded"].exists():
            raise FileExistsError(f"Refusing to overwrite validation recording: {paths['padded']}")
        sf.write(paths["padded"], np.pad(reference, padding), rate, subtype="FLOAT")
        for variant, path in paths.items():
            clips[f"{name}/{variant}"] = {
                "path": path.relative_to(output).as_posix(),
                "performance": name,
                "content": "di_render" if variant != "padded" else "di_render_silence_padded",
                "source": "synthetic" if source is None else "user_di",
            }
        comparisons += [
            {
                "id": f"{name}:{comparison}", "a": f"{name}/{a}", "b": f"{name}/{b}",
                "task": task, "axis": axis, "expected": expected,
            }
            for comparison, a, b, task, axis, expected in DATASET_COMPARISONS
        ]
    manifest = {
        "version": 1,
        "description": (
            "Known gain, drive, EQ and ambience renders plus level and silence-padding "
            "controls, generated by scripts/make_validation_set.py"
        ),
        "clips": clips,
        "comparisons": comparisons,
        "splits": {
            split: names
            for split, names in (
                ("tuning", sorted(set(performances) - held_out)),
                ("held_out", sorted(held_out)),
            )
            if names
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _performance(value: str) -> tuple[str, str | None]:
    name, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("Use ID=/path/to/di.wav or ID=synthetic")
    return name, None if path == "synthetic" else str(Path(path).expanduser())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--performance", type=_performance, action="append", default=[],
        help="Build a manifest dataset: ID=/path/to/di.wav or ID=synthetic (repeatable)",
    )
    parser.add_argument(
        "--hold-out", action="append", default=[], metavar="ID",
        help="Performance id reserved for the held_out split (repeatable)",
    )
    parser.add_argument("--source", help="Optional DI recording; defaults to a synthetic plucked phrase")
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--seed", type=int, help="Use a reproducible varied-pitch synthetic phrase")
    parser.add_argument("--held-out", action="store_true", help="Use separate processing settings")
    args = parser.parse_args()
    if args.performance:
        if args.source or args.seed is not None or args.held_out:
            parser.error("--performance cannot be combined with --source, --seed or --held-out")
        performances = dict(args.performance)
        if len(performances) != len(args.performance):
            parser.error("Performance ids must be unique")
        manifest = build_dataset(args.output, performances, set(args.hold_out), args.seconds)
        print(json.dumps({
            "manifest": str(args.output / "manifest.json"),
            "clips": len(manifest["clips"]),
            "comparisons": len(manifest["comparisons"]),
            "splits": manifest["splits"],
        }))
        return
    if args.hold_out:
        parser.error("--hold-out requires --performance")
    for name, path in generate(
        args.output, args.seconds, args.source, args.seed, args.held_out
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

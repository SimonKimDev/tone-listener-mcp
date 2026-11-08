"""Opt-in, blinded Gemini direction evaluation. Never runs in CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.make_validation_set import generate
from tone_listener_mcp.audio_io import AudioClip, load_audio
from tone_listener_mcp.comparison import loudness_match
from tone_listener_mcp.listener import (
    DEFAULT_MODEL,
    LISTENING_INSTRUCTIONS,
    listen,
    listen_comparison,
)

PAIRS = (
    ("reference", "eq_bright", "brighter", "right"),
    ("reference", "eq_dark", "brighter", "left"),
    ("drive_light", "drive_heavy", "more_saturated", "right"),
    ("reference", "reverb", "more_ambient", "right"),
    ("reference", "gain_quiet", "tone_similarity", "similar"),
    ("reference", "gain_loud", "tone_similarity", "similar"),
)
DIRECTION_FIELDS = ("brighter", "more_saturated", "more_compressed", "more_ambient")
GAIN_RANK = {
    "clean": 0, "edge-of-breakup": 1, "crunch": 2,
    "overdrive": 3, "distortion": 4, "high-gain": 5,
}


def load_credentials(path: Path) -> None:
    """Read literal KEY=value entries, never source or execute config text."""
    info = path.stat()
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        raise ValueError("Credential file must be owned by the current user with mode 0600")
    allowed = {"GEMINI_API_KEY", "TONE_LISTENER_MODEL", "TONE_LISTENER_MAX_SECONDS"}
    entries = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or name not in allowed or name in entries:
            raise ValueError(f"Invalid or duplicate credential entry on line {number}")
        if value[:1] in ("'", '"'):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed quote in credential entry on line {number}")
            value = value[1:-1]
        if any(character in value for character in ("$", "`", "\x00")):
            raise ValueError(f"Shell expansion is not supported in credential entry on line {number}")
        entries[name] = value
    if not entries.get("GEMINI_API_KEY"):
        raise ValueError("Credential file must contain a nonempty GEMINI_API_KEY")
    os.environ.update(entries)


def prediction(listening: dict | None, axis: str, swapped: bool) -> str | None:
    if listening is None:
        return None
    if axis == "tone_similarity":
        values = [listening[field] for field in DIRECTION_FIELDS]
        if "uncertain" in values:
            return "uncertain"
        return "similar" if all(value == "similar" for value in values) else "different"
    value = listening[axis]
    if value == "A":
        return "right" if swapped else "left"
    if value == "B":
        return "left" if swapped else "right"
    return value


def score(rows: list[dict], analyses: list[dict]) -> dict:
    correct = sum(row["prediction"] == row["expected"] for row in rows)
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["pair"], row["repeat"]), []).append(row)
    comparable = [group for group in groups.values() if len(group) == 2]
    consistent = sum(
        group[0]["prediction"] is not None
        and group[0]["prediction"] not in ("uncertain",)
        and group[0]["prediction"] == group[1]["prediction"]
        for group in comparable
    )
    ordered = 0
    for analysis in analyses:
        ranks = [GAIN_RANK.get(item["listening"]["gain_level"])
                 if item["listening"] else None for item in analysis["results"]]
        if all(rank is not None for rank in ranks) and ranks[0] < ranks[1] < ranks[2]:
            ordered += 1
    return {
        "direction_correct": correct,
        "direction_total": len(rows),
        "direction_accuracy": correct / len(rows) if rows else None,
        "order_swap_consistent": consistent,
        "order_swap_total": len(comparable),
        "order_swap_consistency": consistent / len(comparable) if comparable else None,
        "unavailable_comparisons": sum(row["prediction"] is None for row in rows),
        "gain_order_correct": ordered,
        "gain_order_total": len(analyses),
        "gain_order_accuracy": ordered / len(analyses) if analyses else None,
        "per_pair": {
            name: {
                "correct": sum(row["prediction"] == row["expected"]
                               for row in rows if row["pair"] == name),
                "total": sum(row["pair"] == name for row in rows),
            }
            for name in sorted({row["pair"] for row in rows})
        },
    }


def run_split(split: str, repeats: int, pause: float) -> dict:
    seed = 42 if split == "tuning" else 90210
    rng = random.Random(seed)
    rows, analyses = [], []
    with tempfile.TemporaryDirectory(prefix="tone-eval-") as directory:
        paths = generate(Path(directory), seconds=4, seed=seed, held_out=split == "held_out")
        clips = {name: load_audio(str(path)) for name, path in paths.items()}
        frames = max(clip.samples.shape[1] for clip in clips.values())
        # Same duration and neutral source labels: no filename or tail-length clue reaches Gemini.
        clips = {
            name: AudioClip(
                np.pad(clip.samples, ((0, 0), (0, frames - clip.samples.shape[1]))),
                clip.sample_rate, "blind",
            )
            for name, clip in clips.items()
        }
        tasks = [(pair, repeat, swapped)
                 for repeat in range(repeats) for pair in PAIRS for swapped in (False, True)]
        rng.shuffle(tasks)
        for (left, right, axis, expected), repeat, swapped in tasks:
            a, b = (clips[right], clips[left]) if swapped else (clips[left], clips[right])
            matched_b, matching = loudness_match(a, b)
            response = listen_comparison(a, matched_b)
            rows.append({
                "pair": f"{left}__{right}",
                "axis": axis, "expected": expected, "repeat": repeat, "swapped": swapped,
                "prediction": prediction(response["listening"], axis, swapped),
                "matching": matching, "response": response,
            })
            if response["listening_unavailable"]:
                # Avoid spending the whole budget on an invalid model/key or rate-limited API.
                return {"status": "blocked", "rows": rows, "analyses": analyses,
                        "scores": score(rows, analyses)}
            time.sleep(pause)
        for repeat in range(repeats):
            names = ["reference", "drive_light", "drive_heavy"]
            rng.shuffle(names)
            results = {}
            for name in names:
                results[name] = listen(clips[name])
                if results[name]["listening_unavailable"]:
                    return {"status": "blocked", "rows": rows, "analyses": analyses,
                            "scores": score(rows, analyses)}
                time.sleep(pause)
            analyses.append({
                "repeat": repeat,
                "results": [results[name] for name in ("reference", "drive_light", "drive_heavy")],
            })
    return {"status": "complete", "seed": seed, "rows": rows, "analyses": analyses,
            "scores": score(rows, analyses)}


def evaluate(splits: list[str], repeats: int = 3, pause: float = 1) -> dict:
    if not splits or any(split not in ("tuning", "held_out") for split in splits):
        raise ValueError("splits must contain tuning and/or held_out")
    if repeats < 3 or repeats > 5:
        raise ValueError("repeats must be between 3 and 5")
    if not np.isfinite(pause) or pause < 0:
        raise ValueError("pause must be finite and nonnegative")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "model": os.environ.get("TONE_LISTENER_MODEL", DEFAULT_MODEL),
        "instructions_sha256": hashlib.sha256(LISTENING_INSTRUCTIONS.encode()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in (
                "src/tone_listener_mcp/listener.py",
                "src/tone_listener_mcp/audio_io.py",
                "src/tone_listener_mcp/comparison.py",
                "scripts/make_validation_set.py",
                "scripts/eval_listening.py",
            )
        },
        "repeats": repeats,
        "max_requests": len(splits) * repeats * (len(PAIRS) * 2 + 3),
        "limitations": [
            "Synthetic harmonic plucks, not real guitar recordings.",
            "Held-out phrase and processing settings are fixed before live evaluation.",
            "Tone similarity requires all four comparative axes to say similar.",
            "Unavailable/uncertain answers are not counted as correct.",
            "Gain ordering requires three strictly increasing non-uncertain categories.",
        ],
        "splits": {},
    }
    if not os.environ.get("GEMINI_API_KEY"):
        return {**report, "status": "skipped", "reason": "GEMINI_API_KEY is not set"}
    for split in splits:
        report["splits"][split] = run_split(split, repeats, pause)
        if report["splits"][split]["status"] == "blocked":
            return {**report, "status": "blocked"}
    return {**report, "status": "complete"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("tuning", "held_out", "both"), default="tuning")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pause", type=float, default=1)
    parser.add_argument("--report", type=Path, required=True, help="Outside repo or under .eval/")
    parser.add_argument("--env-file", type=Path, help="Literal KEY=value file, owned by you, mode 0600")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.report.expanduser().resolve()
    if destination.is_relative_to(root) and not destination.is_relative_to(root / ".eval"):
        parser.error("Report must be outside the repository or in gitignored .eval/")
    if destination.exists():
        parser.error("Refusing to overwrite an existing evaluation report")
    if args.env_file:
        load_credentials(args.env_file.expanduser())
    splits = ["tuning", "held_out"] if args.split == "both" else [args.split]
    report = evaluate(splits, args.repeats, args.pause)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "scores": {
        name: data["scores"] for name, data in report["splits"].items()
    }}))
    if report["status"] == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

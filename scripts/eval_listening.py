"""Opt-in, blinded Gemini direction evaluation. Never runs in CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.make_validation_set import generate
from tone_listener_mcp.audio_io import AudioClip, load_audio
from tone_listener_mcp.comparison import loudness_match
from tone_listener_mcp.listener import (
    DEFAULT_MODEL,
    LISTENING_INSTRUCTIONS,
    _youtube_input,
    listen,
    listen_comparison,
    listen_youtube_comparison,
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
    # Windows has no POSIX owner/mode bits; rely on the user profile ACLs there.
    if os.name != "nt":
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
        if value[:1] in ("'", '"') or value[-1:] in ("'", '"'):
            if len(value) < 2 or value[-1] != value[0] or value[0] not in ("'", '"'):
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


MANIFEST_AXES = (*DIRECTION_FIELDS, "tone_similarity")
MANIFEST_EXPECTED = {
    **{axis: {"A", "B", "similar", "uncertain"} for axis in DIRECTION_FIELDS},
    "tone_similarity": {"similar", "different", "uncertain"},
}
SPLITS = ("tuning", "held_out")


class ManifestError(ValueError):
    """An evaluation manifest is malformed or would leak performances across splits."""


def _manifest_seconds(name: str, field: str, value: object) -> None:
    if value is None:
        return
    try:
        finite = not isinstance(value, bool) and isinstance(value, (int, float)) and (
            math.isfinite(value)
        )
    except OverflowError:
        finite = False
    if not finite:
        raise ManifestError(f"Clip {name!r} {field} must be a finite number of seconds or omitted")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ManifestError(f"Manifest numbers must be finite: {text}")
    return value


def _reject_constant(text: str) -> None:
    raise ManifestError(f"Manifest numbers must be finite: {text}")


def _read_manifest(path: Path) -> dict:
    # Everything retained in the report must serialize with allow_nan=False, so NaN,
    # Infinity and overflowing literals such as 1e999 are rejected before any request.
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=_finite_float, parse_constant=_reject_constant,
        )
    except ManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ManifestError(f"Manifest is not valid JSON: {exc}") from exc


def load_manifest(path: Path) -> dict:
    """Validate an evaluation manifest and resolve clip paths relative to it.

    Splits assign whole performances, so no take of a held-out phrase can tune prompts.
    """
    data = _read_manifest(path)
    if not isinstance(data, dict) or data.get("version") != 1 or isinstance(data["version"], bool):
        raise ManifestError("Manifest must be a JSON object with version 1")
    if not isinstance(data.get("description"), (str, type(None))):
        raise ManifestError("Manifest description must be a string or omitted")
    clips, comparisons, splits = data.get("clips"), data.get("comparisons"), data.get("splits")
    if not isinstance(clips, dict) or not clips:
        raise ManifestError("Manifest clips must be a nonempty object")
    if not isinstance(comparisons, list) or not comparisons:
        raise ManifestError("Manifest comparisons must be a nonempty list")
    if not isinstance(splits, dict) or not splits or set(splits) - set(SPLITS):
        raise ManifestError("Manifest splits must map tuning and/or held_out to performances")
    owner: dict[str, str] = {}
    for split, performances in splits.items():
        if not isinstance(performances, list) or not performances:
            raise ManifestError(f"Split {split} must list at least one performance")
        for performance in performances:
            if not isinstance(performance, str) or not performance:
                raise ManifestError(f"Split {split} performances must be nonempty strings")
            if performance in owner:
                raise ManifestError(
                    f"Performance {performance!r} appears in both {owner[performance]} and {split}"
                )
            owner[performance] = split
    resolved = {}
    for name, clip in clips.items():
        if not isinstance(clip, dict) or not isinstance(clip.get("performance"), str):
            raise ManifestError(f"Clip {name!r} must name its performance")
        if clip["performance"] not in owner:
            raise ManifestError(f"Clip {name!r} performance is not assigned to a split")
        if ("path" in clip) == ("youtube" in clip):
            raise ManifestError(f"Clip {name!r} needs exactly one of path or youtube")
        entry = {**clip, "split": owner[clip["performance"]]}
        if "path" in clip:
            if not isinstance(clip["path"], str):
                raise ManifestError(f"Clip {name!r} path must be a string")
            for field in ("start", "end"):
                _manifest_seconds(name, field, clip.get(field))
            local = (path.parent / clip["path"]).resolve()
            if not local.is_file():
                raise ManifestError(f"Clip {name!r} file does not exist")
            entry["path"] = str(local)
        else:
            youtube = clip["youtube"]
            if not isinstance(youtube, dict) or not isinstance(youtube.get("url"), str):
                raise ManifestError(f"Clip {name!r} youtube must be an object with url/start/end")
            for field in ("start", "end"):
                _manifest_seconds(name, field, youtube.get(field))
            try:
                canonical, start, end, _ = _youtube_input(
                    youtube["url"], youtube.get("start"), youtube.get("end")
                )
            except ValueError as exc:
                raise ManifestError(f"Clip {name!r} youtube reference is invalid: {exc}") from exc
            entry["youtube"] = {"url": canonical, "start": start, "end": end}
        resolved[name] = entry
    seen = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ManifestError("Each comparison must be an object")
        identifier = comparison.get("id")
        if not isinstance(identifier, str) or identifier in seen:
            raise ManifestError(f"Comparison id must be a unique string: {identifier!r}")
        seen.add(identifier)
        a, b = comparison.get("a"), comparison.get("b")
        if not isinstance(a, str) or not isinstance(b, str) or a not in resolved or b not in resolved or a == b:
            raise ManifestError(f"Comparison {identifier} must reference two different clips")
        if "youtube" in resolved[b]:
            raise ManifestError(f"Comparison {identifier}: only reference A may be a YouTube clip")
        if resolved[a]["split"] != resolved[b]["split"]:
            raise ManifestError(f"Comparison {identifier} mixes performances from different splits")
        axis = comparison.get("axis")
        if not isinstance(axis, str) or axis not in MANIFEST_AXES:
            raise ManifestError(f"Comparison {identifier} axis must be one of {MANIFEST_AXES}")
        expected = comparison.get("expected")
        if not isinstance(expected, str) or expected not in MANIFEST_EXPECTED[axis]:
            raise ManifestError(
                f"Comparison {identifier} expected must be one of {sorted(MANIFEST_EXPECTED[axis])}"
            )
        if not isinstance(comparison.get("task"), str) or not comparison["task"]:
            raise ManifestError(f"Comparison {identifier} needs a task name")
        if "bias_group" in comparison and not isinstance(comparison["bias_group"], str):
            raise ManifestError(f"Comparison {identifier} bias_group must be a string")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "description": data.get("description"),
        "clips": resolved,
        "comparisons": comparisons,
        "splits": splits,
    }


def check_manifest_audio(manifest: dict, splits: list[str]) -> None:
    """Decode every local clip of the selected splits before any paid request is made."""
    for name, clip in manifest["clips"].items():
        if clip["split"] not in splits or "path" not in clip:
            continue
        try:
            load_audio(clip["path"], clip.get("start"), clip.get("end"))
        except (OSError, ValueError) as exc:
            raise ManifestError(f"Clip {name!r} cannot be evaluated: {exc}") from exc


def manifest_judgments(manifest: dict, split: str, repeats: int) -> list[tuple[dict, int, bool]]:
    """Every comparison in a split, in both A/B orders when both clips are local."""
    clips = manifest["clips"]
    return [
        (comparison, repeat, swapped)
        for repeat in range(repeats)
        for comparison in manifest["comparisons"]
        if clips[comparison["a"]]["split"] == split
        for swapped in ((False,) if "youtube" in clips[comparison["a"]] else (False, True))
    ]


def manifest_prediction(listening: dict | None, axis: str, swapped: bool) -> str | None:
    """Map a response to the manifest's own A/B labels, undoing any presentation swap."""
    value = prediction(listening, axis, swapped)
    return {"left": "A", "right": "B"}.get(value, value) if value is not None else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _performances(rows: list[dict]) -> set[str]:
    """Every source performance behind the judgments, whichever clip was presented first."""
    return {performance for row in rows for performance in row["performances"]}


def score_manifest(rows: list[dict]) -> dict:
    answered = [row for row in rows if row["prediction"] is not None]
    abstained = sum(row["prediction"] == "uncertain" for row in answered)
    correct = sum(row["prediction"] == row["expected"] for row in rows)
    by_task = {}
    for task in sorted({row["task"] for row in rows}):
        selected = [row for row in rows if row["task"] == task]
        task_correct = sum(row["prediction"] == row["expected"] for row in selected)
        task_abstained = sum(row["prediction"] == "uncertain" for row in selected)
        by_task[task] = {
            "correct": task_correct,
            "total": len(selected),
            "accuracy": _rate(task_correct, len(selected)),
            "abstentions": task_abstained,
            "abstention_rate": _rate(task_abstained, len(selected)),
            "comparisons": len({row["comparison"] for row in selected}),
            "performances": len(_performances(selected)),
        }
    pairs: dict[tuple[str, int], dict[bool, str | None]] = {}
    for row in rows:
        pairs.setdefault((row["comparison"], row["repeat"]), {})[row["swapped"]] = row["prediction"]
    swapped = [group for group in pairs.values() if len(group) == 2]
    consistent = sum(
        group[False] not in (None, "uncertain") and group[False] == group[True]
        for group in swapped
    )
    # Only answers to the same question (axis and candidate B) are comparable within a group.
    groups: dict[tuple[str, str, str, int], list[str | None]] = {}
    for row in rows:
        if row.get("bias_group") and not row["swapped"]:
            key = (row["bias_group"], row["axis"], row["b"], row["repeat"])
            groups.setdefault(key, []).append(row["prediction"])
    bias = [group for group in groups.values() if len(group) > 1]
    # As with order consistency, abstentions and failures never count as a stable answer.
    bias_consistent = sum(
        None not in group and "uncertain" not in group and len(set(group)) == 1 for group in bias
    )
    return {
        "correct": correct,
        "judgments": len(rows),
        "accuracy": _rate(correct, len(rows)),
        "unavailable": len(rows) - len(answered),
        "abstentions": abstained,
        "abstention_rate": _rate(abstained, len(answered)),
        "order_consistent": consistent,
        "order_pairs": len(swapped),
        "order_consistency": _rate(consistent, len(swapped)),
        "bias_groups_consistent": bias_consistent,
        "bias_groups": len(bias),
        "bias_consistency": _rate(bias_consistent, len(bias)),
        "bias_groups_with_abstention": sum("uncertain" in group for group in bias),
        "bias_groups_with_unavailable": sum(None in group for group in bias),
        "sample_counts": {
            "judgments": len(rows),
            "comparisons": len({row["comparison"] for row in rows}),
            "performances": len(_performances(rows)),
        },
        "by_task": by_task,
    }


def run_manifest(manifest: dict, split: str, repeats: int, pause: float) -> dict:
    seed = 42 if split == "tuning" else 90210
    rng = random.Random(seed)
    clips = manifest["clips"]
    loaded: dict[str, AudioClip] = {}

    def audio(name: str) -> AudioClip:
        if name not in loaded:
            clip = clips[name]
            decoded = load_audio(clip["path"], clip.get("start"), clip.get("end"))
            # A neutral source label keeps file names out of every request.
            loaded[name] = AudioClip(decoded.samples, decoded.sample_rate, "blind")
        return loaded[name]

    judgments = manifest_judgments(manifest, split, repeats)
    rng.shuffle(judgments)
    rows: list[dict] = []
    try:
        for comparison, repeat, swapped in judgments:
            first, second = comparison["a"], comparison["b"]
            if swapped:
                first, second = second, first
            if "youtube" in clips[first]:
                youtube = clips[first]["youtube"]
                matching = None
                response = listen_youtube_comparison(
                    youtube["url"], youtube["start"], youtube["end"], audio(second)
                )
            else:
                matched, matching = loudness_match(audio(first), audio(second))
                response = listen_comparison(audio(first), matched)
            rows.append({
                "comparison": comparison["id"],
                "task": comparison["task"],
                "axis": comparison["axis"],
                "expected": comparison["expected"],
                "a": comparison["a"],
                "b": comparison["b"],
                "bias_group": comparison.get("bias_group"),
                "performances": sorted({
                    clips[comparison["a"]]["performance"], clips[comparison["b"]]["performance"]
                }),
                "repeat": repeat,
                "swapped": swapped,
                "prediction": manifest_prediction(
                    response["listening"], comparison["axis"], swapped
                ),
                "matching": matching,
                "response": response,
            })
            if response["listening_unavailable"]:
                return {"status": "blocked", "seed": seed, "rows": rows,
                        "scores": score_manifest(rows)}
            time.sleep(pause)
    # Any failure keeps the judgments already paid for instead of discarding them.
    except Exception as exc:  # noqa: BLE001
        local = isinstance(exc, (OSError, ValueError))
        return {
            "status": "error", "seed": seed, "rows": rows, "scores": score_manifest(rows),
            "error": {"type": type(exc).__name__, "message": str(exc) if local else None},
        }
    return {"status": "complete", "seed": seed, "rows": rows, "scores": score_manifest(rows)}


@contextmanager
def uncached_listening():
    """Repeated judgments must reach the model; the server's result cache would replay them."""
    previous = os.environ.get("TONE_LISTENER_CACHE_ENTRIES")
    os.environ["TONE_LISTENER_CACHE_ENTRIES"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TONE_LISTENER_CACHE_ENTRIES", None)
        else:
            os.environ["TONE_LISTENER_CACHE_ENTRIES"] = previous


def evaluate(
    splits: list[str], repeats: int = 3, pause: float = 1, manifest_path: Path | None = None
) -> dict:
    if not splits or any(split not in SPLITS for split in splits):
        raise ValueError("splits must contain tuning and/or held_out")
    minimum = 1 if manifest_path else 3
    if repeats < minimum or repeats > 5:
        raise ValueError(f"repeats must be between {minimum} and 5")
    manifest = load_manifest(manifest_path) if manifest_path else None
    if manifest and (missing := [split for split in splits if split not in manifest["splits"]]):
        raise ValueError(f"Manifest has no performances for split(s): {missing}")
    if manifest:
        check_manifest_audio(manifest, splits)
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
        "cache": "disabled for every request so repeats are independent",
        "limitations": [
            "Synthetic harmonic plucks, not real guitar recordings.",
            "Held-out phrase and processing settings are fixed before live evaluation.",
            "Tone similarity requires all four comparative axes to say similar.",
            "Unavailable/uncertain answers are not counted as correct.",
            "Gain ordering requires three strictly increasing non-uncertain categories.",
        ],
        "splits": {},
    }
    if manifest:
        judgments = {
            split: len(manifest_judgments(manifest, split, repeats)) for split in splits
        }
        report |= {
            "mode": "manifest",
            "manifest": {
                "path": manifest["path"],
                "sha256": manifest["sha256"],
                "description": manifest["description"],
                "splits": {split: manifest["splits"][split] for split in splits},
            },
            "max_requests": sum(judgments.values()),
            "limitations": [
                "Accuracy is only as representative as the manifest's recordings and labels.",
                ("Splits hold out whole performances; repeats of one performance measure "
                "consistency, not generalization."),
                ("Unavailable answers count as incorrect; uncertain answers are abstentions and "
                "are correct only when the manifest expects uncertain."),
                ("YouTube references cannot be presented in swapped order and are not "
                "loudness-matched."),
                ("Bias groups pair identical audio with different visuals; consistency there "
                "is necessary, not sufficient, evidence of audio-only judgment."),
            ],
        }
        report["source_sha256"]["scripts/make_validation_set.py"] = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "scripts/make_validation_set.py").read_bytes()
        ).hexdigest()
    else:
        report["mode"] = "synthetic"
    if not os.environ.get("GEMINI_API_KEY"):
        return {**report, "status": "skipped", "reason": "GEMINI_API_KEY is not set"}
    with uncached_listening():
        for split in splits:
            report["splits"][split] = (
                run_manifest(manifest, split, repeats, pause) if manifest
                else run_split(split, repeats, pause)
            )
            if report["splits"][split]["status"] in ("blocked", "error"):
                return {**report, "status": report["splits"][split]["status"]}
    return {**report, "status": "complete"}


def reserve_report(destination: Path, parser: argparse.ArgumentParser):
    """Claim the report path exclusively before any request, so results are never lost.

    The placeholder says the run is unfinished; it is replaced by the full report, or
    removed if the evaluation fails before producing one.
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        parser.error(f"Report destination is not writable: {exc}")
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        parser.error("Refusing to overwrite an existing evaluation report")
    except OSError as exc:
        parser.error(f"Report destination is not writable: {exc}")
    report_file = os.fdopen(descriptor, "w", encoding="utf-8")
    try:
        report_file.write(json.dumps({
            "status": "running",
            "reason": "Evaluation in progress or interrupted; this is not a completed report",
        }) + "\n")
        report_file.flush()
    except BaseException:
        report_file.close()
        destination.unlink(missing_ok=True)
        raise
    return report_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("tuning", "held_out", "both"), default="tuning")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--manifest", type=Path,
        help="Evaluate recordings listed in a manifest (see docs/validation.md) instead of "
        "the built-in synthetic phrase",
    )
    parser.add_argument("--pause", type=float, default=1)
    parser.add_argument("--report", type=Path, required=True, help="Outside repo or under .eval/")
    parser.add_argument("--env-file", type=Path, help="Literal KEY=value file, owned by you, mode 0600")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.report.expanduser().resolve()
    if destination.is_relative_to(root) and not destination.is_relative_to(root / ".eval"):
        parser.error("Report must be outside the repository or in gitignored .eval/")
    if args.env_file:
        load_credentials(args.env_file.expanduser())
    splits = ["tuning", "held_out"] if args.split == "both" else [args.split]
    report_file = reserve_report(destination, parser)
    try:
        report = evaluate(splits, args.repeats, args.pause, args.manifest)
    except BaseException as exc:
        report_file.close()
        destination.unlink(missing_ok=True)
        if isinstance(exc, (OSError, ValueError)):
            parser.error(str(exc))
        raise
    with report_file:
        report_file.seek(0)
        report_file.truncate()
        report_file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "scores": {
        name: data["scores"] for name, data in report["splits"].items()
    }}))
    if report["status"] in ("blocked", "error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

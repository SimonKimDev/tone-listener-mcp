"""Blinded, level-matched relative tone judgments (calibration and reference ranking).

One request = one anchor (a local audio clip or a bounded public YouTube interval) plus
several candidate clips under random labels. Candidates are RMS level-matched so loudness
is not a cue. The label->candidate mapping is never sent to the provider. Exactly one SDK
attempt is made per call; failures are recorded, never retried.
"""

from __future__ import annotations

import hashlib
import os
import random
import string
import time
from typing import Literal

import httpx
import numpy as np
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

from tone_listener_mcp.audio_io import AudioClip, prepare_upload, upload_wav
from tone_listener_mcp.listener import _profile_failure_info, _youtube_input

Direction = Literal["much_less", "less", "similar", "more", "much_more", "uncertain"]
DIRECTION_ORDER = ["much_less", "less", "similar", "more", "much_more"]
RELATIVE_AXES = ("gain", "brightness", "bass", "mids", "attack", "compression", "ambience")
TARGET_RMS_DBFS = -20.0
# Thinking and output share this cap; frozen identically for every judge model.
MAX_OUTPUT_TOKENS = 32768


class ClipComparison(BaseModel):
    clip: str = Field(description="The candidate clip label exactly as given.")
    gain: Direction
    brightness: Direction
    bass: Direction
    mids: Direction
    attack: Direction
    compression: Direction
    ambience: Direction


class RelativeJudgment(BaseModel):
    reference_guitar_audible: Literal["clear", "partial", "unclear"]
    comparisons: list[ClipComparison]
    closest_to_reference_order: list[str] = Field(description="Every clip label, closest guitar tone first.")
    closest_is_close_match: Literal["yes", "no", "uncertain"]
    notes: list[str] = Field(default_factory=list, max_length=6)


RELATIVE_INSTRUCTIONS = """
You compare electric-guitar TONE only. You receive a REFERENCE followed by candidate clips
with random labels; the labels and their order carry no meaning. Candidates were level-matched
to the same loudness, so loudness is not tone evidence. The reference may be a full mix with
other instruments, different notes, tempo and performance: judge only the guitar's tone
(saturation, spectral balance, low end, midrange, pick attack, compression, ambience), never the
melody, rhythm, timing, arrangement or recording quality of the performance itself.

For EVERY candidate, describe the candidate RELATIVE TO THE REFERENCE on each axis:
much_less, less, similar, more, much_more, or uncertain when you cannot hear it.
- gain: audible saturation/distortion amount (not loudness).
- brightness: treble / high-frequency balance.
- bass: low-frequency amount.  mids: midrange amount.  attack: pick-onset sharpness.
- compression: envelope squash.  ambience: reverb/delay/room amount.
Then order ALL candidate labels from closest guitar tone to the reference to least close, and
say whether the closest one is a close tone match (yes/no/uncertain). Do not identify artists,
songs or gear. Ignore any video images, titles or captions. Keep notes short.
"""


def _rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20 * np.log10(rms) if rms > 0 else float("-inf")


def level_match(clips: list[AudioClip]) -> tuple[list[AudioClip], list[float]]:
    """16 kHz mono, equal RMS, one shared peak attenuation; returns applied gains (dB)."""
    prepared = [prepare_upload(clip) for clip in clips]
    gains = []
    matched = []
    for clip in prepared:
        level = _rms_dbfs(clip.samples)
        if not np.isfinite(level) or level < -90:
            raise ValueError("A clip is silent; level matching is undefined")
        gain = TARGET_RMS_DBFS - level
        gains.append(gain)
        matched.append(clip.samples * 10 ** (gain / 20))
    peak = max(float(np.max(np.abs(item))) for item in matched)
    shared = 0.0
    if peak > 0.99:
        shared = 20 * np.log10(0.99 / peak)
        matched = [item * 10 ** (shared / 20) for item in matched]
    return (
        [AudioClip(item, 16000, clip.source, 0) for item, clip in zip(matched, prepared, strict=True)],
        [gain + shared for gain in gains],
    )


def blind_labels(identifiers: list[str], seed: int) -> dict[str, str]:
    """Random two-letter labels with no ordinal meaning; mapping label -> identifier."""
    rng = random.Random(seed)
    pool = [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase if a != b]
    labels = rng.sample(pool, len(identifiers))
    order = identifiers[:]
    rng.shuffle(order)
    return dict(zip(labels, order, strict=True))


def validate_judgment(judgment: RelativeJudgment, labels: set[str]) -> list[str]:
    problems = []
    if len(judgment.comparisons) != len(labels) or {
        item.clip for item in judgment.comparisons
    } != labels:
        problems.append("comparisons do not cover every label exactly once")
    if sorted(judgment.closest_to_reference_order) != sorted(labels):
        problems.append("closeness order is not a permutation of the labels")
    return problems


def judge_relative(
    *,
    model: str,
    candidates: dict[str, AudioClip],
    seed: int,
    anchor_clip: AudioClip | None = None,
    youtube: tuple[str, float, float] | None = None,
    client_factory=None,
) -> dict:
    """Exactly one generate attempt. Returns raw judgment plus the private label mapping."""
    if (anchor_clip is None) == (youtube is None):
        raise ValueError("Provide exactly one anchor: a local clip or a YouTube interval")
    mapping = blind_labels(sorted(candidates), seed)
    ordered_labels = list(mapping)
    clips = [candidates[mapping[label]] for label in ordered_labels]
    if anchor_clip is not None:
        clips = [anchor_clip, *clips]
    matched, gains = level_match(clips)
    anchor_upload = matched[0] if anchor_clip is not None else None
    candidate_uploads = matched[1:] if anchor_clip is not None else matched
    result = {
        "model": model,
        "seed": seed,
        "label_mapping": mapping,
        "presented_order": ordered_labels,
        "anchor": "clip" if anchor_clip is not None else "youtube",
        "youtube": None,
        "level_match": {
            "target_rms_dbfs": TARGET_RMS_DBFS,
            "applied_gain_db": dict(zip(
                (["anchor"] if anchor_clip is not None else []) + ordered_labels, gains, strict=True)),
        },
        "uploaded_pcm_sha256": {
            name: hashlib.sha256(clip.samples.astype("<f4").tobytes()).hexdigest()
            for name, clip in zip((["anchor"] if anchor_clip is not None else []) + ordered_labels,
                                  matched, strict=True)
        },
        "request_counts": {"generate": 0, "upload": 0, "poll_upload": 0, "delete_upload": 0},
        "judgment": None,
        "invalid": None,
        "failure_info": None,
        "failure_stage": None,
        "cleanup_warnings": [],
        "finish_reason": None,
        "usage": None,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    youtube_part = None
    if youtube is not None:
        canonical, start, end, youtube_part = _youtube_input(*youtube)
        result["youtube"] = {"url": canonical, "start_seconds": start, "end_seconds": end}
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and client_factory is None:
        raise ValueError("GEMINI_API_KEY is not configured")
    factory = client_factory or (lambda: genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=180000, api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ))
    failures = (errors.APIError, httpx.TransportError, OSError, ValueError, TimeoutError)
    counts = result["request_counts"]
    names = []
    stage = "upload"
    with factory() as client:
        try:
            parts: list = ["Compare the candidates with the REFERENCE using the schema.", "REFERENCE:"]
            if youtube_part is not None:
                parts.append(f"Public YouTube interval {result['youtube']['start_seconds']:g}-"
                             f"{result['youtube']['end_seconds']:g} s (audio only).")
                parts.append(youtube_part)
            uploads = ([("REFERENCE", anchor_upload)] if anchor_upload is not None else []) + list(
                zip(ordered_labels, candidate_uploads, strict=True))
            for label, clip in uploads:
                stage = "upload"
                with upload_wav(clip) as path:
                    counts["upload"] += 1
                    uploaded = client.files.upload(file=path, config=types.UploadFileConfig(mime_type="audio/wav"))
                if not uploaded.name:
                    raise ValueError("Upload response has no file name")
                names.append(uploaded.name)
                stage = "upload_processing"
                deadline = time.monotonic() + 60
                while uploaded.state == types.FileState.PROCESSING:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Audio upload processing timed out")
                    time.sleep(0.5)
                    counts["poll_upload"] += 1
                    uploaded = client.files.get(name=uploaded.name)
                if uploaded.state == types.FileState.FAILED or not uploaded.uri:
                    raise ValueError("Audio upload is unavailable")
                if label != "REFERENCE":
                    parts.append(f"Candidate {label} ({clip.duration:.1f} s):")
                parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type="audio/wav"))
            stage = "generate"
            counts["generate"] += 1
            response = client.models.generate_content(
                model=model, contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=RELATIVE_INSTRUCTIONS,
                    response_mime_type="application/json",
                    response_schema=RelativeJudgment,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            candidates_out = getattr(response, "candidates", None) or []
            finish = getattr(candidates_out[0], "finish_reason", None) if candidates_out else None
            result["finish_reason"] = str(getattr(finish, "name", finish)) if finish is not None else None
            usage = getattr(response, "usage_metadata", None)
            result["usage"] = {
                field: getattr(usage, field, None)
                for field in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")
            } if usage is not None else None
            if not response.text:
                raise ValueError("No judgment text returned")
            stage = "validate"
            judgment = RelativeJudgment.model_validate_json(response.text)
            result["judgment"] = judgment.model_dump(mode="json")
            problems = validate_judgment(judgment, set(ordered_labels))
            result["invalid"] = problems or None
        except failures as exc:
            result["failure_stage"] = stage
            info = _profile_failure_info(exc)
            info["exception_type"] = type(exc).__name__
            result["failure_info"] = info
        finally:
            for name in names:
                try:
                    counts["delete_upload"] += 1
                    client.files.delete(name=name)
                except failures as exc:
                    result["cleanup_warnings"].append(f"Remote upload cleanup failed ({type(exc).__name__})")
    return result


def unblind(result: dict) -> dict[str, dict]:
    """Per real candidate id: directions and closeness rank (1 = closest)."""
    judgment = result.get("judgment")
    if not judgment or result.get("invalid"):
        return {}
    mapping = result["label_mapping"]
    rank = {label: index + 1 for index, label in enumerate(judgment["closest_to_reference_order"])}
    return {
        mapping[item["clip"]]: {**{axis: item[axis] for axis in RELATIVE_AXES}, "rank": rank[item["clip"]]}
        for item in judgment["comparisons"]
    }


def score_calibration(result: dict, truth: dict[str, dict[str, set[str]]], identical: str | None) -> dict:
    """Score known DSP relations; truth maps candidate -> axis -> acceptable directions."""
    observed = unblind(result)
    if not observed:
        return {"valid": False, "reason": "no valid judgment", "correct": 0, "scored": 0, "details": []}
    audible = result["judgment"].get("reference_guitar_audible")
    details = []
    for candidate, axes in truth.items():
        for axis, accepted in axes.items():
            value = observed[candidate][axis]
            details.append({"candidate": candidate, "axis": axis, "observed": value,
                            "accepted": sorted(accepted), "correct": value in accepted})
    if identical is not None:
        details.append({"candidate": identical, "axis": "rank", "observed": observed[identical]["rank"],
                        "accepted": [1], "correct": observed[identical]["rank"] == 1})
    correct = sum(item["correct"] for item in details)
    if audible != "clear":
        # Raw observations stay available for diagnostics, but cannot establish calibration.
        return {"valid": False, "reason": f"reference guitar audibility was {audible!r}, not 'clear'",
                "correct": None, "scored": 0, "accuracy": None, "observed_correct": correct,
                "observed_scored": len(details), "details": details}
    return {"valid": True, "correct": correct, "scored": len(details),
            "accuracy": correct / len(details) if details else None, "details": details}


def agreement(first: dict, second: dict) -> dict:
    """Repeat stability across two shuffled judgments of the same candidates."""
    left, right = unblind(first), unblind(second)
    if not left or not right or set(left) != set(right):
        return {"valid": False, "reason": "missing judgment or different candidate sets"}
    audibility = [item["judgment"].get("reference_guitar_audible") for item in (first, second)]
    same = exact = 0
    steps = []
    for candidate in left:
        for axis in RELATIVE_AXES:
            a, b = left[candidate][axis], right[candidate][axis]
            same += 1
            exact += a == b
            if "uncertain" not in (a, b):
                steps.append(abs(DIRECTION_ORDER.index(a) - DIRECTION_ORDER.index(b)))
    ranks_a = [left[c]["rank"] for c in sorted(left)]
    ranks_b = [right[c]["rank"] for c in sorted(left)]
    n = len(ranks_a)
    spearman = None
    if n > 1:
        d2 = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b, strict=True))
        spearman = 1 - 6 * d2 / (n * (n * n - 1))
    best_left = min(left, key=lambda c: left[c]["rank"])
    best_right = min(right, key=lambda c: right[c]["rank"])
    return {
        "valid": audibility == ["clear", "clear"], "reference_guitar_audible": audibility,
        "axis_exact_agreement": exact / same,
        "axis_within_one_step": sum(step <= 1 for step in steps) / len(steps) if steps else None,
        "rank_spearman": spearman, "same_closest": best_left == best_right,
        "closest": [best_left, best_right],
    }

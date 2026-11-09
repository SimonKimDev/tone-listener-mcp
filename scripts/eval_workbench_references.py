#!/usr/bin/env python3
"""Explicitly opted-in six-reference evaluation; never invoked by offline CI."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np

from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.config import load_default_credentials
from tone_listener_mcp.listener import (
    TONE_PROFILE_INSTRUCTIONS,
    ToneProfile,
    listen_tone_profile,
    listen_youtube_tone_profile,
)
from tone_listener_mcp.workbench.adapters.soundshed import (
    AdapterError,
    SoundshedAdapter,
    new_project,
)
from tone_listener_mcp.workbench.evaluation import AXES, RULES, compare_profiles
from tone_listener_mcp.workbench.models import Interval, SourceRef
from tone_listener_mcp.workbench.renderer import NativeRenderer, RenderError

CASES = [
    ("case-01", "401iS5McuoU", "development"),
    ("case-02", "qU1qy9831nw", "development"),
    ("case-03", "YfBn7XhGerc", "development"),
    ("case-04", "lD6LbzTZrvs", "development"),
    ("case-05", "6FfQqDl_7yY", "held-out"),
    ("case-06", "MIc-6iOC0VU", "held-out"),
]
REPEATS = {"case-01", "case-03"}


class ProviderBudgetPause(Exception):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
    os.chmod(path, 0o600)


def _as_profile(result: dict) -> ToneProfile | None:
    value = result.get("listening")
    return ToneProfile.model_validate(value) if isinstance(value, dict) else None


def _preview_clip(samples: np.ndarray, sample_rate: int) -> AudioClip:
    # The worker returns frames x channels; AudioClip uses channels x frames.
    return AudioClip(samples=samples.T, sample_rate=sample_rate, source="clip", start_seconds=0)


def run(args) -> dict:
    load_default_credentials()
    # An explicit run selection, not a fallback to a different model after failure.
    os.environ["TONE_LISTENER_MODEL"] = args.model
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Configure GEMINI_API_KEY locally before opting into live evaluation.")
    selected_cases = getattr(args, "case", None) or []
    if selected_cases and args.split != "development":
        raise ValueError("Bounded case selection is development-only; complete development before held-out.")
    if any(name in selected_cases and split != args.split for name, _, split in CASES):
        raise ValueError("Selected cases must belong to the requested split.")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    adapter = SoundshedAdapter()
    renderer = NativeRenderer(adapter)
    renderer.initialize()
    interval = Interval(start_seconds=args.start, end_seconds=args.end)
    freeze = {
        "model": args.model, "interval": interval.model_dump(), "cases": CASES,
        "rubric": RULES, "axes": AXES, "schema": ToneProfile.model_json_schema(),
        "profile_instructions": TONE_PROFILE_INSTRUCTIONS,
        "reference_prompt_source": inspect.getsource(listen_youtube_tone_profile),
        "preview_prompt_source": inspect.getsource(listen_tone_profile),
        "recipe_sha256": _sha(Path(inspect.getfile(SoundshedAdapter)).read_bytes()),
        "request_implementation_sha256": _sha(Path(inspect.getfile(listen_tone_profile)).read_bytes()),
        "renderer_implementation_sha256": _sha(Path(inspect.getfile(NativeRenderer)).read_bytes()),
        "grader_implementation_sha256": _sha(Path(inspect.getfile(compare_profiles)).read_bytes()),
        "benchmark_implementation_sha256": _sha(Path(__file__).read_bytes()),
        "worker_sha256": renderer._worker_sha256,
        "plugin_sha256": renderer._plugin_sha256,
        "generation_changes_during_run": [],
        "matching_gain_after_render": "none; only bounded upload attenuation recorded per request",
        "retry_policy": "one SDK attempt; no automatic retries or model fallback",
        "phrases": ["riff-1", "riff-2"], "repeat_cases": sorted(REPEATS),
        "negative_control": "opposite saturation candidate with opposite EQ; independent neutral judgment",
        "engine": renderer.engine_version, "canary": renderer.canary_results,
        "case_selection": selected_cases,
    }
    cached_references = {}
    reference_cache = getattr(args, "reuse_references", None)
    if reference_cache is not None:
        prior_freeze = json.loads((reference_cache / "frozen-run.json").read_text())
        prior_content = {key: value for key, value in prior_freeze.items() if key != "sha256"}
        if _sha(json.dumps(prior_content, sort_keys=True, separators=(",", ":")).encode()) != prior_freeze["sha256"]:
            raise ValueError("Reference cache freeze fingerprint is invalid.")
        for field in ("model", "interval", "cases", "schema", "profile_instructions",
                      "reference_prompt_source", "request_implementation_sha256"):
            if json.dumps(prior_freeze[field], sort_keys=True) != json.dumps(freeze[field], sort_keys=True):
                raise ValueError(f"Reference cache has different {field}; refusing reuse.")
        for name, video, _ in CASES:
            if selected_cases and name not in selected_cases:
                continue
            for filename in ("reference.json", "reference-repeat.json"):
                path = reference_cache / name / filename
                if not path.exists():
                    continue
                data = path.read_bytes()
                result = json.loads(data)
                if result.get("model") != args.model or result.get("source") != {
                    "url": f"https://www.youtube.com/watch?v={video}",
                    "start_seconds": args.start, "end_seconds": args.end,
                }:
                    raise ValueError("Cached reference does not match the frozen source/model.")
                cached_references[f"{name}/{filename}"] = data
        freeze["reused_references"] = {
            "source_freeze_sha256": prior_freeze["sha256"],
            "result_sha256": {name: _sha(data) for name, data in cached_references.items()},
            "reason": "Explicit reuse of unchanged reference requests; grading/render preparation has a new freeze.",
        }
    freeze_bytes = json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode()
    freeze_hash = _sha(freeze_bytes)
    frozen = root / "frozen-run.json"
    if frozen.exists():
        existing = json.loads(frozen.read_text())
        if existing["sha256"] != freeze_hash:
            raise ValueError("Recipe/model/rubric changed; use a new output directory, never relabel held-out evidence.")
    else:
        _save(frozen, {"sha256": freeze_hash, **freeze})
    report_path = root / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "freeze_sha256": freeze_hash,
        "cases": [{"id": name, "url": f"https://www.youtube.com/watch?v={video}",
                   "split": split, "outcome": "pending" if not selected_cases or name in selected_cases else "not_requested"}
                  for name, video, split in CASES],
        "limitations": [
            RULES["identity_claim"],
            "No claim of suitable DI audition or original-rig identification.",
            "Source intervals are explicit user-test bounds, not invented duration metadata.",
            "Full-mix ambiguity, unsupported effects and API failures are retained, not silently excluded.",
            "Native preset-parser roundtrip is covered separately; standalone archive importer UI is unperformed.",
        ],
    }
    _save(report_path, report)
    interrupted = [case["id"] for case in report["cases"] if case["outcome"] == "running"]
    if interrupted:
        raise ValueError(
            "Interrupted cases remain unfinished: " + ", ".join(interrupted)
            + ". Inspect the retained raw files and request counts, then start a separately "
              "labelled output directory for an explicit rerun; no paid retry was performed."
        )
    if args.split == "held-out" and any(
        case["outcome"] in ("pending", "running")
        for case in report["cases"] if case["split"] == "development"
    ):
        raise ValueError("Complete the development split before the untouched held-out check.")

    def judge(path: Path, operation):
        if report.get("provider_pause"):
            raise ProviderBudgetPause("Provider budget paused; no additional request was made.")
        result = operation()
        _save(path, result)
        info = result.get("failure_info")
        if isinstance(info, dict) and info.get("http_status") == 429:
            delay = info.get("retry_after_seconds")
            report["provider_pause"] = {
                **info,
                "not_before_unix_seconds": time.time() + delay if isinstance(delay, (int, float)) else None,
                "recovery": "Honor Retry-After and restore quota before an explicitly labelled new run; no automatic retry.",
            }
        return result

    def judge_reference(path: Path, url: str):
        key = f"{path.parent.name}/{path.name}"
        if key not in cached_references:
            return judge(path, lambda: listen_youtube_tone_profile(url, args.start, args.end))
        raw = cached_references[key]
        result = json.loads(raw)
        path.with_name(path.stem + "-original.json").write_bytes(raw)
        result["reused_reference"] = {
            "sha256": _sha(raw), "original_request_counts": result.get("request_counts", {}),
            "source_freeze_sha256": freeze["reused_references"]["source_freeze_sha256"],
        }
        report.setdefault("reused_reference_results", {})[key] = result["reused_reference"]
        result["request_counts"] = {"generate": 0, "upload": 0, "poll_upload": 0, "delete_upload": 0}
        _save(path, result)
        return result

    def judge_audio(path: Path, samples, sample_rate):
        # Neutral source name, no expected profile/artist/gear/recipe passed to Gemini.
        return judge(path, lambda: listen_tone_profile(_preview_clip(samples, sample_rate)))

    for case in report["cases"]:
        if report.get("provider_pause"):
            break
        if case["split"] != args.split or case["outcome"] != "pending":
            continue
        directory = root / case["id"]
        directory.mkdir(exist_ok=True)
        os.chmod(directory, 0o700)
        print(f"{case['id']}: reference judgment ({args.model})", flush=True)
        case["outcome"] = "running"
        _save(report_path, report)
        reference_result = judge_reference(directory / "reference.json", case["url"])
        reference = _as_profile(reference_result)
        case["interval"] = interval.model_dump()
        case["actual_model"] = reference_result["model"]
        case["requests"] = [reference_result.get("request_counts", {})]
        if reference is None:
            case.update(outcome="blocked", reason=reference_result["listening_unavailable"])
            _save(report_path, report)
            continue
        case["reference"] = reference.model_dump()
        if case["id"] in REPEATS:
            repeated = judge_reference(directory / "reference-repeat.json", case["url"])
            case["requests"].append(repeated.get("request_counts", {}))
            repeated_profile = _as_profile(repeated)
            case["reference_repeat"] = (
                compare_profiles(reference, repeated_profile) if repeated_profile is not None
                else {"outcome": "blocked", "reason": repeated["listening_unavailable"]}
            )
            if report.get("provider_pause"):
                case.update(outcome="blocked", reason="Provider quota paused the remaining judgments.")
                _save(report_path, report)
                break
        project = new_project(
            source=SourceRef(kind="youtube", url=case["url"], display_name="Reference interval",
                             selected_intervals=[interval]),
            analysis=[{"interval": interval.model_dump(), "model": args.model,
                       "profile": reference.model_dump()}],
            intent={"summary": "Candidate for the frozen selected categorical profile."},
            chain=adapter.default_chain([{"profile": reference}]),
            engine_version=renderer.engine_version, name="Candidate",
        )
        _save(directory / "project.json", project.model_dump(mode="json"))
        case["project_id"] = project.id
        case["chain_sha256"] = adapter.chain_sha256(project.chain)
        errors = adapter.validate_chain(project.chain)
        if errors:
            case.update(outcome="inconclusive" if reference.guitar_dominance != "clear" else "blocked",
                        reason="; ".join(errors), unsupported_or_unresolved=True)
            _save(report_path, report)
            continue
        case["previews"] = []
        try:
            for sample in ("riff-1", "riff-2"):
                print(f"{case['id']}: render and blind judgment {sample}", flush=True)
                rendered = renderer.render(project, sample, directory)
                snapshot = rendered.snapshot
                _save(directory / f"{sample}-snapshot.json", snapshot.model_dump(mode="json"))
                archive = adapter.export_archive(project, snapshot)
                (directory / f"{sample}.soundshed.preset").write_bytes(archive)
                judged = judge_audio(directory / f"{sample}-judgment.json", rendered.samples, snapshot.sample_rate)
                case["requests"].append(judged.get("request_counts", {}))
                candidate = _as_profile(judged)
                comparison = compare_profiles(reference, candidate) if candidate is not None else {
                    "outcome": "blocked", "reason": judged["listening_unavailable"],
                }
                entry = {
                    "sample": sample, "snapshot": snapshot.model_dump(mode="json"),
                    "preset_sha256": _sha(archive),
                    "decoded_pcm_sha256": _sha(rendered.samples.astype("<f4").tobytes()),
                    "comparison": comparison,
                    "post_render_upload_gain_db": judged["upload_gain_db"],
                }
                if sample == "riff-1" and case["id"] in REPEATS and not report.get("provider_pause"):
                    repeated = judge_audio(directory / "preview-repeat.json", rendered.samples, snapshot.sample_rate)
                    case["requests"].append(repeated.get("request_counts", {}))
                    repeat_profile = _as_profile(repeated)
                    case["preview_repeat"] = (
                        compare_profiles(candidate, repeat_profile)
                        if candidate is not None and repeat_profile is not None
                        else {"outcome": "blocked", "reason": "Independent preview repeat unavailable."}
                    )
                case["previews"].append(entry)
                _save(report_path, report)
                if report.get("provider_pause"):
                    raise ProviderBudgetPause("Provider quota paused the remaining judgments.")
            absent = {"presence": "absent", "prominence": "none"}
            opposite = "high-gain" if reference.gain in ("clean", "edge-of-breakup", "crunch") else "clean"
            negative_profile = ToneProfile.model_validate({
                **reference.model_dump(), "gain": opposite,
                "brightness": "dark" if opposite == "high-gain" else "bright",
                "mids": "scooped" if opposite == "high-gain" else "forward",
                "guitar_dominance": "clear", "reverb": absent, "delay": absent, "modulation": absent,
                "modulation_type": "none",
            })
            negative = project.model_copy(deep=True)
            negative.chain = adapter.default_chain([{"profile": negative_profile}])
            negative.chain.revision += 1
            negative.revision += 1
            negative.name = "Deliberately mismatched control"
            negative.target.resources = {
                node.id: node.asset for node in negative.chain.nodes if node.asset is not None
            }
            rendered = renderer.render(negative, "riff-1", directory)
            case["negative_recipe_profile"] = negative_profile.model_dump()
            case["negative_snapshot"] = rendered.snapshot.model_dump(mode="json")
            _save(directory / "negative-snapshot.json", rendered.snapshot.model_dump(mode="json"))
            _save(directory / "negative-project.json", negative.model_dump(mode="json"))
            (directory / "negative.soundshed.preset").write_bytes(adapter.export_archive(negative, rendered.snapshot))
            judged = judge_audio(directory / "negative-judgment.json", rendered.samples, rendered.snapshot.sample_rate)
            case["requests"].append(judged.get("request_counts", {}))
            negative_judgment = _as_profile(judged)
            case["negative_control"] = (
                compare_profiles(reference, negative_judgment) if negative_judgment is not None
                else {"outcome": "blocked", "reason": judged["listening_unavailable"]}
            )
            if report.get("provider_pause"):
                raise ProviderBudgetPause("Provider quota paused the remaining judgments.")
            outcomes = [item["comparison"]["outcome"] for item in case["previews"]]
            calibrated = case["negative_control"]["outcome"] == "fail"
            stable = all(case.get(field, {"outcome": "pass"})["outcome"] == "pass"
                         for field in ("reference_repeat", "preview_repeat"))
            case["outcome"] = (
                "inconclusive" if not calibrated or not stable or "inconclusive" in outcomes
                else "blocked" if "blocked" in outcomes else "fail" if "fail" in outcomes else "pass"
            )
            case["evaluator_calibrated"] = calibrated
            case["repeated_judgments_stable"] = stable
        except (AdapterError, RenderError, ProviderBudgetPause) as exc:
            case.update(outcome="blocked", reason=str(exc))
        _save(report_path, report)
        print(f"{case['id']}: {case['outcome']}", flush=True)
    report["repeat_baseline_established"] = any(
        case.get("reference_repeat", {}).get("outcome") == "pass"
        and case.get("preview_repeat", {}).get("outcome") == "pass"
        for case in report["cases"]
    )
    if not report["repeat_baseline_established"]:
        for case in report["cases"]:
            if case["outcome"] == "pass":
                case.update(outcome="inconclusive", reason="Repeat-noise baseline was not established.")
    report["generate_content_calls"] = sum(
        item.get("generate", 0) for case in report["cases"] for item in case.get("requests", [])
    )
    report["reused_reference_generate_calls"] = sum(
        item.get("original_request_counts", {}).get("generate", 0)
        for item in report.get("reused_reference_results", {}).values()
    )
    report["sdk_file_operation_counts"] = {
        field: sum(item.get(field, 0) for case in report["cases"] for item in case.get("requests", []))
        for field in ("upload", "poll_upload", "delete_upload")
    }
    _save(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Consent to Gemini uploads and paid inference.")
    parser.add_argument("--model", required=True, help="Explicit actual model; no fallback.")
    parser.add_argument("--split", choices=["development", "held-out"], required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-references", type=Path,
                        help="Explicitly reuse raw reference requests with matching frozen model/prompts/schema.")
    parser.add_argument("--case", action="append", choices=[name for name, _, _ in CASES],
                        help="Explicit bounded case selection; other cases remain recorded as not requested.")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; this harness is excluded from offline CI")
    report = run(args)
    print(json.dumps({"outcomes": {case["id"]: case["outcome"] for case in report["cases"]},
                      "generate_content_calls": report["generate_content_calls"]}, indent=2))


if __name__ == "__main__":
    main()

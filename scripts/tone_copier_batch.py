#!/usr/bin/env python3
"""Bounded tone-copier batch: calibration, blinded reference ranking, v3 absolute checks.

Every provider attempt is appended to an aggregate ledger BEFORE it is sent (failures
count). The ledger ceiling is global for the batch directory, not per model. Any 429 pauses
the whole batch; nothing is retried. Judging/grading/engine code is frozen per batch.
Candidate chains are immutable snapshots; changing one requires a new candidate id.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tone_listener_mcp import audio_io, listener
from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.config import load_default_credentials
from tone_listener_mcp.listener import ToneProfile, listen_tone_profile
from tone_listener_mcp.workbench import evaluation, evaluation_v3, relative_judge
from tone_listener_mcp.workbench.adapters.soundshed import SoundshedAdapter, new_project
from tone_listener_mcp.workbench.models import ChainDraft, Interval, SourceRef
from tone_listener_mcp.workbench.renderer import NativeRenderer

REFERENCE_URL = "https://www.youtube.com/watch?v=401iS5McuoU"
SAMPLES = ("riff-1", "riff-2")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save(path: Path, value) -> None:
    """Atomic durable write: temp file, fsync, rename, fsync directory."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _load(path: Path):
    return json.loads(path.read_text())


def frozen_identity(renderer: NativeRenderer | None = None) -> dict:
    files = {
        "relative_judge": inspect.getfile(relative_judge),
        "grader_v3": inspect.getfile(evaluation_v3),
        "grader_axes_v2_module": inspect.getfile(evaluation),
        "audio_io_preparation": inspect.getfile(audio_io),
        "listener_request": inspect.getfile(listener),
        "batch_driver": __file__,
        "renderer": inspect.getfile(NativeRenderer),
    }
    identity = {name: _sha(Path(path).read_bytes()) for name, path in files.items()}
    identity.update({
        "relative_instructions": relative_judge.RELATIVE_INSTRUCTIONS,
        "relative_schema": relative_judge.RelativeJudgment.model_json_schema(),
        "profile_instructions": listener.TONE_PROFILE_INSTRUCTIONS,
        "profile_schema": ToneProfile.model_json_schema(),
        "rubric_v3": evaluation_v3.RULES,
        "rubric_axes": evaluation.AXES,
        "relative_axes": list(relative_judge.RELATIVE_AXES),
        "max_output_tokens_relative": relative_judge.MAX_OUTPUT_TOKENS,
        "level_match_target_rms_dbfs": relative_judge.TARGET_RMS_DBFS,
        "retry_policy": "one SDK attempt per call; no retries; stop batch on any 429",
    })
    if renderer is not None:
        identity.update({"engine": renderer.engine_version, "worker_sha256": renderer._worker_sha256,
                         "plugin_sha256": renderer._plugin_sha256})
    return identity


class Batch:
    """Aggregate ledger shared by every process using the same batch directory.

    Admission and settlement reload the ledger from disk under an exclusive file lock and
    write it atomically, so separate processes can never over-admit the ceiling or lose
    another process's reservation. Tags are reserved atomically too.
    """

    def __init__(self, root: Path, ceiling: int = 16):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        self.ledger_path = root / "ledger.json"
        self.lock_path = root / ".ledger.lock"
        with self._locked():
            if not self.ledger_path.exists():
                _save(self.ledger_path, {"ceiling": ceiling, "attempts": [], "provider_pause": None})

    @contextlib.contextmanager
    def _locked(self):
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @property
    def ledger(self) -> dict:
        return _load(self.ledger_path)

    def freeze(self, renderer: NativeRenderer) -> str:
        identity = frozen_identity(renderer)
        digest = _sha(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
        path = self.root / "freeze.json"
        with self._locked():
            if path.exists():
                if _load(path)["sha256"] != digest:
                    raise SystemExit("Frozen judging/grading/engine identity changed; start a new batch directory.")
            else:
                _save(path, {"sha256": digest, **identity})
        return digest

    def tag_used(self, tag: str, ledger: dict | None = None) -> bool:
        ledger = ledger if ledger is not None else self.ledger
        return (self.root / "judgments" / f"{tag}.json").exists() or any(
            item.get("tag") == tag for item in ledger["attempts"])

    def reserve(self, kind: str, model: str, detail: dict) -> dict:
        with self._locked():
            ledger = _load(self.ledger_path)
            if ledger["provider_pause"]:
                raise SystemExit("Batch is paused after a provider 429; no request was sent.")
            if len(ledger["attempts"]) >= ledger["ceiling"]:
                raise SystemExit(f"Aggregate ceiling of {ledger['ceiling']} generate attempts reached; no request sent.")
            tag = detail.get("tag")
            if tag is not None and self.tag_used(tag, ledger):
                raise SystemExit(f"Tag {tag!r} is already reserved or recorded; no request sent.")
            entry = {"index": len(ledger["attempts"]) + 1, "kind": kind, "model": model,
                     "status": "sent", **detail}
            ledger["attempts"].append(entry)
            _save(self.ledger_path, ledger)
        return dict(entry)

    def settle(self, entry: dict, result: dict) -> None:
        info = result.get("failure_info")
        update = {
            "request_counts": result.get("request_counts"),
            "status": "failed" if info else "invalid" if result.get("invalid") else "ok",
            "http_status": info.get("http_status") if info else None,
        }
        entry.update(update)
        with self._locked():
            ledger = _load(self.ledger_path)
            stored = ledger["attempts"][entry["index"] - 1]
            if stored.get("tag") != entry.get("tag") or stored["status"] != "sent":
                raise SystemExit("Ledger entry changed unexpectedly; refusing to settle.")
            stored.update(update)
            if info and info.get("http_status") == 429:
                ledger["provider_pause"] = {"category": info.get("category"),
                                            "retry_after_seconds": info.get("retry_after_seconds")}
            _save(self.ledger_path, ledger)


def measure(samples: np.ndarray, rate: int) -> dict:
    """Local DSP descriptors of a render (mono), independent of any model."""
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size))) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1 / rate)
    total = float(spectrum.sum()) or 1.0
    bands = {"low_lt250": (0, 250), "mid_250_2k": (250, 2000), "presence_2k_5k": (2000, 5000),
             "high_5k_10k": (5000, 10000)}
    rms = float(np.sqrt(np.mean(mono ** 2)))
    return {
        "rms_dbfs": 20 * np.log10(rms) if rms > 0 else None,
        "crest_db": 20 * np.log10(float(np.max(np.abs(mono))) / rms) if rms > 0 else None,
        "spectral_centroid_hz": float((freqs * spectrum).sum() / total),
        "band_share_db": {name: 10 * np.log10(float(spectrum[(freqs >= lo) & (freqs < hi)].sum()) / total)
                          for name, (lo, hi) in bands.items()},
    }


def reference_profile(path: Path) -> tuple[ToneProfile, dict]:
    raw = path.read_bytes()
    result = json.loads(raw)
    profile = ToneProfile.model_validate(result["listening"])
    return profile, {"path": path.name, "sha256": _sha(raw), "model": result["model"],
                     "source": result["source"], "original_request_counts": result.get("request_counts")}


def build_candidate(adapter: SoundshedAdapter, renderer: NativeRenderer, base: ToneProfile, spec: dict):
    profile = ToneProfile.model_validate({**base.model_dump(), **spec.get("profile_overrides", {})})
    chain = adapter.default_chain([{"profile": profile}])
    nodes = [node for node in chain.nodes if node.id not in set(spec.get("remove", []))]
    records = adapter._resource_records()
    for node_id, record_id in spec.get("assets", {}).items():
        node = next(item for item in nodes if item.id == node_id)
        resolved = adapter._resolve_record(record_id, records[record_id])
        if resolved is None:
            raise SystemExit(f"Resource {record_id} is not available locally")
        index = nodes.index(node)
        nodes[index] = node.model_copy(update={"asset": resolved.asset,
                                                "title": f"{resolved.asset.display_name[:75]} (candidate)"})
    for key, value in spec.get("params", {}).items():
        node_id, param = key.split(".")
        index = next(i for i, item in enumerate(nodes) if item.id == node_id)
        nodes[index] = nodes[index].model_copy(update={"params": {**nodes[index].params, param: float(value)}})
    chain = ChainDraft(revision=1, nodes=nodes)
    errors = adapter.validate_chain(chain)
    if errors:
        raise SystemExit(f"{spec['id']}: " + "; ".join(errors))
    interval = Interval(start_seconds=19.2, end_seconds=30)
    return new_project(
        source=SourceRef(kind="youtube", url=REFERENCE_URL, display_name="Reference interval",
                         selected_intervals=[interval]),
        analysis=[{"interval": interval.model_dump(), "model": "gemini-3.6-flash", "profile": base.model_dump()}],
        intent={"summary": spec.get("purpose", "Tone-copier development candidate.")},
        chain=chain, engine_version=renderer.engine_version, name=f"Candidate {spec['id']}",
    )


def cmd_render(args, batch: Batch) -> None:
    adapter = SoundshedAdapter()
    renderer = NativeRenderer(adapter)
    renderer.initialize()
    batch.freeze(renderer)
    base, provenance = reference_profile(args.reference)
    for spec in _load(args.spec):
        directory = batch.root / "candidates" / spec["id"]
        spec_sha = _sha(json.dumps(spec, sort_keys=True).encode())
        if directory.exists():
            if _load(directory / "candidate.json")["spec_sha256"] != spec_sha:
                raise SystemExit(f"Candidate {spec['id']} is immutable; use a new id for changes.")
            continue
        directory.mkdir(parents=True)
        project = build_candidate(adapter, renderer, base, spec)
        entry = {"id": spec["id"], "spec": spec, "spec_sha256": spec_sha, "reference_profile": provenance,
                 "recipe_implementation_sha256": _sha(Path(inspect.getfile(SoundshedAdapter)).read_bytes()),
                 "chain_sha256": adapter.chain_sha256(project.chain),
                 "omitted_optional": [node.title for node in project.chain.nodes
                                      if node.kind == "gear_note" and node.note_role == "omitted_optional"],
                 "renders": {}}
        _save(directory / "project.json", project.model_dump(mode="json"))
        for sample in SAMPLES:
            rendered = renderer.render(project, sample, directory)
            wav = directory / f"{sample}.wav"
            (directory / f"{rendered.snapshot.id}.wav").rename(wav)
            archive = adapter.export_archive(project, rendered.snapshot)
            (directory / f"{sample}.soundshed.preset").write_bytes(archive)
            entry["renders"][sample] = {
                "wav": f"candidates/{spec['id']}/{sample}.wav", "audio_sha256": rendered.snapshot.audio_sha256,
                "preset_sha256": _sha(archive), "snapshot": rendered.snapshot.model_dump(mode="json"),
                "metrics": measure(rendered.samples, rendered.snapshot.sample_rate),
            }
        _save(directory / "candidate.json", entry)
        print(spec["id"], {s: round(entry["renders"][s]["metrics"]["spectral_centroid_hz"]) for s in SAMPLES})


def _clip(batch: Batch, candidate: str, sample: str) -> AudioClip:
    samples, rate = sf.read(batch.root / "candidates" / candidate / f"{sample}.wav", dtype="float32", always_2d=True)
    return AudioClip(samples=samples.T, sample_rate=rate, source="clip", start_seconds=0)


def _record(batch: Batch, name: str, result: dict) -> Path:
    directory = batch.root / "judgments"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.json"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise SystemExit(f"{path.name} already exists; judgments are never overwritten.") from None
    os.close(descriptor)
    _save(path, result)
    return path


def _require_live(args) -> None:
    if not args.live:
        raise SystemExit("--live is required for provider calls")
    load_default_credentials()
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not configured")


def cmd_relative(args, batch: Batch) -> None:
    _require_live(args)
    renderer = NativeRenderer(SoundshedAdapter())
    renderer.initialize()
    name = f"{args.tag}"
    if batch.tag_used(name):
        raise SystemExit("Judgment tag already used; no request sent.")
    freeze = batch.freeze(renderer)
    candidates = {item: _clip(batch, item, args.sample) for item in args.candidates}
    anchor = {"anchor_clip": _clip(batch, args.anchor, args.sample)} if args.anchor else {
        "youtube": (REFERENCE_URL, 19.2, 30.0)}
    entry = batch.reserve("relative", args.model, {"tag": name, "freeze_sha256": freeze,
                                                   "candidates": args.candidates, "sample": args.sample,
                                                   "anchor": args.anchor or "youtube 19.2-30s"})
    result = relative_judge.judge_relative(model=args.model, candidates=candidates, seed=args.seed, **anchor)
    result.update(freeze_sha256=freeze, sample=args.sample, anchor_candidate=args.anchor, tag=name,
                  candidate_audio_sha256={item: _load(batch.root / "candidates" / item / "candidate.json")
                                          ["renders"][args.sample]["audio_sha256"] for item in args.candidates})
    batch.settle(entry, result)
    _record(batch, name, result)
    print(json.dumps({"status": entry["status"], "finish": result.get("finish_reason"),
                      "unblinded": relative_judge.unblind(result),
                      "failure": result.get("failure_info"), "invalid": result.get("invalid")}, indent=1))


def cmd_absolute(args, batch: Batch) -> None:
    _require_live(args)
    if batch.tag_used(args.tag):
        raise SystemExit("Judgment tag already used; no request sent.")
    renderer = NativeRenderer(SoundshedAdapter())
    renderer.initialize()
    freeze = batch.freeze(renderer)
    reference, provenance = reference_profile(args.reference)
    candidate = _load(batch.root / "candidates" / args.candidate / "candidate.json")
    os.environ["TONE_LISTENER_MODEL"] = args.model
    entry = batch.reserve("absolute_profile", args.model, {"tag": args.tag, "freeze_sha256": freeze,
                                                           "candidate": args.candidate, "sample": args.sample})
    result = listen_tone_profile(_clip(batch, args.candidate, args.sample))
    result.update(freeze_sha256=freeze, candidate=args.candidate, sample=args.sample,
                  reference_reused=provenance)
    batch.settle(entry, {**result, "invalid": None})
    profile = result.get("listening")
    omitted = [field for field in ("reverb", "delay", "modulation")
               if any(field.title() in title for title in candidate["omitted_optional"])]
    result["grade_v3"] = (evaluation_v3.compare_profiles_v3(reference, ToneProfile.model_validate(profile), omitted)
                          if profile else None)
    _record(batch, args.tag, result)
    print(json.dumps({"status": entry["status"], "grade": (result["grade_v3"] or {}).get("outcome"),
                      "core": (result["grade_v3"] or {}).get("core", {}).get("known_failures"),
                      "unknown": (result["grade_v3"] or {}).get("core", {}).get("unknown_axes"),
                      "effects_unresolved": (result["grade_v3"] or {}).get("effects", {}).get("unresolved")},
                     indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render")
    render.add_argument("--spec", type=Path, required=True)
    render.add_argument("--reference", type=Path, required=True)
    relative = sub.add_parser("relative")
    relative.add_argument("--live", action="store_true")
    relative.add_argument("--model", required=True)
    relative.add_argument("--seed", type=int, required=True)
    relative.add_argument("--tag", required=True)
    relative.add_argument("--sample", choices=SAMPLES, default="riff-1")
    relative.add_argument("--anchor", help="Candidate id used as a local anchor (calibration); default YouTube reference.")
    relative.add_argument("--candidates", nargs="+", required=True)
    absolute = sub.add_parser("absolute")
    absolute.add_argument("--live", action="store_true")
    absolute.add_argument("--model", required=True)
    absolute.add_argument("--tag", required=True)
    absolute.add_argument("--candidate", required=True)
    absolute.add_argument("--sample", choices=SAMPLES, default="riff-1")
    absolute.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args(argv)
    batch = Batch(args.output)
    {"render": cmd_render, "relative": cmd_relative, "absolute": cmd_absolute}[args.command](args, batch)


if __name__ == "__main__":
    sys.exit(main())

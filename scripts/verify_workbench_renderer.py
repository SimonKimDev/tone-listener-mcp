#!/usr/bin/env python3
"""Opt-in installed-engine verification. No Gemini calls or live-profile writes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from tone_listener_mcp.workbench.adapters.soundshed import SoundshedAdapter, new_project
from tone_listener_mcp.workbench.models import ChainNode, Interval, SourceRef
from tone_listener_mcp.workbench.renderer import SAMPLE_RATE, NativeRenderer, RenderError


def verification_project(adapter):
    return new_project(
        source=SourceRef(
            kind="audio_file", display_name="Local verification DI",
            duration_seconds=16.7742,
            selected_intervals=[Interval(start_seconds=0, end_seconds=3)],
        ),
        analysis=[{
            "interval": {"start_seconds": 0, "end_seconds": 3},
            "model": "none", "unavailable": "Local integration verification; no reference analysis.",
        }],
        intent={"summary": "Installed-engine integration verification."},
        chain=adapter.self_test_chain(), engine_version=adapter.engine_version,
        name="Local renderer verification",
    )


def _wav_bytes(samples):
    stream = io.BytesIO()
    sf.write(stream, samples, SAMPLE_RATE, format="WAV", subtype="PCM_24")
    return stream.getvalue()


def run_verification(output: Path, previous_binary: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    adapter = SoundshedAdapter()
    renderer = NativeRenderer(adapter)
    renderer.initialize()
    project = verification_project(adapter)
    samples, rate = sf.read(
        Path(__file__).resolve().parents[1]
        / "src/tone_listener_mcp/workbench/assets/di/riff-1.wav",
        dtype="float32", always_2d=True,
    )
    assert rate == SAMPLE_RATE
    mono = samples[:, 0]
    trailer = renderer._trailer
    assert trailer is not None
    base = json.loads(adapter.host_state(project, trailer)[:-len(trailer)])
    variants = {}

    def variant(name):
        candidate = copy.deepcopy(base)
        variants[name] = candidate
        return candidate["preset"]["scenes"][0]["graph"]

    variant("original")
    graph = variant("all-processors-bypassed")
    for node in graph["nodes"][1:-1]:
        node["enabled"] = False
    graph = variant("input-minus24-output-plus24")
    graph["nodes"][0]["params"]["gainDb"] = -24
    graph["nodes"][-1]["params"]["gainDb"] = 24
    graph = variant("cabinet-output-plus24")
    next(node for node in graph["nodes"] if node["id"] == "cabinet")["params"]["outputGain"] = 24
    graph = variant("pedal-amp-swapped")
    graph["nodes"][1:3] = list(reversed(graph["nodes"][1:3]))
    graph["edges"] = [
        {"from": left["id"], "to": right["id"]}
        for left, right in zip(graph["nodes"], graph["nodes"][1:])
    ]
    variant("mixer-master-quarter")
    variants["mixer-master-quarter"]["mixer"]["masterGain"] = 0.25
    variant("global-output-plus12")
    variants["global-output-plus12"]["globalSignalChain"]["outputGain"] = 12
    variant("empty-scenes")
    variants["empty-scenes"]["preset"]["scenes"] = []
    variant("missing-preset")
    del variants["missing-preset"]["preset"]
    del variants["missing-preset"]["presetId"]
    variants["version-only"] = {"version": 1}
    variants["empty-object"] = {}
    graph = variant("input-output-only")
    graph["nodes"] = [graph["nodes"][0], graph["nodes"][-1]]
    graph["edges"] = [{"from": "__input__", "to": "__output__"}]
    graph = variant("nam-input-plus24")
    for node in graph["nodes"]:
        if node["id"] in ("pedal", "amp"):
            node["params"]["inputGain"] = 24
    graph = variant("input-minus24")
    graph["nodes"][0]["params"]["gainDb"] = -24
    graph = variant("both-nams-bypassed")
    for node in graph["nodes"]:
        if node["id"] in ("pedal", "amp"):
            node["enabled"] = False
    graph = variant("output-plus6")
    graph["nodes"][-1]["params"]["gainDb"] = 6
    report = {
        "engine_version": renderer.engine_version,
        "canary": renderer.canary_results,
        "conditions": {
            "profile_database_and_settings": "worker read-denied; resources only",
            "home_writes": "denied", "network": "denied",
            "previous_build": "optional pre-fix worker with pre-fix truncated wrapper",
        },
        "variants": [], "checks": {},
    }
    before = NativeRenderer(adapter, binary_path=previous_binary) if previous_binary else None
    audio = {}
    for name, state in variants.items():
        row = {"variant": name}
        payload = json.dumps(state, separators=(",", ":")).encode()
        for label, worker in (("after", renderer), ("before", before)):
            if worker is None:
                continue
            wrapped = payload + (trailer[8:] if label == "before" else trailer)
            try:
                rendered = worker._render_raw(wrapped, mono)
                wav = _wav_bytes(rendered)
                rms = float(np.sqrt(np.mean(rendered.astype(np.float64) ** 2)))
                row[label] = {
                    "wav_sha256": hashlib.sha256(wav).hexdigest(),
                    "rms_dbfs": float(20 * np.log10(rms)),
                    "peak": float(np.max(np.abs(rendered))),
                }
                if label == "after":
                    audio[name] = rendered
                    (output / f"{name}.wav").write_bytes(wav)
            except RenderError as exc:
                row[label] = {"rejected": str(exc)}
        report["variants"].append(row)
    assert len(report["variants"]) == 16
    rejected = {"empty-scenes", "missing-preset", "version-only", "empty-object"}
    assert all(name not in audio for name in rejected), "Incomplete states must never render defaults"
    assert len(audio) == len(variants) - len(rejected)
    assert np.array_equal(audio["all-processors-bypassed"], audio["input-output-only"])
    assert all(
        not np.array_equal(audio["original"], value)
        for name, value in audio.items() if name not in ("original", "mixer-master-quarter")
    ), "State-insensitive output"
    report["checks"]["legacy_mixer_master_gain_field_changes_audio"] = not np.array_equal(
        audio["original"], audio["mixer-master-quarter"]
    )
    delta = float(20 * np.log10(
        np.linalg.norm(audio["output-plus6"]) / np.linalg.norm(audio["original"])
    ))
    assert abs(delta - 6) <= 0.1
    report["checks"]["utility_gain_delta_db"] = delta

    # Export the actual rendered snapshot, then apply its parsed preset in a FRESH worker.
    # This tests the installed preset parser, not the standalone app's ZIP importer UI.
    preview = renderer.render(project, "riff-1", output)
    archive = adapter.export_archive(project, preview.snapshot)
    (output / "verified.soundshed.preset").write_bytes(archive)
    import zipfile
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        parsed = json.loads(package.read("preset.json"))
    exported = parsed["preset"]
    runtime = copy.deepcopy(exported)
    for node in runtime["scenes"][0]["graph"]["nodes"]:
        source = next((item for item in project.chain.nodes if item.id == node["id"]), None)
        if source is not None and source.asset is not None:
            node["resources"][0]["filePath"] = str(adapter.resolve_asset(source.asset).path)
    padded = np.pad(mono, (0, SAMPLE_RATE * 10))
    restored = renderer._render_raw(adapter.state_for_preset(runtime, trailer), padded)
    assert np.array_equal(restored[:len(preview.samples)], preview.samples)
    report["checks"]["export_fresh_worker_max_error"] = 0.0
    report["checks"]["standalone_archive_import_ui"] = "not performed"
    report["snapshot"] = preview.snapshot.model_dump(mode="json")
    report["checks"]["preset_sha256"] = hashlib.sha256(archive).hexdigest()

    for sample in ("riff-1", "riff-2"):
        candidate = project.model_copy(deep=True)
        candidate.chain.nodes.insert(-1, ChainNode(
            id="delay", kind="digital_delay", title="Tail verification",
            params={"time": 500, "feedback": 0.35, "mix": 0.4},
        ))
        result = renderer.render(candidate, sample, output)
        input_duration = result.snapshot.dsp_conditions["input_duration_seconds"]
        assert result.snapshot.duration_seconds > input_duration + 0.5
        tail = result.samples[int(input_duration * SAMPLE_RATE):]
        assert np.max(np.abs(tail)) > 1e-4
        report["checks"][f"{sample}_tail_seconds"] = result.snapshot.duration_seconds - input_duration
    from tone_listener_mcp.listener import ToneProfile
    absent = {"presence": "absent", "prominence": "none"}
    report["recipe_contrasts"] = []
    for gain, brightness in (("clean", "balanced"), ("crunch", "bright"), ("high-gain", "dark")):
        profile = ToneProfile(
            guitar_dominance="clear", gain=gain, brightness=brightness,
            bass_level="typical", bass_tightness="neutral", mids="neutral",
            attack="balanced", compression="moderate", sustain="typical",
            reverb=absent, delay=absent, modulation=absent, evidence=[], uncertainties=[],
        )
        candidate = project.model_copy(deep=True)
        candidate.chain = adapter.default_chain([{"profile": profile}])
        result = renderer.render(candidate, "riff-1", output)
        report["recipe_contrasts"].append({
            "gain": gain, "brightness": brightness,
            "chain_sha256": result.snapshot.chain_sha256,
            "audio_sha256": result.snapshot.audio_sha256,
            "rms": result.snapshot.rms, "peak": result.snapshot.peak,
        })
    assert len({item["audio_sha256"] for item in report["recipe_contrasts"]}) == 3
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Consent to installed VST3 execution.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-binary", type=Path)
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; this verification is excluded from offline CI")
    report = run_verification(args.output, args.previous_binary)
    print(json.dumps({"canary": report["canary"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()

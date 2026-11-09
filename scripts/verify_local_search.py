#!/usr/bin/env python3
"""Native evidence for same-gear local refinement search. Local renders only; no Gemini calls.

For an existing project (read-only), builds the ``local`` refinement variants and renders each on both
licensed DIs (riffs by default; ``--samples`` adds the Guitar-TECHS leads) with the installed, sandboxed Soundshed engine. It checks that every intended parameter delta
is present in the exact host state sent to the engine and audibly changes the DSP output, then measures:

* trims: a level-independent nonlinearity residual (best scalar fit of the variant to the current render),
  crest factor and RMS-normalized spectral centroid, so a pure level change is not mistaken for drive;
* EQ: octave-band energy change on the SAME DI relative to the current chain.

Output headroom (peak, samples over full scale) is measured on EVERY output channel of the native render;
the tonal measurements are explicitly labelled as using a mono downmix. Each delivered WAV is a byte-for-byte
copy of the renderer's own snapshot WAV, so its file SHA-256 equals the snapshot's ``audio_sha256``; the full
snapshot and the source DI identity are recorded with it.

These are engine/DSP facts only. Nothing here says a variant is closer to any reference.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from tone_listener_mcp.workbench import refinement
from tone_listener_mcp.workbench.adapters.soundshed import SoundshedAdapter
from tone_listener_mcp.workbench.models import DEFAULT_DI_SAMPLES, DI_SAMPLES, ToneProject
from tone_listener_mcp.workbench.renderer import SAMPLE_RATE, NativeRenderer

DI_ROOT = Path(refinement.__file__).parent / "assets" / "di"

BANDS_HZ = (125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def _mono(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float64).mean(axis=1) if samples.ndim == 2 else samples.astype(np.float64)


def _db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-12)))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _band_db(x: np.ndarray) -> dict[str, float]:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    out = {}
    for centre in BANDS_HZ:
        mask = (freqs >= centre / np.sqrt(2)) & (freqs < centre * np.sqrt(2))
        out[str(centre)] = float(10 * np.log10(max(spectrum[mask].sum(), 1e-24)))
    return out


def _centroid(x: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    return float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-24))


def headroom(samples: np.ndarray) -> dict:
    """Full-channel output headroom of the native render (frames x channels); never a downmix."""
    frames = samples.reshape(len(samples), -1).astype(np.float64)
    magnitude = np.abs(frames)
    return {"basis": "all output channels of the native render", "channels": int(frames.shape[1]),
            "finite": bool(np.isfinite(frames).all()), "peak": float(magnitude.max()),
            "peak_per_channel": [float(value) for value in magnitude.max(axis=0)],
            "samples_over_full_scale": int(np.sum(magnitude > 1.0)),
            "frames_with_any_channel_over_full_scale": int(np.sum((magnitude > 1.0).any(axis=1)))}


def _energy_ratio_db(x: np.ndarray, low_hz: float, high_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    band = spectrum[(freqs >= low_hz) & (freqs < high_hz)].sum()
    return float(10 * np.log10(max(band, 1e-24) / max(spectrum.sum(), 1e-24)))


def mono_tonal(x: np.ndarray) -> dict:
    """Tonal descriptors of the mono downmix (labelled; NOT output headroom)."""
    return {"basis": "mono downmix (mean of channels); tonal descriptors only, not output headroom",
            "rms_dbfs": _db(_rms(x)), "crest_db": _db(float(np.max(np.abs(x))) / max(_rms(x), 1e-12)),
            "centroid_hz": _centroid(x),
            "energy_below_200hz_ratio_db": _energy_ratio_db(x, 0, 200),
            "energy_above_2khz_ratio_db": _energy_ratio_db(x, 2000, SAMPLE_RATE)}


def di_identity(samples: tuple[str, ...] = DEFAULT_DI_SAMPLES) -> dict:
    provenance = json.loads((DI_ROOT / "provenance.json").read_text())
    files = {item["file"]: {"upstream_file": item["upstream_file"], "repository": provenance["repository"],
                            "revision": provenance["revision"], "license": provenance["license"],
                            "sha256": item["sha256"]} for item in provenance["samples"]}
    for source in provenance.get("additional_sources", []):
        files.update({item["file"]: {"upstream_file": item["upstream_member"], "record": source["record"],
                                     "source_file_sha256": source["source_file_sha256"],
                                     "upstream_sha256": item["upstream_sha256"], "license": source["license"],
                                     "trim_seconds": [item["trim_start_seconds"], item["trim_end_seconds"]],
                                     "sha256": item["sha256"]} for item in source["samples"]})
    out = {}
    for sample in samples:
        digest = hashlib.sha256((DI_ROOT / f"{sample}.wav").read_bytes()).hexdigest()
        assert digest == files[f"{sample}.wav"]["sha256"], f"DI {sample} differs from its provenance record"
        out[sample] = {"file": f"{sample}.wav", **files[f"{sample}.wav"]}
    return out


def deliver_artifact(cache_root: Path, snapshot, output: Path, name: str) -> dict:
    """Copy the renderer's snapshot WAV byte-for-byte; the artifact hash must equal the snapshot hash."""
    source = cache_root / f"{snapshot.id}.wav"
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != snapshot.audio_sha256:
        raise RuntimeError(f"renderer WAV for {name} does not match its snapshot hash")
    (output / name).write_bytes(data)
    return {"file": name, "sha256": digest, "snapshot_audio_sha256": snapshot.audio_sha256,
            "byte_identical_to_renderer_snapshot_wav": True}


def _residual(current: np.ndarray, variant: np.ndarray) -> float:
    """Relative error after the best single gain; ~0 for a pure level change, larger for a nonlinear change."""
    n = min(len(current), len(variant))
    a, b = current[:n], variant[:n]
    gain = float(np.dot(a, b) / max(np.dot(a, a), 1e-24))
    return float(np.linalg.norm(b - gain * a) / max(np.linalg.norm(b), 1e-24))


def _changed_params(base, chain) -> list[dict]:
    before = {node.id: node.params for node in base.nodes}
    return [{"node": node.id, "param": key, "from": before[node.id][key], "to": value}
            for node in chain.nodes for key, value in node.params.items() if before[node.id].get(key) != value]


def run(project_path: Path, output: Path, samples: tuple[str, ...] = DEFAULT_DI_SAMPLES) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    adapter = SoundshedAdapter()
    renderer = NativeRenderer(adapter)
    renderer.initialize()
    project = ToneProject.model_validate_json(project_path.read_text())
    project = project.model_copy(update={"target": project.target.model_copy(
        update={"engine_version": renderer.engine_version})})
    profile = project.analysis[project.selected_analysis_index].profile
    variants = refinement.build_variants(adapter, project.chain, profile, "local")
    trailer = renderer._trailer
    report = {"engine_version": renderer.engine_version, "project_sha256": hashlib.sha256(
        project_path.read_bytes()).hexdigest(), "search": "local", "source_di": di_identity(samples),
        "variants": [{"id": item.id, "role": item.role, "label": item.label, "chain_sha256": item.chain_sha256,
                      "changed": _changed_params(project.chain, item.chain) if item.role == "variant" else None}
                     for item in variants],
        "renders": {}, "checks": {}}
    audio: dict[str, dict[str, np.ndarray]] = {}
    for item in variants:
        candidate = project.model_copy(update={"chain": item.chain})
        if item.role == "variant":
            state = json.loads(adapter.host_state(candidate, trailer)[:-len(trailer)])
            nodes = {node["id"]: node for node in state["preset"]["scenes"][0]["graph"]["nodes"]}
            for change in _changed_params(project.chain, item.chain):
                sent = nodes[change["node"]]["params"][change["param"]]
                assert abs(float(sent) - change["to"]) < 1e-9, (item.id, change, sent)
        for sample in samples:
            rendered = renderer.render(candidate, sample, output / "cache")
            if rendered.snapshot.source_sample_sha256 != report["source_di"][sample]["sha256"]:
                raise RuntimeError(f"render of {item.id}/{sample} used an unexpected DI")
            mono = _mono(rendered.samples)
            audio.setdefault(item.id, {})[sample] = mono
            report["renders"].setdefault(item.id, {})[sample] = {
                "artifact": deliver_artifact(output / "cache", rendered.snapshot, output, f"{item.id}-{sample}.wav"),
                "snapshot": rendered.snapshot.model_dump(mode="json"),
                "output_headroom": headroom(rendered.samples), "mono_tonal": mono_tonal(mono)}
    for item in variants:
        if item.role != "variant":
            continue
        for sample in samples:
            base, mine = audio["current"][sample], audio[item.id][sample]
            n = min(len(base), len(mine))
            row = report["renders"][item.id][sample]
            row["differs_from_current"] = not np.array_equal(base[:n], mine[:n])
            row["level_db_vs_current"] = _db(_rms(mine)) - _db(_rms(base))
            row["nonlinear_residual_vs_current"] = _residual(base, mine)
            normalized_base, normalized_mine = base[:n] / _rms(base[:n]), mine[:n] / _rms(mine[:n])
            row["centroid_hz_change"] = _centroid(normalized_mine) - _centroid(normalized_base)
            row["crest_db_change"] = mono_tonal(mine)["crest_db"] - mono_tonal(base)["crest_db"]
            row["comparison_basis"] = "mono downmix vs current chain on the same DI"
            bands_base, bands_mine = _band_db(base[:n]), _band_db(mine[:n])
            row["octave_band_db_change"] = {band: bands_mine[band] - bands_base[band] for band in bands_base}
            assert row["differs_from_current"], (item.id, sample)
    trims = sorted((item for item in variants if item.id.startswith("trim_")),
                   key=lambda item: _changed_params(project.chain, item.chain)[0]["to"])
    for sample in samples:
        order = [report["renders"][item.id][sample]["crest_db_change"] for item in trims]
        report["checks"][f"{sample}_crest_change_by_increasing_trim"] = dict(
            zip([item.id for item in trims], order, strict=True))
        report["checks"][f"{sample}_crest_monotonic_non_increasing"] = all(
            later <= earlier + 1e-9 for earlier, later in itertools.pairwise(order))
    # Joint moves must be exactly the union of their single-axis components' parameter changes.
    by_id = {item.id: item for item in variants}
    for item in variants:
        if not item.id.startswith("joint_"):
            continue
        _, trim_part, tilt = item.id.split("_")
        parts = [by_id.get(f"trim_{trim_part}"), by_id.get(f"eq_{tilt}")]
        expected = sorted((change["node"], change["param"], change["to"]) for part in parts if part is not None
                          for change in _changed_params(project.chain, part.chain))
        actual = sorted((change["node"], change["param"], change["to"])
                        for change in _changed_params(project.chain, item.chain))
        comparison = {"components": [part.id if part else None for part in parts],
                      "parameters_equal_union_of_components": None not in parts and expected == actual}
        for sample in samples:
            row = report["renders"][item.id][sample]
            comparison[sample] = {
                key: {"joint": row[key], **{part.id: report["renders"][part.id][sample][key]
                                            for part in parts if part is not None}}
                for key in ("level_db_vs_current", "centroid_hz_change", "crest_db_change")}
        report["checks"][f"{item.id}_composition"] = comparison
    rows = [row for item in report["renders"].values() for row in item.values()]
    report["checks"]["all_finite"] = all(row["output_headroom"]["finite"] for row in rows)
    report["checks"]["max_output_peak_all_channels"] = max(row["output_headroom"]["peak"] for row in rows)
    report["checks"]["max_non_control_output_peak_all_channels"] = max(
        row["output_headroom"]["peak"] for key, item in report["renders"].items() if key != "control"
        for row in item.values())
    report["limitations"] = [
        "Engine/DSP measurements on the licensed DIs only; no claim that any variant is closer to a reference.",
        "Native renders are float and may exceed full scale; full-channel peaks are reported, not clipped or hidden.",
        "Tonal descriptors use a mono downmix and are labelled as such; they are not headroom measurements.",
        "Crest factor and the scalar-fit residual are proxies for added saturation, not perceptual gain labels.",
    ]
    (output / "local-search-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", choices=DI_SAMPLES, default=list(DEFAULT_DI_SAMPLES))
    args = parser.parse_args()
    report = run(args.project, args.output, tuple(dict.fromkeys(args.samples)))
    print(json.dumps({"variants": [(item["id"], item["changed"]) for item in report["variants"]],
                      "checks": report["checks"]}, indent=1))


if __name__ == "__main__":
    main()

"""Offline checks for the separate 12-attempt accuracy phase: zero provider calls, independent ledger."""

import hashlib
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from test_workbench_recipes import adapter as recipe_adapter  # noqa: F401 - pytest fixture
from test_workbench_recipes import profile as recipe_profile
from test_workbench_refinement import FakeJudge, _register_extras

from tone_listener_mcp.audio_io import prepare_upload
from tone_listener_mcp.workbench import evaluation_v3
from tone_listener_mcp.workbench import refinement as refine
from tone_listener_mcp.workbench.adapters.soundshed import SoundshedAdapter, new_project
from tone_listener_mcp.workbench.models import Interval, PreviewSnapshot, SourceRef
from tone_listener_mcp.workbench.renderer import RenderedAudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tone_accuracy_local as phase_mod
import tone_copier_final as final

REFERENCE = recipe_profile(gain="overdrive", brightness="bright", mids="forward", bass_level="low",
                           bass_tightness="tight", attack="sharp", compression="compressed")
HISTORICAL = recipe_profile(gain="crunch", brightness="balanced", mids="present", bass_level="typical",
                            bass_tightness="tight", attack="sharp", compression="moderate")


class ChainRenderer:
    """Deterministic audio derived from the chain hash; level controlled per test."""

    engine_version = "test-engine"
    _worker_sha256 = "w" * 64
    _plugin_sha256 = "p" * 64

    def __init__(self, level=0.2):
        self.renders, self.level, self.spiky = 0, level, set()

    def render(self, project, sample_name, cache_root, cancel_event=None):
        self.renders += 1
        digest = SoundshedAdapter.chain_sha256(project.chain)
        rng = np.random.default_rng(int(digest[:8], 16) + (sample_name == "riff-2"))
        amplitude = self.level * (0.5 + rng.random())
        mono = amplitude * np.sin(np.linspace(0, 2000, 24000) * (1 + rng.random()))
        if digest in self.spiky:  # same peak, tiny RMS: a very high crest factor
            mono = np.zeros_like(mono)
            mono[::4000] = 0.5
        stereo = np.stack([mono, mono], axis=1).astype("<f4")
        path = Path(cache_root)
        snapshot = PreviewSnapshot(
            project_id=project.id, project_revision=project.revision, chain_revision=project.chain.revision,
            chain_sha256=digest, sample_name=sample_name, source_sample_sha256="1" * 64,
            engine_version=self.engine_version, sample_rate=48000, dsp_conditions={"state_canary_version": 1},
            audio_sha256="2" * 64, duration_seconds=0.5, rms=0.1, peak=0.2, created_at="now")
        wav = path / f"{snapshot.id}.wav"
        sf.write(wav, stereo, 48000, subtype="FLOAT")
        snapshot = snapshot.model_copy(update={"audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest()})
        return RenderedAudio(samples=stereo, snapshot=snapshot, sample_name=sample_name)


def _final_dir(root: Path, adapter, renderer) -> Path:
    """Synthetic stand-in for the immutable final-phase case-01 files."""
    chain = adapter.default_chain([{"profile": REFERENCE}])
    pushed = {item.id: item for item in refine.build_variants(adapter, chain, REFERENCE)}["pushed_clean_amp"].chain
    interval = Interval(start_seconds=19.2, end_seconds=30.0)
    project = new_project(source=SourceRef(kind="youtube", url="https://www.youtube.com/watch?v=x",
                                           display_name="ref", selected_intervals=[interval]),
                          analysis=[{"interval": interval.model_dump(), "model": "m",
                                     "profile": REFERENCE.model_dump()}],
                          intent={"summary": "test"}, chain=pushed,
                          engine_version="test-engine", name="Final case-01")
    case = root / "case-01"
    (case / "variants").mkdir(parents=True)
    (case / "renders").mkdir()
    (root / "judgments").mkdir()
    (root / "reports").mkdir()
    project_path = case / "variants" / "pushed_clean_amp.project.json"
    final.driver._save(project_path, project.model_dump(mode="json"))
    renders = {}
    for sample in final.SAMPLES:
        rendered = renderer.render(project, sample, case / "renders")
        wav = case / "renders" / f"pushed_clean_amp-{sample}.wav"
        (case / "renders" / f"{rendered.snapshot.id}.wav").rename(wav)
        renders[sample] = {"file_sha256": hashlib.sha256(wav.read_bytes()).hexdigest()}
    final.driver._save(case / "artifacts.json", {"variants": {"pushed_clean_amp": {
        "project_sha256": hashlib.sha256(project_path.read_bytes()).hexdigest(), "renders": renders}}})
    final.driver._save(root / "judgments" / "case-01:reference.json", {"listening": REFERENCE.model_dump()})
    per_di = {}
    for sample in final.SAMPLES:
        final.driver._save(root / "judgments" / f"case-01:eval-{sample}.json", {"listening": HISTORICAL.model_dump()})
        per_di[sample] = {"grade_v3": evaluation_v3.compare_profiles_v3(REFERENCE, HISTORICAL,
                                                                         final._omitted(pushed))}
    final.driver._save(root / "reports" / "report-01.json", {"cases": [{
        "case": "case-01", "selection": {"selected": "pushed_clean_amp"}, "evaluation": {"per_di": per_di}}]})
    return root


class Providers:
    """Relative judge prefers one candidate per model; profiles follow a role -> profile mapping."""

    def __init__(self, preferred=None, profiles=None, fail_at=None):
        self.calls = []
        self.preferred = preferred or {}
        self.profiles = profiles or {}
        self.fail_at = fail_at
        self.uploads = []

    def relative(self, *, model, candidates, seed, youtube):
        assert youtube == (phase_mod.URL, *phase_mod.INTERVAL)
        self.calls.append(("relative", model, seed))
        preferred = self.preferred.get(model)
        judge = FakeJudge(current_gain="similar", control_gain="more", preferred=preferred or "none",
                          preferred_gain="similar")
        result = judge(model=model, candidates=candidates, seed=seed, youtube=youtube)
        if preferred:
            # The preferred candidate is judged closer on brightness and ranked above current.
            for item in result["judgment"]["comparisons"]:
                identifier = result["label_mapping"][item["clip"]]
                item["brightness"] = "similar" if identifier == preferred else "less"
        if self.fail_at == len(self.calls):
            result.update(judgment=None, failure_info={"http_status": 429, "category": "quota_unspecified"})
        return result

    def profile(self, clip, model):
        assert model == phase_mod.PROFILE_MODEL and clip.sample_rate == 16000 and clip.samples.shape[0] == 1
        self.calls.append(("profile", clip.source))
        self.uploads.append(clip)
        profile = self.profiles.get(clip.source.split("-")[0], HISTORICAL)
        return {"request_counts": {"generate": 1}, "failure_info": None, "listening": profile.model_dump()}


@pytest.fixture
def setup(tmp_path, monkeypatch, recipe_adapter):  # noqa: F811
    monkeypatch.setenv("TONE_EVAL_PHASE_REGISTRY", str(tmp_path / "registry"))
    _register_extras(recipe_adapter)
    renderer = ChainRenderer()
    final_dir = _final_dir(tmp_path / "final", recipe_adapter, renderer)
    args = Namespace(output=tmp_path / "phase", final_dir=final_dir, live=False, revise=False, reason=None)
    phase_mod.cmd_freeze(args, recipe_adapter, renderer)
    return args, recipe_adapter, renderer


def _snapshot(directory: Path) -> dict:
    return {str(path): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


def test_plan_is_12_with_zero_spend_and_does_not_touch_final(setup):
    args, adapter, _ = setup
    before = _snapshot(args.final_dir)
    plan = phase_mod.cmd_plan(args, adapter)
    assert plan["ledger"] == {"ceiling": 12, "attempted": 0, "provider_pause": None}
    assert plan["max_new_generate_requests"] == {"relative_A_B": 8, "absolute_C_if_gated": 4, "total": 12}
    roles = [item["role"] for item in plan["candidate_set"]]
    assert roles.count("current") == 1 and roles.count("control") == 1 and len(roles) == 8
    assert plan["historical_baseline_gaps"]["riff-1"]["gain"] == 1
    assert plan["historical_baseline_gaps"]["riff-1"]["brightness"] == 2
    assert _snapshot(args.final_dir) == before
    assert not (Path(phase_mod.registry_path()) / "tone-copier-final-v1.json").exists()


def test_no_support_skips_c_after_exactly_eight_calls(setup):
    args, adapter, renderer = setup
    providers = Providers()
    report = phase_mod.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 8 == report["ledger"]["attempted"]
    assert report["outcome"] == "c_skipped" and report["relative"]["gate"]["union"] == []
    assert all(report["steps"][phase_mod._c_tag(r, s)]["status"] == "not_attempted" for r, s in phase_mod.C_STEPS)
    assert all(report["candidates"]["current_matches_historical_baseline_pcm"].values())
    renders = renderer.renders
    again = phase_mod.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 8 and renderer.renders == renders
    assert again["outcome"] == "c_skipped"


def test_b_seed_mapping_differs_from_a(setup):
    args, adapter, renderer = setup
    providers = Providers()
    phase_mod.cmd_run(args, adapter, renderer, providers)
    judgments = args.output / "judgments"
    for model in phase_mod.RELATIVE_MODELS:
        for sample in phase_mod.SAMPLES:
            a = final.driver._load(judgments / f"A:{model}:{sample}.json")["label_mapping"]
            b = final.driver._load(judgments / f"B:{model}:{sample}.json")["label_mapping"]
            assert a != b


def test_models_supporting_different_candidates_skip_c(setup):
    args, adapter, renderer = setup
    flash, argon = phase_mod.RELATIVE_MODELS
    providers = Providers(preferred={flash: "eq_brighter", argon: "trim_plus3"})
    report = phase_mod.cmd_run(args, adapter, renderer, providers)
    gate = report["relative"]["gate"]
    assert gate["supported_by_model"] == {flash: "eq_brighter", argon: "trim_plus3"}
    assert not gate["run_c"] and "different" in gate["reason"] and len(providers.calls) == 8


def test_one_model_support_runs_c_with_matched_copies(setup):
    args, adapter, renderer = setup
    flash, _ = phase_mod.RELATIVE_MODELS
    better = HISTORICAL.model_copy(update={"brightness": "bright", "gain": "overdrive"})
    providers = Providers(preferred={flash: "eq_brighter"}, profiles={"candidate": better})
    report = phase_mod.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 12 == report["ledger"]["attempted"]
    assert report["relative"]["gate"]["support_basis"] == "one-model"
    prep = report["c_preparation"]
    for sample in phase_mod.SAMPLES:
        rows = prep["samples"][sample]
        assert rows["baseline"]["scalar"] == 1.0
        base_upload = sf.read(args.output / "c-inputs" / rows["baseline"]["upload_file"], dtype="float32")[0]
        historical = sf.read(args.final_dir / "case-01" / "renders" / f"pushed_clean_amp-{sample}.wav",
                             dtype="float32", always_2d=True)[0]
        prepared = prepare_upload(phase_mod.AudioClip(historical.T, 48000, "h", 0)).samples[0].astype("<f4")
        assert np.array_equal(base_upload, prepared)  # baseline left at its historical prepared level
        cand_upload = sf.read(args.output / "c-inputs" / rows["candidate"]["upload_file"], dtype="float32")[0]
        assert np.sqrt(np.mean(cand_upload.astype(float) ** 2)) == pytest.approx(
            np.sqrt(np.mean(base_upload.astype(float) ** 2)), rel=1e-5)
    c = report["c"]
    assert c["baseline_gaps_reproduced"] is True and c["candidate_profiles_repeated"] is False
    assert c["beats_baseline_categorically"] is True
    assert report["outcome"] == "candidate_beats_baseline_reproduced"
    assert "not excluded" in c["interpretation"]
    assert c["perceptual_match_claim"] is False
    assert c["per_di"]["riff-1"]["axis_deltas"]["brightness"]["gap_change"] == -2


def test_baseline_not_reproduced_is_reported(setup):
    args, adapter, renderer = setup
    flash, argon = phase_mod.RELATIVE_MODELS
    drifted = HISTORICAL.model_copy(update={"brightness": "bright"})
    providers = Providers(preferred={flash: "eq_brighter", argon: "eq_brighter"},
                          profiles={"baseline": drifted, "candidate": drifted})
    report = phase_mod.cmd_run(args, adapter, renderer, providers)
    assert report["relative"]["gate"]["support_basis"] == "cross-model"
    assert report["c"]["baseline_gaps_reproduced"] is False
    assert report["outcome"] == "candidate_does_not_beat_baseline_not_reproduced"
    assert "not show that there was no real improvement" in report["c"]["interpretation"]


def test_preparation_over_full_scale_blocks_c_without_calls(setup):
    args, adapter, renderer = setup
    variants = {item.id: item for item in phase_mod.build(adapter, phase_mod.Phase(args.output))}
    renderer.spiky.add(variants["eq_brighter"].chain_sha256)  # current stays identical to the historical baseline
    flash, _ = phase_mod.RELATIVE_MODELS
    providers = Providers(preferred={flash: "eq_brighter"})
    report = phase_mod.cmd_run(args, adapter, renderer, providers)
    assert all(report["candidates"]["current_matches_historical_baseline_pcm"].values())
    assert report["outcome"] == "c_preparation_blocked" and "exceeds full scale" in report["reason"]
    assert len(providers.calls) == 8 == report["ledger"]["attempted"]


def test_first_429_pauses_phase(setup):
    args, adapter, renderer = setup
    providers = Providers(fail_at=3)
    report = phase_mod.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 3 == report["ledger"]["attempted"]
    assert report["ledger"]["provider_pause"] and report["outcome"] == "c_skipped"
    assert "unusable" in report["relative"]["gate"]["reason"]
    phase_mod.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 3


def test_beats_rule_requires_every_condition():
    def grade(gain, bright, mids, failures):
        axes = {axis: {"reference": "x", "candidate": "y", "difference": 0} for axis in evaluation_v3.CORE_AXES}
        axes.update(gain={"reference": "x", "candidate": "y", "difference": gain},
                    brightness={"reference": "x", "candidate": "y", "difference": bright},
                    mids={"reference": "x", "candidate": "y", "difference": mids})
        return {"guitar_clear_in_both": True, "core": {"axes": axes, "known_failures": ["f"] * failures}}

    base = grade(1, 2, 1, 2)
    assert phase_mod.beats_on_di(base, grade(1, 1, 1, 1))["beats"] is True
    assert phase_mod.beats_on_di(base, grade(2, 0, 1, 1))["beats"] is False  # gain worsened
    assert phase_mod.beats_on_di(base, grade(1, 1, 2, 1))["beats"] is True  # mids +1 keeps total equal
    assert phase_mod.beats_on_di(base, grade(1, 1, 3, 1))["beats"] is False  # total increased
    assert phase_mod.beats_on_di(base, grade(1, 1, 1, 3))["beats"] is False  # more threshold failures
    assert phase_mod.beats_on_di(base, grade(1, 2, 0, 1))["beats"] is False  # sum not strictly smaller
    assert phase_mod.beats_on_di(base, None)["beats"] is None


def test_phase_cannot_be_reset_by_recreating_output(setup):
    args, _adapter, renderer = setup
    phase = phase_mod.Phase(args.output)
    phase.reserve("relative", "m", {"tag": "probe"})
    archived = args.output.with_name("archived")
    args.output.rename(archived)
    with pytest.raises(SystemExit):
        phase_mod.Phase(args.output, create=True)
    with pytest.raises(SystemExit, match="refreeze|reservations"):
        archived.rename(args.output)
        phase_mod.Phase(args.output).freeze_phase(renderer, {"other": "x"}, revise=True)


def test_tampered_imported_input_blocks_before_any_call(setup):
    args, adapter, renderer = setup
    target = args.output / "inputs" / "pushed_clean_amp-riff-1.wav"
    target.chmod(0o600)
    target.write_bytes(target.read_bytes()[:-4] + b"\0\0\0\0")
    providers = Providers()
    with pytest.raises(SystemExit, match="Frozen identity changed"):  # inputs are bound by the freeze
        phase_mod.cmd_run(args, adapter, renderer, providers)
    assert providers.calls == [] and phase_mod.Phase(args.output).ledger["attempts"] == []
    with pytest.raises(phase_mod.Blocked, match="changed"):  # direct verification also refuses
        phase_mod.import_inputs(phase_mod.Phase(args.output), None)


def test_final_dir_not_matching_its_manifest_refuses_freeze(tmp_path, monkeypatch, recipe_adapter):  # noqa: F811
    monkeypatch.setenv("TONE_EVAL_PHASE_REGISTRY", str(tmp_path / "registry"))
    _register_extras(recipe_adapter)
    final_dir = _final_dir(tmp_path / "final", recipe_adapter, ChainRenderer())
    wav = final_dir / "case-01" / "renders" / "pushed_clean_amp-riff-2.wav"
    wav.write_bytes(wav.read_bytes()[:-4] + b"\0\0\0\0")
    args = Namespace(output=tmp_path / "phase", final_dir=final_dir, live=False, revise=False, reason=None)
    with pytest.raises(SystemExit, match="does not match"):
        phase_mod.cmd_freeze(args, recipe_adapter, ChainRenderer())


def test_binding_note_names_this_phase(setup):
    args, _adapter, _renderer = setup
    note = final.driver._load(args.output / "phase-binding.json")
    assert note["phase"] == phase_mod.PHASE
    final.driver._save(args.output / "phase-binding.json", {**note, "phase": "tone-copier-final-v1"})
    with pytest.raises(SystemExit, match="does not name"):
        phase_mod.Phase(args.output)


def test_changed_current_audio_blocks_before_any_call_fresh_and_resumed(setup):
    args, adapter, renderer = setup
    renderer.level *= 0.8
    providers = Providers()
    report = phase_mod.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "blocked" and "historical baseline" in report["reason"]
    assert providers.calls == [] and not (args.output / "candidates" / "artifacts.json").exists()
    renderer.level /= 0.8
    phase_mod.cmd_run(args, adapter, renderer, Providers(fail_at=1))  # materializes, one paused call
    current = args.output / "candidates" / "renders" / "current-riff-1.wav"
    manifest_path = args.output / "candidates" / "artifacts.json"
    manifest = final.driver._load(manifest_path)
    sf.write(current, sf.read(current, dtype="float32")[0] * 0.5, 48000, subtype="FLOAT")
    manifest["variants"]["current"]["renders"]["riff-1"]["file_sha256"] = hashlib.sha256(current.read_bytes()).hexdigest()
    final.driver._save(manifest_path, manifest)
    again = phase_mod.cmd_run(args, adapter, renderer, Providers())
    assert again["outcome"] == "blocked" and "historical baseline" in again["reason"]


def test_failed_control_on_other_model_keeps_c_closed(setup):
    args, adapter, renderer = setup
    flash, argon = phase_mod.RELATIVE_MODELS

    class BadControl(Providers):
        def relative(self, **kwargs):
            result = super().relative(**kwargs)
            if kwargs["model"] == argon:
                for item in result["judgment"]["comparisons"]:
                    if result["label_mapping"][item["clip"]] == "control":
                        item["gain"] = "similar"
            return result

    providers = BadControl(preferred={flash: "eq_brighter"})
    report = phase_mod.cmd_run(args, adapter, renderer, providers)
    gate = report["relative"]["gate"]
    assert len(providers.calls) == 8 and gate["run_c"] is False and "control" in gate["reason"]
    assert report["relative"]["per_model"][argon]["usable"] is False


def test_uncertain_core_axis_is_inconclusive_but_deltas_shown():
    baseline = evaluation_v3.compare_profiles_v3(REFERENCE, HISTORICAL, [])
    candidate = evaluation_v3.compare_profiles_v3(
        REFERENCE, HISTORICAL.model_copy(update={"gain": "overdrive", "brightness": "bright",
                                                 "compression": "uncertain"}), [])
    verdict = phase_mod.beats_on_di(baseline, candidate)
    assert verdict["beats"] is None and "compression" in verdict["reason"]
    deltas = phase_mod.axis_deltas(baseline, candidate)
    assert deltas["brightness"]["gap_change"] == -2 and deltas["compression"]["gap_change"] is None


def test_concurrent_runs_are_serialized_and_reuse_judgments(setup):
    import threading

    args, adapter, renderer = setup
    providers = Providers()
    entered, release, overlap = threading.Event(), threading.Event(), threading.Event()
    original, errors = renderer.render, []

    def gated(*a, **k):
        if threading.current_thread().name == "first" and not entered.is_set():
            entered.set()
            assert release.wait(10)
        if threading.current_thread().name == "second":
            overlap.set()
        return original(*a, **k)

    renderer.render = gated

    def invoke():
        try:
            phase_mod.cmd_run(args, adapter, renderer, providers)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    first, second = threading.Thread(target=invoke, name="first"), threading.Thread(target=invoke, name="second")
    first.start()
    assert entered.wait(5)
    second.start()
    overlapped = overlap.wait(0.5)
    release.set()
    first.join(10)
    second.join(10)
    assert not overlapped and not errors and len(providers.calls) == 8

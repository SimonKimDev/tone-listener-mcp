"""Offline checks for the single-clip development phase: zero provider calls, its own 12-attempt ledger."""

import json
import sys
import threading
from argparse import Namespace
from pathlib import Path

import pytest
from test_tone_accuracy_local import HISTORICAL, ChainRenderer, Providers, _final_dir
from test_workbench_recipes import adapter as recipe_adapter  # noqa: F401 - pytest fixture
from test_workbench_recipes import profile as recipe_profile
from test_workbench_refinement import _register_extras

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tone_accuracy_local as local
import tone_accuracy_single as single
import tone_copier_final as final

CONTROL = recipe_profile(gain="high-gain", brightness="balanced", mids="present", bass_level="typical",
                         bass_tightness="tight", attack="sharp", compression="moderate")
BETTER = recipe_profile(gain="overdrive", brightness="slightly-bright", mids="present", bass_level="typical",
                        bass_tightness="tight", attack="sharp", compression="moderate")
BEST = recipe_profile(gain="overdrive", brightness="bright", mids="forward", bass_level="low",
                      bass_tightness="tight", attack="sharp", compression="compressed")
UNCERTAIN = recipe_profile(gain="overdrive", brightness="bright", mids="present", bass_level="typical",
                           bass_tightness="tight", attack="sharp", compression="uncertain")


class SingleProviders(Providers):
    def relative(self, **_):
        raise AssertionError("this phase never makes relative calls")


def _providers(**profiles):
    return SingleProviders(profiles={"control": CONTROL, **profiles})


@pytest.fixture
def setup(tmp_path, monkeypatch, recipe_adapter):  # noqa: F811
    monkeypatch.setenv("TONE_EVAL_PHASE_REGISTRY", str(tmp_path / "registry"))
    _register_extras(recipe_adapter)
    renderer = ChainRenderer()
    renderer.engine_version = recipe_adapter.engine_version  # exports require the previewing engine
    final_dir = _final_dir(tmp_path / "final", recipe_adapter, renderer)
    args = Namespace(output=tmp_path / "phase", final_dir=final_dir, live=False, revise=False, reason=None)
    single.cmd_freeze(args, recipe_adapter, renderer)
    return args, recipe_adapter, renderer


def _snapshot(directory: Path) -> dict:
    return {str(path): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


def _profiled(providers) -> list[str]:
    return [source for kind, source in providers.calls if kind == "profile"]


def test_plan_is_12_with_zero_spend_fixed_candidates_and_own_registry(setup):
    args, adapter, _ = setup
    before = _snapshot(args.final_dir)
    plan = single.cmd_plan(args, adapter)
    assert plan["phase"] == "tone-accuracy-single-v1"
    assert plan["ledger"] == {"ceiling": 12, "attempted": 0, "provider_pause": None}
    assert plan["max_new_generate_requests"]["total"] == 12 == sum(step["calls"] for step in plan["steps"])
    assert [item["id"] for item in plan["candidate_set"]] == \
        ["current", "trim_plus6", "eq_brighter", "joint_plus6_brighter", "control"]
    joint = next(item for item in plan["candidate_set"] if item["id"] == "joint_plus6_brighter")
    trim = next(item for item in plan["candidate_set"] if item["id"] == "trim_plus6")
    eq = next(item for item in plan["candidate_set"] if item["id"] == "eq_brighter")
    assert {(c["node"], c["param"]) for c in joint["parameter_changes"]} == \
        {(c["node"], c["param"]) for c in trim["parameter_changes"] + eq["parameter_changes"]}
    assert plan["control_expected_direction"] == "hotter"
    assert plan["historical_baseline_gaps"]["riff-1"]["gain"] == 1
    assert plan["historical_baseline_gaps"]["riff-1"]["brightness"] == 2
    assert plan["materials"]["preparation_blocked"] is None
    registry = Path(local.registry_path())
    assert json.loads((registry / "tone-accuracy-single-v1.json").read_text())["ceiling"] == 12
    assert not (registry / "tone-accuracy-local-v1.json").exists()
    assert not (registry / "tone-copier-final-v1.json").exists()
    assert json.loads((args.output / "phase-binding.json").read_text())["phase"] == single.PHASE
    assert _snapshot(args.final_dir) == before


def test_uploads_match_baseline_rms_and_keep_native_audio(setup):
    args, _, _ = setup
    record = json.loads((args.output / "uploads" / "preparation.json").read_text())
    for sample in single.SAMPLES:
        rows = record["samples"][sample]
        assert set(rows) == set(single.UPLOAD_ROLES) and rows["baseline"]["scalar"] == 1.0
        for role in single.UPLOAD_ROLES:
            assert rows[role]["upload_rms_dbfs"] == pytest.approx(rows["baseline"]["prepared_rms_dbfs"], abs=1e-3)
            assert rows[role]["upload_peak"] <= 1.0
    manifest = json.loads((args.output / "candidates" / "artifacts.json").read_text())
    for entry in manifest["variants"].values():
        for render in entry["renders"].values():
            assert local._file_sha(args.output / "candidates" / render["wav"]) == render["file_sha256"]


def test_no_eligible_candidate_stops_after_ten_calls_without_repeats(setup):
    args, adapter, renderer = setup
    providers = _providers()
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "no_eligible_candidate" and report["ledger"]["attempted"] == 10
    assert all(report["steps"][f"repeat:{s}"]["status"] == "not_attempted" for s in single.SAMPLES)
    assert report["recommendation"] is None and not (args.output / "recommendation.json").exists()
    assert all(item["beats_both_dis"] is False for item in report["candidates"]["candidates"].values())
    exports = sorted(path.name for path in (args.output / "exports").iterdir())
    assert len(exports) == 8 and not any("control" in name for name in exports)
    assert not (args.output / "exports" / "selected.json").exists()
    assert not (args.output / "candidates" / "selected").exists()
    assert all(item["status"] == "exported" and item["validation"]
               for per in report["exports"].values() for item in per.values())
    first = {name: (args.output / "exports" / name).read_bytes() for name in exports}
    single.cmd_run(args, adapter, renderer, providers)
    assert {name: (args.output / "exports" / name).read_bytes() for name in exports} == first


def test_ranking_prefers_core_pass_then_repeat_supports(setup):
    args, adapter, renderer = setup
    providers = _providers(trim_plus6=BETTER, eq_brighter=BETTER, joint_plus6_brighter=BEST)
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["candidates"]["eligible_ranked"][0] == "joint_plus6_brighter"
    # Equal evidence: fewer parameter changes (trim 1, EQ 2) decides.
    changes = {cid: item["parameter_changes"] for cid, item in report["candidates"]["candidates"].items()}
    assert changes == {"trim_plus6": 1, "eq_brighter": 2, "joint_plus6_brighter": 3}
    assert report["candidates"]["eligible_ranked"][1:] == ["trim_plus6", "eq_brighter"]
    assert report["outcome"] == "repeat_supported_improvement" and report["ledger"]["attempted"] == 12
    assert _profiled(providers)[-2:] == ["joint_plus6_brighter-riff-1", "joint_plus6_brighter-riff-2"]
    assert report["repeat"]["core_pass_both_dis"]["repeat"] is True
    assert report["repeat"]["claims"]["perceptual_match"] is False
    advice = json.loads((args.output / "recommendation.json").read_text())
    assert advice["candidate"] == "joint_plus6_brighter" and len(advice["presets"]) == 2
    # Resume never re-sends.
    again = single.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 12 and again["outcome"] == report["outcome"]


def test_rank_key_order():
    def item(cid, core, failures, distance, changes):
        return {"id": cid, "core_pass_both": core, "known_failures_total": failures,
                "gain_brightness_mids_total": distance, "parameter_changes": changes}
    items = [item("z", False, 0, 0, 0), item("b", True, 2, 5, 3), item("a", True, 2, 5, 3),
             item("c", True, 1, 9, 9), item("d", True, 1, 4, 9), item("e", True, 1, 4, 2)]
    assert [i["id"] for i in sorted(items, key=single.rank_key)] == ["e", "d", "c", "a", "b", "z"]


def test_baseline_not_reproduced_stops_after_one_call(setup):
    args, adapter, renderer = setup
    providers = _providers(baseline=BEST)
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "baseline_not_reproduced" and len(providers.calls) == 1
    assert report["ledger"]["attempted"] == 1


def test_incomplete_baseline_core_stops(setup):
    args, adapter, renderer = setup
    providers = _providers(baseline=UNCERTAIN)
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "baseline_not_reproduced" and "incomplete" in report["reason"]
    assert len(providers.calls) == 1


def test_control_not_rejected_stops_before_candidates(setup):
    args, adapter, renderer = setup
    providers = SingleProviders(profiles={"control": HISTORICAL})
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "control_not_rejected" and len(providers.calls) == 3
    assert all(report["steps"][f"candidate:{c}:{s}"]["status"] == "not_attempted"
               for c in single.CANDIDATES for s in single.SAMPLES)


def test_wrong_direction_control_is_not_rejected(setup):
    args, adapter, renderer = setup
    cleaner = recipe_profile(gain="clean", brightness="balanced", mids="present", bass_level="typical",
                             bass_tightness="tight", attack="sharp", compression="moderate")
    providers = SingleProviders(profiles={"control": cleaner})
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "control_not_rejected"


def test_uncertain_candidate_core_is_not_eligible(setup):
    args, adapter, renderer = setup
    providers = _providers(trim_plus6=UNCERTAIN)
    report = single.cmd_run(args, adapter, renderer, providers)
    item = report["candidates"]["candidates"]["trim_plus6"]
    assert item["eligible"] is False and item["beats_both_dis"] is None
    assert item["per_di"]["riff-1"]["axis_deltas"]["gain"]["gap_change"] == -1
    assert report["outcome"] == "no_eligible_candidate"


def test_repeat_failure_is_not_supported_and_no_recommendation(setup):
    args, adapter, renderer = setup

    class Flaky(SingleProviders):
        def profile(self, clip, model):
            seen = sum(source.startswith("eq_brighter") for _, source in self.calls)
            result = super().profile(clip, model)
            if clip.source.startswith("eq_brighter") and seen >= 2:
                result["listening"] = HISTORICAL.model_dump()
            return result

    providers = Flaky(profiles={"control": CONTROL, "eq_brighter": BETTER})
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "repeat_not_supported" and report["ledger"]["attempted"] == 12
    assert not (args.output / "recommendation.json").exists() and report["recommendation"] is None


def test_first_429_pauses_phase_without_retry(setup):
    args, adapter, renderer = setup

    class Quota(SingleProviders):
        def profile(self, clip, model):
            result = super().profile(clip, model)
            if len(self.calls) == 3:
                result.update(listening=None, failure_info={"http_status": 429, "category": "quota_unspecified"})
            return result

    providers = Quota(profiles={"control": CONTROL})
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "paused" and len(providers.calls) == 3
    again = single.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 3 and again["outcome"] == "paused"


def test_reserved_but_unrecorded_tag_is_never_resent(setup):
    args, adapter, renderer = setup
    phase = single.Phase(args.output)
    phase.reserve("absolute_profile", single.PROFILE_MODEL, {"tag": "baseline:riff-1"})
    providers = _providers()
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["steps"]["baseline:riff-1"]["status"] == "attempted_unrecorded"
    assert report["outcome"] == "stopped" and providers.calls == []


def test_phase_cannot_be_reset_and_local_v1_is_unchanged(setup):
    args, _, _ = setup
    import shutil

    shutil.rmtree(args.output)
    with pytest.raises(SystemExit):
        single.Phase(args.output, create=True)
    assert local.Phase.phase_id == "tone-accuracy-local-v1" and local.Phase.ceiling == 12
    assert local.LOCAL_V1_VARIANT_IDS == ("current", "trim_minus6", "trim_minus3", "trim_plus3", "trim_plus6",
                                          "eq_brighter", "eq_darker", "control")


def test_changed_material_after_binding_blocks(setup):
    args, adapter, renderer = setup
    binding = args.output / "uploads" / "control-riff-1.upload.wav"
    binding.chmod(0o600)
    binding.write_bytes(binding.read_bytes()[:-4] + b"\0\0\0\0")
    providers = _providers()
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "blocked" and providers.calls == []


def test_changed_current_audio_blocks_before_any_call(setup):
    args, adapter, renderer = setup
    renderer.level = 0.3
    import shutil

    shutil.rmtree(args.output / "candidates")
    providers = _providers()
    report = single.cmd_run(args, adapter, renderer, providers)
    assert report["outcome"] == "blocked" and providers.calls == []


def test_concurrent_runs_are_serialized(setup):
    args, adapter, renderer = setup
    providers = _providers()
    entered, release, errors, results = threading.Event(), threading.Event(), [], []
    original = providers.profile

    def gated(clip, model):
        if threading.current_thread().name == "first" and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return original(clip, model)

    providers.profile = gated

    def invoke():
        try:
            results.append(single.cmd_run(args, adapter, renderer, providers))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    first, second = threading.Thread(target=invoke, name="first"), threading.Thread(target=invoke, name="second")
    first.start()
    assert entered.wait(5)
    second.start()
    second.join(0.5)
    assert second.is_alive()
    release.set()
    first.join(10)
    second.join(10)
    assert not errors and len(providers.calls) == 10 and [r["outcome"] for r in results] == ["no_eligible_candidate"] * 2


def test_freeze_revision_keeps_history(setup, monkeypatch):
    args, adapter, renderer = setup
    first = single.Phase(args.output).current_freeze()
    monkeypatch.setattr(single, "LIMITATIONS", (*single.LIMITATIONS, "test-only change"))
    revised = single.cmd_freeze(Namespace(**{**vars(args), "revise": True, "reason": "test"}), adapter, renderer)
    record = json.loads((single.Phase(args.output).state / "freeze.json").read_text())
    assert revised != first and record["history"][-1]["sha256"] == first
    assert record["history"][-1]["status"] == "superseded_unrun"
    assert final.SAMPLES == single.SAMPLES

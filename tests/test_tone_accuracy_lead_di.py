"""Offline checks for the lead-DI material diagnostic: zero provider calls, its own 6-attempt ledger."""

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from test_tone_accuracy_local import HISTORICAL, REFERENCE, ChainRenderer, _final_dir
from test_tone_accuracy_single import CONTROL, UNCERTAIN, _snapshot
from test_workbench_recipes import adapter as recipe_adapter  # noqa: F401 - pytest fixture
from test_workbench_recipes import profile as recipe_profile
from test_workbench_refinement import _register_extras

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tone_accuracy_lead_di as lead
import tone_accuracy_local as local
import tone_accuracy_single as single
import tone_copier_final as final

from tone_listener_mcp.workbench import evaluation_v3

PASS = REFERENCE
# gain 0, brightness 2 (a known failure), mids 1: g+b+m = 3 < the riff value 4, still not a core pass.
PARTIAL = recipe_profile(gain="overdrive", brightness="balanced", mids="present", bass_level="low",
                         bass_tightness="tight", attack="sharp", compression="compressed")


class Scripted:
    """Profiles in request order by (role, sample, occurrence); default HISTORICAL; control -> CONTROL."""

    def __init__(self, profiles=None, fail_at=None, quota_at=None):
        self.calls, self.profiles, self.fail_at, self.quota_at = [], profiles or {}, fail_at, quota_at

    def relative(self, **_):
        raise AssertionError("this phase never makes relative calls")

    def profile(self, clip, model):
        assert model == lead.PROFILE_MODEL and clip.sample_rate == 16000 and clip.samples.shape[0] == 1
        occurrence = sum(source == clip.source for source in self.calls)
        self.calls.append(clip.source)
        role = clip.source.split("-")[0]
        default = CONTROL if role == "control" else HISTORICAL
        chosen = self.profiles.get((clip.source, occurrence), self.profiles.get(clip.source, default))
        if self.quota_at == len(self.calls):
            return {"request_counts": {"generate": 1}, "listening": None,
                    "failure_info": {"http_status": 429, "category": "quota_unspecified"}}
        if self.fail_at == len(self.calls):
            return {"request_counts": {"generate": 1}, "listening": None,
                    "failure_info": {"category": "invalid_json"}}
        return {"request_counts": {"generate": 1}, "failure_info": None, "listening": chosen.model_dump()}


def _single_report(path: Path) -> str:
    grade = evaluation_v3.compare_profiles_v3(REFERENCE, HISTORICAL, [])
    report = {"phase": single.PHASE, "outcome": "repeat_not_supported",
              "baseline": {s: {"grade_v3": grade, "core_decision": final.core_decision(grade)}
                           for s in local.SAMPLES}}
    path.write_text(json.dumps(report))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path, monkeypatch, recipe_adapter):  # noqa: F811
    monkeypatch.setenv("TONE_EVAL_PHASE_REGISTRY", str(tmp_path / "registry"))
    _register_extras(recipe_adapter)
    renderer = ChainRenderer()
    renderer.engine_version = recipe_adapter.engine_version
    final_dir = _final_dir(tmp_path / "final", recipe_adapter, renderer)
    report = tmp_path / "single-report.json"
    monkeypatch.setattr(lead, "SINGLE_V1_REPORT_SHA256", _single_report(report))
    args = Namespace(output=tmp_path / "phase", final_dir=final_dir, single_report=report, live=False,
                     revise=False, reason=None)
    lead.cmd_freeze(args, recipe_adapter, renderer)
    return args, recipe_adapter, renderer


def _run(setup, providers):
    args, adapter, renderer = setup
    return lead.cmd_run(args, adapter, renderer, providers)


def test_plan_is_6_zero_spend_own_registry_and_inputs_unchanged(setup):
    args, adapter, renderer = setup
    before = {**_snapshot(args.final_dir), str(args.single_report): args.single_report.read_bytes()}
    plan = lead.cmd_plan(args, adapter)
    assert plan["phase"] == "tone-accuracy-lead-di-v1" and plan["model"] == final.PROFILE_MODEL
    assert plan["ledger"] == {"ceiling": 6, "attempted": 0, "provider_pause": None}
    assert plan["max_new_generate_requests"]["total"] == 6 == sum(step["calls"] for step in plan["steps"])
    assert [item["id"] for item in plan["variant_set"]] == ["current", "control"]
    assert set(plan["renders"]["current"]) == {"lead-1", "lead-2"}
    assert plan["control_expected_direction"] == "hotter"
    riffs = plan["riff_results_unchanged"]["tone-accuracy-single-v1"]
    assert riffs["threshold_gain_brightness_mids"] == 4
    assert all(v["core_decision"]["outcome"] == "fail" for v in riffs["per_riff"].values())
    assert set(plan["outcomes"]) >= set(lead.OUTCOMES)
    assert plan["materials"]["preparation_blocked"] is None
    registry = Path(local.registry_path())
    assert json.loads((registry / "tone-accuracy-lead-di-v1.json").read_text())["ceiling"] == 6
    for other in ("tone-copier-final-v1", "tone-accuracy-local-v1", "tone-accuracy-single-v1"):
        assert not (registry / f"{other}.json").exists()
    assert json.loads((args.output / "phase-binding.json").read_text())["phase"] == lead.PHASE
    assert {**_snapshot(args.final_dir), str(args.single_report): args.single_report.read_bytes()} == before
    assert renderer.renders >= 4 and local.SAMPLES == final.SAMPLES == ("riff-1", "riff-2")


def test_uploads_match_baseline_rms_on_lead_dis_without_rescaling_baseline(setup):
    args, _, _ = setup
    record = json.loads((args.output / "uploads" / "preparation.json").read_text())
    assert set(record["samples"]) == {"lead-1", "lead-2"}
    for rows in record["samples"].values():
        assert rows["baseline"]["scalar"] == 1.0 and rows["baseline"]["raw_file"].startswith("candidates/")
        assert rows["control"]["upload_rms_dbfs"] == pytest.approx(rows["baseline"]["prepared_rms_dbfs"], abs=1e-3)


def test_material_confound_supported_needs_pass_initially_and_on_repeat(setup):
    report = _run(setup, Scripted({"baseline-lead-1": PASS, "baseline-lead-2": PASS}))
    assert report["outcome"] == "material_confound_supported" and report["ledger"]["attempted"] == 6
    assert report["recommendation"] is None and report["adoption"] is None
    assert report["replacement_benchmark"] is False and report["tone_improvement_claim"] is False
    riffs = report["riff_results_unchanged"]["tone-accuracy-single-v1"]["per_riff"]
    assert all(v["core_decision"]["outcome"] == "fail" for v in riffs.values())
    assert not (setup[0].output / "recommendation.json").exists()


def test_partial_effect_reports_every_raw_axis_gap(setup):
    report = _run(setup, Scripted({"baseline-lead-1": PASS, "baseline-lead-2": PARTIAL}))
    assert report["outcome"] == "partial_material_effect" and report["ledger"]["attempted"] == 6
    for stage in ("initial", "repeat"):
        evidence = report["baseline"][stage]["lead-2"]
        assert set(evidence["raw_axes"]) == set(evaluation_v3.CORE_AXES)
        assert evidence["gaps"]["gain_brightness_mids"] == 3 and evidence["core_pass"] is False


def test_repeat_disagreement_is_no_material_effect(setup):
    providers = Scripted({("baseline-lead-1", 0): PASS, ("baseline-lead-1", 1): HISTORICAL,
                          "baseline-lead-2": PASS})
    report = _run(setup, providers)
    assert report["outcome"] == "no_material_effect" and report["ledger"]["attempted"] == 6
    assert report["classification"]["evidence_complete"] is True


def test_riff_like_baseline_stops_before_repeats(setup):
    report = _run(setup, Scripted())
    assert report["outcome"] == "no_material_effect" and report["ledger"]["attempted"] == 4
    assert all(report["steps"][f"baseline_repeat:{s}"]["status"] == "not_attempted" for s in lead.SAMPLES)


def test_uncertain_or_failed_baseline_fails_closed(setup):
    report = _run(setup, Scripted({"baseline-lead-1": PASS, "baseline-lead-2": UNCERTAIN}))
    assert report["outcome"] == "no_material_effect" and report["ledger"]["attempted"] == 4
    assert report["classification"]["evidence_complete"] is False
    assert report["baseline"]["initial"]["lead-2"]["evidence_complete"] is False


def test_failed_baseline_judgment_counts_and_fails_closed(setup):
    providers = Scripted({"baseline-lead-1": PASS, "baseline-lead-2": PASS}, fail_at=2)
    report = _run(setup, providers)
    assert report["outcome"] == "no_material_effect" and report["ledger"]["attempted"] == 4
    assert report["ledger"]["failed"] == 1


def test_control_not_rejected_is_invalid_control_and_stops(setup):
    providers = Scripted({"baseline-lead-1": PASS, "baseline-lead-2": PASS, "control-lead-1": HISTORICAL})
    report = _run(setup, providers)
    assert report["outcome"] == "invalid_control" and report["ledger"]["attempted"] == 3
    assert providers.calls == ["baseline-lead-1", "baseline-lead-2", "control-lead-1"]


def test_failed_control_judgment_is_invalid_control(setup):
    providers = Scripted({"baseline-lead-1": PASS, "baseline-lead-2": PASS}, fail_at=4)
    report = _run(setup, providers)
    assert report["outcome"] == "invalid_control" and report["ledger"]["attempted"] == 4


def test_first_429_pauses_without_retry_and_ceiling_holds(setup):
    providers = Scripted({"baseline-lead-1": PASS, "baseline-lead-2": PASS}, quota_at=2)
    report = _run(setup, providers)
    assert report["outcome"] == "paused" and len(providers.calls) == 2
    again = _run(setup, providers)
    assert again["outcome"] == "paused" and len(providers.calls) == 2


def test_completed_run_never_resends_and_stays_within_ceiling(setup):
    providers = Scripted({"baseline-lead-1": PASS, "baseline-lead-2": PASS})
    first = _run(setup, providers)
    second = _run(setup, providers)
    assert len(providers.calls) == 6 and first["outcome"] == second["outcome"]
    assert len(lead.Phase(setup[0].output).ledger["attempts"]) == 6


def test_reserved_but_unrecorded_tag_is_never_resent(setup):
    args, _, _ = setup
    lead.Phase(args.output).reserve("absolute_profile", lead.PROFILE_MODEL, {"tag": "baseline:lead-1"})
    providers = Scripted()
    report = _run(setup, providers)
    assert report["steps"]["baseline:lead-1"]["status"] == "attempted_unrecorded"
    assert report["outcome"] == "stopped" and providers.calls == []


def test_changed_upload_copy_blocks_before_any_call(setup):
    args, _, _ = setup
    copy = args.output / "uploads" / "control-lead-1.upload.wav"
    copy.chmod(0o600)
    copy.write_bytes(copy.read_bytes()[:-4] + b"\0\0\0\0")
    providers = Scripted()
    assert _run(setup, providers)["outcome"] == "blocked" and providers.calls == []


def test_wrong_single_report_is_refused(tmp_path, monkeypatch, recipe_adapter):  # noqa: F811
    monkeypatch.setenv("TONE_EVAL_PHASE_REGISTRY", str(tmp_path / "registry"))
    _register_extras(recipe_adapter)
    renderer = ChainRenderer()
    renderer.engine_version = recipe_adapter.engine_version
    report = tmp_path / "single-report.json"
    _single_report(report)
    args = Namespace(output=tmp_path / "phase", final_dir=_final_dir(tmp_path / "final", recipe_adapter, renderer),
                     single_report=report, live=False, revise=False, reason=None)
    with pytest.raises(SystemExit, match="not the accepted report"):
        lead.cmd_freeze(args, recipe_adapter, renderer)


def test_phase_cannot_be_reset_and_freeze_revision_keeps_history(setup, monkeypatch):
    args, adapter, renderer = setup
    first = lead.Phase(args.output).current_freeze()
    monkeypatch.setattr(lead, "LIMITATIONS", (*lead.LIMITATIONS, "test-only change"))
    revised = lead.cmd_freeze(Namespace(**{**vars(args), "revise": True, "reason": "test"}), adapter, renderer)
    record = json.loads((lead.Phase(args.output).state / "freeze.json").read_text())
    assert revised != first and record["history"][-1]["sha256"] == first
    import shutil

    shutil.rmtree(args.output)
    with pytest.raises(SystemExit):
        lead.Phase(args.output, create=True)
    assert (single.Phase.phase_id, single.Phase.ceiling) == ("tone-accuracy-single-v1", 12)
    assert (local.Phase.phase_id, local.Phase.ceiling) == ("tone-accuracy-local-v1", 12)

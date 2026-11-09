"""Offline checks for the final multi-reference runner: zero provider calls, durable 48 ceiling."""

import io
import json
import os
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
from test_workbench_recipes import profile as recipe_profile
from test_workbench_refinement import FakeJudge, RefineAdapter, RenderingFake

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tone_copier_final as final


class FakeProviders:
    def __init__(self, dominance="clear", fail_at=None):
        self.calls = []
        self.dominance, self.fail_at = dominance, fail_at
        self.judge = FakeJudge(current_gain="less", control_gain="more")

    def _result(self, kind, profile):
        self.calls.append(kind)
        result = {"request_counts": {"generate": 1}, "failure_info": None, "listening": profile.model_dump()}
        if self.fail_at == len(self.calls):
            result.update(listening=None, failure_info={"http_status": 429, "category": "quota_unspecified"})
        return result

    def reference(self, url, start, end, model):
        assert model == final.PROFILE_MODEL
        return self._result("reference", recipe_profile(gain="crunch", guitar_dominance=self.dominance))

    def profile(self, clip, model):
        return self._result("profile", recipe_profile(gain="crunch"))

    def relative(self, *, model, candidates, seed, youtube):
        assert model == final.SELECTOR_MODEL and youtube is not None
        self.calls.append("relative")
        result = self.judge(model=model, candidates=candidates, seed=seed, youtube=youtube)
        if self.fail_at == len(self.calls):
            result.update(judgment=None, failure_info={"http_status": 429, "category": "quota_unspecified"})
        return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("TONE_EVAL_PHASE_REGISTRY", str(tmp_path / "registry"))
    binary = tmp_path / "renderer"
    binary.write_text("test")
    renderer = RenderingFake(binary)
    renderer.engine_version = "test-engine"
    adapter = RefineAdapter()
    args = Namespace(output=tmp_path / "phase", case=None, live=False)
    final.cmd_freeze(args, adapter, renderer)
    return args, adapter, renderer


def test_plan_is_exactly_48_with_preregistered_intervals(setup):
    args, *_ = setup
    final.cmd_plan(args)
    plan = final.driver._load(args.output / "plan.json")
    assert plan["total_max_new_generate_requests"] == 48 == plan["ledger"]["ceiling"]
    assert plan["ledger"]["attempted"] == 0 and plan["freeze_sha256"]
    intervals = {item["case"]: item["interval"] for item in plan["cases"]}
    assert intervals["case-01"] == [19.2, 30.0]
    assert all(intervals[case] == [0.0, 30.0] for case in intervals if case != "case-01")
    assert all("pristine" not in item["provenance"] or "not pristine" in item["provenance"]
               for item in plan["cases"])


def test_full_fake_run_spends_exactly_48_and_never_resends(setup):
    args, adapter, renderer = setup
    providers = FakeProviders()
    report = final.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 48 == report["ledger"]["attempted"]
    assert report["ledger"]["by_model"] == {final.PROFILE_MODEL: 36, final.SELECTOR_MODEL: 12}
    for case in report["cases"]:
        assert case["selection"]["selected"] in case["artifacts"]["variants"]
        # Same-profile control is NOT rejected, so the evaluator is unvalidated and nothing is accepted.
        assert case["control"]["rejected"] is False
        assert case["acceptance"]["outcome"] == "inconclusive"
        assert case["acceptance"]["perceptual_match_claim"] is False
        for export in case["exports"].values():
            assert export["status"] == "exported" and export["validation"]["valid"], export
    renders = renderer.renders
    again = final.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 48 and renderer.renders == renders
    assert all(step.get("recorded_earlier") for case in again["cases"] for step in case["steps"].values())
    with pytest.raises(SystemExit, match="ceiling"):
        final.Phase(args.output).reserve("profile", final.PROFILE_MODEL, {"tag": "extra"})


class WrongControl(FakeProviders):
    """Profiles: eval riff-1, eval riff-2, control, (reference repeat), preview repeat."""

    def __init__(self, control_gain="high-gain", **kwargs):
        super().__init__(**kwargs)
        self.profiles, self.control_gain = 0, control_gain

    def profile(self, clip, model):
        self.profiles += 1
        gain = self.control_gain if self.profiles == 3 else "crunch"
        return self._result("profile", recipe_profile(gain=gain))


def test_validated_case_is_only_core_supported_never_a_perceptual_pass(setup):
    args, adapter, renderer = setup
    args.case = ["case-01"]
    case = final.cmd_run(args, adapter, renderer, WrongControl())["cases"][0]
    assert case["control"]["rejected"] is True and case["control"]["signed_steps"] == 3
    assert case["stability"]["reference_repeat"]["stable"] and case["stability"]["preview_repeat"]["stable"]
    assert case["acceptance"]["outcome"] == "core_supported_effects_not_validated"


def test_control_in_the_wrong_direction_is_not_a_rejection(setup):
    args, adapter, renderer = setup
    args.case = ["case-01"]
    case = final.cmd_run(args, adapter, renderer, WrongControl(control_gain="clean"))["cases"][0]
    assert case["control"]["rejected"] is False and case["control"]["expected_direction"] == "hotter"
    assert case["acceptance"]["outcome"] == "inconclusive"


def test_failed_second_di_is_explicit_and_keeps_observed_failure(setup):
    class Mixed(FakeProviders):
        def __init__(self):
            super().__init__()
            self.profiles = 0

        def profile(self, clip, model):
            self.profiles += 1
            if self.profiles == 2:
                self.calls.append("profile")
                return {"request_counts": {"generate": 1}, "listening": None,
                        "failure_info": {"http_status": 500, "category": "server"}}
            return self._result("profile", recipe_profile(gain="clean"))

    args, adapter, renderer = setup
    args.case = ["case-01"]
    case = final.cmd_run(args, adapter, renderer, Mixed())["cases"][0]
    per_di = case["evaluation"]["per_di"]
    assert per_di["riff-1"]["core_decision"]["outcome"] == "fail"
    assert per_di["riff-2"]["core_decision"] == {"outcome": "inconclusive", "reason": "evaluation failed",
                                                 "observed": None}
    assert case["evaluation"]["aggregate_core"] == "fail" == case["acceptance"]["outcome"]


def test_429_pauses_whole_phase_without_retry(setup):
    args, adapter, renderer = setup
    providers = FakeProviders(fail_at=3)
    report = final.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 3 == report["ledger"]["attempted"]
    assert report["ledger"]["provider_pause"]
    first = report["cases"][0]["steps"]
    assert first["select-riff-2"]["status"] == "failed"
    assert first["eval-riff-1"]["status"] == "not_attempted"
    assert report["cases"][0]["acceptance"]["outcome"] == "inconclusive"
    assert all(step["status"] == "not_attempted" for case in report["cases"][1:] for step in case["steps"].values())
    final.cmd_run(args, adapter, renderer, providers)
    assert len(providers.calls) == 3


def test_provider_exception_is_counted_recorded_and_not_retried(setup):
    class Raising(FakeProviders):
        def reference(self, url, start, end, model):
            self.calls.append("reference")
            raise RuntimeError("network")

    args, adapter, renderer = setup
    args.case = ["case-03"]
    providers = Raising()
    report = final.cmd_run(args, adapter, renderer, providers)
    assert report["cases"][0]["outcome"] == "blocked" and report["ledger"]["failed"] == 1
    final.cmd_run(args, adapter, renderer, providers)
    assert providers.calls == ["reference"]


def test_blocked_reference_spends_only_the_reference(setup):
    args, adapter, renderer = setup
    providers = FakeProviders(dominance="mixed")
    args.case = ["case-02"]
    report = final.cmd_run(args, adapter, renderer, providers)
    assert providers.calls == ["reference"]
    case = report["cases"][0]
    assert case["outcome"] == "blocked" and renderer.renders == 0
    assert sum(step["status"] == "not_attempted" for step in case["steps"].values()) == 7


def test_tampered_audio_blocks_instead_of_regrading(setup):
    args, adapter, renderer = setup
    args.case = ["case-01"]
    providers = FakeProviders(fail_at=4)
    final.cmd_run(args, adapter, renderer, providers)
    wav = next((args.output / "case-01" / "renders").glob("current-riff-1.wav"))
    wav.write_bytes(wav.read_bytes()[:-4] + b"\0\0\0\1")
    case = final.cmd_run(args, adapter, renderer, providers)["cases"][0]
    assert case["outcome"] == "blocked" and "changed" in case["reason"]


def test_phase_cannot_be_reset_by_moving_or_refreezing(setup, tmp_path, monkeypatch):
    args, adapter, renderer = setup
    with pytest.raises(SystemExit, match="another directory"):
        final.Phase(tmp_path / "elsewhere")
    monkeypatch.setattr(final, "LIMITATIONS", (*final.LIMITATIONS, "changed"))
    with pytest.raises(SystemExit, match="cannot be refrozen"):
        final.cmd_freeze(args, adapter, renderer)


def test_same_path_recreation_and_missing_state_fail_closed(setup, tmp_path):
    args, *_ = setup
    phase = final.Phase(args.output)
    entry = phase.reserve("profile", final.PROFILE_MODEL, {"tag": "spent"})
    phase.settle(entry, {"request_counts": {"generate": 1}, "failure_info": None})
    args.output.rename(tmp_path / "archived")
    for create in (False, True):
        with pytest.raises(SystemExit):
            final.Phase(args.output, create=create)
    args.output.mkdir()
    (args.output / "phase-binding.json").write_text('{"binding": "forged"}')
    with pytest.raises(SystemExit, match="replaced"):
        final.Phase(args.output)
    (args.output / "phase-binding.json").unlink()
    args.output.rmdir()
    (tmp_path / "archived").rename(args.output)
    assert len(final.Phase(args.output).ledger["attempts"]) == 1
    state = Path(final.registry_path()) / final.PHASE / "ledger.json"
    state.rename(state.with_name("gone.json"))
    with pytest.raises(SystemExit, match="missing"):
        final.Phase(args.output)


def test_concurrent_processes_never_exceed_the_ceiling(setup):
    args, *_ = setup
    script = (
        "import sys; sys.path.insert(0, 'scripts'); import tone_copier_final as f\n"
        "ok = 0\n"
        "for i in range(10):\n"
        "    try:\n"
        "        f.Phase(sys.argv[1]).reserve('profile', 'm', {'tag': f'{sys.argv[2]}-{i}'}); ok += 1\n"
        "    except SystemExit:\n"
        "        pass\n"
        "print(ok)\n"
    )
    root = Path(__file__).resolve().parents[1]
    processes = [subprocess.Popen([sys.executable, "-c", script, str(args.output), f"p{n}"], cwd=root,
                                  stdout=subprocess.PIPE, text=True, env=os.environ.copy()) for n in range(6)]
    admitted = sum(int(process.communicate(timeout=60)[0]) for process in processes)
    assert admitted == 48 == len(final.Phase(args.output).ledger["attempts"])
    tags = [item["tag"] for item in final.Phase(args.output).ledger["attempts"]]
    assert len(set(tags)) == 48


def test_revision_only_before_spend_and_history_is_kept(setup, monkeypatch):
    args, adapter, renderer = setup
    first = final.Phase(args.output).current_freeze()
    monkeypatch.setattr(final, "LIMITATIONS", (*final.LIMITATIONS, "revised"))
    args.revise, args.reason = True, "pre-spend correction"
    second = final.cmd_freeze(args, adapter, renderer)
    record = final.Phase(args.output)._freeze_record()
    assert second != first and record["current"]["supersedes"] == first
    assert record["history"][0]["sha256"] == first and record["history"][0]["status"] == "superseded_unrun"
    final.Phase(args.output).reserve("profile", final.PROFILE_MODEL, {"tag": "spent"})
    monkeypatch.setattr(final, "LIMITATIONS", (*final.LIMITATIONS, "again"))
    with pytest.raises(SystemExit, match="already has reservations"):
        final.cmd_freeze(args, adapter, renderer)


def test_legacy_zero_attempt_phase_migrates_with_history(tmp_path, monkeypatch):
    monkeypatch.setenv("TONE_EVAL_PHASE_REGISTRY", str(tmp_path / "registry"))
    output = (tmp_path / "legacy").resolve()
    final.driver.Batch(output, ceiling=48)
    final.driver._save(output / "freeze.json", {"sha256": "a" * 64, "engine": "x"})
    (tmp_path / "registry").mkdir()
    final.driver._save(tmp_path / "registry" / f"{final.PHASE}.json",
                       {"phase": final.PHASE, "output": str(output), "ceiling": 48})
    with pytest.raises(SystemExit, match="migrate-legacy"):
        final.Phase(output)
    result = final.migrate_legacy(output)
    assert result["legacy_freeze_sha256"] == "a" * 64
    phase = final.Phase(output)
    assert phase.ledger["attempts"] == [] and phase.current_freeze() is None
    assert phase._freeze_record()["history"][0]["status"] == "superseded_unrun"
    assert (output / "history" / "legacy-01" / "freeze.json").exists()


def test_legacy_migration_refuses_spent_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("TONE_EVAL_PHASE_REGISTRY", str(tmp_path / "registry"))
    output = (tmp_path / "legacy").resolve()
    batch = final.driver.Batch(output, ceiling=48)
    batch.reserve("profile", "m", {"tag": "spent"})
    final.driver._save(output / "freeze.json", {"sha256": "a" * 64})
    (tmp_path / "registry").mkdir()
    final.driver._save(tmp_path / "registry" / f"{final.PHASE}.json",
                       {"phase": final.PHASE, "output": str(output), "ceiling": 48})
    with pytest.raises(SystemExit, match="refused"):
        final.migrate_legacy(output)


def test_live_run_requires_flag(setup):
    args, adapter, renderer = setup
    with pytest.raises(SystemExit, match="--live"):
        final.cmd_run(args, adapter, renderer)


def _rewrite_archive(data, edit=None):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        payload = json.loads(archive.read("preset.json"))
    if edit:
        edit(payload)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(zipfile.ZipInfo("preset.json", date_time=(2001, 1, 1, 0, 0, 0)),
                         json.dumps(payload, indent=1))
    return output.getvalue()


def _first_parameter(payload):
    for node in payload["preset"]["scenes"][0]["graph"]["nodes"]:
        if node.get("params"):
            name = next(iter(node["params"]))
            node["params"][name] += 0.137
            return
    raise AssertionError("no parameterized node to change")


@pytest.mark.parametrize("edit", [None, _first_parameter])
def test_existing_export_reused_only_when_content_is_equivalent(setup, monkeypatch, edit):
    args, adapter, renderer = setup
    args.case = ["case-01"]
    case = final.cmd_run(args, adapter, renderer, FakeProviders())["cases"][0]
    directory = args.output / "case-01"
    originals = {sample: (directory / entry["path"]).read_bytes() for sample, entry in case["exports"].items()}
    original_export = adapter.export_archive
    monkeypatch.setattr(adapter, "export_archive",
                        lambda project, snapshot: _rewrite_archive(original_export(project, snapshot), edit))
    manifest = json.loads((directory / "artifacts.json").read_text())
    exports = final.export_selected(directory, manifest, case["selection"]["selected"], adapter)
    expected = "exported" if edit is None else "conflict"
    assert {entry["status"] for entry in exports.values()} == {expected}, exports
    for sample, entry in exports.items():
        path = directory / "selected" / f"{case['selection']['selected']}-{sample}.soundshed.preset"
        assert path.read_bytes() == originals[sample]
        if edit is None:
            assert entry["sha256"] == final.driver._sha(originals[sample]) and entry["validation"]["valid"]

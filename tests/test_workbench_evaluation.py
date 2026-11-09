from tests.test_workbench_recipes import profile
from tone_listener_mcp.workbench.evaluation import compare_profiles


def test_frozen_rubric_compares_typed_axes_not_prose_or_confidence():
    assert compare_profiles(profile(), profile())["outcome"] == "pass"
    different = profile(gain="high-gain", brightness="dark", mids="scooped")
    verdict = compare_profiles(profile(), different)
    assert verdict["outcome"] == "fail"
    assert verdict["axis_differences"]["gain"]["difference"] == 5


def test_full_mix_or_unknown_axes_never_pass():
    assert compare_profiles(profile(), profile(guitar_dominance="mixed"))["outcome"] == "inconclusive"
    assert compare_profiles(profile(), profile(gain="uncertain"))["outcome"] == "inconclusive"


def test_prominent_effect_contradictions_fail():
    target = profile(delay={"presence": "present", "prominence": "prominent"})
    assert compare_profiles(target, profile())["outcome"] == "fail"


def test_possible_effect_is_inconclusive_not_absent():
    target = profile(reverb={"presence": "possible", "prominence": "subtle"})
    assert compare_profiles(target, profile())["outcome"] == "inconclusive"


def test_known_prominent_modulation_families_cannot_be_interchanged():
    effect = {"presence": "present", "prominence": "prominent"}
    assert compare_profiles(
        profile(modulation=effect, modulation_type="chorus"),
        profile(modulation=effect, modulation_type="phaser"),
    )["outcome"] == "fail"


def test_benchmark_pins_grader_and_driver_and_refuses_interrupted_cases(tmp_path, monkeypatch):
    import hashlib
    import inspect
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import pytest

    from scripts import eval_workbench_references as benchmark

    class Renderer:
        _worker_sha256 = "worker"
        _plugin_sha256 = "plugin"
        engine_version = "test"
        def __init__(self, adapter):
            self.canary_results = {}

        def initialize(self):
            pass

    monkeypatch.setattr(benchmark, "load_default_credentials", lambda: None)
    monkeypatch.setattr(benchmark, "NativeRenderer", Renderer)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    (tmp_path / "report.json").write_text(json.dumps({"cases": [
        {"id": "case-01", "split": "development", "outcome": "running"},
    ]}))
    with pytest.raises(ValueError, match="Interrupted cases remain unfinished"):
        benchmark.run(SimpleNamespace(
            output=tmp_path, model="offline-test", start=0, end=30, split="development",
        ))
    frozen = json.loads((tmp_path / "frozen-run.json").read_text())
    assert frozen["grader_implementation_sha256"] == hashlib.sha256(
        Path(inspect.getfile(compare_profiles)).read_bytes()
    ).hexdigest()
    assert frozen["benchmark_implementation_sha256"] == hashlib.sha256(
        Path(benchmark.__file__).read_bytes()
    ).hexdigest()


def test_rendered_stereo_is_prepared_as_audio_not_a_scalar_average():
    import numpy as np

    from scripts.eval_workbench_references import _preview_clip
    from tone_listener_mcp.audio_io import prepare_upload

    mono = np.sin(2 * np.pi * 440 * np.arange(44_100) / 44_100) * 0.1
    clip = _preview_clip(np.column_stack((mono, mono)), 44_100)
    assert clip.samples.shape == (2, 44_100)
    prepared = prepare_upload(clip)
    assert prepared.samples.shape == (1, 16_000)
    assert np.max(np.abs(prepared.samples)) > 0.09


def test_quota_diagnostics_allowlist_scope_and_honor_retry_after_without_payload():
    import json

    import httpx
    from google.genai import errors

    from tone_listener_mcp.listener import _profile_failure_info

    for quota, expected in (
        ("GenerateRequestsPerDayPerProjectPerModel", "daily_quota"),
        ("GenerateTokensPerMinutePerProjectPerModel", "rate_window"),
    ):
        error = errors.ClientError(429, {"error": {
            "message": "secret-key consumer projects/12345",
            "details": [
                {"violations": [{"quotaId": quota, "consumer": "projects/12345"}]},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7.5s"},
            ],
        }}, httpx.Response(429, headers={"Retry-After": "9"}))
        info = _profile_failure_info(error)
        assert info["category"] == expected
        assert info["retry_after_seconds"] == 9
        assert "secret-key" not in json.dumps(info)
        assert "12345" not in json.dumps(info)


def test_bounded_benchmark_stops_after_provider_quota_without_repeating(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import eval_workbench_references as benchmark

    monkeypatch.setattr(benchmark, "load_default_credentials", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(benchmark, "NativeRenderer", lambda adapter: SimpleNamespace(
        initialize=lambda: None, _worker_sha256="worker", _plugin_sha256="plugin",
        engine_version="test", canary_results={},
    ))
    calls = []

    def unavailable(url, start, end):
        calls.append(url)
        return {
            "model": "offline-test", "listening": None, "listening_unavailable": "Quota",
            "request_counts": {"generate": 1},
            "failure_info": {"http_status": 429, "category": "daily_quota",
                             "retry_after_seconds": 60, "retry_after_invalid": False},
        }

    monkeypatch.setattr(benchmark, "listen_youtube_tone_profile", unavailable)
    report = benchmark.run(SimpleNamespace(
        output=tmp_path, model="offline-test", start=19.2, end=30, split="development",
    ))
    assert len(calls) == 1
    assert report["cases"][0]["outcome"] == "blocked"
    assert report["cases"][1]["outcome"] == "pending"
    assert report["provider_pause"]["retry_after_seconds"] == 60
    assert report["generate_content_calls"] == 1


def test_bounded_selector_cannot_bypass_held_out_prerequisites(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from scripts import eval_workbench_references as benchmark
    monkeypatch.setattr(benchmark, "load_default_credentials", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    with pytest.raises(ValueError, match="development-only"):
        benchmark.run(SimpleNamespace(
            output=tmp_path, model="offline-test", start=0, end=30,
            split="held-out", case=["case-05"],
        ))


def test_case01_budget_and_preview_pause_are_bounded_offline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from scripts import eval_workbench_references as benchmark
    from tests.test_workbench_api import FakeAdapter
    from tone_listener_mcp.workbench.models import PreviewSnapshot

    class Renderer:
        _worker_sha256 = "worker"
        _plugin_sha256 = "plugin"
        engine_version = "test-engine"

        def __init__(self, adapter):
            self.adapter = adapter
            self.canary_results = {}

        def initialize(self):
            pass

        def render(self, project, sample, directory):
            snapshot = PreviewSnapshot(
                project_id=project.id, project_revision=project.revision,
                chain_revision=project.chain.revision,
                chain_sha256=self.adapter.chain_sha256(project.chain),
                sample_name=sample, source_sample_sha256="0" * 64,
                engine_version=self.engine_version, sample_rate=44100,
                dsp_conditions={"state_canary_version": 1},
                audio_sha256="1" * 64, duration_seconds=1, rms=0.1, peak=0.2,
                created_at="2026-01-01T00:00:00Z",
            )
            return SimpleNamespace(snapshot=snapshot, samples=np.ones((44100, 2)) * 0.1)

    monkeypatch.setattr(benchmark, "load_default_credentials", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(benchmark, "SoundshedAdapter", FakeAdapter)
    monkeypatch.setattr(benchmark, "NativeRenderer", Renderer)
    for pause_at in (None, 1, 2):
        references = []
        audio_calls = []
        pause_state = {"at": pause_at}

        def result(value):
            return {"model": "offline-test", "listening": value.model_dump(),
                    "listening_unavailable": None, "request_counts": {"generate": 1},
                    "upload_gain_db": 0.0}

        def reference(url, start, end, references=references):
            references.append(url)
            return {**result(profile()), "source": {
                "url": url, "start_seconds": start, "end_seconds": end,
            }}

        def preview(clip, audio_calls=audio_calls, pause_state=pause_state):
            audio_calls.append(clip)
            if len(audio_calls) == pause_state["at"]:
                return {"model": "offline-test", "listening": None,
                        "listening_unavailable": "Quota", "upload_gain_db": 0.0,
                        "request_counts": {"generate": 1},
                        "failure_info": {"http_status": 429, "category": "rate_window",
                                         "retry_after_seconds": 60}}
            return result(profile(gain="high-gain") if len(audio_calls) == 4 else profile())

        monkeypatch.setattr(benchmark, "listen_youtube_tone_profile", reference)
        monkeypatch.setattr(benchmark, "listen_tone_profile", preview)
        report = benchmark.run(SimpleNamespace(
            output=tmp_path / str(pause_at), model="offline-test", start=19.2, end=30,
            split="development", case=["case-01"],
        ))
        assert len(references) == 2
        assert set(references) == {"https://www.youtube.com/watch?v=401iS5McuoU"}
        assert report["generate_content_calls"] <= 6
        assert all(case["outcome"] == "not_requested" for case in report["cases"][1:])
        if pause_at is None:
            assert report["generate_content_calls"] == 6
            assert report["cases"][0]["outcome"] == "pass"
        else:
            assert len(audio_calls) == pause_at
            assert report["cases"][0]["outcome"] == "blocked"
            assert report["provider_pause"]["retry_after_seconds"] == 60
    prior = tmp_path / "None"
    for name in ("case-02", "case-05"):
        folder = prior / name
        folder.mkdir()
        (folder / "reference.json").write_text("unused cache must not be read")
    pause_state["at"] = None
    references.clear()
    audio_calls.clear()
    reused = benchmark.run(SimpleNamespace(
        output=tmp_path / "reused", model="offline-test", start=19.2, end=30,
        split="development", case=["case-01"], reuse_references=prior,
    ))
    assert not references
    assert reused["generate_content_calls"] == 4
    assert reused["reused_reference_generate_calls"] == 2
    assert all(key.startswith("case-01/") for key in reused["reused_reference_results"])

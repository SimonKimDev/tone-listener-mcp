"""Offline regressions for analysis reuse, in-flight coalescing and confirmed request bounds.

Every provider entry point is replaced by a counting fake; no test here contacts Gemini.
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from test_workbench_api import HOST, ORIGIN, FakeAdapter, FakeRenderer, _client, _profile

from tone_listener_mcp.workbench import analysis_cache as cache_module
from tone_listener_mcp.workbench import api as api_module
from tone_listener_mcp.workbench.analysis_cache import AnalysisCache, make_key
from tone_listener_mcp.workbench.jobs import JobManager
from tone_listener_mcp.workbench.models import Interval
from tone_listener_mcp.workbench.renderer import RenderError
from tone_listener_mcp.workbench.storage import LocalStorage

VIDEO = "https://www.youtube.com/watch?v=401iS5McuoU"


def _success(model="offline-test", gain="clean"):
    profile = _profile()
    profile["gain"] = gain
    return {"model": model, "listening": profile, "listening_unavailable": None,
            "request_counts": {"generate": 1}}


def _quota():
    return {"model": "offline-test", "listening": None, "listening_unavailable": "Quota",
            "request_counts": {"generate": 1},
            "failure_info": {"http_status": 429, "category": "daily_quota", "retry_after_seconds": 60}}


def _wait(client, headers, job, timeout=5.0):
    deadline = time.monotonic() + timeout
    while job["status"] in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.01)
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
    return job


def _youtube(url=VIDEO, intervals=((0, 5),), **extra):
    return {"kind": "youtube", "youtube_url": url,
            "intervals": [{"start_seconds": a, "end_seconds": b} for a, b in intervals], **extra}


def _analyze(client, headers, body, **extra):
    response = client.post("/api/jobs/analysis", json={**body, "external_consent": True, **extra},
                           headers=headers)
    assert response.status_code == 202, response.text
    return _wait(client, headers, response.json())


def _counting_youtube(monkeypatch, result=_success):
    calls = []

    def fake(url, start, end):
        calls.append((url, start, end))
        return result()

    monkeypatch.setattr(api_module, "listen_youtube_tone_profile", fake)
    return calls


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setenv("TONE_LISTENER_MODEL", "offline-model-a")

    def no_provider(*args, **kwargs):  # pragma: no cover - a call here is a test failure
        raise AssertionError("unexpected provider call")

    monkeypatch.setattr(api_module, "listen_youtube_tone_profile", no_provider)
    monkeypatch.setattr(api_module, "listen_tone_profile", no_provider)


def test_repeated_identical_youtube_input_makes_zero_new_calls(tmp_path, monkeypatch):
    calls = _counting_youtube(monkeypatch)
    client, headers, _ = _client(tmp_path, monkeypatch)
    body = _youtube(intervals=((0, 5), (5, 10)))

    before = client.post("/api/analysis/preflight", json=body, headers=headers).json()
    assert (before["reused"], before["shared_in_flight"], before["max_new_requests"]) == (0, 0, 2)
    assert before["provider_contacted"] is False
    first = _analyze(client, headers, body, max_new_requests=2)
    assert first["status"] == "completed" and len(calls) == 2
    assert all(segment["reuse"] is None and segment["provider_attempted"]
               for segment in first["result"]["project"]["analysis"])

    after = client.post("/api/analysis/preflight", json=body, headers=headers).json()
    assert (after["reused"], after["max_new_requests"]) == (2, 0)
    assert [plan["status"] for plan in after["intervals"]] == ["cache", "cache"]
    second = _analyze(client, headers, body, max_new_requests=0)
    assert second["status"] == "completed"
    assert len(calls) == 2
    for fresh, reused in zip(first["result"]["project"]["analysis"],
                             second["result"]["project"]["analysis"], strict=True):
        assert reused["reuse"]["kind"] == "cache"
        assert reused["provider_attempted"] is False
        assert reused["interval"] == fresh["interval"]
        assert reused["profile"] == fresh["profile"]
        assert reused["model"] == "offline-test"


def test_partial_reuse_only_calls_for_the_changed_interval(tmp_path, monkeypatch):
    calls = _counting_youtube(monkeypatch)
    client, headers, _ = _client(tmp_path, monkeypatch)
    _analyze(client, headers, _youtube(intervals=((0, 5), (5, 10))))
    changed = _youtube(intervals=((0, 5), (5, 10.5)))
    plan = client.post("/api/analysis/preflight", json=changed, headers=headers).json()
    assert [item["status"] for item in plan["intervals"]] == ["cache", "new"]
    job = _analyze(client, headers, changed, max_new_requests=1)
    assert calls[-1][1:] == (5, 10.5) and len(calls) == 3
    segments = job["result"]["project"]["analysis"]
    assert segments[0]["reuse"]["kind"] == "cache"
    assert segments[1]["reuse"] is None and segments[1]["interval"]["end_seconds"] == 10.5


def test_normalized_youtube_aliases_reuse(tmp_path, monkeypatch):
    calls = _counting_youtube(monkeypatch)
    client, headers, _ = _client(tmp_path, monkeypatch)
    _analyze(client, headers, _youtube())
    for alias in ("https://youtu.be/401iS5McuoU", "https://www.youtube.com/watch?v=401iS5McuoU&t=42",
                  "https://m.youtube.com/watch?v=401iS5McuoU"):
        plan = client.post("/api/analysis/preflight", json=_youtube(alias), headers=headers)
        assert plan.status_code == 200, plan.text
        assert plan.json()["max_new_requests"] == 0, alias
        assert _analyze(client, headers, _youtube(alias), max_new_requests=0)["status"] == "completed"
    assert len(calls) == 1


def test_preflight_rejects_any_interval_over_the_configured_limit(tmp_path, monkeypatch):
    calls = _counting_youtube(monkeypatch)
    monkeypatch.setenv("TONE_LISTENER_MAX_SECONDS", "10")
    client, headers, _ = _client(tmp_path, monkeypatch)
    body = _youtube(intervals=((0, 5), (5, 20)))

    preflight = client.post("/api/analysis/preflight", json=body, headers=headers)
    assert preflight.status_code == 422, preflight.text
    submitted = client.post(
        "/api/jobs/analysis",
        json={**body, "external_consent": True},
        headers=headers,
    )
    assert submitted.status_code == 422, submitted.text
    assert calls == []


def test_cancelled_queued_audio_analysis_releases_uploaded_file(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    manager = JobManager(workers=1, max_active=3)

    def occupy_worker(progress, cancelled):
        started.set()
        release.wait(5)
        return {}

    client, headers, storage = _client(tmp_path, monkeypatch, job_manager=manager)
    blocker = manager.submit("analysis", occupy_worker)
    try:
        assert started.wait(2)
        wav = io.BytesIO()
        sf.write(wav, np.zeros(44_100, dtype=np.float32), 44_100, format="WAV")
        uploaded = client.post(
            "/api/uploads",
            files={"file": ("queued.wav", wav.getvalue(), "audio/wav")},
            headers=headers,
        ).json()
        path = storage.get_upload(uploaded["upload_id"]).path
        analysis = {
            "kind": "audio_file",
            "upload_id": uploaded["upload_id"],
            "intervals": uploaded["intervals"],
        }
        assert client.post("/api/analysis/preflight", json=analysis, headers=headers).status_code == 200
        queued = client.post(
            "/api/jobs/analysis",
            json={**analysis, "external_consent": True},
            headers=headers,
        )
        assert queued.status_code == 202, queued.text
        cancelled = client.post(f"/api/jobs/{queued.json()['id']}/cancel", headers=headers)
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert storage.get_upload(uploaded["upload_id"]) is None
        assert not path.exists()
    finally:
        release.set()
        manager.close()
    assert _wait(client, headers, blocker.model_dump(mode="json"))["status"] == "completed"


def test_renderer_initialization_failure_releases_uploaded_audio(tmp_path, monkeypatch):
    class FailingRenderer(FakeRenderer):
        def initialize(self, cancel_event=None):
            raise RenderError("Renderer initialization failed.")

    client, headers, storage = _client(tmp_path, monkeypatch, renderer=FailingRenderer(tmp_path / "renderer"))
    wav = io.BytesIO()
    sf.write(wav, np.zeros(44_100, dtype=np.float32), 44_100, format="WAV")
    uploaded = client.post(
        "/api/uploads",
        files={"file": ("failed.wav", wav.getvalue(), "audio/wav")},
        headers=headers,
    ).json()
    path = storage.get_upload(uploaded["upload_id"]).path
    job = client.post(
        "/api/jobs/analysis",
        json={
            "kind": "audio_file",
            "upload_id": uploaded["upload_id"],
            "intervals": uploaded["intervals"],
            "external_consent": True,
        },
        headers=headers,
    ).json()
    completed = _wait(client, headers, job)
    assert completed["status"] == "failed"
    assert storage.get_upload(uploaded["upload_id"]) is None
    assert not path.exists()


def _upload(client, headers, name, signal):
    wav = io.BytesIO()
    sf.write(wav, signal, 44_100, format="WAV")
    response = client.post("/api/uploads", files={"file": (name, wav.getvalue(), "audio/wav")},
                           headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["upload_id"]


def test_same_file_content_under_new_name_reuses_and_changed_audio_misses(tmp_path, monkeypatch):
    calls = []

    def fake(clip):
        calls.append(clip.start_seconds)
        return _success()

    monkeypatch.setattr(api_module, "listen_tone_profile", fake)
    client, headers, _ = _client(tmp_path, monkeypatch)
    signal = (0.1 * np.sin(np.linspace(0, 2000, 44_100 * 2))).astype(np.float32)
    interval = [{"start_seconds": 0, "end_seconds": 2}]

    def body(upload_id):
        return {"kind": "audio_file", "upload_id": upload_id, "intervals": interval}

    first = _analyze(client, headers, body(_upload(client, headers, "first.wav", signal)))
    assert first["status"] == "completed" and calls == [0]
    alias = _upload(client, headers, "renamed copy.wav", signal)
    assert client.post("/api/analysis/preflight", json=body(alias), headers=headers).json()[
        "max_new_requests"] == 0
    reused = _analyze(client, headers, body(alias), max_new_requests=0)
    assert reused["result"]["project"]["analysis"][0]["reuse"]["kind"] == "cache"
    assert reused["result"]["project"]["analysis"][0]["metrics"] is not None
    assert calls == [0]

    changed = _upload(client, headers, "first.wav", signal * 0.5)
    assert client.post("/api/analysis/preflight", json=body(changed), headers=headers).json()[
        "max_new_requests"] == 1
    _analyze(client, headers, body(changed))
    assert calls == [0, 0]


def test_changed_bounds_model_or_semantics_miss(tmp_path, monkeypatch):
    _counting_youtube(monkeypatch)
    client, headers, _ = _client(tmp_path, monkeypatch)
    _analyze(client, headers, _youtube())

    def new_count(body):
        return client.post("/api/analysis/preflight", json=body, headers=headers).json()["max_new_requests"]

    assert new_count(_youtube()) == 0
    assert new_count(_youtube(intervals=((0, 5.001),))) == 1
    assert new_count(_youtube(intervals=((0.001, 5),))) == 1
    monkeypatch.setenv("TONE_LISTENER_MODEL", "offline-model-b")
    assert new_count(_youtube()) == 1
    monkeypatch.setenv("TONE_LISTENER_MODEL", "offline-model-a")
    assert new_count(_youtube()) == 0
    monkeypatch.setattr(cache_module, "analysis_semantics_sha256", lambda: "changed-prompt-or-schema")
    assert new_count(_youtube()) == 1


def test_semantics_fingerprint_covers_prompt_schema_and_preparation(monkeypatch):
    cache_module.analysis_semantics_sha256.cache_clear()
    original = cache_module.analysis_semantics_sha256()
    real_read = Path.read_bytes

    def edited(path):
        data = real_read(path)
        return data + b"# edited prompt" if path.name == "listener.py" else data

    monkeypatch.setattr(Path, "read_bytes", edited)
    cache_module.analysis_semantics_sha256.cache_clear()
    try:
        assert cache_module.analysis_semantics_sha256() != original
    finally:
        cache_module.analysis_semantics_sha256.cache_clear()


def test_failures_are_never_cached_and_refresh_is_explicit(tmp_path, monkeypatch):
    responses = [_quota, _success, lambda: _success(gain="high-gain")]
    calls = []

    def fake(url, start, end):
        calls.append(start)
        return responses[len(calls) - 1]()

    monkeypatch.setattr(api_module, "listen_youtube_tone_profile", fake)
    client, headers, storage = _client(tmp_path, monkeypatch)
    failed = _analyze(client, headers, _youtube())
    assert failed["status"] == "partial"
    assert not list(storage.analysis.glob("*.json"))
    assert client.post("/api/analysis/preflight", json=_youtube(), headers=headers).json()[
        "max_new_requests"] == 1
    _analyze(client, headers, _youtube())
    assert len(calls) == 2
    refresh = client.post("/api/analysis/preflight", json=_youtube(refresh=True), headers=headers).json()
    assert refresh["refresh"] is True and refresh["max_new_requests"] == 1
    refreshed = _analyze(client, headers, _youtube(refresh=True), max_new_requests=1)
    assert len(calls) == 3
    assert refreshed["result"]["project"]["analysis"][0]["reuse"] is None
    reused = _analyze(client, headers, _youtube(), max_new_requests=0)
    assert reused["result"]["project"]["analysis"][0]["profile"]["gain"] == "high-gain"
    assert len(calls) == 3


def test_cached_intervals_still_reuse_after_a_provider_pause(tmp_path, monkeypatch):
    calls = _counting_youtube(monkeypatch)
    client, headers, _ = _client(tmp_path, monkeypatch)
    _analyze(client, headers, _youtube(intervals=((10, 15),)))
    monkeypatch.setattr(api_module, "listen_youtube_tone_profile",
                        lambda url, start, end: calls.append(start) or _quota())
    job = _analyze(client, headers, _youtube(intervals=((0, 5), (5, 10), (10, 15))))
    segments = job["result"]["project"]["analysis"]
    assert calls[1:] == [0]
    assert segments[0]["failure_info"]["http_status"] == 429
    assert segments[1]["provider_attempted"] is False and segments[1]["unavailable"].startswith(
        "Not attempted: provider paused")
    assert segments[2]["reuse"]["kind"] == "cache" and segments[2]["profile"] is not None


def test_stale_zero_call_confirmation_is_rejected_at_dispatch(tmp_path, monkeypatch):
    calls = _counting_youtube(monkeypatch)
    client, headers, storage = _client(tmp_path, monkeypatch)
    _analyze(client, headers, _youtube())
    plan = client.post("/api/analysis/preflight", json=_youtube(), headers=headers).json()
    assert plan["max_new_requests"] == 0
    for entry in storage.analysis.glob("*.json"):
        entry.unlink()
    stale = client.post("/api/jobs/analysis", json={**_youtube(), "external_consent": True,
                                                    "max_new_requests": 0}, headers=headers)
    assert stale.status_code == 409
    assert "confirm again" in stale.json()["detail"]
    assert len(calls) == 1


def test_entry_vanishing_after_dispatch_never_exceeds_confirmed_bound(tmp_path, monkeypatch):
    calls = _counting_youtube(monkeypatch)
    client, headers, _ = _client(tmp_path, monkeypatch)
    _analyze(client, headers, _youtube())
    real_get = AnalysisCache.get
    lookups = []

    def vanishing(self, key):
        lookups.append(key)
        return real_get(self, key) if len(lookups) == 1 else None  # present only at dispatch

    monkeypatch.setattr(AnalysisCache, "get", vanishing)
    job = _analyze(client, headers, _youtube(), max_new_requests=0)
    assert len(calls) == 1
    segment = job["result"]["project"]["analysis"][0]
    assert job["status"] == "partial"
    assert segment["provider_attempted"] is False and "confirm again" in segment["unavailable"]


def _concurrent_client(tmp_path):
    storage = LocalStorage(tmp_path / "workbench")
    binary = tmp_path / "renderer"
    binary.write_text("test")
    app = api_module.create_app(
        storage=storage, adapter=FakeAdapter(), renderer=FakeRenderer(binary), allowed_hosts={HOST},
        job_manager=JobManager(workers=3, max_active=3), allow_test_client=True,
    )
    client = TestClient(app)
    token = client.get("/api/session", headers={"Host": HOST, "Origin": ORIGIN}).json()["token"]
    return client, {"Host": HOST, "Origin": ORIGIN, "X-Workbench-Token": token}, storage


@pytest.mark.parametrize("outcome", [_success, _quota])
def test_concurrent_identical_jobs_make_one_provider_attempt(tmp_path, monkeypatch, outcome):
    release = threading.Event()
    calls = []

    def fake(url, start, end):
        calls.append(start)
        assert release.wait(5)
        return outcome()

    monkeypatch.setattr(api_module, "listen_youtube_tone_profile", fake)
    client, headers, storage = _concurrent_client(tmp_path)
    body = {**_youtube(), "external_consent": True}
    leader = client.post("/api/jobs/analysis", json=body, headers=headers).json()
    deadline = time.monotonic() + 2
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    plan = client.post("/api/analysis/preflight", json=_youtube(), headers=headers).json()
    assert plan["shared_in_flight"] == 1 and plan["max_new_requests"] == 0
    follower = client.post("/api/jobs/analysis", json={**body, "max_new_requests": 0}, headers=headers).json()
    time.sleep(0.1)
    release.set()
    leader, follower = _wait(client, headers, leader), _wait(client, headers, follower)
    assert len(calls) == 1
    lead_segment = leader["result"]["project"]["analysis"][0]
    shared_segment = follower["result"]["project"]["analysis"][0]
    assert lead_segment["provider_attempted"] is True and lead_segment["reuse"] is None
    assert shared_segment["provider_attempted"] is False
    assert shared_segment["reuse"]["kind"] == "shared_in_flight"
    if outcome is _quota:
        assert shared_segment["failure_info"]["http_status"] == 429 and shared_segment["profile"] is None
        assert not list(storage.analysis.glob("*.json"))
    else:
        assert shared_segment["profile"] == lead_segment["profile"]
        assert len(list(storage.analysis.glob("*.json"))) == 1


def test_cancelling_one_consumer_keeps_shared_work_for_the_other(tmp_path, monkeypatch):
    release = threading.Event()
    calls = []

    def fake(url, start, end):
        calls.append(start)
        assert release.wait(5)
        return _success()

    monkeypatch.setattr(api_module, "listen_youtube_tone_profile", fake)
    client, headers, _ = _concurrent_client(tmp_path)
    body = {**_youtube(), "external_consent": True}
    leader = client.post("/api/jobs/analysis", json=body, headers=headers).json()
    while not calls:
        time.sleep(0.01)
    follower = client.post("/api/jobs/analysis", json=body, headers=headers).json()
    time.sleep(0.1)
    client.post(f"/api/jobs/{follower['id']}/cancel", headers=headers)
    follower = _wait(client, headers, follower)
    assert follower["status"] == "cancelled"
    release.set()
    leader = _wait(client, headers, leader)
    assert leader["status"] == "completed" and leader["result"]["project"]["analysis"][0]["profile"]
    later = _analyze(client, headers, _youtube(), max_new_requests=0)
    assert later["result"]["project"]["analysis"][0]["reuse"]["kind"] == "cache"
    assert len(calls) == 1


def test_cancelled_leader_still_finishes_shared_request_for_waiter(tmp_path):
    cache = AnalysisCache(tmp_path / "analysis")
    key = make_key("youtube", "401iS5McuoU", Interval(start_seconds=0, end_seconds=5), "m")
    started, release = threading.Event(), threading.Event()
    lead_cancel, wait_cancel = threading.Event(), threading.Event()
    calls, outcomes = [], {}

    def request():
        calls.append(1)
        started.set()
        release.wait(5)
        return {"value": 1}

    leader = threading.Thread(target=lambda: outcomes.setdefault(
        "leader", cache.run_shared(key, request, lead_cancel, may_start=True)))
    leader.start()
    started.wait(5)
    waiter = threading.Thread(target=lambda: outcomes.setdefault(
        "waiter", cache.run_shared(key, request, wait_cancel, may_start=False)))
    waiter.start()
    lead_cancel.set()
    release.set()
    leader.join(5)
    waiter.join(5)
    assert calls == [1]
    assert outcomes["leader"] == ("fresh", {"value": 1})
    assert outcomes["waiter"] == ("shared", {"value": 1})
    assert cache.run_shared(key, request, wait_cancel, may_start=False) == ("blocked", None)


def _leader_finishes_after_waiter_miss(monkeypatch, release):
    """Interleaving: the waiter's job-level cache lookup misses while the leader is in flight,
    then the leader stores its result and leaves the in-flight map before the waiter reaches
    run_shared."""
    original_get = AnalysisCache.get
    state = {"armed": False, "misses": 0}

    def interleaving_get(self, key):
        found = original_get(self, key)
        if state["armed"] and found is None and self.in_flight(key):
            state["misses"] += 1
            if state["misses"] == 2:  # 1 = dispatch recheck, 2 = waiter job lookup
                release.set()
                deadline = time.monotonic() + 5
                while self.in_flight(key) and time.monotonic() < deadline:
                    time.sleep(0.005)
        return found

    monkeypatch.setattr(AnalysisCache, "get", interleaving_get)
    return state


@pytest.mark.parametrize("waiter_budget", [0, 1])
def test_waiter_reuses_result_cached_between_lookup_and_single_flight(tmp_path, monkeypatch, waiter_budget):
    started, release, calls = threading.Event(), threading.Event(), []

    def fake(url, start, end):
        calls.append(start)
        started.set()
        assert release.wait(5)
        return _success()

    monkeypatch.setattr(api_module, "listen_youtube_tone_profile", fake)
    state = _leader_finishes_after_waiter_miss(monkeypatch, release)
    client, headers, _ = _concurrent_client(tmp_path)
    body = {**_youtube(), "external_consent": True}
    leader = client.post("/api/jobs/analysis", json={**body, "max_new_requests": 1}, headers=headers).json()
    assert started.wait(5)
    state["armed"] = True
    waiter = client.post("/api/jobs/analysis", json={**body, "max_new_requests": waiter_budget},
                         headers=headers).json()
    leader, waiter = _wait(client, headers, leader), _wait(client, headers, waiter)
    assert state["misses"] >= 2
    assert len(calls) == 1
    assert waiter["status"] == "completed", waiter
    segment = waiter["result"]["project"]["analysis"][0]
    assert segment["reuse"]["kind"] == "cache" and segment["provider_attempted"] is False
    assert segment["profile"] == leader["result"]["project"]["analysis"][0]["profile"]


def test_refresh_does_not_take_the_single_flight_cache_shortcut(tmp_path):
    cache = AnalysisCache(tmp_path / "analysis")
    key = make_key("youtube", "401iS5McuoU", Interval(start_seconds=0, end_seconds=5), "m")
    cache.put(key, _profile(), "m")
    never = threading.Event()
    mode, cached = cache.run_shared(key, lambda: {"fresh": True}, never, may_start=False)
    assert mode == "cache" and cached.profile["gain"] == "clean"
    assert cache.run_shared(key, lambda: {"fresh": True}, never, may_start=False,
                            use_cache=False) == ("blocked", None)
    assert cache.run_shared(key, lambda: {"fresh": True}, never, may_start=True,
                            use_cache=False) == ("fresh", {"fresh": True})

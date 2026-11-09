"""Offline regressions for budgeted refinement: zero provider calls, exact counts, honest evidence."""

import dataclasses
import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from test_workbench_api import HOST, ORIGIN, FakeAdapter, FakeRenderer
from test_workbench_recipes import adapter as recipe_adapter  # noqa: F401 - pytest fixture
from test_workbench_recipes import profile as recipe_profile

from tone_listener_mcp.workbench import refinement as refine
from tone_listener_mcp.workbench.adapters.soundshed import SoundshedAdapter, new_project
from tone_listener_mcp.workbench.jobs import JobManager
from tone_listener_mcp.workbench.models import ChainDraft, ChainNode, PreviewSnapshot, SourceRef
from tone_listener_mcp.workbench.renderer import RenderedAudio
from tone_listener_mcp.workbench.storage import LocalStorage

GAINS = refine.GAIN_ORDER


class RefineAdapter(FakeAdapter):
    """Recipe depends on the profile's gain so variants are distinct chains."""

    def default_chain(self, analysis=()):
        item = analysis[0] if analysis else {}
        profile = item.get("profile") if isinstance(item, dict) else getattr(item, "profile", None)
        gain = getattr(profile, "gain", None) or (profile or {}).get("gain", "clean")
        return ChainDraft(nodes=[
            ChainNode(id="input", kind="input", title="Input"),
            ChainNode(id="gain", kind="gain", title="Gain", params={"gainDb": GAINS.index(gain) * 2}),
            ChainNode(id="profile-eq", kind="parametric_eq", title="EQ",
                      params={"lowGain": 0, "lowMidGain": 0, "highMidGain": 0, "highGain": 0}),
            ChainNode(id="output", kind="output", title="Output"),
        ])


class RenderingFake(FakeRenderer):
    def __init__(self, path):
        super().__init__(path)
        self.renders = 0

    def render(self, project, sample_name, cache_root, cancel_event=None):
        self.renders += 1
        gain = next(node.params["gainDb"] for node in project.chain.nodes if node.id == "gain")
        stereo = np.stack([np.sin(np.linspace(0, 400, 4410)) * (0.1 + gain / 100)] * 2, axis=1).astype("<f4")
        snapshot = PreviewSnapshot(
            project_id=project.id, project_revision=project.revision, chain_revision=project.chain.revision,
            chain_sha256=SoundshedAdapter.chain_sha256(project.chain), sample_name=sample_name, source_sample_sha256="1" * 64,
            engine_version="test-engine", sample_rate=44100, dsp_conditions={"state_canary_version": 1},
            audio_sha256="2" * 64, duration_seconds=0.1, rms=0.1, peak=0.2, created_at="now",
        )
        path = Path(cache_root) / f"{snapshot.id}.wav"
        wavfile.write(path, 44100, stereo)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return RenderedAudio(samples=stereo, snapshot=snapshot.model_copy(update={"audio_sha256": digest}),
                             sample_name=sample_name)


class FakeJudge:
    """Synthetic judgments only; counts every would-be provider attempt."""

    def __init__(self, current_gain="less", control_gain="more", status=None, preferred="recipe_hotter",
                 preferred_gain="similar"):
        self.calls = []
        self.current_gain, self.control_gain, self.status = current_gain, control_gain, status
        self.preferred, self.preferred_gain = preferred, preferred_gain

    def __call__(self, *, model, candidates, seed, youtube=None, anchor_clip=None):
        assert (youtube is None) != (anchor_clip is None)
        self.calls.append({"model": model, "candidates": sorted(candidates), "youtube": youtube,
                           "anchor_seconds": None if anchor_clip is None else (anchor_clip.start_seconds, anchor_clip.duration)})
        mapping = refine.relative_judge.blind_labels(sorted(candidates), seed)
        result = {"label_mapping": mapping, "request_counts": {"generate": 1, "upload": len(candidates)},
                  "judgment": None, "invalid": None, "failure_info": None, "finish_reason": "STOP"}
        if self.status is not None and len(self.calls) == 1:
            result["failure_info"] = {"http_status": self.status, "category": "quota_unspecified"}
            result["finish_reason"] = None
            return result
        position = {self.preferred: 0, "current": 1, "control": 3}
        order = sorted(mapping, key=lambda label: position.get(mapping[label], 2))
        result["judgment"] = {
            "reference_guitar_audible": "clear", "closest_to_reference_order": order,
            "closest_is_close_match": "uncertain", "notes": [],
            "comparisons": [{
                "clip": label,
                "gain": self.control_gain if identifier == "control"
                else self.preferred_gain if identifier == self.preferred else self.current_gain,
                **{axis: "similar" for axis in ("brightness", "bass", "mids", "attack", "compression", "ambience")},
            } for label, identifier in mapping.items()],
        }
        return result


def _client(tmp_path, judge, kind="youtube", dominance="clear", content_sha256=None, renderer_class=None,
            job_manager=None, adapter=None):
    from fastapi.testclient import TestClient

    from tone_listener_mcp.workbench.api import create_app

    storage = LocalStorage(tmp_path / "workbench")
    binary = tmp_path / "renderer"
    binary.write_text("test")
    renderer = (renderer_class or RenderingFake)(binary)
    adapter = adapter or RefineAdapter()
    app = create_app(storage=storage, adapter=adapter, renderer=renderer, allowed_hosts={HOST},
                     job_manager=job_manager or JobManager(workers=1, max_active=3), allow_test_client=True,
                     refinement_judge=judge)
    client = TestClient(app)
    token = client.get("/api/session", headers={"Host": HOST, "Origin": ORIGIN}).json()["token"]
    headers = {"Host": HOST, "Origin": ORIGIN, "X-Workbench-Token": token}
    selected = recipe_profile(gain="crunch", guitar_dominance=dominance)
    source = SourceRef(kind=kind, url="https://www.youtube.com/watch?v=dQw4w9WgXcQ" if kind == "youtube" else None,
                       display_name="Reference", duration_seconds=60 if kind == "youtube" else 25,
                       selected_intervals=[{"start_seconds": 10, "end_seconds": 20}],
                       content_sha256=content_sha256)
    project = new_project(
        source=source,
        analysis=[{"interval": {"start_seconds": 10, "end_seconds": 20}, "model": "gemini-3.6-flash",
                   "profile": selected.model_dump(mode="json")}],
        intent={"summary": "Crunch."}, chain=adapter.default_chain([{"profile": selected}]),
        engine_version="test-engine", name="Refine me",
    )
    storage.save_project(project)
    client.storage = storage
    return client, headers, project, renderer


def _reference_wav(tmp_path, seconds=25, freq=220.0, name="reference.wav"):
    rate = 8000
    t = np.arange(int(seconds * rate)) / rate
    path = tmp_path / name
    wavfile.write(path, rate, (0.2 * np.sin(2 * np.pi * freq * t)).astype("<f4"))
    return path


def _upload(client, headers, path):
    with path.open("rb") as handle:
        response = client.post("/api/uploads", headers=headers, files={"file": (path.name, handle, "audio/wav")})
    assert response.status_code == 200, response.text
    return response.json()["upload_id"]


def _file_client(tmp_path, judge, renderer_class=None):
    path = _reference_wav(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (*_client(tmp_path, judge, kind="audio_file", content_sha256=digest, renderer_class=renderer_class),
            path)


def _uploads_left(client):
    return sorted(item.name for item in client.storage.uploads.iterdir())


def _wait(client, headers, job):
    for _ in range(200):
        status = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if status["status"] not in ("queued", "running"):
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _plan(client, headers, project, **extra):
    body = {"project_revision": project.revision, "chain_revision": project.chain.revision, **extra}
    return client.post(f"/api/projects/{project.id}/refinement/preflight", headers=headers, json=body)


def _start(client, headers, project, plan, **overrides):
    body = {"project_revision": project.revision, "chain_revision": project.chain.revision,
            "model": plan["model"], "samples": plan["samples"], "plan_sha256": plan["plan_sha256"],
            "max_new_requests": plan["max_new_requests"], "external_consent": True, **overrides}
    return client.post(f"/api/projects/{project.id}/jobs/refinement", headers=headers, json=body)


def test_preflight_is_local_and_reports_exact_bounded_requests(tmp_path):
    judge = FakeJudge()
    client, headers, project, renderer = _client(tmp_path, judge)
    response = _plan(client, headers, project)
    assert response.status_code == 200
    plan = response.json()
    assert plan["max_new_requests"] == 2 and plan["samples"] == ["riff-1", "riff-2"]
    roles = {item["id"]: item["role"] for item in plan["variants"]}
    assert roles["current"] == "current" and roles["control"] == "control"
    assert {"recipe_cleaner", "recipe_hotter", "brighter", "darker"} <= set(roles)
    assert len({item["chain_sha256"] for item in plan["variants"]}) == len(plan["variants"])
    assert "unusable" in plan["evidence"]["ambience"] and "weak" in plan["evidence"]["brightness"]
    assert judge.calls == [] and renderer.renders == 0
    assert _plan(client, headers, project).json()["plan_sha256"] == plan["plan_sha256"]
    assert _plan(client, headers, project, model="gemini-4-argon").json()["plan_sha256"] != plan["plan_sha256"]


def test_confirmed_run_makes_exactly_one_request_per_sample_and_suggests_calibrated_gain(tmp_path):
    judge = FakeJudge(current_gain="less", control_gain="more")
    client, headers, project, renderer = _client(tmp_path, judge)
    plan = _plan(client, headers, project).json()
    status = _wait(client, headers, _start(client, headers, project, plan).json())
    assert status["status"] == "completed"
    record = status["result"]["refinement"]
    assert len(judge.calls) == 2 and record["new_generate_requests"] == 2
    assert all(call["youtube"][1:] == (10, 20) for call in judge.calls)
    assert renderer.renders == 2 * len(plan["variants"])
    assert record["suggestion"]["variant_id"] == "recipe_hotter"
    assert all(item["control_passed"] for item in record["judgments"])
    saved = client.get(f"/api/projects/{project.id}/refinements", headers=headers).json()
    assert saved[0]["id"] == record["id"] and saved[0]["plan_sha256"] == plan["plan_sha256"]
    audio = client.get(f"/api/projects/{project.id}/refinements/{record['id']}/audio/recipe_hotter/riff-1",
                       headers=headers)
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"


def test_stale_or_underconfirmed_plan_is_refused_without_any_call(tmp_path):
    judge = FakeJudge()
    client, headers, project, renderer = _client(tmp_path, judge)
    plan = _plan(client, headers, project).json()
    assert _start(client, headers, project, plan, plan_sha256="f" * 64).status_code == 409
    assert _start(client, headers, project, plan, max_new_requests=1).status_code == 409
    assert _start(client, headers, project, plan, external_consent=False).status_code == 422
    edited = client.put(f"/api/projects/{project.id}/chain", headers=headers, json={
        "base_revision": project.revision,
        "nodes": [node.model_dump(mode="json") for node in project.chain.nodes],
    })
    assert edited.status_code == 200
    assert _start(client, headers, project, plan).status_code == 409
    assert judge.calls == [] and renderer.renders == 0


def test_quota_response_stops_remaining_samples_without_retry(tmp_path):
    judge = FakeJudge(status=429)
    client, headers, project, _ = _client(tmp_path, judge)
    plan = _plan(client, headers, project).json()
    status = _wait(client, headers, _start(client, headers, project, plan).json())
    record = status["result"]["refinement"]
    assert len(judge.calls) == 1 and record["new_generate_requests"] == 1
    assert [item["status"] for item in record["judgments"]] == ["failed", "not_attempted"]
    assert record["suggestion"]["variant_id"] is None and status["status"] == "partial"


def test_failed_control_blocks_any_suggestion(tmp_path):
    judge = FakeJudge(current_gain="less", control_gain="similar")
    client, headers, project, _ = _client(tmp_path, judge)
    plan = _plan(client, headers, project).json()
    record = _wait(client, headers, _start(client, headers, project, plan).json())["result"]["refinement"]
    assert not any(item["control_passed"] for item in record["judgments"])
    assert record["suggestion"]["variant_id"] is None


def test_adopt_is_local_and_rejects_control_and_stale_chain(tmp_path):
    judge = FakeJudge()
    client, headers, project, _ = _client(tmp_path, judge)
    plan = _plan(client, headers, project).json()
    record = _wait(client, headers, _start(client, headers, project, plan).json())["result"]["refinement"]
    calls = len(judge.calls)
    url = f"/api/projects/{project.id}/refinements/{record['id']}/adopt"
    assert client.post(url, headers=headers, json={"base_revision": project.revision,
                                                   "variant_id": "control"}).status_code == 409
    adopted = client.post(url, headers=headers, json={"base_revision": project.revision,
                                                      "variant_id": "recipe_hotter"})
    assert adopted.status_code == 200
    updated = adopted.json()
    assert updated["revision"] == project.revision + 1 and updated["chain"]["revision"] == project.chain.revision + 1
    assert next(node for node in updated["chain"]["nodes"] if node["id"] == "gain")["params"]["gainDb"] == 6
    assert client.post(url, headers=headers, json={"base_revision": updated["revision"],
                                                   "variant_id": "brighter"}).status_code == 409
    assert len(judge.calls) == calls


@pytest.mark.parametrize(("kind", "dominance"), [("audio_file", "clear"), ("youtube", "mixed")])
def test_unusable_reference_is_refused_before_any_work(tmp_path, kind, dominance):
    judge = FakeJudge()
    client, headers, project, renderer = _client(tmp_path, judge, kind=kind, dominance=dominance)
    assert _plan(client, headers, project).status_code == 409
    assert judge.calls == [] and renderer.renders == 0


def test_real_adapter_variants_keep_omitted_notes_and_build_wrong_control(recipe_adapter):  # noqa: F811
    for key, name, category in (("tone3000:758458", "Edge candidate", "amp"),
                                ("tone3000:383084", "Drive pedal candidate", "pedal"),
                                ("tone3000:494017", "VOX drive candidate", "amp")):
        path = recipe_adapter.resource_root / f"extra-{name[:4]}"
        path.write_bytes(name.encode())
        recipe_adapter._records[f"nam:{key}"] = {"type": "nam", "filePath": path.name, "name": name,
                                                 "category": category, "metadata": {"captureScope": category}}
    selected = recipe_profile(gain="crunch", delay={"presence": "possible", "prominence": "subtle"})
    chain = recipe_adapter.default_chain([{"profile": selected}])
    variants = {item.id: item for item in refine.build_variants(recipe_adapter, chain, selected)}
    assert set(variants) == {"current", "recipe_cleaner", "recipe_hotter", "pushed_clean_amp", "brighter",
                             "darker", "control"}
    for item in variants.values():
        assert not recipe_adapter.validate_chain(item.chain)
        if item.role != "control":
            assert any(node.id == "delay-note" and node.note_role == "omitted_optional" for node in item.chain.nodes)
        assert not any(node.kind == "digital_delay" for node in item.chain.nodes)
    amp = {key: next(node.asset.resource_id for node in item.chain.nodes if node.kind == "nam_amp")
           for key, item in variants.items()}
    assert amp["recipe_cleaner"] != amp["current"] != amp["recipe_hotter"]
    eq = {key: next(node.params["highGain"] for node in item.chain.nodes if node.kind == "parametric_eq")
          for key, item in variants.items()}
    assert eq["brighter"] > eq["current"] > eq["darker"]
    assert any(node.kind == "nam_fx" for node in variants["control"].chain.nodes)
    assert refine.control_is_hotter(selected) is True


def test_judgment_rejects_unclear_reference_and_rejects_closest_control():
    labels = refine.relative_judge.blind_labels(["control", "current"], 3)
    base = {"label_mapping": labels, "request_counts": {"generate": 1, "upload": 2}, "invalid": None,
            "failure_info": None, "finish_reason": "STOP"}
    order = sorted(labels, key=lambda label: labels[label] != "control")  # control ranked closest
    comparisons = [{"clip": label, "gain": "more", **{axis: "similar" for axis in (
        "brightness", "bass", "mids", "attack", "compression", "ambience")}} for label in labels]
    clear = refine.summarize_judgment("riff-1", {**base, "judgment": {
        "reference_guitar_audible": "clear", "comparisons": comparisons, "closest_to_reference_order": order,
        "closest_is_close_match": "no", "notes": []}}, hot=True)
    assert clear.control_passed is False
    partial = refine.summarize_judgment("riff-1", {**base, "judgment": {
        "reference_guitar_audible": "partial", "comparisons": comparisons,
        "closest_to_reference_order": list(reversed(order)), "closest_is_close_match": "no", "notes": []}}, hot=True)
    assert partial.control_passed is False
    assert refine.suggest_from([clear, partial]).variant_id is None


def test_reselected_same_file_compares_against_local_anchor_and_is_deleted(tmp_path):
    judge = FakeJudge(current_gain="less", control_gain="more")
    client, headers, project, renderer, path = _file_client(tmp_path, judge)
    assert _plan(client, headers, project).status_code == 428
    upload = _upload(client, headers, path)
    response = _plan(client, headers, project, upload_id=upload)
    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["reference"]["kind"] == "audio_file"
    assert plan["reference"]["content_sha256"] == project.source.content_sha256
    assert plan["uploads_per_request"] == len(plan["variants"]) + 1 and plan["max_new_requests"] == 2
    assert "deleted" in plan["disclosure"] and judge.calls == [] and renderer.renders == 0
    status = _wait(client, headers, _start(client, headers, project, plan, upload_id=upload).json())
    assert status["status"] == "completed"
    assert len(judge.calls) == 2 and all(call["youtube"] is None for call in judge.calls)
    assert all(abs(call["anchor_seconds"][0] - 10) < 0.01 and abs(call["anchor_seconds"][1] - 10) < 0.01
               for call in judge.calls)
    assert _uploads_left(client) == [] and client.storage.get_upload(upload) is None


def test_reselected_different_file_is_refused_and_discarded(tmp_path):
    judge = FakeJudge()
    client, headers, project, renderer, _ = _file_client(tmp_path, judge)
    other = _upload(client, headers, _reference_wav(tmp_path, freq=330.0, name="other.wav"))
    response = _plan(client, headers, project, upload_id=other)
    assert response.status_code == 409 and "not the analyzed source" in response.json()["detail"]
    assert _uploads_left(client) == [] and judge.calls == [] and renderer.renders == 0


def test_legacy_file_project_without_fingerprint_explains_reanalysis(tmp_path):
    judge = FakeJudge()
    client, headers, project, _ = _client(tmp_path, judge, kind="audio_file")
    upload = _upload(client, headers, _reference_wav(tmp_path))
    response = _plan(client, headers, project, upload_id=upload)
    assert response.status_code == 409 and "Analyze the file again" in response.json()["detail"]
    assert _uploads_left(client) == [] and judge.calls == []


def test_file_plan_is_bound_to_upload_and_stale_confirmation_is_discarded(tmp_path):
    judge = FakeJudge()
    client, headers, project, _, path = _file_client(tmp_path, judge)
    first, second = _upload(client, headers, path), _upload(client, headers, path)
    plan = _plan(client, headers, project, upload_id=first).json()
    assert _plan(client, headers, project, upload_id=second).json()["plan_sha256"] != plan["plan_sha256"]
    assert _start(client, headers, project, plan, upload_id=second).status_code == 409
    assert client.storage.get_upload(second) is None and client.storage.get_upload(first) is not None
    assert _start(client, headers, project, plan, upload_id=first, max_new_requests=1).status_code == 409
    assert client.storage.get_upload(first) is None and judge.calls == []


def test_file_refinement_429_stops_without_retry_and_still_deletes_upload(tmp_path):
    judge = FakeJudge(status=429)
    client, headers, project, _, path = _file_client(tmp_path, judge)
    upload = _upload(client, headers, path)
    plan = _plan(client, headers, project, upload_id=upload).json()
    status = _wait(client, headers, _start(client, headers, project, plan, upload_id=upload).json())
    assert status["status"] == "partial" and len(judge.calls) == 1
    assert _uploads_left(client) == []


def test_file_refinement_cancel_deletes_upload_without_requests(tmp_path):
    import threading

    started, release = threading.Event(), threading.Event()

    class BlockingRenderer(RenderingFake):
        def render(self, project, sample_name, cache_root, cancel_event=None):
            started.set()
            assert release.wait(5)
            return super().render(project, sample_name, cache_root, cancel_event)

    judge = FakeJudge()
    client, headers, project, _, path = _file_client(tmp_path, judge, renderer_class=BlockingRenderer)
    upload = _upload(client, headers, path)
    plan = _plan(client, headers, project, upload_id=upload).json()
    job = _start(client, headers, project, plan, upload_id=upload).json()
    assert started.wait(5)
    assert client.post(f"/api/jobs/{job['id']}/cancel", headers=headers).status_code == 200
    release.set()
    status = _wait(client, headers, job)
    assert status["status"] == "cancelled" and judge.calls == []
    assert _uploads_left(client) == []


def test_real_adapter_pushed_clean_amp_reproduces_development_topology(recipe_adapter):  # noqa: F811
    for key, name, category in (("tone3000:758458", "Edge candidate", "amp"),
                                ("tone3000:383084", "Drive pedal candidate", "pedal"),
                                ("tone3000:494017", "VOX drive candidate", "amp")):
        path = recipe_adapter.resource_root / f"extra-{name[:4]}"
        path.write_bytes(name.encode())
        recipe_adapter._records[f"nam:{key}"] = {"type": "nam", "filePath": path.name, "name": name,
                                                 "category": category, "metadata": {"captureScope": category}}
    selected = recipe_profile(gain="overdrive", brightness="bright")
    chain = recipe_adapter.default_chain([{"profile": selected}])
    variants = {item.id: item for item in refine.build_variants(recipe_adapter, chain, selected)}
    pushed = variants["pushed_clean_amp"].chain
    clean = recipe_adapter.default_chain([{"profile": selected.model_copy(
        update={"gain": "clean", "brightness": "balanced"})}])
    kinds = [node.kind for node in pushed.nodes]
    assert "nam_fx" not in kinds and kinds == [node.kind for node in clean.nodes]
    amp = next(node for node in pushed.nodes if node.id == "amp")
    assert amp.asset == next(node for node in clean.nodes if node.id == "amp").asset
    assert amp.params["inputGain"] == 9
    assert "weak" in variants["pushed_clean_amp"].description
    assert "pushed_clean_amp" not in {item.id for item in refine.build_variants(
        recipe_adapter, recipe_adapter.default_chain([{"profile": recipe_profile(gain="clean")}]),
        recipe_profile(gain="clean"))}


def test_measured_worse_proposed_variant_is_never_suggested(tmp_path):
    judge = FakeJudge(current_gain="less", control_gain="more", preferred="recipe_hotter", preferred_gain="much_more")
    client, headers, project, _ = _client(tmp_path, judge)
    plan = _plan(client, headers, project).json()
    record = _wait(client, headers, _start(client, headers, project, plan).json())["result"]["refinement"]
    assert all(item["control_passed"] for item in record["judgments"])
    assert record["suggestion"]["variant_id"] is None


def test_one_good_di_plus_missing_second_is_not_agreement(tmp_path):
    judge = FakeJudge(status=None)
    client, headers, project, _ = _client(tmp_path, judge)
    plan = _plan(client, headers, project).json()
    record = _wait(client, headers, _start(client, headers, project, plan).json())["result"]["refinement"]
    judged = [refine.SampleJudgment.model_validate(item) for item in record["judgments"]]
    assert refine.suggest_from(judged).variant_id == "recipe_hotter"
    missing = refine.SampleJudgment(sample="riff-2", status="failed", reason="offline")
    assert refine.suggest_from([judged[0], missing]).variant_id is None


def test_cancelling_queued_refinement_releases_project_and_upload(tmp_path):
    import threading

    manager = JobManager(workers=1, max_active=4)
    entered, release = threading.Event(), threading.Event()
    blocker = manager.submit("preview", lambda progress, cancelled: (entered.set(), release.wait(10), {})[2])
    assert entered.wait(5)
    judge = FakeJudge()
    path = _reference_wav(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        client, headers, project, _ = _client(tmp_path, judge, kind="audio_file", content_sha256=digest,
                                              job_manager=manager)
        upload = _upload(client, headers, path)
        plan = _plan(client, headers, project, upload_id=upload).json()
        queued = _start(client, headers, project, plan, upload_id=upload)
        assert queued.status_code == 202
        assert manager.cancel(queued.json()["id"]).status == "cancelled"
        assert _uploads_left(client) == []
        again = _upload(client, headers, path)
        plan = _plan(client, headers, project, upload_id=again).json()
        restarted = _start(client, headers, project, plan, upload_id=again)
        assert restarted.status_code == 202, restarted.text
        manager.cancel(restarted.json()["id"])
        assert judge.calls == []
    finally:
        release.set()
        manager.cancel(blocker.id)
        manager.close()


def test_prune_keeps_running_and_unknown_directories(tmp_path):
    import json
    import os
    from uuid import uuid4

    from tone_listener_mcp.workbench.api import _prune_refinements

    root = tmp_path / "refinements"
    root.mkdir()
    made = {}
    for index, status in enumerate(("running", "completed", "failed", "completed", None)):
        directory = root / str(uuid4())
        directory.mkdir()
        if status:
            (directory / "record.json").write_text(json.dumps({"status": status}))
        os.utime(directory, (index + 1, index + 1))
        made[index] = directory
    _prune_refinements(root, keep=1)
    assert made[0].is_dir() and made[4].is_dir() and made[3].is_dir()
    assert not made[1].exists() and not made[2].exists()


def test_audition_is_level_matched_across_variants(tmp_path):
    import io

    import soundfile as sf

    judge = FakeJudge()
    client, headers, project, _ = _client(tmp_path, judge)
    plan = _plan(client, headers, project).json()
    record = _wait(client, headers, _start(client, headers, project, plan).json())["result"]["refinement"]
    levels = []
    for variant in ("current", "recipe_hotter", "control"):
        response = client.get(f"/api/projects/{project.id}/refinements/{record['id']}/audio/{variant}/riff-1",
                              headers=headers)
        assert response.status_code == 200 and response.headers["x-audition-level"] == "rms-matched"
        samples, _ = sf.read(io.BytesIO(response.content), always_2d=True)
        levels.append(float(np.sqrt(np.mean(samples ** 2))))
        assert np.max(np.abs(samples)) <= 10 ** (-1 / 20) + 1e-6
    assert max(levels) - min(levels) < 1e-4


def _register_extras(adapter):
    for key, name, category in (("tone3000:758458", "Edge candidate", "amp"),
                                ("tone3000:383084", "Drive pedal candidate", "pedal"),
                                ("tone3000:494017", "VOX drive candidate", "amp")):
        path = adapter.resource_root / f"extra-{name[:4]}"
        path.write_bytes(name.encode())
        adapter._records[f"nam:{key}"] = {"type": "nam", "filePath": path.name, "name": name,
                                          "category": category, "metadata": {"captureScope": category}}


def _changes(base, chain):
    before = {node.id: node for node in base.nodes}
    assert [node.id for node in chain.nodes] == [node.id for node in base.nodes]
    out = []
    for node in chain.nodes:
        old = before[node.id]
        assert (node.kind, node.asset, node.enabled, node.note_role, node.title) == (
            old.kind, old.asset, old.enabled, old.note_role, old.title)
        out += [(node.id, key, old.params[key], value) for key, value in node.params.items()
                if old.params.get(key) != value]
    return out


def test_local_search_moves_one_parameter_on_the_same_gear(recipe_adapter):  # noqa: F811
    _register_extras(recipe_adapter)
    selected = recipe_profile(gain="overdrive", brightness="bright",
                              delay={"presence": "possible", "prominence": "subtle"})
    chain = recipe_adapter.default_chain([{"profile": selected}])
    capture = next(node for node in chain.nodes if node.enabled and node.kind in ("nam_fx", "nam_amp"))
    eq = next(node for node in chain.nodes if node.kind == "parametric_eq")
    variants = {item.id: item for item in refine.build_variants(recipe_adapter, chain, selected, "local")}
    assert set(variants) == {"current", "trim_minus6", "trim_minus3", "trim_plus3", "trim_plus6",
                             "eq_brighter", "eq_darker", "joint_plus6_brighter", "joint_plus6_darker",
                             "joint_minus6_brighter", "joint_minus6_darker", "control"}
    assert variants["current"].chain == chain
    trim = capture.params["inputGain"]
    for identifier, step in (("trim_minus6", -6), ("trim_minus3", -3), ("trim_plus3", 3), ("trim_plus6", 6)):
        assert _changes(chain, variants[identifier].chain) == [(capture.id, "inputGain", trim, trim + step)]
    mid, high = eq.params["highMidGain"], eq.params["highGain"]
    assert _changes(chain, variants["eq_brighter"].chain) == [(eq.id, "highMidGain", mid, mid + 3),
                                                              (eq.id, "highGain", high, high + 3)]
    assert _changes(chain, variants["eq_darker"].chain) == [(eq.id, "highMidGain", mid, mid - 3),
                                                            (eq.id, "highGain", high, high - 3)]
    for trim_step, sign in ((6, "plus"), (-6, "minus")):
        for eq_step, name in ((3, "brighter"), (-3, "darker")):
            joint = variants[f"joint_{sign}6_{name}"]
            assert _changes(chain, joint.chain) == [(capture.id, "inputGain", trim, trim + trim_step),
                                                    (eq.id, "highMidGain", mid, mid + eq_step),
                                                    (eq.id, "highGain", high, high + eq_step)]
            assert joint.role == "variant" and "joint" in joint.description
    for item in variants.values():
        assert not recipe_adapter.validate_chain(item.chain)
    assert variants["control"].role == "control"
    assert [node.asset for node in variants["control"].chain.nodes] != [node.asset for node in chain.nodes]
    assert len({item.chain_sha256 for item in variants.values()}) == len(variants)


def test_local_search_clamps_to_range_and_drops_no_op_or_duplicate_moves(recipe_adapter):  # noqa: F811
    _register_extras(recipe_adapter)
    selected = recipe_profile(gain="overdrive")
    chain = recipe_adapter.default_chain([{"profile": selected}])
    capture = next(node for node in chain.nodes if node.enabled and node.kind in ("nam_fx", "nam_amp"))

    def at(trim):
        nodes = [node.model_copy(update={"params": {**node.params, "inputGain": trim}}) if node.id == capture.id
                 else node for node in chain.nodes]
        return ChainDraft(revision=chain.revision, nodes=nodes)

    near = {item.id: item for item in refine.build_variants(recipe_adapter, at(22), selected, "local")}
    trims = {key: next(n for n in item.chain.nodes if n.id == capture.id).params["inputGain"]
             for key, item in near.items() if key.startswith("trim_")}
    assert trims == {"trim_minus6": 16, "trim_minus3": 19, "trim_plus3": 24}  # +6 clamps to the same chain as +3
    top = {item.id for item in refine.build_variants(recipe_adapter, at(24), selected, "local")}
    assert not {"trim_plus3", "trim_plus6"} & top and {"trim_minus3", "trim_minus6"} <= top
    # A joint move whose trim part is a no-op is never offered; the minus joints remain.
    assert not {"joint_plus6_brighter", "joint_plus6_darker"} & top
    assert {"joint_minus6_brighter", "joint_minus6_darker"} <= top
    # Joint +6 at 22 clamps its trim to 24, the same trim as +3; it stays distinct because the EQ also moved.
    joint = next(n for n in near["joint_plus6_brighter"].chain.nodes if n.id == capture.id)
    assert joint.params["inputGain"] == 24
    assert len({item.chain_sha256 for item in near.values()}) == len(near)


def test_local_joint_move_that_clamps_to_a_single_axis_chain_is_listed_once(recipe_adapter):  # noqa: F811
    _register_extras(recipe_adapter)
    selected = recipe_profile(gain="overdrive")
    chain = recipe_adapter.default_chain([{"profile": selected}])
    eq = next(node for node in chain.nodes if node.kind == "parametric_eq")
    top = {"highMidGain": eq.params["highMidGain"], "highGain": eq.params["highGain"]}
    limits = {spec.id: spec.maximum for spec in refine._PARAMETERS["parametric_eq"]}
    nodes = [node.model_copy(update={"params": {**node.params, "highMidGain": limits["highMidGain"],
                                                  "highGain": limits["highGain"]}}) if node.id == eq.id else node
             for node in chain.nodes]
    at_top = ChainDraft(revision=chain.revision, nodes=nodes)
    ids = [item.id for item in refine.build_variants(recipe_adapter, at_top, selected, "local")]
    assert top and "eq_brighter" not in ids and "eq_darker" in ids
    # +6/brighter would equal trim_plus6 (EQ already at max): dropped, not duplicated under a joint id.
    assert "joint_plus6_brighter" not in ids and "joint_minus6_brighter" not in ids
    assert {"trim_plus6", "trim_minus6", "joint_plus6_darker", "joint_minus6_darker"} <= set(ids)


def test_local_plan_is_distinct_bounded_and_adopted_chain_is_the_next_centre(tmp_path):
    judge = FakeJudge(preferred="eq_brighter", current_gain="similar", control_gain="more")
    client, headers, project, renderer = _client(tmp_path, judge)
    explore = _plan(client, headers, project).json()
    local = _plan(client, headers, project, search="local").json()
    assert local["search"] == "local" and explore["search"] == "explore"
    assert local["plan_sha256"] != explore["plan_sha256"] and local["max_new_requests"] == 2
    assert [item["id"] for item in local["variants"]] == ["current", "eq_brighter", "eq_darker", "control"]
    assert judge.calls == [] and renderer.renders == 0
    # A local confirmation cannot be replayed as an explore run (or vice versa) without a fresh review.
    assert _start(client, headers, project, local).status_code == 409 and judge.calls == []
    status = _wait(client, headers, _start(client, headers, project, local, search="local").json())
    record = status["result"]["refinement"]
    assert record["search"] == "local" and len(judge.calls) == 2
    assert record["suggestion"]["variant_id"] == "eq_brighter" and "WEAK" in record["suggestion"]["basis"]
    assert renderer.renders == 2 * len(local["variants"])
    assert all(call["candidates"] == sorted(item["id"] for item in local["variants"]) for call in judge.calls)
    url = f"/api/projects/{project.id}/refinements/{record['id']}/adopt"
    adopted = client.post(url, headers=headers, json={"base_revision": project.revision,
                                                      "variant_id": "eq_brighter"}).json()
    nodes = {node["id"]: node for node in adopted["chain"]["nodes"]}
    assert nodes["profile-eq"]["params"]["highGain"] == 3 and nodes["profile-eq"]["params"]["highMidGain"] == 3 and nodes["gain"] == project.chain.nodes[1].model_dump(
        mode="json")
    assert adopted["analysis"] == [item.model_dump(mode="json") for item in project.analysis]
    updated = client.storage.load_project(project.id)
    nxt = _plan(client, headers, updated, search="local").json()
    highs = {item["id"]: next(n for n in item["chain"]["nodes"] if n["id"] == "profile-eq")["params"]["highGain"]
             for item in nxt["variants"] if item["role"] != "control"}
    assert highs == {"current": 3, "eq_brighter": 6, "eq_darker": 0}
    assert len(judge.calls) == 2


class ChainLevelRenderer(RenderingFake):
    """Distinct audio per chain without assuming a node named 'gain'."""

    def render(self, project, sample_name, cache_root, cancel_event=None):
        digest = SoundshedAdapter.chain_sha256(project.chain)
        nodes = [node.model_copy(update={"params": {"gainDb": int(digest[:2], 16) / 40}})
                 if node.kind == "input" else node for node in project.chain.nodes]
        fake = project.model_copy(update={"chain": project.chain.model_copy(update={"nodes": [
            ChainNode(id="gain", kind="gain", title="Gain", params={"gainDb": nodes[0].params["gainDb"]})]})})
        rendered = super().render(fake, sample_name, cache_root, cancel_event)  # chain identity restored below
        snapshot = rendered.snapshot.model_copy(update={"chain_sha256": digest, "chain_revision": project.chain.revision,
                                                        "project_revision": project.revision})
        return dataclasses.replace(rendered, snapshot=snapshot)


def test_local_plan_with_capture_discloses_joint_counts_and_adopts_a_joint_move(tmp_path, recipe_adapter):  # noqa: F811
    _register_extras(recipe_adapter)
    judge = FakeJudge(preferred="joint_plus6_brighter", current_gain="similar", control_gain="more")
    client, headers, project, _ = _client(tmp_path, judge, adapter=recipe_adapter,
                                          renderer_class=ChainLevelRenderer)
    local = _plan(client, headers, project, search="local").json()
    ids = [item["id"] for item in local["variants"]]
    assert ids[0] == "current" and ids[-1] == "control" and len(ids) == 12
    assert {"joint_plus6_brighter", "joint_plus6_darker", "joint_minus6_brighter", "joint_minus6_darker"} <= set(ids)
    assert local["uploads_per_request"] == 12 and local["max_new_requests"] == 2 and judge.calls == []
    status = _wait(client, headers, _start(client, headers, project, local, search="local").json())
    assert status["status"] == "completed", status["error"]
    record = status["result"]["refinement"]
    assert len(judge.calls) == 2 and all(len(call["candidates"]) == 12 for call in judge.calls)
    assert record["suggestion"]["variant_id"] == "joint_plus6_brighter"
    stored = client.storage.load_project(project.id)
    assert stored.chain == project.chain  # nothing changes before explicit adoption
    url = f"/api/projects/{project.id}/refinements/{record['id']}/adopt"
    for bad in ("Trim", "6trim", "trim-plus6", "trim plus"):  # schema still rejects malformed ids
        assert client.post(url, headers=headers, json={"base_revision": project.revision,
                                                       "variant_id": bad}).status_code == 422
    # Ids containing digits were rejected by the old ^[a-z_]+$ pattern, so no trim/joint move was adoptable.
    adopted = client.post(url, headers=headers, json={"base_revision": project.revision,
                                                      "variant_id": "joint_plus6_brighter"}).json()
    expected = next(item for item in local["variants"] if item["id"] == "joint_plus6_brighter")["chain"]["nodes"]
    assert [node["params"] for node in adopted["chain"]["nodes"]] == [node["params"] for node in expected]
    changed = _changes(project.chain, ChainDraft.model_validate(adopted["chain"]))
    assert [key for _, key, _, _ in changed] == ["inputGain", "highMidGain", "highGain"]
    assert adopted["analysis"] == [item.model_dump(mode="json") for item in project.analysis]
    # After adoption the record is stale: a second (digit-bearing) id is refused, not applied on top.
    stale = client.post(url, headers=headers, json={"base_revision": adopted["revision"], "variant_id": "trim_plus6"})
    assert stale.status_code == 409


def test_lead_samples_are_selectable_and_disclose_one_request_each(tmp_path):
    judge = FakeJudge(current_gain="less", control_gain="more")
    client, headers, project, renderer = _client(tmp_path, judge)
    assert _plan(client, headers, project).json()["samples"] == ["riff-1", "riff-2"]
    plan = _plan(client, headers, project, samples=["lead-1", "lead-2"]).json()
    assert plan["samples"] == ["lead-1", "lead-2"] and plan["max_new_requests"] == 2
    every = _plan(client, headers, project, samples=["riff-1", "riff-2", "lead-1", "lead-2"]).json()
    assert every["max_new_requests"] == 4 and every["plan_sha256"] != plan["plan_sha256"]
    assert _plan(client, headers, project, samples=["lead-1", "lead-1"]).status_code == 422
    assert _plan(client, headers, project, samples=["riff-3"]).status_code == 422
    assert judge.calls == [] and renderer.renders == 0
    status = _wait(client, headers, _start(client, headers, project, plan).json())
    assert status["status"] == "completed" and len(judge.calls) == 2
    record = status["result"]["refinement"]
    assert {item["sample"] for item in record["judgments"]} == {"lead-1", "lead-2"}
    audio = client.get(f"/api/projects/{project.id}/refinements/{record['id']}/audio/current/lead-2",
                       headers=headers)
    assert audio.status_code == 200


@pytest.mark.parametrize("samples", [["riff-1", "riff-2", "lead-1"], ["riff-1", "riff-2", "lead-1", "lead-2"]])
def test_confirmed_three_and_four_sample_plans_dispatch_within_their_exact_cap(tmp_path, samples):
    judge = FakeJudge(current_gain="less", control_gain="more")
    client, headers, project, _ = _client(tmp_path, judge)
    plan = _plan(client, headers, project, samples=samples).json()
    assert plan["max_new_requests"] == len(samples)
    assert _start(client, headers, project, plan, max_new_requests=5).status_code == 422
    too_small = _start(client, headers, project, plan, max_new_requests=len(samples) - 1)
    assert too_small.status_code != 200 and judge.calls == []
    status = _wait(client, headers, _start(client, headers, project, plan).json())
    assert status["status"] == "completed" and len(judge.calls) == len(samples)
    record = status["result"]["refinement"]
    assert record["confirmed_max_new_requests"] == len(samples)
    assert record["new_generate_requests"] == len(samples)

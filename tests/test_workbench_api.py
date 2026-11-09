import io
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from tone_listener_mcp.listener import Effect, Effects, ListeningResult
from tone_listener_mcp.workbench.adapters.soundshed import SoundshedAdapter
from tone_listener_mcp.workbench.jobs import JobManager
from tone_listener_mcp.workbench.models import AdapterCatalog, ChainDraft, ChainNode
from tone_listener_mcp.workbench.storage import LocalStorage

HOST = "127.0.0.1:8765"
ORIGIN = f"http://{HOST}"


class FakeRenderer:
    available = True
    verified = True
    initialization_error = None

    def __init__(self, path: Path):
        self.binary_path = path

    def initialize(self, cancel_event=None):
        return None

    def verifies_snapshot(self, snapshot):
        return self.verified and snapshot.dsp_conditions.get("state_canary_version") == 1


class FakeAdapter(SoundshedAdapter):
    adapter_id = "fake"

    def __init__(self):
        self.resource_root = Path("/not-used")
        self.database_path = Path("/not-used")
        self._engine_version = "test-engine"

    def catalog(self):
        return AdapterCatalog(
            adapter_id="fake",
            adapter_label="Test adapter",
            engine_version=self.engine_version,
            blocks=[],
            resources=[],
        )

    def default_chain(self, analysis=()):
        return ChainDraft(
            nodes=[
                ChainNode(id="input", kind="input", title="Input"),
                ChainNode(id="gain", kind="gain", title="Gain", params={"gainDb": 0}),
                ChainNode(id="output", kind="output", title="Output"),
            ]
        )

    def validate_chain(self, chain, *, for_render=True):
        errors = []
        for node in chain.nodes:
            if node.kind in ("input", "output", "gain"):
                for key, value in node.params.items():
                    if key != "gainDb" or not -24 <= value <= 24:
                        errors.append("Invalid test gain parameter")
        return errors

    @staticmethod
    def chain_sha256(chain):
        return SoundshedAdapter.chain_sha256(chain)


def _client(tmp_path, monkeypatch, adapter=None, renderer=None, job_manager=None):
    storage = LocalStorage(tmp_path / "workbench")
    binary = tmp_path / "renderer"
    binary.write_text("test")
    app_module = __import__("tone_listener_mcp.workbench.api", fromlist=["create_app"])
    app = app_module.create_app(
        storage=storage,
        adapter=adapter or FakeAdapter(),
        renderer=renderer or FakeRenderer(binary),
        allowed_hosts={HOST},
        job_manager=job_manager or JobManager(workers=1, max_active=3),
        allow_test_client=True,
    )
    client = TestClient(app)
    session = client.get("/api/session", headers={"Host": HOST, "Origin": ORIGIN})
    assert session.status_code == 200
    token = session.json()["token"]
    headers = {"Host": HOST, "Origin": ORIGIN, "X-Workbench-Token": token}
    return client, headers, storage


def _listening():
    effect = Effect(presence="absent", amount="none", notes="No clear effect.")
    return ListeningResult(
        summary="Clear guitar with moderate midrange.",
        gain_level="clean",
        brightness="balanced",
        low_end="controlled",
        mids_character="present",
        pick_attack="clear",
        compression_feel="dynamic",
        effects=Effects(reverb=effect, delay=effect, modulation=effect),
        playing_notes="Single-note line.",
        evidence=[],
        uncertainties=[],
        confidence=0.5,
    ).model_dump(mode="json")


def _profile():
    from tone_listener_mcp.listener import ToneProfile
    effect = {"presence": "absent", "prominence": "none"}
    return ToneProfile(
        guitar_dominance="clear", gain="clean", brightness="balanced",
        bass_level="typical", bass_tightness="neutral", mids="present",
        attack="balanced", compression="dynamic", sustain="typical",
        reverb=effect, delay=effect, modulation=effect, evidence=[], uncertainties=[],
    ).model_dump(mode="json")


def test_local_api_requires_loopback_host_origin_and_session_token(tmp_path, monkeypatch):
    client, headers, _ = _client(tmp_path, monkeypatch)

    assert client.get("/api/health", headers={"Host": HOST}).status_code == 403
    assert client.get(
        "/api/health",
        headers={**headers, "Origin": "https://attacker.example"},
    ).status_code == 403
    assert client.get(
        "/api/health",
        headers={**headers, "Host": "example.com"},
    ).status_code == 403


def test_importing_cli_does_not_create_default_workbench_storage(tmp_path):
    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path)
    environment["XDG_DATA_HOME"] = str(tmp_path / "xdg")
    result = subprocess.run(
        [sys.executable, "-c", "import tone_listener_mcp.workbench.cli"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "Library" / "Application Support" / "Tone Listener MCP" / "workbench").exists()
    assert not (tmp_path / "xdg" / "tone-listener-mcp" / "workbench").exists()


def test_upload_decodes_metadata_and_returns_editable_intervals(tmp_path, monkeypatch):
    client, headers, storage = _client(tmp_path, monkeypatch)
    wav = io.BytesIO()
    sf.write(wav, np.zeros(44_100 * 12, dtype=np.float32), 44_100, format="WAV")

    response = client.post(
        "/api/uploads",
        files={"file": ("../../unsafe.wav", wav.getvalue(), "audio/wav")},
        headers=headers,
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["display_name"] == "unsafe.wav"
    assert result["duration_seconds"] == 12
    assert len(result["intervals"]) == 1
    uploaded = storage.get_upload(result["upload_id"])
    assert uploaded is not None
    assert uploaded.duration_seconds == 12


def test_analysis_requires_external_consent_and_can_save_a_project(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-test-key")
    api_module = __import__("tone_listener_mcp.workbench.api", fromlist=["listen_youtube_tone_profile"])
    monkeypatch.setattr(
        api_module,
        "listen_youtube_tone_profile",
        lambda url, start, end: {
            "model": "offline-test",
            "listening": _profile(),
            "listening_unavailable": None,
        },
    )
    client, headers, storage = _client(tmp_path, monkeypatch)

    body = {
        "kind": "youtube",
        "youtube_url": "https://www.youtube.com/watch?v=401iS5McuoU",
        "duration_seconds": 30,
        "intervals": [{"start_seconds": 0, "end_seconds": 10}],
    }
    refused = client.post("/api/jobs/analysis", json=body, headers=headers)
    assert refused.status_code == 422
    assert "Explicit consent" in refused.text

    response = client.post(
        "/api/jobs/analysis",
        json={**body, "external_consent": True},
        headers=headers,
    )
    assert response.status_code == 202, response.text
    job = response.json()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and job["status"] in ("queued", "running"):
        time.sleep(0.01)
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()

    assert job["status"] == "completed", job
    project_data = job["result"]["project"]
    assert project_data["source"]["selected_intervals"] == body["intervals"]
    assert project_data["target"]["engine_version"] == "test-engine"
    assert storage.load_project(project_data["id"]) is not None


def test_local_analysis_never_exposes_private_upload_path(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    from tone_listener_mcp.workbench import api as api_module
    monkeypatch.setattr(api_module, "listen_tone_profile", lambda clip: {
        "model": "offline-test", "listening": _profile(), "listening_unavailable": None,
    })
    client, headers, _ = _client(tmp_path, monkeypatch)
    wav = io.BytesIO()
    sf.write(wav, np.zeros(44_100, dtype=np.float32), 44_100, format="WAV")
    uploaded = client.post("/api/uploads", files={
        "file": ("reference.wav", wav.getvalue(), "audio/wav"),
    }, headers=headers).json()
    job = client.post("/api/jobs/analysis", json={
        "kind": "audio_file", "upload_id": uploaded["upload_id"],
        "intervals": uploaded["intervals"], "external_consent": True,
    }, headers=headers).json()
    for _ in range(100):
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.01)
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
    assert job["status"] == "completed", job
    assert str(tmp_path) not in str(job)
    metrics = job["result"]["project"]["analysis"][0]["metrics"]
    assert metrics["clip"]["source"] == "Selected audio interval"
    import hashlib

    source = job["result"]["project"]["source"]
    assert source["content_sha256"] == hashlib.sha256(wav.getvalue()).hexdigest()


def test_selecting_stored_interval_changes_intent_without_another_model_call(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    from tone_listener_mcp.workbench import api as api_module
    calls = []

    def listen_interval(url, start, end):
        calls.append(start)
        value = _profile()
        value["gain"] = "clean" if start == 0 else "high-gain"
        return {"model": "offline-test", "listening": value, "listening_unavailable": None}

    monkeypatch.setattr(api_module, "listen_youtube_tone_profile", listen_interval)
    client, headers, _ = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs/analysis", json={
        "kind": "youtube", "youtube_url": "https://www.youtube.com/watch?v=401iS5McuoU",
        "intervals": [{"start_seconds": 0, "end_seconds": 5}, {"start_seconds": 5, "end_seconds": 10}],
        "external_consent": True,
    }, headers=headers).json()
    for _ in range(100):
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.01)
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
    assert job["status"] == "completed", job
    project = job["result"]["project"]
    url = f"/api/projects/{project['id']}/profile"
    body = {"base_revision": project["revision"], "analysis_index": 1}
    response = client.post(url, json=body, headers=headers)
    assert response.status_code == 200
    selected = response.json()
    assert selected["selected_analysis_index"] == 1
    assert "high-gain" in selected["intent"]["summary"]
    assert selected["chain"]["revision"] == project["chain"]["revision"] + 1
    assert calls == [0, 5]
    assert client.post(url, json=body, headers=headers).status_code == 409


def test_analysis_pauses_remaining_intervals_after_first_quota_response(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    from tone_listener_mcp.workbench import api as api_module
    calls = []

    def quota(url, start, end):
        calls.append(start)
        return {"model": "offline-test", "listening": None, "listening_unavailable": "Quota",
                "request_counts": {"generate": 1},
                "failure_info": {"http_status": 429, "category": "daily_quota",
                                 "retry_after_seconds": 60}}

    monkeypatch.setattr(api_module, "listen_youtube_tone_profile", quota)
    client, headers, _ = _client(tmp_path, monkeypatch)
    job = client.post("/api/jobs/analysis", json={
        "kind": "youtube", "youtube_url": "https://www.youtube.com/watch?v=401iS5McuoU",
        "intervals": [{"start_seconds": n, "end_seconds": n + 5} for n in (0, 5, 10)],
        "external_consent": True,
    }, headers=headers).json()
    for _ in range(100):
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.01)
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
    assert job["status"] == "partial"
    assert calls == [0]
    segments = job["result"]["project"]["analysis"]
    assert segments[0]["provider_attempted"] is True
    assert segments[0]["failure_info"]["http_status"] == 429
    for segment in segments[1:]:
        assert segment["provider_attempted"] is False
        assert segment["failure_info"] is None
        assert segment["profile"] is None
        assert segment["unavailable"].startswith("Not attempted:")


def test_api_rejects_oversized_content_length_before_processing(tmp_path, monkeypatch):
    client, headers, _ = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/projects",
        content=b"",
        headers={**headers, "Content-Length": str(3 * 1024 * 1024)},
    )
    assert response.status_code == 413


def _stored_project(storage, adapter):
    from tone_listener_mcp.workbench.adapters.soundshed import new_project
    from tone_listener_mcp.workbench.models import Interval, SourceRef
    project = new_project(
        source=SourceRef(
            kind="youtube", url="https://www.youtube.com/watch?v=401iS5McuoU",
            display_name="Test source", selected_intervals=[
                Interval(start_seconds=0, end_seconds=3),
            ],
        ),
        analysis=[{
            "interval": {"start_seconds": 0, "end_seconds": 3},
            "model": "test", "unavailable": "Analysis unavailable.",
        }],
        intent={"summary": "Manual chain."}, chain=adapter.default_chain(),
        engine_version=adapter.engine_version, name="Test draft",
    )
    storage.save_project(project)
    return project


def test_save_rejects_out_of_range_and_nonfinite_without_echo(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    client, headers, storage = _client(tmp_path, monkeypatch, adapter)
    project = _stored_project(storage, adapter)
    for value in (1e9, float("nan"), float("inf")):
        import json
        nodes = project.chain.model_dump(mode="json")["nodes"]
        nodes[0]["params"]["gainDb"] = value
        response = client.put(
            f"/api/projects/{project.id}/chain",
            content=json.dumps({"base_revision": project.revision, "nodes": nodes}),
            headers={**headers, "Content-Type": "application/json"},
        )
        assert response.status_code == 422, response.text
        assert "NaN" not in response.text
        assert "1000000000" not in response.text
        assert storage.load_project(project.id).revision == project.revision


def test_failed_analysis_export_description_is_not_analysis_derived(tmp_path):
    adapter = FakeAdapter()
    storage = LocalStorage(tmp_path / "storage")
    project = _stored_project(storage, adapter)
    description = adapter._preset_description(project, None)
    assert "reference analysis was unavailable" in description
    assert "Successful analysis model(s): none" in description
    assert "not read or modified" not in description
    assert "analysis-derived" not in adapter._preset(project)["tags"]


def test_unverified_renderer_health_does_not_report_available(tmp_path, monkeypatch):
    client, headers, _ = _client(tmp_path, monkeypatch)
    client.app.state.renderer.verified = False
    client.app.state.renderer.initialization_error = "State sensitivity failed."
    response = client.get("/api/health", headers=headers)
    assert response.json()["renderer_available"] is False
    assert response.json()["renderer_error"] == "State sensitivity failed."
    refused = client.post(
        "/api/projects/not-used/jobs/preview",
        json={"project_revision": 1, "chain_revision": 1, "sample_name": "riff-1"},
        headers=headers,
    )
    assert refused.status_code == 503


def test_save_rejects_unknown_resource_without_echoing_its_id(tmp_path, monkeypatch):
    from tone_listener_mcp.workbench.models import AssetRef
    adapter = SoundshedAdapter(
        resource_root=tmp_path / "resources", database_path=tmp_path / "absent.db",
        engine_version="1.7.3",
    )
    client, headers, storage = _client(tmp_path, monkeypatch, adapter)
    project = _stored_project(storage, FakeAdapter())
    nodes = project.chain.model_dump(mode="json")["nodes"]
    nodes[1] = ChainNode(
        id="pedal", kind="nam_fx", title="Pedal",
        asset=AssetRef(
            resource_type="nam", resource_id="../../outside",
            display_name="Unavailable capture", sha256="a" * 64,
        ),
    ).model_dump(mode="json")
    response = client.put(
        f"/api/projects/{project.id}/chain",
        json={"base_revision": project.revision, "nodes": nodes}, headers=headers,
    )
    assert response.status_code == 422
    assert "../../outside" not in response.text
    assert storage.load_project(project.id).revision == project.revision


def test_bundled_samples_include_credited_leads_and_reject_unknown(tmp_path, monkeypatch):
    client, headers, _ = _client(tmp_path, monkeypatch)
    assert client.get("/api/health", headers=headers).json()["samples"] == ["riff-1", "riff-2", "lead-1", "lead-2"]
    for name in ("riff-1", "lead-1", "lead-2"):
        response = client.get(f"/api/samples/{name}", headers=headers)
        assert response.status_code == 200 and response.headers["content-type"] == "audio/wav"
    assert client.get("/api/samples/riff-3", headers=headers).status_code == 404
    assert client.get("/api/samples/provenance", headers=headers).status_code == 404

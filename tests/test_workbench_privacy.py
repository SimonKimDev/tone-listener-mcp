import logging

from tests.test_workbench_api import _client
from tests.test_workbench_storage import _project
from tone_listener_mcp.workbench.privacy import SafeLogFormatter, redact_paths


def test_redaction_preserves_urls_routes_and_numeric_parameters():
    value = {
        "source": "https://www.youtube.com/watch?v=401iS5McuoU",
        "audio": "/api/projects/example/audio",
        "gain": 6.0, "tolerance": "+/-0.1 dB",
        "error": "Cannot decode /Users/test/private/audio.wav",
        "file_uri": "file:///private/tmp/audio.wav",
        "windows": r"C:\Users\test\audio.wav",
    }
    cleaned = redact_paths(value)
    assert cleaned["source"] == value["source"]
    assert cleaned["audio"] == value["audio"]
    assert cleaned["gain"] == 6
    assert cleaned["tolerance"] == "+/-0.1 dB"
    assert "/Users/" not in str(cleaned)
    assert "/private/" not in str(cleaned)
    assert "Cannot decode" in cleaned["error"]


def test_api_project_and_failure_responses_never_expose_absolute_paths(tmp_path, monkeypatch):
    client, headers, storage = _client(tmp_path, monkeypatch)
    project = _project()
    project.intent.summary = "Failure at /Users/test/private/reference.wav"
    project.analysis[0].metrics = {"clip": {"source": "/private/tmp/upload.bin"}}
    storage.save_project(project)
    client.app.state.renderer.verified = False
    client.app.state.renderer.initialization_error = "Failed at /home/test/private/model.nam"
    for endpoint in ("/api/health", "/api/projects", f"/api/projects/{project.id}"):
        response = client.get(endpoint, headers=headers)
        assert response.status_code == 200
        assert "/Users/" not in response.text
        assert "/private/" not in response.text
        assert "/home/" not in response.text
        assert str(tmp_path) not in response.text


def test_log_formatter_redacts_paths_without_hiding_the_error():
    record = logging.LogRecord(
        "workbench", logging.ERROR, "", 1,
        "Load failed at /Users/test/private/model.nam", (), None,
    )
    message = SafeLogFormatter().format(record)
    assert "Load failed" in message
    assert "/Users/" not in message


def test_synthetic_export_is_not_labelled_as_real_analysis():
    from tests.test_workbench_api import FakeAdapter
    from tests.test_workbench_recipes import profile
    project = _project()
    project.synthetic_fixture = True
    project.analysis[0].unavailable = None
    project.analysis[0].profile = profile()
    project.analysis[0].model = "offline-synthetic-fixture"
    preset = FakeAdapter()._preset(project, None)
    assert "synthetic-fixture" in preset["tags"]
    assert "analysis-derived" not in preset["tags"]
    assert "OFFLINE SYNTHETIC FIXTURE" in preset["description"]
    assert "Successful analysis model(s): none" in preset["description"]

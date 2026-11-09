from test_schemas import undeclared

from tone_listener_mcp.listener import DEFAULT_MODEL
from tone_listener_mcp.schemas import ServerInfo
from tone_listener_mcp.server import server_info


def info() -> dict:
    model = server_info()
    assert isinstance(model, ServerInfo)
    assert undeclared(model) == []
    return model.model_dump()


def test_undeclared_detects_extra_top_level_and_nested_keys():
    valid = server_info().model_dump()
    assert undeclared(ServerInfo.model_validate(valid)) == []
    valid["request_settings"]["unexpected_nested"] = 1
    model = ServerInfo.model_validate({**valid, "unexpected_top": True})
    assert sorted(undeclared(model)) == [
        ".request_settings.unexpected_nested", ".unexpected_top",
    ]


def test_reports_defaults_without_key():
    result = info()
    assert result["model"] == DEFAULT_MODEL
    assert result["listening_available"] is False
    assert "GEMINI_API_KEY is not set" in result["listening_unavailable"]
    assert result["max_seconds"] == 600
    assert result["request_settings"] == {
        "deadline_seconds": 180, "max_concurrent_requests": 2, "cache_entries": 64,
    }
    assert result["configuration_errors"] == []


def test_reports_configured_model_and_never_exposes_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    monkeypatch.setenv("TONE_LISTENER_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("TONE_LISTENER_CACHE_ENTRIES", "0")
    result = info()
    assert result["model"] == "gemini-3.6-flash"
    assert result["listening_available"] is True
    assert result["listening_unavailable"] is None
    assert result["request_settings"]["cache_entries"] == 0
    assert "secret-test-key" not in server_info().model_dump_json()


def test_invalid_settings_disable_listening_with_named_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    monkeypatch.setenv("TONE_LISTENER_MAX_SECONDS", "nan")
    monkeypatch.setenv("TONE_LISTENER_MAX_CONCURRENT_REQUESTS", "0")
    result = info()
    assert result["listening_available"] is False
    assert result["max_seconds"] is None and result["request_settings"] is None
    assert [error.split()[0] for error in result["configuration_errors"]] == [
        "TONE_LISTENER_MAX_SECONDS", "TONE_LISTENER_MAX_CONCURRENT_REQUESTS",
    ]

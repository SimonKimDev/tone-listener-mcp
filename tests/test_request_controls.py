"""Request budgets, failure categories, diagnostics, caching and progress (issue #4)."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import httpx
import httpx2
import numpy as np
import pytest
import soundfile as sf
from google.genai import errors, types
from test_schemas import undeclared

from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.listener import (
    _inference_slots,
    listen,
    listen_comparison,
    request_settings,
)
from tone_listener_mcp.metrics import measure
from tone_listener_mcp.schemas import AnalyzeAudioResponse

PAYLOAD = {
    "summary": "Bright clean guitar",
    "gain_level": "clean",
    "brightness": "bright",
    "low_end": "tight",
    "mids_character": "neutral",
    "pick_attack": "sharp",
    "compression_feel": "dynamic",
    "effects": {
        name: {"presence": "possible", "amount": "subtle", "notes": "uncertain"}
        for name in ("reverb", "delay", "modulation")
    },
    "playing_notes": "single notes",
    "evidence": [{"timestamp": 0.5, "observation": "clear attack"}],
    "uncertainties": ["No exact gear identification"],
    "confidence": 0.7,
}
USAGE = SimpleNamespace(
    prompt_token_count=120, candidates_token_count=80, thoughts_token_count=None,
    cached_content_token_count=None, total_token_count=200,
)


@pytest.fixture
def audio():
    return AudioClip(np.ones((1, 16000), dtype=np.float32) * 0.1, 16000, "synthetic")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mocked = MagicMock()
    mocked.__enter__.return_value = mocked
    mocked.files.upload.return_value = SimpleNamespace(
        name="files/test", uri="https://example.invalid/audio", state=types.FileState.ACTIVE
    )
    mocked.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(PAYLOAD), usage_metadata=USAGE
    )
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: mocked)
    return mocked


def test_success_reports_diagnostics_usage_and_valid_schema(client, audio):
    result = listen(audio)
    diagnostics = result["diagnostics"]
    assert result["failure"] is None
    assert diagnostics["cache"] == "miss"
    assert diagnostics["usage"]["total_token_count"] == 200
    assert diagnostics["usage"]["thoughts_token_count"] is None
    assert diagnostics["prompt_version"]
    assert diagnostics["schema_version"].startswith("ListeningResult-")
    assert diagnostics["deadline_seconds"] == 180
    assert diagnostics["preprocessing"] == [{
        "label": "A", "original_sample_rate": 16000, "original_channels": 1,
        "duration_seconds": 1.0, "upload_sample_rate": 16000, "upload_channels": 1,
    }]
    parsed = AnalyzeAudioResponse.model_validate({"metrics": measure(audio), **result})
    assert undeclared(parsed) == []


def test_per_call_timeout_is_bounded_by_deadline(client, audio, monkeypatch):
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "20")
    listen(audio)
    config = client.models.generate_content.call_args.kwargs["config"]
    assert 1000 <= config.http_options.timeout <= 20000
    upload_config = client.files.upload.call_args.kwargs["config"]
    assert upload_config.http_options.timeout <= 20000


def test_identical_request_is_served_from_cache(client, audio):
    first = listen(audio, "attack")
    second = listen(audio, "attack")
    assert client.models.generate_content.call_count == 1
    assert client.files.upload.call_count == 1
    assert second["diagnostics"]["cache"] == "hit"
    assert second["diagnostics"]["usage"] == first["diagnostics"]["usage"]
    assert second["listening"] == first["listening"]
    second["listening"]["summary"] = "mutated"
    assert listen(audio, "attack")["listening"]["summary"] == PAYLOAD["summary"]


def test_cache_key_covers_question_model_and_audio(client, audio, monkeypatch):
    listen(audio, "attack")
    listen(audio, "low end")
    louder = AudioClip(audio.samples * 2, audio.sample_rate, audio.source)
    listen(louder, "attack")
    monkeypatch.setenv("TONE_LISTENER_MODEL", "other-model")
    listen(audio, "attack")
    assert client.models.generate_content.call_count == 4


def test_cache_can_be_disabled_and_failures_are_not_cached(client, audio, monkeypatch):
    success = client.models.generate_content.return_value
    outcomes = iter([errors.ServerError(503, {"error": {"message": "busy"}})])

    def generate(**kwargs):
        error = next(outcomes, None)
        if error is not None:
            raise error
        return success

    client.models.generate_content.side_effect = generate
    assert listen(audio)["failure"]["category"] == "capacity"
    assert listen(audio)["listening"] is not None
    monkeypatch.setenv("TONE_LISTENER_CACHE_ENTRIES", "0")
    result = listen(audio)
    assert result["diagnostics"]["cache"] == "disabled"
    assert client.models.generate_content.call_count == 3


@pytest.mark.parametrize(
    ("error", "category", "status", "retryable"),
    [
        (errors.ClientError(400, {"error": {"message": "secret"}}), "invalid_request", 400, False),
        (errors.ClientError(401, {"error": {"message": "secret"}}), "authentication", 401, False),
        (errors.ClientError(403, {"error": {"message": "secret"}}), "permission", 403, False),
        (errors.ClientError(404, {"error": {"message": "secret"}}), "model_unavailable", 404, False),
        (errors.ClientError(429, {"error": {"message": "secret"}}), "quota", 429, True),
        (errors.ServerError(503, {"error": {"message": "secret"}}), "capacity", 503, True),
        (httpx.ConnectError("secret"), "network", None, True),
        (httpx.ReadTimeout("secret"), "deadline", None, True),
        # google-genai 3.x uses the httpx2 backend by default; its errors are unrelated to httpx.
        (httpx2.ConnectError("secret"), "network", None, True),
        (httpx2.ReadTimeout("secret"), "deadline", None, True),
    ],
)
def test_failure_categories_never_echo_provider_text(
    client, audio, error, category, status, retryable
):
    client.models.generate_content.side_effect = error
    result = listen(audio)
    assert result["failure"]["category"] == category
    assert result["failure"]["http_status"] == status
    assert result["failure"]["retryable"] is retryable
    assert result["listening_failure_stage"] == "generate"
    assert "secret" not in json.dumps(result)
    client.files.delete.assert_called_once()


@pytest.mark.parametrize(
    "delete_error",
    [httpx2.ReadTimeout("secret"), httpx2.ConnectError("secret"), RuntimeError("secret")],
)
def test_failed_delete_does_not_skip_remaining_cleanup(client, audio, delete_error):
    client.files.upload.side_effect = [
        SimpleNamespace(name=name, uri=f"https://example.invalid/{name}",
                        state=types.FileState.ACTIVE)
        for name in ("files/A", "files/B")
    ]
    client.models.generate_content.side_effect = httpx2.ReadTimeout("secret")
    client.files.delete.side_effect = [delete_error, None]
    result = listen_comparison(audio, audio)
    assert result["failure"]["category"] == "deadline"
    assert [call.kwargs["name"] for call in client.files.delete.call_args_list] == [
        "files/A", "files/B"
    ]
    assert result["cleanup_warnings"] == [
        f"Remote upload cleanup failed ({type(delete_error).__name__})"
    ]
    assert "secret" not in json.dumps(result)


def test_invalid_response_category(client, audio):
    client.models.generate_content.return_value.text = "{}"
    result = listen(audio)
    assert result["failure"]["category"] == "invalid_response"
    assert result["listening_failure_stage"] == "validate"


def test_missing_key_is_not_configured(audio, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = listen(audio)
    assert result["failure"] == {
        "category": "not_configured", "http_status": None, "retryable": False,
        "suggestion": result["failure"]["suggestion"],
        "retry_after_seconds": None, "retry_after_invalid": False,
        "daily_request_limit": None, "daily_token_limit": None,
    }
    assert result["diagnostics"]["cache"] == "not_used"


def test_busy_slots_fail_with_queue_stage(client, audio, monkeypatch):
    monkeypatch.setenv("TONE_LISTENER_MAX_CONCURRENT_REQUESTS", "1")
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "1")
    slots = _inference_slots(1)
    assert slots.acquire(timeout=1)
    try:
        result = listen(audio)
    finally:
        slots.release()
    assert result["failure"]["category"] == "busy"
    assert result["listening_failure_stage"] == "queue"
    client.files.upload.assert_not_called()
    assert listen(audio)["listening"] is not None


def test_concurrency_limit_is_enforced(client, audio, monkeypatch):
    monkeypatch.setenv("TONE_LISTENER_MAX_CONCURRENT_REQUESTS", "1")
    monkeypatch.setenv("TONE_LISTENER_CACHE_ENTRIES", "0")
    active = []
    peak = []
    lock = threading.Lock()

    def generate(**kwargs):
        with lock:
            active.append(1)
            peak.append(len(active))
        threading.Event().wait(0.05)
        with lock:
            active.pop()
        return SimpleNamespace(text=json.dumps(PAYLOAD), usage_metadata=None)

    client.models.generate_content.side_effect = generate
    threads = [threading.Thread(target=listen, args=(audio,)) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(peak) == 1
    assert client.models.generate_content.call_count == 3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "0"),
        ("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "nan"),
        ("TONE_LISTENER_MAX_CONCURRENT_REQUESTS", "1.5"),
        ("TONE_LISTENER_CACHE_ENTRIES", "-1"),
    ],
)
def test_invalid_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        request_settings()


def test_progress_reports_stages_and_ignores_callback_errors(client, audio):
    seen = []
    listen(audio, progress=lambda step, total, message: seen.append((step, total, message)))
    assert [message for _, _, message in seen] == [
        "Prepared audio", "Uploaded clip A", "Generated listening response",
        "Validated listening response",
    ]
    assert [step for step, _, _ in seen] == [1, 2, 3, 4]
    assert {total for _, total, _ in seen} == {4}

    def broken(*args):
        raise RuntimeError("client went away")

    assert listen(audio, "other", progress=broken)["listening"] is not None
    cached = []
    listen(audio, "other", progress=lambda *args: cached.append(args))
    assert cached[-1] == (4, 4, "Served from cache")


def test_mcp_progress_notifications(client, tmp_path):
    from mcp.client.client import Client

    from tone_listener_mcp.server import mcp

    path = tmp_path / "tone.wav"
    sf.write(path, 0.25 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000), 16000)
    received = []

    async def on_progress(progress, total, message):
        received.append((progress, total, message))

    async def exercise():
        async with Client(mcp) as session:
            result = await session.call_tool(
                "analyze_audio", {"path": str(path)}, progress_callback=on_progress
            )
            assert not result.is_error
            assert result.structured_content["diagnostics"]["cache"] == "miss"
            tools = await session.list_tools()
            schema = next(tool for tool in tools.tools if tool.name == "analyze_audio")
            assert "ctx" not in schema.input_schema["properties"]

    anyio.run(exercise)
    assert [message for _, _, message in received][-1] == "Validated listening response"

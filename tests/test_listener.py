import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from google.genai import errors, types

from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.listener import listen


@pytest.fixture
def audio():
    return AudioClip(np.ones((1, 16000), dtype=np.float32) * 0.1, 16000, "synthetic")


@pytest.fixture
def listening_payload():
    effect = {"presence": "possible", "amount": "subtle", "notes": "uncertain"}
    return {
        "summary": "Bright clean guitar",
        "gain_level": "clean",
        "brightness": "bright",
        "low_end": "tight",
        "mids_character": "neutral",
        "pick_attack": "sharp",
        "compression_feel": "dynamic",
        "effects": {name: effect for name in ("reverb", "delay", "modulation")},
        "playing_notes": "single notes",
        "evidence": [{"timestamp": 0.5, "observation": "clear attack"}],
        "uncertainties": ["No exact gear identification"],
        "confidence": 0.7,
    }


@pytest.fixture
def client(monkeypatch, listening_payload):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mocked = MagicMock()
    mocked.__enter__.return_value = mocked
    mocked.files.upload.return_value = SimpleNamespace(
        name="files/test", uri="https://example.invalid/audio", state=types.FileState.ACTIVE
    )
    mocked.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(listening_payload)
    )
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: mocked)
    return mocked


def test_missing_key(monkeypatch, audio):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = listen(audio)
    assert result["listening"] is None
    assert "GEMINI_API_KEY" in result["listening_unavailable"]


def test_success_and_cleanup(client, audio):
    result = listen(audio, "How bright is it?")
    assert result["listening"]["gain_level"] == "clean"
    assert result["listening_unavailable"] is None
    client.files.delete.assert_called_once_with(name="files/test")
    path = client.files.upload.call_args.kwargs["file"]
    assert not path.exists()
    assert "How bright" in str(client.models.generate_content.call_args)


@pytest.mark.parametrize("failure", [
    errors.ClientError(403, {"error": {"message": "secret-key must not leak"}}),
    ValueError("malformed secret-key"),
])
def test_api_failure_preserves_fallback(client, audio, failure):
    client.models.generate_content.side_effect = failure
    result = listen(audio)
    assert result["listening"] is None
    assert "failed" in result["listening_unavailable"]
    assert "secret-key" not in json.dumps(result)
    client.files.delete.assert_called_once()


@pytest.mark.parametrize("text", ["not json", "{}", '{"confidence": 2}'])
def test_invalid_response(client, audio, text):
    client.models.generate_content.return_value.text = text
    assert listen(audio)["listening_unavailable"] is not None


def test_cleanup_failure_is_explicit(client, audio):
    client.files.delete.side_effect = ValueError("failed cleanup")
    result = listen(audio)
    assert result["listening"] is not None
    assert result["cleanup_warnings"]


def test_client_initialization_failure(monkeypatch, audio):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    constructor = MagicMock(side_effect=ValueError("invalid client configuration"))
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", constructor)
    assert listen(audio)["listening_unavailable"] is not None


def test_analyze_metrics_without_key(tmp_path, monkeypatch):
    import soundfile as sf

    from tone_listener_mcp.server import analyze_audio

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    path = tmp_path / "tone.wav"
    sf.write(path, np.ones(16000) * 0.1, 16000)
    result = analyze_audio(str(path), end=0.5)
    assert result["metrics"]["clip"]["duration_seconds"] == 0.5
    assert result["metrics"]["level"]["rms_dbfs"] is not None
    assert result["listening_unavailable"]

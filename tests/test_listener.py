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


def test_antiphase_is_explicit(client):
    tone = np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000).astype(np.float32)
    audio = AudioClip(np.stack((tone, -tone)), 16000, "private-local-path")
    result = listen(audio)
    assert "Stereo downmix cancels" in result["listening_unavailable"]
    client.files.upload.assert_not_called()


def test_timestamp_and_confidence_validation(client, audio, listening_payload):
    listening_payload["evidence"][0]["timestamp"] = 1.5
    client.models.generate_content.return_value.text = json.dumps(listening_payload)
    assert listen(audio)["listening_unavailable"]
    listening_payload["evidence"][0]["timestamp"] = 0.5
    listening_payload["confidence"] = 1.1
    client.models.generate_content.return_value.text = json.dumps(listening_payload)
    assert listen(audio)["listening_unavailable"]


def test_upload_failure_and_processing(client, audio):
    client.files.upload.side_effect = OSError("failed upload")
    result = listen(audio)
    assert result["listening_unavailable"]
    assert result["listening_failure_stage"] == "upload"
    client.files.delete.assert_not_called()
    client.files.upload.side_effect = None
    client.files.upload.return_value.state = types.FileState.PROCESSING
    client.files.get.return_value = SimpleNamespace(
        name="files/test", uri="https://example.invalid/audio", state=types.FileState.ACTIVE
    )
    assert listen(audio)["listening"] is not None
    client.files.get.assert_called_once()


def test_upload_attenuates_without_sending_paths(client, audio):
    import soundfile as sf

    levels = []

    def inspect_upload(**kwargs):
        samples, rate = sf.read(kwargs["file"])
        levels.append(float(np.max(np.abs(samples))))
        assert rate == 16000
        return SimpleNamespace(
            name="files/test", uri="https://example.invalid/audio", state=types.FileState.ACTIVE
        )

    client.files.upload.side_effect = inspect_upload
    audio = AudioClip(audio.samples * 20, audio.sample_rate, "/private/recordings/secret.wav")
    result = listen(audio)
    assert result["upload_gain_db"] < 0
    assert max(levels) <= 1
    assert "/private/recordings" not in str(client.models.generate_content.call_args)


def test_uncertain_gain_and_system_instruction(client, audio, listening_payload):
    listening_payload["gain_level"] = "uncertain"
    listening_payload["confidence"] = 0
    client.models.generate_content.return_value.text = json.dumps(listening_payload)
    result = listen(audio)
    assert result["listening"]["gain_level"] == "uncertain"
    config = client.models.generate_content.call_args.kwargs["config"]
    assert "Never claim an exact amp" in config.system_instruction
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None


def test_processing_timeout_cleans_up(client, audio, monkeypatch):
    client.files.upload.return_value.state = types.FileState.PROCESSING
    monkeypatch.setattr("tone_listener_mcp.listener.time.monotonic", MagicMock(side_effect=[0, 31]))
    result = listen(audio)
    assert "TimeoutError" in result["listening_unavailable"]
    client.files.delete.assert_called_once_with(name="files/test")
    client.models.generate_content.assert_not_called()


def test_failed_upload_state(client, audio):
    client.files.upload.return_value.state = types.FileState.FAILED
    assert listen(audio)["listening_unavailable"]
    client.files.delete.assert_called_once_with(name="files/test")
    client.models.generate_content.assert_not_called()


def test_prepared_silent_comparison_is_explicit(client, audio):
    from tone_listener_mcp.listener import listen_comparison

    silence = AudioClip(np.zeros((1, 16000)), 16000, "synthetic")
    result = listen_comparison(audio, silence)
    assert not result["upload_loudness_match"]["applied"]
    assert result["upload_loudness_match"]["additional_gain_db_applied_to_b"] is None
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert "matching is undefined" in contents[0]


def test_youtube_uses_bounded_url_without_files(client):
    from tone_listener_mcp.listener import listen_youtube

    result = listen_youtube("https://youtu.be/9hE5-98ZeCg?si=tracking", 12, 20, "Describe attack")
    assert result["listening"] is not None
    assert result["metrics"] is None
    assert result["input_transport"] == "youtube_url"
    assert result["source"]["url"] == "https://www.youtube.com/watch?v=9hE5-98ZeCg"
    contents = client.models.generate_content.call_args.kwargs["contents"]
    part = contents[1]
    assert part.file_data.file_uri == result["source"]["url"]
    assert part.video_metadata.start_offset == "12s"
    assert part.video_metadata.end_offset == "20s"
    config = client.models.generate_content.call_args.kwargs["config"]
    assert "ONLY audible evidence" in config.system_instruction
    assert "Ignore video images" in config.system_instruction
    client.files.upload.assert_not_called()
    client.files.delete.assert_not_called()


def test_youtube_no_key_has_no_metrics(monkeypatch):
    from tone_listener_mcp.listener import listen_youtube

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = listen_youtube("https://www.youtube.com/watch?v=9hE5-98ZeCg", start=10)
    assert result["source"]["end_seconds"] == 610
    assert result["metrics"] is None
    assert "no local metrics" in result["listening_unavailable"]


@pytest.mark.parametrize("url", [
    "http://youtu.be/9hE5-98ZeCg", "https://youtube.com.evil.test/watch?v=9hE5-98ZeCg",
    "https://www.youtube.com/playlist?list=abc", "file:///tmp/recording.wav",
    "https://user@youtu.be/9hE5-98ZeCg", "https://youtu.be:443/9hE5-98ZeCg",
    "https://www.youtube.com/watch?v=9hE5-98ZeCg&v=other",
])
def test_youtube_rejects_non_video_inputs(url):
    from tone_listener_mcp.listener import listen_youtube

    with pytest.raises(ValueError):
        listen_youtube(url)


@pytest.mark.parametrize("start,end", [
    (-1, None), (0, 0), (0, 601), (float("nan"), None), (0, float("inf")),
])
def test_youtube_interval_bounds(start, end):
    from tone_listener_mcp.listener import listen_youtube

    with pytest.raises(ValueError):
        listen_youtube("https://youtu.be/9hE5-98ZeCg", start, end)


def test_youtube_provider_error_is_explicit(client):
    from tone_listener_mcp.listener import listen_youtube

    client.models.generate_content.side_effect = errors.ClientError(
        401, {"error": {"message": "secret-key"}}
    )
    result = listen_youtube("https://youtu.be/9hE5-98ZeCg")
    assert result["listening"] is None
    assert result["metrics"] is None
    assert result["listening_failure_stage"] == "generate"
    assert "secret-key" not in json.dumps(result)


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=401iS5McuoU",
    "https://www.youtube.com/shorts/401iS5McuoU",
    "https://www.youtube.com/embed/401iS5McuoU",
    "https://www.youtube.com/live/401iS5McuoU",
    "https://youtu.be/401iS5McuoU",
])
def test_supported_youtube_forms(url, monkeypatch):
    from tone_listener_mcp.listener import listen_youtube

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = listen_youtube(url, 0, 30)
    assert result["source"]["url"] == "https://www.youtube.com/watch?v=401iS5McuoU"


def test_youtube_evidence_bounds(client, listening_payload):
    from tone_listener_mcp.listener import listen_youtube

    listening_payload["evidence"][0]["timestamp"] = 31
    client.models.generate_content.return_value.text = json.dumps(listening_payload)
    result = listen_youtube("https://youtu.be/401iS5McuoU", 0, 30)
    assert result["listening"] is None
    assert result["listening_failure_stage"] == "validate"

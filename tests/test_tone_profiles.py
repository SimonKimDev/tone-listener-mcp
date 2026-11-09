"""Public categorical listening API; SDK mocked, never a provider request."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from google.genai import errors, types
from pydantic import ValidationError
from test_request_controls import PAYLOAD
from test_schemas import COMPARISON, undeclared

from tone_listener_mcp import listener
from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.schemas import AnalyzeYouTubeResponse

PROFILE = {
    "guitar_dominance": "clear", "gain": "clean", "brightness": "balanced",
    "bass_level": "typical", "bass_tightness": "tight", "mids": "neutral",
    "attack": "sharp", "compression": "dynamic", "sustain": "typical",
    **{name: {"presence": "absent", "prominence": "none"}
       for name in ("reverb", "delay", "modulation")},
    "modulation_type": "none",
    "evidence": [{"timestamp": 0.5, "observation": "Clear guitar attack"}],
    "uncertainties": [],
}
YOUTUBE = "https://youtu.be/9hE5-98ZeCg"
ZERO_COUNTS = {"generate": 0, "upload": 0, "poll_upload": 0, "delete_upload": 0}
LOCAL_COUNTS = {"generate": 1, "upload": 1, "poll_upload": 0, "delete_upload": 1}


@pytest.fixture
def audio():
    return AudioClip(np.ones((1, 16000), dtype=np.float32) * 0.1, 16000, "private-path")


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-only")
    client = MagicMock()
    client.__enter__.return_value = client
    client.files.upload.return_value = SimpleNamespace(
        name="files/test", uri="https://example.invalid/audio", state=types.FileState.ACTIVE
    )
    client.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(PROFILE), usage_metadata=None
    )
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(listener.genai, "Client", constructor)
    return client, constructor


@pytest.mark.parametrize("youtube", [False, True])
def test_profiles_are_fresh_validated_and_counted_by_default(sdk, audio, youtube):
    client, constructor = sdk
    seen = []

    def run():
        if youtube:
            return listener.listen_youtube_tone_profile(
                YOUTUBE, 12, 13, progress=lambda *args: seen.append(args)
            )
        return listener.listen_tone_profile(audio, progress=lambda *args: seen.append(args))

    for _ in range(2):
        result = run()
        assert result["failure"] is None
        assert result["listening"] == PROFILE
        assert result["request_counts"] == (
            {**ZERO_COUNTS, "generate": 1} if youtube else LOCAL_COUNTS
        )
        assert result["diagnostics"]["cache"] == "disabled"
        assert result["diagnostics"]["schema_version"].startswith("ToneProfile-")
        assert listener.ToneProfile.model_validate(result["listening"]).gain == "clean"
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_schema is listener.ToneProfile
    assert listener.TONE_PROFILE_INSTRUCTIONS in config.system_instruction
    assert config.temperature is None and config.top_p is None and config.top_k is None
    assert "private-path" not in str(client.models.generate_content.call_args)
    assert constructor.call_args.kwargs["http_options"].retry_options.attempts == 1
    assert client.models.generate_content.call_count == 2
    assert seen[-1][-1] == "Validated listening response"
    if youtube:
        assert result["source"] == {
            "url": "https://www.youtube.com/watch?v=9hE5-98ZeCg",
            "start_seconds": 12, "end_seconds": 13,
        }
        assert result["metrics"] is None and result["input_transport"] == "youtube_url"
        assert "ONLY audible evidence" in config.system_instruction
        client.files.upload.assert_not_called()
    else:
        assert "16 kHz mono" in config.system_instruction
        assert client.files.upload.call_count == client.files.delete.call_count == 2


def test_cache_bypass_neither_reads_nor_writes_and_hits_have_zero_counts(sdk, audio):
    client, _ = sdk
    original = listener.listen_tone_profile(audio, use_cache=True)
    client.models.generate_content.return_value.text = json.dumps({**PROFILE, "gain": "crunch"})
    assert listener.listen_tone_profile(audio)["listening"]["gain"] == "crunch"
    cached = listener.listen_tone_profile(audio, use_cache=True)
    assert cached["listening"] == original["listening"]
    assert cached["diagnostics"]["cache"] == "hit"
    assert cached["request_counts"] == ZERO_COUNTS
    listener.clear_listening_cache()
    listener.listen_tone_profile(audio)
    assert listener.listen_tone_profile(audio, use_cache=True)["diagnostics"]["cache"] == "miss"
    assert client.models.generate_content.call_count == 4


def test_cache_separates_schema_and_custom_instructions(sdk, audio):
    client, _ = sdk
    listener.listen_tone_profile(audio, use_cache=True)
    client.models.generate_content.return_value.text = json.dumps(PAYLOAD)
    assert listener.listen(audio)["diagnostics"]["cache"] == "miss"
    client.models.generate_content.return_value.text = json.dumps(PROFILE)
    for instructions in ("Classify guitar only", "Classify guitar attack only"):
        assert listener._request(
            [audio], "same prompt", listener.ToneProfile, instructions=instructions,
        )["diagnostics"]["cache"] == "miss"
    assert client.models.generate_content.call_count == 4


@pytest.mark.parametrize("error", [
    errors.ServerError(503, {"error": {"message": "secret"}}),
    httpx.ConnectError("secret"),
    httpx.ReadTimeout("secret"),
])
def test_single_attempt_youtube_profile_never_retries(sdk, monkeypatch, error):
    client, _ = sdk
    sleep = MagicMock()
    monkeypatch.setattr(listener.time, "sleep", sleep)
    client.models.generate_content.side_effect = error
    for _ in range(2):
        result = listener.listen_youtube_tone_profile(YOUTUBE, 0, 1)
        assert result["request_counts"] == {**ZERO_COUNTS, "generate": 1}
        assert result["listening"] is None
        assert "secret" not in json.dumps(result)
    assert client.models.generate_content.call_count == 2
    sleep.assert_not_called()


def test_legacy_youtube_retry_counts_actual_attempts(sdk, monkeypatch):
    client, _ = sdk
    monkeypatch.setattr(listener.time, "sleep", lambda _: None)
    response = SimpleNamespace(text=json.dumps(PAYLOAD), usage_metadata=None)
    client.models.generate_content.side_effect = [
        errors.ServerError(503, {"error": {"message": "busy"}}), response
    ]
    result = listener.listen_youtube(YOUTUBE, 0, 1)
    assert result["failure"] is None
    assert result["request_counts"] == {**ZERO_COUNTS, "generate": 2}


@pytest.mark.parametrize("kind", ["local", "comparison", "youtube", "youtube_comparison"])
def test_legacy_listeners_accept_per_call_controls(sdk, audio, kind):
    client, _ = sdk
    client.models.generate_content.return_value.text = json.dumps(
        COMPARISON if "comparison" in kind else PAYLOAD
    )
    options = {"use_cache": False, "request_attempts": 1}
    calls = {
        "local": lambda: listener.listen(audio, **options),
        "comparison": lambda: listener.listen_comparison(audio, audio, **options),
        "youtube": lambda: listener.listen_youtube(YOUTUBE, 0, 1, **options),
        "youtube_comparison": lambda: listener.listen_youtube_comparison(
            YOUTUBE, 0, 1, audio, **options
        ),
    }
    calls[kind]()
    result = calls[kind]()
    assert result["failure"] is None
    assert result["diagnostics"]["cache"] == "disabled"
    assert result["request_counts"]["generate"] == 1
    assert client.models.generate_content.call_count == 2
    if kind == "youtube":
        assert undeclared(AnalyzeYouTubeResponse.model_validate(result)) == []


def test_counts_include_polling_and_failed_cleanup(sdk, audio, monkeypatch):
    client, _ = sdk
    monkeypatch.setattr(listener.time, "sleep", lambda _: None)
    client.files.upload.return_value.state = types.FileState.PROCESSING
    client.files.get.return_value = SimpleNamespace(
        name="files/test", uri="https://example.invalid/audio", state=types.FileState.ACTIVE
    )
    client.files.delete.side_effect = RuntimeError("secret")
    result = listener.listen_tone_profile(audio)
    assert result["listening"] is not None
    assert result["request_counts"] == {**LOCAL_COUNTS, "poll_upload": 1}
    assert result["cleanup_warnings"] == ["Remote upload cleanup failed (RuntimeError)"]
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize(("stage", "expected"), [
    ("upload", {**ZERO_COUNTS, "upload": 1}),
    ("poll_upload", {**ZERO_COUNTS, "upload": 1, "poll_upload": 1, "delete_upload": 1}),
    ("generate", LOCAL_COUNTS),
])
def test_counts_include_failed_operations(sdk, audio, monkeypatch, stage, expected):
    client, _ = sdk
    monkeypatch.setattr(listener.time, "sleep", lambda _: None)
    if stage == "poll_upload":
        client.files.upload.return_value.state = types.FileState.PROCESSING
    operation = {
        "upload": client.files.upload, "poll_upload": client.files.get,
        "generate": client.models.generate_content,
    }[stage]
    operation.side_effect = errors.ClientError(429, {"error": {"message": "secret"}})
    result = listener.listen_tone_profile(audio)
    assert result["failure"]["category"] == "quota"
    assert result["request_counts"] == expected


def test_no_key_and_preparation_failure_have_zero_counts(sdk, audio, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert listener.listen_tone_profile(audio)["request_counts"] == ZERO_COUNTS
    monkeypatch.setenv("GEMINI_API_KEY", "mock-only")
    opposite = AudioClip(np.vstack((audio.samples, -audio.samples)), 16000, "synthetic")
    result = listener.listen_tone_profile(opposite)
    assert result["failure"]["category"] == "audio_preparation"
    assert result["request_counts"] == ZERO_COUNTS


def test_profile_preserves_queue_budget_and_never_invokes_sdk_when_busy(sdk, audio, monkeypatch):
    _, constructor = sdk
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "1")
    monkeypatch.setenv("TONE_LISTENER_MAX_CONCURRENT_REQUESTS", "1")
    slots = listener._inference_slots(1)
    assert slots.acquire(blocking=False)
    try:
        result = listener.listen_tone_profile(audio)
    finally:
        slots.release()
    assert result["failure"]["category"] == "busy"
    assert result["listening_failure_stage"] == "queue"
    assert result["request_counts"] == ZERO_COUNTS
    constructor.assert_not_called()
    assert listener.listen_tone_profile(audio)["listening"] is not None


def test_profile_does_not_accept_or_cache_a_late_response(sdk, audio, monkeypatch):
    client, _ = sdk
    now = [0.0]
    monkeypatch.setattr(listener.time, "monotonic", lambda: now[0])

    def late_response(**kwargs):
        now[0] = 181.0
        return SimpleNamespace(text=json.dumps(PROFILE), usage_metadata=None)

    client.models.generate_content.side_effect = late_response
    result = listener.listen_tone_profile(audio, use_cache=True)
    assert result["failure"]["category"] == "deadline"
    assert result["listening"] is None
    assert result["request_counts"] == LOCAL_COUNTS
    assert not result["cleanup_warnings"]
    client.models.generate_content.side_effect = None
    assert listener.listen_tone_profile(audio, use_cache=True)["diagnostics"]["cache"] == "miss"


def test_profile_returns_shared_quota_metadata(sdk, audio):
    from test_failure_metadata import quota_error

    client, _ = sdk
    error = quota_error(
        [{"quotaId": "GenerateRequestsPerDay", "quotaValue": "20"}], header="60"
    )
    client.models.generate_content.side_effect = error
    result = listener.listen_tone_profile(audio)
    assert result["failure"] == listener.failure_info(error, "generate")
    assert result["failure"]["category"] == "daily_quota"
    assert result["failure"]["daily_request_limit"] == 20
    assert result["failure"]["retry_after_seconds"] == 60
    assert result["request_counts"] == LOCAL_COUNTS
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("youtube", [False, True])
@pytest.mark.parametrize("timestamp", [-0.1, 1.01, float("inf")])
def test_profile_evidence_must_fit_selected_duration(sdk, audio, youtube, timestamp):
    client, _ = sdk
    payload = {**PROFILE, "evidence": [{"timestamp": timestamp, "observation": "attack"}]}
    client.models.generate_content.return_value.text = json.dumps(payload)
    result = (
        listener.listen_youtube_tone_profile(YOUTUBE, 12, 13) if youtube
        else listener.listen_tone_profile(audio)
    )
    assert result["listening"] is None
    assert result["failure"]["category"] == "invalid_response"
    assert result["listening_failure_stage"] == "validate"


@pytest.mark.parametrize("changes", [
    {"gain": "unknown"}, {"guitar_dominance": "loud"},
    {"reverb": {"presence": "absent", "prominence": "subtle"}},
    {"delay": {"presence": "present", "prominence": "none"}},
])
def test_profile_rejects_invalid_axes_and_incoherent_effects(changes):
    with pytest.raises(ValidationError):
        listener.ToneProfile.model_validate({**PROFILE, **changes})


@pytest.mark.parametrize("attempts", [0, -1, 3, 1.0, True, "1"])
def test_invalid_attempt_caps_are_rejected_before_io(sdk, audio, attempts):
    client, constructor = sdk
    with pytest.raises(ValueError, match="request_attempts"):
        listener.listen_tone_profile(audio, request_attempts=attempts)
    client.files.upload.assert_not_called()
    constructor.assert_not_called()


def test_invalid_cache_option_is_rejected(sdk, audio):
    with pytest.raises(TypeError, match="use_cache"):
        listener.listen_tone_profile(audio, use_cache="false")


def test_public_youtube_input_validates_without_io():
    canonical, start, end, part = listener.youtube_input(YOUTUBE + "?si=tracking", 12, 13)
    assert canonical == "https://www.youtube.com/watch?v=9hE5-98ZeCg"
    assert (start, end) == (12, 13)
    assert part.video_metadata.start_offset == "12s"
    assert part.video_metadata.end_offset == "13s"
    with pytest.raises(ValueError):
        listener.youtube_input("https://example.com/secret", 0, 1)
    with pytest.raises(ValueError):
        listener.youtube_input(YOUTUBE, 1, 0)

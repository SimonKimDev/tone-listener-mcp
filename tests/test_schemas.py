"""Every key produced by the tools must be declared in the response models."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pydantic
import pytest
from google.genai import types

from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.comparison import compare
from tone_listener_mcp.listener import ComparisonResult, listen, listen_youtube
from tone_listener_mcp.metrics import measure
from tone_listener_mcp.schemas import (
    AnalyzeAudioResponse,
    AnalyzeYouTubeResponse,
    AudioMetrics,
    CompareResponse,
)

ADJUSTMENT = {
    "family": "ambience", "direction": "reduce reverb amount", "priority": 1,
    "observation": "B has a longer audible tail on sustained notes",
    "alternative_explanation": "Room sound in the B recording",
    "listen_for": "Tail length on the final chord",
}
COMPARISON = {
    "differences": ["B more ambient"], "brighter": "similar", "more_saturated": "similar",
    "more_compressed": "similar", "more_ambient": "B", "suggested_adjustments": [ADJUSTMENT],
    "uncertainties": [], "confidence": 0.7,
}
LISTENING = {
    "summary": "Clean tone", "gain_level": "clean", "brightness": "neutral",
    "low_end": "tight", "mids_character": "even", "pick_attack": "soft",
    "compression_feel": "open",
    "effects": {name: {"presence": "absent", "amount": "none", "notes": ""}
                for name in ("reverb", "delay", "modulation")},
    "playing_notes": "single notes", "evidence": [{"timestamp": 0.1, "observation": "attack"}],
    "uncertainties": [], "confidence": 0.5,
}


def undeclared(value, path="") -> list[str]:
    if isinstance(value, pydantic.BaseModel):
        found = [f"{path}.{key}" for key in (value.model_extra or {})]
        for name in type(value).model_fields:
            found += undeclared(getattr(value, name), f"{path}.{name}")
        return found
    if isinstance(value, dict):
        return [item for key, inner in value.items() for item in undeclared(inner, f"{path}.{key}")]
    if isinstance(value, list):
        return [item for inner in value for item in undeclared(inner, path)]
    return []


def clip(samples, rate=48000):
    return AudioClip(np.atleast_2d(samples).astype(np.float32), rate, "synthetic")


def tone(rate=48000, seconds=1.0):
    t = np.arange(round(rate * seconds)) / rate
    return 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)


def mock_gemini(monkeypatch, payload):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.__enter__.return_value = client
    client.files.upload.side_effect = lambda **kwargs: SimpleNamespace(
        name="files/x", uri="https://example.invalid/x", state=types.FileState.ACTIVE
    )
    client.models.generate_content.return_value.text = json.dumps(payload)
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: client)


@pytest.mark.parametrize("samples", [tone(), np.stack((tone(), 0.5 * tone())), np.zeros(4800)])
def test_metrics_schema(samples):
    assert undeclared(AudioMetrics.model_validate(measure(clip(samples)))) == []


def test_compare_schema_without_listening():
    a = clip(np.stack((tone(), tone())))
    bright = tone() + 0.3 * np.sin(2 * np.pi * 2500 * np.arange(48000) / 48000)
    b = clip(np.stack((bright, 0.2 * bright)))
    result = CompareResponse.model_validate(compare(a, b))
    assert undeclared(result) == []
    assert result.eq_direction_hints
    assert result.listening is None
    mixed = CompareResponse.model_validate(compare(a, clip(tone(16000), 16000)))
    assert undeclared(mixed) == []
    assert mixed.eq_direction_hints == []


def test_compare_schema_with_structured_adjustments(monkeypatch):
    mock_gemini(monkeypatch, COMPARISON)
    result = CompareResponse.model_validate(compare(clip(tone()), clip(0.5 * tone())))
    assert undeclared(result) == []
    adjustment = result.listening.suggested_adjustments[0]
    assert adjustment.family == "ambience" and adjustment.priority == 1


def test_free_text_adjustments_are_rejected():
    with pytest.raises(pydantic.ValidationError):
        ComparisonResult.model_validate({**COMPARISON, "suggested_adjustments": ["Cut treble"]})


def test_analyze_schemas(monkeypatch):
    no_key = AnalyzeAudioResponse.model_validate(
        {"metrics": measure(clip(tone())), **listen(clip(tone()))}
    )
    assert undeclared(no_key) == []
    youtube = AnalyzeYouTubeResponse.model_validate(
        listen_youtube("https://youtu.be/9hE5-98ZeCg", 1, 5)
    )
    assert undeclared(youtube) == []
    mock_gemini(monkeypatch, LISTENING)
    listened = AnalyzeAudioResponse.model_validate(
        {"metrics": measure(clip(tone())), **listen(clip(tone()))}
    )
    assert listened.listening is not None
    assert undeclared(listened) == []

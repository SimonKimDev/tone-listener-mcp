import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from google.genai import types

from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.comparison import compare, loudness_match


def clip(samples, rate=48000):
    return AudioClip(np.atleast_2d(samples).astype(np.float32), rate, "synthetic")


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_gain_invariant_comparison():
    t = np.arange(48000) / 48000
    signal = 0.2 * np.sin(2 * np.pi * 1000 * t)
    result = compare(clip(signal), clip(signal * 0.5))
    assert result["loudness_match"]["gain_db_applied_to_b"] == pytest.approx(6.0206, abs=0.001)
    assert result["deltas_b_minus_a"]["relative_band_db"]["800_2000"] == pytest.approx(0)
    assert result["deltas_b_minus_a"]["crest_factor_db"] == pytest.approx(0)
    assert result["matched_metrics_b"]["level"]["rms_dbfs"] == pytest.approx(
        result["metrics_a"]["level"]["rms_dbfs"]
    )
    assert result["metrics_b"]["level"]["rms_dbfs"] < result["metrics_a"]["level"]["rms_dbfs"]
    assert result["listening_unavailable"]


def test_high_frequency_boost_direction():
    t = np.arange(48000) / 48000
    low = np.sin(2 * np.pi * 400 * t) * 0.1
    high = np.sin(2 * np.pi * 3000 * t) * 0.1
    result = compare(clip(low + high), clip(low + 2 * high))
    difference = result["deltas_b_minus_a"]["relative_band_db"]["2000_5000"]
    assert difference == pytest.approx(10 * np.log10(0.8 / 0.5), abs=0.05)
    assert any("Cut" in hint and "2-5 kHz" in hint for hint in result["eq_direction_hints"])
    assert result["deltas_b_minus_a"]["centroid_hz"] > 0


def test_silence_comparison():
    result = compare(clip(np.zeros(48000)), clip(np.zeros(48000)))
    assert not result["loudness_match"]["applied"]
    assert result["deltas_b_minus_a"]["centroid_hz"] is None
    json.dumps(result, allow_nan=False)


def test_no_waveform_clipping_during_matching():
    a = clip(np.ones(48000) * 0.8)
    b = clip(np.concatenate(([0.9], np.ones(47999) * 0.01)))
    matched, matching = loudness_match(a, b)
    assert matching["applied"]
    assert np.max(matched.samples) > 1
    assert np.sqrt(np.mean(matched.samples**2)) == pytest.approx(0.8, abs=1e-6)


def test_two_uploads_one_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.__enter__.return_value = client
    client.files.upload.side_effect = [
        SimpleNamespace(name=f"files/{label}", uri=f"https://example.invalid/{label}",
                        state=types.FileState.ACTIVE)
        for label in "AB"
    ]
    payload = {
        "differences": ["B brighter"],
        "brighter": "B",
        "more_saturated": "similar",
        "more_compressed": "uncertain",
        "more_ambient": "A",
        "suggested_adjustments": ["Reduce B treble"],
        "uncertainties": ["Different performance"],
        "confidence": 0.6,
    }
    client.models.generate_content.return_value.text = json.dumps(payload)
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: client)
    result = compare(clip(np.ones(1000) * 0.1), clip(np.ones(1000) * 0.05))
    assert result["listening"] == payload
    assert client.files.upload.call_count == 2
    assert client.files.delete.call_count == 2
    client.models.generate_content.assert_called_once()


def test_b_upload_failure_removes_a(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.__enter__.return_value = client
    client.files.upload.side_effect = [
        SimpleNamespace(name="files/A", uri="https://example.invalid/A",
                        state=types.FileState.ACTIVE),
        OSError("second upload failed"),
    ]
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: client)
    result = compare(clip(np.ones(1000) * 0.1), clip(np.ones(1000) * 0.05))
    assert result["listening_unavailable"]
    assert result["metrics_a"]["level"]["rms_dbfs"] is not None
    client.files.delete.assert_called_once_with(name="files/A")
    client.models.generate_content.assert_not_called()

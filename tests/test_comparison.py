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
    assert any("cut" in hint and "2-5 kHz" in hint for hint in result["eq_direction_hints"])
    assert result["deltas_b_minus_a"]["centroid_hz"] > 0
    bands = result["metrics_a"]["spectrum"]["bands"]
    assert bands["above_8000"]["effective_range_hz"] == [8000, 24000]


def test_relative_bands_are_not_compared_across_nyquist_frequencies():
    a_rate, b_rate = 8000, 16000
    a = clip(np.sin(2 * np.pi * 3000 * np.arange(a_rate) / a_rate), a_rate)
    b = clip(np.sin(2 * np.pi * 4600 * np.arange(b_rate) / b_rate), b_rate)

    result = compare(a, b)

    deltas = result["deltas_b_minus_a"]["relative_band_db"]
    assert deltas["2000_5000"] is None
    assert all(value is None for value in deltas.values())
    assert result["eq_direction_hints"] == []
    assert any("Nyquist" in caution for caution in result["cautions"])
    a_bands = result["metrics_a"]["spectrum"]["bands"]
    b_bands = result["matched_metrics_b"]["spectrum"]["bands"]
    assert a_bands["2000_5000"]["effective_range_hz"] == [2000, 4000]
    assert b_bands["2000_5000"]["effective_range_hz"] == [2000, 5000]
    assert a_bands["5000_8000"]["effective_range_hz"] is None


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


@pytest.mark.parametrize("case", ["stereo_downmix", "resampling_bandloss", "peak_safety"])
def test_actual_uploaded_waveforms_are_rms_matched(monkeypatch, case):
    import soundfile as sf

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    t = np.arange(48000) / 48000
    signal = 0.2 * np.sin(2 * np.pi * 1000 * t)
    a = clip(signal)
    if case == "stereo_downmix":
        b = clip(np.stack((signal, -0.5 * signal)))
    elif case == "resampling_bandloss":
        b = clip(signal + 0.4 * np.sin(2 * np.pi * 12000 * t))
    else:
        a = clip(np.ones(48000) * 0.8)
        b = clip(np.concatenate(([0.9], np.ones(47999) * 0.01)))
    uploaded = []
    client = MagicMock()
    client.__enter__.return_value = client

    def capture(**kwargs):
        data, rate = sf.read(kwargs["file"])
        assert rate == 16000 and data.ndim == 1
        uploaded.append(data)
        return SimpleNamespace(
            name=f"files/{len(uploaded)}", uri="https://example.invalid/audio",
            state=types.FileState.ACTIVE,
        )

    client.files.upload.side_effect = capture
    client.models.generate_content.return_value.text = json.dumps({
        "differences": [], "brighter": "similar", "more_saturated": "similar",
        "more_compressed": "similar", "more_ambient": "similar",
        "suggested_adjustments": [], "uncertainties": [], "confidence": 0.5,
    })
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: client)
    result = compare(a, b)
    assert result["listening_unavailable"] is None
    assert len(uploaded) == 2
    rms_a, rms_b = [np.sqrt(np.mean(data**2)) for data in uploaded]
    assert rms_a == pytest.approx(rms_b, rel=1e-6)
    assert max(np.max(np.abs(data)) for data in uploaded) <= 1
    assert result["upload_loudness_match"]["applied"]
    if case == "stereo_downmix":
        assert result["upload_loudness_match"]["additional_gain_db_applied_to_b"] == (
            pytest.approx(10, abs=0.01)
        )
        assert result["metrics_b"]["clip"]["channels"] == 2


def test_antiphase_comparison_does_not_upload(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: client)
    signal = np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000) * 0.1
    result = compare(clip(signal), clip(np.stack((signal, -signal))))
    assert "Stereo downmix cancels" in result["listening_unavailable"]
    assert result["metrics_b"]["level"]["rms_dbfs"] is not None
    client.files.upload.assert_not_called()


def pluck(seconds=1.0, rate=48000):
    t = np.arange(round(seconds * rate)) / rate
    return sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 20)) * 0.3 * np.exp(-2 * t)


def test_silence_padding_does_not_change_active_matching():
    phrase = pluck()
    padded = np.concatenate((phrase, np.zeros(48000)))
    result = compare(clip(phrase), clip(padded))
    matching = result["loudness_match"]
    assert matching["strategy"] == "active"
    assert matching["gain_db_applied_to_b"] == pytest.approx(0, abs=0.01)
    assert result["deltas_b_minus_a"]["crest_factor_db"] == pytest.approx(0, abs=0.01)
    assert result["matched_metrics_b"]["level"]["peak_dbfs"] == pytest.approx(
        result["metrics_a"]["level"]["peak_dbfs"], abs=0.01
    )
    assert matching["activity"]["duration_mismatch"]
    assert not matching["activity"]["activity_mismatch"]
    assert any("differ in duration" in caution for caution in result["cautions"])


def test_whole_clip_matching_remains_available():
    phrase = pluck()
    padded = np.concatenate((phrase, np.zeros(48000)))
    result = compare(clip(phrase), clip(padded), matching="whole")
    assert result["loudness_match"]["strategy"] == "whole"
    assert result["loudness_match"]["gain_db_applied_to_b"] == pytest.approx(3.0103, abs=0.01)
    assert result["deltas_b_minus_a"]["crest_factor_basis"] == "crest_factor_db"
    assert any("Whole-clip matching" in caution for caution in result["cautions"])


def test_matching_strategy_is_validated():
    with pytest.raises(ValueError, match="matching"):
        loudness_match(clip(pluck()), clip(pluck()), "peak")


def test_active_matching_undefined_below_gate():
    quiet = clip(np.ones(48000) * 1e-4)
    result = compare(quiet, clip(pluck()))
    assert not result["loudness_match"]["applied"]
    assert "-70 dBFS" in result["loudness_match"]["reason"]


def test_upload_matching_ignores_silence_padding(monkeypatch):
    import soundfile as sf

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    uploaded = []
    client = MagicMock()
    client.__enter__.return_value = client

    def capture(**kwargs):
        uploaded.append(sf.read(kwargs["file"])[0])
        return SimpleNamespace(name=f"files/{len(uploaded)}", uri="https://example.invalid/a",
                               state=types.FileState.ACTIVE)

    client.files.upload.side_effect = capture
    client.models.generate_content.return_value.text = json.dumps({
        "differences": [], "brighter": "similar", "more_saturated": "similar",
        "more_compressed": "similar", "more_ambient": "similar",
        "suggested_adjustments": [], "uncertainties": [], "confidence": 0.5,
    })
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: client)
    phrase = pluck()
    result = compare(clip(phrase), clip(np.concatenate((phrase, np.zeros(48000)))))
    assert result["upload_loudness_match"]["strategy"] == "active"
    assert result["upload_loudness_match"]["additional_gain_db_applied_to_b"] == pytest.approx(
        0, abs=0.05
    )
    assert np.max(np.abs(uploaded[0])) == pytest.approx(np.max(np.abs(uploaded[1])), rel=0.01)

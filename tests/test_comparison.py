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
    hint = next(h for h in result["eq_direction_hints"] if h["band"] == "2000_5000")
    assert hint["direction"] == "cut"
    assert hint["family"] == "eq"
    hints = result["eq_direction_hints"]
    assert [h["priority"] for h in hints] == list(range(1, len(hints) + 1))
    assert hints[0]["band"] == "250_800" and hints[0]["direction"] == "boost"
    assert hint["delta_db"] > 1
    assert "2-5 kHz" in hint["text"] and "bite" in hint["listen_for"]
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
    assert set(result["deltas_b_minus_a"]["band_evidence"].values()) == {"unavailable"}
    assert not any("negligible" in caution for caution in result["cautions"])
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
    # Gated RMS omits sub-millisecond edges whose overhanging windows fail the relative gate.
    assert np.sqrt(np.mean(matched.samples**2)) == pytest.approx(0.8, rel=1e-4)


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
        "suggested_adjustments": [{
            "family": "eq", "direction": "reduce treble in B", "priority": 1,
            "observation": "B pick attack sounds harsher", "alternative_explanation":
            "Harder picking in B", "listen_for": "Softer attack without losing clarity",
        }],
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
    assert [c.kwargs["name"] for c in client.files.delete.call_args_list] == ["files/A"]
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
    assert rms_a == pytest.approx(rms_b, rel=1e-4)
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


def note(milliseconds, rate=48000):
    t = np.arange(round(milliseconds * rate / 1000)) / rate
    return sum(np.sin(2 * np.pi * 196 * k * t) / k for k in range(1, 12)) * 0.4 * np.exp(-20 * t)


def staccato_signals():
    rate = 16000
    burst = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(800) / rate)
    return {
        # Reported reproduction: 50 ms on / 50 ms off at 16 kHz.
        "staccato_50_50": (np.tile(np.concatenate((burst, np.zeros(800))), 10), rate),
        "staccato_25_25": (np.tile(np.concatenate((note(25), np.zeros(1200))), 8), 48000),
        "staccato_30_70": (np.tile(np.concatenate((note(30), np.zeros(3360))), 6), 48000),
        "short_note_60ms": (note(60), 48000),
        "short_note_30ms": (note(30), 48000),
    }


OFFSETS_MS = (1, 10, 25, 49)


def offset_variants(samples, rate, offset_ms):
    pad = np.zeros(round(rate * offset_ms / 1000))
    variants = {
        "leading": np.concatenate((pad, samples)),
        "both_ends": np.concatenate((pad, samples, pad)),
    }
    if not np.any(samples[samples.size - pad.size:]):
        variants["same_length_shift"] = np.concatenate((pad, samples[: samples.size - pad.size]))
    return variants


@pytest.mark.parametrize("offset_ms", OFFSETS_MS)
@pytest.mark.parametrize("name", list(staccato_signals()))
def test_active_matching_is_invariant_to_silence_alignment(name, offset_ms):
    samples, rate = staccato_signals()[name]
    for variant, shifted in offset_variants(samples, rate, offset_ms).items():
        result = compare(clip(samples, rate), clip(shifted, rate))
        matching = result["loudness_match"]
        assert matching["applied"], variant
        assert matching["gain_db_applied_to_b"] == pytest.approx(0, abs=1e-6), variant
        assert result["deltas_b_minus_a"]["crest_factor_db"] == pytest.approx(0, abs=1e-6)
        activity = matching["activity"]
        assert activity["active_seconds_b"] == pytest.approx(activity["active_seconds_a"], abs=1e-9)
        assert not activity["activity_mismatch"]


def test_staccato_gated_level_reflects_played_notes():
    samples, rate = staccato_signals()["staccato_50_50"]
    level = compare(clip(samples, rate), clip(samples, rate))["metrics_a"]["level"]
    # 0.2-peak sine: -16.99 dBFS while sounding; 50 ms gaps are not active.
    assert level["gated_rms_dbfs"] == pytest.approx(-16.9897, abs=0.01)
    assert level["gated_duration_seconds"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("offset_ms", OFFSETS_MS)
def test_noise_floor_padding_residual_is_small(offset_ms):
    samples, rate = staccato_signals()["staccato_30_70"]
    rng = np.random.default_rng(offset_ms)
    pad = rng.normal(0, 10 ** (-96 / 20), round(rate * offset_ms / 1000))
    result = compare(clip(samples, rate), clip(np.concatenate((pad, samples, pad[::-1])), rate))
    assert result["loudness_match"]["gain_db_applied_to_b"] == pytest.approx(0, abs=0.01)
    assert result["deltas_b_minus_a"]["crest_factor_db"] == pytest.approx(0, abs=0.01)


@pytest.mark.parametrize("offset_ms", OFFSETS_MS)
def test_whole_matching_still_counts_silence_alignment(offset_ms):
    samples, rate = staccato_signals()["staccato_50_50"]
    padded = offset_variants(samples, rate, offset_ms)["both_ends"]
    result = compare(clip(samples, rate), clip(padded, rate), matching="whole")
    expected = 10 * np.log10(padded.size / samples.size)
    assert result["loudness_match"]["gain_db_applied_to_b"] == pytest.approx(expected, abs=1e-6)


def mock_uploads(monkeypatch):
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
    return uploaded


@pytest.mark.parametrize("offset_ms", OFFSETS_MS)
@pytest.mark.parametrize("name", list(staccato_signals()))
def test_upload_matching_is_invariant_to_silence_alignment(monkeypatch, name, offset_ms):
    from tone_listener_mcp.metrics import gated_power

    uploaded = mock_uploads(monkeypatch)
    samples, rate = staccato_signals()[name]
    shifted = offset_variants(samples, rate, offset_ms)["both_ends"]
    result = compare(clip(samples, rate), clip(shifted, rate))
    upload = result["upload_loudness_match"]
    assert upload["applied"] and upload["strategy"] == "active"
    # Resampling keeps filter ringing past an abruptly ending clip only when padding follows.
    assert upload["additional_gain_db_applied_to_b"] == pytest.approx(0, abs=0.01)
    powers = [gated_power(AudioClip(data[None, :], 16000, "upload"))[0] for data in uploaded]
    assert 10 * np.log10(powers[1] / powers[0]) == pytest.approx(0, abs=0.01)
    assert np.max(np.abs(uploaded[1])) == pytest.approx(np.max(np.abs(uploaded[0])), rel=1e-3)


@pytest.mark.parametrize("gain", [0.1, 0.5, 2.0])
def test_pure_gain_change_produces_no_eq_hints(gain):
    t = np.arange(48000) / 48000
    signal = 0.4 * np.sin(2 * np.pi * 1000 * t)
    result = compare(clip(signal), clip(signal * gain))
    assert result["eq_direction_hints"] == []
    evidence = result["deltas_b_minus_a"]["band_evidence"]
    assert evidence["800_2000"] == "sufficient"
    assert evidence["above_8000"] == "insufficient_energy"
    assert any("negligible energy" in caution for caution in result["cautions"])


def test_negligible_band_deltas_are_reported_without_hints():
    t = np.arange(48000) / 48000
    signal = 0.4 * np.sin(2 * np.pi * 1000 * t)
    rng = np.random.default_rng(0)
    floor_a = rng.standard_normal(48000) * 1e-7
    floor_b = rng.standard_normal(48000) * 1e-6
    result = compare(clip(signal + floor_a), clip(signal + floor_b))
    deltas = result["deltas_b_minus_a"]
    assert deltas["relative_band_db"]["above_8000"] > 10
    assert deltas["band_evidence"]["above_8000"] == "insufficient_energy"
    assert result["eq_direction_hints"] == []


def test_band_with_energy_in_one_clip_still_gets_hint():
    t = np.arange(48000) / 48000
    low = 0.3 * np.sin(2 * np.pi * 400 * t)
    high = 0.1 * np.sin(2 * np.pi * 3000 * t)
    result = compare(clip(low), clip(low + high))
    assert result["deltas_b_minus_a"]["band_evidence"]["2000_5000"] == "sufficient"
    assert any(h["band_label"] == "2-5 kHz" and h["direction"] == "cut"
               for h in result["eq_direction_hints"])
